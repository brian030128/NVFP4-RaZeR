"""Real-quant execution of the N16K64 campaign policies on the mixfp4 SM120 kernel.

Every scoped nn.Linear runs as a native W4A4 block-scaled FP4 GEMM:

    weights      packed once per policy: 4-bit codes + UE4M3 scale bytes (bit 7 = E0M3 flag)
                 + one FP32 global scale, derived with the campaign's own quantizer arithmetic;
                 decode(packed) is asserted value-equal to the fake-quant weight the campaign
                 installs (campaign.policies.Installer / campaign.tiles.apply_mask).
    activations  quantized per call with the campaign's per-token rule (FourOverSix rows for the
                 MixFP4 policies, NVFP4 rows for the NVFP4 policy), packed the same way; the first
                 calls of every module are asserted value-equal to the fake-quant activation.
    GEMM         the patched mixfp4 kernel: D = gs_w * sum_k decode(A) decode(B), bf16 output.
    epilogue     the per-token activation global scale gs_x[t] is applied to D in FP32 and the
                 result rounded to bf16 (the kernel's epilogue only carries a scalar alpha).

Configurations (repro_local/realquant/build.sh):
    wt_as_A  weights are operand A, format granule 16 rows x 64 K  -> N16K64 maps
    b8x64    weights are operand B, format granule 8 cols x 64 K   -> N8K64 maps
Both pin the activation operand to E2M1.
"""
import contextlib
import ctypes
import types
from pathlib import Path

import torch

from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import _quant_e2m1, quant_nvfp4, quant_nvfp4_4over6

LIB_DIR = Path('/home/dev/n16k64_campaign/realquant/bin')
WEIGHT_OPERAND = {'wt_as_A': 0, 'b8x64': 1, 'wt_as_A_colD': 0}
TYPE_BLOCK = {'wt_as_A': (16, 64), 'b8x64': (8, 64), 'wt_as_A_colD': (16, 64)}
# Builds whose D is stored column-major: D = W X^T (out x tokens) lands in memory as row-major
# [tokens, out], so the weights-on-A output needs no transpose (build.sh wt_as_A_colD).
COLUMN_MAJOR_D = {'wt_as_A_colD'}
# Latency-study builds (repro_local/realquant/build.sh): E2M1-only kernels with no format dispatch.
# The value is the default weight operand; the stock kernel accepts either.
LATENCY_BUILDS = {'stock': 1, 'wt_as_A_nodisp': 0, 'b8x64_nodisp': 1}
# Grid/exactness checks force a host sync per call. They are verification, not part of the
# algorithm: the accuracy runs keep them on; latency runs switch them off after correctness is shown.
VALIDATE = True
# Wrap each RealLinear stage in a torch.profiler.record_function range (latency breakdowns only).
PROFILE_RANGES = False
# Epilogue as ONE elementwise pass: bf16 D times the FP32 per-token scale, computed in FP32 and
# written straight into a contiguous bf16 [tokens, out] tensor (for weights-on-A this read is the
# transpose). Bit-identical to the default three-step path (FP32 multiply, bf16 cast, contiguous
# copy), which was verified on widths 48..14336; kept opt-in so the recorded runs stay reproducible.
SINGLE_PASS_EPILOGUE = False
# Quantize + pack + place activations with the fused Triton kernel (fused_quant.py), one launch per
# Linear, instead of the PyTorch reference path. Bit-identical on real Llama-3.1-8B activations
# (results/fused_quant_bitwise_llama8b.json); opt-in so the recorded runs stay reproducible.
FUSED_ACT_QUANT = False
_FUSED = None


def _fused():
    global _FUSED
    if _FUSED is None:
        import fused_quant
        _FUSED = fused_quant
    return _FUSED
FP8_MIN, FP8_MAX = 2 ** -9, 448.0
E2M1_LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


# ----------------------------------------------------------------------------- kernel binding

class Kernel:
    def __init__(self, cfg, lib_path=None, weight_operand=None):
        self.cfg = cfg
        self.path = Path(lib_path or LIB_DIR / f'lib{cfg}.so')
        lib = ctypes.CDLL(str(self.path))
        i, i64, p, sz = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t
        lib.rq_sf_size.argtypes, lib.rq_sf_size.restype = [i, i, i, i], i64
        lib.rq_sf_offsets.argtypes, lib.rq_sf_offsets.restype = [i, i, i, i, p], i
        lib.rq_granule_map.argtypes, lib.rq_granule_map.restype = [i, p, i], i
        lib.rq_granule_shape.argtypes, lib.rq_granule_shape.restype = [p], None
        lib.rq_pinned.argtypes, lib.rq_pinned.restype = [p], None
        lib.rq_workspace_size.argtypes, lib.rq_workspace_size.restype = [i, i, i], sz
        lib.rq_gemm.argtypes = [p, p, p, p, p, i, i, i, ctypes.c_float, p, sz, p]
        lib.rq_gemm.restype = i
        self.lib = lib
        g = (ctypes.c_int * 4)()
        lib.rq_granule_shape(g)
        self.granule = dict(a_rows=g[0], a_k=g[1], b_cols=g[2], b_k=g[3])
        pn = (ctypes.c_int * 2)()
        lib.rq_pinned(pn)
        self.pinned = (bool(pn[0]), bool(pn[1]))
        self._sf = {}
        self._ws = {}
        self.d_colmajor = cfg in COLUMN_MAJOR_D
        self.latency_only = cfg in LATENCY_BUILDS
        if self.latency_only:
            # No E0M3 site: only NVFP4 / FourOverSix weights (all flags clear) are legal here.
            self.weight_operand = LATENCY_BUILDS[cfg] if weight_operand is None else weight_operand
            return
        self.weight_operand = WEIGHT_OPERAND[cfg]
        # The weight operand carries the format map; the activation operand must be pinned E2M1.
        if self.pinned[self.weight_operand] or not self.pinned[1 - self.weight_operand]:
            raise RuntimeError(f'{cfg}: unexpected pinning {self.pinned}')
        gran = ((self.granule['a_rows'], self.granule['a_k']) if self.weight_operand == 0
                else (self.granule['b_cols'], self.granule['b_k']))
        if gran != TYPE_BLOCK[cfg]:
            raise RuntimeError(f'{cfg}: library granule {gran} != expected {TYPE_BLOCK[cfg]}')

    def granule_map(self, operand):
        buf = (ctypes.c_int * 1024)()
        n = self.lib.rq_granule_map(operand, buf, 1024)
        if n <= 0:
            raise RuntimeError('granule map too large')
        return list(buf[:n])

    def sf_index(self, operand, m, n, k, device):
        """(flat offsets of the row-major (rows, k/16) scale grid, buffer size) on `device`."""
        key = (operand, m, n, k, str(device))
        hit = self._sf.get(key)
        if hit is None:
            rows = m if operand == 0 else n
            offs = torch.empty(rows * (k // 16), dtype=torch.int64)
            if self.lib.rq_sf_offsets(operand, m, n, k, ctypes.c_void_p(offs.data_ptr())) != 0:
                raise RuntimeError('rq_sf_offsets failed')
            size = int(self.lib.rq_sf_size(operand, m, n, k))
            if offs.min() < 0 or offs.max() >= size or offs.unique().numel() != offs.numel():
                raise RuntimeError(f'scale-factor layout is not injective for {key}')
            hit = (offs.to(device), size)
            self._sf[key] = hit
        return hit

    def place_scales(self, scale_bytes, operand, m, n, k):
        idx, size = self.sf_index(operand, m, n, k, scale_bytes.device)
        buf = torch.zeros(size, dtype=torch.uint8, device=scale_bytes.device)
        buf[idx] = scale_bytes.reshape(-1)
        return buf

    def workspace(self, m, n, k, device):
        key = (m, n, k, str(device))
        ws = self._ws.get(key)
        if ws is None:
            size = int(self.lib.rq_workspace_size(m, n, k))
            ws = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
            self._ws[key] = ws
        return ws

    def gemm(self, a, sfa, b, sfb, m, n, k, alpha):
        """D[m, n] (bf16) = alpha * decode(a) @ decode(b).T; a: [m, k/2] uint8, b: [n, k/2] uint8.

        Column-major-D builds return the same D as its row-major transpose, a [n, m] tensor."""
        for t in (a, sfa, b, sfb):
            assert t.is_cuda and t.dtype == torch.uint8 and t.is_contiguous()
        assert a.shape == (m, k // 2) and b.shape == (n, k // 2)
        d = torch.empty((n, m) if self.d_colmajor else (m, n), dtype=torch.bfloat16, device=a.device)
        ws = self.workspace(m, n, k, a.device)
        stream = torch.cuda.current_stream(a.device).cuda_stream
        rc = self.lib.rq_gemm(ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(sfa.data_ptr()),
                              ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(sfb.data_ptr()),
                              ctypes.c_void_p(d.data_ptr()), m, n, k, ctypes.c_float(alpha),
                              ctypes.c_void_p(ws.data_ptr()), ws.numel(), ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f'rq_gemm({m},{n},{k}) failed with code {rc}')
        return d


# ----------------------------------------------------------------------------- codes and packing

_E2M1_INDEX = {}


def e2m1_nibbles(code):
    """Codes on the signed E2M1 grid -> 4-bit nibbles (sign << 3 | magnitude index)."""
    lut = _E2M1_INDEX.get(code.device)
    if lut is None:
        lut = torch.full((13,), -1, dtype=torch.int64, device=code.device)
        for i, v in enumerate(E2M1_LEVELS):
            lut[int(2 * v)] = i
        _E2M1_INDEX[code.device] = lut
    idx = lut[(code.abs() * 2).round().long().clamp(0, 12)]
    if VALIDATE and bool((idx < 0).any()):
        raise ValueError('value off the E2M1 grid')
    return (idx | ((code < 0).long() << 3)).to(torch.uint8)


def e0m3_nibbles(code):
    mag = code.abs()
    if VALIDATE and (bool((mag > 7).any()) or bool((mag != mag.round()).any())):
        raise ValueError('value off the E0M3 grid')
    return (mag.long() | ((code < 0).long() << 3)).to(torch.uint8)


def pack_nibbles(nib):
    """[..., K] nibbles -> [..., K/2] bytes, element 2j in the low nibble (CUTLASS subbyte order)."""
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def scale_bytes(scale):
    """FP32 scales that are exactly E4M3-representable -> their UE4M3 bytes (bit 7 clear)."""
    b = scale.to(torch.float8_e4m3fn)
    if VALIDATE and not torch.equal(b.float(), scale):
        raise ValueError('scale is not E4M3-exact')
    return b.view(torch.uint8)


_DECODE = {}


def decode(nib, flags, scale, gs):
    """Reference decode: bf16((codebook[nib] * scale) * gs), fake-quant operation order."""
    lut = _DECODE.get(nib.device)
    if lut is None:
        lut = torch.tensor(list(E2M1_LEVELS) + [-v for v in E2M1_LEVELS]
                           + [float(i) for i in range(8)] + [-float(i) for i in range(8)],
                           device=nib.device)
        _DECODE[nib.device] = lut
    vals = lut[nib.long() + 16 * flags.long()]
    return ((vals * scale) * gs).bfloat16()


# ----------------------------------------------------------------------------- weight quantizers
# Each mirrors the campaign function named in its docstring line for line, but returns codes,
# block scales and the global scale instead of the dequantized product.

@torch.no_grad()
def weight_four_over_six(w):
    """quant_nvfp4_4over6(w, 4, 16)."""
    quant_value = sorted([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mid_value = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]
    n, k = w.shape
    w_fp_new = w.reshape(-1, 16).to(torch.float32)
    global_scale = w_fp_new.abs().amax() / (6.0 * 448)
    w_scaled = w_fp_new / global_scale
    w_dq_sign = w_scaled.sign()
    block_scale = w_scaled.abs().amax(dim=-1, keepdim=True)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    bs6 = (block_scale / 6.0).clamp(max=fmax, min=FP8_MIN).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    bs4 = (block_scale / 4.0).clamp(max=fmax, min=FP8_MIN).to(torch.float8_e4m3fn).to(w_scaled.dtype)

    def grid(xs):
        q = torch.zeros_like(w_scaled)
        for i, data in enumerate(quant_value):
            if i == 0:
                q += torch.where(xs <= mid_value[i], data, 0)
            elif i == len(quant_value) - 1:
                q += torch.where(xs > mid_value[i - 1], data, 0)
            else:
                q += torch.where((mid_value[i - 1] < xs) & (xs <= mid_value[i]), data, 0)
        return q * w_dq_sign

    q6, q4 = grid((w_scaled / bs6).abs()), grid((w_scaled / bs4).abs())
    e6 = ((q6 * bs6 - w_scaled) ** 2).sum(dim=-1)
    e4 = ((q4 * bs4 - w_scaled) ** 2).sum(dim=-1)
    sel4 = (e4 < e6)[:, None]
    code = torch.where(sel4, q4, q6).reshape(n, k)
    scale = torch.where(sel4, bs4, bs6).reshape(n, k // 16)
    return code, scale, global_scale


@torch.no_grad()
def weight_e0m3(w):
    """quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always') -- every block E0M3."""
    n, k = w.shape
    w32 = w.reshape(-1, k).to(torch.float32)
    global_scale = (w32.abs().amax() / (6.0 * 448)).clamp(min=torch.finfo(torch.float32).tiny)
    blocks = (w32 / global_scale).reshape(n, k // 16, 16)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (block_max * (1.0 / 7.0)).clamp(max=FP8_MAX, min=FP8_MIN).to(torch.float8_e4m3fn).to(blocks.dtype)
    code = (blocks / scale).round().clamp(min=-7.0, max=7.0)
    return code.reshape(n, k), scale.reshape(n, k // 16), global_scale


@torch.no_grad()
def weight_nvfp4(w):
    """quant_nvfp4(w, 4, 16)."""
    n, k = w.shape
    w_fp_new = w.reshape(-1, 16).to(torch.float32)
    global_scale = w_fp_new.abs().amax() / (6.0 * 448)
    w_scaled = w_fp_new / global_scale
    block_max = w_scaled.abs().amax(dim=-1, keepdim=True)
    scale = (block_max / 6.0).clamp(max=448, min=FP8_MIN).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    code = _quant_e2m1(w_scaled, scale) / scale
    return code.reshape(n, k), scale.reshape(n, k // 16), global_scale


class PackedWeight:
    """Kernel-ready weight: packed nibbles, placed scale bytes, global scale, E0M3 tile count."""

    def __init__(self, packed, sf_bytes, gs, n, k, e0m3_tiles):
        self.packed, self.sf_bytes, self.gs, self.n, self.k = packed, sf_bytes, gs, n, k
        self.e0m3_tiles = e0m3_tiles


@torch.no_grad()
def pack_weight(w, kind, mask=None, type_block=None, expected=None):
    """Pack one [out, in] weight for `kind` in {'nvfp4', 'four_over_six', 'map'}.

    Returns (PackedWeight without placed scales, row-major scale bytes). `expected`, if given, is
    the bf16 fake-quant weight the campaign installs; decode(packed) must equal it value for value.
    """
    n, k = w.shape
    if kind == 'nvfp4':
        code, scale, gs = weight_nvfp4(w)
        nib, flags = e2m1_nibbles(code), torch.zeros_like(scale, dtype=torch.uint8)
        tiles = 0
    else:
        code, scale, gs = weight_four_over_six(w)
        nib, flags, tiles = e2m1_nibbles(code), torch.zeros_like(scale, dtype=torch.uint8), 0
        if kind == 'map' and mask is not None and bool(mask.any()):
            code0, scale0, gs0 = weight_e0m3(w)
            if not torch.equal(gs0, gs):
                raise ValueError('E0M3 and FourOverSix global scales differ')
            bm, bk = type_block
            f = mask.to(w.device).repeat_interleave(bm, 0).repeat_interleave(bk // 16, 1)[:n]
            flags = f.to(torch.uint8)
            per_elem = f.repeat_interleave(16, 1)
            nib = torch.where(per_elem, e0m3_nibbles(code0), nib)
            scale = torch.where(f, scale0, scale)
            tiles = int(mask.sum())
        elif kind not in ('four_over_six', 'map'):
            raise ValueError(kind)
    sbytes = scale_bytes(scale) | (flags << 7)
    if expected is not None:
        got = decode(nib, flags.repeat_interleave(16, 1), scale.repeat_interleave(16, 1), gs)
        if not torch.equal(got, expected):
            bad = (got != expected).sum().item()
            raise AssertionError(f'packed weight differs from the fake-quant weight in {bad} elements')
    return PackedWeight(pack_nibbles(nib), None, float(gs), n, k, tiles), sbytes


# ----------------------------------------------------------------------------- activation quantizers

@torch.no_grad()
def act_four_over_six_rows(x2d):
    """quantize_rows (causal per-token FourOverSix) on a [T, K] tensor -> nib, scale, gs[T]."""
    t, k = x2d.shape
    blocks = x2d.float().reshape(-1, k // 16, 16)
    gs = (blocks.abs().amax((1, 2), keepdim=True) / (6 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = blocks / gs
    sign = scaled.sign()
    peak = scaled.abs().amax(-1, keepdim=True)
    levels = scaled.new_tensor(list(E2M1_LEVELS))
    mids = (levels[:-1] + levels[1:]) / 2
    codes, scales, deq = [], [], []
    for qmax in (6., 4.):
        scale = (peak / qmax).clamp(FP8_MIN, 448).to(torch.float8_e4m3fn).float()
        code = levels[torch.bucketize((scaled / scale).abs().contiguous(), mids, right=False)] * sign
        codes.append(code)
        scales.append(scale)
        deq.append(code * scale)
    choose4 = (deq[1] - scaled).square().sum(-1, keepdim=True) < (deq[0] - scaled).square().sum(-1, keepdim=True)
    code = torch.where(choose4, codes[1], codes[0]).reshape(t, k)
    scale = torch.where(choose4, scales[1], scales[0]).reshape(t, k // 16)
    return code, scale, gs.reshape(t)


@torch.no_grad()
def act_nvfp4_rows(x2d):
    """campaign.quant.nvfp4_rows on a [T, K] tensor -> code, scale, gs[T]."""
    t, k = x2d.shape
    b = x2d.float().reshape(-1, k // 16, 16)
    gs = (b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448, min=FP8_MIN).to(torch.float8_e4m3fn).to(s.dtype)
    code = _quant_e2m1(s, scale) / scale
    return code.reshape(t, k), scale.reshape(t, k // 16), gs.reshape(t)


ACT = {'four_over_six_rows': act_four_over_six_rows, 'nvfp4_rows': act_nvfp4_rows}


def _range(name):
    return torch.profiler.record_function(name) if PROFILE_RANGES else contextlib.nullcontext()


def _fake_act(kind):
    if kind == 'four_over_six_rows':
        return quantize_rows
    from campaign.quant import nvfp4_rows
    return nvfp4_rows


# ----------------------------------------------------------------------------- model installation

class RealLinear:
    """Replaces one nn.Linear's forward with the native kernel path."""

    def __init__(self, kernel, weight, bias, act_kind, check_calls):
        self.kernel, self.w, self.bias, self.act_kind = kernel, weight, bias, act_kind
        self.act = ACT[act_kind]
        self.check_calls = check_calls
        self.calls = 0
        self.checked = 0

    def __call__(self, x):
        w, kern = self.w, self.kernel
        k = w.k
        lead = x.shape[:-1]
        x2 = x.reshape(-1, k)
        t = x2.shape[0]
        # the activation is the GEMM operand the weights are not on
        a_op = 1 if kern.weight_operand == 0 else 0
        am, an = (w.n, t) if a_op == 1 else (t, w.n)
        if FUSED_ACT_QUANT:
            # one Triton launch: quantize, pack, and place scale bytes (bit-identical, see fused_quant)
            with _range('rq/act_quant'):
                idx, size = kern.sf_index(a_op, am, an, k, x2.device)
                packed, asf, gs = _fused().quantize(x2.contiguous(), self.act_kind, idx, size)
            self.calls += 1
        else:
            with _range('rq/act_quant'):
                code, scale, gs = self.act(x2)
            with _range('rq/encode'):
                nib = e2m1_nibbles(code)
            if self.calls < self.check_calls:
                fake = _fake_act(self.act_kind)(x2)
                got = decode(nib, torch.zeros_like(nib), scale.repeat_interleave(16, 1), gs[:, None])
                if not torch.equal(got, fake):
                    raise AssertionError(f'activation packing differs from fake quant in '
                                         f'{(got != fake).sum().item()} elements')
                self.checked += 1
            self.calls += 1
            with _range('rq/encode'):
                packed = pack_nibbles(nib)
                sbytes = scale_bytes(scale)
            with _range('rq/place'):
                asf = kern.place_scales(sbytes, a_op, am, an, k)
        if kern.weight_operand == 0:            # D[out, t] = W X^T
            with _range('rq/gemm'):
                d = kern.gemm(w.packed, w.sf_bytes, packed, asf, w.n, t, k, w.gs)
            dt = d if kern.d_colmajor else d.t()  # [t, out]: already row-major for column-major-D builds
            if SINGLE_PASS_EPILOGUE and self.bias is None:
                with _range('rq/epilogue'):
                    out = torch.empty((t, w.n), dtype=x.dtype, device=x.device)
                    torch.mul(dt, gs[:, None], out=out)
                    return out.reshape(*lead, w.n)
            with _range('rq/epilogue'):
                if kern.d_colmajor:
                    y = d.float() * gs[:, None]
                else:
                    y = (d.float() * gs[None, :]).t()
        else:                                    # D[t, out] = X W^T
            with _range('rq/gemm'):
                d = kern.gemm(packed, asf, w.packed, w.sf_bytes, t, w.n, k, w.gs)
            if SINGLE_PASS_EPILOGUE and self.bias is None:
                with _range('rq/epilogue'):
                    out = torch.empty((t, w.n), dtype=x.dtype, device=x.device)
                    torch.mul(d, gs[:, None], out=out)
                    return out.reshape(*lead, w.n)
            with _range('rq/epilogue'):
                y = d.float() * gs[:, None]
        with _range('rq/epilogue'):
            if self.bias is not None:
                y = y + self.bias.float()
            # Weights-on-A produce D^T; hand the model a row-major [t, out] tensor like nn.Linear.
            # A strided view here makes downstream attention see non-contiguous q/k/v, which sends
            # SDPA to its FP32 math backend instead of flash attention (and slows every
            # elementwise op after it). The copy moves bytes only; the values are unchanged.
            return y.to(x.dtype).contiguous().reshape(*lead, w.n)


class RealInstaller:
    """Installs packed weights and native forwards on the campaign's scoped modules.

    policy kinds follow campaign.policies.KINDS: 'nvfp4' (NVFP4 weights, nvfp4_rows activations),
    'four_over_six' and 'map' (FourOverSix weights +/- E0M3 tiles, four_over_six_rows activations).
    """

    ACT_KIND = {'nvfp4': 'nvfp4_rows', 'four_over_six': 'four_over_six_rows', 'map': 'four_over_six_rows'}

    def __init__(self, modules, tokens, check_calls=2):
        self.modules = modules
        self.tokens = tokens
        self.check_calls = check_calls
        self.saved = {n: m.forward for n, m in modules.items()}
        self.state = {}

    @torch.no_grad()
    def install(self, kind, kernel, masks=None, expected_weight=None):
        """expected_weight(name, weight) -> bf16 fake-quant weight, for the bitwise packing check."""
        self.remove()
        type_block = TYPE_BLOCK.get(kernel.cfg)
        if kernel.latency_only and masks is not None and any(bool(m.any()) for m in masks.values()):
            raise ValueError(f'{kernel.cfg} has no E0M3 site; it cannot run a map with E0M3 tiles')
        tiles = 0
        for name, mod in self.modules.items():
            w = mod.weight.detach()
            n, k = w.shape
            mask = masks[name] if masks is not None else None
            exp = expected_weight(name, w) if expected_weight is not None else None
            pw, sbytes = pack_weight(w, kind, mask, type_block, exp)
            del exp
            if kernel.weight_operand == 0:
                pw.sf_bytes = kernel.place_scales(sbytes, 0, n, self.tokens, k)
            else:
                pw.sf_bytes = kernel.place_scales(sbytes, 1, self.tokens, n, k)
            tiles += pw.e0m3_tiles
            rl = RealLinear(kernel, pw, mod.bias, self.ACT_KIND[kind], self.check_calls)
            self.state[name] = rl
            mod.forward = types.MethodType(lambda self_, x, _rl=rl: _rl(x), mod)
        return dict(e0m3_tiles=tiles, modules=len(self.modules))

    def activation_checks(self):
        return {n: rl.checked for n, rl in self.state.items()}

    def remove(self):
        for name, mod in self.modules.items():
            mod.forward = self.saved[name]
        self.state = {}
