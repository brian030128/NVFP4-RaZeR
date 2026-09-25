"""The frozen numeric specification of the native path (sm120/NUMERICS.md), as code.

Every function here mirrors, operation for operation, the fake quantizer the selector and the
fake-quant evaluation use, but returns codes and scales instead of the dequantized product:

    weight_four_over_six  quantize.quantizer.quant_nvfp4_4over6(w, 4, 16)          (map baseline)
    weight_e0m3           quantize.quantizer.quant_mix_4_6(w, 4, 16, type_block=(8, 64),
                                                           clip='a1', elect='always') (map alternative)
    weight_nvfp4          quantize.quantizer.quant_nvfp4(w, 4, 16)
    act_four_over_six_rows quantize.causal_four_over_six.quantize_rows             (per-token FourOverSix)
    act_nvfp4_rows        per-token-global-scale quant_nvfp4 (the campaign's nvfp4_rows)

tests/test_numerics.py asserts decode(codes, scales, gs) == fake quantizer output bit for bit.

Encodings (hardware-defined, confirmed on SM120 by kernel/tests/mma_intrinsics):
  * nibble = (sign << 3) | magnitude_index, two per byte, element 2j in the LOW nibble;
  * E2M1 magnitudes by index: 0, 0.5, 1, 1.5, 2, 3, 4, 6;  E0M3 magnitudes by index: 0, 1, ..., 7;
  * scale byte = UE4M3 of the block scale (always positive, so its bit 7 is free) with bit 7 set
    iff the 16-element block is E0M3 (the tensor core ignores bit 7; the kernel dispatches on it);
  * one FP32 global scale per weight tensor and one per activation token row.
"""
import torch

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0M3_LEVELS = tuple(float(i) for i in range(8))
E2M1_MAX, E0M3_MAX = 6.0, 7.0
FP8_MIN, FP8_MAX = 2.0 ** -9, 448.0
GLOBAL_QMAX = 6.0 * 448.0          # both formats share the NVFP4 global-scale convention
SCALE_BLOCK = 16
FP32_TINY = torch.finfo(torch.float32).tiny


class NumericsError(ValueError):
    pass


# ------------------------------------------------------------------------------------ encodings

_LUT = {}


def _lut(device, name):
    key = (str(device), name)
    t = _LUT.get(key)
    if t is None:
        if name == 'e2m1_index':      # 2 * |code| -> magnitude index (or -1)
            t = torch.full((13,), -1, dtype=torch.int64)
            for i, v in enumerate(E2M1_LEVELS):
                t[int(2 * v)] = i
        elif name == 'decode':        # nibble + 16 * is_e0m3 -> value
            t = torch.tensor(list(E2M1_LEVELS) + [-v for v in E2M1_LEVELS]
                             + list(E0M3_LEVELS) + [-v for v in E0M3_LEVELS], dtype=torch.float64)
        t = t.to(device)
        _LUT[key] = t
    return t


def e2m1_nibbles(code, validate=True):
    """Signed E2M1 grid values -> nibbles. Negative zero maps to nibble 0 (both decode to 0)."""
    idx = _lut(code.device, 'e2m1_index')[(code.abs() * 2).round().long().clamp(0, 12)]
    if validate and bool((idx < 0).any() | (code.abs() > E2M1_MAX).any()):
        raise NumericsError('value off the E2M1 grid')
    if validate and not torch.equal(_lut(code.device, 'decode')[idx].to(code.dtype), code.abs()):
        raise NumericsError('value off the E2M1 grid')
    return (idx | ((code < 0).long() << 3)).to(torch.uint8)


def e0m3_nibbles(code, validate=True):
    mag = code.abs()
    if validate and (bool((mag > E0M3_MAX).any()) or bool((mag != mag.round()).any())):
        raise NumericsError('value off the E0M3 grid')
    return (mag.long() | ((code < 0).long() << 3)).to(torch.uint8)


def pack_nibbles(nib):
    """[..., K] nibbles -> [..., K/2] bytes, element 2j in the low nibble (CUTLASS sub-byte order)."""
    if nib.shape[-1] % 2:
        raise NumericsError('K must be even')
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def scale_bytes(scale, e0m3_flags=None, validate=True):
    """E4M3-exact positive FP32 scales -> UE4M3 bytes, bit 7 = E0M3 tag."""
    b = scale.to(torch.float8_e4m3fn)
    if validate and not torch.equal(b.float(), scale.float()):
        raise NumericsError('scale is not E4M3-representable')
    if validate and bool((scale < 0).any() | ~torch.isfinite(scale).all()):
        raise NumericsError('scale must be finite and non-negative')
    out = b.view(torch.uint8)
    if e0m3_flags is not None:
        out = out | (e0m3_flags.to(torch.uint8) << 7)
    return out


def split_scale_bytes(sbytes):
    """UE4M3 bytes with tag -> (FP32 scale, bool E0M3 flag)."""
    return (sbytes & 0x7F).view(torch.float8_e4m3fn).float(), (sbytes >> 7).bool()


def decode_exact(nib, sbytes, gs):
    """FP64 value of every element: codebook(nib, tag) * scale * gs (gs: scalar or per-row [R, 1])."""
    scale, flag = split_scale_bytes(sbytes)
    per_elem = flag.repeat_interleave(SCALE_BLOCK, -1)
    vals = _lut(nib.device, 'decode')[nib.long() + 16 * per_elem.long()]
    gs = gs.double() if torch.is_tensor(gs) else float(gs)
    return vals * scale.double().repeat_interleave(SCALE_BLOCK, -1) * gs


def decode_fake_order(nib, sbytes, gs):
    """bf16((codebook * scale) * gs) in FP32 -- the fake quantizers' operation order."""
    scale, flag = split_scale_bytes(sbytes)
    per_elem = flag.repeat_interleave(SCALE_BLOCK, -1)
    vals = _lut(nib.device, 'decode').float()[nib.long() + 16 * per_elem.long()]
    return ((vals * scale.repeat_interleave(SCALE_BLOCK, -1)) * gs).bfloat16()


# ------------------------------------------------------------------------------------ weights

def _check_weight(w):
    if w.ndim != 2 or w.shape[1] % SCALE_BLOCK:
        raise NumericsError(f'weight must be [out, in] with in % 16 == 0, got {tuple(w.shape)}')
    if not torch.isfinite(w).all():
        raise NumericsError('non-finite weight')
    if not bool(w.abs().amax() > 0):
        raise NumericsError('all-zero weight has an undefined global scale')


@torch.no_grad()
def weight_four_over_six(w):
    """quant_nvfp4_4over6(w, 4, 16) -> (code [n,k] on the E2M1 grid, scale [n,k/16], gs)."""
    _check_weight(w)
    quant_value = sorted(E2M1_LEVELS)
    mid_value = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]
    n, k = w.shape
    w_fp_new = w.reshape(-1, SCALE_BLOCK).to(torch.float32)
    global_scale = w_fp_new.abs().amax() / GLOBAL_QMAX
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
    scale = torch.where(sel4, bs4, bs6).reshape(n, k // SCALE_BLOCK)
    return code, scale, global_scale


@torch.no_grad()
def weight_e0m3(w):
    """quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'): every block E0M3."""
    _check_weight(w)
    n, k = w.shape
    w32 = w.reshape(-1, k).to(torch.float32)
    global_scale = (w32.abs().amax() / GLOBAL_QMAX).clamp(min=FP32_TINY)
    blocks = (w32 / global_scale).reshape(n, k // SCALE_BLOCK, SCALE_BLOCK)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (block_max * (1.0 / E0M3_MAX)).clamp(max=FP8_MAX, min=FP8_MIN).to(torch.float8_e4m3fn).to(blocks.dtype)
    code = (blocks / scale).round().clamp(min=-E0M3_MAX, max=E0M3_MAX)
    return code.reshape(n, k), scale.reshape(n, k // SCALE_BLOCK), global_scale


def _quant_e2m1_codes(x_s):
    """quantize.quantizer._quant_e2m1 on already-divided values, returning codes."""
    private_exp = torch.floor(torch.log2(torch.abs(x_s) + (x_s == 0).type(x_s.dtype))).clamp(min=0)
    x_m = x_s / (2 ** private_exp) * 2
    x_m = torch.sign(x_m) * torch.floor(torch.abs(x_m) + 0.5)
    return (x_m * (2 ** private_exp) / 2).clamp(min=-E2M1_MAX, max=E2M1_MAX)


@torch.no_grad()
def weight_nvfp4(w):
    """quant_nvfp4(w, 4, 16)."""
    _check_weight(w)
    n, k = w.shape
    w_fp_new = w.reshape(-1, SCALE_BLOCK).to(torch.float32)
    global_scale = w_fp_new.abs().amax() / GLOBAL_QMAX
    w_scaled = w_fp_new / global_scale
    block_max = w_scaled.abs().amax(dim=-1, keepdim=True)
    scale = (block_max / E2M1_MAX).clamp(max=FP8_MAX, min=FP8_MIN).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    code = _quant_e2m1_codes(w_scaled / scale)
    return code.reshape(n, k), scale.reshape(n, k // SCALE_BLOCK), global_scale


def expand_mask(mask, type_block, shape):
    """[n/bm, k/bk] tile mask -> [n, k/16] per-scale-block flags."""
    bm, bk = type_block
    n, k = shape
    if bk % SCALE_BLOCK or tuple(mask.shape) != (n // bm, k // bk) or n % bm or k % bk:
        raise NumericsError(f'mask {tuple(mask.shape)} does not tile weight {tuple(shape)} with block {type_block}')
    return mask.to(torch.bool).repeat_interleave(bm, 0).repeat_interleave(bk // SCALE_BLOCK, 1)


@torch.no_grad()
def quantize_weight(w, kind, mask=None, type_block=None):
    """Quantize one [out, in] weight for the native path.

    kind: 'nvfp4' | 'four_over_six' | 'map' (FourOverSix baseline, E0M3 on the masked tiles).
    Returns (nibbles uint8 [n, k], scale bytes uint8 [n, k/16] with the E0M3 tag, gs float32 scalar).
    """
    n, k = w.shape
    if kind == 'nvfp4':
        code, scale, gs = weight_nvfp4(w)
        return e2m1_nibbles(code), scale_bytes(scale), gs.float()
    if kind not in ('four_over_six', 'map'):
        raise NumericsError(f'unknown weight kind {kind!r}')
    code, scale, gs = weight_four_over_six(w)
    nib = e2m1_nibbles(code)
    flags = torch.zeros((n, k // SCALE_BLOCK), dtype=torch.bool, device=w.device)
    if kind == 'map' and mask is not None and bool(mask.any()):
        code0, scale0, gs0 = weight_e0m3(w)
        if not torch.equal(gs0, gs):
            raise NumericsError('E0M3 and FourOverSix global scales differ')
        flags = expand_mask(mask.to(w.device), type_block, (n, k))
        per_elem = flags.repeat_interleave(SCALE_BLOCK, 1)
        nib = torch.where(per_elem, e0m3_nibbles(code0), nib)
        scale = torch.where(flags, scale0, scale)
    return nib, scale_bytes(scale, flags), gs.float()


# ------------------------------------------------------------------------------------ activations

@torch.no_grad()
def act_four_over_six_rows(x2d):
    """quantize_rows (per-token FourOverSix) -> (code [T, K], scale [T, K/16], gs [T])."""
    t, k = x2d.shape
    if k % SCALE_BLOCK:
        raise NumericsError('activation width must be a multiple of 16')
    blocks = x2d.float().reshape(-1, k // SCALE_BLOCK, SCALE_BLOCK)
    gs = (blocks.abs().amax((1, 2), keepdim=True) / (6 * 448)).clamp_min(FP32_TINY)
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
    scale = torch.where(choose4, scales[1], scales[0]).reshape(t, k // SCALE_BLOCK)
    return code, scale, gs.reshape(t)


@torch.no_grad()
def act_nvfp4_rows(x2d):
    """Per-token NVFP4 (campaign.quant.nvfp4_rows) -> (code, scale, gs [T])."""
    t, k = x2d.shape
    if k % SCALE_BLOCK:
        raise NumericsError('activation width must be a multiple of 16')
    b = x2d.float().reshape(-1, k // SCALE_BLOCK, SCALE_BLOCK)
    gs = (b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)).clamp_min(FP32_TINY)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448, min=FP8_MIN).to(torch.float8_e4m3fn).to(s.dtype)
    code = _quant_e2m1_codes(s / scale)
    return code.reshape(t, k), scale.reshape(t, k // SCALE_BLOCK), gs.reshape(t)


ACT_QUANTIZERS = {'four_over_six_rows': act_four_over_six_rows, 'nvfp4_rows': act_nvfp4_rows}


@torch.no_grad()
def quantize_act(x2d, kind):
    """Reference (PyTorch) activation quantization -> (nibbles [T, K], scale bytes [T, K/16], gs [T])."""
    code, scale, gs = ACT_QUANTIZERS[kind](x2d)
    return e2m1_nibbles(code), scale_bytes(scale), gs


def fake_quant_act_rows(x, kind):
    """The dequantized bf16 activation the fake-quant path feeds its BF16 GEMM (any leading shape)."""
    shape = x.shape
    nib, sb, gs = quantize_act(x.reshape(-1, shape[-1]), kind)
    return decode_fake_order(nib, sb, gs[:, None]).reshape(shape)
