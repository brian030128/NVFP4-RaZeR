"""One store for both MixFP4 weight candidates, in the native packed format (lean memory mode).

run_multiround.py --memory-mode lean keeps the FourOverSix E2M1 base and the E0M3 alpha=1
alternative of every quantized matrix only here. Per candidate that is the packed 4-bit codes
[n, k/2] (element 2j in the low nibble) and the row-major UE4M3 scale bytes [n, k/16] (bit 7 set
on every E0M3 scale byte); both share one FP32 global scale. That is 1.125 bytes per weight for the
two candidates together, the layout native_dev.NativeDev packs and selects from. Packing is
native_dev.pack_candidates, the code NativeDev.add runs: each candidate's decode must equal the
fake-quant candidate it replaces (decode_base / decode_alt, or the reference quantizers' output)
bitwise, signed zeros included, or packing raises.

decode(name, sel) is the fake path's weight for map `sel` [ceil(n/rows), k/cols]: element (r, c) is
the alternative's iff sel[r // rows, c // cols], i.e. the weight run_multiround.py's apply()
installs in legacy mode, torch.where(expand(sel), decode_alt, decode_base). One Triton pass
computes rq.decode's bf16((codebook[nibble] * scale) * gs) from the selected code and scale bytes,
with the same two separately rounded FP32 multiplies. reference_decode() is the PyTorch path it is
checked against: NativeDev.select + native_dev.decode_packed.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_dev  # noqa: E402
import rq  # noqa: E402

WHICH = {None: 0, 'base': 1, 'alt': 2}


@triton.jit
def _decode_kernel(b4_ptr, b0_ptr, sb4_ptr, sb0_ptr, sel_ptr, vlut_ptr, slut_ptr, gs_ptr, out_ptr,
                   K, SEL_W, ROWS: tl.constexpr, COLS: tl.constexpr, WHICH: tl.constexpr,
                   BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = c < K
    if WHICH == 0:
        alt = tl.load(sel_ptr + (r // ROWS) * SEL_W + c // COLS, mask=inside, other=0) != 0
    elif WHICH == 1:
        alt = c < 0
    else:
        alt = c >= 0
    byte_at = r * (K // 2) + c // 2
    scale_at = r * (K // 16) + c // 16
    code = tl.where(alt, tl.load(b0_ptr + byte_at, mask=inside & alt, other=0),
                    tl.load(b4_ptr + byte_at, mask=inside & (~alt), other=0)).to(tl.int32)
    sbyte = tl.where(alt, tl.load(sb0_ptr + scale_at, mask=inside & alt, other=0),
                     tl.load(sb4_ptr + scale_at, mask=inside & (~alt), other=0)).to(tl.int32)
    nibble = (code >> ((c & 1) * 4)) & 15
    value = tl.load(vlut_ptr + nibble + 16 * (sbyte >> 7))     # rq.decode's codebook row
    scale = tl.load(slut_ptr + (sbyte & 127))                 # the UE4M3 scale as FP32
    gs = tl.load(gs_ptr)
    y = (value * scale) * gs                                  # rq.decode: (vals * scale) * gs
    tl.store(out_ptr + r * K + c, y.to(tl.bfloat16), mask=inside)


class CandidateStore:
    def __init__(self, rows, cols, alt_signed_zero=False):
        self.rows, self.cols = rows, cols
        self.alt_signed_zero = alt_signed_zero
        self.cand = {}
        self.nvfp4 = {}
        self.lut = {}

    def _luts(self, device):
        hit = self.lut.get(device)
        if hit is None:
            # rq.decode's codebook: signed E2M1 (flag 0) then signed E0M3 (flag 1), 16 nibbles each
            values = torch.tensor(list(rq.E2M1_LEVELS) + [-v for v in rq.E2M1_LEVELS]
                                  + [float(i) for i in range(8)] + [-float(i) for i in range(8)],
                                  dtype=torch.float32, device=device)
            # decode_packed's (sbytes & 0x7F).view(float8_e4m3fn).float() for every byte value
            scales = torch.arange(128, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float()
            hit = self.lut[device] = (values, scales)
        return hit

    @torch.no_grad()
    def add(self, name, w, base, alt):
        """Pack both candidates of the source weight w; their decode must equal base and alt bitwise."""
        self.cand[name] = native_dev.pack_candidates(name, w, base, alt, self.alt_signed_zero)

    @torch.no_grad()
    def add_nvfp4(self, name, w, reference):
        """Optional third candidate for evaluation: plain NVFP4 (quant_nvfp4) in the native format, checked
        bitwise (signed zeros included) against reference = quant_nvfp4(w, 4, 16)."""
        n, k = w.shape
        code, scale, gs = rq.weight_nvfp4(w)
        packed, sbytes, gs = rq.pack_nibbles(native_dev.e2m1_nibbles(code)), rq.scale_bytes(scale), gs.reshape(()).float()
        got = native_dev.decode_packed(packed, sbytes, gs, n, k)
        if not torch.equal(got.view(torch.int16), reference.view(torch.int16)):
            raise AssertionError(f'{name}: packed NVFP4 differs from quant_nvfp4 in {(got != reference).sum().item()} elements')
        self.nvfp4[name] = (packed, sbytes, gs, n, k)

    def shape(self, name):
        return self.cand[name][5:7]

    @torch.no_grad()
    def decode(self, name, sel=None, which=None, block=1024):
        """bf16 [n, k]: the map `sel`'s weight (which=None), or the whole 'base' / 'alt' / 'nvfp4' candidate."""
        if which == 'nvfp4':
            packed, sbytes, gs, n, k = self.nvfp4[name]
            b4 = b0 = packed
            sb4 = sb0 = sbytes
            which = 'base'
        else:
            b4, b0, sb4, sb0, gs, n, k = self.cand[name]
        values, scales = self._luts(b4.device)
        if which is None:
            assert sel.shape == (-(-n // self.rows), k // self.cols) and sel.dtype == torch.bool
            sel_bytes = sel.contiguous().view(torch.uint8)
            width = sel.shape[1]
        else:
            sel_bytes, width = b4, 0                           # unused
        out = torch.empty((n, k), dtype=torch.bfloat16, device=b4.device)
        _decode_kernel[(n, triton.cdiv(k, block))](
            b4, b0, sb4, sb0, sel_bytes, values, scales, gs.reshape(1), out, k, width,
            ROWS=self.rows, COLS=self.cols, WHICH=WHICH[which], BLOCK=block, num_warps=4,
            enable_fp_fusion=False)
        return out

    @torch.no_grad()
    def reference_decode(self, name, sel=None, which=None):
        """The same weight through the PyTorch path: NativeDev.select + decode_packed."""
        if which == 'nvfp4':
            return native_dev.decode_packed(*self.nvfp4[name])
        b4, b0, sb4, sb0, gs, n, k = self.cand[name]
        if which == 'base':
            return native_dev.decode_packed(b4, sb4, gs, n, k)
        if which == 'alt':
            return native_dev.decode_packed(b0, sb0, gs, n, k)
        packed, sbytes, gs, n, k = native_dev.select_candidates(self.cand[name], sel, self.rows, self.cols)
        return native_dev.decode_packed(packed, sbytes, gs, n, k)

    @torch.no_grad()
    def verify_map(self, sel):
        """Modules whose Triton decode of map `sel` differs from reference_decode (bitwise, as int16)."""
        return [name for name in self.cand
                if not torch.equal(self.decode(name, sel[name]).view(torch.int16),
                                   self.reference_decode(name, sel[name]).view(torch.int16))]

    def nbytes(self):
        return (sum(t.numel() * t.element_size() for c in self.cand.values() for t in c[:4])
                + sum(t.numel() * t.element_size() for c in self.nvfp4.values() for t in c[:2]))


def lean_forward(weight_for, current, bias):
    """nn.Linear.forward with the weight weight_for(current()) decoded on every call.

    current() returns the key the weight is decoded from (e.g. the module's map); under autograd the
    decoded weight is not kept for the backward. The saved-tensor hooks replace whatever the linear
    saves of it (the transposed view mm saves) by a snapshot of the key and the view's geometry, and
    decode it again when the backward needs it: the backward runs on a bitwise identical tensor in
    the same view, so it computes exactly what nn.Linear's backward computes on a resident weight.
    """
    def forward(x):
        key = current()
        w = weight_for(key)
        if not torch.is_grad_enabled():
            return F.linear(x, w, bias)
        snapshot = tuple(k.clone() if torch.is_tensor(k) else k for k in key)
        storage = w.untyped_storage().data_ptr()

        def pack_hook(t):
            if t.untyped_storage().data_ptr() == storage:
                return ('decoded weight', t.shape, t.stride(), t.storage_offset())
            return t

        def unpack_hook(saved):
            if isinstance(saved, tuple) and saved[0] == 'decoded weight':
                return weight_for(snapshot).as_strided(*saved[1:])
            return saved

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            return F.linear(x, w, bias)
    return forward
