"""Native (mixfp4 b8x64 kernel) development evaluation for run_multiround.py --shadow-native.

Weights (operand B, format granule 8 output rows x 64 K): both candidates, E2M1 FourOverSix
(base) and E0M3 alpha=1 (alternative), are packed once from the source weight with rq.py's
mirrors of the reference quantizers. decode(packed) is checked bitwise against the fake-quant
candidates. For a map, every byte of codes and every scale byte is taken from the base or the
alternative tile by tile (bit 7 of a scale byte = E0M3). A map unit (256x64 or 8x64) is a
union of 8x64 granules.

Activations: per-document tensor-wide FourOverSix, the rule of quant_per_document. One FP32
global scale per document, codes and UE4M3 block scales against it (fused Triton kernel with
the scales given), and every token of the document gets that scale in the epilogue. The first
calls are checked bitwise against quant_per_document.

GEMM: D[t, out] = gs_w * sum_k decode(X) decode(W), in bf16. Output: bf16(D * gs_doc[t]).
lm_head and everything outside the 224 matrices run as in the fake path.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fused_quant  # noqa: E402
import rq  # noqa: E402
from quantize.fast_act import quant_per_document  # noqa: E402

CONFIG = 'b8x64'


_E2M1_INDEX = {}


def e2m1_nibbles(code):
    """rq.e2m1_nibbles with the sign taken from the sign bit, so that a negative value rounded to
    zero is stored as -0. That is what quantize/packed_candidates.decode_base returns: level 0
    times the sign of a negative input. torch.sign(-0.0) is +0, so an exact -0 input stays +0.
    The E0M3 alternative uses rq.e0m3_nibbles (sign from code < 0): decode_alt stores codes in
    offset binary, so its zeros are +0."""
    lut = _E2M1_INDEX.get(code.device)
    if lut is None:
        lut = torch.full((13,), -1, dtype=torch.int64, device=code.device)
        for i, v in enumerate(rq.E2M1_LEVELS):
            lut[int(2 * v)] = i
        _E2M1_INDEX[code.device] = lut
    idx = lut[(code.abs() * 2).round().long().clamp(0, 12)]
    if bool((idx < 0).any()):
        raise ValueError('value off the E2M1 grid')
    return (idx | (torch.signbit(code).long() << 3)).to(torch.uint8)


def unpack_nibbles(packed, n, k):
    """[n, k/2] bytes -> [n, k] nibbles (element 2j in the low nibble)."""
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(n, k)


def decode_packed(packed, sbytes, gs, n, k):
    """bf16 weight or activation from packed codes and row-major scale bytes (bit 7 = E0M3)."""
    flags = (sbytes >> 7).repeat_interleave(16, 1)
    scale = (sbytes & 0x7F).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return rq.decode(unpack_nibbles(packed, n, k), flags, scale, gs)


class NativeDev:
    def __init__(self, modules, rows, cols, tokens, check_calls=64):
        if (rows, cols) not in ((256, 64), (8, 64)):
            raise ValueError('a map unit must be a union of the kernel\'s 8x64 format granules')
        self.kern = rq.Kernel(CONFIG)
        assert self.kern.weight_operand == 1 and rq.TYPE_BLOCK[CONFIG] == (8, 64)
        self.modules, self.rows, self.cols, self.tokens = modules, rows, cols, tokens
        self.check_calls, self.checked = check_calls, 0
        self.cand, self.built, self.current = {}, {}, {}
        self.saved = {n: m.forward for n, m in modules.items()}
        self.documents = 1
        self.rebuild_seconds = 0.0

    @torch.no_grad()
    def add(self, name, w, base, alt):
        """Pack both candidates of the source weight w; their decode must equal base and alt bitwise."""
        n, k = w.shape
        c4, s4, g4 = rq.weight_four_over_six(w)
        c0, s0, g0 = rq.weight_e0m3(w)
        if not torch.equal(g4, g0):
            raise AssertionError(f'{name}: E0M3 and FourOverSix global scales differ')
        sb4 = rq.scale_bytes(s4)
        sb0 = rq.scale_bytes(s0) | 128
        b4, b0 = rq.pack_nibbles(e2m1_nibbles(c4)), rq.pack_nibbles(rq.e0m3_nibbles(c0))
        gs = g4.reshape(()).float()
        for packed, sb, ref, label in ((b4, sb4, base, 'base'), (b0, sb0, alt, 'alternative')):
            got = decode_packed(packed, sb, gs, n, k)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                raise AssertionError(f'{name}: packed {label} differs from the fake-quant candidate in '
                                     f'{(got != ref).sum().item()} elements')
        self.cand[name] = (b4, b0, sb4, sb0, gs, n, k)

    @torch.no_grad()
    def select(self, name, sel):
        """Codes and row-major scale bytes of module `name` for map `sel` [ceil(n/rows), k/cols]."""
        b4, b0, sb4, sb0, gs, n, k = self.cand[name]
        rows = sel.repeat_interleave(self.rows, 0)[:n]
        packed = torch.where(rows.repeat_interleave(self.cols // 2, 1), b0, b4).contiguous()
        sbytes = torch.where(rows.repeat_interleave(self.cols // 16, 1), sb0, sb4).contiguous()
        return packed, sbytes, gs, n, k

    @torch.no_grad()
    def verify_map(self, sel, installed):
        """Bitwise: decode(selected packed weight) == the weight apply() installed, every module."""
        bad = []
        for name in self.cand:
            packed, sbytes, gs, n, k = self.select(name, sel[name])
            if not torch.equal(decode_packed(packed, sbytes, gs, n, k).view(torch.int16),
                               installed(name).view(torch.int16)):
                bad.append(name)
        return bad

    @torch.no_grad()
    def install(self, sel):
        """Native forwards for map `sel`; only modules whose map changed since the last build are repacked."""
        t0 = time.time()
        for name, mod in self.modules.items():
            snap = self.built.get(name)
            if snap is None or not torch.equal(snap, sel[name]):
                packed, sbytes, gs, n, k = self.select(name, sel[name])
                sf = self.kern.place_scales(sbytes, 1, self.tokens, n, k)
                self.current[name] = rq.PackedWeight(packed, sf, gs, n, k, int(sel[name].sum()))
                self.built[name] = sel[name].clone()
            mod.forward = self._forward(name)
        torch.cuda.synchronize()
        self.rebuild_seconds = time.time() - t0

    def remove(self):
        for name, mod in self.modules.items():
            mod.forward = self.saved[name]

    def _forward(self, name):
        w, kern = self.current[name], self.kern

        def forward(x):
            k, docs = w.k, self.documents
            x2 = x.reshape(-1, k).contiguous()
            t = x2.shape[0]
            assert t == self.tokens and t % docs == 0, (tuple(x.shape), docs)
            # quant_per_document's global scale, one per document, broadcast to its tokens
            gs_doc = x.reshape(docs, -1, 16).float().abs().amax(dim=(1, 2)) / (6 * 448)
            gs_rows = gs_doc.repeat_interleave(t // docs).contiguous()
            idx, size = kern.sf_index(0, t, w.n, k, x2.device)
            packed, asf, _ = fused_quant.quantize(x2, 'four_over_six_rows', idx, size, gs=gs_rows,
                                                  signed_zero=True)
            if self.checked < self.check_calls:
                got = decode_packed(packed, asf[idx].reshape(t, k // 16), gs_rows[:, None], t, k)
                ref = quant_per_document(x.reshape(docs, -1, k)).reshape(t, k)
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    raise AssertionError(f'{name}: native activation differs from quant_per_document in '
                                         f'{(got != ref).sum().item()} elements')
                self.checked += 1
            d = kern.gemm(packed, asf, w.packed, w.sf_bytes, t, w.n, k, w.gs)
            return (d.float() * gs_rows[:, None]).to(x.dtype).reshape(*x.shape[:-1], w.n)
        return forward
