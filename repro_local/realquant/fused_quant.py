"""Fused per-token activation quantization + packing for the native W4A4 path (Triton).

One program per token row. Pass 1 finds the row amax (the per-token FP32 global scale); pass 2 walks
the row in chunks of 16-element blocks and, per block, computes the UE4M3 scale candidate(s),
rounds to E2M1 codes, packs two codes per byte (element 2j in the low nibble), and writes the scale
byte straight into the GEMM's swizzled scale-factor layout through a precomputed offset table.
It replaces rq.act_* + e2m1_nibbles + pack_nibbles + scale_bytes + place_scales (about 30 PyTorch
ops, several full-tensor FP32 round trips) with one launch.

Arithmetic mirrors the campaign references operation for operation, so results can be compared bit
for bit:
  MODE 0  campaign.quant.nvfp4_rows   scale = e4m3(clamp(bmax / 6)); _quant_e2m1 rounding
                                      (floor(log2) exponent, round-half-away at one mantissa bit)
  MODE 1  quantize_rows (FourOverSix) scales e4m3(peak / 6) and e4m3(peak / 4); codes by bucketize
                                      (right=False) against the E2M1 midpoints; keep the /4
                                      candidate iff its block squared error is strictly smaller
Division follows torch's CUDA semantics exactly: a tensor divisor is IEEE division (tl.div_rn), but a
Python-scalar divisor is computed by torch as a multiply by the FP32 reciprocal (div_true_kernel_cuda's
CPU-scalar path), which differs from true division in ~20% of cases -- so amax / 2688 and peak / 6
are `* fp32(1/2688)` and `* fp32(1/6)` here. log2/exp2/floor use libdevice (the same __nv_* functions
torch's CUDA kernels call); fp32 -> e4m3 rounds to nearest even, as torch.
The FourOverSix block error is a 16-term FP32 sum, so its association order decides near-tied
candidates. torch's `.sum(-1)` over 16 contiguous values is the adjacent pairwise tree
((x0+x1)+(x2+x3))+... (measured: 100% of 400k rows; sequential and other trees match 47-81%), and
_tree_sum16 spells exactly that tree out instead of relying on tl.sum's own order. The kernel is
compiled with enable_fp_fusion=False, and the squares and tree additions are scalar inline-PTX
mul.rn.f32 / add.rn.f32: torch rounds every multiply and add separately, and on sm_120 ptxas was
observed to contract Triton's packed mul.rn.f32x2 + add.rn.f32x2 into FFMAs (changing tie
decisions), which explicitly rounded scalar ops are not allowed to be.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FP32_TINY = torch.finfo(torch.float32).tiny
# torch divides by a Python scalar as a multiply by its FP32 reciprocal; these are those reciprocals
INV_2688 = float(np.float32(1.0) / np.float32(2688.0))      # 1 / (6 * 448)
INV_6 = float(np.float32(1.0) / np.float32(6.0))
MODES = {'nvfp4_rows': 0, 'four_over_six_rows': 1}


@triton.jit
def _e4m3(v):
    """clamp to [2^-9, 448], round to e4m3 (nearest even), back to fp32."""
    v = tl.minimum(tl.maximum(v, 0.001953125), 448.0)
    return v.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _sign(v):
    return tl.where(v > 0, 1.0, tl.where(v < 0, -1.0, 0.0))


@triton.jit
def _e2m1_level(idx):
    # levels [0, .5, 1, 1.5, 2, 3, 4, 6] indexed by bucketize position 0..7
    return tl.where(idx <= 4, idx.to(tl.float32) * 0.5,
                    tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)))


@triton.jit
def _bucketize(r):
    # number of E2M1 midpoints strictly below r (torch.bucketize(..., right=False))
    return ((r > 0.25).to(tl.int32) + (r > 0.75).to(tl.int32) + (r > 1.25).to(tl.int32)
            + (r > 1.75).to(tl.int32) + (r > 2.5).to(tl.int32) + (r > 3.5).to(tl.int32)
            + (r > 5.0).to(tl.int32))


@triton.jit
def _mul_rn(a, b):
    # scalar mul.rn.f32: an explicitly rounded op that ptxas may not contract into an FMA
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _tree_sum16(v):
    """Sum [NB, 16] over axis 1 as torch does: adjacent pairwise tree, one fp32 add per node."""
    NB: tl.constexpr = v.shape[0]
    a, b = tl.split(tl.reshape(v, (NB, 8, 2)))
    v8 = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v8, (NB, 4, 2)))
    v4 = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v4, (NB, 2, 2)))
    v2 = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v2, (NB, 1, 2)))
    return tl.reshape(_add_rn(a, b), (NB,))


@triton.jit
def _quant_rows_kernel(x_ptr, packed_ptr, sf_ptr, sf_idx_ptr, gs_ptr, K, KB,
                       MODE: tl.constexpr, BLOCK: tl.constexpr, TINY: tl.constexpr,
                       INV2688: tl.constexpr, INV6: tl.constexpr, GIVEN_GS: tl.constexpr = False,
                       SIGNED_ZERO: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)
    NB: tl.constexpr = BLOCK // 16
    x_row = x_ptr + row * K
    if GIVEN_GS:
        # the row's global scale is supplied (e.g. one per document, see quantize(gs=...))
        gs = tl.load(gs_ptr + row)
    else:
        # pass 1: row amax -> per-token global scale
        amax = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK):
            offs = k0 + tl.arange(0, BLOCK)
            v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
        gs = tl.maximum(amax * INV2688, TINY)                   # torch: amax / (6 * 448), scalar divisor
        tl.store(gs_ptr + row, gs)
    # pass 2: per 16-element block
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        s = tl.reshape(tl.div_rn(v, gs), (NB, 16))
        peak = tl.max(tl.abs(s), axis=1)
        if MODE == 1:
            sc6 = _e4m3(peak * INV6)                        # torch: peak / 6.0 (scalar divisor)
            sc4 = _e4m3(peak * 0.25)                        # peak / 4.0: the reciprocal is exact
            sgn = _sign(s)
            i6 = _bucketize(tl.abs(tl.div_rn(s, sc6[:, None])))
            i4 = _bucketize(tl.abs(tl.div_rn(s, sc4[:, None])))
            c6 = _e2m1_level(i6) * sgn
            c4 = _e2m1_level(i4) * sgn
            d6 = c6 * sc6[:, None] - s
            d4 = c4 * sc4[:, None] - s
            e6 = _tree_sum16(_mul_rn(d6, d6))
            e4 = _tree_sum16(_mul_rn(d4, d4))
            take4 = e4 < e6
            code = tl.where(take4[:, None], c4, c6)
            idx = tl.where(take4[:, None], i4, i6)
            scale = tl.where(take4, sc4, sc6)
        else:
            scale = _e4m3(peak * INV6)                      # torch: bmax / 6.0 (scalar divisor)
            xs = tl.div_rn(s, scale[:, None])
            ax = tl.abs(xs)
            e = tl.maximum(libdevice.floor(libdevice.log2(ax + (xs == 0.0).to(tl.float32))), 0.0)
            p = libdevice.exp2(e)
            xm = tl.div_rn(xs, p) * 2.0
            xm = _sign(xm) * libdevice.floor(tl.abs(xm) + 0.5)
            code = tl.minimum(tl.maximum((xm * p) * 0.5, -6.0), 6.0)   # torch: / 2 (scalar, exact)
            t2 = (tl.abs(code) * 2.0).to(tl.int32)            # 0,1,2,3,4,6,8,12
            idx = tl.where(t2 <= 4, t2, tl.where(t2 == 6, 5, tl.where(t2 == 8, 6, 7)))
        # SIGNED_ZERO: sign bit from the scaled input, so a negative value rounded to zero is
        # stored as -0, as the fake quantizers' level * sign produces (bitwise decode checks)
        if SIGNED_ZERO:
            neg = s < 0
        else:
            neg = code < 0
        nib = (idx | tl.where(neg, 8, 0)).to(tl.uint8)
        lo, hi = tl.split(tl.reshape(nib, (BLOCK // 2, 2)))
        byte = lo | (hi << 4)
        boffs = k0 // 2 + tl.arange(0, BLOCK // 2)
        tl.store(packed_ptr + row * (K // 2) + boffs, byte, mask=boffs < K // 2)
        kb = k0 // 16 + tl.arange(0, NB)
        dst = tl.load(sf_idx_ptr + row * KB + kb, mask=kb < KB, other=0)
        sbyte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(sf_ptr + dst, sbyte, mask=kb < KB)


def quantize(x2d, mode, sf_idx, sf_size, block=1024, gs=None, signed_zero=False):
    """x2d [T, K] bf16 -> (packed [T, K/2] uint8, scale-factor buffer [sf_size] uint8, gs [T] fp32).

    sf_idx: int64 offsets of the row-major (T, K/16) scale grid in the GEMM's layout (Kernel.sf_index).
    gs: optional FP32 [T] global scale per row (e.g. each token's document scale); by default each
    row's scale is computed from its own amax (the per-token rule).
    signed_zero: store a negative value rounded to zero as -0 (default: +0, as rq.e2m1_nibbles).
    """
    t, k = x2d.shape
    assert k % 16 == 0 and x2d.is_contiguous()
    packed = torch.empty((t, k // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.zeros(sf_size, dtype=torch.uint8, device=x2d.device)
    given = gs is not None
    if given:
        assert gs.dtype == torch.float32 and gs.shape == (t,) and gs.is_contiguous()
    else:
        gs = torch.empty(t, dtype=torch.float32, device=x2d.device)
    _quant_rows_kernel[(t,)](x2d, packed, sf, sf_idx, gs, k, k // 16, MODE=MODES[mode], BLOCK=block,
                             TINY=FP32_TINY, INV2688=INV_2688, INV6=INV_6, GIVEN_GS=given,
                             SIGNED_ZERO=signed_zero, num_warps=4,
                             enable_fp_fusion=False)
    return packed, sf, gs
