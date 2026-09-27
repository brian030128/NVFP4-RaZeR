"""Fused per-token activation quantization + packing for the native Linear (one Triton launch).

One program per padded token row. Pass 1 finds the row amax (the per-token FP32 global scale gs);
pass 2 walks the row in 16-element blocks: UE4M3 scale candidate(s), rounding to E2M1 codes, two
codes per byte (element 2j in the low nibble), and the scale byte written straight into the GEMM's
block-scaled layout (lib.sf_offset_formula, computed in-kernel). Rows past T and K blocks past K/16
up to the layout's 64-K atom are written as zero scale bytes, so the buffer needs no memset: the
kernel reads those padding bytes (multiplying zero-filled operand data), and an uninitialised 0x7F
byte would be an E4M3 NaN.

Arithmetic mirrors the reference quantizers operation for operation, so the output is bit-identical
to numerics.quantize_act (tests/test_quant_act.py; ported from the repro branch's fused_quant.py,
which was verified on 5.1 G real Llama-3.1-8B activation elements):
  MODE 0  nvfp4_rows          scale = e4m3(clamp(bmax / 6)); _quant_e2m1 rounding
  MODE 1  four_over_six_rows  scales e4m3(peak / 6), e4m3(peak / 4); codes by bucketize against the
                              E2M1 midpoints; the /4 candidate iff its squared error is strictly lower
torch computes division by a Python scalar as a multiply by its FP32 reciprocal, so amax / 2688 and
peak / 6 are `* fp32(1/2688)` and `* fp32(1/6)`; a tensor divisor is IEEE division (tl.div_rn).
torch's 16-term `.sum(-1)` is the adjacent pairwise tree, spelled out in _tree_sum16; squares and
tree additions are scalar mul.rn / add.rn so ptxas cannot contract them into FMAs.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FP32_TINY = float(torch.finfo(torch.float32).tiny)
INV_2688 = float(np.float32(1.0) / np.float32(2688.0))      # 1 / (6 * 448)
INV_6 = float(np.float32(1.0) / np.float32(6.0))
MODES = {'nvfp4_rows': 0, 'four_over_six_rows': 1}


@triton.jit
def _e4m3(v):
    v = tl.minimum(tl.maximum(v, 0.001953125), 448.0)
    return v.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _sign(v):
    return tl.where(v > 0, 1.0, tl.where(v < 0, -1.0, 0.0))


@triton.jit
def _e2m1_level(idx):
    return tl.where(idx <= 4, idx.to(tl.float32) * 0.5,
                    tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)))


@triton.jit
def _bucketize(r):
    return ((r > 0.25).to(tl.int32) + (r > 0.75).to(tl.int32) + (r > 1.25).to(tl.int32)
            + (r > 1.75).to(tl.int32) + (r > 2.5).to(tl.int32) + (r > 3.5).to(tl.int32)
            + (r > 5.0).to(tl.int32))


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _tree_sum16(v):
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
def _sf_offset(row, kb, K_ATOMS):
    # lib.sf_offset_formula: 128-row x 4-block atoms of 512 bytes, row tiles outermost
    return ((row // 128) * K_ATOMS + kb // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4


@triton.jit
def _quant_rows_kernel(x_ptr, x_stride, packed_ptr, sf_ptr, gs_ptr, T, K, KB, KB_PAD, K_ATOMS,
                       MODE: tl.constexpr, BLOCK: tl.constexpr, TINY: tl.constexpr,
                       INV2688: tl.constexpr, INV6: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    NB: tl.constexpr = BLOCK // 16
    if row >= T:
        # padding row of the scale-factor layout: zero scale bytes, no data
        for kb0 in range(0, KB_PAD, NB):
            kb = kb0 + tl.arange(0, NB)
            tl.store(sf_ptr + _sf_offset(row, kb, K_ATOMS), tl.zeros((NB,), tl.uint8), mask=kb < KB_PAD)
        return
    x_row = x_ptr + row * x_stride
    amax = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    gs = tl.maximum(amax * INV2688, TINY)
    tl.store(gs_ptr + row, gs)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        s = tl.reshape(tl.div_rn(v, gs), (NB, 16))
        peak = tl.max(tl.abs(s), axis=1)
        if MODE == 1:
            sc6 = _e4m3(peak * INV6)
            sc4 = _e4m3(peak * 0.25)
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
            scale = _e4m3(peak * INV6)
            xs = tl.div_rn(s, scale[:, None])
            ax = tl.abs(xs)
            e = tl.maximum(libdevice.floor(libdevice.log2(ax + (xs == 0.0).to(tl.float32))), 0.0)
            p = libdevice.exp2(e)
            xm = tl.div_rn(xs, p) * 2.0
            xm = _sign(xm) * libdevice.floor(tl.abs(xm) + 0.5)
            code = tl.minimum(tl.maximum((xm * p) * 0.5, -6.0), 6.0)
            t2 = (tl.abs(code) * 2.0).to(tl.int32)
            idx = tl.where(t2 <= 4, t2, tl.where(t2 == 6, 5, tl.where(t2 == 8, 6, 7)))
        nib = (idx | tl.where(code < 0, 8, 0)).to(tl.uint8)
        lo, hi = tl.split(tl.reshape(nib, (BLOCK // 2, 2)))
        byte = lo | (hi << 4)
        boffs = k0 // 2 + tl.arange(0, BLOCK // 2)
        tl.store(packed_ptr + row * (K // 2) + boffs, byte, mask=boffs < K // 2)
        kb = k0 // 16 + tl.arange(0, NB)
        sbyte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(sf_ptr + _sf_offset(row, kb, K_ATOMS), sbyte, mask=kb < KB)
    # K blocks past K/16 inside the last 64-K atom
    kb = KB + tl.arange(0, 4)
    tl.store(sf_ptr + _sf_offset(row, kb, K_ATOMS), tl.zeros((4,), tl.uint8), mask=kb < KB_PAD)


def _block_for(k):
    for b in (1024, 512, 256, 128, 64, 32):
        if k >= b:
            return b
    return 32


def quantize(x2d, mode, out=None, block=None):
    """x2d [T, K] (bf16/fp16/fp32, unit stride along K) ->
    (packed [T, K/2] uint8, scale-factor buffer uint8 in the kernel layout, gs [T] fp32).

    `out`, if given, is a (packed, sf, gs) triple of preallocated buffers to write into."""
    t, k = x2d.shape
    if k % 32:
        raise ValueError(f'K={k} must be a multiple of 32')
    if x2d.stride(1) != 1:
        x2d = x2d.contiguous()
    kb = k // 16
    k_atoms = (kb + 3) // 4
    rows_pad = (t + 127) // 128 * 128
    sf_size = rows_pad // 128 * k_atoms * 512
    if out is None:
        packed = torch.empty((t, k // 2), dtype=torch.uint8, device=x2d.device)
        sf = torch.empty(sf_size, dtype=torch.uint8, device=x2d.device)
        gs = torch.empty(t, dtype=torch.float32, device=x2d.device)
    else:
        packed, sf, gs = out
        if packed.shape != (t, k // 2) or sf.numel() < sf_size or gs.numel() < t:
            raise ValueError('preallocated activation buffers have the wrong size')
    block = block or _block_for(k)
    _quant_rows_kernel[(rows_pad,)](x2d, x2d.stride(0), packed, sf, gs, t, k, kb, k_atoms * 4, k_atoms,
                                    MODE=MODES[mode], BLOCK=block, TINY=FP32_TINY, INV2688=INV_2688,
                                    INV6=INV_6, num_warps=4, enable_fp_fusion=False)
    return packed, sf, gs
