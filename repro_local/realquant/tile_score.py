"""B1: fused per-sequence tile scores for run_multiround.py's scoring pass (--tile-score-kernel).

The legacy backward hook computes, for every sequence s of a batch and every tile of a quantized matrix,

    g[s, tile] = sum_{(r, c) in tile} G_s[r, c] * D[r, c],   G_s = dy_s^T x_s (FP32),
    D = (alt - base) * (-1 if the tile is E0M3 in the current map else +1)   (alt, base: the bf16 candidates),

by materializing G_s (N x K, FP32) per sequence, then G_s * D, then a tile reduction, and adds g in FP64 to the
per-tile sum and sum of squares, sequence after sequence. This kernel never materializes G, D or G * D. Each program
owns a block of rows and one 64-column tile, decodes its D block from the packed candidates (the store's decode:
bf16((codebook * scale) * gs), then FP32 (alt - base) * sign, bitwise the legacy D), forms G's block with IEEE FP32
tl.dot over the tokens (products of bf16 values are exact in FP32; no TF32 / BF16 math), multiplies by D, reduces
each tile, and accumulates g into the FP64 sum and sum of squares in sequence order, as the legacy loop does. The only
numeric difference from the legacy hook is the FP32 summation order inside G and the tile reduction.
"""
import torch
import triton
import triton.language as tl



@triton.jit
def _decode_block(b_ptr, sb_ptr, vlut_ptr, slut_ptr, gs, rows, cols, rmask, K):
    """bf16-rounded candidate values (as FP32) of block rows x cols, the store's decode arithmetic."""
    code = tl.load(b_ptr + rows[:, None].to(tl.int64) * (K // 2) + (cols // 2)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    sbyte = tl.load(sb_ptr + rows[:, None].to(tl.int64) * (K // 16) + (cols // 16)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    nibble = (code >> ((cols[None, :] & 1) * 4)) & 15
    value = tl.load(vlut_ptr + nibble + 16 * (sbyte >> 7))
    scale = tl.load(slut_ptr + (sbyte & 127))
    return ((value * scale) * gs).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tile_score_kernel(dy_ptr, x_ptr, b4_ptr, b0_ptr, sb4_ptr, sb0_ptr, sel_ptr, vlut_ptr, slut_ptr, gs_ptr,
                       sum_ptr, sq_ptr, B, T, N, K, SEL_W,
                       R: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BT: tl.constexpr):
    PR: tl.constexpr = R if R > BR else BR          # rows owned by one program
    SUBS: tl.constexpr = PR // BR                   # row sub-blocks per program (R = 256: 4)
    TPS: tl.constexpr = BR // R if R < BR else 1    # tiles per sub-block along rows (R = 8: 8)
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    cols = pc * C + tl.arange(0, C)
    gs = tl.load(gs_ptr)
    trow = pr * (PR // R) + tl.arange(0, TPS)       # tile rows this program writes
    tmask = trow * R < N
    acc_sum = tl.load(sum_ptr + trow.to(tl.int64) * SEL_W + pc, mask=tmask, other=0.0)
    acc_sq = tl.load(sq_ptr + trow.to(tl.int64) * SEL_W + pc, mask=tmask, other=0.0)
    # D of the first (for 8x64 tiles: the only) row sub-block, decoded once
    rows0 = pr * PR + tl.arange(0, BR)
    rmask0 = rows0 < N
    alt0 = tl.load(sel_ptr + (rows0 // R).to(tl.int64) * SEL_W + pc, mask=rmask0, other=0) != 0
    d0 = (_decode_block(b0_ptr, sb0_ptr, vlut_ptr, slut_ptr, gs, rows0, cols, rmask0, K)
          - _decode_block(b4_ptr, sb4_ptr, vlut_ptr, slut_ptr, gs, rows0, cols, rmask0, K)) * tl.where(alt0, -1.0, 1.0)[:, None]
    for s in range(B):
        val = tl.zeros((TPS,), tl.float32)
        for sub in tl.static_range(SUBS):
            rows = pr * PR + sub * BR + tl.arange(0, BR)
            rmask = rows < N
            if sub == 0:
                d = d0
            else:
                alt = tl.load(sel_ptr + (rows // R).to(tl.int64) * SEL_W + pc, mask=rmask, other=0) != 0
                d = (_decode_block(b0_ptr, sb0_ptr, vlut_ptr, slut_ptr, gs, rows, cols, rmask, K)
                     - _decode_block(b4_ptr, sb4_ptr, vlut_ptr, slut_ptr, gs, rows, cols, rmask, K)) * tl.where(alt, -1.0, 1.0)[:, None]
            acc = tl.zeros((BR, C), tl.float32)
            for t0 in range(0, T, BT):
                ts = t0 + tl.arange(0, BT)
                tmask_t = ts < T
                dyb = tl.load(dy_ptr + (s * T + ts[:, None]).to(tl.int64) * N + rows[None, :],
                              mask=tmask_t[:, None] & rmask[None, :], other=0.0).to(tl.float32)
                xb = tl.load(x_ptr + (s * T + ts[:, None]).to(tl.int64) * K + cols[None, :],
                             mask=tmask_t[:, None], other=0.0).to(tl.float32)
                acc = tl.dot(tl.trans(dyb), xb, acc, input_precision='ieee')
            prod = acc * d
            if R < BR:
                val += tl.sum(tl.sum(tl.reshape(prod, (TPS, R, C)), axis=2), axis=1)
            else:
                val += tl.sum(tl.sum(prod, axis=1), axis=0)
        v64 = val.to(tl.float64)
        acc_sum += v64
        acc_sq += v64 * v64
    tl.store(sum_ptr + trow.to(tl.int64) * SEL_W + pc, acc_sum, mask=tmask)
    tl.store(sq_ptr + trow.to(tl.int64) * SEL_W + pc, acc_sq, mask=tmask)


def tile_scores(dy, x, cand, sel, rows, cols, luts, total, square, br=64, bt=32, num_warps=4):
    """Add every sequence's tile score of one matrix into (total, square), FP64 [ceil(N/rows), K/cols], in place.

    dy: (B, T, N) bf16; x: (B, T, K) bf16; cand: the store's (b4, b0, sb4, sb0, gs, n, k); sel: the current map;
    luts: CandidateStore._luts(device) (codebook, scale table)."""
    b4, b0, sb4, sb0, gs, n, k = cand
    batch, tokens = x.shape[0], x.shape[1]
    assert dy.shape == (batch, tokens, n) and x.shape[2] == k and cols == 64 and k % cols == 0
    assert rows in (8, 256) and total.dtype == square.dtype == torch.float64
    dy, x = dy.contiguous(), x.contiguous()
    values, scales = luts
    pr = max(rows, br)
    grid = (triton.cdiv(n, pr), k // cols)
    _tile_score_kernel[grid](dy, x, b4, b0, sb4, sb0, sel.contiguous().view(torch.uint8), values, scales,
                             gs.reshape(1), total, square, batch, tokens, n, k, sel.shape[1],
                             R=rows, C=cols, BR=br, BT=bt, num_warps=num_warps)
