"""Fused FourOverSix fake quantization (Triton) for distillation training.

`fourover6(x, documents)` returns exactly quant_per_document(x) (documents = x.shape[0]) or
quant_nvfp4_4over6(x, 4, 16) (documents = 1): one FP32 global scale per document,
E4M3 block scales block_max/6 and block_max/4 for every 16-element block, E2M1 codes by
round-to-nearest (ties toward the lower level, as bucketize(right=False)), the /4
candidate kept iff its block squared error is strictly smaller, and the dequantized
value (code * scale) * global_scale rounded to BF16. The global scale is computed with
the same torch expression as the references; inside the kernel every operation
follows torch's CUDA arithmetic (IEEE division for tensor divisors, the FP32
reciprocal multiply torch uses for Python-scalar divisors, the adjacent pairwise
16-term summation tree of `.sum(-1)`, and explicitly rounded scalar mul/add that
ptxas cannot contract into FMAs); see repro_local/realquant/fused_quant.py, from
which these helpers are taken. Callers verify bitwise equality with the reference
quantizers on real tensors before relying on it (`verify_weights`).

`scaled_e2m1(x, scales, global_scale)` is the same element rounding with given E4M3
block scales (scale-only distillation); with the FourOverSix scales it again equals
quant_nvfp4_4over6. `scale_grad` is its LSQ step-size gradient.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from quantize.quantizer import quant_nvfp4_4over6

INV_6 = float(np.float32(1.0) / np.float32(6.0))
E4M3_MIN, E4M3_MAX = 2.0 ** -9, 448.0


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
def _sub_rn(a, b):
    return tl.inline_asm_elementwise('sub.rn.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32,
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
def _fourover6_kernel(x_ptr, out_ptr, gs_ptr, n_per_doc, INV6: tl.constexpr, BLOCK: tl.constexpr):
    doc = tl.program_id(1).to(tl.int64)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_per_doc
    base = doc * n_per_doc
    gs = tl.load(gs_ptr + doc)
    NB: tl.constexpr = BLOCK // 16
    v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.reshape(tl.div_rn(v, gs), (NB, 16))
    peak = tl.max(tl.abs(s), axis=1)
    sc6 = _e4m3(peak * INV6)                       # torch: peak / 6.0 (Python-scalar divisor)
    sc4 = _e4m3(peak * 0.25)                       # peak / 4.0: the reciprocal is exact
    sgn = _sign(s)
    c6 = _e2m1_level(_bucketize(tl.abs(tl.div_rn(s, sc6[:, None])))) * sgn
    c4 = _e2m1_level(_bucketize(tl.abs(tl.div_rn(s, sc4[:, None])))) * sgn
    d6 = _sub_rn(_mul_rn(c6, sc6[:, None]), s)
    d4 = _sub_rn(_mul_rn(c4, sc4[:, None]), s)
    take4 = _tree_sum16(_mul_rn(d4, d4)) < _tree_sum16(_mul_rn(d6, d6))
    code = tl.where(take4[:, None], c4, c6)
    scale = tl.where(take4, sc4, sc6)
    out = _mul_rn(_mul_rn(code, scale[:, None]), gs)
    tl.store(out_ptr + base + offs, tl.reshape(out, (BLOCK,)).to(tl.bfloat16), mask=mask)


@triton.jit
def _given_scale_kernel(x_ptr, out_ptr, scale_ptr, gs_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    NB: tl.constexpr = BLOCK // 16
    mask = offs < n
    bidx = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
    gs = tl.load(gs_ptr)
    scale = tl.load(scale_ptr + bidx, mask=bidx < n // 16, other=1.0)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.reshape(tl.div_rn(v, gs), (NB, 16))
    code = _e2m1_level(_bucketize(tl.abs(tl.div_rn(s, scale[:, None])))) * _sign(s)
    out = _mul_rn(_mul_rn(code, scale[:, None]), gs)
    tl.store(out_ptr + offs, tl.reshape(out, (BLOCK,)).to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_grad_kernel(x_ptr, g_ptr, scale_ptr, gs_ptr, grad_ptr, n, BLOCK: tl.constexpr):
    # d/ds of s * R(x / s) with a straight-through R inside the grid (LSQ): R(u) - u for
    # |u| <= 6, R(u) = +-6 when saturated; times global_scale and the output gradient.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    NB: tl.constexpr = BLOCK // 16
    mask = offs < n
    bidx = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
    bmask = bidx < n // 16
    gs = tl.load(gs_ptr)
    scale = tl.load(scale_ptr + bidx, mask=bmask, other=1.0)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.reshape(tl.div_rn(v, gs), (NB, 16))
    u = tl.div_rn(s, scale[:, None])
    code = _e2m1_level(_bucketize(tl.abs(u))) * _sign(s)
    d = tl.where(tl.abs(u) <= 6.0, code - u, code)
    total = tl.sum(tl.reshape(g, (NB, 16)) * d, axis=1) * gs
    tl.store(grad_ptr + bidx, total, mask=bmask)


def global_scales(x, documents):
    """Per-document FP32 global scale, the same torch expression as quant_per_document."""
    return (x.reshape(documents, -1, 16).float().abs().amax(dim=(1, 2)) / (6 * 448)).contiguous()


def fourover6(x, documents=1, block=1024):
    assert x.dtype == torch.bfloat16 and x.is_cuda and x.numel() % (16 * documents) == 0
    x = x.contiguous()
    n_per_doc = x.numel() // documents
    gs = global_scales(x, documents)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_per_doc, block), documents)
    _fourover6_kernel[grid](x, out, gs, n_per_doc, INV6=INV_6, BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return out


def fourover6_rows(x, block=1024):
    """quantize_rows(x) (quantize/causal_four_over_six.py): one FP32 global scale per token row, clamped
    at the smallest normal as there, codes and scales as fourover6. Bitwise equal to quantize_rows."""
    assert x.dtype == torch.bfloat16 and x.is_cuda and x.shape[-1] % 16 == 0
    x = x.contiguous()
    k = x.shape[-1]
    rows = x.numel() // k
    gs = (x.reshape(rows, k // 16, 16).float().abs().amax((1, 2), keepdim=True) / (6 * 448)).clamp_min(
        torch.finfo(torch.float32).tiny).reshape(rows).contiguous()
    out = torch.empty_like(x)
    _fourover6_kernel[(triton.cdiv(k, block), rows)](x, out, gs, k, INV6=INV_6, BLOCK=block, num_warps=4,
                                                    enable_fp_fusion=False)
    return out


def scaled_e2m1(x, scales, global_scale, block=1024):
    """E2M1 round-to-nearest of x (one tensor) with given per-block scales; BF16 dequantized."""
    assert x.dtype == torch.bfloat16 and scales.dtype == torch.float32 and scales.numel() * 16 == x.numel()
    x = x.contiguous()
    out = torch.empty_like(x)
    _given_scale_kernel[(triton.cdiv(x.numel(), block),)](x, out, scales.contiguous(), global_scale.reshape(1),
                                                          x.numel(), BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return out


def scale_grad(x, grad_out, scales, global_scale, block=1024):
    grad = torch.empty_like(scales)
    _scale_grad_kernel[(triton.cdiv(x.numel(), block),)](x.contiguous(), grad_out.contiguous(), scales.contiguous(),
                                                         global_scale.reshape(1), grad, x.numel(), BLOCK=block,
                                                         num_warps=4)
    return grad


@torch.no_grad()
def fourover6_block_scales(w):
    """Pre-rounding FourOverSix block scales of one tensor, chosen exactly as quant_nvfp4_4over6.

    Returns (global_scale, scale_before_e4m3 [blocks], e4m3 scale [blocks]); the E4M3 value is
    the block scale quant_nvfp4_4over6 uses."""
    w = w.reshape(-1, 16).float()
    gs = w.abs().amax() / (6.0 * 448)
    scaled = w / gs
    peak = scaled.abs().amax(-1)
    levels = scaled.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    mids = (levels[:-1] + levels[1:]) / 2
    best = None
    for qmax in (6.0, 4.0):
        pre = peak / qmax
        s = pre.clamp(max=E4M3_MAX, min=E4M3_MIN).to(torch.float8_e4m3fn).float()
        q = levels[torch.bucketize((scaled / s[:, None]).abs().contiguous(), mids, right=False)] * scaled.sign()
        error = (q * s[:, None] - scaled).square().sum(-1)
        if best is None:
            best = (pre, s, error)
        else:
            use4 = error < best[2]
            best = (torch.where(use4, pre, best[0]), torch.where(use4, s, best[1]), None)
    return gs, best[0], best[1]


class STEFourOverSix(torch.autograd.Function):
    """FourOverSix fake quantization with a straight-through (identity) gradient."""

    @staticmethod
    def forward(ctx, x, documents):
        return fourover6(x, documents)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class LearnedScaleE2M1(torch.autograd.Function):
    """Frozen weight w, learned factor f per 16-element block: scale = e4m3(clamp(f * pre)).

    Straight-through across the E4M3 rounding; zero gradient where the clamp is active;
    LSQ gradient across the element rounding. No gradient to w."""

    @staticmethod
    def forward(ctx, factor, w, global_scale, pre):
        target = factor * pre
        scales = target.clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()
        ctx.save_for_backward(w, global_scale, pre, target, scales)
        return scaled_e2m1(w, scales, global_scale)

    @staticmethod
    def backward(ctx, grad):
        w, global_scale, pre, target, scales = ctx.saved_tensors
        grad_scale = scale_grad(w, grad, scales, global_scale)
        live = (target >= E4M3_MIN) & (target <= E4M3_MAX)
        return torch.where(live, grad_scale * pre, 0.), None, None, None


@torch.no_grad()
def verify_weights(weights):
    """Bitwise check of fourover6 and scaled_e2m1 (FourOverSix scales) against quant_nvfp4_4over6."""
    bad = []
    for name, w in weights.items():
        ref = quant_nvfp4_4over6(w, 4, 16).view(torch.int16)
        if not torch.equal(fourover6(w, 1).view(torch.int16), ref):
            bad.append((name, 'fourover6'))
        gs, _, scales = fourover6_block_scales(w)
        if not torch.equal(scaled_e2m1(w, scales, gs).view(torch.int16), ref):
            bad.append((name, 'scaled_e2m1'))
        del ref
    return bad
