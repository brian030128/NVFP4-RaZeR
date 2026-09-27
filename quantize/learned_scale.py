"""Learned per-block scales for E0M3 blocks and for mixed E2M1/E0M3 tensors (results/scale_additivity/PROTOCOL.md).

The E2M1 part is run_cost_distill.py's arm D (quantize/fused_fourover6.LearnedScaleE2M1): a factor f per 16-element
block, scale = e4m3(clamp(f * pre)). pre is the FourOverSix pre-rounding scale, so f = 1 reproduces
quant_nvfp4_4over6. The gradients are straight-through across the E4M3 rounding and LSQ across the element rounding.

The E0M3 part is the same construction on the E0M3 alpha = 1 grid {0, +-1, ..., +-7}, as quant_mix_4_6(clip='a1',
elect='always') and sm120's weight_e0m3 compute it:
- pre = block_max / 7 in the globally scaled domain;
- code = round-half-even(x / scale), clamped to +-7;
- value = bf16((code * scale) * global_scale).
So f = 1 reproduces the E0M3 candidate. LSQ: d/ds of s * R(x / s) is R(u) - u for |u| <= 7, and R(u) = +-7 when
saturated.

A mixed tensor (a fixed tile map: E0M3 on the flagged blocks) uses one factor per block, with the E2M1 or E0M3 pre by
the block's flag. The global scale is the tensor's FourOverSix / E0M3 global scale amax / (6 * 448), shared by both
formats as in MixFP4. Every scale stays a valid UE4M3 value, so it deploys with the E0M3 flag in bit 7.
"""
import torch

from quantize.fused_fourover6 import E4M3_MAX, E4M3_MIN, fourover6_block_scales, scale_grad, scaled_e2m1

E0M3_MAX = 7.0
E2M1_LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def e4m3_scales(factor, pre):
    """UE4M3 block scales (as FP32 values) from factors and pre-rounding scales."""
    return (factor * pre).clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()


@torch.no_grad()
def block_pre(w, flags=None):
    """(global_scale, pre [blocks]): FourOverSix pre-rounding scales, and block_max / 7 on the flagged (E0M3) blocks."""
    gs, pre4, _ = fourover6_block_scales(w)
    if flags is None or not bool(flags.any()):
        return gs, pre4
    pre0 = (w.reshape(-1, 16).float() / gs).abs().amax(-1) / E0M3_MAX
    return gs, torch.where(flags.reshape(-1), pre0, pre4)


def e2m1_codes(w, scales, gs):
    """E2M1 codes of w with given block scales, by run_cost_distill.scaled_dequant_reference's rule (a midpoint rounds
    down; the sign is kept, so -0 appears)."""
    x = w.reshape(-1, 16).float() / gs
    levels = x.new_tensor(E2M1_LEVELS)
    q = levels[torch.bucketize((x / scales[:, None]).abs().contiguous(), (levels[:-1] + levels[1:]) / 2, right=False)]
    return (q * x.sign()).reshape(w.shape)


def e0m3_codes(w, scales, gs):
    x = w.reshape(-1, 16).float() / gs
    return (x / scales[:, None]).round().clamp(min=-E0M3_MAX, max=E0M3_MAX).reshape(w.shape)


def dequant(codes, scales, gs):
    """bf16((code * scale) * global_scale): the fake quantizers' order, and sm120 numerics.decode_fake_order's."""
    shape = codes.shape
    return ((codes.reshape(-1, 16) * scales[:, None]) * gs).reshape(shape).bfloat16()


def scaled_e0m3(w, scales, gs):
    return dequant(e0m3_codes(w, scales, gs), scales, gs)


def e0m3_scale_grad(w, grad, scales, gs):
    x = w.reshape(-1, 16).float() / gs
    u = x / scales[:, None]
    code = u.round().clamp(min=-E0M3_MAX, max=E0M3_MAX)
    d = torch.where(u.abs() <= E0M3_MAX, code - u, code)
    return (grad.reshape(-1, 16).float() * d).sum(-1) * gs


class LearnedScaleMixed(torch.autograd.Function):
    """Frozen weight w and a fixed per-block format flag (True = E0M3); a learned factor per block."""

    @staticmethod
    def forward(ctx, factor, w, global_scale, pre, flags):
        target = factor * pre
        scales = target.clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()
        per_elem = flags.reshape(-1, 1).expand(-1, 16).reshape(w.shape)
        out = torch.where(per_elem, scaled_e0m3(w, scales, global_scale), scaled_e2m1(w, scales, global_scale))
        ctx.save_for_backward(w, global_scale, pre, target, scales, flags)
        return out

    @staticmethod
    def backward(ctx, grad):
        w, global_scale, pre, target, scales, flags = ctx.saved_tensors
        per_elem = flags.reshape(-1, 1).expand(-1, 16).reshape(w.shape)
        g2 = scale_grad(w, torch.where(per_elem, 0., grad).to(grad.dtype), scales, global_scale)
        g0 = e0m3_scale_grad(w, grad, scales, global_scale)
        grad_scale = torch.where(flags.reshape(-1), g0, g2)
        live = (target >= E4M3_MIN) & (target <= E4M3_MAX)
        return torch.where(live, grad_scale * pre, 0.), None, None, None, None


@torch.no_grad()
def deployed(w, factor, pre, gs, flags=None):
    """The deployed bf16 weight and its (codes, scales, flags): E2M1 codes by e2m1_codes, E0M3 codes on flagged blocks."""
    scales = e4m3_scales(factor, pre)
    codes = e2m1_codes(w, scales, gs)
    if flags is not None and bool(flags.any()):
        per_elem = flags.reshape(-1, 1).expand(-1, 16).reshape(w.shape)
        codes = torch.where(per_elem, e0m3_codes(w, scales, gs), codes)
    return dequant(codes, scales, gs), codes, scales


def tile_flags(mask, rows, cols, shape):
    """Per-16-block E0M3 flags [n, k/16] of a tile map [ceil(n/rows), k/cols] (rows beyond n trimmed)."""
    n, k = shape
    return mask.repeat_interleave(rows, 0)[:n].repeat_interleave(cols // 16, 1).contiguous()
