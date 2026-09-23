"""Vectorized FourOverSix activation fake-quantization with per-document scales.

Bitwise identical to calling quant.quantizer.quant_nvfp4_4over6 separately on each
document x[i] of a (B, ..., K) batch: the tensor-wide FP32 global scale is taken
per document, and the E2M1 rounding uses bucketize on the same midpoints instead
of a chain of torch.where passes. `check` compares against the reference.
"""
import torch

from quantize.quantizer import quant_nvfp4_4over6

_LEVELS = {}


def _levels(device):
    if device not in _LEVELS:
        levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)
        _LEVELS[device] = (levels, (levels[:-1] + levels[1:]) / 2)
    return _LEVELS[device]


@torch.no_grad()
def quant_per_document(x):
    shape = x.shape
    batch = shape[0] if x.dim() >= 3 else 1
    w = x.reshape(batch, -1, 16).float()
    gs = w.abs().amax(dim=(1, 2), keepdim=True) / (6 * 448)
    scaled = w / gs
    sign = scaled.sign()
    peak = scaled.abs().amax(-1, keepdim=True)
    levels, mids = _levels(x.device)
    best = None
    for qmax in (6., 4.):
        s = (peak / qmax).clamp(max=torch.finfo(torch.float8_e4m3fn).max, min=2 ** -9).to(torch.float8_e4m3fn).float()
        q = levels[torch.bucketize((scaled / s).abs().contiguous(), mids, right=False)] * sign
        error = (q * s - scaled).square().sum(-1, keepdim=True)
        if best is None:
            best = (q, s, error)
        else:
            use4 = error < best[2]
            best = (torch.where(use4, q, best[0]), torch.where(use4, s, best[1]), None)
    return (best[0] * best[1] * gs).reshape(shape).to(torch.bfloat16)


@torch.no_grad()
def check(x):
    """True iff the vectorized result equals the per-document reference bitwise."""
    ref = torch.stack([quant_nvfp4_4over6(x[i], 4, 16) for i in range(x.shape[0])]) if x.dim() >= 3 \
        else quant_nvfp4_4over6(x, 4, 16)
    return torch.equal(quant_per_document(x).view(torch.int16), ref.view(torch.int16))
