"""FourOverSix with one FP32 tensor factor per token, never across tokens.

This changes the activation-scale convention; it is not claimed to be a
drop-in implementation of the original single-tensor-factor CUDA interface.
"""
import torch


@torch.no_grad()
def quantize_rows(x):
    if x.shape[-1] % 16:
        raise ValueError('Input width must be divisible by16')
    shape = x.shape
    blocks = x.float().reshape(-1, shape[-1]//16, 16)
    global_scale = (blocks.abs().amax((1,2), keepdim=True)/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = blocks/global_scale; sign = scaled.sign(); peak = scaled.abs().amax(-1, keepdim=True)
    levels = scaled.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    mids = (levels[:-1]+levels[1:])/2
    candidates = []
    for qmax in (6., 4.):
        scale = (peak/qmax).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
        code = levels[torch.bucketize((scaled/scale).abs().contiguous(), mids, right=False)]*sign
        candidates.append(code*scale)
    c6, c4 = candidates
    choose4 = (c4-scaled).square().sum(-1, keepdim=True) < (c6-scaled).square().sum(-1, keepdim=True)
    return (torch.where(choose4, c4, c6)*global_scale).reshape(shape).bfloat16()
