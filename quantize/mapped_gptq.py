"""Columnwise GPTQ compensation under a frozen MixFP4 tile-format map.

The format of every (tile_rows x 64) weight tile is fixed in advance (E0M3 alpha=1
where the mask is set, FourOverSix E2M1 elsewhere). Compensation changes only the
4-bit codes and E4M3 block scales, never the map, so the result is a legal MixFP4
tensor for the same kernel and format metadata. Block scales are fixed at the start
of each 16-column scale group from the current compensated weights (the convention
of quantize/branched_format.py). Quantization arithmetic mirrors quant_nvfp4_4over6
in float32; only the compensation state is float64. With a diagonal factor `u` the
result is round-to-nearest FourOverSix / E0M3 under the map.
"""
import torch

from quantize.branched_format import inverse_factor

LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def _fp8(x):
    return x.clamp(2 ** -9, 448).to(torch.float8_e4m3fn).float()


@torch.no_grad()
def quantize_mapped(w, u, mask, tile_rows):
    n, k = w.shape
    if k % 64 or tuple(mask.shape) != ((n + tile_rows - 1) // tile_rows, k // 64):
        raise ValueError('Mask does not match weight tiles')
    gs = (w.float().abs().amax() / (6 * 448)).clamp(min=torch.finfo(torch.float32).tiny)
    levels = torch.tensor(LEVELS, device=w.device)
    mid = (levels[:-1] + levels[1:]) / 2
    rows = mask.to(w.device).repeat_interleave(tile_rows, 0)[:n]
    work = w.double().clone()
    out = torch.empty_like(w, dtype=torch.bfloat16)

    def e2m1(scaled):
        return levels[torch.bucketize(scaled.abs().contiguous(), mid)] * scaled.sign()

    for start in range(0, k, 64):
        stop = start + 64
        e0 = rows[:, start // 64]
        local = u[start:stop, start:stop]
        errors = torch.empty((n, 64), device=w.device, dtype=torch.float64)
        for j in range(64):
            c = start + j
            if j % 16 == 0:
                group = work[:, c:c + 16].float() / gs
                maximum = group.abs().amax(-1, keepdim=True)
                s6, s4 = _fp8(maximum / 6), _fp8(maximum / 4)
                err6 = (e2m1(group / s6) * s6 - group).square().sum(-1, keepdim=True)
                err4 = (e2m1(group / s4) * s4 - group).square().sum(-1, keepdim=True)
                s2 = torch.where(err4 < err6, s4, s6)[:, 0]
                s0 = _fp8(maximum / 7)[:, 0]
            x = work[:, c]
            scaled = x.float() / gs
            q2 = e2m1(scaled / s2) * s2 * gs
            q0 = (scaled / s0).round().clamp(-7, 7) * s0 * gs
            q = torch.where(e0, q0, q2).to(torch.bfloat16)
            out[:, c] = q
            err = (q.double() - x) / local[j, j]
            errors[:, j] = err
            work[:, c + 1:stop] += err[:, None] * local[j, j + 1:]
        if stop < k:
            work[:, stop:] += errors @ u[start:stop, stop:]
    return out


def factor(h):
    """Upper Cholesky factor of the 1%-damped inverse second moment."""
    return inverse_factor(h)[0]
