"""Dataset-free reference solver for fixed 8x64 format choices.

Exact dense SVD is intentional: this is a small-matrix mechanism reference,
not a scalable LLM quantizer. Only weight-error objectives enter selection.
"""

import torch

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6

TILE = (8, 64)
MAX_STEPS = 16
REL_TOL = 1e-10


def _check(w):
    if w.ndim != 2 or not w.numel() or w.shape[0] % 8 or w.shape[1] % 64:
        raise ValueError("Reference solver requires a nonempty matrix divisible by 8x64")
    if not torch.isfinite(w).all():
        raise ValueError("Weights must be finite")


@torch.no_grad()
def candidates(w):
    """Canonical BF16-dequantized candidates, including production scale rounding."""
    _check(w)
    if not torch.count_nonzero(w):
        return torch.zeros_like(w, dtype=torch.bfloat16), torch.zeros_like(w, dtype=torch.bfloat16)
    baseline = quant_nvfp4_4over6(w, 4, 16)
    alternative = quant_mix_4_6(w, 4, 16, type_block=TILE, clip='a1', elect='always')
    if not torch.isfinite(baseline).all() or not torch.isfinite(alternative).all():
        raise ValueError("Nonfinite quantized candidate")
    return baseline, alternative


def tile_mask(mask, shape):
    expected = (shape[0] // 8, shape[1] // 64)
    if mask.dtype != torch.bool or tuple(mask.shape) != expected:
        raise ValueError(f"Expected boolean type map of shape {expected}")
    return mask.repeat_interleave(8, 0).repeat_interleave(64, 1)


def reconstruct(base, alt, mask):
    return torch.where(tile_mask(mask, base.shape), alt, base)


def spectral_squared(error):
    """Largest squared singular value, evaluated in float64."""
    return torch.linalg.svdvals(error.double())[0].square().item()


def mse_map(w, base, alt):
    shape = (w.shape[0] // 8, 8, w.shape[1] // 64, 64)
    diff = (alt.double() - w.double()).square() - (base.double() - w.double()).square()
    return diff.reshape(shape).sum(dim=(1, 3)) < 0


@torch.no_grad()
def select(w, base=None, alt=None):
    """Best-improvement coordinate descent from all E2M1, fixed 16-step cap.

    Both switch directions are considered each step. Row-major order breaks
    ties. Every accepted step reduces the full matrix spectral objective;
    neither global optimality nor downstream accuracy is guaranteed.
    """
    _check(w)
    if (base is None) != (alt is None):
        raise ValueError("Supply both candidates or neither")
    if base is None:
        base, alt = candidates(w)
    if base.shape != w.shape or alt.shape != w.shape:
        raise ValueError("Candidate shapes must match weights")
    if not torch.isfinite(base).all() or not torch.isfinite(alt).all():
        raise ValueError("Candidates must be finite")
    mask = torch.zeros((w.shape[0] // 8, w.shape[1] // 64), dtype=torch.bool, device=w.device)
    error = base.double() - w.double()
    value = spectral_squared(error)
    history = [value]
    changes = []
    evaluations = 1
    reason = 'step_cap'
    for _ in range(MAX_STEPS):
        best_value, best_index = value, None
        for index in range(mask.numel()):
            row, col = divmod(index, mask.shape[1])
            rs, cs = slice(row * 8, (row + 1) * 8), slice(col * 64, (col + 1) * 64)
            trial = error.clone()
            replacement = base if mask[row, col] else alt
            trial[rs, cs] = replacement[rs, cs].double() - w[rs, cs].double()
            score = spectral_squared(trial)
            evaluations += 1
            if score < best_value - REL_TOL * max(value, torch.finfo(torch.float64).tiny):
                best_value, best_index = score, index
        if best_index is None:
            reason = 'coordinate_stationary'
            break
        mask.flatten()[best_index] = ~mask.flatten()[best_index]
        error = reconstruct(base, alt, mask).double() - w.double()
        value = best_value
        history.append(value)
        changes.append(best_index)
    return mask, dict(objective_history=history, changed_indices=changes,
                      objective_evaluations=evaluations, stop_reason=reason)
