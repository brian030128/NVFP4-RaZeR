"""One legal bit per update, with current-model CE and teacher-KL scores."""
import math
import torch


@torch.no_grad()
def common_descent_scores(ce, kl, selected):
    if ce.shape != kl.shape or ce.ndim != 2 or ce.shape[0] < 2:
        raise ValueError('Need paired sequence-by-tile score matrices')
    direction = torch.where(selected, -1., 1.).to(ce)
    bounds = []
    for scores in (ce, kl):
        signed = scores * direction
        bounds.append(signed.mean(0) + 2 * signed.std(0, unbiased=True) / math.sqrt(len(scores)))
    # Both losses must have a negative directional score. These two-SE
    # quantities are heuristics, not simultaneous confidence bounds over tiles.
    return torch.maximum(*bounds)


@torch.no_grad()
def next_bit(ce, kl, selected):
    upper = common_descent_scores(ce, kl, selected)
    if not torch.isfinite(upper).all():
        raise ValueError('Nonfinite tile score')
    index = int(upper.argmin())
    return (index if float(upper[index]) < 0 else None), float(upper[index])


@torch.no_grad()
def stale_map(ce, kl, count=256):
    selected = torch.zeros(ce.shape[1], dtype=torch.bool, device=ce.device)
    upper = common_descent_scores(ce, kl, selected)
    order = torch.argsort(upper, stable=True)[:count]
    selected[order[upper[order] < 0]] = True
    return selected
