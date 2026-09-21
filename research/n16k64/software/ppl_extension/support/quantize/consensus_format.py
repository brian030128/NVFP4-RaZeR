"""Consensus of per-source CE/KL directional scores at one shared baseline."""
import torch
from quantize.relinearized_format import common_descent_scores, stale_map


@torch.no_grad()
def consensus_map(ce, kl, count=256):
    # Source x example x tile. A common descent direction for all sources
    # is a descent direction for their convex mixtures in the linear model.
    if ce.ndim != 3 or ce.shape != kl.shape:
        raise ValueError('Expected paired source-by-example-by-tile scores')
    zero = torch.zeros(ce.shape[-1], dtype=torch.bool, device=ce.device)
    upper = torch.stack([common_descent_scores(a, b, zero) for a, b in zip(ce, kl)]).amax(0)
    order = torch.argsort(upper, stable=True)[:count]
    zero[order[upper[order] < 0]] = True
    return zero


@torch.no_grad()
def elect_all(ce, kl, count=256):
    domains = ('web', 'math', 'code')
    maps = {'all_consensus': consensus_map(ce, kl, count),
            'all_pooled': stale_map(ce.flatten(0, 1), kl.flatten(0, 1), count)}
    for i, heldout in enumerate(domains):
        keep = [j for j in range(3) if i != j]
        maps[f'without_{heldout}'] = consensus_map(ce[keep], kl[keep], count)
        maps[f'pooled_without_{heldout}'] = stale_map(ce[keep].flatten(0, 1), kl[keep].flatten(0, 1), count)
    return maps
