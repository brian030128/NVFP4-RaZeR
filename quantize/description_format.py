"""A fixed sparse prior replaces a chosen tile-count or predicted-loss budget."""
import math
import torch
from quantize.relinearized_format import common_descent_scores


@torch.no_grad()
def elect(ce, kl, tokens_per_sequence):
    d = ce.shape[1]; n = ce.shape[0]*tokens_per_sequence
    if d < 2 or tokens_per_sequence < 1:
        raise ValueError('Need at least two tiles and positive token count')
    # Independent Bernoulli p=1/(D+1). Relative to the all-zero map,
    # -log prior rises by exactly log(D) per switched tile.
    penalty = math.log(d)/n
    zero = torch.zeros(d, dtype=torch.bool, device=ce.device)
    upper = common_descent_scores(ce, kl, zero)
    mask = upper + penalty < 0
    return mask, dict(tiles=d, tokens=n, penalty=penalty, prior_probability=1/(d+1),
                      selected=int(mask.sum()), predicted_upper=float(upper[mask].sum()),
                      relative_description_nats=int(mask.sum())*math.log(d))
