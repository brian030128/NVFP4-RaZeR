"""Evaluate the unchanged sparse rule module by module to bound CPU memory."""
import torch
from quantize.relinearized_format import common_descent_scores


def modulewise_maps(tables, names, count=256):
    """tables[name] is (CE[192, tiles], KL[192, tiles]) in web/math/code order."""
    vectors = {p: [] for p in ('c4_64', 'mixed64', 'pooled192')}
    slices = {}
    offset = 0
    mixed = torch.tensor(list(range(22)) + list(range(64, 85)) + list(range(128, 149)))
    for name in names:
        ce, kl = tables[name]
        if ce.shape != kl.shape or ce.ndim != 2 or ce.shape[0] != 192:
            raise ValueError('Need paired 192-sequence score tables')
        selected = torch.zeros(ce.shape[1], dtype=torch.bool, device=ce.device)
        for p, a, b in [('c4_64', ce[:64], kl[:64]),
                        ('mixed64', ce[mixed], kl[mixed]), ('pooled192', ce, kl)]:
            v = common_descent_scores(a, b, selected)
            if not torch.isfinite(v).all():
                raise ValueError('Nonfinite tile score')
            vectors[p].append(v)
        slices[name] = (offset, offset + ce.shape[1])
        offset += ce.shape[1]
    maps = {}
    for p, parts in vectors.items():
        upper = torch.cat(parts)
        order = torch.argsort(upper, stable=True)[:count]
        mask = torch.zeros_like(upper, dtype=torch.bool)
        mask[order[upper[order] < 0]] = True
        maps[p] = {n: mask[lo:hi].clone() for n, (lo, hi) in slices.items()}
    return maps
