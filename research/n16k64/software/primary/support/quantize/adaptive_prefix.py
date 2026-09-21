"""Count-adaptive prefix selection with a PSD curvature interaction majorizer.

For any PSD H, s.T H s <= (sum_j s_j sqrt(H_jj))**2 for binary s.
Here H is a sampled predictive-GGN estimate, not the true finite-step Hessian.
"""
import math
import torch


def source_subsets():
    """Only math rows 0:64 and code rows 64:128; no C4 or Wiki calibration."""
    result = {}
    for source, start in (('math', 0), ('code', 64)):
        for count in (16, 32, 64):
            result[f'{source}{count}'] = list(range(start, start + count))
    for count in (16, 32, 64, 128):
        half = count // 2
        result[f'math_code{count}'] = list(range(half)) + list(range(64, 64 + half))
    return result


@torch.no_grad()
def adaptive_prefix(upper, diagonal):
    if upper.ndim != 1 or upper.shape != diagonal.shape:
        raise ValueError('Need aligned linear scores and curvature diagonals')
    if not torch.isfinite(upper).all() or not torch.isfinite(diagonal).all() or (diagonal < 0).any():
        raise ValueError('Nonfinite score or negative curvature')
    upper, diagonal = upper.double(), diagonal.double()
    eligible = (upper < 0).nonzero().flatten()
    order = eligible[torch.argsort(upper[eligible], stable=True)]
    linear = upper[order].cumsum(0)
    radius = diagonal[order].sqrt().cumsum(0)
    objectives = torch.cat((upper.new_zeros(1), linear + .5 * radius.square()))
    # Index 0 is the unchanged baseline, so zero switches are a legal result.
    count = int(objectives.argmin())
    chosen = order[:count]
    stats = dict(selected_blocks=count, eligible_blocks=len(order),
                 predicted_upper_objective=float(objectives[count]),
                 predicted_linear=float(linear[count-1]) if count else 0.,
                 predicted_curvature_penalty=float(.5*radius[count-1].square()) if count else 0.,
                 count_cap=None, objective_at_all_eligible=float(objectives[-1]))
    return chosen, stats


@torch.no_grad()
def derive_maps(modules, include_fixed=True):
    """Iterate (name, shape, CE[128,T], KL[128,T], sampled_score[128,T])."""
    subsets = source_subsets()
    vectors = {p: {'upper': [], 'diagonal': [], 'global_indices': []} for p in subsets}
    slices, offset = {}, 0
    for name, shape, ce, kl, fisher in modules:
        if name in slices or ce.shape != kl.shape or ce.shape != fisher.shape or ce.shape[0] != 128:
            raise ValueError('Need one aligned 128-row table per unique module')
        if ce.ndim != 2 or ce.shape[1] != (shape[0]//8)*(shape[1]//64):
            raise ValueError('Invalid type-block shape')
        slices[name] = (offset, offset+ce.shape[1])
        for policy, ids in subsets.items():
            a, b, f = ce[ids].double(), kl[ids].double(), fisher[ids].double()
            upper = torch.maximum(a.mean(0)+2*a.std(0, unbiased=True)/math.sqrt(len(ids)),
                                  b.mean(0)+2*b.std(0, unbiased=True)/math.sqrt(len(ids)))
            diagonal = 511 * f.square().mean(0)
            if not torch.isfinite(upper).all() or not torch.isfinite(diagonal).all():
                raise ValueError('Nonfinite score table')
            eligible = (upper < 0).nonzero().flatten()
            vectors[policy]['upper'].append(upper[eligible])
            vectors[policy]['diagonal'].append(diagonal[eligible])
            vectors[policy]['global_indices'].append(eligible + offset)
        offset += ce.shape[1]
    maps, statistics = {}, {}
    for setting, parts in vectors.items():
        upper = torch.cat(parts['upper']); diagonal = torch.cat(parts['diagonal'])
        global_indices = torch.cat(parts['global_indices'])
        chosen, stats = adaptive_prefix(upper, diagonal)
        modes = {f'adaptive_{setting}': (chosen, stats)}
        if include_fixed:
            fixed = torch.argsort(upper, stable=True)[:256]
            modes[f'fixed256_{setting}'] = (fixed, dict(selected_blocks=len(fixed),
                eligible_blocks=len(upper), count_cap=256))
        for policy, (indices, stats) in modes.items():
            chosen_global = global_indices[indices].sort().values
            maps[policy] = {n: (chosen_global[(chosen_global >= lo) & (chosen_global < hi)]-lo).tolist()
                            for n, (lo, hi) in slices.items()}
            statistics[policy] = dict(**stats, calibration_sequences=len(subsets[setting]),
                math_sequences=sum(i < 64 for i in subsets[setting]),
                code_sequences=sum(i >= 64 for i in subsets[setting]), total_type_blocks=offset,
                selected_fraction=len(indices)/offset, selected_weights=len(indices)*512,
                selected_scale_blocks=len(indices)*32)
    return maps, statistics
