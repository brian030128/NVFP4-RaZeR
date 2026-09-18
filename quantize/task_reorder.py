"""Joint channel partitioning and CE/KL format election for coarse MixFP4.

Input scores are directional derivatives, not MSE gains: negative favors E0M3.
Each [sequence, row_atom, column_atom] cell must be additive under permutation.
Use 1x16 atoms for full row/scale-group search; 8x64 historical scores allow only
row-band regrouping. Sequence vectors are retained to preserve tile covariance.

The solver alternates capacity-constrained linearized assignments and exact
pair exchanges, from several starting layouts and with a smooth-to-hard
continuation. It optimizes a surrogate on FIT data, not actual model loss.
Call elect_layout on disjoint sequences after freezing the winning layout.
All large operations belong on Slurm workers, including CPU-only searches.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SearchConfig:
    tile_rows: int = 256
    tile_cols: int = 64
    atom_rows: int = 1
    atom_cols: int = 16
    k: float = 3.0
    seed: int = 0
    starts: int = 4
    rounds: int = 6
    swap_samples: int = 4096
    swap_passes: int = 2
    scratch_mb: int = 32
    axes: str = "both"

    def validate(self):
        for field in ("tile_rows", "tile_cols", "atom_rows", "atom_cols", "starts",
                      "scratch_mb"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) < 1:
                raise ValueError(f'{field} must be a positive integer')
        for field in ("rounds", "swap_samples", "swap_passes"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        if self.tile_rows % self.atom_rows or self.tile_cols % self.atom_cols:
            raise ValueError('Type tile must be divisible by score atom shape')
        if self.atom_cols % 16 or not math.isfinite(self.k) or self.k < 0:
            raise ValueError('Atoms must preserve 16-column scales; k must be finite and >= 0')
        if self.axes not in ('both', 'rows', 'cols'):
            raise ValueError('axes must be both, rows, or cols')


def _features(ce, kl):
    if ce.shape != kl.shape or ce.ndim != 3 or min(ce.shape) < 1 or ce.shape[0] < 2:
        raise ValueError('Need matching [sequences >= 2, row atoms, column atoms] scores')
    if not ce.is_floating_point() or not kl.is_floating_point():
        raise ValueError('Directional scores must be floating point')
    if not torch.isfinite(ce).all() or not torch.isfinite(kl).all():
        raise ValueError('Nonfinite directional score')
    # CPU float64 avoids cancellation changing an election near zero. The two
    # sequence axes remain independent objectives, not independent random tiles.
    return torch.stack((ce.cpu().double(), kl.cpu().double()), dim=0).permute(
        2, 3, 0, 1).contiguous().flatten(2)


def _labels(order, size):
    labels = torch.empty_like(order)
    labels[order] = torch.arange(order.numel()) // size
    return labels


def _sum_axis(x, labels, axis):
    shape = list(x.shape)
    shape[axis] = int(labels.max()) + 1
    return x.new_zeros(shape).index_add_(axis, labels, x)


def _tiles(features, rows, cols):
    return _sum_axis(_sum_axis(features, cols, 1), rows, 0)


def _upper(phi, k, scales):
    values = phi.reshape(*phi.shape[:-1], 2, -1)
    upper = values.mean(-1) + k * values.std(-1, unbiased=True) / math.sqrt(values.shape[-1])
    return upper / scales


def _values(phi, k, scales, temperature=0.0):
    upper = _upper(phi, k, scales)
    if temperature == 0:
        return (-upper.amax(-1)).clamp_min(0)
    # Smooth both the CE/KL conjunction and the inactive-tile hinge. This gives
    # a search direction even when no tile initially clears the hard threshold.
    bound = temperature * torch.logsumexp(upper / temperature, dim=-1)
    return temperature * F.softplus(-bound / temperature)


def _gradient(phi, config, scales, temperature):
    with torch.enable_grad():
        p = phi.detach().requires_grad_(True)
        return torch.autograd.grad(_values(p, config.k, scales, temperature).sum(), p)[0]


def _capacities(n, size):
    return torch.tensor([min(size, n - start) for start in range(0, n, size)])


def _balanced_assign(profit, capacities):
    """Regret-ordered capacitated assignment; exact objective checks follow it.

    This is a heuristic for the transportation subproblem, not an optimality
    claim. The final partial group has its own fixed capacity: padding cannot
    drift into interior tiles and then disappear from the exported permutation.
    """
    if profit.shape[1] == 1:
        return torch.zeros(profit.shape[0], dtype=torch.long)
    top = profit.topk(2, dim=1).values
    order = torch.argsort(top[:, 0] - top[:, 1], descending=True, stable=True)
    remaining = capacities.clone()
    labels = torch.empty(profit.shape[0], dtype=torch.long)
    for i in order.tolist():
        group = int(profit[i].masked_fill(remaining == 0, -torch.inf).argmax())
        labels[i] = group
        remaining[group] -= 1
    return labels


def _profit(x, phi, config, scales, temperature):
    gradient = _gradient(phi, config, scales, temperature).flatten(1)
    return x.flatten(1) @ gradient.T


def _assignment(x, labels, size, config, scales, temperature):
    if int(labels.max()) == 0:
        return labels
    phi = _sum_axis(x, labels, 0)
    profit = _profit(x, phi, config, scales, temperature)
    proposed = _balanced_assign(profit, _capacities(x.shape[0], size))
    old = float(_values(phi, config.k, scales, temperature).sum())
    new = float(_values(_sum_axis(x, proposed, 0), config.k, scales, temperature).sum())
    return proposed if new > old + 1e-12 * max(1., abs(old)) else labels


def _swap_refine(x, labels, size, config, scales, temperature, generator):
    groups = int(labels.max()) + 1
    if groups < 2 or config.swap_samples == 0:
        return labels
    phi = _sum_axis(x, labels, 0)
    n = x.shape[0]
    # Half uniform, half directed towards promising destination groups. A
    # capacity-preserving exchange lets rows cross an inactive-tile plateau.
    i = torch.randint(n, (config.swap_samples,), generator=generator)
    j = torch.randint(n, (config.swap_samples,), generator=generator)
    profit = _profit(x, phi, config, scales, temperature)
    profit[torch.arange(n), labels] = -torch.inf
    half = config.swap_samples // 2
    destinations = profit[i[:half]].argmax(-1)
    capacities = _capacities(n, size)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), capacities.cumsum(0)[:-1]))
    positions = (torch.rand(half, generator=generator) * capacities[destinations]).long()
    members = torch.argsort(labels, stable=True)
    j[:half] = members[starts[destinations] + positions]
    keep = labels[i] != labels[j]
    i, j = i[keep], j[keep]
    if not i.numel():
        return labels
    a, b = labels[i], labels[j]
    val = lambda p: _values(p, config.k, scales, temperature).sum(-1)
    old = val(phi)
    # Several live temporaries per candidate; bound scratch rather than allocating
    # [all_pairs, other_groups, 2 * sequences] at once.
    batch = max(1, config.scratch_mb * 1024**2 // (8 * x[0].numel() * 12))
    gains = []
    for start in range(0, len(i), batch):
        sl = slice(start, start + batch)
        delta = x[j[sl]] - x[i[sl]]
        gains.append(val(phi[a[sl]] + delta) + val(phi[b[sl]] - delta)
                     - old[a[sl]] - old[b[sl]])
    gains = torch.cat(gains)
    tolerance = 1e-12 * max(1., float(old.abs().sum()))
    candidates = torch.nonzero(gains > tolerance).flatten()
    candidates = candidates[torch.argsort(gains[candidates], descending=True, stable=True)]
    touched = set()
    result = labels.clone()
    for index in candidates.tolist():
        ga, gb = int(a[index]), int(b[index])
        if ga in touched or gb in touched:
            continue
        result[i[index]], result[j[index]] = gb, ga
        touched.update((ga, gb))
    # Only disjoint GROUP pairs are committed, so their exact gains remain valid.
    return result


def _initial_orders(gain, generator, starts, axes):
    r, c = gain.shape
    identity = (torch.arange(r), torch.arange(c))
    candidates = [('identity', *identity),
                  ('marginal', torch.argsort(gain.sum(1), stable=True),
                   torch.argsort(gain.sum(0), stable=True))]
    if min(r, c) >= 2:
        # Fork the global RNG so randomized SVD is reproducible without altering
        # the caller's random state. The solver itself uses its private generator.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(torch.randint(2**31, (), generator=generator)))
            u, s, v = torch.svd_lowrank(gain, q=min(4, r, c), niter=3)
        candidates.extend([
            ('spectral1', torch.argsort(u[:, 0], stable=True), torch.argsort(v[:, 0], stable=True)),
            ('spectral2', torch.argsort(torch.atan2(u[:, 1]*s[1], u[:, 0]*s[0]), stable=True),
             torch.argsort(torch.atan2(v[:, 1]*s[1], v[:, 0]*s[0]), stable=True))])
    for index in range(max(2, starts)):
        candidates.append((f'random{index}', torch.randperm(r, generator=generator),
                           torch.randperm(c, generator=generator)))
    return [(name, identity[0] if axes == 'cols' else ro,
             identity[1] if axes == 'rows' else co) for name, ro, co in candidates]


@torch.no_grad()
def search_layout(ce, kl, config=None, objective_scales=None):
    """Return permutations, fit-only map, trace, and exact fit objective.

    Orders map NEW positions to OLD indices. The physical weight transform is
    Wp = W[row_perm][:, col_perm], Xp = X[:, col_perm]; the epilogue writes
    Y[:, row_perm] = Xp @ Wp.T. Column permutation cancels algebraically only
    when this same order is applied to the activation operand.
    """
    config = config or SearchConfig()
    config.validate()
    features = _features(ce, kl)
    r, c = features.shape[:2]
    rs, cs = config.tile_rows // config.atom_rows, config.tile_cols // config.atom_cols
    means = features.reshape(r, c, 2, -1).mean(-1)
    if objective_scales is None:
        scales = means.square().mean((0, 1)).sqrt().clamp_min(1e-30)
    else:
        scales = torch.as_tensor(objective_scales, dtype=torch.float64, device='cpu')
        if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Need two finite positive objective scales')
    identity_r, identity_c = torch.arange(r), torch.arange(c)
    rows, cols = _labels(identity_r, rs), _labels(identity_c, cs)
    phi = _tiles(features, rows, cols)
    baseline = float(_values(phi, config.k, scales).sum())
    temperature = max(float(_upper(phi, config.k, scales).abs().median()), 1e-8)
    generator = torch.Generator().manual_seed(config.seed)
    gain = -(means / scales).amax(-1)
    initial = []
    for name, ro, co in _initial_orders(gain, generator, config.starts, config.axes):
        if cs == 1:
            co = identity_c  # Whole type-width moves only relabel tile columns.
        rl, cl = _labels(ro, rs), _labels(co, cs)
        score = float(_values(_tiles(features, rl, cl), config.k, scales, temperature).sum())
        initial.append((score, name, rl, cl))
    # Always refine identity as well as the best other starts. Do not throw away
    # all but the best initializer: their local optima can rank differently.
    initial = initial[:1] + sorted(initial[1:], key=lambda v: v[0], reverse=True)[:config.starts-1]
    best, best_rows, best_cols = baseline, rows.clone(), cols.clone()
    trace = []
    for _, name, rows, cols in initial:
        for temp in (temperature, temperature * .25, 0.):
            for iteration in range(config.rounds + 1):
                phi = _tiles(features, rows, cols)
                hard = float(_values(phi, config.k, scales).sum())
                soft = float(_values(phi, config.k, scales, temp).sum())
                trace.append(dict(start=name, temperature=temp, iteration=iteration,
                                  objective=hard, continuation_objective=soft))
                if hard > best:
                    best, best_rows, best_cols = hard, rows.clone(), cols.clone()
                if iteration == config.rounds:
                    break
                old_rows, old_cols = rows.clone(), cols.clone()
                if config.axes != 'cols':
                    x = _sum_axis(features, cols, 1)
                    rows = _assignment(x, rows, rs, config, scales, temp)
                    for _ in range(config.swap_passes):
                        rows = _swap_refine(x, rows, rs, config, scales, temp, generator)
                if config.axes != 'rows' and cs > 1:
                    x = _sum_axis(features, rows, 0).transpose(0, 1).contiguous()
                    cols = _assignment(x, cols, cs, config, scales, temp)
                    for _ in range(config.swap_passes):
                        cols = _swap_refine(x, cols, cs, config, scales, temp, generator)
                if torch.equal(rows, old_rows) and torch.equal(cols, old_cols):
                    break
    ro, co = torch.argsort(best_rows, stable=True), torch.argsort(best_cols, stable=True)
    row_perm = (ro[:, None] * config.atom_rows + torch.arange(config.atom_rows)).flatten()
    col_perm = (co[:, None] * config.atom_cols + torch.arange(config.atom_cols)).flatten()
    phi = _tiles(features, best_rows, best_cols)
    return dict(schema='mixfp4_task_reorder_v1', config=asdict(config),
                row_atom_perm=ro, col_atom_perm=co, row_perm=row_perm, col_perm=col_perm,
                inverse_row_perm=torch.argsort(row_perm), inverse_col_perm=torch.argsort(col_perm),
                fit_mask=_upper(phi, config.k, scales).amax(-1) < 0,
                fit_objective=best, identity_objective=baseline,
                objective_scales=scales, trace=trace,
                weight_shape=[r * config.atom_rows, c * config.atom_cols],
                padded_shape=[math.ceil(r / rs)*config.tile_rows, math.ceil(c / cs)*config.tile_cols])


def _validate_order(order, n):
    order = torch.as_tensor(order, device='cpu')
    if order.dtype != torch.long or order.shape != (n,) or not torch.equal(
            torch.sort(order).values, torch.arange(n)):
        raise ValueError('Invalid permutation')
    return order


@torch.no_grad()
def elect_layout(ce, kl, layout):
    """Re-elect a FROZEN layout on independent sequences; never change its order."""
    config = SearchConfig(**layout['config'])
    config.validate()
    features = _features(ce, kl)
    r, c = features.shape[:2]
    if layout['weight_shape'] != [r * config.atom_rows, c * config.atom_cols]:
        raise ValueError('Election scores have a different weight shape')
    ro, co = _validate_order(layout['row_atom_perm'], r), _validate_order(layout['col_atom_perm'], c)
    phi = _tiles(features, _labels(ro, config.tile_rows // config.atom_rows),
                 _labels(co, config.tile_cols // config.atom_cols))
    upper = _upper(phi, config.k, torch.ones(2, dtype=torch.float64))
    scales = layout['objective_scales']
    return dict(mask=upper.amax(-1) < 0, upper_ce=upper[..., 0], upper_kl=upper[..., 1],
                objective=float(_values(phi, config.k, scales).sum()), sequences=ce.shape[0])


@torch.no_grad()
def reordered_weight_reference(base, alternative, layout, mask):
    """Dequantized reference IN DEPLOYED ORDER, not a native packed kernel."""
    if base.shape != alternative.shape or list(base.shape) != layout['weight_shape']:
        raise ValueError('Candidate weight shape mismatch')
    config = SearchConfig(**layout['config'])
    config.validate()
    rp = _validate_order(layout['row_perm'], base.shape[0]).to(base.device)
    cp = _validate_order(layout['col_perm'], base.shape[1]).to(base.device)
    want = tuple(math.ceil(n / b) for n, b in zip(base.shape, (config.tile_rows, config.tile_cols)))
    if mask.shape != want or mask.dtype != torch.bool:
        raise ValueError('Type mask shape/dtype mismatch')
    expanded = mask.to(base.device).repeat_interleave(config.tile_rows, 0).repeat_interleave(
        config.tile_cols, 1)[:base.shape[0], :base.shape[1]]
    return torch.where(expanded, alternative[rp][:, cp], base[rp][:, cp])


@torch.no_grad()
def scale_block_scores(gradient, direction):
    """Fine additive CE/KL atoms: one output row by 16 contiguous K elements."""
    if gradient.shape != direction.shape or gradient.ndim != 2 or gradient.shape[1] % 16:
        raise ValueError('Need matching weight gradient/direction with K divisible by 16')
    return (gradient.float() * direction.float()).reshape(gradient.shape[0], -1, 16).sum(-1)
