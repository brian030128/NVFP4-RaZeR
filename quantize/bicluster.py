"""Checkerboard biclustering of the CE/KL score matrix into legal type tiles.

The reordering problem is a co-clustering problem stated in other words. Scores
form a matrix `G[i, j]` over row atoms and 16-element column groups; a type tile
is the submatrix cut out by one row group and one column group, and it elects a
single format. Wanting tiles whose members agree is wanting the K x L grid of
blocks to be internally homogeneous, which is the checkerboard model of spectral
biclustering (Kluger et al. 2003) and of information-theoretic co-clustering
(Dhillon et al. 2003).

`quantize/task_reorder.py` already computes a degenerate version: it takes the
plain SVD of the unnormalized gain matrix and sorts rows by one singular vector.
Two things are missing.

**Normalization.** Output channels differ enormously in gradient magnitude, so an
unnormalized SVD reports which channels are loud rather than which format they
prefer, and the leading singular vector is the likeliest place for that nuisance
to sit. Standardizing rows and columns alternately removes both marginals and
leaves the interaction structure, which is the checkerboard pattern itself.

**Partition class.** Sorting a one-dimensional projection and cutting it into
equal blocks can only express interval partitions of a line. Balanced k-means on
a rank-q embedding expresses any partition whose cells are separable in q
dimensions, which is strictly larger and still a small model.

The point of the low rank is regularization, not speed. The deployed search fits
about 14,592 free assignments for one Llama gate/up matrix against 64 calibration
documents, and a sign-flip placebo shows it extracts as much objective from noise
as from signal. A rank-q biclustering commits to structure shared across many
rows and columns, and exposes exactly one tunable number, q, which can be chosen
on held-out sequences.

This module returns a layout in the same schema as `search_layout`, so
`elect_layout` and every downstream consumer are unchanged. It performs no format
election and no hinge refinement: grouping and electing stay separate, and
electing keeps the frozen CE/KL rule.
"""
import math

import torch

from quantize.task_reorder import (SearchConfig, _balanced_assign, _capacities, _features,
                                   _labels, _tiles, _upper, _values)


def standardize(matrix, passes=3, eps=1e-12):
    """Alternately z-score rows and columns, leaving the interaction structure.

    Removes both marginals, so neither a loud output channel nor a loud column
    group can present itself as a format preference. Three passes is the usual
    stopping point; this is not iterated to a fixed point.
    """
    out = matrix.clone()
    for _ in range(max(0, passes)):
        out = (out - out.mean(1, keepdim=True)) / out.std(1, keepdim=True).clamp_min(eps)
        out = (out - out.mean(0, keepdim=True)) / out.std(0, keepdim=True).clamp_min(eps)
    if not torch.isfinite(out).all():
        raise ValueError('Standardization produced a nonfinite score matrix')
    return out


def structure(matrix, rank=4):
    """Report whether the checkerboard is row dominant, column dominant, or both.

    For each retained component, compares how much of the singular vector pair's
    energy is concentrated in few rows against few columns, using a normalized
    participation ratio. Values near 1 mean the component spreads over the whole
    axis, near 0 that it lives on a handful of entries. A component that is
    structured on K and flat on N is column dominant, which is what
    `CLAUDE.md` predicts for this data, and it would mean the row axis deserves
    little of the grouping budget.

    This is a descriptive diagnostic. It selects nothing.
    """
    u, s, v = torch.svd_lowrank(matrix, q=min(rank + 1, *matrix.shape), niter=4)
    report = []
    for k in range(min(rank, s.numel())):
        def spread(vec):
            p = vec.square()
            total = float(p.sum())
            if total <= 0:
                return float('nan')
            p = p / total
            return float(1.0 / (p.square().sum() * p.numel()))
        report.append(dict(component=k, singular_value=float(s[k]),
                           row_spread=spread(u[:, k]), column_spread=spread(v[:, k])))
    energy = float(s.square().sum())
    for row in report:
        row['energy_fraction'] = row['singular_value'] ** 2 / energy if energy > 0 else float('nan')
    return report


def balanced_kmeans(points, size, iterations=10, seed=0):
    """k-means with exact per-cluster capacity, returning group labels.

    Capacity is not optional here: a type tile holds exactly `size` rows, so the
    assignment step is a capacitated transportation problem rather than a nearest
    centroid lookup. It reuses the deployed regret-ordered balanced assignment,
    which is a heuristic for that subproblem and does not claim optimality.
    Initial centroids come from cutting the leading coordinate, which makes the
    result deterministic and at least as expressive as the sort-and-cut layout
    the deployed search starts from.
    """
    n = points.shape[0]
    groups = max(1, math.ceil(n / size))
    if groups == 1:
        return torch.zeros(n, dtype=torch.long)
    capacities = _capacities(n, size)
    labels = _labels(torch.argsort(points[:, 0], stable=True), size)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(max(1, iterations)):
        centroids = torch.zeros(groups, points.shape[1], dtype=points.dtype)
        centroids.index_add_(0, labels, points)
        counts = torch.zeros(groups, dtype=points.dtype).index_add_(
            0, labels, torch.ones(n, dtype=points.dtype))
        empty = counts == 0
        centroids = centroids / counts.clamp_min(1).unsqueeze(1)
        if bool(empty.any()):
            # Reseed a vacated centroid onto a random point rather than leaving
            # it at the origin, where it would attract arbitrary members.
            picks = torch.randint(n, (int(empty.sum()),), generator=generator)
            centroids[empty] = points[picks]
        # Profit is negative squared distance, so the capacitated assignment
        # maximizes exactly what k-means minimizes.
        profit = -torch.cdist(points, centroids).square()
        proposed = _balanced_assign(profit, capacities)
        if torch.equal(proposed, labels):
            break
        labels = proposed
    return labels


def _order_from_labels(labels, size):
    """Permutation placing each cluster's members contiguously, capacity exact."""
    order = torch.argsort(labels, stable=True)
    if not torch.equal(_labels(order, size), labels):
        raise ValueError('Balanced labels do not match their contiguous ordering')
    return order


@torch.no_grad()
def bicluster_layout(ce, kl, config=None, rank=4, normalize=True, passes=3,
                     iterations=10, objective_scales=None):
    """Return a layout in `search_layout`'s schema, fitted by co-clustering only.

    Rows and columns are embedded jointly from the rank-q factorization of the
    standardized gain matrix and grouped by capacitated k-means. No hinge
    objective is optimized and no pair swaps are made, so nothing here selects on
    whether a tile's confidence bound happens to clear zero.
    """
    config = config or SearchConfig()
    config.validate()
    if rank < 1:
        raise ValueError('rank must be a positive integer')
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

    # Same gain convention as the deployed search: positive favors E0M3 on the
    # worse of the two objectives, so the conjunction is respected up front.
    gain = -(means / scales).amax(-1)
    matrix = standardize(gain, passes) if normalize else gain
    diagnostics = structure(matrix, rank)
    q = min(rank, r, c)
    u, s, v = torch.svd_lowrank(matrix, q=q, niter=4)
    row_points = (u[:, :q] * s[:q]).contiguous()
    col_points = (v[:, :q] * s[:q]).contiguous()

    identity_r, identity_c = torch.arange(r), torch.arange(c)
    baseline = float(_values(_tiles(features, _labels(identity_r, rs), _labels(identity_c, cs)),
                             config.k, scales).sum())
    row_order = identity_r if config.axes == 'cols' else _order_from_labels(
        balanced_kmeans(row_points, rs, iterations, config.seed), rs)
    # A single column group per tile means permuting columns only relabels tiles.
    col_order = identity_c if (config.axes == 'rows' or cs == 1) else _order_from_labels(
        balanced_kmeans(col_points, cs, iterations, config.seed), cs)

    rows, cols = _labels(row_order, rs), _labels(col_order, cs)
    phi = _tiles(features, rows, cols)
    fit_objective = float(_values(phi, config.k, scales).sum())
    ro, co = torch.argsort(rows, stable=True), torch.argsort(cols, stable=True)
    row_perm = (ro[:, None] * config.atom_rows + torch.arange(config.atom_rows)).flatten()
    col_perm = (co[:, None] * config.atom_cols + torch.arange(config.atom_cols)).flatten()
    return dict(schema='mixfp4_task_reorder_v1', config=asdict_config(config),
                row_atom_perm=ro, col_atom_perm=co, row_perm=row_perm, col_perm=col_perm,
                inverse_row_perm=torch.argsort(row_perm), inverse_col_perm=torch.argsort(col_perm),
                fit_mask=_upper(phi, config.k, scales).amax(-1) < 0,
                fit_objective=fit_objective, identity_objective=baseline,
                objective_scales=scales,
                trace=[dict(start='bicluster', rank=q, normalized=bool(normalize),
                            objective=fit_objective, structure=diagnostics)],
                weight_shape=[r * config.atom_rows, c * config.atom_cols],
                padded_shape=[math.ceil(r / rs) * config.tile_rows,
                              math.ceil(c / cs) * config.tile_cols])


def asdict_config(config):
    from dataclasses import asdict
    return asdict(config)
