"""
    Column reordering that changes WHICH ELEMENTS SHARE A SCALE BLOCK.

    WHY THIS IS A DIFFERENT (AND BETTER) LEVER THAN `reorder.py`
    -----------------------------------------------------------
    `reorder.py` permutes whole 16-column chunks so that the E0M3/E2M1 tag grid stays invariant and
    its objective stays exact. The price of that choice is fatal: a chunk-of-16 permutation CANNOT
    change which elements share a scale block, so it cannot touch the block maxima. All it can do is
    reshuffle which tiles vote together, and measurement (`REPORT.md`) shows that lever is worth
    nothing -- the search beats a cell-shuffled control by +0.003 and perplexity moves with no
    consistent sign.

    This module permutes INDIVIDUAL columns, which is what actually matters for NVFP4:

        block_scale = alpha * block_max / grid_max

    Every element of a 16-element block is quantized on a grid whose step is set by the LARGEST
    element in that block. One outlier channel forces a coarse step on the other fifteen. Grouping
    channels of similar magnitude means the outliers are concentrated into a few blocks that pay for
    themselves, instead of poisoning one block each.

    THE OBJECTIVE, AND WHY IT HAS STRUCTURE
    ---------------------------------------
    Round-to-nearest on a uniform grid of step `s` has squared error ~ s^2/12 per element, and
    `s ∝ block_max`, so the total squared error is approximately

        sum over blocks of  block_max^2                        (`block_cost`)

    For ONE row this is a partition of K columns into groups of 16 minimizing the sum of squared
    group maxima -- and that is solved EXACTLY by sorting. The only thing that makes it hard is that
    a single column order has to serve every row of the matrix at once, because the permutation is
    applied to the shared input-channel axis.

    So the question is entirely empirical and cheap to answer: do the rows agree about which columns
    are large? `column_profile_agreement` measures it. If the column magnitudes are consistent
    across rows, sorting wins for a real, structural reason; if each row has its own idea of which
    columns are big, no single permutation can help and this dies the same way the tag-grid version
    did. That check runs before any search.

    DEPLOYABILITY is identical to `reorder.py` (see ALGORITHM.md §7): a per-layer column permutation
    is free for `down_proj` (absorbed into gate/up_proj's rows) and within-head for `o_proj`; the
    other matrices share the residual axis and admit one global permutation.
"""

from typing import Optional

import torch


@torch.no_grad()
def block_cost(w, groupsize: int = 16, cols=None, importance=None):
    """
        sum over 16-element blocks of block_max^2 -- the quantity a column permutation controls.

        `importance` (length K, the per-input-channel E[x_j^2]) weights each block by the mean
        importance of the channels in it, turning raw weight error into the diagonal-Hessian
        estimate of layer output error. Optional; without it this is plain MSE.
    """
    x = w if cols is None else w[:, cols]
    x = x.reshape(x.shape[0], -1, groupsize)
    cost = x.abs().amax(dim=-1).pow(2)                       # (M, num_block)
    if importance is not None:
        imp = importance if cols is None else importance[cols]
        cost = cost * imp.reshape(1, -1, groupsize).mean(dim=-1)
    return float(cost.sum())


@torch.no_grad()
def column_profile_agreement(w, num_sample: int = 512, generator=None):
    """
        Do the rows agree about which columns are large?

        Sorting columns by magnitude can only help if `|W[i, j]|` is dominated by a per-COLUMN
        factor shared across rows. Modelling `|W[i,j]| ~ r_i * c_j`, the agreement is the fraction
        of the variance of `log|W|` explained by the column term -- i.e. exactly the two-way
        decomposition, but applied to log-magnitudes, where a shared multiplicative column scale is
        what sorting exploits.

        Returns (row_share, col_share, resid_share). A large `col_share` is the go/no-go for this
        whole module, and it is the statistic the tag-grid approach failed on (col_share 0.004).
    """
    x = w.to(torch.float32).abs()
    x = torch.log(x.clamp(min=x[x > 0].min() if (x > 0).any() else 1e-30))
    if x.shape[0] > num_sample:
        idx = torch.randperm(x.shape[0], generator=generator)[:num_sample]
        x = x[idx]
    M, N = x.shape
    mu = x.mean()
    a  = x.mean(dim=1, keepdim=True) - mu
    b  = x.mean(dim=0, keepdim=True) - mu
    tot = (x - mu).pow(2).sum().clamp(min=1e-30)
    return (float(a.pow(2).sum() * N / tot),
            float(b.pow(2).sum() * M / tot),
            float((x - mu - a - b).pow(2).sum() / tot))


@torch.no_grad()
def magnitude_order(w, stat: str = "rms", importance=None):
    """
        Sort columns by a per-column magnitude statistic.

        For a single row, sorting is the EXACT minimizer of `sum block_max^2`; across rows it is the
        natural heuristic, and which summary statistic best stands in for "this column is large"
        is an empirical question, hence the choice.

          "rms"  -- sqrt(mean_i W[i,j]^2), the column's typical magnitude
          "max"  -- max_i |W[i,j]|, dominated by the single worst row
          "mean" -- mean_i |W[i,j]|
          "q90"  -- the 90th percentile of |W[:,j]|, a robust stand-in for "large"
    """
    x = w.to(torch.float32).abs()
    if stat == "rms":
        key = x.pow(2).mean(dim=0).sqrt()
    elif stat == "max":
        key = x.amax(dim=0)
    elif stat == "mean":
        key = x.mean(dim=0)
    elif stat == "q90":
        key = torch.quantile(x, 0.9, dim=0)
    else:
        raise ValueError(f'Unknown stat "{stat}".')
    if importance is not None:
        # a channel that matters more should be placed by how much error it CONTRIBUTES
        key = key * importance.to(key.dtype).clamp(min=0).sqrt()
    return torch.argsort(key, descending=True)


@torch.no_grad()
def refine_column_order(w, cols, groupsize: int = 16, rounds: int = 6, samples: int = 20000,
                        importance=None, generator=None, verbose: bool = False):
    """
        Kernighan-Lin swap refinement on `block_cost`, starting from `cols`.

        Sorting is exact for one row but only a heuristic once a single order must serve every row,
        so pairwise swaps between different blocks recover part of that gap. Each candidate swap is
        evaluated exactly, on the two blocks it touches, and accepted only if it lowers the cost.
    """
    cols = cols.clone()
    K    = cols.numel()
    nb   = K // groupsize
    if nb < 2:
        return cols

    for it in range(rounds):
        x    = w[:, cols].reshape(w.shape[0], nb, groupsize)
        best = x.abs().amax(dim=-1)                                   # (M, nb)
        imp  = None
        if importance is not None:
            imp = importance[cols].reshape(nb, groupsize).mean(dim=-1)

        i = torch.randint(0, K, (samples,), generator=generator)
        j = torch.randint(0, K, (samples,), generator=generator)
        bi, bj = i // groupsize, j // groupsize
        keep = bi != bj
        i, j, bi, bj = i[keep], j[keep], bi[keep], bj[keep]
        if i.numel() == 0:
            break

        accepted, touched = 0, torch.zeros(nb, dtype=torch.bool)
        # evaluate in one batch, then apply greedily over disjoint block pairs so every accepted
        # swap's delta stays exactly valid
        col_i, col_j = w[:, cols[i]], w[:, cols[j]]                   # (M, S)
        gain = torch.zeros(i.numel())
        for lo in range(0, i.numel(), 4096):
            sl = slice(lo, min(lo + 4096, i.numel()))
            bi_s, bj_s = bi[sl], bj[sl]
            # recompute the two affected block maxima with the two columns exchanged
            xi = x[:, bi_s, :].clone()
            xj = x[:, bj_s, :].clone()
            pi = (i[sl] % groupsize)
            pj = (j[sl] % groupsize)
            ar = torch.arange(bi_s.numel())
            xi[:, ar, pi] = col_j[:, sl]
            xj[:, ar, pj] = col_i[:, sl]
            new_i = xi.abs().amax(dim=-1).pow(2)                      # (M, S)
            new_j = xj.abs().amax(dim=-1).pow(2)
            old_i = best[:, bi_s].pow(2)
            old_j = best[:, bj_s].pow(2)
            if imp is not None:
                d = ((old_i - new_i) * imp[bi_s] + (old_j - new_j) * imp[bj_s]).sum(dim=0)
            else:
                d = (old_i - new_i + old_j - new_j).sum(dim=0)
            gain[sl] = d

        order = torch.argsort(gain, descending=True)
        total = 0.0
        for k in order.tolist():
            if gain[k] <= 0:
                break
            a, b = int(bi[k]), int(bj[k])
            if touched[a] or touched[b]:
                continue
            touched[a] = touched[b] = True
            ci, cj = int(i[k]), int(j[k])
            cols[ci], cols[cj] = cols[cj].clone(), cols[ci].clone()
            total += float(gain[k])
            accepted += 1
        if verbose:
            print(f"      refine round {it}: {accepted} swaps, cost -{total:.4g}")
        if accepted == 0:
            break
    return cols


@torch.no_grad()
def spread_order(w, stat: str = "rms", groupsize: int = 16, importance=None):
    """
        The OPPOSITE of `magnitude_order`, and the right pre-conditioner for rotation.

        A Hadamard rotation over a chunk helps precisely when that chunk holds ONE outlier among
        otherwise normal values: it turns {big, small x 15} into 16 medium values and collapses
        block_max by ~4x. If every element of the chunk is large, rotation redistributes nothing and
        the block still needs a coarse scale.

        So rotation wants outliers SPREAD, one per chunk -- exactly what sorting destroys. This
        sorts by magnitude and then deals the columns round-robin into chunks, so each chunk
        receives one column from every magnitude stratum: one large, one medium, ..., one small.
        Every chunk then has a high max/rms and every chunk is worth rotating.

        `magnitude_order` and `spread_order` are the two extremes of the same axis, which makes them
        a clean A/B for what rotation actually wants.
    """
    order = magnitude_order(w, stat, importance)
    K     = order.numel()
    nb    = K // groupsize
    if nb < 1:
        return order
    # deal round-robin: position p of chunk c receives the (p * nb + c)-th largest column
    dealt = order[: nb * groupsize].reshape(groupsize, nb).transpose(0, 1).reshape(-1)
    return torch.cat([dealt, order[nb * groupsize:]])


@torch.no_grad()
def rotation_split_error(w_scaled, cols=None, groupsize: int = 16, clip: str = "a1",
                         metric: str = "mse", rotate_size: int = 16, min_gain: float = 0.0):
    """
        Per-column-chunk rotation, evaluated exactly, for a given column order.

        Returns the total E2M1 error under three policies -- never rotate, always rotate, and rotate
        a chunk only when it beats no-rotation by at least `min_gain` (the `rotmin<t>` rule, which
        CLAUDE.md measures as the best realizable configuration in the study).

        The decision is per COLUMN CHUNK and shared down every row, because the activation side
        rotates a chunk for all tokens at once -- so the per-chunk errors are summed over rows
        before the comparison. A Hadamard is orthogonal, so a chunk's squared error is the same
        measured in either basis and the two candidates are directly comparable.

        WHY `rotate_size` IS THE INTERESTING KNOB HERE. QuaRot rotates the WHOLE hidden dimension
        with one randomized Hadamard, absorbed into the weights. Against a full-dimension rotation a
        permutation is provably useless -- a dense Hadamard already mixes every channel, and `PH` is
        just another orthogonal matrix, so reordering first changes nothing. Reordering has leverage
        only for a BLOCK-DIAGONAL rotation, which is what a `rotate_size`-wide chunked Hadamard is
        and what is cheap to apply on the fly. There, which columns land in a chunk decides how much
        the rotation can dissolve, so the prediction is that any gain from `spread_order` SHRINKS as
        `rotate_size` grows and vanishes once a chunk spans the whole dimension.
    """
    from .quantizer import _rotate_chunks
    from .reorder import scale_block_gain

    x = w_scaled if cols is None else w_scaled[:, cols]
    assert rotate_size % groupsize == 0, \
        f"rotate_size {rotate_size} must be a multiple of the scale block {groupsize}"
    per = rotate_size // groupsize                     # scale blocks per rotation chunk

    _, e_id, _  = scale_block_gain(x, groupsize, metric, clip, return_losses=True)
    _, e_rot, _ = scale_block_gain(_rotate_chunks(x, rotate_size), groupsize, metric, clip,
                                   return_losses=True)

    # the rotation decision is per ROTATION CHUNK and shared down the rows, so fold the scale-block
    # errors it spans together before comparing
    fold  = lambda e: e.to(torch.float64).sum(dim=0).reshape(-1, per).sum(dim=-1)
    c_id  = fold(e_id)                                 # (num_rotation_chunk,)
    c_rot = fold(e_rot)
    take  = c_rot < c_id * (1.0 - min_gain)
    return dict(
        norot=float(c_id.sum()),
        allrot=float(c_rot.sum()),
        percol=float(torch.where(take, c_rot, c_id).sum()),
        rotated_share=float(take.to(torch.float32).mean()),
    )
