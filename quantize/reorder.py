"""
    Row / column reordering for MixFP4 type blocks -- a balanced co-clustering search.

    THE PROBLEM
    -----------
    Fix a weight matrix W (M x K). MixFP4 cuts it into 16-element SCALE blocks along K, each with
    its own ue4m3 scale, and into much coarser TYPE blocks (BM x BK) that must elect ONE element
    data type (E2M1 or E0M3) for every scale block they contain. Write

        N        = K / 16                                  (scale blocks per row)
        G[i, j]  = loss_E2M1(block i,j) - loss_E0M3(block i,j)          G in R^(M x N)

    so G[i, j] > 0 means scale block (i, j) would rather be E0M3. G is the "tag grid": a signed,
    WEIGHTED tag per cell, not a binary one, because the election sums the magnitudes.

    A type block is a tile of BM rows x c = BK/16 columns of G. Under the plain `argmin` election
    the loss a tile saves relative to all-E2M1 is max(0, sum of its cells), so the total gain is

        F(R, C) = sum_{b in R} sum_{c in C}  v( sum_{i in b, j in c} G[i, j] )

    where R partitions the M rows into groups of exactly BM and C partitions the N scale-block
    columns into groups of exactly c. Reordering rows and columns of W is exactly choosing R and C
    (the ORDER inside a group is irrelevant -- only the partition matters -- and any equal-size
    partition is realizable by some permutation).

    F is bounded above by sum_ij max(0, G[i,j]), which is what a 1x16 type block achieves, so

        recovered = F(R, C) / sum_ij relu(G[i, j])

    is a directly interpretable score: the fraction of the (non-realizable) per-scale-block choice
    that a realizable tile shape keeps. `mix_4_6` today is this quantity at R = C = identity.

    WHY COLUMNS MOVE IN CHUNKS OF 16
    --------------------------------
    Permuting INDIVIDUAL columns changes which elements share a scale block, hence the block
    maximum, hence G itself -- the objective would depend on the variable being optimized. Permuting
    whole 16-column chunks makes G exactly permutation-covariant, so the search above is exact.
    Finer permutation is still available afterwards, CONFINED TO ONE COLUMN GROUP, where it can only
    change cell values inside tiles that already exist and never the partition.

    THIS IS NP-HARD
    ---------------
    Balanced co-clustering / biclustering with fixed cluster cardinalities. It contains balanced
    k-way graph partitioning. Classical relatives: the Bond Energy Algorithm (McCormick 1972),
    Consecutive Block Minimization / the Consecutive Ones Property (NP-hard, Kou 1977), matrix
    seriation, and Dhillon's spectral / information-theoretic co-clustering. The solver here follows
    the standard recipe from that literature: spectral initializer -> alternating balanced
    assignment (Lloyd) -> Kernighan-Lin swap refinement.

    THE STRUCTURAL FACT THAT MAKES IT CHEAP
    ---------------------------------------
    Every election rule in `quantizer._elect_e0m3` is a function of a handful of ADDITIVE tile
    statistics. Mapping each cell g to the feature vector

        phi(g) = [ g,  relu(g),  relu(-g),  1{g > 0},  g^2 ]

    the tile statistic is just the sum of phi over its cells, and

        argmin      elect iff  s > 0
        harm(l)     elect iff  p > l * n                (the `h<l>` rule)
        dominance   elect iff  n == 0 and s > 0
        vote(t)     elect iff  cnt > t * nb and s > 0
        margin(z)   elect iff  s/nb > z * sqrt(sq/nb - (s/nb)^2) / sqrt(nb)

    all read off (s, p, n, cnt, sq). So the objective is a sum over tiles of a cheap function of a
    5-vector, every move has an O(#tiles touched) incremental evaluation, and the search optimizes
    the DEPLOYED election rule rather than a proxy for it.
"""

from typing import Optional

import torch

from .quantizer import CLIP_PRESETS, _quant_e0m3, _quant_e2m1, _selection_loss


NUM_FEAT = 5    # s, p, n, cnt, sq


# ----------------------------------------------------------------------------------------------
# The tag grid
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def scale_block_gain(w_scaled, groupsize: int = 16, metric: str = "mse", clip: str = "base",
                     importance=None, return_losses: bool = False):
    """
        Per SCALE block, how much it prefers E0M3 over E2M1:

            G[i, j] = loss_E2M1(i, j) - loss_E0M3(i, j)

        computed at the 1x16 granularity with each block free to pick its own clip ratio -- exactly
        the quantity `quant_mix_4_6` later sums over a type block, and exactly `row_preference`
        before its row-sum. `w_scaled` is (M, K) in the globally scaled domain.

        `importance` (optional, length K) is the per-input-channel E[x_j^2]; passing it switches the
        loss to the diagonal-Hessian estimate of the layer output error instead of plain MSE.
    """
    E2M1_MAX, E0M3_MAX = 6.0, 7.0
    FP8_SCALE_MAX, FP8_SCALE_MIN = 448.0, 2 ** (-9)

    blocks    = w_scaled.reshape(w_scaled.shape[0], -1, groupsize).to(torch.float32)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    alphas    = CLIP_PRESETS[clip]

    imp = None
    if importance is not None:
        imp = importance.to(blocks.dtype).reshape(1, -1, groupsize)

    def best(quant_fn, grid_max, alpha_list):
        best_err = None
        for alpha in alpha_list:
            scale = (block_max * (alpha / grid_max)).clamp(
                max=FP8_SCALE_MAX, min=FP8_SCALE_MIN
            ).to(torch.float8_e4m3fn).to(blocks.dtype)
            err = _selection_loss(blocks, quant_fn(blocks, scale), metric, imp)
            best_err = err if best_err is None else torch.minimum(best_err, err)
        return best_err

    err_e2m1 = best(_quant_e2m1, E2M1_MAX, alphas["e2m1"])
    err_e0m3 = best(_quant_e0m3, E0M3_MAX, alphas["e0m3"])
    gain     = (err_e2m1 - err_e0m3).squeeze(-1)        # (M, N)
    if not return_losses:
        return gain
    # The all-E2M1 total is the DENOMINATOR that turns a "fraction of the 1x16 ceiling" back into a
    # fraction of the quantization error actually being paid. It is the loss of plain NVFP4 under
    # this clip preset -- i.e. what the E0M3 election is trying to improve on.
    return gain, err_e2m1.squeeze(-1), err_e0m3.squeeze(-1)


@torch.no_grad()
def gain_features(gain):
    """ phi(g) = [g, relu(g), relu(-g), 1{g>0}, g^2], stacked on a trailing axis. """
    g = gain.to(torch.float64)      # tile sums cancel to ~0; float32 loses the sign of small s
    return torch.stack(
        [g, g.clamp(min=0), (-g).clamp(min=0), (g > 0).to(g.dtype), g * g], dim=-1
    )


# ----------------------------------------------------------------------------------------------
# The election rule, as a function of the additive tile statistics
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def elect_mask(phi, rule: str = "argmin", margin: float = 0.0, cells_per_tile: int = 1,
               eps: float = 0.0):
    """
        `phi` is (..., NUM_FEAT). Returns a boolean mask of the same leading shape, matching
        `quantizer._elect_e0m3` term for term.
    """
    s, p, n, cnt, sq = phi.unbind(-1)

    if rule == "never":
        return torch.zeros_like(s, dtype=torch.bool)
    if rule == "always":
        return torch.ones_like(s, dtype=torch.bool)
    if rule == "argmin":
        return s > 0
    if rule == "dominance":
        return (n <= eps) & (s > 0)
    if rule == "harm":
        return p > margin * n
    if rule == "vote":
        return (cnt > margin * cells_per_tile) & (s > 0)
    if rule == "margin":
        nb   = float(cells_per_tile)
        mean = s / nb
        var  = (sq / nb - mean * mean).clamp(min=0)
        return mean > margin * var.sqrt() / (nb ** 0.5)
    raise ValueError(f'Unsupported election rule "{rule}".')


@torch.no_grad()
def tile_value(phi, rule: str = "argmin", margin: float = 0.0, cells_per_tile: int = 1):
    """ Loss reduction the tile realizes versus all-E2M1: `elect * total_gain`. """
    s = phi[..., 0]
    return torch.where(elect_mask(phi, rule, margin, cells_per_tile), s, torch.zeros_like(s))


# ----------------------------------------------------------------------------------------------
# Partition bookkeeping
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _group_sums(x, labels, num_group: int):
    """ x: (n_item, ...); returns (num_group, ...) with rows of the same label summed. """
    out = torch.zeros((num_group,) + x.shape[1:], dtype=x.dtype, device=x.device)
    return out.index_add_(0, labels, x)


@torch.no_grad()
def _colgroup_sums(feat, clab, num_colgroup: int):
    """ feat: (M, N, F) -> (M, num_colgroup, F). """
    out = torch.zeros(feat.shape[0], num_colgroup, feat.shape[-1],
                      dtype=feat.dtype, device=feat.device)
    return out.index_add_(1, clab, feat)


@torch.no_grad()
def _labels_from_order(order, group_size: int, num_item: int):
    lab = torch.empty(num_item, dtype=torch.long)
    lab[order] = torch.arange(num_item) // group_size
    return lab


@torch.no_grad()
def objective(feat, rlab, clab, num_rowgroup: int, num_colgroup: int,
              rule: str, margin: float, cells_per_tile: int):
    A   = _colgroup_sums(feat, clab, num_colgroup)
    phi = _group_sums(A, rlab, num_rowgroup)
    return float(tile_value(phi, rule, margin, cells_per_tile).sum())


# ----------------------------------------------------------------------------------------------
# Stage 1 -- initializers
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _spectral_orders(gain, rank: int = 1):
    """
        Rank-1 (or rank-2 angular) seriation of the tag grid.

        G ~ sigma * u v^T means sign(G[i,j]) ~ sign(u_i) sign(v_j): a CHECKERBOARD, which is exactly
        the structure a row-partition x column-partition product can represent. Sorting rows by u
        and columns by v therefore puts cells that agree on their preferred data type into the same
        tiles. If G's sign structure is far from rank-1, no product partition can do well, and the
        rank-1 fit is a cheap go/no-go test for the whole idea.

        Sign ambiguity of the SVD is irrelevant here: flipping (u, v) -> (-u, -v) reverses both
        orders, which yields the same PARTITION with its groups relabelled.
    """
    g = gain.to(torch.float32)
    q = max(min(rank + 4, min(g.shape) - 1), 2)
    try:
        U, S, V = torch.svd_lowrank(g, q=q, niter=4)
    except Exception:                                    # tiny / degenerate matrices
        U, S, Vh = torch.linalg.svd(g, full_matrices=False)
        V = Vh.transpose(-1, -2)

    if rank <= 1 or S.shape[0] < 2:
        return torch.argsort(U[:, 0], descending=True), torch.argsort(V[:, 0], descending=True)

    ru = torch.atan2(U[:, 1] * S[1], U[:, 0] * S[0])
    rv = torch.atan2(V[:, 1] * S[1], V[:, 0] * S[0])
    return torch.argsort(ru), torch.argsort(rv)


@torch.no_grad()
def _initial_partitions(gain, feat, block_m: int, chunks: int, rule: str, margin: float,
                        cells_per_tile: int, generator=None, num_random: int = 2):
    """
        Several cheap starting points; the caller keeps whichever scores best. `marginal` is the
        1-D generalization of the existing `permute="rows"` mode to both axes.
    """
    M, N  = gain.shape
    n_rg  = M // block_m
    n_cg  = N // chunks
    cands = {}

    cands["identity"] = (torch.arange(M), torch.arange(N))
    cands["marginal"] = (torch.argsort(gain.sum(dim=1), descending=True),
                         torch.argsort(gain.sum(dim=0), descending=True))
    cands["spectral1"] = _spectral_orders(gain, rank=1)
    cands["spectral2"] = _spectral_orders(gain, rank=2)
    # spectral rows only / columns only: isolates which axis the structure lives on
    cands["spectral1_rows"] = (cands["spectral1"][0], torch.arange(N))
    cands["spectral1_cols"] = (torch.arange(M), cands["spectral1"][1])

    for r in range(num_random):
        cands[f"random{r}"] = (torch.randperm(M, generator=generator),
                               torch.randperm(N, generator=generator))

    scored = {}
    for name, (ro, co) in cands.items():
        rlab = _labels_from_order(ro, block_m, M)
        clab = _labels_from_order(co, chunks, N)
        scored[name] = (objective(feat, rlab, clab, n_rg, n_cg, rule, margin, cells_per_tile),
                        rlab, clab)
    return scored


# ----------------------------------------------------------------------------------------------
# Stage 2 -- alternating balanced assignment (Lloyd for co-clustering)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def balanced_assign(profit, group_size: int):
    """
        Assign each item to one of `num_group` groups of EXACTLY `group_size` items, maximizing the
        total profit. This is a transportation problem; solved here by the regret-ordered greedy
        used in balanced k-means -- items whose best group beats their second best by the most get
        to choose first. Stage 3 repairs what greedy leaves on the table.
    """
    n_item, n_group = profit.shape
    assert n_item == n_group * group_size

    if n_group == 1:
        return torch.zeros(n_item, dtype=torch.long)

    top2   = profit.topk(2, dim=1).values
    regret = top2[:, 0] - top2[:, 1]
    order  = torch.argsort(regret, descending=True)

    remaining = torch.full((n_group,), group_size, dtype=torch.long)
    lab       = torch.empty(n_item, dtype=torch.long)
    neg       = torch.full((), torch.finfo(profit.dtype).min, dtype=profit.dtype)

    for i in order.tolist():
        b = int(torch.where(remaining > 0, profit[i], neg).argmax())
        lab[i] = b
        remaining[b] -= 1
    return lab


@torch.no_grad()
def _lloyd_step(X, lab, num_group: int, group_size: int,
                rule: str, margin: float, cells_per_tile: int):
    """
        One alternating half-step. `X` is (n_item, n_other_group, F): the feature mass item `i`
        contributes to each tile column. Linearize the objective around the CURRENT elections:
        with the elect bits s[g, o] held fixed the objective is

            sum_i  sum_o  s[g(i), o] * X[i, o, 0]

        which is linear in the assignment, so the optimal reassignment is a balanced transportation
        problem. Re-electing after the move and iterating is Lloyd's algorithm.
    """
    phi    = _group_sums(X, lab, num_group)                              # (n_group, n_other, F)
    elect  = elect_mask(phi, rule, margin, cells_per_tile).to(X.dtype)   # (n_group, n_other)
    profit = X[..., 0] @ elect.transpose(0, 1)                           # (n_item, n_group)
    return balanced_assign(profit.to(torch.float32), group_size)


# ----------------------------------------------------------------------------------------------
# Stage 3 -- Kernighan-Lin swap refinement on the exact objective
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _swap_round(X, lab, num_group: int, rule: str, margin: float, cells_per_tile: int,
                samples: int, generator=None):
    """
        Sample `samples` candidate item swaps between different groups, evaluate every delta
        exactly and in one batch, and greedily accept the positive ones that touch disjoint groups
        (so the deltas stay valid without recomputation). Returns (new_lab, gain, n_accept).

        Swapping items i, j between groups g1, g2 changes only those two rows of tiles, and the tile
        statistics are additive, so the delta is a closed-form function of X[i] and X[j].
    """
    n_item = X.shape[0]
    if num_group < 2:
        return lab, 0.0, 0
    phi = _group_sums(X, lab, num_group)

    i = torch.randint(0, n_item, (samples,), generator=generator)
    j = torch.randint(0, n_item, (samples,), generator=generator)
    keep = lab[i] != lab[j]
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return lab, 0.0, 0

    g1, g2 = lab[i], lab[j]
    val = lambda t: tile_value(t, rule, margin, cells_per_tile).sum(dim=-1)

    # Batched: the delta tensors are (batch, n_other_group, NUM_FEAT), which on a down_proj-sized
    # column pass would be gigabytes in one shot.
    batch  = max(1, int(2 ** 22 // max(X.shape[1] * X.shape[2], 1)))
    deltas = []
    for lo in range(0, i.numel(), batch):
        sl  = slice(lo, min(lo + batch, i.numel()))
        d   = X[j[sl]] - X[i[sl]]
        p1, p2 = phi[g1[sl]], phi[g2[sl]]
        deltas.append((val(p1 + d) + val(p2 - d)) - (val(p1) + val(p2)))
    delta = torch.cat(deltas)

    pos = torch.nonzero(delta > 0, as_tuple=False).flatten()
    if pos.numel() == 0:
        return lab, 0.0, 0
    # at most num_group/2 swaps can be accepted (each consumes two untouched groups), so there is
    # nothing to gain from scanning far beyond that many candidates
    pos = pos[torch.argsort(delta[pos], descending=True)][:4 * num_group]

    lab     = lab.clone()
    touched = torch.zeros(num_group, dtype=torch.bool)
    total, n_acc = 0.0, 0
    for k in pos.tolist():
        a, b = int(g1[k]), int(g2[k])
        if touched[a] or touched[b]:
            continue
        touched[a] = touched[b] = True
        lab[int(i[k])], lab[int(j[k])] = b, a
        total += float(delta[k])
        n_acc += 1
    return lab, total, n_acc


# ----------------------------------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def search_permutation(gain, block_m: int, block_k: int, groupsize: int = 16,
                       rule: str = "argmin", margin: float = 0.0,
                       rounds: int = 12, swap_samples: int = 40000, lloyd_iters: int = 3,
                       seed: int = 0, axes: str = "both", verbose: bool = False):
    """
        Find a row permutation and a 16-column-chunk permutation of the tag grid `gain` (M x N)
        that maximize the realized E0M3 gain for a `block_m` x `block_k` type block under the
        election rule `rule`.

        `axes` restricts which permutations are searched -- "both", "rows" or "cols". The
        restriction is not cosmetic: §7 of results/reorder/ALGORITHM.md shows the column axis is the
        freely permutable one for `down_proj` and `o_proj` while the row axis is not, so "cols" is
        the deployable search for those and "both" is an upper bound.

        Returns a dict with `row_perm` (length M), `chunk_perm` (length N), the achieved score, the
        identity-order score, and the 1x16 ceiling.
    """
    assert axes in ("both", "rows", "cols"), f'axes must be both/rows/cols, got "{axes}".'
    assert block_k % groupsize == 0
    chunks = block_k // groupsize
    gain   = gain.to(torch.float64).cpu()
    M, N   = gain.shape

    pad_m = (-M) % block_m
    pad_n = (-N) % chunks
    if pad_m or pad_n:
        gain = torch.nn.functional.pad(gain, (0, pad_n, 0, pad_m))     # zero cells: neutral
    Mp, Np = gain.shape

    feat = gain_features(gain)
    n_rg, n_cg = Mp // block_m, Np // chunks
    cpt  = block_m * chunks

    gen = torch.Generator().manual_seed(seed)

    ceiling  = float(gain.clamp(min=0).sum())
    ident_r  = _labels_from_order(torch.arange(Mp), block_m, Mp)
    ident_c  = _labels_from_order(torch.arange(Np), chunks, Np)
    baseline = objective(feat, ident_r, ident_c, n_rg, n_cg, rule, margin, cpt)

    scored = _initial_partitions(gain, feat, block_m, chunks, rule, margin, cpt, generator=gen)
    if axes == "rows":
        scored = {k: v for k, v in scored.items() if k in ("identity", "marginal", "spectral1_rows")}
        scored = {k: (v[0], v[1], ident_c) for k, v in scored.items()}
        scored = {k: (objective(feat, v[1], v[2], n_rg, n_cg, rule, margin, cpt), v[1], v[2])
                  for k, v in scored.items()}
    elif axes == "cols":
        scored = {k: (v[0], ident_r, v[2]) for k, v in scored.items()}
        scored = {k: (objective(feat, v[1], v[2], n_rg, n_cg, rule, margin, cpt), v[1], v[2])
                  for k, v in scored.items()}
    init_name, (best, rlab, clab) = max(scored.items(), key=lambda kv: kv[1][0])
    if verbose:
        print("      init: " + ", ".join(f"{k}={v[0] / max(ceiling, 1e-30):.3f}"
                                         for k, v in scored.items()))

    for it in range(rounds):
        before = best

        # ---- Lloyd: rows, then columns ----
        if axes != "cols":
            for _ in range(lloyd_iters):
                A    = _colgroup_sums(feat, clab, n_cg)                 # (M, n_cg, F)
                cand = _lloyd_step(A, rlab, n_rg, block_m, rule, margin, cpt)
                sc   = objective(feat, cand, clab, n_rg, n_cg, rule, margin, cpt)
                if sc > best:
                    best, rlab = sc, cand
                else:
                    break

        if axes != "rows":
            for _ in range(lloyd_iters):
                B    = _group_sums(feat, rlab, n_rg).transpose(0, 1).contiguous()  # (N, n_rg, F)
                cand = _lloyd_step(B, clab, n_cg, chunks, rule, margin, cpt)
                sc   = objective(feat, rlab, cand, n_rg, n_cg, rule, margin, cpt)
                if sc > best:
                    best, clab = sc, cand
                else:
                    break

        # ---- Kernighan-Lin swaps on the exact objective ----
        if axes != "cols":
            A = _colgroup_sums(feat, clab, n_cg)
            for _ in range(3):
                rlab, gained, n_acc = _swap_round(A, rlab, n_rg, rule, margin, cpt,
                                                  swap_samples, gen)
                if n_acc == 0:
                    break

        if axes != "rows":
            B = _group_sums(feat, rlab, n_rg).transpose(0, 1).contiguous()
            for _ in range(3):
                clab, gained, n_acc = _swap_round(B, clab, n_cg, rule, margin, cpt,
                                                  swap_samples, gen)
                if n_acc == 0:
                    break
                B = _group_sums(feat, rlab, n_rg).transpose(0, 1).contiguous()

        best = objective(feat, rlab, clab, n_rg, n_cg, rule, margin, cpt)   # resync exactly
        if verbose:
            print(f"      round {it}: {best / max(ceiling, 1e-30):.4f}")
        if best <= before * (1 + 1e-9):
            break

    row_perm   = torch.argsort(rlab, stable=True)
    chunk_perm = torch.argsort(clab, stable=True)
    row_perm   = row_perm[row_perm < M]
    chunk_perm = chunk_perm[chunk_perm < N]

    return {
        "row_perm":   row_perm,
        "chunk_perm": chunk_perm,
        "score":      best,
        "baseline":   baseline,
        "ceiling":    ceiling,
        "init":       init_name,
        "recovered":  best / ceiling if ceiling > 0 else float("nan"),
        "baseline_recovered": baseline / ceiling if ceiling > 0 else float("nan"),
    }


@torch.no_grad()
def expand_chunk_perm(chunk_perm, groupsize: int = 16):
    """ A permutation of 16-column chunks, expanded to a permutation of columns. """
    base = chunk_perm.to(torch.long).reshape(-1, 1) * groupsize
    return (base + torch.arange(groupsize).reshape(1, -1)).reshape(-1)


# ----------------------------------------------------------------------------------------------
# The noise floor -- the control that decides whether a measured gain is structure
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def shuffle_control(gain, generator=None):
    """
        Randomly permute ALL cells of the tag grid.

        This destroys every row/column structure while preserving the exact multiset of gains, so
        the ceiling is unchanged and the search faces a statistically identical but structureless
        problem. Running the same search on it gives the OVERFITTING FLOOR: a balanced co-clustering
        search with tens of thousands of tiles and full permutation freedom will always concentrate
        some positive mass, even in i.i.d. noise (measured at 0.38 of the ceiling on 256x64 standard
        normal cells, against 0.18 for the identity order).

        Only the EXCESS of the real score over this floor is evidence that rows and columns actually
        share a preference. Report both.
    """
    flat = gain.reshape(-1)
    perm = torch.randperm(flat.numel(), generator=generator)
    return flat[perm].reshape(gain.shape)


@torch.no_grad()
def additive_shares(gain):
    """
        Two-way additive decomposition of the tag grid,

            G[i, j] = mu + a_i + b_j + e_ij          (a, b, e mutually orthogonal)

        returning the share of the total variance carried by the row effects `a`, the column effects
        `b`, and the residual `e`.

        This is the sharpest statement of what a reordering CAN do. A tile's statistic is the mean
        over its cells,

            mean_tile = mu + mean_{i in b} a_i + mean_{j in c} b_j + mean_tile e

        and a permutation moves only which rows and which columns share a tile. The `a` and `b`
        terms it can concentrate arbitrarily well -- just sort them. The residual `e` it cannot
        touch in any systematic way: if `e` is exchangeable, every arrangement is equally likely and
        rearranging it is fitting noise, which is exactly what `shuffle_control` measures.

        So `row_share + col_share` is the fraction of the signal any row/column reordering scheme --
        this one, seriation, or an exact solver -- has to work with. If it is a few percent, the
        idea is dead regardless of how good the search is, and the measured "gain" is the noise
        floor.
    """
    g   = gain.to(torch.float64)
    M, N = g.shape
    mu  = g.mean()
    a   = g.mean(dim=1, keepdim=True) - mu
    b   = g.mean(dim=0, keepdim=True) - mu
    tot = (g - mu).pow(2).sum().clamp(min=1e-300)
    return (float(a.pow(2).sum() * N / tot),
            float(b.pow(2).sum() * M / tot),
            float((g - mu - a - b).pow(2).sum() / tot))


@torch.no_grad()
def interaction_structure(gain, num_sample: int = 2048, rank: int = 16, generator=None):
    """
        Probe for CO-CLUSTER structure -- the kind `additive_shares` is blind to.

        `additive_shares` fits G = mu + a_i + b_j + e, so it only sees whether a row is uniformly
        more E0M3-preferring than another ACROSS THE WHOLE reduction dimension. That is not what a
        tile needs. A grid whose rows fall into k clusters with k different PROFILES is perfectly
        co-clusterable while having a_i ~ 0 and b_j ~ 0, because the profiles cancel in the row
        means. All of that structure is booked as "residual" by the additive decomposition.

        What the tiling actually requires is profile similarity, and it requires it GLOBALLY: a row
        group is formed once and then meets every column group, so its 8 rows must resemble each
        other over the entire reduction dimension, not merely on the 4 chunks of one tile. Two
        statistics capture that without assuming additivity:

          * the distribution of pairwise correlations between mean-centred row profiles -- if rows
            cluster at all, some pairs must be far more similar than chance;
          * the singular spectrum of the centred grid -- k co-clusters produce k large singular
            values, so this is the rank-k generalization of the rank-1 sign test.

        Both are meaningless alone and only interpretable against `shuffle_control`, which is why
        the caller should probe the real grid and a shuffled copy and compare.
    """
    g = gain.to(torch.float32)
    g = g - g.mean()

    def _corr(x):
        n   = min(num_sample, x.shape[0])
        idx = torch.randperm(x.shape[0], generator=generator)[:n]
        v   = x[idx]
        v   = v - v.mean(dim=1, keepdim=True)
        v   = v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        c   = v @ v.transpose(0, 1)
        off = c[~torch.eye(n, dtype=torch.bool)]
        return float(off.abs().mean()), float(torch.quantile(off, 0.999))

    row_mean, row_p999 = _corr(g)
    col_mean, col_p999 = _corr(g.transpose(0, 1).contiguous())

    q = max(min(rank, min(g.shape) - 1), 2)
    U, S, V = torch.svd_lowrank(g, q=q, niter=4)
    total   = float(g.pow(2).sum())
    spec    = (S.pow(2) / max(total, 1e-30)).tolist()

    return dict(
        row_corr_abs=row_mean, row_corr_p999=row_p999,
        col_corr_abs=col_mean, col_corr_p999=col_p999,
        sv1=spec[0], sv_top4=sum(spec[:4]), sv_top16=sum(spec[:16]),
    )


@torch.no_grad()
def election_stats(gain, rlab, clab, num_rowgroup: int, num_colgroup: int,
                   rule: str, margin: float, cells_per_tile: int, e2m1_total: float = 0.0):
    """
        What the election did to the INDIVIDUAL scale blocks, not just in aggregate.

        A tile that elects E0M3 forces that grid on every scale block it contains, including the
        ones that preferred E2M1. Those blocks are HARMED -- they end up worse than they would have
        been under plain NVFP4. The aggregate gain can rise while the harm rises with it, and
        CLAUDE.md measures repeatedly that this is the regime where a weight-MSE win fails to reach
        perplexity ("a rule that fires on small gains is fitting noise").

        This distinguishes the two ways of banking the same aggregate MSE:

          * from STRUCTURE -- elected tiles are homogeneous, few blocks are overruled;
          * from ARRANGEMENT -- the search packs negative blocks in with positive ones so the sum
            clears the bar, which raises the harmed share by construction.

        Returns the realized gain, the share of scale blocks harmed, the harm mass as a fraction of
        the all-E2M1 loss, and the share of tiles that elected E0M3.
    """
    feat  = gain_features(gain)
    A     = _colgroup_sums(feat, clab, num_colgroup)
    phi   = _group_sums(A, rlab, num_rowgroup)
    elect = elect_mask(phi, rule, margin, cells_per_tile)          # (n_rg, n_cg)

    # broadcast the per-tile decision back down to every cell
    cell_elect = elect[rlab][:, clab]                              # (M, N)
    harmed     = cell_elect & (gain < 0)
    harm_mass  = float((-gain)[harmed].to(torch.float64).sum())

    return dict(
        realized=float(tile_value(phi, rule, margin, cells_per_tile).sum()),
        harmed_share=float(harmed.to(torch.float32).mean()),
        harm_pct_of_mse=(100.0 * harm_mass / e2m1_total) if e2m1_total else float("nan"),
        elected_tile_share=float(elect.to(torch.float32).mean()),
    )
