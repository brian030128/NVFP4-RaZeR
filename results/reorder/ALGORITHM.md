# Reordering rows and columns to group E0M3-preferring blocks

How to permute a weight matrix so that coarse MixFP4 type blocks come out **unbalanced** — mostly
E0M3-preferring or mostly E2M1-preferring — instead of mixed.

Implementation: `quantize/reorder.py`. Tests: `tests/test_reorder.py`. Measurement driver:
`run_reorder_sim.py`.

---

## 1. The problem, stated exactly

Fix a weight matrix `W` of shape `(M, K)`. MixFP4 cuts it two ways:

- a **scale block** is 16 contiguous columns and owns one ue4m3 scale — always 16, not configurable;
- a **type block** is `BM x BK` and owns one element data type, E2M1 or E0M3, shared by every scale
  block inside it.

So the natural grid is not the element grid, it is the **scale-block grid**

```
N        = K / 16
G[i, j]  = loss_E2M1(scale block i,j) − loss_E0M3(scale block i,j)        G ∈ R^(M × N)
```

`G[i,j] > 0` means "this 1×16 block prefers E0M3". This is exactly the quantity `quant_mix_4_6`
already computes and sums per tile — it is `row_preference` with the row-sum removed
(`reorder.scale_block_gain`, asserted equal in `test_gain_grid_matches_row_preference`).

Two things about the tags that shape everything downstream:

- **They are signed magnitudes, not binary.** The election sums them, so a tile with three
  weakly-E0M3 cells and one strongly-E2M1 cell is an E2M1 tile. Counting tags would get this wrong.
- **The grid is `M × K/16`, not `M × K`.** A type block of `8x64` is a `8 × 4` tile of this grid —
  32 cells. This is much smaller than it looks, which matters for the noise floor in §6.

A type block is a tile of `BM` rows × `c = BK/16` columns of `G`. Under the plain `argmin` election
the loss a tile saves relative to all-E2M1 is `max(0, Σ_tile G)`. Reordering is choosing a partition
`R` of the rows into groups of exactly `BM` and a partition `C` of the columns into groups of exactly
`c`, and the objective is

```
maximize   F(R, C) = Σ_{b∈R} Σ_{c∈C}  v( Σ_{i∈b, j∈c} G[i,j] )
```

Two facts make this the right formulation:

- **Only the partition matters, not the order.** Any equal-size partition is realizable by some
  permutation, and permutations that induce the same partition are equivalent. So this is a set
  partitioning problem, not an ordering problem — which is why pure seriation approaches (sort by a
  key) are strictly weaker than what follows.
- **There is an exact, interpretable ceiling.** `F ≤ Σ_ij max(0, G[i,j])`, achieved by `1x16` where
  every scale block chooses for itself. Define

  ```
  recovered = F(R, C) / Σ_ij relu(G[i,j])
  ```

  `mix_4_6` today is this quantity at `R = C = identity`. Everything below is reported in these
  units, so "0.28 → 0.44" means the reordering recovered an extra 16% of the gap between a coarse
  tile and per-block choice.

---

## 2. The decision that makes it well-posed: columns move in chunks of 16

**Permute whole 16-column chunks, never individual columns.**

Permuting individual columns changes which elements share a scale block, hence the block maximum,
hence the block scale, hence `G` itself. The objective would then depend on the variable being
optimized — a fixed-point problem with no guarantee of convergence and no exact evaluation.

Moving whole 16-column chunks makes `G` exactly permutation-covariant: permuting chunks permutes
`G`'s columns and changes nothing else. This is verified end-to-end on real quantizer output in
`test_permuted_weights_reproduce_gain` — permute the weights, recompute the tag grid from scratch,
and it equals the permuted tag grid to 1e-9.

Finer permutation is not lost, it is **deferred** — see §7, where it returns as a strictly safe
within-group refinement that cannot disturb the partition.

---

## 3. Yes, it is NP-hard — and it has a name

This is **balanced co-clustering** (equivalently biclustering with fixed cluster cardinalities, or
"checkerboard" co-clustering). It contains balanced k-way graph partitioning, so it is NP-hard, and
even the row-only subproblem with several column groups is a vector-partitioning problem that is
NP-hard.

The literature to borrow from, and what each contributes:

| Prior problem | What it gives here |
|---|---|
| **Bond Energy Algorithm** (McCormick 1972) | The original "permute rows and columns to make dense blocks" heuristic. Pure seriation — the weak baseline. |
| **Consecutive Block Minimization / Consecutive Ones Property** (Kou 1977, NP-hard) | Establishes hardness for exactly this shape of problem. |
| **Spectral co-clustering** (Dhillon 2001) | The initializer in §5, stage 1. |
| **Information-theoretic co-clustering** (Dhillon, Mallela, Modha 2003) | The alternating-optimization skeleton: fix columns, reassign rows, repeat. |
| **Balanced k-means / capacitated clustering** | The assignment step is a transportation problem, not a free argmin — capacity is what makes it hard. |
| **Kernighan–Lin / Fiduccia–Mattheyses, METIS, KaHyPar** | The swap-based refinement that closes the last few percent, and the standard way to hold a hard balance constraint. |

The solver below is the standard recipe from that literature — spectral init → capacitated Lloyd →
KL refinement — specialized to this objective.

---

## 4. The structural fact that makes it cheap

**Every election rule in `quantizer._elect_e0m3` is a function of a handful of additive tile
statistics.** Map each cell to a 5-vector

```
φ(g) = [ g,  relu(g),  relu(−g),  1{g>0},  g² ]     →   (s, p, n, cnt, sq) summed over the tile
```

then

| rule | elect iff |
|---|---|
| `argmin` | `s > 0` |
| `harm(λ)` — the `h<λ>` rule | `p > λ·n` |
| `dominance` | `n = 0` and `s > 0` |
| `vote(t)` | `cnt > t·nb` and `s > 0` |
| `margin(z)` | `s/nb > z·√(sq/nb − (s/nb)²)/√nb` |

and the realized gain is `elect · s`. This is asserted rule-for-rule against the real
`_elect_e0m3` in `test_elect_matches_quantizer`.

Three consequences, and they are the whole reason this is tractable:

1. **The objective is a sum over tiles of a cheap function of a 5-vector.** Evaluating a full
   `11008 × 688` grid partition is two `index_add_`s.
2. **Every move has a closed-form incremental delta.** Swapping two rows changes only two rows of
   tiles, and additivity gives the delta without recomputing anything.
3. **The search optimizes the rule you will actually deploy.** CLAUDE.md's central finding is that
   `argmin` loses and `h1.5` wins — a "when it decisively helps" rule. Optimizing the permutation
   under `argmin` and then deploying `h1.5` would be optimizing the wrong thing. Here `rule` is a
   parameter of the search.

---

## 5. The algorithm

### Stage 0 — build the tag grid
`scale_block_gain(W / global_scale, 16, metric, clip)`. Weights only, no calibration data. Use the
`clip` preset you will deploy (`heade0` if the E0M3 branch is in play at all).

### Stage 1 — spectral initialization, and a go/no-go test
Take the top singular triplet, `G ≈ σ·u·vᵀ`. Then `sign(G[i,j]) ≈ sign(u_i)·sign(v_j)`: a
**checkerboard**, which is precisely the structure a row-partition × column-partition product can
represent. Sorting rows by `u` and columns by `v` puts agreeing cells in the same tiles.

This is also the cheap **go/no-go test for the whole idea**. If `sign(G)` is far from rank-1, no
product partition can be good, whatever the search does. `run_reorder_sim.sign_structure` reports
the fraction of `|G|` mass the best rank-1 sign model explains. The SVD's sign ambiguity is
irrelevant: flipping `(u,v) → (−u,−v)` reverses both orders, giving the same partition relabelled.

Several initializers are scored and the best kept: identity, marginal sort (the two-axis
generalization of the existing `permute="rows"`), rank-1 spectral, rank-2 angular spectral,
rows-only, columns-only, and random restarts. Which one wins is itself diagnostic — `spectral1_cols`
winning says the structure lives on K, `marginal` winning says a simple sort suffices.

### Stage 2 — capacitated Lloyd (alternating balanced assignment)
Fix the column partition. Precompute `A[i,c] = Σ_{j∈c} φ(G[i,j])`, the feature mass row `i`
contributes to each tile column. Now **linearize around the current elections**: with the elect bits
`s[b,c] ∈ {0,1}` held fixed, the objective is

```
Σ_i Σ_c  s[b(i), c] · A[i, c, 0]
```

which is **linear in the assignment**, so the optimal reassignment of rows to row-groups is a
balanced transportation problem. Solve it (regret-ordered greedy from balanced k-means), re-elect,
repeat. Then do the same for columns. This is Lloyd's algorithm; the linearization is exact at
fixed elections and the outer loop keeps only improving moves, so it is monotone.

### Stage 3 — Kernighan–Lin swap refinement, on the exact objective
Sample candidate swaps between different groups, evaluate every delta exactly and in one batch via
the additive statistics, and greedily accept the positive ones that touch **disjoint groups** — so
the deltas stay valid without recomputation. Alternate rows and columns. This is where the last few
points come from, and unlike stage 2 it uses the exact `v`, including the non-linear `h(λ)` gate.

Swaps are capped at `4·num_group` candidates per round (at most `num_group/2` can be accepted) and
batched so the delta tensors stay small — a `down_proj` column pass would otherwise allocate
gigabytes.

### Cost
One `index_add_` per objective evaluation, one thin randomized SVD per init, and batched delta
evaluation. Seconds per matrix. It runs on CPU, offline, once per model.

---

## 6. The control that decides whether any of this is real

**A balanced partition search with tens of thousands of tiles and full permutation freedom will
concentrate positive mass even in i.i.d. noise.** This is not a bug — it is the combinatorial
analogue of overfitting, and it is large.

Measured on synthetic tensors (`8x64`, `argmin`, fraction of the 1×16 ceiling):

| tensor | rank-1 sign fit | identity | search | **cell-shuffle control** | real lift |
|---|---|---|---|---|---|
| i.i.d. Gaussian `[1024,1024]` | 0.582 | 0.290 | 0.492 | **0.491** | **+0.001** |
| planted heavy-tail | 0.667 | 0.167 | 0.432 | 0.342 | +0.090 |
| column outliers | 0.972 | 0.000 | 0.399 | 0.190 | +0.209 |

On a Gaussian matrix the search reports **+0.203 over the identity order and +0.001 over the
control**. Reporting "reordering recovers 20% more of the 1×16 gain" from that number would be
entirely an artifact.

So `shuffle_control` permutes **all cells** of `G`, destroying every row/column structure while
preserving the exact multiset of gains — same ceiling, statistically identical, structureless. The
same search is run on it, and the quantity to believe is

```
lift_vs_control = search − control
```

Every row of `run_reorder_sim.py` output carries it. `search − identity` is what a perplexity run
would see; `search − control` is what is attributable to rows and columns genuinely sharing a
preference. Note the column-outlier row: rank-1 sign fit 0.972 and identity 0.000 — a case where the
structure is real, strong, and completely invisible to the unpermuted order.

---

## 7. Which permutations are actually free (this constrains the answer)

For `Y = X Wᵀ`, permuting `W`'s **rows** permutes the layer's output channels and must be undone
downstream; permuting `W`'s **columns** requires `X`'s channels permuted identically, upstream.
Confirmed against `models/qmodule_llama.py`:

| matrix | reduction axis (K) — the one that matters | output axis (M) |
|---|---|---|
| **`down_proj`** | **free, per layer** (`π_ff`) | global `π_hidden` only |
| **`o_proj`** | **free within a head**; cross-head needs a runtime gather | global `π_hidden` only |
| `gate_proj`, `up_proj` | global `π_hidden` only | free per layer, but **tied to each other and to `down_proj`'s columns** (`π_ff`) |
| `v_proj` | global `π_hidden` only | free within a head, tied to `o_proj`'s columns |
| `q_proj`, `k_proj` | global `π_hidden` only | RoPE pairs dim `d` with `d+D/2`, and q/k are tied — heavily constrained |

Three consequences:

- **`down_proj` is the target.** Its K axis — the axis CLAUDE.md identifies as the one that matters
  ("the disagreement is mostly *within* a row across its k-blocks") — is freely permutable per
  layer at zero runtime cost, because `down_proj(act(gate(x)) * up(x))` is elementwise in the
  intermediate axis. `down_proj` is also the single largest matrix in the block.
- **The `π_ff` tie is free, not a conflict.** `π_ff` moves 16-channel atoms. Since 16 is a multiple
  of `BM = 8`, permuting whole atoms leaves `gate/up_proj`'s row grouping **completely unchanged** —
  each atom is exactly two intact row groups. So optimizing `down_proj`'s columns costs the MLP
  nothing elsewhere.
- **The hidden axis is one global permutation.** Columns of q/k/v/gate/up and rows of o/down all
  share it, along with the RMSNorm gains, the embedding and the lm_head. It is free but must be
  chosen once for the whole model — a joint objective summed over every layer, not a per-layer one.

This is also the reason the existing `permute="rows"` experiment (CLAUDE.md: "every variant is worse")
was not a fair test of the idea. It permuted the axis that is *least* free and *least* structured.

### The two-level scheme this suggests

- **Level 1, between atoms** — permute 16-channel atoms. Sets `down_proj`'s column partition. `G` is
  invariant, the objective is exact, this is what `search_permutation` does.
- **Level 2, within one atom** — permute the 16 channels inside an atom. This changes `down_proj`'s
  scale-block *composition* for that one column chunk (so its `G` cell moves) but **never the
  partition**, since the chunk stays in its group. And it fully regroups those 16 rows of
  `gate/up_proj` into their two groups of 8. Strictly safe, and the only way to reach `gate/up`'s
  row axis at all.

Level 2 is not implemented yet; level 1 is.

---

## 8. How to read the output, and what would count as success

`run_reorder_sim.py` prints, per matrix and per (type block, rule): `identity`, `search`, `control`,
and the two lifts, all as fractions of the 1×16 ceiling. Plus `pos_share` (how mixed the tensor is
at all) and `rank1_sign_fit` (whether a product partition can even express its structure).

Calibrating against CLAUDE.md's measured perplexities on Llama-3.1-8B W4A16 at 8x64: `mix_4_6_h1.5`
is −0.0117 wikitext and `nvif4` (per-scale-block choice, the ceiling) is −0.0387. So the realizable
rule keeps about a quarter of the ceiling, and **each extra 0.1 of `recovered` is worth roughly
0.004 wikitext** if the relationship is linear. A reordering that moves `recovered` by +0.15 with a
matching `lift_vs_control` would be worth about −0.006 — real, and comparable to the best knobs in
the study, but not transformative. If `lift_vs_control` is near zero on real weights, the honest
conclusion is that the E0M3 preference of a scale block is **not** a product of a row property and a
column property, and no permutation scheme of this family will help.

That last outcome is a live possibility and the reason the control is built in rather than bolted
on. The i.i.d. Gaussian row of §6 is what the failure mode looks like.

---

## 9. Two known caveats

- **MSE-optimal ≠ perplexity-optimal.** CLAUDE.md documents this repeatedly. `recovered` is a
  weight-space quantity; the search maximizes it under the deployed election rule, which is the
  best available proxy, but it is still a proxy. Confirm with `run_ppl_sweep.py`.
- **The search overfits, so always report against the control.** See §6. Any single number quoted
  without its shuffle control should be treated as unmeasured.
