# Reordering rows and columns does not help MixFP4 — and here is the proof it cannot

Algorithm and rationale: `ALGORITHM.md`. Implementation: `quantize/reorder.py`.

**Result: at the deployable 8x64 weight type block on Llama-3.1-8B, the co-clustering reorder
recovers +0.135 of the 1×16 ceiling over the current order — and +0.002 over a cell-shuffled
control. The entire gain is the partition search overfitting noise. A two-way variance decomposition of the tag grid over all 224 layers explains
why: row and column effects carry only 2.5% of the variance in the E0M3 preference, while 97.5% is
residual — idiosyncratic to the individual 16-element scale block and invariant to every row and
column permutation.**

This is a bound on the whole idea, not on this particular solver.

---

## 1. What was measured

For every weight matrix, the tag grid

```
G[i,j] = loss_E2M1(scale block i,j) − loss_E0M3(scale block i,j)
```

and then, for a given type block and election rule, the realized E0M3 gain as a fraction of the
1×16 ceiling `Σ relu(G)`:

| column | meaning |
|---|---|
| `identity` | the current, unpermuted order — what `mix_4_6` gets today |
| `search` | after the balanced co-clustering reorder |
| `control` | **the same search on a cell-shuffled copy of `G`** — same multiset of gains, same ceiling, zero structure |

`search − identity` is what a perplexity run would see. `search − control` is the part attributable
to rows and columns genuinely sharing a preference.

The control is not a formality. On an i.i.d. Gaussian matrix the search reports **+0.203 over the
identity order and +0.001 over the control** — a balanced partition search with tens of thousands of
tiles and full permutation freedom concentrates positive mass in pure noise. Any number quoted
without this control is unmeasured.

Validation that the search itself works: on a planted rank-1 checkerboard it goes 0.171 → **0.998**
of the ceiling, and on a column-outlier tensor whose structure is invisible to the identity order
(0.000) it reaches 0.399 against a control of 0.190. The solver is not the limitation.

---

## 2. Llama-3.1-8B weights at 8x64

`8x64` is the smallest hardware-realizable type block for the weight operand (`n8 x k64`, one MMA
B-tile), so it is the only shape that matters for deployment. Every 4th layer,
`q_proj`/`v_proj`/`o_proj`/`up_proj`/`down_proj`, clip preset `heade0`, mean over the 40 tensors
(`llama-3.1-8b.csv`):

| rule | identity | search | control | lift vs identity | **lift vs control** |
|---|---|---|---|---|---|
| `argmin` | 0.232 | 0.345 | 0.344 | +0.114 | **+0.001** |
| `h1.5`   | 0.179 | 0.314 | 0.312 | +0.135 | **+0.002** |

A large, entirely spurious lift over the identity order; nothing over the control.

The `h1.5` row is the one that counts: that is the election rule CLAUDE.md establishes as the one
worth deploying, and the search optimizes it directly rather than a proxy for it. Reordering takes
it from 0.179 to 0.314 of the ceiling — and a grid with its cells shuffled reaches 0.312.

Per-tensor it never separates either. `down_proj` is the best case at +0.013 over control — and
`down_proj` is precisely the matrix whose reduction axis is freely permutable per layer (§7 of
`ALGORITHM.md`), so if anything were there, that is where it would show. `q_proj` and `v_proj` come
out **negative** against their controls (−0.021, −0.009), i.e. within search-to-search noise of zero.

---

## 3. Why — the two-way variance decomposition

Decompose the tag grid into row effects, column effects and residual:

```
G[i,j] = mu + a_i + b_j + e_ij            (orthogonal; `reorder.additive_shares`)
```

A tile's statistic is the mean over its cells, `mu + mean(a over its rows) + mean(b over its
columns) + mean(e over the tile)`. A permutation can concentrate `a` and `b` arbitrarily well — just
sort them. It cannot touch `e` in any systematic way: if the residual is exchangeable, every
arrangement is equally likely, and rearranging it is exactly the overfitting the control measures.

So **`row_share + col_share` is the fraction of the signal that any row/column reordering scheme has
to work with** — this solver, seriation, spectral co-clustering, or an exact NP-hard solver.

The decomposition assigns a share to rows and columns even when there is no structure at all: for
an `M x N` grid of i.i.d. cells, `E[row_share] = (M-1)/(MN-1)` and `E[col_share] = (N-1)/(MN-1)`.
So the measured share only means something read against that baseline (`analyze_reorder.py`).

Llama-3.1-8B, **all 224 linear layers**, clip preset `heade0`
(`diag_llama-3.1-8b_heade0.csv`, `x` = multiple of the i.i.d. expectation):

| projection | row | noise | x | col | noise | x | **row+col** | resid | rank1 |
|---|---|---|---|---|---|---|---|---|---|
| `q_proj`    | 0.0234 | 0.0039 | 6.0 | 0.0048 | 0.0002 | 19.6 | 0.0282 | 0.9718 | 0.568 |
| `k_proj`    | 0.0333 | 0.0039 | 8.5 | 0.0121 | 0.0010 | 12.5 | 0.0455 | 0.9546 | 0.596 |
| `v_proj`    | 0.0116 | 0.0039 | 3.0 | 0.0050 | 0.0010 |  5.1 | 0.0166 | 0.9834 | 0.566 |
| `o_proj`    | 0.0111 | 0.0039 | 2.8 | 0.0010 | 0.0002 |  4.3 | 0.0121 | 0.9878 | 0.558 |
| `gate_proj` | 0.0320 | 0.0039 | 8.2 | 0.0008 | 0.0001 | 11.1 | 0.0328 | 0.9672 | 0.562 |
| `up_proj`   | 0.0131 | 0.0039 | 3.4 | 0.0009 | 0.0001 | 13.3 | 0.0140 | 0.9859 | 0.554 |
| `down_proj` | 0.0242 | 0.0011 | 21.7 | 0.0011 | 0.0002 |  4.4 | 0.0253 | 0.9747 | 0.551 |
| **ALL** | **0.0213** | 0.0035 | **6.1** | **0.0037** | 0.0004 | **9.1** | **0.0249** | **0.9751** | **0.565** |

Read this carefully, because the honest version is not "it is all noise". The row and column effects
are **statistically real** — 3–22× the chance level, far outside sampling error at these grid sizes.
They are simply **negligible in magnitude**: together they carry **2.5% of the variance**, and the
remaining **97.5% is residual**, idiosyncratic to the individual 16-element scale block and
invariant to every row and column permutation.

That is the whole result. A reordering scheme is competing for 2.5% of the signal, and it has to
express that 2.5% through tiles that average 32 cells — which is why the measured lift over the
control is +0.002 rather than something merely small.

The corroborating statistic points the same way: the best rank-1 sign model of `G` explains
**0.565** of the `|G|` mass against 0.5 for a coin flip. The sign pattern of the tag grid is very
nearly not a product of a row property and a column property — and a product of a row property and
a column property is exactly, and only, what a row-partition × column-partition tiling can represent.

Note `down_proj`: the largest row effect relative to noise (21.7×) and the matrix whose reduction
axis is freely permutable per layer. It is also the one tensor where the search beat its control at
all (+0.013). Both facts agree, and both are too small to matter.

---

## 4. What this settles

- **The E0M3-vs-E2M1 preference is a property of the individual 16-element scale block**, not of its
  output channel and not of its input channel. Two scale blocks in the same row and two in the same
  column disagree about as often as two picked at random.
- **No reordering scheme in this family can work**, however good its solver. That includes the
  existing `_perm` row-sort (which CLAUDE.md already measured as a loss), seriation, the Bond Energy
  Algorithm, spectral co-clustering, and an exact solve of the NP-hard problem. The earlier `_perm`
  failure was not "the wrong axis" — CLAUDE.md's own diagnosis, that "the axis that matters is K" —
  it is that **neither axis carries the signal**. Permuting K freely, which this study does and
  which is genuinely available per-layer for `down_proj`, changes nothing.
- **The gap between `1x16` and a realizable tile is irreducible by permutation.** It is the price of
  coarse granularity itself, and the only ways left to attack it are a finer type block, a better
  election rule (where `h<λ>` already lives), or a different format.

This also closes a specific trap: the search "works", in the sense that it reliably lifts the
measured objective by +0.115 of the ceiling. Calibrating against CLAUDE.md's perplexities that would
have read as roughly −0.005 wikitext, comparable to the best knobs in the study. It would have been
entirely false.

---

## 5. What was built anyway, and when it would be worth reusing

`quantize/reorder.py` is general balanced co-clustering over any additive tile objective, with
`quant_mix_4_6` hooks (`_cocl`, `_coclcol`, `_coclrow`) so a perplexity run needs no plumbing. It is
worth pointing at a different problem if one turns up where `additive_shares` says the row and
column effects are large — the diagnostic costs one pass over the weights and answers the question
before any search runs. On the evidence here, run `--diagnostics_only` first, always.

## 6. Reproducing

Everything goes through Slurm, including the CPU-only test suites — see the cluster rules in
`/home/u4320956/CLAUDE.md`. The GPU each job requests is left idle; it is there only because cores
are rationed at 12 per GPU and there is no CPU-only partition.

```bash
sbatch slurm/reorder_tests.sbatch                # test_reorder.py + test_mixfp4.py + analyze_reorder.py
sbatch slurm/reorder_diag.sbatch llama-3.1-8b    # the decisive diagnostic, ~10 min for all 224 layers
sbatch slurm/reorder_sim.sbatch  llama-3.1-8b 8  # the full search sweep
```

Note both sbatch scripts read the model from the existing `/work/$USER/hf` cache with
`HF_HUB_OFFLINE=1` — this account has no HuggingFace token, so gated repos cannot be downloaded and
only `meta-llama/Llama-3.1-8B` and the Qwen3 models are available locally. Llama-2-7B could not be
measured for that reason.
