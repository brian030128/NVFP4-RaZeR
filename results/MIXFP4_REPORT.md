# MixFP4: what helps, what doesn't, and what it costs

Perplexity at seq 2048, type block **8x64** for weights (the smallest hardware-realizable weight
tile, one `n8 x k64` MMA B-operand). Five models, wikitext and c4.

**Sections 1-11 are W4A16 (weight-only).** Section 1a gives the W4A4 results, which are the
deployment-relevant ones and where several conclusions differ. W4A4 activations are pinned at
`nvfp4_4over6` throughout so that only the weight side varies.

---

## 0. Read this first — measurement noise

Configurations that use calibration (`hess`) are **not bit-reproducible**. The same configuration
run twice:

    mix_4_6_clipheade0_hess_coclcol_m1   6.539287   and   6.543658     (spread 0.0044)

Configurations without calibration are exactly reproducible (`nvfp4_4over6` gave
6.598703861236572 in three separate jobs). So the variance comes from `collect_importance` —
GPU reductions over the calibration batches are not deterministic, and that propagates through
the elections.

**Consequence: differences below ~0.005 between two `hess` configurations are not meaningful.**
This matters for two headline numbers below — the E0M3 type block's marginal contribution
(-0.011) and reordering's contribution (-0.007) are only 2.5x and 1.5x this noise floor. They
are reported as measured, but a single run should not be trusted to that precision.

---

## 1. Headline

| lever | worth (Llama-3.1-8B wikitext) | needs |
|---|---|---|
| **alpha chosen by importance instead of MSE** | **-0.042** | nothing — stock NVFP4 kernel |
| widening alpha from {1,1.5} to 5 candidates | -0.006 | nothing |
| E0M3 type block (election by importance) | -0.011 | E0M3 hardware path + type block |
| reordering on top | -0.007 | offline search; free only if constrained (§6) |

**The scale search dominates.** Choosing between FourOverSix's existing two candidates with an
importance-weighted criterion is worth about four times the entire E0M3 type block, and it
requires no format change at all: alpha only changes the value written into the ue4m3 scale
field that already exists.

---

## 1a. W4A4 (prefill) — activations fixed at `nvfp4_4over6`

The section that matters for deployment. Only the WEIGHT configuration varies; activations are
`nvfp4_4over6` in every row so the comparison is like for like.

Note `quant_act` never receives `importance` -- the `use_importance` field is discarded in its
dispatch -- so **calibration applies to weights only**. A `hess` qualifier on an activation dtype is
a silent no-op.

### Llama-3.1-8B

| weight config | wikitext / c4 | vs `4over6` | vs `nvfp4` |
|---|---|---|---|
| `nvfp4` (both operands plain) | 6.9454 / 9.9339 | +0.0699 / +0.1077 | — |
| `nvfp4_4over6` (both) | 6.8755 / 9.8262 | — | -0.0699 / -0.1077 |
| `base_hess_e2m1` (importance-alpha, **no type block**) | 6.8323 / 9.7791 | **-0.0432 / -0.0471** | -0.1131 / -0.1548 |
| `heade0_hess_m1` | 6.8084 / 9.7569 | **-0.0671 / -0.0693** | -0.1369 / -0.1770 |
| `heade0_hess_coclcol_m1` | 6.8133 / 9.7518 | -0.0623 / -0.0743 | -0.1321 / -0.1821 |

Two differences from the W4A16 picture:

* **The weight-side alpha gain survives activation quantization unchanged**: -0.0432 / -0.0471 in
  W4A4 against -0.0419 / -0.0391 in W4A16.
* **The type block is worth about twice as much here.** It adds -0.0239 on top of importance-alpha
  (W4A4) against -0.011 (W4A16). So §1's ranking -- "the scale search dominates, the type block is a
  small increment" -- weakens under W4A4: the type block roughly doubles in relative value.
* Reordering is mixed and inside the noise floor of §0: worse on wikitext (-0.0623 vs -0.0671),
  better on c4.

### Qwen3-4B

Same inversion as W4A16 -- plain `nvfp4` on both operands is the best configuration, and every
added mechanism costs.

| weight config | wikitext / c4 | vs `4over6` | vs `nvfp4` |
|---|---|---|---|
| `nvfp4` (both plain) | **13.9488 / 17.2751** | -0.3202 / -0.0409 | — |
| `nvfp4_4over6` (both) | 14.2691 / 17.3160 | — | +0.3202 / +0.0409 |
| `base_hess_e2m1` | 14.4448 / 17.3682 | +0.1757 / +0.0522 | +0.4960 / +0.0931 |
| `heade0_hess_m1` | 14.8122 / 17.5409 | +0.5431 / +0.2249 | +0.8634 / +0.2659 |
| `heade0_hess_coclcol_m1` | 14.8634 / 17.5491 | +0.5944 / +0.2331 | +0.9146 / +0.2740 |

`4over6` costs +0.3202 wikitext here, closely matching the +0.3823 it costs in W4A16, so the
alpha-widening defect is a property of the weights and is not changed by quantizing activations.

### Not yet measured in W4A4

* **alpha = 1 weight configs** (`clipa1_hess_h10`, `_v0.7`, `_m1`, `_h5`). These are the only
  configurations that reached parity with `nvfp4` in W4A16 on Qwen3-4B, so they are the ones that
  matter there. Queued.
* **Qwen3-8B** and the second-wave weight ladder (`h1.5` MSE, `hessa`, `hesst`) on both models.
* W4A4 runs are far slower than W4A16 because `quant_act` executes on every forward pass; one
  7-config sweep took over 3.5 hours, and a 2-hour `dev` allocation timed out.

### One W4A4 measurement on the ACTIVATION type block

From a separate sweep with weights fixed at `mix_4_6_clipheade0_h1.5@8x64`, varying only the
activation dtype at a 16x64 tile (the A-operand `m16 x k64`):

| activations | wikitext |
|---|---|
| `nvfp4_4over6` | 6.8499 |
| `mix_4_6_clipheade0_e2m1` (type block, never elects) | 6.8455 |
| `mix_4_6_clipheade0_h1.5` | **6.8358** |

The activation type block is worth **-0.0141** with no calibration, no reordering and no search --
decided per tile at runtime. A peakedness veto (`pv<tau>`) on top was worse at every threshold
tried, because the error-based election already encodes peakedness (rho = -0.59, §8).

---

## 2. Baselines

| model | `nvfp4` (alpha=1) | `nvfp4_4over6` |
|---|---|---|
| Llama-3.1-8B | 6.6236 / 9.4796 | 6.5987 / 9.4232 |
| Llama-3.1-8B-Instruct | — | 7.5313 / 10.8714 |
| Llama-3.2-1B-Instruct | — | 14.3933 / 20.4528 |
| Qwen3-8B | 9.9067 / 13.5420 | 9.8898 / 13.5430 |
| Qwen3-4B | **13.6584 / 16.8725** | 14.0407 / 17.0142 |

**FourOverSix is not universally safe.** It helps Llama-3.1-8B (-0.025 / -0.056) and Qwen3-8B
(-0.017 / +0.000) but costs Qwen3-4B **+0.382 / +0.142**. Every wider preset is worse still there.

This contradicts CLAUDE.md Part 1, which states the wider alpha search is "neutral-to-positive and
never harmful" and should be taken unconditionally. It holds on the Llama models measured and
fails badly on Qwen3-4B.

---

## 3. Best configuration per model

Delta against `nvfp4_4over6`, wikitext / c4:

| model | best config | delta |
|---|---|---|
| Llama-3.1-8B | `heade0_hess_coclcol_m1` | **-0.059 / -0.060** |
| Llama-3.1-8B-Instruct | `heade0_hess_coclcol_m1` | **-0.052 / -0.055** |
| Llama-3.2-1B-Instruct | `heade0_hess_coclcol_m1` | **-0.166 / -0.434** |
| Qwen3-8B | `heade0_hess_coclcol_h1.5` | **-0.069 / -0.080** |
| Qwen3-4B | `a1_hess_h10` | -0.380 / -0.137 |

Calibrated MixFP4 beats `4over6` on all five models. But on Qwen3-4B the right reference is
`nvfp4`, and against that no single configuration wins on both datasets:

| config | wikitext | c4 |
|---|---|---|
| `a1_hess_h10` | +0.0028 | +0.0049 |
| `a1_hess_v0.7` | +0.0097 | **-0.0271** |
| `a1_hess_m1` | +0.3370 | **-0.0259** |

`a1_hess_h10` is indistinguishable from `nvfp4` -- both deltas sit inside the +-0.0044 noise of §0.
`a1_hess_v0.7` is a genuine c4 win (-0.0271) traded against a small wikitext loss (+0.0097, which is
above the noise floor and so real). Both require freezing alpha at 1 AND an extreme election margin;
every default configuration on this model is far behind.

---

## 4. The decomposition: scale vs type

`hess` weights one loss that drives BOTH decisions (`_selection_loss` feeds `_best_over_alphas`,
whose output feeds `_elect_e0m3`). The `hesst` / `hessa` qualifiers split them.

Llama-3.1-8B, delta vs `nvfp4`:

| config | alpha by | type by | wikitext | c4 |
|---|---|---|---|---|
| `nvfp4_4over6` | MSE, {1,1.5} | — | -0.025 | -0.056 |
| `headx_e2m1` | MSE, 5 cand | — | -0.034 | -0.060 |
| `base_hess_e2m1` | **importance, {1,1.5}** | — | **-0.067** | **-0.095** |
| `headx_hessa_e2m1` | **importance, 5 cand** | — | **-0.073** | **-0.106** |
| `a1_h1.5` | none (alpha=1) | MSE | **+0.003** | +0.002 |
| `a1_hess_h1.5` | none (alpha=1) | importance | -0.033 | -0.050 |
| `headx_hesst_h1.5` | MSE | importance | -0.048 | -0.071 |
| `heade0_hess_coclcol_m1` | importance | importance + reorder | -0.084 | -0.116 |

Three readings:

1. **The MSE-based type election is harmful.** At alpha=1 it is +0.003 on Llama-3.1-8B and
   **+1.289** on Qwen3-4B. The E0M3 type block only becomes useful once the criterion is
   importance-weighted. What looked like its gain in earlier work was the alpha search bundled
   alongside it.
2. **Most of the alpha gain is in the 4-or-6 decision itself**: -0.067 of the -0.073 comes from
   choosing between {1, 1.5} correctly. Widening to five candidates adds only -0.006.
3. **The halves are sub-additive**: -0.073 (alpha) + -0.014 (type, measured on top of MSE-alpha)
   would be -0.087; measured together it is -0.084.

Matched comparison for the type block alone — same alpha criterion, E0M3 off vs on:

| alpha by | E0M3 off | E0M3 on | type block's own gain |
|---|---|---|---|
| MSE | 6.5896 / 9.4192 | 6.5755 / 9.4086 | **-0.014 / -0.011** |
| importance | 6.5510 / 9.3737 | 6.5403 / 9.3660 | **-0.011 / -0.008** |

So E0M3 does win against its matched baseline, consistently, by about 0.01 — near the noise floor
of §0.

---

## 5. Why Qwen3-4B is different

Every MixFP4 mechanism inverts on this model:

| mechanism | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| alpha widening (MSE) | -0.025 | **+0.382** |
| alpha widening (importance) | -0.067 | **+0.663** |
| type election (MSE, alpha=1) | +0.003 | **+1.289** |
| type election (importance, alpha=1) | -0.033 | +0.010 (strict `v0.7`) |

The mechanism is block peakedness. From the selection analysis (§8), E2M1 is log-spaced and coarse
at the top, and those top codes are what absorb an outlier; `alpha > 1` maps the block maximum to a
lower code and **discards exactly those codes**. E0M3 is uniform and has no top-code headroom at
all. A model with peakier blocks can afford neither. Llama's flatter blocks can afford both.

Two facts confirm the format is not forcing anything:

* `clipa1_e2m1` reproduces `nvfp4` **bit for bit** at both 8x64 and 1x16. E0M3 is a genuine
  addition and the floor is exactly reachable.
* At **1x16** — one free, independent choice per 16-element block, the finest granularity the
  format allows — MSE-selected E0M3 still costs **+0.509** on Qwen3-4B (`nvif4` 14.1675 vs
  `nvfp4` 13.6584). So weight MSE is anti-correlated with perplexity for this decision on this
  model, and no election rule reading that signal is safe at any tile size.

Also note **coarse beats fine here**: `a1_hess_h3` at 8x64 (+0.239) is far better than anything at
1x16 (+0.509). The coarse tile plus a strict rule acts as a regulariser — it elects less, and
electing less is what helps — inverting the usual reading of 1x16 as the ceiling.

---

## 6. Reordering: worth and cost

### What it is worth

Only measured usefully once the tag grid is importance-weighted. On the MSE grid the search matched
its own cell-shuffle control (+0.003 of the 1x16 ceiling) and perplexity moved with no consistent
sign across 5 models x 2 datasets (7 of 10 cells worse).

On the corrected grid it contributes about **-0.007** on top of importance-picked alpha
(Llama-3.1-8B), and it is in the best configuration on 4 of 5 models.

### What it costs

Prefill only (W4A4 is prefill-bound), seq 2048, batch 1, H100. Block GEMM total 1.159 ms.

Permuting `W`'s columns requires the activation permuted identically — `(XP)(WP)^T = XW^T`. The
question is never whether that cancels, it is **who produces the activation and who else reads it**.

| axis | per-layer order? | absorbed into |
|---|---|---|
| `x_norm` -> `q/k/v` | **free** | RMSNorm output write + gamma (the rms reduction is order-invariant) |
| `x_norm2` -> `gate/up` | **free** | RMSNorm output write + gamma |
| `h` -> `down_proj` columns | **free** | `gate/up` rows |
| `attn_out` -> `o_proj` columns | **free** | `v` rows, within head |
| residual itself | one global order | `o/down` **rows**, so they write back in global order |

**The only real constraint is that `q`, `k`, `v` must share one order and `gate`, `up` must share
another**, because each trio/pair reads the same normalized tensor.

| variant | gathers/layer | ms | % of block GEMM |
|---|---|---|---|
| independent order per matrix — **what `coclcol` measures** | 5 | 0.155 | **13.4%** |
| one order per norm site, per layer | 0 (fused into RMSNorm) | 0 | **0%** |
| one global residual order | 0 | 0 | 0% |
| `down_proj` + `o_proj` per layer | 0 | 0 | 0% |

Per-matrix the unfused cost is worse than the average suggests: at seq 2048 a 4096-wide gather is
0.031 ms while `k_proj`/`v_proj`'s GEMM is 0.025 ms — the gather costs more than the matmul.

**Every `coclcol` number in this report is the first row** — per-layer freedom on all seven
matrices, which is not purchasable at zero cost. The deployable variant (one order per norm site)
is strictly more constrained and **has not been measured**, so -0.007 is an upper bound on it.

Offline cost is a non-issue: ~22.7 min for a full Llama-3.1-8B quantization pass against ~8 s
without, one-off at quantization time.

---

## 7. Calibration

`hess` needs `E[x_j^2]` per input channel. Measured against a 4-batch wikitext reference, fraction
of 8x64 tile elections unchanged:

| source | agreement |
|---|---|
| one 2048-token wikitext batch | **0.957** |
| 4 batches of c4 (different corpus) | **0.943** |
| 4 batches of random token ids | 0.881 |
| no calibration (plain MSE) | 0.828 |

**One forward pass of arbitrary real text recovers 95%+.** The corpus barely matters; one batch
matches four. Random tokens do not substitute, so it needs real text — just very little of it.

Cross-domain transfer is already implicit in every number here: calibration is drawn from wikitext
**train**, and the c4 column is evaluated on a different corpus, where the gains are largest
(Llama-3.1-8B c4 -0.057, Qwen3-8B -0.080). `build_calibration` draws from the train split
explicitly to avoid leaking the evaluation tokens.

**No calibration-free substitute was found.** The tiles `hess` declines are indistinguishable from
the ones it keeps on every weight statistic (peakedness 2.809 vs 2.839, energy concentration 0.0620
vs 0.0622, gain concentration 0.1082 vs 0.1096); only `imp_max` separates them, 47% higher. No rule
beat the do-nothing baseline of 0.828 agreement: `relgain` 0.839, `mse_h3` 0.832, `veto3.0` 0.829,
`veto2.5` 0.812, `sqnr` 0.749. Re-ranking by `sqnr`/`relgain` is *worse* than plain MSE at
reproducing `hess` (rho 0.741 vs 0.758).

---

## 8. How the type is chosen

Rank correlation between the E0M3 gain and block peakedness `block_max / block_rms`:

| | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| rho(gain, max/rms) | **-0.591** | **-0.592** |
| rho(gain, kurtosis) | -0.478 | -0.474 |
| max/rms, E0M3-preferring | 2.007 | 1.979 |
| max/rms, E2M1-preferring | 2.359 | 2.325 |
| blocks preferring E0M3 | 0.537 | 0.552 |
| tiles electing at 8x64 (MSE -> hess) | 0.263 -> 0.176 | 0.295 -> 0.223 |
| importance ratio E2M1/E0M3 | 1.18x | **2.35x** |

A 16-sample Gaussian block sits at max/rms 2.0, and the E0M3-preferring population averages
1.98-2.01 with sub-Gaussian kurtosis. **E0M3 is the "this block has no outlier" grid** — uniform
spacing suits a flat block; E2M1's log spacing is coarse at the top and absorbs an outlier cheaply.

`hess` does not find new wins: it flips 11-16% of blocks, elects on ~33% **fewer** tiles, and the
tiles it does elect sit on lower-importance channels. It declines elections that would damage
channels the output depends on. The last row is why Qwen3-4B suffers most — its peaked,
E2M1-preferring blocks sit on channels 2.35x more important than elsewhere, so every wrong election
costs double.

---

## 9. Negative results

Each was measured with a control, and each is a null or a loss.

| idea | result |
|---|---|
| Reordering the **weight** tag grid (MSE) | search matched its cell-shuffle control (+0.003 of ceiling); ppl worse on 7 of 10 model x dataset cells |
| Row axis | `row_share` 0.021, profile correlation 1.07x control — no structure |
| Optimizing **purity** directly | harm rose 12.4% -> 14.0%; mass purity 0.673 vs shuffled 0.674 |
| Stricter election rules | `argmin`/`h3`/`h5`/`v0.6-0.75`/`tol`/`dominance` all worse than `h1.5` |
| `dominance` (purity = 1.0 required) | elects **0.0%** of tiles before and after reordering |
| Reordering by weight **magnitude** | +-0.1%, same band as a random control |
| Reordering **activation channels** by magnitude | error **worse** by 4.27%, worst (-55%) on the layer with the strongest channel structure |
| Reorder then **rotate** (block-diagonal Hadamard) | `spread - identity` +0.06/-0.35/-0.34% at rot 16/64/128 — noise |
| Peakedness **veto** on activations (`pv<tau>`) | worse at every threshold; redundant with the error-based election, which already encodes peakedness |
| 10x search budget | purity +0.002, gap over control unchanged to 4 decimals |

Two structural facts explain most of these:

* **Retention is 1/sqrt(cells per tile)** on the MSE grid. `1x128` (8 cells along a row) and
  `8x16` (8 cells down a column) both retain ~0.36; `8x64` (32 cells) retains 0.168 against
  1/sqrt(32) = 0.177. Geometry is irrelevant, only the count — and a permutation cannot change the
  count.
* A **partition search overfits**: on an i.i.d. Gaussian matrix it reports +0.203 of the ceiling
  over the identity order and **+0.001** over a cell-shuffled control. Any reordering claim without
  that control is unmeasured.

The one positive structural result: the **activation** tag grid has the channel structure the
weight one lacks — `col_share` **0.169** vs 0.004, and reordering beats its shuffle control by
**+0.081** at 16x64 vs +0.003 on weights. Applying the type block to activations on the fly at
16x64 is worth **-0.014** wikitext in W4A4 with no reordering at all.

---

## 10. Corrections to CLAUDE.md

1. **"The wider alpha search is neutral-to-positive and never harmful."** False on Qwen3-4B:
   `4over6` is +0.382 wikitext worse than plain `nvfp4`, and every wider preset is worse still.
2. **The E0M3 type block treated as a modest model-dependent extra** (worst case +0.0165 on
   Llama-2-7B). Measured worst case is **+1.289** (Qwen3-4B at alpha=1). It helps 1 of 5 models on
   the MSE criterion and 4 of 5 once importance-weighted.
3. **"The axis that matters is K."** On perplexity the two axes contribute about equally —
   rows +0.008, columns +0.009, both +0.022 (all harmful on the MSE grid).
4. **`headx` is a 5-candidate scale search**, not the 4-or-6 pair. Most of its benefit (-0.067 of
   -0.073) comes from the 2-candidate decision alone.

---

## 11. Gaps

* The **deployable** reordering variant (one order per norm site per layer) is unmeasured. All
  `coclcol` numbers assume per-layer freedom on all seven matrices, which costs 13.4% of prefill
  GEMM.
* `hess` is unmeasured on Llama-2-7B and Llama-3.2-3B (gated, no HF token on this account).
* Run-to-run variance of `hess` configurations (~0.004) is the same order as the type block's
  contribution. Repeat runs would be needed to separate them.
* The W4A4 activation-permutation path (`QuantConfig.a_perm`) is implemented but its perplexity
  run did not complete.

---

## 12. Reproduction

Everything runs through Slurm; the CPU-only jobs request a GPU only because cores are rationed at
12 per GPU on this cluster.

```bash
sbatch slurm/hess_ppl.sbatch llama-3.1-8b-local      # the main sweep (run_ppl_sweep.py path)
sbatch slurm/baseline_cmp.sbatch qwen3-4b            # nvfp4 vs 4over6 vs wider presets
sbatch slurm/e0m3_choice.sbatch llama-3.1-8b-local   # how the type is chosen
sbatch slurm/calib_rob.sbatch llama-3.1-8b-local     # calibration robustness
sbatch slurm/bench_cost.sbatch                       # runtime cost of the gathers
sbatch slurm/reorder_tests.sbatch                    # 9 reorder + 52 MixFP4 CPU tests
```

`run_ppl.py` has **no importance plumbing** — `hess` configurations run through it silently
quantize without importance. Use `run_ppl_sweep.py`.

This account has no HuggingFace token; gated models load from local snapshot paths recorded in
`model2path.json` (`*-local`), which never contact the hub.
