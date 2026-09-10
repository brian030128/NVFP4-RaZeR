# MixFP4: choosing the FP4 element type per tile

NVFP4 hardware can already read a weight operand tile as either E2M1 or E0M3 at
no cost. This report is about how to set that one bit per 8x64 tile, and what it
buys. Every number is the released 2048-token evaluation: WikiText-2 raw test in
141 full nonoverlapping windows, C4 as 256 seed-0 crops from validation shard
00000, tensor-wide activation factors, SDPA, WikiText cached per window with C4
uncached, and the released float32 perplexity aggregation. Measurements taken
under any other protocol are not reported here.

## 1. Results

Baselines are NVFP4 and NVFP4 FourOverSix, both W4A4. The method is MixFP4: the same
FourOverSix E2M1 weights, with the tiles the rule elects switched to E0M3. Calibration
uses OpenWebMath and CodeParrot only, so WikiText-2 and C4 are held out for every row.
BF16 is the unquantized reference, not a competitor.

### Llama-3.1-8B

13,631,488 type blocks of 8x64 across the quantized text linear weights. The rule elects **3,345 of them, 0.0245%** — about one block in 4,075.

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 reference | — | 6.240087 | 8.958212 |
| NVFP4 W4A4 | 0 | 6.940252 | 9.925099 |
| NVFP4 FourOverSix W4A4 | 0 | 6.875525 | 9.823733 |
| **MixFP4 (k=3), ours** | 3,345 | **6.849275** | **9.773040** |

| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 − FourOverSix | -0.026249 | -0.003825 ±0.001588 | -0.050694 | -0.005174 ±0.001778 |
| MixFP4 − NVFP4 | -0.090977 | -0.013195 ±0.002273 | -0.152060 | -0.015439 ±0.002769 |
| FourOverSix − NVFP4 | -0.064728 | -0.009370 ±0.002000 | -0.101366 | -0.010266 ±0.002287 |

### Qwen3-4B

7,096,320 type blocks of 8x64 across the quantized text linear weights. The rule elects **7,912 of them, 0.1115%** — about one block in 897.

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 reference | — | 13.662473 | 16.643560 |
| NVFP4 W4A4 | 0 | 13.936539 | 17.293613 |
| NVFP4 FourOverSix W4A4 | 0 | 14.269062 | 17.326633 |
| **MixFP4 (k=3), ours** | 7,912 | **11.862908** | **15.824034** |

| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 − FourOverSix | -2.406154 | -0.184677 ±0.007504 | -1.502600 | -0.090714 ±0.003758 |
| MixFP4 − NVFP4 | -2.073630 | -0.161098 ±0.006276 | -1.469580 | -0.088807 ±0.003390 |
| FourOverSix − NVFP4 | +0.332523 | +0.023580 ±0.004994 | +0.033020 | +0.001907 ±0.002283 |

### Qwen3.8-27B

47,559,680 type blocks of 8x64 across the quantized text linear weights. The rule elects **3,785 of them, 0.0080%** — about one block in 12,565.

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 reference | — | 7.050375 | 9.893323 |
| NVFP4 W4A4 | 0 | 7.579994 | 10.230958 |
| NVFP4 FourOverSix W4A4 | 0 | 7.287076 | 10.188365 |
| **MixFP4 (k=3), ours** | 3,785 | **7.214750** | **10.149866** |

| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 − FourOverSix | -0.072327 | -0.009975 ±0.003516 | -0.038499 | -0.003786 ±0.000809 |
| MixFP4 − NVFP4 | -0.365244 | -0.049385 ±0.008052 | -0.081092 | -0.007958 ±0.001083 |
| FourOverSix − NVFP4 | -0.292918 | -0.039410 ±0.007762 | -0.042593 | -0.004172 ±0.001171 |

MixFP4 improves both datasets on every model against both baselines, with every paired
two-SE interval excluding zero. Note FourOverSix is not uniformly the stronger baseline:
on Qwen3-4B plain NVFP4 beats it, so the method is measured against the better of the
two, not only against its own base.

### The count is not a tuned constant

The same rule at other values of k, for reference. k is fixed at 3 for every model and
is not selected per model or per dataset.

**Llama-3.1-8B**

| k | Tiles | % of blocks | ΔWiki | ΔC4 |
|---|---:|---:|---:|---:|
| 2 | 99,024 | 0.7264% | +0.032289 | +0.018516 |
| **3** | 3,345 | 0.0245% | -0.026249 | -0.050694 |
| 4 | 541 | 0.0040% | -0.023511 | -0.046163 |
| 5 | 267 | 0.0020% | -0.014463 | -0.028150 |
| 6 | 145 | 0.0011% | -0.014984 | -0.022989 |

**Qwen3-4B**

| k | Tiles | % of blocks | ΔWiki | ΔC4 |
|---|---:|---:|---:|---:|
| 2 | 66,856 | 0.9421% | -3.418157 | -2.164186 |
| **3** | 7,912 | 0.1115% | -2.406154 | -1.502600 |
| 4 | 1,837 | 0.0259% | -1.553578 | -0.968634 |
| 5 | 576 | 0.0081% | -0.858928 | -0.545380 |
| 6 | 222 | 0.0031% | -0.446252 | -0.289104 |

**Qwen3.8-27B**

| k | Tiles | % of blocks | ΔWiki | ΔC4 |
|---|---:|---:|---:|---:|
| 2 | 149,033 | 0.3134% | -0.187205 | -0.086909 |
| **3** | 3,785 | 0.0080% | -0.072327 | -0.038499 |
| 4 | 593 | 0.0012% | -0.031829 | -0.020767 |
| 5 | 165 | 0.0003% | -0.004445 | -0.013606 |
| 6 | 47 | 0.0001% | -0.001054 | -0.001950 |

## 2. How the element type is chosen

### The two candidates

A **scale block** is always 16 elements along K and owns one E4M3 scale; this is
inherited from NVFP4 unchanged. A **type block** is a 2-D tile that owns one
element data type and contains many scale blocks. The two types are:

- **E2M1**, the standard FP4 grid, maximum magnitude 6, log-spaced so it is fine
  near zero and coarse near the block maximum.
- **E0M3**, the evenly spaced signed grid, maximum magnitude 7, uniform.

Both encode 15 values in 16 codes and share the same ue4m3 scale. Only the
spacing differs, so which one is better depends on the distribution inside the
tile, which is why the choice must be data driven rather than fixed.

The tile shape is not free. The public NVFP4 path issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read either operand as E0M3. One instruction cannot
subdivide its operand tile, so for weights in operand B the smallest realizable
type block is `n8 x k64`. Everything here uses exactly that 8x64 tile, with the
E0M3 branch pinned at alpha 1 so no extra scale search is smuggled in.

### The score

Form the canonical FourOverSix Q0 and the E0M3-alpha1 Q1 once, so each tile j has
a fixed difference D_j. At the unchanged FourOverSix W4A4 model, one forward pass
per calibration sequence supports two backward passes, giving each tile a
directional derivative for next-token cross entropy and for KL from the pristine
BF16 teacher:

    g_CE[i,j] = <grad_Wj CE_i, D_j>,   g_KL[i,j] = <grad_Wj KL(teacher||student)_i, D_j>

A negative value predicts that switching tile j to E0M3 lowers that loss. The
gradient is taken at the **quantized** model, not the pristine one, so it finds
corrections that are useful given the errors already present. Activation
quantization uses an identity straight-through derivative during scoring.

### The rule

Over n calibration sequences take each tile's mean and standard error, and switch
tile j to E0M3 when

    max( mean(g_CE) + k SE(g_CE),  mean(g_KL) + k SE(g_KL) ) < 0,  with k = 3

Everything else stays E2M1. There is no cap and no per-model search: the number
of switched tiles is whatever passes.

## 3. Why this rule

### Why a task gradient and not the weight error

The obvious selector compares the two candidates' squared weight error per tile
and takes the smaller. That measures how well a candidate approximates the
pristine weights, not which inputs reach the block or how its outputs move the
prediction. Even for one linear layer the output error is

    E||dW x||^2 = tr(dW E[x x^T] dW^T)

which depends on the input second moment, and that still ignores the downstream
loss and interactions between blocks. A weight-error gain of factor g certifies
an output-error reduction only when g > 1 - 1/kappa(S), and the measured
conditioning of S puts that threshold far above the gains available at a
realizable tile size. So the criterion has to be a task loss.

### Why both CE and KL

The two objectives fail in opposite directions. Cross entropy alone can be
lowered by fitting the calibration corpus's particular observed tokens, which
does not transfer. Teacher KL alone can be lowered by restoring agreement with
the unquantized model on positions the task does not care about, and it is blind
to whether the recovered probability mass sits on the right tokens. Requiring
both means a switch is kept only when it improves the actual task loss **and**
moves the quantized model back toward its own unquantized reference.

Taking the maximum is what makes that a single objective: for each loss
separately, the sum of its estimated upper directional scores is bounded above by
the sum of the per-tile maxima, so one selection bounds both. It also needs no
weighting constant between two quantities that have no common scale, and the
eligible set is unchanged if either objective is rescaled by a positive factor.

### Why k = 3, and why a threshold rather than a fixed count

Rearranged, `mean + k SE < 0` is `|mean|/SE > k`: k is a t-statistic cutoff, the
number of standard errors of evidence a tile must show. It adapts on its own,
because SE measures each model's own score noise — the count is never set, it
falls out.

k = 2 is too loose here, and the reason is multiple comparisons. There are
millions of tiles. Under a null where a tile has no real effect, the chance it
clears the bar on both objectives is about Phi(-k)^2, so at k = 2 one expects
thousands of false positives, and the elected set is measurably polluted by them:
at k = 2 the rule elects 0.73% of Llama-3.1-8B's tiles and **harms** the model.
At k = 3 the expected null count falls to a few dozen out of thousands elected.
k = 3 is the smallest value at which that expected contamination becomes
negligible relative to the selected set, computed from the calibration table
alone with no evaluation data involved.

That null estimate treats the CE and KL tests as independent when they share a
forward pass and are correlated, so it is optimistic in magnitude. The
conclusion it supports is the qualitative one — a 2 SE bar is far too loose
across millions of comparisons — not a precise contamination figure.

## 4. Which tiles the rule elects, and where they are

<!-- BEGIN TILE SELECTION -->
The rule is a threshold, so the map it produces is an outcome rather than a design: nothing in it constrains which matrices, which layers, or which part of a matrix the elected tiles come from. Rebuilt from the frozen calibration scores, every model's per-k count reproduces the published election exactly.

### The elected tiles are spread over matrices and concentrated inside them

| Model | Tiles | Elected | Rate | Matrices holding at least one | Largest single matrix |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 13,631,488 | 3,345 | 1 in 4,075 | 216/224 | 836 (25%) |
| Qwen3-4B | 7,096,320 | 7,912 | 1 in 897 | 241/252 | 1,307 (17%) |
| Qwen3.8-27B | 47,559,680 | 3,785 | 1 in 12,565 | 379/496 | 177 (5%) |

Almost every weight matrix contributes something, so this is not a rule that fires on a handful of layers. The mass is nevertheless very uneven: the five largest contributors hold 32% (Llama-3.1-8B), 38% (Qwen3-4B), 19% (Qwen3.8-27B) of all elected tiles, and the single largest is `L31.down_proj` with 836, `L35.down_proj` with 1,307, `L48.in_proj_qkv` with 177.

### Which projections, and how deep

| Projection | Llama-3.1-8B per 1M | share | Qwen3-4B per 1M | share | Qwen3.8-27B per 1M | share |
|---|---|---|---|---|---|---|
| `q_proj` | 218 | 6.8% | 589 | 5.5% | 95 | 4.9% |
| `k_proj` | 423 | 3.3% | 722 | 1.7% | 18 | 0.1% |
| `v_proj` | 671 | 5.3% | 2729 | 6.4% | 519 | 2.2% |
| `o_proj` | 310 | 9.7% | 2482 | 23.1% | 241 | 6.3% |
| `gate_proj` | 124 | 13.6% | 338 | 7.5% | 34 | 10.0% |
| `up_proj` | 146 | 16.1% | 636 | 14.1% | 48 | 14.2% |
| `down_proj` | 412 | 45.2% | 1889 | 41.8% | 41 | 12.1% |
| `out_proj` | — | — | — | — | 105 | 8.2% |
| `in_proj_qkv` | — | — | — | — | 240 | 31.2% |
| `in_proj_z` | — | — | — | — | 136 | 10.6% |
| `in_proj_b` | — | — | — | — | 260 | 0.2% |
| `in_proj_a` | — | — | — | — | 174 | 0.1% |

Rates are elected tiles per million tiles of that kind, so they are comparable across matrices of very different sizes; shares are of the model's elected set. By rate, `v_proj` carries the highest rate on all three models (671, 2729, 519 per 1M), while the largest share of the elected set goes to the MLP `down_proj` on the two dense models and to the hybrid model's linear-attention `in_proj_qkv`. No kind is exempt and none is anywhere near saturated: the most enriched projection in the panel is `v_proj` on Qwen3-4B at 2729 per million, one tile in 366.

| Depth quarter | Llama-3.1-8B | Qwen3-4B | Qwen3.8-27B |
|---|---|---|---|
| first | 634 (19%) | 1,777 (22%) | 409 (11%) |
| second | 535 (16%) | 1,238 (16%) | 632 (17%) |
| third | 490 (15%) | 1,087 (14%) | 958 (25%) |
| last | 1,686 (50%) | 3,810 (48%) | 1,786 (47%) |

Depth is the strongest single predictor: about half of every model's elected tiles are in its last quarter of layers (50%, 48%, 47%), and the last layer alone holds 976 of 3,345 on Llama-3.1-8B, 1,854 of 7,912 on Qwen3-4B, 328 of 3,785 on Qwen3.8-27B. The first layers are the secondary peak, so the profile is a shallow U with a much heavier top end, not a monotone trend.

### Position inside a matrix

The reference for every row below is a seeded uniform placement that keeps each matrix's elected count and its tile grid and randomises only the positions, so a departure is structure the rule found rather than an artefact of where the tiles are.

| Statistic | Llama-3.1-8B obs. | unif. | Qwen3-4B obs. | unif. | Qwen3.8-27B obs. | unif. |
|---|---|---|---|---|---|---|
| Most tiles in one row band (output channels) | 19 | 7 | 25 | 11 | 21 | 3 |
| Most tiles in one column band (input channels) | 122 | 10 | 56 | 17 | 28 | 7 |
| Most tiles on one column index, one projection kind | 125 | 16 | 269 | 43 | 137 | 25 |
| Most matrices sharing a column index, one kind | 15 | 13 | 34 | 24 | 25 | 15 |
| Most tiles on one residual channel band | 79 | 35 | 644 | 88 | 358 | 50 |
| Most residual-reading matrices sharing that band | 44 | 31 | 146 | 58 | 102 | 38 |

Both axes are clustered well beyond chance on every model, in two different shapes. Along the input axis, one 64-channel band collects tiles from many output rows: 122 of the 836 tiles elected in `L31.down_proj` on Llama-3.1-8B share the single input band 12672–12735 while spreading over 391 of 512 output bands. Along the output axis the pattern is the mirror image — one row band, meaning one group of eight output channels, elected across much of its own row: 19 of the 64 column bands in row 1491 of `L29.up_proj` on Llama-3.1-8B are elected, 59% of everything that matrix contributes.

The clustering also runs across layers, not only inside a matrix. For the matrices that read the residual stream directly (`q/k/v_proj`, `gate/up_proj`, the hybrid model's `in_proj_*`) a column index means the same hidden channels in every layer, and the elected tiles pile onto a few such bands:

| Model | Hidden channels | Elected tiles there | Residual-reading matrices hit | Uniform placement |
|---|---|---:|---:|---:|
| Llama-3.1-8B | 4032–4095 | 79 | 44/152 | 31 matrices |
| Qwen3-4B | 0–63 | 644 | 146/169 | 58 matrices |
| Qwen3.8-27B | 3968–4031 | 358 | 102/257 | 38 matrices |

On the two Qwen models this is the clearest structure in the whole map: one 64-channel band of the residual stream is elected in 146 of 169 matrices on Qwen3-4B and 102 of 257 matrices on Qwen3.8-27B, against 58 and 38 under uniform placement. On Llama-3.1-8B the cross-layer version is much weaker (44 against 31) and the concentration lives inside single matrices instead. Either way the elected set is a property of particular input channels rather than of particular matrices. This report does not establish the mechanism; it is consistent with the known concentration of activation magnitude in a small number of channels, which is exactly where a uniform grid and a log-spaced grid differ most.

### Which objective binds

Both objectives must clear the bar, but the one that decides is model dependent: Llama-3.1-8B 56% CE, Qwen3-4B 13% CE, Qwen3.8-27B 15% CE. Median t statistics at elected tiles are around -3.4/-3.6, -4.7/-3.5, -3.7/-3.4 (CE/KL), so the elected set is not sitting on the threshold — it clears it comfortably on both. Neither objective is redundant: each is the binding constraint for a substantial share of the elected tiles on all three models.

[Per-model tables, top matrices and hot column indices](results/kse_selection/job_337344/REPORT.md)
<!-- END TILE SELECTION -->

## 5. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is measured,
  and no speedup is claimed.
- Tensor-wide activation factors span the whole teacher-forced window, so these
  are reference-text perplexities, not causal generation likelihoods.
- k = 3 is prespecified from the calibration score distribution and was fixed
  using Llama-3.1-8B and Qwen3-4B. Qwen3.8-27B played no part in choosing it and
  is a held-out check of the rule, not a third fitting model. Three models is
  still a small panel, and one calibration draw is used per model.
- The released evaluator predates the Qwen3.8 architecture, so its BF16 reference
  is measured through this report's own path. That path reproduces the released
  BF16 values exactly for Llama-3.1-8B and Qwen3-4B, which is the basis for
  trusting the 27B row; it is not an independent implementation.
- Two-SE intervals are descriptive evaluation-window intervals. They do not
  adjust for multiple comparisons, WikiText article dependence, or
  calibration-draw variability; one calibration draw per model is used.
- Gradient selection, distillation and sparse optimization are established tools.
  Their use here is not by itself a novelty claim.
- Section 4 describes where one realized map lands, from one calibration draw per
  model. The uniform-placement reference there is a descriptive baseline for how
  clustered the positions are, not a significance test, and nothing in that
  section is evidence that a position-based rule would work without the scores.
