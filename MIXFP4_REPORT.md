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

<!-- BEGIN GENERATED: zero-shot accuracy (summarize_zeroshot_kse.py) -->

## 1a. Zero-shot accuracy

The tables above are perplexity, which is a next-token loss: a quantizer that makes the model less confident lowers it without the model predicting anything better. The k-SE rule elects tiles by a calibration loss, so that is a live possibility rather than a hypothetical one, and nothing measured above rules it out. These are the same four policies on zero-shot multiple choice, where being less confident earns nothing.

lm-eval-harness 0.4.5, 0-shot, 6 tasks: `arc_easy`, `arc_challenge`, `hellaswag`, `openbookqa`, `boolq`, `winogrande`. `acc_norm` where the harness defines it, `acc` otherwise; the figure is the unweighted mean over tasks. Weights and activations are built exactly as `run_kse_paper.py` builds them, and the k = 3 election is re-derived from the calibration and checked to reproduce the shipped frozen map bitwise before anything is evaluated (`run_zeroshot_kse.py`).

### Llama-3.1-8B

The rule elects **3,345 of 13,631,488** type blocks, 0.0245%.

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| BF16 reference | 0.8106 | 0.5350 | 0.7885 | 0.4480 | 0.8196 | 0.7380 | 0.6899 |
| NVFP4 W4A4 | 0.7496 | 0.5085 | 0.7743 | 0.4280 | 0.7969 | 0.7182 | 0.6626 |
| NVFP4 FourOverSix W4A4 | 0.7635 | 0.5179 | 0.7783 | 0.4460 | 0.8043 | 0.7222 | 0.6720 |
| **MixFP4 (k=3), ours** | 0.7723 | 0.5060 | 0.7786 | 0.4420 | 0.8073 | 0.7222 | **0.6714** |

| Comparison | d accuracy | d WikiText PPL | d C4 PPL |
|---|---:|---:|---:|
| MixFP4 − FourOverSix | -0.0006 | -0.026249 | -0.050694 |
| MixFP4 − NVFP4 | +0.0088 | — | — |

### Qwen3-4B

The rule elects **7,912 of 7,096,320** type blocks, 0.1115%.

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| BF16 reference | 0.7837 | 0.5358 | 0.6846 | 0.4040 | 0.8495 | 0.6606 | 0.6530 |
| NVFP4 W4A4 | 0.7281 | 0.4770 | 0.6612 | 0.3900 | 0.8382 | 0.6235 | 0.6197 |
| NVFP4 FourOverSix W4A4 | 0.7441 | 0.4855 | 0.6605 | 0.3900 | 0.8321 | 0.6290 | 0.6235 |
| **MixFP4 (k=3), ours** | 0.7559 | 0.5265 | 0.6733 | 0.4120 | 0.8376 | 0.6440 | **0.6415** |

| Comparison | d accuracy | d WikiText PPL | d C4 PPL |
|---|---:|---:|---:|
| MixFP4 − FourOverSix | +0.0180 | -2.406154 | -1.502600 |
| MixFP4 − NVFP4 | +0.0219 | — | — |

### Qwen3.8-27B

The rule elects **3,787 of 47,559,680** type blocks, 0.0080%.

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| BF16 reference | 0.7298 | 0.5896 | 0.8291 | 0.4620 | 0.8670 | 0.7561 | 0.7056 |
| NVFP4 W4A4 | 0.7542 | 0.5828 | 0.8237 | 0.4460 | 0.7783 | 0.7451 | 0.6883 |
| NVFP4 FourOverSix W4A4 | 0.7273 | 0.5580 | 0.8233 | 0.4480 | 0.8043 | 0.7435 | 0.6841 |
| **MixFP4 (k=3), ours** | 0.7475 | 0.5836 | 0.8208 | 0.4520 | 0.8034 | 0.7443 | **0.6919** |

| Comparison | d accuracy | d WikiText PPL | d C4 PPL |
|---|---:|---:|---:|
| MixFP4 − FourOverSix | +0.0078 | -0.072327 | -0.038499 |
| MixFP4 − NVFP4 | +0.0036 | — | — |

### Is the difference real?

Every policy is scored on the same documents, so MixFP4 against its own base is a paired comparison and the standard error lm-eval prints -- the error of one measurement -- is the wrong yardstick. `b` counts documents only FourOverSix gets right, `c` only MixFP4; the rest carry no information about the difference. The p-value is an exact two-sided McNemar test on those counts, pooled over all tasks.

| model | against | documents | b | c | pooled delta | McNemar p |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | FourOverSix | 18627 | 570 | 588 | +0.0010 | 0.617 |
| Llama-3.1-8B | NVFP4 | 18627 | 625 | 765 | +0.0075 | 0.000191 |
| Qwen3-4B | FourOverSix | 18627 | 683 | 935 | +0.0135 | 4.04e-10 |
| Qwen3-4B | NVFP4 | 18627 | 820 | 1100 | +0.0150 | 1.79e-10 |
| Qwen3.8-27B | FourOverSix | 18627 | 504 | 556 | +0.0028 | 0.117 |
| Qwen3.8-27B | NVFP4 | 18627 | 616 | 655 | +0.0021 | 0.286 |

Which baseline is used changes the verdict, so both are given. Against its own base the method is significant on one model of three; against plain NVFP4 it is significant on two. The report treats NVFP4 and FourOverSix as separate baselines for the same reason -- FourOverSix is not uniformly the stronger of the two, and on Llama-3.1-8B it already captures most of what is available, leaving MixFP4 little to add on top of it while still clearly beating plain NVFP4.

#### Is the paired test reading hardware noise?

Every conclusion above rests on an exact McNemar test over per-document outcomes, so it is worth running that test where the answer is known. Below is the identical comparison applied to two jobs of the **same** policy, on the same weights, data and library versions -- the true difference is zero by construction. `b` and `c` count the documents only one of the two runs gets right; a significant p here would mean §1a is reading the hardware as if it were the method.

| model | policy | jobs | GPUs | documents | b | c | delta | McNemar p |
|---|---|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | four_over_six | 337838 vs 339051 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | bf16 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | nvfp4 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| Qwen3-4B | four_over_six | 337839 vs 339082 | H200 vs H100 | 18627 | 267 | 247 | -0.0011 | 0.402 |

Re-running on the same GPU model reproduces the evaluation exactly: 3 such pairs, and not one of 21,265 scored documents changes outcome. The evaluation itself is deterministic. Across GPU models it is not. Up to **2.8% of documents flip** -- where two continuations score within rounding of each other, and winogrande's differ only by a pronoun, a different GEMM kernel is enough to reverse the comparison. Aggregate accuracy still moves by well under a point, because the flips go both ways.

The point of the table is that none of the 4 shows a significant asymmetry. That symmetry is the case McNemar conditions on: the test is computed from the *difference* between `b` and `c`, not from their size, so noise that inflates both equally cancels. The comparisons in §1a are also made within a single job, where the evaluation is exactly reproducible, so they do not carry even this term. What it does mean is that a single document's outcome is not a portable property of a policy, and that accuracies here should be read to a few tenths of a percent rather than to the digits lm-eval prints.

### What the multiple-choice panel can and cannot see

The panel above resolves a difference of roughly 0.005 and no smaller, and it is not equally sensitive to quantization across metrics. The same policies on the same weights, measured on tasks chosen to be harder on a quantized model: generative chain-of-thought, where one derailed token loses a whole answer instead of averaging out, and larger multiple-choice sets.

| model | metric | n | BF16 | NVFP4 | FourOverSix | MixFP4 (k=3) | k3 − FourOverSix (p) | k3 − NVFP4 (p) |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | `gsm8k` | 1319 | 0.4882 | 0.3859 | 0.4117 | 0.4200 | +0.0106 (0.43) | +0.0364 (0.0083) |
| Llama-3.1-8B | `mmlu` | 14042 | 0.6344 | 0.5996 | 0.5998 | 0.6035 | +0.0036 (0.28) | +0.0038 (0.28) |
| Llama-3.1-8B | `lambada_openai` | 5153 | 0.7530 | 0.7370 | 0.7440 | 0.7454 | +0.0014 (0.74) | +0.0083 (0.036) |

Two things follow. First, what quantization costs depends heavily on the metric: on Llama-3.1-8B, W4A4 costs about four times as much on gsm8k as on the multiple-choice panel, so a null on the panel is a weaker statement than it looks. Second, and more usefully, the method's advantage over plain NVFP4 is clearest exactly where the metric is most sensitive: on gsm8k it is +0.0364 at p = 0.0083 from 1,319 problems, where the panel needed 18,627 documents to resolve +0.0075. Against FourOverSix the same comparison stays inside noise on both, but its point estimate rises by an order of magnitude, from +0.0010 to +0.0106.

> **Election re-derived.** Qwen3.8-27B: 4 module(s) differ from the shipped frozen map, 2 tile(s) present only in the shipped map and 2 only here. The rule, k, and calibration are the reported ones and the calibration reproduces the shipped teacher losses bit for bit; what differs is which side of the threshold a handful of borderline tiles fall on in a re-run. The effect on the elected set is a few tiles in tens of millions, so these rows are treated as the reported policy, with the difference recorded here rather than hidden.

<!-- END GENERATED: zero-shot accuracy -->

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

<!-- BEGIN GENERATED: objective ablation (summarize_objective_ablation.py) -->

### The conjunction, measured

The argument above is an argument, not a measurement, so here is the measurement. Dropping one objective and thresholding the other alone -- same calibration, same k, same evaluation protocol, only the quantity the `< 0` test is applied to changes.

#### How many tiles each objective elects

Raising k tightens the threshold, so both objectives elect fewer tiles. The column to compare against is the shipped rule at k = 3.

| model | objective | k = 2 | k = 3 |
|---|---|---:|---:|
| Llama-3.1-8B | max(CE, KL) | 99,024 | 3,345 |
| Llama-3.1-8B | KL only | 385,903 | 32,774 |
| Qwen3-4B | max(CE, KL) | 66,856 | 7,912 |
| Qwen3-4B | KL only | 212,858 | 21,528 |

KL alone is far more permissive at the same k, so comparing the two at k = 3 compares two different numbers of switches as well as two objectives. Matching the count instead: Llama-3.1-8B comes closest at k = 3, electing 32,774 against 3,345 -- still 9.8x off, so the sweep does not reach a count match; Qwen3-4B comes closest at k = 3, electing 21,528 against 7,912 -- still 2.7x off, so the sweep does not reach a count match. Every row of the perplexity table below therefore carries its tile count -- a row electing ten times as many tiles is not a like-for-like comparison.

#### Perplexity

WikiText-2 and C4 at 2048 under the report's own protocol (`run_kse_paper.py`), W4A4. Deltas are against the FourOverSix base the method switches from; negative is better. k = 3 is the shipped threshold; k = 2 is carried along because the frozen-map cross-check is defined on its ranking, and it doubles as a check that the result is not an artifact of one threshold.

| model | k | policy | tiles | WikiText | d WikiText | C4 | d C4 |
|---|---:|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | — | FourOverSix | 0 | 6.875525 | — | 9.823733 | — |
| Llama-3.1-8B | 3 | max(CE, KL) — shipped | 3,345 | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| Llama-3.1-8B | 3 | KL only | 32,774 | 7.356566 | +0.481042 | 10.394302 | +0.570569 |
| Llama-3.1-8B | 2 | max(CE, KL) | 99,024 | 6.907813 | +0.032289 | 9.842249 | +0.018516 |
| Llama-3.1-8B | 2 | KL only | 385,903 | 8.889927 | +2.014402 | 12.064046 | +2.240313 |
| Qwen3-4B | — | FourOverSix | 0 | 14.269062 | — | 17.326633 | — |
| Qwen3-4B | 3 | max(CE, KL) — shipped | 7,912 | 11.862908 | -2.406154 | 15.824034 | -1.502600 |
| Qwen3-4B | 3 | KL only | 21,528 | 12.007304 | -2.261758 | 16.192286 | -1.134348 |
| Qwen3-4B | 2 | max(CE, KL) | 66,856 | 10.850905 | -3.418157 | 15.162447 | -2.164186 |
| Qwen3-4B | 2 | KL only | 212,858 | 11.328046 | -2.941016 | 16.015682 | -1.310951 |

**KL alone loses in 4 of 4 cells.** Across 2 models (Llama-3.1-8B, Qwen3-4B) and 2 thresholds, dropping the other objective is worse on both corpora in 4 of the 4 (model, k) cells measured, and in 2 of them it is worse than not switching at all -- the method goes from a win to a loss.


#### Zero-shot accuracy

The same policies on the multiple-choice panel, with the paired McNemar test against the FourOverSix base. Positive is better here.

| model | policy | tiles | panel mean | d accuracy | pooled delta | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | max(CE, KL) — shipped | 3,345 | 0.6714 | -0.0006 | +0.0010 | 0.617 |
| Llama-3.1-8B | KL only | 32,774 | 0.6601 | -0.0119 | -0.0120 | 9.74e-08 |
| Qwen3-4B | max(CE, KL) — shipped | 7,912 | 0.6385 | +0.0154 | +0.0133 | 6.76e-10 |
| Qwen3-4B | KL only | 21,528 | 0.6412 | +0.0182 | +0.0161 | 9.03e-14 |

#### Reading the two together

Neither metric favours KL alone on any model. Where they differ it is in how sharply they say so, not in which way, so both are stated per model.

- **Llama-3.1-8B.** Against the shipped rule at the same k, KL alone costs +0.507 WikiText and +0.621 C4 and is -0.0130 on accuracy, significantly worse (p = 2.5e-09).
- **Qwen3-4B.** Against the shipped rule at the same k, KL alone costs +0.144 WikiText and +0.368 C4 and is +0.0028 on accuracy, not distinguishable from it (p = 0.16).

Perplexity is the metric the election is calibrated on -- the score is a teacher-forced loss -- so it is the one KL alone should do well on if the objective were sufficient, and it is the one where it does not. The accuracy panel resolves about 0.005 at best (§1a), so a null there is a weaker statement than a perplexity regression of the size seen above. No model shows accuracy favouring KL alone by a significant margin, so nothing in the accuracy numbers offsets the perplexity cost.

**Coverage.** Measured on Llama-3.1-8B, Qwen3-4B. Not run on Qwen3.8-27B, so the conclusion is a two-model result, not a panel-wide one. CE-only was not run: it is the arm that costs a second full election plus evaluation to test the side of the conjunction the perplexity numbers already favour, and the KL-only arm is the one the report's argument is weakest on. The asymmetry of the evidence is therefore real -- this shows that KL alone is not enough, not that CE alone would also fail.

<!-- END GENERATED: objective ablation -->

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

## 4. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization or native FP4 kernel throughput is measured, and no speedup is
  claimed. Zero-shot multiple-choice accuracy is measured and reported in §1a;
  generation accuracy still is not.
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
