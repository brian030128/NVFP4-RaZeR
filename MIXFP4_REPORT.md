# MixFP4: choosing the FP4 element type per tile

**GB200 follow-up:** [Weight reordering and rotation plan](MIXFP4_GB200_PLAN.md)
targets `m256n256k64` with 256x64 weight type tiles and fixed E2M1 activations.
The arranging results below extend the measured 8x64 reference to 256x64 tiles.

NVFP4 hardware can already read a weight operand tile as either E2M1 or E0M3 at
no cost. This report is about how to set that one bit per 8x64 tile, and what it
buys. The main PPL tables use the released 2048-token evaluation: WikiText-2 raw test in
full nonoverlapping windows (141 for Llama-3.1-8B, 145 for Qwen3.8-27B), C4 as 256 seed-0 crops from validation shard
00000, tensor-wide activation factors, SDPA, WikiText cached per window with C4
uncached, and the released float32 perplexity aggregation. The 256x64 follow-up explicitly separates fresh calibration checks and native
permutation timings from these PPL measurements.

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
| Llama-3.1-8B | four_over_six | 337838 vs 339115 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | k3 | 337838 vs 339115 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | four_over_six | 337838 vs 339800 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | k3 | 337838 vs 339800 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | bf16 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | nvfp4 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | four_over_six | 339051 vs 339115 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | four_over_six | 339051 vs 339800 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | four_over_six | 339115 vs 339800 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Llama-3.1-8B | k3 | 339115 vs 339800 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Qwen3-4B | four_over_six | 337839 vs 339082 | H200 vs H100 | 18627 | 267 | 247 | -0.0011 | 0.402 |
| Qwen3-4B | k3 | 337839 vs 339082 | H200 vs H100 | 18627 | 238 | 213 | -0.0013 | 0.258 |
| Qwen3-4B | four_over_six | 337839 vs 339801 | H200 vs H100 | 18627 | 267 | 247 | -0.0011 | 0.402 |
| Qwen3-4B | k3 | 337839 vs 339801 | H200 vs H100 | 18627 | 238 | 213 | -0.0013 | 0.258 |
| Qwen3-4B | four_over_six | 339082 vs 339801 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| Qwen3-4B | k3 | 339082 vs 339801 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |

Re-running on the same GPU model reproduces the evaluation exactly: 13 such pairs, and not one of 207,535 scored documents changes outcome. The evaluation itself is deterministic. Across GPU models it is not. Up to **2.8% of documents flip** -- where two continuations score within rounding of each other, and winogrande's differ only by a pronoun, a different GEMM kernel is enough to reverse the comparison. Aggregate accuracy still moves by well under a point, because the flips go both ways.

The point of the table is that none of the 17 shows a significant asymmetry. That symmetry is the case McNemar conditions on: the test is computed from the *difference* between `b` and `c`, not from their size, so noise that inflates both equally cancels. The comparisons in §1a are also made within a single job, where the evaluation is exactly reproducible, so they do not carry even this term. What it does mean is that a single document's outcome is not a portable property of a policy, and that accuracies here should be read to a few tenths of a percent rather than to the digits lm-eval prints.

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

The original argument for requiring both was that the two objectives fail in
opposite directions. Cross entropy alone can be lowered by fitting the
calibration corpus's particular observed tokens, which does not transfer.
Teacher KL alone can be lowered by restoring agreement with the unquantized
model on positions the task does not care about, and it is blind to whether the
recovered probability mass sits on the right tokens.

That argument is now measured rather than asserted, and it survives, but not in
the form it was put. Both single objectives were elected and evaluated at five
thresholds on two models, on perplexity and on zero-shot accuracy; the tables
below are the result. Three things in them change how this subsection should be
read.

**Neither objective alone is safe, and they are not equally unsafe.** Dropping
CE is the worse of the two: at an equal budget of switches, KL alone is beaten
by a conjunction setting electing no more tiles at 9 of 10 settings, and on
Llama-3.1-8B at the shipped threshold it does not merely give back the win but
inverts it, to +0.48 WikiText against the FourOverSix base the method switches
from. Dropping KL is beaten at 4 of 10. Neither single objective beats the
conjunction on accuracy at any setting measured.

**Most of what separated them at a fixed k was permissiveness, not the
objective.** At the same k the two elect very different numbers of tiles --
3,345 against 32,774 on Llama-3.1-8B at k = 3 -- so the obvious comparison
confounds which objective is thresholded with how much it lets through.
Tightening k rescues KL alone substantially. What does not follow is that the
objective was fine all along: at an equal budget the conjunction still reaches a
lower perplexity, which is CE selecting different tiles rather than merely
fewer.

**KL's role is narrower than "both must agree" suggests.** Because the rule
elects on the intersection, adding KL to CE can only remove tiles, so its
contribution is exactly the value of the tiles it vetoes. Measured that way it
is essential at the loosest threshold -- where CE alone is worse than not
switching at all on both models, and the veto is worth 3.65 C4 on Qwen3-4B --
and harmful at strict ones, costing up to +2.86 WikiText on Qwen3-4B at k = 6 by
discarding 14,252 tiles CE had accepted correctly. KL is insurance against a
threshold that is too loose, not a second opinion on which tiles are good.

At the shipped k = 3 both halves still earn their place, which is the claim this
subsection needs and the one the evidence supports: KL alone is far worse on
Llama-3.1-8B, and CE alone costs 0.0161 accuracy at p = 1e-11 on Qwen3-4B while
electing 15.9 times as many tiles. What the evidence does not support is that
the conjunction is the best selector at a given budget. It is not, on either
model; what it is, is the one that is never badly wrong, and which single
objective fails is model-dependent and not something the calibration predicts in
advance.

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

| model | objective | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | max(CE, KL) | 99,024 | 3,345 | 541 | 267 | 145 |
| Llama-3.1-8B | KL only | 385,903 | 32,774 | 6,591 | 2,950 | 1,653 |
| Llama-3.1-8B | CE only | 356,302 | 22,906 | 1,104 | 277 | 147 |
| Qwen3-4B | max(CE, KL) | 66,856 | 7,912 | 1,837 | 576 | 222 |
| Qwen3-4B | KL only | 212,858 | 21,528 | 3,569 | 1,226 | 630 |
| Qwen3-4B | CE only | 423,120 | 125,611 | 51,278 | 25,869 | 14,474 |

KL alone is far more permissive at the same k, so comparing the two at k = 3 compares two different numbers of switches as well as two objectives. Matching the count instead: Llama-3.1-8B comes closest at k = 5, electing 2,950 against 3,345; Qwen3-4B comes closest at k = 4, electing 3,569 against 7,912, which is 0.45x the shipped count -- near but not a match. Every row of the perplexity table below therefore carries its tile count -- a row electing ten times as many tiles is not a like-for-like comparison.

#### Perplexity

WikiText-2 and C4 at 2048 under the report's own protocol (`run_kse_paper.py`), W4A4. Deltas are against the FourOverSix base the method switches from; negative is better. k = 3 is the shipped threshold; k = 2 is carried along because the frozen-map cross-check is defined on its ranking, and it doubles as a check that the result is not an artifact of one threshold.

| model | k | policy | tiles | WikiText | d WikiText | C4 | d C4 |
|---|---:|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | — | FourOverSix | 0 | 6.875525 | — | 9.823733 | — |
| Llama-3.1-8B | 2 | max(CE, KL) | 99,024 | 6.907813 | +0.032289 | 9.842249 | +0.018516 |
| Llama-3.1-8B | 2 | KL only | 385,903 | 8.889927 | +2.014402 | 12.064046 | +2.240313 |
| Llama-3.1-8B | 2 | CE only | 356,302 | 6.938570 | +0.063045 | 9.876889 | +0.053156 |
| Llama-3.1-8B | 3 | max(CE, KL) — shipped | 3,345 | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| Llama-3.1-8B | 3 | KL only | 32,774 | 7.356566 | +0.481042 | 10.394302 | +0.570569 |
| Llama-3.1-8B | 3 | CE only | 22,906 | 6.836686 | -0.038839 | 9.768658 | -0.055076 |
| Llama-3.1-8B | 4 | max(CE, KL) | 541 | 6.852014 | -0.023511 | 9.777571 | -0.046163 |
| Llama-3.1-8B | 4 | KL only | 6,591 | 6.971170 | +0.095645 | 9.953737 | +0.130004 |
| Llama-3.1-8B | 4 | CE only | 1,104 | 6.853590 | -0.021935 | 9.781658 | -0.042075 |
| Llama-3.1-8B | 5 | max(CE, KL) | 267 | 6.861062 | -0.014463 | 9.795584 | -0.028150 |
| Llama-3.1-8B | 5 | KL only | 2,950 | 6.881298 | +0.005773 | 9.827334 | +0.003601 |
| Llama-3.1-8B | 5 | CE only | 277 | 6.862983 | -0.012541 | 9.793248 | -0.030485 |
| Llama-3.1-8B | 6 | max(CE, KL) | 145 | 6.860541 | -0.014984 | 9.800744 | -0.022989 |
| Llama-3.1-8B | 6 | KL only | 1,653 | 6.858618 | -0.016906 | 9.784282 | -0.039452 |
| Llama-3.1-8B | 6 | CE only | 147 | 6.858060 | -0.017465 | 9.791759 | -0.031975 |
| Qwen3-4B | — | FourOverSix | 0 | 14.269062 | — | 17.326633 | — |
| Qwen3-4B | 2 | max(CE, KL) | 66,856 | 10.850905 | -3.418157 | 15.162447 | -2.164186 |
| Qwen3-4B | 2 | KL only | 212,858 | 11.328046 | -2.941016 | 16.015682 | -1.310951 |
| Qwen3-4B | 2 | CE only | 423,120 | 13.080847 | -1.188215 | 18.817131 | +1.490498 |
| Qwen3-4B | 3 | max(CE, KL) — shipped | 7,912 | 11.862908 | -2.406154 | 15.824034 | -1.502600 |
| Qwen3-4B | 3 | KL only | 21,528 | 12.007304 | -2.261758 | 16.192286 | -1.134348 |
| Qwen3-4B | 3 | CE only | 125,611 | 11.804390 | -2.464672 | 16.745670 | -0.580963 |
| Qwen3-4B | 4 | max(CE, KL) | 1,837 | 12.715484 | -1.553578 | 16.358000 | -0.968634 |
| Qwen3-4B | 4 | KL only | 3,569 | 12.719244 | -1.549818 | 16.505108 | -0.821526 |
| Qwen3-4B | 4 | CE only | 51,278 | 11.195740 | -3.073322 | 15.752458 | -1.574176 |
| Qwen3-4B | 5 | max(CE, KL) | 576 | 13.410134 | -0.858928 | 16.781254 | -0.545380 |
| Qwen3-4B | 5 | KL only | 1,226 | 13.375737 | -0.893325 | 16.817934 | -0.508699 |
| Qwen3-4B | 5 | CE only | 25,869 | 10.974319 | -3.294744 | 15.322808 | -2.003825 |
| Qwen3-4B | 6 | max(CE, KL) | 222 | 13.822810 | -0.446252 | 17.037529 | -0.289104 |
| Qwen3-4B | 6 | KL only | 630 | 13.801343 | -0.467719 | 17.055054 | -0.271580 |
| Qwen3-4B | 6 | CE only | 14,474 | 10.966258 | -3.302804 | 15.188874 | -2.137759 |

Tightening the threshold is the obvious way to try to rescue a single objective, and it helps a great deal -- on Llama-3.1-8B the KL-only penalty falls from +0.481 WikiText at k = 3 to +0.009 at k = 6. But raising k also shrinks the election, so much of that is buying back permissiveness rather than showing the objective was fine all along.

The fair test is at an equal budget of switches: for each single-objective setting, is there a conjunction setting that elects **no more tiles** and is at least as good on **both** corpora?

| model | objective | k | its tiles | its WikiText | beaten by k | tiles | WikiText |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | KL only | 6 | 1,653 | 6.8586 | 4 | 541 | 6.8520 |
| Llama-3.1-8B | KL only | 5 | 2,950 | 6.8813 | 6 | 145 | 6.8605 |
| Llama-3.1-8B | KL only | 4 | 6,591 | 6.9712 | 6 | 145 | 6.8605 |
| Llama-3.1-8B | KL only | 3 | 32,774 | 7.3566 | 6 | 145 | 6.8605 |
| Llama-3.1-8B | KL only | 2 | 385,903 | 8.8899 | 6 | 145 | 6.8605 |
| Llama-3.1-8B | CE only | 6 | 147 | 6.8581 | — | — | — |
| Llama-3.1-8B | CE only | 5 | 277 | 6.8630 | — | — | — |
| Llama-3.1-8B | CE only | 4 | 1,104 | 6.8536 | 4 | 541 | 6.8520 |
| Llama-3.1-8B | CE only | 3 | 22,906 | 6.8367 | — | — | — |
| Llama-3.1-8B | CE only | 2 | 356,302 | 6.9386 | 6 | 145 | 6.8605 |
| Qwen3-4B | KL only | 6 | 630 | 13.8013 | 5 | 576 | 13.4101 |
| Qwen3-4B | KL only | 5 | 1,226 | 13.3757 | — | — | — |
| Qwen3-4B | KL only | 4 | 3,569 | 12.7192 | 4 | 1,837 | 12.7155 |
| Qwen3-4B | KL only | 3 | 21,528 | 12.0073 | 3 | 7,912 | 11.8629 |
| Qwen3-4B | KL only | 2 | 212,858 | 11.3280 | 2 | 66,856 | 10.8509 |
| Qwen3-4B | CE only | 6 | 14,474 | 10.9663 | — | — | — |
| Qwen3-4B | CE only | 5 | 25,869 | 10.9743 | — | — | — |
| Qwen3-4B | CE only | 4 | 51,278 | 11.1957 | — | — | — |
| Qwen3-4B | CE only | 3 | 125,611 | 11.8044 | 2 | 66,856 | 10.8509 |
| Qwen3-4B | CE only | 2 | 423,120 | 13.0808 | 4 | 1,837 | 12.7155 |

**13 of 20 single-objective settings are beaten by a conjunction setting using no more tiles.** The ones that are not: CE only at Llama-3.1-8B k = 6 (147 tiles), Llama-3.1-8B k = 5 (277 tiles), Llama-3.1-8B k = 3 (22,906 tiles), Qwen3-4B k = 6 (14,474 tiles), Qwen3-4B k = 5 (25,869 tiles), Qwen3-4B k = 4 (51,278 tiles); KL only at Qwen3-4B k = 5 (1,226 tiles).

Asked the other way -- is any **conjunction** setting beaten by a single objective electing no more tiles? -- the answer is not empty, and this is the part the report's argument does not predict:

| model | objective | conjunction k | its tiles | its WikiText | beaten by k | tiles | WikiText |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | KL only | 2 | 99,024 | 6.9078 | 6 | 1,653 | 6.8586 |
| Llama-3.1-8B | CE only | 5 | 267 | 6.8611 | 6 | 147 | 6.8581 |
| Llama-3.1-8B | CE only | 2 | 99,024 | 6.9078 | 6 | 147 | 6.8581 |

So the conjunction is not uniformly the best objective at a given budget. The two single objectives are not equally weak either -- KL only is beaten at 9 of 10 settings; CE only is beaten at 4 of 10 settings -- so the case for requiring both rests much more heavily on KL than on CE. What the conjunction has going for it on this evidence is not that it wins everywhere, but that it is never badly wrong, and which single objective fails is not something the calibration predicts in advance.

Held at the same k instead of the same budget -- the naive swap -- KL alone is worse on both corpora in 7 of 10 (model, k) cells, and in 4 of them worse than not switching at all. That comparison flatters neither objective, since at a fixed k the two elect different numbers of tiles; the budget-matched table above is the one to read.

Held at the same k instead of the same budget -- the naive swap -- CE alone is worse on both corpora in 3 of 10 (model, k) cells, and in 1 of them worse than not switching at all. That comparison flatters neither objective, since at a fixed k the two elect different numbers of tiles; the budget-matched table above is the one to read.


#### Zero-shot accuracy

The same question on the multiple-choice panel of §1a, with the paired McNemar test on per-document outcomes. Each row is compared against the shipped rule **measured in the same job**: §1a's control shows the evaluation is exact on one GPU model and flips 2.8% of documents across two, so policies from different jobs are not a paired comparison. Positive favours the single objective.

| model | policy | tiles | vs shipped k = 3 | panel mean | d vs shipped | p | d vs FourOverSix | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | KL only | 32,774 | 9.80x | 0.6601 | -0.0130 | 2.5e-09 | -0.0120 | 9.74e-08 |
| Llama-3.1-8B | KL only | 2,950 | 0.88x | 0.6706 | -0.0012 | 0.533 | -0.0002 | 0.93 |
| Llama-3.1-8B | KL only | 1,653 | 0.49x | 0.6701 | -0.0007 | 0.718 | +0.0003 | 0.906 |
| Llama-3.1-8B | CE only | 22,906 | 6.85x | 0.6677 | -0.0012 | 0.505 | -0.0003 | 0.905 |
| Qwen3-4B | KL only | 21,528 | 2.72x | 0.6412 | +0.0028 | 0.157 | +0.0161 | 9.03e-14 |
| Qwen3-4B | KL only | 3,569 | 0.45x | 0.6345 | -0.0028 | 0.182 | +0.0105 | 1.07e-06 |
| Qwen3-4B | KL only | 1,226 | 0.15x | 0.6287 | -0.0065 | 0.00166 | +0.0067 | 0.00148 |
| Qwen3-4B | CE only | 125,611 | 15.88x | 0.6223 | -0.0161 | 1.14e-11 | -0.0028 | 0.284 |

**None of the 8 settings beats the conjunction.** 3 are significantly worse and 5 are indistinguishable from it. The rows where a single objective elects far more tiles than the shipped rule are the ones that look closest to it, which is the tile count talking rather than the objective -- the ratio column is there to make that visible. Note also what the last two columns do not say together: a setting can beat the FourOverSix base convincingly and still not reach the conjunction, and several do exactly that.

#### Is KL needed, or would CE alone do?

The rule elects when **both** bounds are negative, so the elected set is the intersection: adding KL to CE can only take tiles away. That gives the question an exact form -- are the tiles CE accepts and KL vetoes worth keeping? Holding the CE threshold fixed and varying only the veto is the one comparison here where matching k is right rather than misleading, because the tile count difference *is* the effect being measured.

| model | k | CE elects | KL vetoes | d WikiText from vetoing | d C4 | CE alone vs the base |
|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | 2 | 356,302 | 257,278 | -0.0308 | -0.0346 | **worse than not switching** |
| Llama-3.1-8B | 3 | 22,906 | 19,561 | +0.0126 | +0.0044 | an improvement |
| Llama-3.1-8B | 4 | 1,104 | 563 | -0.0016 | -0.0041 | an improvement |
| Llama-3.1-8B | 5 | 277 | 10 | -0.0019 | +0.0023 | an improvement |
| Llama-3.1-8B | 6 | 147 | 2 | +0.0025 | +0.0090 | an improvement |
| Qwen3-4B | 2 | 423,120 | 356,264 | -2.2299 | -3.6547 | **worse than not switching** |
| Qwen3-4B | 3 | 125,611 | 117,699 | +0.0585 | -0.9216 | an improvement |
| Qwen3-4B | 4 | 51,278 | 49,441 | +1.5197 | +0.6055 | an improvement |
| Qwen3-4B | 5 | 25,869 | 25,293 | +2.4358 | +1.4584 | an improvement |
| Qwen3-4B | 6 | 14,474 | 14,252 | +2.8566 | +1.8487 | an improvement |

Negative means the veto helps. The answer is not uniform, and the pattern is the useful part:

- **At the loosest threshold the veto is essential.** CE alone is worse than not switching at all in 2 of the 10 cells, and the veto is worth up to 3.65 C4 there (Qwen3-4B, k = 2). This is KL working as the safety net the rule claims.
- **At strict thresholds it costs.** In 3 cells the veto is harmful on both corpora, by as much as +2.86 WikiText (Qwen3-4B, k = 6), where it discards 14,252 tiles CE had accepted correctly.

So KL is not selecting tiles; it is insuring against a threshold that is too loose. Where the threshold is already strict, its veto mostly destroys value. That is a narrower role than "a switch is kept only when it improves the actual task loss **and** moves the quantized model back toward its own unquantized reference" suggests, and the accuracy table above is what keeps it from being an argument for dropping KL at k = 3: on Qwen3-4B, CE alone there costs 0.0161 accuracy at p = 1e-11 while electing 15.9 times as many tiles.

#### Does the budget-matched result hold on accuracy?

The perplexity table above compares the two objectives at an equal budget of switches, because at a fixed k they elect very different numbers of tiles. The same comparison on the multiple-choice panel, with every delta measured against the FourOverSix base by paired McNemar over documents.

**Llama-3.1-8B**

| objective | k | tiles | panel mean | d vs base (paired) | p |
|---|---:|---:|---:|---:|---:|
| — | — | 0 | 0.6720 | — | — |
| max(CE, KL) | 6 | 145 | 0.6737 | +0.0013 | 0.467 |
| max(CE, KL) | 5 | 267 | 0.6767 | +0.0032 | 0.079 |
| max(CE, KL) | 4 | 541 | 0.6712 | +0.0000 | 1 |
| max(CE, KL) **(shipped)** | 3 | 3,345 | 0.6714 | +0.0010 | 0.617 |
| KL only | 6 | 1,653 | 0.6701 | +0.0003 | 0.906 |
| KL only | 5 | 2,950 | 0.6706 | -0.0002 | 0.93 |
| CE only | 3 | 22,906 | 0.6677 | -0.0003 | 0.905 |
| KL only | 3 | 32,774 | 0.6601 | -0.0120 | 9.74e-08 |

Head to head against the best conjunction setting that elects no more tiles:

- `k6_kl` (1,653 tiles) against `k5` (267): -0.0029, p = 0.112 -- indistinguishable
- `k5_kl` (2,950 tiles) against `k5` (267): -0.0034, p = 0.0721 -- indistinguishable
- `k3_ce` (22,906 tiles) against `k5` (267): -0.0034, p = 0.0622 -- indistinguishable
- `k3_kl` (32,774 tiles) against `k5` (267): -0.0152, p = 8.57e-12 -- significantly different

**Qwen3-4B**

| objective | k | tiles | panel mean | d vs base (paired) | p |
|---|---:|---:|---:|---:|---:|
| — | — | 0 | 0.6231 | — | — |
| max(CE, KL) | 6 | 222 | 0.6252 | +0.0025 | 0.235 |
| max(CE, KL) | 5 | 576 | 0.6266 | +0.0060 | 0.00412 |
| max(CE, KL) | 4 | 1,837 | 0.6222 | +0.0061 | 0.00457 |
| max(CE, KL) **(shipped)** | 3 | 7,912 | 0.6385 | +0.0133 | 6.76e-10 |
| KL only | 5 | 1,226 | 0.6287 | +0.0067 | 0.00148 |
| KL only | 4 | 3,569 | 0.6345 | +0.0105 | 1.07e-06 |
| KL only | 3 | 21,528 | 0.6412 | +0.0161 | 9.03e-14 |
| CE only | 3 | 125,611 | 0.6223 | -0.0028 | 0.284 |

Head to head against the best conjunction setting that elects no more tiles:

- `k5_kl` (1,226 tiles) against `k5` (576): +0.0007, p = 0.754 -- indistinguishable
- `k4_kl` (3,569 tiles) against `k4` (1,837): +0.0044, p = 0.0335 -- significantly different
- `k3_kl` (21,528 tiles) against `k3` (7,912): +0.0028, p = 0.157 -- indistinguishable
- `k3_ce` (125,611 tiles) against `k3` (7,912): -0.0161, p = 1.14e-11 -- significantly different

Each frontier is pooled across jobs but confined to one GPU model (Llama-3.1-8B on H100 (2 point(s) on other hardware dropped); Qwen3-4B on H100 (2 point(s) on other hardware dropped)), because §1a's control found the evaluation exact within a GPU model and 2.8% of documents flipped across two. The 12 policies measured twice on that hardware agree on every document, so the pooling is exact rather than assumed.

#### Reading the two together

Neither metric favours KL alone on any model. Where they differ it is in how sharply they say so, not in which way, so both are stated per model.

- **Llama-3.1-8B.** Against the shipped rule at the same k, KL alone costs +0.507 WikiText and +0.621 C4 and spans -0.0130 to -0.0007 on accuracy across 3 threshold(s), none of them significantly better and 1 significantly worse.
- **Qwen3-4B.** Against the shipped rule at the same k, KL alone costs +0.144 WikiText and +0.368 C4 and spans -0.0065 to +0.0028 on accuracy across 3 threshold(s), none of them significantly better and 1 significantly worse.

Perplexity is the metric the election is calibrated on -- the score is a teacher-forced loss -- so it is the one KL alone should do well on if the objective were sufficient, and it is the one where it does not. The accuracy panel resolves about 0.005 at best (§1a), so a null there is a weaker statement than a perplexity regression of the size seen above. No model shows accuracy favouring KL alone by a significant margin, so nothing in the accuracy numbers offsets the perplexity cost.

**Coverage.** Measured on Llama-3.1-8B, Qwen3-4B. Not run on Qwen3.8-27B, so the conclusion is a two-model result, not a panel-wide one.

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
  quantization is measured. Native permutation overhead is measured separately
  in the 256×64 follow-up; no end-to-end model speedup is claimed. Zero-shot multiple-choice accuracy is in §1a, and one generative
  metric -- 8-shot chain-of-thought gsm8k -- is reported there for
  Llama-3.1-8B only, so generation accuracy is measured on one model of three
  rather than not at all.
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
- The CE-and-KL ablation in §3 covers Llama-3.1-8B and Qwen3-4B, not
  Qwen3.8-27B, so it is a two-model result. It also shows the conjunction is not
  the best selector at a given budget of switches on either model -- CE alone
  reaches a lower perplexity at several budgets -- and the case for it rests on
  its being the arm that is never badly wrong, with which single objective fails
  differing by model. Related: k = 3 is not the per-model optimum. On Qwen3-4B
  k = 2 is far better (WikiText 10.85 against 11.86); it is on Llama-3.1-8B that
  k = 2 is worse than not switching at all, which is what makes k = 3 the value
  that works on both rather than the best value for either.
- Agentic evaluation was attempted and is not reported. Terminal-Bench 4.0 and
  2.0 were run on Qwen3.8-27B across the three policies; the runs are recorded
  in `results/terminal_bench/TBENCH_STATUS.md` with their failure modes. The
  binding constraint is throughput: the in-process server exists because neither
  vLLM nor SGLang can run the simulated activation hooks, and at roughly 2
  tokens per second the agent timeout truncates most trials, so the pass rates
  obtained are lower bounds that separate nothing at this sample size. No
  agentic claim is made either way.
- Gradient selection, distillation and sparse optimization are established tools.
  Their use here is not by itself a novelty claim.
- Section 4 describes where one realized map lands, from one calibration draw per
  model. The uniform-placement reference there is a descriptive baseline for how
  clustered the positions are, not a significance test, and nothing in that
  section is evidence that a position-based rule would work without the scores.

## Arranging weights for 256×64 MixFP4 tiles

The Blackwell follow-up keeps the two weight codebooks and the joint CE/KL
`k=3` rule, but makes each format decision cover **256 rows × 64 columns**.
A large tile often mixes weights that benefit from E0M3 with weights that are
hurt by it. Arranging groups compatible weights into the same legal tile.
The accepted Qwen result uses permutations only; it uses no rotation.

### How the arranging algorithm works

1. **Score small pieces using the task loss.** Starting from FourOverSix,
   compute the weight-gradient inner product with the E0M3-minus-E2M1 weight
   change. Store separate CE and teacher-KL scores for each **1-row × 16-column**
   atom on each of 128 original OpenWebMath/CodeParrot calibration documents.
   This is a first-order loss prediction, not an MSE ranking. Keeping scores
   per document preserves covariance when many atoms are summed into a tile.
2. **Fit a legal arrangement.** Use 64 source-balanced documents to group rows
   into sets of 256 and intact 16-column scale groups into sets of four.
   Alternate row and column assignments, using capacity-constrained linearized
   assignments and exact pair swaps. A smooth objective transitions toward the
   hard joint confidence-bound objective. The fixed search uses four starts,
   six rounds, 4,096 sampled swaps, two swap passes, seed 0, and 32 MiB scratch.
   Columns never split a 16-element quantization scale group.
3. **Elect formats independently.** Freeze the arrangement, then use the other
   64 source-balanced documents to recompute each tile's summed CE and KL
   scores. Choose E0M3 only when **both `mean + 3 SE < 0`**. Otherwise retain
   FourOverSix E2M1. Election data do not choose the permutation.
4. **Keep the scope small.** The accepted experiment arranges only the final
   MLP's gate, up, and down projections (Qwen layer 63). All other matrices keep
   their raw 256×64 map. The elected final-MLP counts are 0, 6, and 14 tiles,
   respectively; 192 tiles elsewhere give **212 E0M3 tiles total**.
5. **Compact without changing the selected weights.** Reassign active row
   groups and column bands to positions that retain as many original indices
   as possible; fill unused positions with identities. This step uses no loss
   data. It preserves the original-coordinate format map and was verified to
   preserve the actual quantized weights **bitwise**. The zero-tile gate
   projection becomes identity.

For an output-by-input weight matrix, write the arranged matrix as
`W′ = P W Qᵀ`. Its input is `X′ = X Qᵀ`, so
`X′ W′ᵀ = X Wᵀ Pᵀ`; restoring output order cancels `P`.
Weights can be arranged offline. Runtime work is the matching input permutation
and output restoration, potentially fused into activation production and the
GEMM epilogue. Whole 16-element codes/scale groups must move together, preserving
the tensor-wide activation quantizer decision. Independent gate/up permutations
also require consistent ordering before their elementwise MLP product.

### Measured Qwen3.8-27B results

All PPL rows below use the same released 2,048-token protocol as the main report.
Lower is better. Deltas are relative to FourOverSix.

| Policy | E0M3 tiles | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 7.287076 | — | 10.188365 | — |
| Published MixFP4 8×64, k=3 | 3,785 | 7.214750 | −0.072327 | 10.149866 | −0.038499 |
| Raw 256×64, supplied reference | 198 | 7.275704 | −0.011373 | 10.177685 | −0.010680 |
| Raw 256×64, local reconstruction | 195 | 7.266300 | −0.020776 | 10.176030 | −0.012335 |
| Earlier row-only arrangement | 207 | 7.259481 | −0.027595 | 10.169099 | −0.019266 |
| **Accepted both-axis arrangement** | **212** | **7.255834** | **−0.031243** | **10.167336** | **−0.021029** |
| Independently confirmed compacted row candidate | 201 | 7.262465 | −0.024611 | 10.172032 | −0.016333 |

The accepted both-axis model beats the earlier row-only result on both datasets.
It recovers **43.2% of the 8×64 WikiText improvement and 54.6% of the C4
improvement** over FourOverSix. It does **not** reach the former 90% target;
the user accepted this endpoint. The supplied raw-198 and reconstructed raw-195
maps differ and are retained as separate controls.

The both-axis model has measured held-out PPL gains, but did not pass the later
strict fresh joint CE/KL confirmation procedure. The independently confirmed
row candidate is a separate model. Later extensions and rotations did not produce
a new validated PPL leader; in particular, the 218-tile Fisher extension failed
its frozen fresh CE-primary gate and was not evaluated for PPL. These failures
are not relabeled as successes.

### Exact compaction and deployment cost

| Final-MLP projection | Rows moved before → after | Columns moved before → after |
|---|---:|---:|
| Gate | 17,408 → 0 | 4,768 → 0 |
| Up | 17,407 → 1,921 | 5,072 → 288 |
| Down | 5,119 → 3,080 | 17,376 → 1,104 |
| **Total** | **39,934 → 5,001** | **27,216 → 1,392** |

Compaction reduces moved rows by 87.5% and moved columns by 94.9%, with identical
quantized weights and therefore reused parent PPL. These percentages measure
index movement, **not latency savings**. The user has verified that the raw
256×64 MixFP4 GEMM matches NVFP4 throughput; the added permutation cost is a
separate measurement. Llama transfer and native GB200 overhead use the frozen
[follow-up plan](results/task_reorder/transfer_20260920/plan.json).
The measured native costs are reported below; fused overhead remains unmeasured.

### Llama-3.1-8B transfer: fresh confirmation failed

We transferred the accepted search settings without tuning: last MLP (layer 31),
1×16 atoms, both axes, four starts, six rounds, and independent `k=3` election.
The regenerated calibration reproduced all historical adaptive/fixed maps exactly
and recovered the expected counts of 187 raw 256×64 and 3,345 fine 8×64 tiles.
The new arrangement elected **4 gate, 7 up, and 39 down tiles**, giving **178 total**
with the 128 unchanged tiles elsewhere. Compaction preserved all three actual
quantized matrices bitwise.

Before any PPL promotion, one frozen candidate was checked on 64 new, balanced
math/code documents (512 tokens each), excluding both models' calibration data,
six earlier Qwen fresh-data manifests, and published C4 evaluation documents.
The prospective gate required pooled CE `mean + 2 SE < 0` against both raw256
and matched identity, plus nonpositive math/code mean CE changes versus raw256.
KL was diagnostic. Matched identity used the same 64-document tile-election
split, yielding 161 total tiles, and therefore controls for the split as well
as the unchanged background.

| Fresh comparison | Documents | ΔCE | 2 SE | ΔCE + 2 SE | ΔKL |
|---|---:|---:|---:|---:|---:|
| Arranged − raw256, pooled | 64 | +0.005206 | 0.001771 | +0.006978 | +0.004720 |
| Arranged − matched identity, pooled | 64 | +0.005148 | 0.001795 | +0.006942 | +0.004581 |
| Arranged − raw256, math | 32 | +0.004309 | 0.002867 | +0.007176 | +0.003467 |
| Arranged − raw256, code | 32 | +0.006104 | 0.002079 | +0.008183 | +0.005972 |

**This transfer failed:** loss worsened on both sources, and also against matched
identity. Thus the failure cannot be explained solely by changing from 128 to
64 election documents. Successful Qwen arranging did not establish a model-general
method. No Llama arranging PPL gain is claimed; the failed candidate was **not
promoted to full WikiText/C4 PPL evaluation**, preserving the frozen gate and GPU budget.

| Llama policy | E0M3 tiles | WikiText-2 PPL | ΔWiki vs FourOverSix | C4 PPL | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|---:|
| FourOverSix, published | 0 | 6.875525 | — | 9.823733 | — |
| MixFP4 8×64, k=3, published | 3,345 | 6.849275 | −0.026249 | 9.773040 | −0.050694 |
| Raw MixFP4 256×64, supplied reference | 187 | 6.866879 | −0.008645 | 9.801361 | −0.022372 |
| Both-axis 256×64 transfer | 178 | Not run: fresh gate failed | — | Not run: fresh gate failed | — |

The first three PPL rows reuse existing results. Matching regenerated tile counts
alone is not a bitwise audit of the supplied raw-256 mask, whose full artifact was
not provided. The new measurement here is the controlled fresh-loss comparison,
not another raw-PPL run. Sources: [confirmation report](results/task_reorder/transfer_20260920/llama_confirmation/report.json),
[frozen plan](results/task_reorder/transfer_20260920/llama_confirmation/plan.json), and
[fresh document manifest](results/task_reorder/transfer_20260920/llama_confirmation/fresh_manifest.json).

The PPL quality path restores quantized weights to original coordinates. It does
not measure rounding changes from native GEMM accumulation in permuted K order.

### Native GB200 permutation overhead

Jobs 405441 and 405458 ran on **NVIDIA GB200, SM100, driver 580.105.08**,
using the existing `../mixfp4/src/mixed_nvfp4_gemm_sm100.cu` kernel and the exact
compacted Qwen up/down permutations. Gate needs no permutation. We measured:

- A full out-of-place gather of packed FP4 activation codes and their existing
  16-element scales, followed by GEMM and full BF16 output restoration.
- The same input gather and GEMM, with two-pass **in-place restoration of only
  moved output channels**: first read all moved values into scratch, then write
  their destinations. Separating reads and writes prevents permutation-cycle
  races. Fixed output channels remain where GEMM wrote them.

The table reports graph-replay medians of five repetitions, 100 iterations each,
with baseline/pipeline order alternating. Setup, allocation, graph construction,
weight preparation, and activation quantization are excluded. Timings are per
projection, in microseconds; these are not full-model latencies.

| Projection | Tokens | GEMM only µs | Full input/output permutation + GEMM µs | Full overhead | Sparse-output pipeline µs | Sparse overhead |
|---|---:|---:|---:|---:|---:|---:|
| down | 1 | 21.15 | 27.57 | +30.3% | 29.47 | +39.5% |
| down | 16 | 20.81 | 26.49 | +27.3% | 28.28 | +36.0% |
| down | 128 | 21.11 | 28.19 | +33.5% | 30.03 | +42.2% |
| down | 512 | 22.45 | 36.65 | +63.3% | 38.56 | +73.2% |
| down | 2048 | 63.16 | 111.27 | +76.2% | 115.47 | +82.8% |
| up | 1 | 14.23 | 19.62 | +37.9% | 21.36 | +50.2% |
| up | 16 | 11.83 | 18.40 | +55.6% | 19.63 | +65.8% |
| up | 128 | 12.18 | 23.20 | +90.4% | 20.41 | +67.4% |
| up | 512 | 15.30 | 46.48 | +203.7% | 28.80 | +88.4% |
| up | 2048 | 57.33 | 174.27 | +204.0% | 92.17 | +60.9% |

Each overhead uses its own job's matched GEMM baseline; the displayed GEMM column
is from the full-permutation job. Sparse restoration helps the up projection at
larger token counts, reducing its 2,048-token pipeline from 174.27 to 92.17 µs.
It does not help the down projection: 3,080 of 5,120 output channels still move,
and two launches outweigh the saved traffic. At small token counts, launch cost
also makes sparse restoration slower. Use full restoration for down, and choose
up restoration according to workload; those initial jobs did not implement fusion; the follow-up below does.

**The added permutations are not free.** Compaction preserves quality but does
not by itself remove memory passes or launches. The measured per-projection
cost cannot be extrapolated to a whole-model slowdown: only the last MLP is
arranged, and no native end-to-end model benchmark was run.

Correctness checks used nonconstant FP4 codes/scales and verified their gather
byte-for-byte; both output restoration variants passed bitwise checks. A separate
[eight-second correctness job](results/task_reorder/transfer_20260920/gb200_check_405476/checks.txt)
used distinct output bit patterns at all 20 shape/variant combinations. Its
unchanged-output negative control failed for exactly every moved channel. This
strengthens the initial synthetic-GEMM check, whose output columns can coincide. The
existing GEMM also passed its small host-reference check. This overhead harness
uses the mixed kernel's fixed E2M1 format path, isolating permutation cost; it
**does not validate execution of the complete 212-tile mixed-format model**.
It adds no activation quantization or rotation. Those initial jobs make no fused-kernel latency claim.
Ordinary-launch timings, individual pass timings, all repetitions, and provenance
are retained in the [full-gather results](results/task_reorder/transfer_20260920/gb200_405441/SUMMARY.md)
and [sparse-output results](results/task_reorder/transfer_20260920/gb200_405458/SUMMARY.md).
The two native jobs used one GPU each for about 2 minutes per job.

Implementation: `quantize/task_reorder.py`, `run_task_reorder.py`,
`quantize/compact_rows.py`, and `quantize/compact_both.py`.
See [study status](REORDERING_STUDY_STATUS.md), the
[full comparison](results/task_reorder/cluster_20260919/FULL_COMPARISON.md), and
[measurement snapshots](results/task_reorder/cluster_20260919/published_evidence/INDEX.json)
for provenance and the unsuccessful follow-ups.

### Continued Llama research: independent gate passed, both PPLs improved

The initial Llama transfer failure above remains a failure. The renewed study
found that most of its degradation came from `down_proj`, and that summing
single-tile improvements was unreliable: tile effects interacted in the actual
quantized model. A gate/up-only candidate and an eight-down-tile refinement both
improved mean fresh CE but failed their frozen confidence gates. Neither was
sent to perplexity evaluation.

The successful refinement keeps the learned row/column arrangements and searches
**subsets of the already joint-CE/KL-k3-elected tiles**. It changes actual format
masks, not the original codebooks, quantization scales, or legal 256×64 geometry.
It uses no rotation or correction GEMM.

1. Cache the final MLP input and residual from the raw quantized model. Replay
   the final MLP, final norm, and vocabulary head for each trial. Verify cached
   losses bit-for-bit against full-model losses before searching.
2. Use all **192 previously observed documents as development data**, including
   the rejected confirmation sets. These documents are explicitly no longer
   independent validation. Preserve each prior failure.
3. Evaluate each proposed tile flip with the actual joint quantized-model CE,
   retaining interactions between gate, up, and down. Search at most four flips
   among the 50 originally elected tiles. Minimize the worst of pooled CE+2SE,
   each-domain CE+1SE, and each observed 64-document set's CE mean versus raw.
4. The four changes remove one gate tile and one up tile and add two down tiles.
   The final MLP has **3 gate + 6 up + 10 down = 19 E0M3 tiles**. The unchanged
   raw background contributes 128, giving **147 total**. The 192-document
   development CE change is −0.001490 versus raw.
5. Freeze this new map before drawing **64 new documents**, excluding all prior
   calibration, confirmation, and published C4 documents. Require pooled CE+2SE
   below zero versus both raw and matched identity, and nonpositive math/code
   CE means versus raw. Only after that gate passes, evaluate PPL on the exact
   published WikiText/C4 token windows.

The independent confirmation (job 405692) passed:

| Comparison | Mean ΔCE | SE | Mean + 2SE |
|---|---:|---:|---:|
| Candidate − raw 256×64 | -0.001675449 | 0.000447776 | -0.000779897 |
| Candidate − matched identity | -0.001712369 | 0.000417071 | -0.000878227 |

Math/code mean ΔCE versus raw are −0.002173/−0.001178. KL also improves,
but remains diagnostic under this prospective CE-primary protocol. No failed
candidate was promoted or retested unchanged on another fresh draw.

| Llama-3.1-8B policy | E0M3 tiles | WikiText PPL | Δ vs FourOverSix | C4 PPL | Δ vs FourOverSix |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 6.875525 | +0.000000 | 9.823733 | +0.000000 |
| Published MixFP4 8×64 | 3,345 | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| Supplied raw MixFP4 256×64 | 187 | 6.866879 | -0.008646 | 9.801361 | -0.022372 |
| **Refined both-axis 256×64** | 147 | 6.864886 | -0.010638 | 9.796946 | -0.026788 |

The new result improves on supplied raw 256×64 by **-0.001993 WikiText**
and **-0.004415 C4**, recovering **40.5%/52.8%** of the published 8×64
gain over FourOverSix. The original controls were reused; the raw deltas use
the user-supplied rounded values. Job 405707 verified identical published token
windows and source weights. This is a modest, independently confirmed quality
gain; it does not meet the historical 90% target.

[Fresh confirmation](results/task_reorder/transfer_20260920/renewed_llama/joint192_confirm/report.json),
[PPL report](results/task_reorder/transfer_20260920/renewed_llama/joint192_ppl/report.json),
and [joint search](results/task_reorder/transfer_20260920/renewed_llama/joint192/report.json)
preserve the evidence, including the preceding rejected candidates.

### GB200: remove standalone permutations through producer/consumer fusion

The successful native approach preserves the existing SM100 GEMM mainloop and
TMA epilogue. A direct scatter epilogue was bitwise correct but much slower
(job 405618), so it is not the recommended implementation.

**Columns:** the FourOverSix activation quantizer stores each original 16-value
group's packed 64-bit code word and E4M3 scale byte directly at its permuted
destination. The scale byte uses the actual CUTLASS SM100 SFA layout. Whole-group
permutation preserves tensor-wide absolute maximum and every within-group
calculation. The final pipeline reapplies the common global activation scale
through GEMM's alpha; the common maximum computation is outside timing.

**Rows:** keep GEMM's fast output order and make the next consumer read the
inverse permutation. For up, SiLU/multiply reads the matching gate and up
channels and emits down's column order. For down, residual-add reads the inverse
row map. The vector version processes two BF16 values per thread, using paired
loads when consecutive and scalar loads where needed. These operations replace
standalone restoration passes rather than adding more kernels. In a connected
MLP, apply down's column permutation exactly once: either the SiLU consumer
emits that order or the down quantizer applies it. These independent projection
prototypes are not an end-to-end MLP implementation.

The native quantizer's first independent Python comparison found a signed-zero
mismatch. After preserving negative zero, **all 22,528 tested BF16 values match
`quant_nvfp4_4over6` bit-for-bit**. Separate checks compare fused/unfused packed
codes and scale bytes. Every measured pipeline shape passes bitwise output
comparison and a deliberately wrong no-permutation negative control.

The final table (job 405810) times **FourOverSix producer + GEMM + consumer**,
using the exact compacted **Qwen** maps. All three variants use the same
quantizer, GEMM, and BF16 arithmetic. Values are microseconds, medians of five
alternating baseline/fused repetitions with 100 CUDA-graph iterations each.

| Projection | Tokens | No permutation µs | Separate passes µs | Fully fused µs | Added cost µs | Overhead |
|---|---:|---:|---:|---:|---:|---:|
| up | 1 | 20.78 | 26.35 | 21.16 | +0.38 | +1.8% |
| up | 128 | 28.49 | 43.49 | 29.63 | +1.14 | +4.0% |
| up | 512 | 69.93 | 111.54 | 72.99 | +3.06 | +4.4% |
| up | 2,048 | 240.62 | 417.37 | 253.00 | +12.38 | +5.1% |
| down | 1 | 28.97 | 32.62 | 29.71 | +0.75 | +2.6% |
| down | 128 | 48.33 | 54.76 | 49.24 | +0.91 | +1.9% |
| down | 512 | 117.14 | 132.00 | 118.90 | +1.76 | +1.5% |
| down | 2,048 | 421.93 | 465.68 | 427.89 | +5.96 | +1.4% |

This is a per-projection prototype, **not a full-model speedup claim**. It uses
synthetic weights and the mixed kernel's fixed E2M1 path to isolate permutation
cost; it does not execute the complete 212-tile mixed-format model or benchmark
Llama's native latency. Its FourOverSix producer is not claimed to be an optimal
quantizer, so its absolute cost affects percentage overhead. Common tensor-amax,
setup, weight preparation, and graph construction are excluded. Do not compare
these percentages directly with the earlier GEMM-only baseline.

The intermediate GEMM+consumer benchmark, which still includes input gathering,
reduces up's 2,048-token pipeline from 319.50 to 159.79 µs (142.21 µs baseline).
Producer fusion then removes that remaining gather. The isolated producer
experiment adds only about 0.02–0.03 µs at one token; small negative differences
at other shapes are timing variation, not a claimed quantization speedup.

[Final pipeline measurements](results/task_reorder/transfer_20260920/full_pipeline_405810/summary.json),
[independent quantizer check](results/task_reorder/transfer_20260920/quant_audit_405697/python_reference.json),
and [implementation notes](native/PERMUTATION_FUSION.md) document scope and code.
The renewed [job ledger](results/task_reorder/transfer_20260920/renewed_job_ledger.json)
includes failed builds and rejected candidates. All work used `gov113008`, with
Slurm workers for heavy compute and attached completion monitors; the measured
peak concurrency remained below the four-GPU limit.
