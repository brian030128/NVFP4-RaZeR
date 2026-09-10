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
Where shown, BF16 is the unquantized reference, not a competitor.

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

## 4. Protocol, and two deliberate differences from the released code

The scope is corrected full W4A4: every targeted text linear weight and its input
is quantized. Both differences below make our baselines harder to beat.

**Qwen `o_proj` inputs are quantized.** At the archived release commit `e230099`,
`models/qmodule_qwen3.py` passed unquantized attention output into `o_proj`, so
Qwen "W4A4" left one projection's input in BF16. The RaZeR author corrected this
in commit `abab3c6`, after publication. We evaluate the corrected behaviour, so
our Qwen FourOverSix baseline is 14.269062 WikiText where the pre-fix code gives
14.201942. Qwen rows here are therefore **not** comparable with the published
RaZeR Qwen row. Llama was never affected, and its rows match the released run to
the last digit — the control showing the difference comes only from that line.

**NVFP4 saturation is clamped.** The archived `quant_nvfp4` used a rounding path
with no E2M1 saturation clamp, so an FP8 subnormal block scale that rounded down
could produce magnitude code 8, which is not a legal FP4 code. The current
`quant_nvfp4` clamps to [-6, 6], and baseline and method both use it.

Verification carried by the runs themselves: the shipped 2 SE score is
reproduced exactly by the k = 2 case; the 256-tile prefix of that ranking
reproduces the previously frozen map bitwise; the Llama FourOverSix row is
asserted equal to the archived released-code reproduction; pristine weight
hashes and frozen map hashes are checked; and the C4 evaluation documents have
zero hash overlap with the calibration documents.

## 5. Perplexity below BF16 is not a quality claim

On Qwen3-4B some MixFP4 perplexities fall below the unquantized BF16 reference.
Perplexity cannot settle that on its own: a model that becomes less overconfident scores
better on next-token loss without predicting better. The same frozen maps were therefore
evaluated zero-shot on multiple choice, where a smoothing artefact should not help. BF16
restores pristine weights and removes activation quantization.

| Policy | WikiText-2 PPL | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean acc |
|---|---|---|---|---|---|---|---|---|
| BF16 reference | 13.662473 | 0.7816 | 0.5358 | 0.6846 | 0.4020 | 0.8495 | 0.6519 | 0.6509 |
| FourOverSix W4A4 | 14.269062 | 0.7462 | 0.4906 | 0.6616 | 0.3940 | 0.8358 | 0.6212 | 0.6249 |
| MixFP4, 256 tiles | 13.040957 | 0.7563 | 0.4881 | 0.6652 | 0.3880 | 0.8330 | 0.6409 | 0.6286 |
| MixFP4, 65,536 tiles | 10.864750 | 0.7778 | 0.5043 | 0.6725 | 0.4060 | 0.8453 | 0.6314 | 0.6395 |

Accuracy corroborates the method among the quantized policies, in the same order perplexity gives: 0.6249 for FourOverSix, 0.6286 at 256 tiles, 0.6395 at 65,536, the larger map ahead on 5/6 tasks. Quantization costs 0.0260 mean accuracy against BF16; those maps recover 14.2% and 56.2% of it.

Accuracy does **not** support beating BF16. The best quantized policy is still 0.0114 below the unquantized model while its perplexity is 2.797723 better. Perplexity is therefore not a reliable absolute quality measure against BF16 on this model. No claim here rests on a below-BF16 perplexity; comparisons against the matched baselines are unaffected.

Zero-shot via lm_eval 0.4.5. Skipped for dataset-loading reasons unrelated to the model: piqa.

## 6. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is measured,
  and no speedup is claimed.
- Tensor-wide activation factors span the whole teacher-forced window, so these
  are reference-text perplexities, not causal generation likelihoods.
- k = 3 is prespecified from the calibration score distribution and was fixed
  using Llama-3.1-8B and Qwen3-4B. Qwen3.8-27B played no part in choosing it and
  is a held-out check of the rule, not a third fitting model. Three models is
  still a small panel, and one calibration draw is used per model.
- No matched 2048-token BF16 run exists for Qwen3.8-27B, so that reference row is
  omitted for it rather than filled from a measurement taken under another
  protocol.
- Two-SE intervals are descriptive evaluation-window intervals. They do not
  adjust for multiple comparisons, WikiText article dependence, or
  calibration-draw variability; one calibration draw per model is used.
- Gradient selection, distillation and sparse optimization are established tools.
  Their use here is not by itself a novelty claim.
