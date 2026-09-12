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

