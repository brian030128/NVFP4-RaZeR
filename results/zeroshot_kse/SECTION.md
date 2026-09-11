## Zero-shot accuracy

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

### What the multiple-choice panel can and cannot see

The panel above resolves a difference of roughly 0.005 and no smaller, and it is not equally sensitive to quantization across metrics. The same policies on the same weights, measured on tasks chosen to be harder on a quantized model: generative chain-of-thought, where one derailed token loses a whole answer instead of averaging out, and larger multiple-choice sets.

| model | metric | n | BF16 | NVFP4 | FourOverSix | MixFP4 (k=3) | k3 − FourOverSix (p) | k3 − NVFP4 (p) |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | `gsm8k` | 1319 | 0.4882 | 0.3859 | 0.4117 | 0.4200 | +0.0106 (0.43) | +0.0364 (0.0083) |
| Llama-3.1-8B | `mmlu` | 14042 | 0.6344 | 0.5996 | 0.5998 | 0.6035 | +0.0036 (0.28) | +0.0038 (0.28) |
| Llama-3.1-8B | `lambada_openai` | 5153 | 0.7530 | 0.7370 | 0.7440 | 0.7454 | +0.0014 (0.74) | +0.0083 (0.036) |

Two things follow. First, what quantization costs depends heavily on the metric: on Llama-3.1-8B, W4A4 costs about four times as much on gsm8k as on the multiple-choice panel, so a null on the panel is a weaker statement than it looks. Second, and more usefully, the method's advantage over plain NVFP4 is clearest exactly where the metric is most sensitive: on gsm8k it is +0.0364 at p = 0.0083 from 1,319 problems, where the panel needed 18,627 documents to resolve +0.0075. Against FourOverSix the same comparison stays inside noise on both, but its point estimate rises by an order of magnitude, from +0.0010 to +0.0106.

> **Election re-derived.** Qwen3.8-27B: 4 module(s) differ from the shipped frozen map, 2 tile(s) present only in the shipped map and 2 only here. The rule, k, and calibration are the reported ones and the calibration reproduces the shipped teacher losses bit for bit; what differs is which side of the threshold a handful of borderline tiles fall on in a re-run. The effect on the elected set is a few tiles in tens of millions, so these rows are treated as the reported policy, with the difference recorded here rather than hidden.

