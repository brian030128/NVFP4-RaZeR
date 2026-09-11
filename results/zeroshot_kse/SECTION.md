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

### Is the difference real?

Every policy is scored on the same documents, so MixFP4 against its own base is a paired comparison and the standard error lm-eval prints -- the error of one measurement -- is the wrong yardstick. `b` counts documents only FourOverSix gets right, `c` only MixFP4; the rest carry no information about the difference. The p-value is an exact two-sided McNemar test on those counts, pooled over all tasks.

| model | documents | b | c | pooled delta | McNemar p |
|---|---|---|---|---|---|
| Llama-3.1-8B | 18627 | 570 | 588 | +0.0010 | 0.617 |
| Qwen3-4B | 18627 | 683 | 935 | +0.0135 | 4.04e-10 |

### Coverage

This section covers Llama-3.1-8B, Qwen3-4B. It does not cover Qwen3.8-27B. Qwen3.8-27B does not reproduce its shipped election in this environment: the rule elects 3,787 tiles where the report records 3,785, and the 256-tile cross-check differs, while the calibration reproduces all 128 teacher losses bit for bit and matches weight_mse_sha256. That is threshold sensitivity in a 24B scoring pass, not a different calibration, but it means an accuracy number measured here would be the k-SE rule re-derived rather than the shipped artifact. No claim is made about the missing model either way.

