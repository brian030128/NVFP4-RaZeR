## 1a. Zero-shot accuracy (current scope)

§1 is perplexity only. Perplexity is a next-token loss, and a quantizer that makes a model *less confident* lowers it without the model predicting anything better -- a real possibility here, since the election rule is chosen to reduce a squared error rather than to preserve a decision. So the identical seven configurations were re-run on zero-shot multiple choice, where a smoothing artefact earns nothing.

Same weights, same code path: W4A4 prefill, weights 8x64 / alpha = 1, activations `nvfp4_4over6` at 16x64, importance from 4 wikitext-train windows on the unquantized model (`run_zeroshot_sweep.py`, which reuses `run_ppl_sweep.py`'s quantization loop unchanged). lm-eval-harness 0.4.5, 0-shot, 6 tasks: `arc_easy`, `arc_challenge`, `hellaswag`, `openbookqa`, `boolq`, `winogrande`. `acc_norm` where the harness reports it, `acc` otherwise; the panel figure is the unweighted mean.

### Panel mean accuracy, absolute

`bf16` is the unquantized model and `nvfp4` the quantized reference §1 measures against; the gap between them is what W4A4 costs before any element-type decision is made.

| model | bf16 | `nvfp4` | `e2m1` | `h10` | `hess_h1.5` | `hess_h10` | `hess_m1` | `hess_impg16_h10` | 1 SE |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.6894 | 0.6654 | 0.6654 | 0.6679 | 0.6751 | 0.6645 | 0.6713 | 0.6614 | 0.0053 |
| Llama-3.1-8B-Ins | 0.6909 | 0.6749 | 0.6749 | 0.6760 | 0.6792 | 0.6764 | 0.6789 | 0.6729 | 0.0053 |
| Llama-3.2-1B-Ins | 0.5432 | 0.5202 | 0.5202 | 0.5181 | 0.5186 | 0.5224 | 0.5206 | 0.5201 | 0.0054 |
| Qwen3-4B | 0.6509 | 0.6218 | 0.6218 | 0.6180 | 0.6236 | 0.6209 | 0.6300 | 0.6274 | 0.0053 |
| Qwen3-8B | 0.6813 | 0.6674 | 0.6674 | 0.6718 | 0.6695 | 0.6701 | 0.6726 | 0.6703 | 0.0053 |
| Qwen3-14B | 0.7173 | 0.7040 | 0.7040 | 0.7035 | 0.7069 | 0.7017 | 0.7056 | 0.7051 | 0.0052 |

### Panel mean accuracy, delta against `nvfp4`

Positive is better here -- the opposite sign convention from the perplexity tables, because this is an accuracy. `e2m1` is again the validation row: alpha frozen at 1 with the election disabled leaves MixFP4 no freedom, so it must reproduce `nvfp4` exactly.

| model | bf16 | `e2m1` | `h10` | `hess_h1.5` | `hess_h10` | `hess_m1` | `hess_impg16_h10` |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | +0.0240 | +0.0000 | +0.0025 | +0.0097 | -0.0009 | +0.0059 | -0.0040 |
| Llama-3.1-8B-Ins | +0.0160 | +0.0000 | +0.0011 | +0.0043 | +0.0015 | +0.0040 | -0.0020 |
| Llama-3.2-1B-Ins | +0.0230 | +0.0000 | -0.0021 | -0.0016 | +0.0022 | +0.0004 | -0.0001 |
| Qwen3-4B | +0.0291 | +0.0000 | -0.0039 | +0.0017 | -0.0009 | +0.0082 | +0.0056 |
| Qwen3-8B | +0.0139 | +0.0000 | +0.0044 | +0.0020 | +0.0027 | +0.0052 | +0.0029 |
| Qwen3-14B | +0.0132 | +0.0000 | -0.0006 | +0.0029 | -0.0024 | +0.0015 | +0.0011 |

### Summary over the panel

| rule | mean delta | worst model | best model |
|---|---|---|---|
| `e2m1` | +0.0000 | Llama-3.1-8B +0.0000 | Qwen3-8B +0.0000 |
| `h10` | +0.0002 | Qwen3-4B -0.0039 | Qwen3-8B +0.0044 |
| `hess_h1.5` | +0.0032 | Llama-3.2-1B-Ins -0.0016 | Llama-3.1-8B +0.0097 |
| `hess_h10` | +0.0004 | Qwen3-14B -0.0024 | Qwen3-8B +0.0027 |
| `hess_m1` | +0.0042 | Llama-3.2-1B-Ins +0.0004 | Qwen3-4B +0.0082 |
| `hess_impg16_h10` | +0.0006 | Llama-3.1-8B -0.0040 | Qwen3-4B +0.0056 |

### The noise floor, measured

`nvfp4` uses no calibration, so its quantization is bit-deterministic and re-running it must give the same number. The paired jobs below re-ran it from scratch on every model, which turns that into a measurement of the evaluation's own reproducibility -- the floor any delta above has to clear.

| model | first run | re-run | difference |
|---|---|---|---|
| Llama-3.1-8B | 0.6654 | 0.6639 | -0.0016 |
| Llama-3.1-8B-Ins | 0.6749 | 0.6758 | +0.0010 |
| Llama-3.2-1B-Ins | 0.5202 | 0.5202 | +0.0000 |
| Qwen3-8B | 0.6674 | 0.6674 | +0.0000 |
| Qwen3-14B | 0.7040 | 0.7040 | +0.0000 |

The spread reaches **0.0016**, and it is a property of the model rather than of the run: Llama-3.2-1B-Ins, Qwen3-8B, Qwen3-14B reproduce exactly, while Llama-3.1-8B, Llama-3.1-8B-Ins do not. The likely mechanism is that the two runs reach this configuration in a different order, so allocator and cuBLAS state differ, and tiny logit differences flip multiple-choice items that were already near ties -- which also explains why the models that drift are the ones with the most near ties. Whatever the cause, a delta of 0.0016 on one of those models is not evidence of anything, and several deltas in the table above are that size. That is what the paired test below is for.

### Paired test against `nvfp4`

Both configurations are scored on the same documents, so the comparison is paired and the question is not how much each one varies but how often they disagree. `b` counts documents only the reference gets right, `c` documents only the variant gets right; everything else carries no information about the difference. The p-value is an exact two-sided McNemar test on those counts, pooled over all documents of all tasks (`analyze_zeroshot_paired.py`).

| model | rule | documents | b | c | pooled delta | McNemar p |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | `hess_h1.5` | 18627 | 663 | 769 | +0.0057 | 0.00551 |
| Llama-3.1-8B | `hess_impg16_h10` | 18627 | 573 | 554 | -0.0010 | 0.592 |
| Llama-3.1-8B-Ins | `hess_h1.5` | 18627 | 585 | 633 | +0.0026 | 0.178 |
| Llama-3.1-8B-Ins | `hess_impg16_h10` | 18627 | 523 | 472 | -0.0027 | 0.113 |
| Llama-3.2-1B-Ins | `hess_h1.5` | 18627 | 1013 | 1021 | +0.0004 | 0.877 |
| Llama-3.2-1B-Ins | `hess_impg16_h10` | 18627 | 866 | 843 | -0.0012 | 0.595 |
| Qwen3-8B | `hess_h1.5` | 18627 | 640 | 690 | +0.0027 | 0.179 |
| Qwen3-8B | `hess_impg16_h10` | 18627 | 545 | 563 | +0.0010 | 0.61 |
| Qwen3-14B | `hess_h1.5` | 18627 | 513 | 585 | +0.0039 | 0.0321 |
| Qwen3-14B | `hess_impg16_h10` | 18627 | 443 | 439 | -0.0002 | 0.92 |

10 tests were run, so the 0.05 threshold is worth 0.0050 after a Bonferroni correction. Uncorrected, 2 row(s) fall below 0.05: Llama-3.1-8B `hess_h1.5` (p = 0.00551), Qwen3-14B `hess_h1.5` (p = 0.0321). Corrected, 0 survive. Read the table accordingly: it is evidence about the SIZE of these effects, and the honest summary of that size is that it is small enough to need 18,600 documents to see at all.

<details>
<summary>Per-task accuracy</summary>

**Llama-3.1-8B**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.8123 | 0.5367 | 0.7884 | 0.4440 | 0.8196 | 0.7356 | 0.6894 |
| `nvfp4` | 0.7567 | 0.4940 | 0.7750 | 0.4420 | 0.8024 | 0.7222 | 0.6654 |
| `e2m1` | 0.7567 | 0.4940 | 0.7750 | 0.4420 | 0.8024 | 0.7222 | 0.6654 |
| `h10` | 0.7559 | 0.5034 | 0.7781 | 0.4380 | 0.7994 | 0.7324 | 0.6679 |
| `hess_h1.5` | 0.7887 | 0.5154 | 0.7751 | 0.4400 | 0.8052 | 0.7261 | 0.6751 |
| `hess_h10` | 0.7521 | 0.5034 | 0.7758 | 0.4540 | 0.7963 | 0.7056 | 0.6645 |
| `hess_m1` | 0.7862 | 0.5043 | 0.7785 | 0.4460 | 0.8113 | 0.7017 | 0.6713 |
| `hess_impg16_h10` | 0.7504 | 0.4991 | 0.7744 | 0.4360 | 0.7988 | 0.7096 | 0.6614 |

**Llama-3.1-8B-Ins**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.7976 | 0.5495 | 0.7921 | 0.4280 | 0.8410 | 0.7372 | 0.6909 |
| `nvfp4` | 0.7727 | 0.5247 | 0.7728 | 0.4220 | 0.8294 | 0.7277 | 0.6749 |
| `e2m1` | 0.7727 | 0.5247 | 0.7728 | 0.4220 | 0.8294 | 0.7277 | 0.6749 |
| `h10` | 0.7778 | 0.5188 | 0.7741 | 0.4260 | 0.8294 | 0.7301 | 0.6760 |
| `hess_h1.5` | 0.7795 | 0.5273 | 0.7769 | 0.4400 | 0.8278 | 0.7238 | 0.6792 |
| `hess_h10` | 0.7803 | 0.5247 | 0.7738 | 0.4160 | 0.8318 | 0.7316 | 0.6764 |
| `hess_m1` | 0.7820 | 0.5307 | 0.7775 | 0.4240 | 0.8284 | 0.7309 | 0.6789 |
| `hess_impg16_h10` | 0.7761 | 0.5094 | 0.7717 | 0.4260 | 0.8248 | 0.7293 | 0.6729 |

**Llama-3.2-1B-Ins**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.6338 | 0.3780 | 0.6082 | 0.3460 | 0.6917 | 0.6014 | 0.5432 |
| `nvfp4` | 0.5854 | 0.3720 | 0.5697 | 0.3340 | 0.6752 | 0.5848 | 0.5202 |
| `e2m1` | 0.5854 | 0.3720 | 0.5697 | 0.3340 | 0.6752 | 0.5848 | 0.5202 |
| `h10` | 0.5968 | 0.3584 | 0.5704 | 0.3320 | 0.6703 | 0.5809 | 0.5181 |
| `hess_h1.5` | 0.5985 | 0.3618 | 0.5724 | 0.3340 | 0.6657 | 0.5793 | 0.5186 |
| `hess_h10` | 0.5918 | 0.3584 | 0.5721 | 0.3520 | 0.6697 | 0.5904 | 0.5224 |
| `hess_m1` | 0.6002 | 0.3695 | 0.5735 | 0.3460 | 0.6691 | 0.5651 | 0.5206 |
| `hess_impg16_h10` | 0.5939 | 0.3584 | 0.5688 | 0.3440 | 0.6667 | 0.5888 | 0.5201 |

**Qwen3-4B**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.7816 | 0.5358 | 0.6846 | 0.4020 | 0.8495 | 0.6519 | 0.6509 |
| `nvfp4` | 0.7311 | 0.4957 | 0.6558 | 0.3900 | 0.8428 | 0.6156 | 0.6218 |
| `e2m1` | 0.7311 | 0.4957 | 0.6558 | 0.3900 | 0.8428 | 0.6156 | 0.6218 |
| `h10` | 0.7298 | 0.4829 | 0.6617 | 0.3760 | 0.8394 | 0.6180 | 0.6180 |
| `hess_h1.5` | 0.7407 | 0.4855 | 0.6619 | 0.3880 | 0.8401 | 0.6251 | 0.6236 |
| `hess_h10` | 0.7357 | 0.4855 | 0.6637 | 0.3820 | 0.8376 | 0.6212 | 0.6209 |
| `hess_m1` | 0.7521 | 0.4829 | 0.6623 | 0.4140 | 0.8443 | 0.6243 | 0.6300 |
| `hess_impg16_h10` | 0.7399 | 0.4915 | 0.6629 | 0.4080 | 0.8450 | 0.6172 | 0.6274 |

**Qwen3-8B**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.8093 | 0.5657 | 0.7489 | 0.4160 | 0.8661 | 0.6819 | 0.6813 |
| `nvfp4` | 0.7925 | 0.5461 | 0.7310 | 0.4080 | 0.8584 | 0.6685 | 0.6674 |
| `e2m1` | 0.7925 | 0.5461 | 0.7310 | 0.4080 | 0.8584 | 0.6685 | 0.6674 |
| `h10` | 0.7883 | 0.5478 | 0.7308 | 0.4200 | 0.8621 | 0.6819 | 0.6718 |
| `hess_h1.5` | 0.7963 | 0.5529 | 0.7349 | 0.4040 | 0.8554 | 0.6732 | 0.6695 |
| `hess_h10` | 0.7938 | 0.5418 | 0.7329 | 0.4160 | 0.8550 | 0.6811 | 0.6701 |
| `hess_m1` | 0.7908 | 0.5503 | 0.7325 | 0.4140 | 0.8560 | 0.6922 | 0.6726 |
| `hess_impg16_h10` | 0.7854 | 0.5478 | 0.7338 | 0.4220 | 0.8535 | 0.6796 | 0.6703 |

**Qwen3-14B**

| config | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.8291 | 0.6041 | 0.7883 | 0.4600 | 0.8930 | 0.7293 | 0.7173 |
| `nvfp4` | 0.8030 | 0.5811 | 0.7778 | 0.4540 | 0.8838 | 0.7245 | 0.7040 |
| `e2m1` | 0.8030 | 0.5811 | 0.7778 | 0.4540 | 0.8838 | 0.7245 | 0.7040 |
| `h10` | 0.8047 | 0.5742 | 0.7762 | 0.4640 | 0.8841 | 0.7174 | 0.7035 |
| `hess_h1.5` | 0.8144 | 0.5939 | 0.7811 | 0.4520 | 0.8881 | 0.7119 | 0.7069 |
| `hess_h10` | 0.8047 | 0.5828 | 0.7782 | 0.4440 | 0.8813 | 0.7190 | 0.7017 |
| `hess_m1` | 0.8102 | 0.5904 | 0.7814 | 0.4500 | 0.8872 | 0.7143 | 0.7056 |
| `hess_impg16_h10` | 0.8013 | 0.5768 | 0.7776 | 0.4640 | 0.8832 | 0.7277 | 0.7051 |

</details>

