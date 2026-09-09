# Paper-aligned W4A4 panel: NVFP4, FourOverSix and calibrated MixFP4

Every number uses the released 2048-token evaluation: WikiText-2 raw test in full
nonoverlapping windows (141) and 256 seed-0 C4 validation crops from shard 00000 (256
windows), tensor-wide activation factors, SDPA, WikiText cached per window with C4
uncached, and the released float32 perplexity aggregation. The scope is corrected full
W4A4: every targeted text linear weight and its input is quantized, including Qwen
`o_proj`, which the released code omitted until the RaZeR author's own fix `abab3c6`.

MixFP4 maps are 256 8x64 E0M3 type blocks over a FourOverSix E2M1 base, elected by the
CE/KL two-SE task-gradient rule on OpenWebMath and CodeParrot only. Neither WikiText-2
nor C4 supplies calibration data, gradients or any selection feedback.

## Llama-3.1-8B

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 (unquantized reference) | — | 6.240087 | 8.958212 |
| NVFP4 W4A4 | 0 | 6.940252 | 9.925099 |
| NVFP4 FourOverSix W4A4 | 0 | 6.875525 | 9.823733 |
| **MixFP4 ours (math_code128)** | 256 | **6.848383** | **9.764387** |

Paired differences, negative favours the first policy. Two-SE intervals are
descriptive over evaluation windows.

| Comparison | WikiText ΔPPL | WikiText ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 ours − FourOverSix | -0.027142 | -0.003956 ±0.001604 | -0.059346 | -0.006059 ±0.002295 |
| MixFP4 ours − NVFP4 | -0.091869 | -0.013326 ±0.001994 | -0.160712 | -0.016325 ±0.002911 |
| FourOverSix − NVFP4 | -0.064728 | -0.009370 ±0.002000 | -0.101366 | -0.010266 ±0.002287 |

Movement relative to the unquantized BF16 reference:

- wikitext: FourOverSix is +0.635437 above BF16; MixFP4 moves 0.027142 toward it (4.3% of the gap).
- c4: FourOverSix is +0.865521 above BF16; MixFP4 moves 0.059346 toward it (6.9% of the gap).

### Llama-3.1-8B: all ten frozen calibration settings

Every setting elects exactly 256 blocks with the same rule and differs only in the
calibration corpus and sequence count. Reported so the primary setting is not read
as a selected best.

| Calibration | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|
| math16 | 6.864338 | 9.798604 | -0.011187 | -0.025129 |
| math32 | 6.859864 | 9.786333 | -0.015661 | -0.037400 |
| math64 | 6.847213 | 9.784532 | -0.028311 | -0.039202 |
| code16 | 6.858930 | 9.795424 | -0.016595 | -0.028309 |
| code32 | 6.855278 | 9.784207 | -0.020246 | -0.039526 |
| code64 | 6.846576 | 9.785232 | -0.028948 | -0.038502 |
| math_code16 | 6.870324 | 9.808887 | -0.005200 | -0.014847 |
| math_code32 | 6.859400 | 9.785341 | -0.016125 | -0.038392 |
| math_code64 | 6.854017 | 9.782113 | -0.021507 | -0.041620 |
| **math_code128** | 6.848383 | 9.764387 | -0.027142 | -0.059346 |

## Qwen3-4B

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 (unquantized reference) | — | 13.662473 | 16.643560 |
| NVFP4 W4A4 | 0 | 13.936539 | 17.293613 |
| NVFP4 FourOverSix W4A4 | 0 | 14.269062 | 17.326633 |
| **MixFP4 ours (math_code128)** | 256 | **13.040957** | **16.615953** |

Paired differences, negative favours the first policy. Two-SE intervals are
descriptive over evaluation windows.

| Comparison | WikiText ΔPPL | WikiText ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 ours − FourOverSix | -1.228105 | -0.089999 ±0.005297 | -0.710680 | -0.041881 ±0.002294 |
| MixFP4 ours − NVFP4 | -0.895581 | -0.066419 ±0.004484 | -0.677660 | -0.039974 ±0.002311 |
| FourOverSix − NVFP4 | +0.332523 | +0.023580 ±0.004994 | +0.033020 | +0.001907 ±0.002283 |

Movement relative to the unquantized BF16 reference:

- wikitext: FourOverSix is +0.606589 above BF16; MixFP4 moves 1.228105 toward it (202.5% of the gap). MixFP4 lands **below** the BF16 reference here, so the share exceeds 100%; a quantized perplexity below an unquantized one is a known effect on this model and is not evidence of a better model.
- c4: FourOverSix is +0.683073 above BF16; MixFP4 moves 0.710680 toward it (104.0% of the gap). MixFP4 lands **below** the BF16 reference here, so the share exceeds 100%; a quantized perplexity below an unquantized one is a known effect on this model and is not evidence of a better model.

### Qwen3-4B: all ten frozen calibration settings

Every setting elects exactly 256 blocks with the same rule and differs only in the
calibration corpus and sequence count. Reported so the primary setting is not read
as a selected best.

| Calibration | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|
| math16 | 13.707123 | 16.965935 | -0.561939 | -0.360699 |
| math32 | 13.470236 | 16.790638 | -0.798826 | -0.535995 |
| math64 | 13.379006 | 16.736881 | -0.890056 | -0.589752 |
| code16 | 12.995038 | 16.632469 | -1.274024 | -0.694164 |
| code32 | 12.868947 | 16.563389 | -1.400115 | -0.763245 |
| code64 | 12.862923 | 16.527679 | -1.406139 | -0.798954 |
| math_code16 | 13.289850 | 16.754166 | -0.979212 | -0.572468 |
| math_code32 | 13.090026 | 16.662802 | -1.179036 | -0.663832 |
| math_code64 | 13.061725 | 16.623228 | -1.207337 | -0.703405 |
| **math_code128** | 13.040957 | 16.615953 | -1.228105 | -0.710680 |

## Verification

- For Llama-3.1-8B the NVFP4 and FourOverSix rows are asserted **exactly equal** to the
  archived released-code reproduction (job 335297), which the evaluator checks at run
  time. Llama is unaffected by the `o_proj` fix, so this anchors the whole path.
- The FourOverSix and MixFP4 rows reproduce the published fixed-256 evaluation
  (job 335428) to the last digit, which shows that adding the plain-NVFP4 policy left
  the evaluator numerically inert.
- Input token hashes are asserted identical to the released run for both models and
  every policy; pristine weight hashes and frozen map hashes are verified.

## Scope and limits

- Qwen3-4B numbers are not comparable to RaZeR's published Qwen row. That row was
  produced before commit `abab3c6`, when `o_proj` consumed unquantized attention
  output. Our Qwen baseline is correspondingly harder, not easier.
- Simulated W4A4 on text linear layers; no KV-cache quantization, generation accuracy,
  or native FP4 kernel throughput is measured.
- Tensor-wide activation factors span the whole teacher-forced window, so these are
  reference-text perplexities and not causal generation likelihoods.
- Two-SE intervals are descriptive evaluation-window intervals. They do not adjust for
  multiple comparisons across the ten calibration settings, for WikiText article
  dependence, or for calibration-draw variability (one draw per setting).
