# MixFP4 with task-calibrated type selection

Every result in this report uses the released 2048-token evaluation protocol.
WikiText-2 raw test is read as 141 full nonoverlapping windows; C4 is 256 seed-0
crops from validation shard 00000. Activation factors are tensor-wide, attention
is SDPA, WikiText is cached per window and C4 uncached, and perplexity uses the
released float32 aggregation. Non-aligned 512-token studies have been removed
from this report; their records remain under `results/` and are listed in the
final section.

[Paper-aligned panel](results/paper_baseline/REPORT.md) (job 335962).

## 1. Paper-aligned W4A4 result

Baselines are NVFP4 and NVFP4 FourOverSix. The method is MixFP4: the same FourOverSix
E2M1 weights with 256 8x64 tiles switched to E0M3, elected by our CE/KL task-gradient
calibration on OpenWebMath and CodeParrot only. WikiText-2 and C4 supply no calibration
data, no gradients and no selection feedback, so both are held out for every row.

### Llama-3.1-8B

Scope: 224 text linear matrices, 13,631,488 type blocks of 8x64, 6,979,321,856 weights. MixFP4 switches **256 blocks = 0.001878% of type blocks** (131,072 weights, about 1 block in 53,248).

| Policy | E0M3 blocks | % of type blocks | WikiText-2 | C4 |
|---|---:|---:|---:|---:|
| BF16 reference | — | — | 6.240087 | 8.958212 |
| NVFP4 W4A4 | 0 | 0 | 6.940252 | 9.925099 |
| NVFP4 FourOverSix W4A4 | 0 | 0 | 6.875525 | 9.823733 |
| **MixFP4, ours** | 256 | 0.001878% | **6.848383** | **9.764387** |

| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 − FourOverSix | -0.027142 | -0.003956 ±0.001604 | -0.059346 | -0.006059 ±0.002295 |
| MixFP4 − NVFP4 | -0.091869 | -0.013326 ±0.001994 | -0.160712 | -0.016325 ±0.002911 |
| FourOverSix − NVFP4 | -0.064728 | -0.009370 ±0.002000 | -0.101366 | -0.010266 ±0.002287 |

Movement relative to the unquantized BF16 reference:

- wikitext: FourOverSix sits +0.635437 above BF16; MixFP4 moves 0.027142 toward it (4.3% of the gap).
- c4: FourOverSix sits +0.865521 above BF16; MixFP4 moves 0.059346 toward it (6.9% of the gap).

Against the archived released NVFP4 run: WikiText 6.941772 (-0.001520), C4 9.940895 (-0.015796). See "Two deliberate differences from the released code" for why these differ.

The FourOverSix row reproduces the archived released run exactly, asserted at run time.

### Qwen3-4B

Scope: 252 text linear matrices, 7,096,320 type blocks of 8x64, 3,633,315,840 weights. MixFP4 switches **256 blocks = 0.003608% of type blocks** (131,072 weights, about 1 block in 27,720).

| Policy | E0M3 blocks | % of type blocks | WikiText-2 | C4 |
|---|---:|---:|---:|---:|
| BF16 reference | — | — | 13.662473 | 16.643560 |
| NVFP4 W4A4 | 0 | 0 | 13.936539 | 17.293613 |
| NVFP4 FourOverSix W4A4 | 0 | 0 | 14.269062 | 17.326633 |
| **MixFP4, ours** | 256 | 0.003608% | **13.040957** | **16.615953** |

| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| MixFP4 − FourOverSix | -1.228105 | -0.089999 ±0.005297 | -0.710680 | -0.041881 ±0.002294 |
| MixFP4 − NVFP4 | -0.895581 | -0.066419 ±0.004484 | -0.677660 | -0.039974 ±0.002311 |
| FourOverSix − NVFP4 | +0.332523 | +0.023580 ±0.004994 | +0.033020 | +0.001907 ±0.002283 |

Movement relative to the unquantized BF16 reference:

- wikitext: FourOverSix sits +0.606589 above BF16; MixFP4 moves 1.228105 toward it (202.5% of the gap). MixFP4 lands **below** the BF16 reference here, so this share exceeds 100%. Zero-shot accuracy shows this is not a better model than BF16; see "Perplexity below BF16 is not a quality claim".
- c4: FourOverSix sits +0.683073 above BF16; MixFP4 moves 0.710680 toward it (104.0% of the gap). MixFP4 lands **below** the BF16 reference here, so this share exceeds 100%. Zero-shot accuracy shows this is not a better model than BF16; see "Perplexity below BF16 is not a quality claim".

<!-- FIXED256_PAPER_EVAL_START -->
## 2. Math/code-only calibration: fixed-256 W4A4, aligned 2048-token benchmark

Calibration uses **OpenWebMath and CodeParrot only; neither WikiText nor C4 is used for calibration**. One 128-sequence causal scoring pass per model (64 math + 64 code, 512 tokens each) supplies the ten frozen source/sample-count settings below. The numeric suffix denotes the total calibration sequence count; math_code settings use equal numbers from each source.

This evaluation replays all ten frozen math/code-only fixed-256 maps for each of three models. **No adaptive maps, weight-MSE controls, recalibration, or best-setting selection are included.** The same map is used for both datasets. All maps contain exactly **256 E0M3 type blocks** of 8×64 weights; all other targeted weights use FourOverSix E2M1.

**Evaluation:** WikiText-2 raw test text concatenated with double newlines, nonoverlapping 2048-token windows; C4 validation shard 00000, 256 seed-0 random 2048-token crops using the released sampling rule. As in the release, WikiText uses a fresh cache per window and C4 disables it; no cache is carried between windows. Incomplete WikiText tails are omitted. PPL uses the released float32 NLL aggregation. WikiText and C4 token hashes match the released-code reproduction exactly for Llama-3.1-8B and Qwen3-4B.

**W4A4 definition:** tensor-wide FourOverSix activation factors, 16-element scale blocks, BF16 simulation, SDPA attention, and unquantized KV cache. Every targeted linear input is quantized, including Qwen o_proj. The small-model baseline is checked against the corrected repository wrapper. Qwen3.8-27B uses native Transformers 5.16.1 text-linear modules; its vision, convolution, recurrent state operations, normalization, and LM head are outside that linear quantization scope. It has no corresponding RaZeR Table 3 entry.

**Alignment means matching the evaluation protocol, not reproducing every published Table 3 value.** Baselines below are freshly measured with matching inputs and quantization scope. In particular, quantizing Qwen o_proj corrects an omission in the archived release. All improvements reported here are relative to **FourOverSix**, not NVFP4.

The older 512-token tables remain historical measurements under different context, C4 sampling, and per-token activation scales. These new tensor-wide factors can depend on future tokens, as in the paper-style simulation; these results are not a causal deployment claim. Maps were originally fitted with causal 512-token math/code calibration and are transferred here unchanged.

ΔPPL = fixed-256 PPL minus matched FourOverSix PPL; **negative is better**. Average is the unweighted arithmetic mean of the separate Wiki/C4 PPL values.

### Qwen3-4B

| Calibration | E0M3 blocks | Wiki PPL | C4 PPL | Average | Δ Wiki | Δ C4 | Δ average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix baseline | 0 | 14.269062 | 17.326633 | 15.797848 | +0.000000 | +0.000000 | +0.000000 |
| math16 | 256 | 13.707123 | 16.965935 | 15.336529 | -0.561939 | -0.360699 | -0.461319 |
| math32 | 256 | 13.470236 | 16.790638 | 15.130437 | -0.798826 | -0.535995 | -0.667411 |
| math64 | 256 | 13.379006 | 16.736881 | 15.057944 | -0.890056 | -0.589752 | -0.739904 |
| code16 | 256 | 12.995038 | 16.632469 | 14.813754 | -1.274024 | -0.694164 | -0.984094 |
| code32 | 256 | 12.868947 | 16.563389 | 14.716168 | -1.400115 | -0.763245 | -1.081680 |
| code64 | 256 | 12.862923 | 16.527679 | 14.695301 | -1.406139 | -0.798954 | -1.102547 |
| math_code16 | 256 | 13.289850 | 16.754166 | 15.022008 | -0.979212 | -0.572468 | -0.775840 |
| math_code32 | 256 | 13.090026 | 16.662802 | 14.876414 | -1.179036 | -0.663832 | -0.921434 |
| math_code64 | 256 | 13.061725 | 16.623228 | 14.842476 | -1.207337 | -0.703405 | -0.955371 |
| math_code128 | 256 | 13.040957 | 16.615953 | 14.828455 | -1.228105 | -0.710680 | -0.969392 |

### Llama-3.1-8B

| Calibration | E0M3 blocks | Wiki PPL | C4 PPL | Average | Δ Wiki | Δ C4 | Δ average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix baseline | 0 | 6.875525 | 9.823733 | 8.349629 | +0.000000 | +0.000000 | +0.000000 |
| math16 | 256 | 6.864338 | 9.798604 | 8.331471 | -0.011187 | -0.025129 | -0.018158 |
| math32 | 256 | 6.859864 | 9.786333 | 8.323098 | -0.015661 | -0.037400 | -0.026531 |
| math64 | 256 | 6.847213 | 9.784532 | 8.315872 | -0.028311 | -0.039202 | -0.033756 |
| code16 | 256 | 6.858930 | 9.795424 | 8.327177 | -0.016595 | -0.028309 | -0.022452 |
| code32 | 256 | 6.855278 | 9.784207 | 8.319743 | -0.020246 | -0.039526 | -0.029886 |
| code64 | 256 | 6.846576 | 9.785232 | 8.315904 | -0.028948 | -0.038502 | -0.033725 |
| math_code16 | 256 | 6.870324 | 9.808887 | 8.339605 | -0.005200 | -0.014847 | -0.010024 |
| math_code32 | 256 | 6.859400 | 9.785341 | 8.322371 | -0.016125 | -0.038392 | -0.027258 |
| math_code64 | 256 | 6.854017 | 9.782113 | 8.318065 | -0.021507 | -0.041620 | -0.031564 |
| math_code128 | 256 | 6.848383 | 9.764387 | 8.306385 | -0.027142 | -0.059346 | -0.043244 |

### Qwen3.8-27B

| Calibration | E0M3 blocks | Wiki PPL | C4 PPL | Average | Δ Wiki | Δ C4 | Δ average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix baseline | 0 | 7.287076 | 10.188365 | 8.737721 | +0.000000 | +0.000000 | +0.000000 |
| math16 | 256 | 7.308025 | 10.178003 | 8.743014 | +0.020948 | -0.010362 | +0.005293 |
| math32 | 256 | 7.304955 | 10.178423 | 8.741689 | +0.017879 | -0.009942 | +0.003968 |
| math64 | 256 | 7.290631 | 10.165107 | 8.727869 | +0.003555 | -0.023258 | -0.009852 |
| code16 | 256 | 7.270402 | 10.183651 | 8.727027 | -0.016674 | -0.004714 | -0.010694 |
| code32 | 256 | 7.296406 | 10.175853 | 8.736130 | +0.009330 | -0.012512 | -0.001591 |
| code64 | 256 | 7.286375 | 10.170633 | 8.728504 | -0.000701 | -0.017732 | -0.009217 |
| math_code16 | 256 | 7.302811 | 10.188348 | 8.745579 | +0.015734 | -0.000017 | +0.007859 |
| math_code32 | 256 | 7.271716 | 10.181243 | 8.726480 | -0.015360 | -0.007122 | -0.011241 |
| math_code64 | 256 | 7.269203 | 10.174075 | 8.721639 | -0.017874 | -0.014290 | -0.016082 |
| math_code128 | 256 | 7.292438 | 10.167717 | 8.730077 | +0.005361 | -0.020648 | -0.007643 |

Fixed-256 improves point PPL in **54/60** matched model/setting/dataset cells. Qwen3-4B and Llama-3.1-8B improve in all 20 cells each; Qwen3.8-27B improves in 14/20, with six WikiText regressions and ten C4 improvements. These settings share calibration data and evaluation windows; this count is descriptive, not a count of independent confirmations. Per-window losses and paired NLL diagnostics are retained in the machine-readable summary; intervals do not account for WikiText article dependence or calibration-draw variability.

Execution jobs: 335407, 335428; consolidated results: `job_335428`. Each job uses `gov113008/taide_h200`, an eight-H200 allocation, and independent one-GPU Slurm steps. Cache-only repairs reuse C4 after exact identity checks and retain the pre-fix reports. Every dataset comparison uses identical inputs and activation quantization scope within its model.

[Full results](results/fixed256_paper_eval/job_335428/REPORT.md) · [Protocol](results/fixed256_paper_eval/PROTOCOL.md).

<!-- FIXED256_PAPER_EVAL_END -->

<!-- BASELINE_PROTOCOL_AUDIT_START -->
## 3. Baseline audit against RaZeR Table 3

The historical math/code-only study used **512-token evaluation windows**. A separate baseline-only audit at **2048 tokens**, with the released C4 sampling procedure and tensor-wide activation factors, gives the values below. No calibration or E0M3 map selection occurs in this audit. The unquantized baselines match the paper to its displayed precision; shorter context accounts for most of the large gap in the original study.

| Model | W4A4 method | Paper Wiki PPL | Audit Wiki PPL | Paper C4 PPL | Audit C4 PPL |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | NVFP4 | 13.88 | 13.981052 | 17.21 | 17.281944 |
| Qwen3-4B | FourOverSix | 13.88 | 14.213994 | 17.21 | 17.299487 |
| Llama-3.1-8B | NVFP4 | 6.95 | 6.939384 | 9.94 | 9.929869 |
| Llama-3.1-8B | FourOverSix | 6.88 | 6.879243 | 9.83 | 9.823488 |

The Qwen wrapper formerly computed quantized attention-output activations but passed the unquantized tensor into o_proj; repository commit abab3c6 fixed this on 2026-06-17. A separate historical-behavior diagnostic leaves only those projection inputs unquantized at the same 2048-token context. Its results are:

| Qwen3-4B historical behavior | Wiki PPL | C4 PPL |
|---|---:|---:|
| NVFP4 | 13.948580 | 17.222591 |
| FourOverSix | 14.137439 | 17.286469 |

The corrected full-W4A4 measurements remain the appropriate baseline for methods quantizing all these inputs. Residual differences from the paper remain visible above; a historical-behavior match does not independently establish the exact code that generated the published table. **This baseline-only audit did not re-evaluate E0M3 maps. The subsequent Math/code-only fixed-256 benchmark above reports their aligned 2048-token results.**

[Full matched-context audit and validation](results/baseline_protocol_audit/REPORT.md) · [RaZeR Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).

<!-- BASELINE_PROTOCOL_AUDIT_END -->

<!-- RELEASED_REPRODUCTION_START -->
## 4. Direct reproduction with the February RaZeR release

The archived released evaluator at commit `e230099` was run directly for 14 cases (28 separate WikiText-2/C4 cells). **15/28 cells match Table 3 at its published two-decimal precision.** No calibration or parameter search was used. The evaluator uses 2048-token windows, seed 0, 256 sampled C4 windows, and its original float32 PPL aggregation.

Signed Δ is reproduced PPL minus published PPL; negative means lower perplexity. A displayed-precision match is not a claim of bitwise agreement with unpublished author outputs.

| Model | Method | Paper Wiki | Reproduced Wiki | Δ Wiki | Paper C4 | Reproduced C4 | Δ C4 | Matches |
|---|---|---:|---:|---:|---:|---:|---:|---|
| llama-3.1-8b | BF16 (paper: FP16) | 6.24 | 6.240087 | +0.000087 | 8.96 | 8.958212 | -0.001788 | wikitext, c4 |
| llama-3.1-8b | NVFP4 W4A4 | 6.95 | 6.941772 | -0.008228 | 9.94 | 9.940895 | +0.000895 | c4 |
| llama-3.1-8b | FourOverSix W4A4 | 6.88 | 6.875525 | -0.004475 | 9.83 | 9.823733 | -0.006267 | wikitext |
| llama-3.1-8b | RaZeR W4A4 | 6.74 | 6.744553 | +0.004553 | 9.63 | 9.630655 | +0.000655 | wikitext, c4 |
| llama-3.1-8b | NVFP4 W4A16 | 6.63 | 6.625925 | -0.004075 | 9.48 | 9.480332 | +0.000332 | wikitext, c4 |
| llama-3.1-8b | FourOverSix W4A16 | 6.60 | 6.598704 | -0.001296 | 9.42 | 9.423216 | +0.003216 | wikitext, c4 |
| llama-3.1-8b | RaZeR W4A16 | 6.50 | 6.500746 | +0.000746 | 9.29 | 9.292561 | +0.002561 | wikitext, c4 |
| qwen3-4b | BF16 (paper: FP16) | 13.66 | 13.662473 | +0.002473 | 16.65 | 16.643564 | -0.006436 | wikitext |
| qwen3-4b | NVFP4 W4A4 | 13.88 | 13.935143 | +0.055143 | 17.21 | 17.209829 | -0.000171 | c4 |
| qwen3-4b | FourOverSix W4A4 | 13.88 | 14.201942 | +0.321942 | 17.21 | 17.280434 | +0.070434 | none |
| qwen3-4b | RaZeR W4A4 | 13.82 | 14.104443 | +0.284443 | 17.11 | 17.260754 | +0.150754 | none |
| qwen3-4b | NVFP4 W4A16 | 13.83 | 13.631868 | -0.198132 | 16.85 | 16.851393 | +0.001393 | c4 |
| qwen3-4b | FourOverSix W4A16 | 13.83 | 14.040705 | +0.210705 | 16.85 | 17.015335 | +0.165335 | none |
| qwen3-4b | RaZeR W4A16 | 13.83 | 13.969723 | +0.139723 | 16.85 | 17.033857 | +0.183857 | none |

Execution: job `335302`, account `gov113008`; NVIDIA H200. Independent one-GPU Slurm steps share one eight-GPU allocation. Python 3.10.18, Torch 2.7.1+cu126, Transformers 4.57.3, datasets 4.4.1; recorded attention backend(s): `sdpa`. All policies within each model passed exact input-token-hash equality checks; recorded losses reproduce the original evaluator’s PPL exactly.

**Historical behavior and limits.** The archived Qwen wrapper leaves `o_proj` inputs unquantized, despite calculating a quantized copy. Its W4A4 labels therefore describe the release’s command-line setting, with this omission; they must not replace a corrected full-W4A4 baseline without disclosure. The original NVFP4 midpoint lookup also differs from the current arithmetic quantizer. The first January evaluator could load author-local cached C4 tokens; those tokens are unavailable, and the February release always regenerates seed-0 windows. The released environment omits a Torch version, so exact environment reconstruction is not established. These are reproduction limits, not explanations proven to account for every residual.

This audit does not change or validate the existing 512-token E0M3 calibration gains at 2048 tokens.

[Published Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).

### Fixed environment diagnostic

Both complete attempts are retained. The second uses Python 3.10.18, the released core package pins, and Torch 2.7.1/CUDA 12.6 inferred from the listed Triton/CUDA dependencies. The first uses Python 3.11/Torch 2.9. This does not establish the authors’ exact environment.

All input windows match exactly between attempts. Maximum absolute PPL change: **0.000004**. Δ below is second environment minus first; every case is shown.

| Case | Δ Wiki | Δ C4 | Changed weight matrices |
|---|---:|---:|---:|
| llama-3.1-8b_bf16 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_nvfp4_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_four_over_six_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_razer_w4a4 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_nvfp4_w4a16 | +0.000000 | +0.000000 | 0/224 |
| llama-3.1-8b_four_over_six_w4a16 | +0.000000 | +0.000002 | 0/224 |
| llama-3.1-8b_razer_w4a16 | +0.000000 | +0.000000 | 0/224 |
| qwen3-4b_bf16 | +0.000000 | +0.000004 | 0/252 |
| qwen3-4b_nvfp4_w4a4 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_four_over_six_w4a4 | +0.000000 | -0.000004 | 0/252 |
| qwen3-4b_razer_w4a4 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_nvfp4_w4a16 | -0.000003 | +0.000004 | 0/252 |
| qwen3-4b_four_over_six_w4a16 | +0.000000 | +0.000000 | 0/252 |
| qwen3-4b_razer_w4a16 | +0.000000 | +0.000000 | 0/252 |

Qwen3-4B NVFP4 weight-only WikiText is listed as **13.63 in Table 1** and **13.83 in Table 3**. The primary table above keeps the Table 3 target. This internal difference is documented separately rather than changing the reproduction target.

[Full reproduction record](results/released_reproduction/job_335302/REPORT.md) · [Protocol](results/released_reproduction/PROTOCOL.md).

<!-- RELEASED_REPRODUCTION_END -->

## 5. Perplexity below BF16 is not a quality claim

On Qwen3-4B, MixFP4 perplexity falls below the unquantized BF16 reference. Perplexity
cannot settle that on its own: a model that becomes less overconfident scores better on
next-token loss without predicting better. The same frozen policies were therefore
evaluated zero-shot on multiple choice, where a smoothing artefact should not help.
BF16 restores pristine weights and removes activation quantization; every other row uses
the paper-aligned W4A4 path.

| Policy | WikiText-2 PPL | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean acc |
|---|---|---|---|---|---|---|---|---|
| BF16 reference | 13.662473 | 0.7816 | 0.5358 | 0.6846 | 0.4020 | 0.8495 | 0.6519 | 0.6509 |
| FourOverSix W4A4 | 14.269062 | 0.7462 | 0.4906 | 0.6616 | 0.3940 | 0.8358 | 0.6212 | 0.6249 |
| MixFP4 256 tiles | 13.040957 | 0.7563 | 0.4881 | 0.6652 | 0.3880 | 0.8330 | 0.6409 | 0.6286 |
| MixFP4 65,536 tiles | 10.864750 | 0.7778 | 0.5043 | 0.6725 | 0.4060 | 0.8453 | 0.6314 | 0.6395 |

**Two conclusions, and they point in different directions.**

Accuracy corroborates the method among the quantized policies: FourOverSix 0.6249 to 0.6286 at 256 tiles to 0.6395 at 65,536, the same ordering perplexity gives, with the larger map ahead on 5/6 tasks. Quantization costs 0.0260 mean accuracy against BF16; the 256-tile map recovers 14.2% of that and the larger map 56.2%.

Accuracy does **not** support beating BF16. The best quantized policy remains 0.0114 below the unquantized model, while its perplexity is 2.797723 better. Perplexity is therefore not a reliable absolute quality measure against BF16 on this model, and no claim in this report rests on the below-BF16 perplexities. Comparisons against the matched FourOverSix baseline are unaffected.

Tasks are 0-shot via lm_eval 0.4.5. Skipped for dataset-loading reasons unrelated to the model: piqa.

## 6. Two deliberate differences from the released code

Both make our baselines harder to beat, not easier.

**Qwen `o_proj` inputs are quantized.** At the archived release commit
`e230099`, `models/qmodule_qwen3.py` passed unquantized attention output into
`o_proj`, so Qwen "W4A4" left one projection's input in BF16. The RaZeR author
corrected this himself in commit `abab3c6`, after publication. We evaluate the
corrected behaviour, so our Qwen FourOverSix baseline is 14.269062 WikiText
where the pre-fix code gives 14.201942. Llama was never affected, and its rows
match the released run to the last digit; that is the control which shows the
difference comes only from that one line. Qwen rows here are therefore **not**
comparable with the published RaZeR Qwen row.

**NVFP4 saturation is clamped.** The archived `quant_nvfp4` used a rounding
path with no E2M1 saturation clamp, so an FP8 subnormal block scale that rounded
down could produce magnitude code 8, which is not a legal FP4 code. The current
`quant_nvfp4` clamps to [-6, 6]. Baseline and method both use the corrected
quantizer. Exact equality with the released NVFP4 run is therefore not expected
and not asserted; the released values are recorded as a comparison in each
report. The FourOverSix path is unchanged since the release and keeps a hard
exact-equality assertion.

## 7. The format

MixFP4 is NVFP4 plus a second, coarser block granularity that selects the FP4
element data type. The FP32 per-tensor global scale, the E4M3 block scale and
the 16-element scale block are inherited from NVFP4 unchanged.

A **scale block** is always 16 elements along K and owns one E4M3 scale. A
**type block** is a 2-D tile that owns one element data type and contains many
scale blocks. The two element types are **E2M1**, the standard FP4 grid with
maximum magnitude 6, and **E0M3**, the evenly spaced signed grid with maximum
magnitude 7. Both encode 15 distinct values in 16 codes and share the same
ue4m3 scale; only the spacing differs, so the better choice depends on the
distribution inside the tile.

The tile shape is not free. The public NVFP4 path issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read either operand as E0M3. A single instruction
cannot subdivide its operand tile, so for weights in operand B the smallest
realizable type block is `n8 x k64`. All results here use exactly that 8x64
weight tile, with alpha fixed at 1 on the E0M3 branch.

## 8. Why a weight-error election is insufficient

The natural selector compares the squared weight error of the two candidates
per tile and takes the smaller. It measures how well a candidate approximates
the pristine weights, not which inputs reach the block or how its outputs move
the prediction. Even at one linear layer the output error is

$$
\mathbb E\|\Delta W x\|^2=\operatorname{tr}(\Delta W\,\mathbb E[xx^T]\,\Delta W^T),
$$

which depends on the input second moment, and that still omits the downstream
loss and interactions between quantized blocks. A weight-error gain of factor
$g$ certifies an output-error reduction only when $g > 1-1/\kappa(S)$, and the
measured conditioning of $S$ puts that threshold far above the gains available
at a realizable tile size. This is why the election below is driven by a task
loss rather than a reconstruction loss.

MixFP4 can also appear to help merely because its implementation searches more
scales. We therefore hold the scale search fixed: the E2M1 baseline is
FourOverSix, the E0M3 alternative is fixed at alpha 1, and the experiment
changes only the tile's element type. No rotation, permutation, scale search or
weight training is added.

## 9. The selection method

Form the canonical FourOverSix Q0 and the E0M3-alpha1 Q1 once, so each 8x64
tile j has a fixed difference D_j. At the unchanged FourOverSix W4A4 student,
one forward per calibration sequence supports two backward passes, giving each
tile a directional derivative for next-token cross entropy and for KL from the
pristine BF16 teacher:

    g_CE[i,j] = <grad_Wj CE_i, D_j>,   g_KL[i,j] = <grad_Wj KL(teacher||student)_i, D_j>.

Activation quantization uses an identity straight-through derivative during
scoring. With n calibration sequences,

    u_j = max(mean(g_CE[:,j]) + 2 SE(g_CE[:,j]), mean(g_KL[:,j]) + 2 SE(g_KL[:,j])),

and the 256 most negative u_j are switched to E0M3. Requiring both objectives to
favour a switch is the point of the maximum: for each objective separately, the
sum of its estimated upper directional scores is bounded above by the sum of
per-tile maxima, so one selection bounds both. This is an approximate predictor
of a discrete switch, not a proof that a finite joint flip lowers the network
loss, and two SE does not control the many tile comparisons simultaneously. The
count 256 and the factor 2 are fixed empirical constants, never tuned per model
or per destination dataset.

Calibration uses OpenWebMath and CodeParrot only. Neither evaluation corpus
contributes gradients or any selection feedback.

## 10. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is
  measured, and no speedup is claimed.
- Tensor-wide activation factors span the whole teacher-forced window, so these
  are reference-text perplexities, not causal generation likelihoods.
- Two-SE intervals are descriptive evaluation-window intervals. They do not
  adjust for comparisons across the ten calibration settings, for WikiText
  article dependence, or for calibration-draw variability; one calibration draw
  per setting is used.
- The 256-tile count is a fixed constant here. Evidence that the count/gain
  curve has a model-dependent interior optimum exists only under the
  512-token protocol and is deliberately excluded from this report until it is
  re-measured under the aligned protocol.
- Gradient selection, distillation and sparse optimization are established
  tools. Their use here is not by itself a novelty claim.

## 11. Records not included in this report

These studies use protocols other than the aligned 2048-token one and are
excluded from the tables above. Their records are retained.

- [Pooled calibration transfer panel](results/transfer_rule/REPORT.md) and
  [held-out C4](results/c4_frozen/REPORT_332781.md): 512-token windows, causal
  per-token activation factors, C4 among the calibration sources.
- [Qwen3.8-27B pooled extension](results/pooled_qwen27b/model_332840/REPORT.md).
- [Tile-count sweep](results/cap_sweep/REPORT.md) and
  [calibration-chosen count](results/adaptive_count/REPORT.md): 512-token.
- [Adaptive-count study](results/math_code_adaptive/summary_333786_333788/REPORT.md):
  512-token evaluation of the curvature-penalised selector.
- [Earlier format exploration](results/MIXFP4_REPORT.md) and
  [decision-rule rounds](results/DECIDE_SUMMARY.md).
