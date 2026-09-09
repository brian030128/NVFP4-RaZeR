# Making MixFP4 Work Across Domains

**Motivation, method, experiments, and limitations**

<!-- FIXED256_CURRENT_POINTER -->
**Paper-protocol fixed-256 results (2048 tokens):** [matched WikiText/C4 measurements](results/fixed256_paper_eval/job_335428/REPORT.md). The Math/code-only section below contains these results; other 512-token tables are historical.


Updated September 9, 2026 · Original report: commit `5dc17ea`; subsequent studies are in the workspace.

<!-- BEGIN POOLED PERFORMANCE TABLES -->
## Pooled calibration: all evaluated models and datasets

Each cell is **FourOverSix baseline PPL → pooled-map PPL**; lower is better. The pooled map uses all 192 calibration sequences with the common CE/KL two-SE rule and 256-tile cap (`pooled192`, named `all_pooled` in the initial study). These tables cover its held-out/reference-text evaluations across all eight models and all seven dataset families evaluated. Other selection methods and calibration fitting losses are not included in the dataset averages.

**Mean PPL** is the unweighted arithmetic mean of the available dataset PPLs in that row, computed separately for baseline and pooled map from unrounded values. It is a descriptive dataset average, not pooled-corpus perplexity or a statistical significance test. **—** means not evaluated and is excluded from the mean. Different dataset coverage and model tokenizers limit comparisons between rows; the 27B mean below covers C4 and WikiText-2 only. Qwen3-4B and Llama-3.1-8B have five evaluated datasets; the other five models have four.

### Current causal evaluation

All policies use per-token activation factors. C4 has 256 validation documents per model; literature (PG19), science (arXiv articles), and government (GovReport reports) each have 64 test documents per model. Each document contributes a 512-token crop. Causal WikiText-2 uses all nonoverlapping512-token windows of the concatenated raw test split, omitting only the final incomplete window. The same frozen map serves all available domains of a model.

| Model | C4 | WikiText-2 | PG19 / literature | arXiv / science | GovReport / government | Mean PPL |
|---|---:|---:|---:|---:|---:|---:|
| OPT-350M | 26.7439 → 26.4696 | — | 26.7803 → 26.3180 | 39.9885 → 38.9790 | 19.3052 → 19.0420 | 28.2045 → 27.7021 |
| Qwen3-0.6B | 37.9007 → 34.7924 | — | 42.2868 → 38.2347 | 22.9779 → 21.3853 | 20.9650 → 19.2934 | 31.0326 → 28.4265 |
| Llama-3.2-1B-Instruct | 24.8986 → 24.1244 | — | 27.4281 → 26.1702 | 22.9979 → 22.1912 | 15.9390 → 15.4375 | 22.8159 → 21.9808 |
| OLMo-1B | 14.9844 → 14.8830 | — | 18.3616 → 18.2721 | 21.5055 → 21.1115 | 12.1762 → 12.0859 | 16.7569 → 16.5881 |
| Pythia-1.4B | 20.4712 → 20.1627 | — | 17.3559 → 17.0860 | 19.7099 → 19.4682 | 14.1871 → 13.9786 | 17.9310 → 17.6739 |
| Qwen3-4B | 21.9282 → 20.9027 | 19.4149 → 17.5212 | 21.7798 → 20.5380 | 13.6565 → 13.0184 | 12.7575 → 12.0811 | 17.9074 → 16.8123 |
| Llama-3.1-8B | 11.6001 → 11.5099 | 9.0929 → 9.0113 | 11.6750 → 11.5913 | 10.8050 → 10.7011 | 8.2520 → 8.1814 | 10.2850 → 10.1990 |
| Qwen3.8-27B | 12.6450 → 12.6353 | 9.0409 → 8.9971 | — | — | — | 10.8429 → 10.8162 |

The 27B C4 difference is inconclusive (paired ΔNLL −0.000764 ±0.001538, descriptive two-SE); its small point gain must not be presented as a supported improvement. OLMo-1B literature is also inconclusive. Of the preceding 29 causal comparisons excluding WikiText-2, 27 have supporting descriptive paired two-SE intervals. The three added WikiText-2 comparisons have 3 supported gains and 0 supported harms under descriptive two-SE intervals; adjacent WikiText windows may share articles, so these intervals do not account for article dependence. [Transfer results](results/transfer_rule/REPORT.md), [seven-model C4 results](results/c4_frozen/REPORT_332781.md), [27B C4 result](results/pooled_qwen27b/model_332840/REPORT.md), [causal WikiText-2 results](results/wiki_frozen/REPORT_332974_332976.md).

### Earlier development evaluations: window-wide activation factors

These are the original three pooled maps, replayed unchanged in subsequent confirmation. WikiText-2 uses 32 held-out 512-token windows; GSM8K and MBPP use 32 held-out reference-text examples each, truncated to at most 512 tokens. Their reported PPL is exp(mean example NLL); math and code scores are not answer accuracy or pass@k. These historical runs cover only the three models below; the causal WikiText-2 runs for three larger models are in the current table. GSM8K and MBPP were not evaluated for the other five models with this pooled rule.

**Historical only:** window-wide activation factors can depend on future tokens. These numbers are retained for completeness and are not causal-likelihood evidence. Their averages are kept separate from the causal table.

| Model | WikiText-2 | GSM8K reference text | MBPP reference text | Mean PPL |
|---|---:|---:|---:|---:|
| OPT-350M | 37.8453 → 37.1308 | 19.1657 → 19.1168 | 29.7292 → 28.7015 | 28.9134 → 28.3164 |
| Qwen3-0.6B | 39.4416 → 34.5269 | 5.9890 → 5.6115 | 11.4359 → 10.4532 | 18.9555 → 16.8639 |
| Llama-3.2-1B-Instruct | 21.0039 → 20.3201 | 5.0743 → 4.9452 | 9.6906 → 9.5024 | 11.9229 → 11.5892 |

[Development study](results/consensus_format/REPORT_332332.md). The all-source pooled map was a secondary control; it does not rescue the failed leave-source-out consensus screen.

### Original confirmation: window-wide activation factors

These five-model measurements used the same maps and inputs later replayed in the causal table. They are shown to preserve every evaluation setting, and must not be counted again as independent confirmation. The full prespecified confirmation screen failed its C4-only comparison despite the baseline gains.

| Model | PG19 / literature | arXiv / science | GovReport / government | Mean PPL |
|---|---:|---:|---:|---:|
| OPT-350M | 26.8110 → 26.3857 | 39.9208 → 38.9119 | 19.3680 → 19.0120 | 28.6999 → 28.1032 |
| Qwen3-0.6B | 42.4125 → 38.1521 | 22.9353 → 21.3439 | 20.9340 → 19.3883 | 28.7606 → 26.2947 |
| Llama-3.2-1B-Instruct | 27.5652 → 26.3203 | 23.0250 → 22.1103 | 15.9851 → 15.4686 | 22.1917 → 21.2997 |
| OLMo-1B | 18.3940 → 18.2765 | 21.7249 → 20.9924 | 12.1758 → 12.0902 | 17.4315 → 17.1197 |
| Pythia-1.4B | 18.9405 → 18.5658 | 21.5648 → 20.9412 | 15.4993 → 15.1421 | 18.6682 → 18.2164 |

[Original confirmation](results/pooled_confirmation/REPORT_332349.md). No window-wide confirmation run was performed for Qwen3-4B, Llama-3.1-8B, or Qwen3.8-27B under this pooled protocol.

Source values and per-cell report paths: [table data](results/transfer_rule/pooled_performance_tables.json). Reproduce with `build_pooled_performance_tables.py` via `slurm/pooled_performance_tables.sbatch`.
<!-- END POOLED PERFORMANCE TABLES -->

<!-- FIXED256_PAPER_EVAL_START -->
## Math/code-only calibration: fixed-256 W4A4, aligned 2048-token benchmark

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

<!-- MATH_CODE_ADAPTIVE_START -->
**Historical 512-token adaptive study:** [Full tables and diagnostics](results/math_code_adaptive/summary_333786_333788/REPORT.md). Its adaptive counts, fixed-256 controls, and weight-MSE controls use a different evaluation protocol and are not part of the aligned benchmark above.
<!-- MATH_CODE_ADAPTIVE_END -->

<!-- BASELINE_PROTOCOL_AUDIT_START -->
## Baseline audit against RaZeR Table 3

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
## Direct reproduction with the February RaZeR release

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




## Current result: a frozen rule across seven models

With causal per-token activation factors, the unchanged rule now improves
**21/21** matched FourOverSix perplexity comparisons across seven models,
from OPT350M to Llama8B, on literature, science and government text.
Twenty gains have supporting descriptive paired2SE intervals; none has
supported harm. It also beats weight-MSE election in21/21 point comparisons.
The [consolidated result](results/transfer_rule/REPORT.md) separates the
initial confirmation, causal replay and4B/8B size extension, and preserves
the unsuccessful comparator criterion described below.

One shared scoring pass on a fixed192-sequence pool supports a simple tile
rule: estimate each8x64 switch's next-token-loss and teacher-KL derivatives,
require both to favor the switch after a common2SE filter, and take at most256
tiles. All models use the same recipe. There is no candidate-loss backtracking,
per-domain configuration election, or separate calibration per configuration.
The [method](results/pooled_confirmation/METHOD.md) gives the equations and
states the approximation and fixed-constant assumptions explicitly.

The frozen confirmation covers five model families and three previously
uninspected data families: literature, scientific articles and government
reports. Pythia1.4B and OLMo1B are new model families relative to development;
the other three models' primary maps replay exactly. The same map is used for
every domain of each model. The method improves **15/15 comparisons over
FourOverSix**, with all15 paired descriptive2SE intervals supporting gains,
and beats weight-MSE election in15/15. Raw losses, source/data hashes, controls
and fitting audits are in the
[confirmation report](results/pooled_confirmation/REPORT_332349.md).

The complete prespecified screen nevertheless **fails** its stronger C4-only
comparison:8/15 point wins, versus the required9. At equal calibration-token
count, mixing sources improves only6/15 point estimates, with two supported
gains and two supported harms. The evidence supports transfer of the frozen
procedure; it does not establish that source diversity is the cause or is
uniformly preferable. No criterion has been relaxed after seeing these results.

This remains a calibrated procedure, not a weight-only universal theorem or
a novelty claim for gradient selection. The original confirmation used
window-wide activation factors. The subsequent
[causal audit](results/causal_replay/REPORT_332374.md) retained improvements
in15/15 comparisons with per-token factors,14 supported, and passed its
declared retention/prefix-independence screen. It reused the same frozen maps
and inputs, without recalibration. The
[4B/8B extension](results/pooled_scale/REPORT_332389.md) then passed its
size-transfer screen with6/6 supported gains and the same256-tile cap.

Changing future tokens changed earlier logits under the older window-wide
factor in all five models tested under both conventions. The per-token
version produced bitwise-identical prefix logits for both baseline and
selected maps in all seven models. Per-token factors change the activation
representation; this is not claimed as a free native-kernel modification.
All results simulate W4A4 nonhead linear operations. Native FP4-kernel speed,
generation accuracy and multi-pool seed replication are not measured.

## Held-out C4 evaluation of the frozen maps

The unchanged maps improve held-out C4 PPL on **7/7 models**, all seven with
supporting descriptive paired two-SE intervals. They beat weight-MSE on 7/7
and C4-only selection on 4/7 point comparisons.

| Model | FourOverSix C4 PPL | Selected C4 PPL | ΔPPL |
|---|---:|---:|---:|
| OPT-350M | 26.743945 | 26.469613 | -0.274332 |
| Qwen3-0.6B | 37.900671 | 34.792433 | -3.108238 |
| Llama-3.2-1B-Instruct | 24.898634 | 24.124422 | -0.774213 |
| OLMo-1B | 14.984432 | 14.883008 | -0.101424 |
| Pythia-1.4B | 20.471159 | 20.162666 | -0.308493 |
| Qwen3-4B | 21.928243 | 20.902742 | -1.025501 |
| Llama-3.1-8B | 11.600109 | 11.509863 | -0.090246 |

The [C4 follow-up](results/c4_frozen/REPORT_332781.md) evaluates the same seven
saved maps on 256 distinct validation documents per model, with one seeded
512-token crop per document. Exact text hashes exclude all 192 calibration
documents for each model. Model revisions, source linear weights and map-file
hashes are verified; no calibration, gradient scoring, map selection or
backtracking is performed. All policies use the causal per-token activation
convention. The report includes FourOverSix, pooled192, C4-only64, mixed64 and
weight-MSE PPL, raw paired document losses, and descriptive two-SE intervals.

C4 is a calibration source. This is held-out within-source evaluation,
separate from the 21 literature/science/government transfer comparisons.
The `C4-64` control means a map selected earlier using 64 C4 training sequences;
it does not mean that its evaluation examples are calibration examples.

## Qwen3.8-27B: the same rule gives an inconclusive C4 gain

The [27B extension](results/pooled_qwen27b/model_332840/REPORT.md) retains the
same 192-sequence calibration recipe, CE/KL two-SE score and 256-tile cap.
It uses the pinned native hybrid text architecture with Transformers 5.16.1.
Held-out C4 PPL changes from **12.644977 to 12.635323**, a reduction of only
**0.009654**. Paired ΔNLL is **-0.000764 ±0.001538** (descriptive two-SE),
which includes zero. This is below the earlier 0.01-PPL practical reference
threshold and is not a supported improvement. The seven-model gains above
therefore do not establish meaningful C4 improvement on this 27B target.

C4-only64 reaches 12.630390, mixed64 reaches 12.627442, and weight-MSE reaches
12.671271. None of these controls is used to replace the primary pooled map.
All policies use the same 256 held-out documents and causal activation factors.
Exact calibration/test text-hash overlap is zero; baseline and selected prefix
independence checks pass. The [eight-model comparison and independent overlap
audit](results/pooled_qwen27b/C4_COMPARISON_332840.md) preserve this inconclusive
target result separately from the preceding seven-model confirmation.

## Earlier report and exploratory research record

**Research continuation (September 7; subsequent workspace experiments):**
The calibration/backtracking method below does not resolve the request for a
strong, transferable tile-selection rule. Six additional fixed-method studies
test conditional compensation, asymmetric compensation, calibration-free
activation election, interacting weight tiles, teacher-Fisher sensitivity, and
complete tile branches with columnwise GPTQ compensation.
Their protocols, unsuccessful cases, and paired results are collected in
[the continuation report](results/format_directions/REPORT.md). The
[method analysis](results/format_directions/METHOD_AND_EVIDENCE.md) separates
what each objective guarantees from what must generalize empirically. These
experiments are exploratory and do not establish a paper-ready universal rule.
The [findings](results/format_directions/FINDINGS.md) summarize why all six
screens failed; the final panel improves seven of nine comparisons over
FourOverSix but has two supported math regressions.

**September 8 continuation:** Additional fixed-method studies and a replay of
failed maps on their original fitting examples are recorded in the
[continuation log](results/format_directions/CONTINUATION_20260908.md).
The replay identifies failure of the actual fitting objective, before domain
transfer, for the large curvature-selected maps. Current-model binary updates
and subsequent shared-score methods were tested under frozen recipes. The
validated transfer results and their claim limits are summarized above.

MixFP4 gives a quantizer an additional choice: represent a block with E2M1 or
E0M3 while keeping its element width at four bits. The difficult part is choosing
where that flexibility helps the model. A format that reconstructs a weight
block more accurately need not improve the predictions of an already quantized
network. A selection that improves WikiText need not improve another domain.

The earlier approach used **task-gradient selection on C4, followed by
actual-loss backtracking and independent validation**. The same algorithm
produced accepted maps on three models, with two calibration seeds for the
27B target. It reduced perplexity by at least **0.01 in 15 of 16 evaluated
model/seed/domain cases**. Fourteen cases had paired uncertainty intervals
supporting an improvement; no case had an interval supporting harm. The
remaining case, Llama code, was inconclusive rather than an established gain.

This is evidence for a useful common calibration procedure. It is not a
universal weight-statistic rule, a fixed map that can be reused across models,
or a guarantee for arbitrary domains. The experiments also identify clear
failures of Wiki-only selection and teacher-based selection on code.

## 1. Motivation: use four bits more effectively

At four bits, the placement of representable values matters. The two formats
used in this implementation have the following value sets before scaling:

| Format | Nonnegative values | Main distinction |
|---|---|---|
| E2M1 | 0, 0.5, 1, 1.5, 2, 3, 4, 6 | Nonuniform spacing, with smaller gaps near zero |
| E0M3 | 0, 1, 2, 3, 4, 5, 6, 7 | Uniform spacing |

Both also represent the negative values. Each has 15 distinct numerical values
in 16 codes because zero has redundant encodings. E0M3 here denotes the
implemented signed uniform grid; it does not mean that three mantissa bits
make every value more accurate than E2M1.

The tradeoff depends on scaling. With an ideal maximum-fitting scale, E2M1
places more resolution near zero, whereas E0M3 distributes resolution evenly
across the range. Real block scales are also quantized, so rounding those scales
can change which candidate is preferable. Different blocks can therefore
benefit from different grids.

The study uses two granularities:

| Quantity | Granularity |
|---|---|
| Element representation | Four bits |
| Block scale | One E4M3 scale per 16 elements |
| Format choice | One E0M3/E2M1 decision per 8×64 weight tile |
| Tensor normalization | Shared tensor-scale convention |

An 8×64 tile contains 512 weights and 32 scale groups. Its format choice is
shared by all those weights, while each 16-element group keeps its own scale.
This constrains the search to legal tile changes rather than independent
per-weight decisions. It also means that adding a format option does not
increase the element width. The experiments do **not** establish the packed
metadata cost or runtime speed of a production mixed-format kernel.

The practical objective is to improve accuracy at this fixed quantization
scope. A reduction of 0.01 absolute PPL already counts as worthwhile in this
project; a method need not produce a dramatic percentage gain to be useful.

## 2. Why a straightforward format election is insufficient

### Local reconstruction does not measure the full task

A natural selector compares the squared weight errors of the two candidates:

$$
\|Q_{E0M3}(W_b)-W_b\|_F^2
\quad\text{and}\quad
\|Q_{E2M1}(W_b)-W_b\|_F^2.
$$

This measures how well each candidate approximates the pristine weights.
It does not account for which inputs reach the block or how its outputs affect
the final prediction. Even at the linear-layer level, the output error is

$$
\mathbb E\|\Delta W x\|^2
=\operatorname{tr}(\Delta W\,\mathbb E[xx^T]\,\Delta W^T),
$$

which depends on the input second moment. Replacing weight error with output
error adds useful information, but still does not include the full downstream
loss or interactions with other quantized blocks.

The earlier mechanism experiments illustrate the distinction. High-energy
input columns identified a useful region of the target model, but randomizing
output-row groups within the selected regions lost much of the benefit.
The input location alone was not a sufficient selector. Separately useful
sets of corrections also did not combine additively. These observations
motivated a task-level selector rather than a universal channel-index rule.
See the [earlier mechanism findings](results/task_sensitivity_four_over_six/FINDINGS.md).

### Scale improvements must be separated from format improvements

MixFP4 can appear to help because its implementation searches more scales,
even when the additional E0M3 representation is not responsible for the gain.
We therefore used a strong fixed baseline and fixed candidate formulas:

- **E2M1 baseline:** FourOverSix, choosing between normalization toward code 6
  and code 4 independently in each scale group, using reconstruction error.
- **E0M3 alternative:** maximum-fitting normalization toward code 7, with
  alpha fixed at 1.
- **Selection experiment:** change only the tile's candidate representation;
  add no scale search, rotation, permutation, or weight training.

FourOverSix's two E2M1 normalizations correspond to alpha values 1 and 1.5
relative to normalization toward code 6. The candidates include the existing
E4M3 scale rounding and saturation behavior. Thus the final experiment asks
whether E0M3 can improve an already strong E2M1 baseline, not whether a weaker
baseline can be beaten by giving one branch extra scale optimization.

The implementation is in [quantizer.py](quantize/quantizer.py). Earlier scale
search results motivated this control; their numerical gains should not be
substituted for the type-selection results below.

## 3. The selection method

The central question is: **at the quantized model we will actually deploy,
which finite E0M3 substitutions are likely to improve task loss?**

Let $W^B$ denote the FourOverSix baseline and define the fixed candidate
change for tile $b$ as

$$
\Delta W_b=Q_{E0M3}(W_b)-W^B_b.
$$

For calibration sequence $i$, calculate

$$
s_{i,b}=\left\langle
\nabla_{W_b}\ell_i(W^B),\Delta W_b
\right\rangle.
$$

Here $\ell_i$ is mean next-token negative log-likelihood (NLL), in nats per
token. A negative score predicts that moving toward the E0M3 candidate reduces
loss. Crucially, the gradient is evaluated at the **quantized baseline**, not
the pristine model. It can identify corrections useful in the presence of the
baseline's existing errors.

The forward pass uses the actual fake-quantized weights and activations.
Backward propagation uses an identity straight-through estimator (STE) through
activation quantization. For a linear projection, its weight gradient is
formed from the actual quantized input $X$ and output adjoint $D$ as
$D^T X$, then reduced against $\Delta W_b$ per tile. Parameters remain
frozen: this collects sensitivity scores, not trained weights.

This is an approximate discrete-switch predictor. The STE does not make
activation rounding differentiable in the ordinary sense, and a full switch
can differ substantially from its first-order forecast.

### Stable scores and a bounded proposal

Across the fitting sequences, estimate the mean $\mu_b$ and standard error
$SE_b$. A tile is eligible only when

$$
\mu_b+2SE_b<0.
$$

Rank eligible tiles by their mean, most negative first. Starting with budget
$B=0.1$, take the largest ranked prefix satisfying

$$
-\sum_{b\in S}\mu_b\le B.
$$

The budget is in predicted NLL units rather than a fixed number or fraction of
tiles. This lets the same procedure select a different count on each model.
The budget limits extrapolation empirically; it is not an upper bound on the
actual loss change. The per-tile two-SE check is also a heuristic, not a
multiple-comparison certificate over millions of tiles.

### Backtracking on actual joint loss

We then install the **entire proposed map** and evaluate its fitting loss.
Let $\widehat\Delta=\sum_{b\in S}\mu_b$ be the predicted change, and let
$\bar d$ and $SE_d$ describe the measured paired per-sequence NLL change.
The fitting check requires

$$
\bar d+2SE_d<0,
\qquad
\frac{\bar d}{\widehat\Delta}\ge0.25.
$$

If either condition fails, halve the budget and construct a smaller prefix.
There are at most eight attempts, with budgets $0.1\times2^{-k}$ for
$k=0,\ldots,7$. If none succeeds, retain the baseline.

Backtracking was the decisive repair. In the first Qwen3.8 C4 run:

| Budget | E0M3 tiles | Predicted fitting ΔNLL | Measured fitting ΔNLL | Decision |
|---|---:|---:|---:|---|
| 0.1000 | 53,859 | -0.1000 | +0.09948 | Reject |
| 0.0500 | 12,202 | -0.0500 | +0.02287 | Reject |
| 0.0250 | 3,145 | -0.0250 | -0.00290 | Reject: only 11.6% of forecast |
| 0.0125 | 905 | -0.01250 | -0.00611 | Pass: 48.9% of forecast |

The second seed reproduced the initial wrong-direction prediction: predicted
-0.1, measured +0.09430. It also passed after shrinking the budget to 0.0125.
Because these failures occurred on the fitting domain itself, matching the
calibration domain alone could not solve the problem. The residual can include
STE error, finite-step curvature, and interactions; these experiments do not
uniquely assign the failure to one cause.

### Independent acceptance

After fitting selects one map, a separate validation set checks its actual
NLL change once. Acceptance requires validation mean plus two SE below zero.
A rejected map is preserved for analysis, but its export contains the baseline.
Validation does not choose the backtracking budget, and held-out evaluation
does not change the map or acceptance decision.

The recommended C4 variant uses 64 fitting windows and 16 independent
validation windows. Its operation can be summarized as:

```text
Construct the fixed FourOverSix baseline and E0M3 alternatives.
Collect task-gradient tile scores on 64 C4 fitting windows.
Keep tiles with mean + 2 SE < 0 and rank by predicted benefit.
For budget = 0.1, 0.05, ..., 0.00078125:
    Construct the prefix within the predicted-loss budget.
    Measure the combined map on fitting data.
    Stop when the paired improvement and prediction-ratio checks pass.
Validate that frozen map once on 16 separate C4 windows.
Export it if validation passes; otherwise export FourOverSix.
```

This avoids an exhaustive search over tile combinations, but it still needs
backward passes and a bounded number of joint forward evaluations. It is not
a calibration-free or purely analytical performance predictor.

## 4. Experimental design

The final domain study used native Transformers 5.16.1 implementations and
matched baselines for each model. It quantized the selected text linear weights
and their inputs to W4A4; the target's other native components, embeddings and
language-model head retained their existing behavior. The smaller-model panel
quantized non-head linear weights and inputs. Format decisions concerned
**weights**; activation quantization stayed fixed at FourOverSix throughout.

| Model | Calibration runs | Held-out Wiki windows | Held-out C4 documents | Math / code examples |
|---|---|---:|---:|---:|
| Qwen/Qwen3.8-27B | Seeds 20260912 and 20260913 | 127 | 256 | 128 / 128 |
| Qwen3-4B | Seed 20260918 | 128 | 256 | 128 / 128 |
| Llama-3.1-8B | Seed 20260918 | 123 | 256 | 128 / 128 |

Wiki calibration uses disjoint regions of the training split for fitting,
validation and diagnostic probes. Held-out Wiki uses the official raw
validation split. C4 calibration uses distinct training documents; each
qualifying document contributes one random 2,048-token window. Held-out C4
uses 256 documents from a separate validation shard, with recorded document
and token hashes. Calibration and evaluation C4 documents are hash-disjoint.

Math and code use GSM8K and MBPP reference text. They are **uncalibrated domain
stress tests of language-model loss**, not generated-answer accuracy, reasoning
success, or code pass@k. Their main metric weights examples equally; the raw
reports also contain token-weighted perplexity.

The two target seeds share held-out examples. Different model tokenizers change
Wiki window boundaries and can change which C4 documents satisfy the minimum
length condition. All policy comparisons are paired on identical examples
within a model. Native panel numbers must not be directly equated with older
results obtained using copied model implementations or different evaluation
splits.

We evaluated six selection variants:

| Rule | Fitting data / objective | Independent validation |
|---|---|---|
| Wiki | 64 Wiki windows, observed-token NLL | 16 Wiki windows |
| C4 | 64 C4 windows, observed-token NLL | 16 C4 windows |
| Mixed | 32 Wiki + 32 C4, pooled NLL scores and fitting loss | Pooled 8 + 8 windows |
| Consensus | 32 Wiki + 32 C4, stable negative scores and measured fitting improvement in each domain | Each 8-window domain separately |
| Teacher mixed | Same 32 + 32, teacher KL for scoring and fitting | Actual NLL on pooled 8 + 8 |
| Teacher consensus | Same 32 + 32, per-domain teacher-KL checks | Actual NLL in each domain |

Consensus ranks eligible tiles by their worst domain mean score. Its joint
fitting checks require a supported reduction and sufficient actual/predicted
agreement separately in both domains. A worst-domain score budget alone does
not bound the predicted change in every other domain.

Teacher variants were tested on the two smaller models, using cached logits
from their pristine, unquantized BF16 reference models. The fitting objective
was $KL(p_{teacher}\|q_{quantized})$. Validation still checked actual NLL,
so better imitation of the teacher could not override an NLL validation failure.
This follow-up was motivated by fitting-score diagnostics before new held-out
results were inspected.

The study contains 20 candidate maps and 80 candidate/domain evaluations.
All proposals and acceptance decisions were frozen before their held-out
evaluations. All heavy CPU work, GPU work, tests and numerical summaries ran
through Slurm on H100 allocations. No H200 or login-node heavy compute was used.

## 5. Results: what transfers

For these tables, **ΔPPL = candidate PPL − matched baseline PPL**. Negative is
better. An absolute reduction of 0.01 is practically meaningful. Statistical
support is reported separately through paired NLL mean ± two SE, transformed
to PPL units. Those intervals are descriptive, not simultaneous guarantees.

### C4 calibration gives the strongest overall coverage

All four C4 maps passed independent validation. Their held-out changes were:

| Model / seed | E0M3 tiles | Final budget | Wiki ΔPPL | C4 ΔPPL | Math ΔPPL | Code ΔPPL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B / 20260912 | 905 | 0.0125 | -0.1094 | -0.0500 | -0.0343 | -0.0462 |
| Qwen3.8-27B / 20260913 | 864 | 0.0125 | -0.0775 | -0.0532 | -0.0199 | -0.0352 |
| Qwen3-4B / 20260918 | 125 | 0.1000 | -1.9695 | -1.1781 | -0.4542 | -0.5216 |
| Llama-3.1-8B / 20260918 | 2,924 | 0.0250 | -0.0452 | -0.0546 | -0.0342 | +0.0103 |

For scale, the target's Wiki baseline was 7.6890 PPL and its C4 baseline was
10.2257. The first C4 map changed these to 7.5796 and 10.1758. Qwen3-4B changed
from 15.2160 to 13.2465 on Wiki and from 17.8187 to 16.6406 on C4.

All eight target cells and all four Qwen3-4B cells have intervals supporting
improvement. On Llama, Wiki and C4 improvements are supported, while math and
code are inconclusive. The Llama code interval is [-0.0565, +0.0778] PPL:
the +0.0103 mean is not a verified gain, but neither is it confirmed harm.

The 15/16 count describes point improvements of at least 0.01 PPL, not 15
independent statistical confirmations. The final budget also varies by model;
0.0125 should not be substituted for adaptive backtracking as a universal
constant.

On the target, there are 47,559,680 candidate weight tiles. Selecting 905 and
864 corresponds to roughly 0.0019% and 0.0018% of them. Very sparse changes can
therefore give useful gains; maximizing the fraction of E0M3 tiles is not the
objective.

### Wiki-only gains do not establish domain transfer

The complete target comparison shows why Wiki performance alone was insufficient:

| Rule | Wiki ΔPPL, seed 1 / seed 2 | C4 ΔPPL, seed 1 / seed 2 | Maps accepted |
|---|---|---|---:|
| Wiki | -0.3684 / -0.4088 | +0.0028 / +0.0064 | 2/2 |
| C4 | -0.1094 / -0.0775 | -0.0500 / -0.0532 | 2/2 |
| Mixed | -0.4509 / -0.4183 | -0.0026 / +0.0004 | 2/2 |
| Consensus | -0.1781 / -0.2092 | -0.0209 / -0.0353 | 1/2 |

Both Wiki-only C4 changes are inconclusive and neither meets the 0.01 gain
criterion. Mixed calibration mainly strengthens Wiki: equal sample counts did
not produce meaningful C4 gains on this model. In seed one, mixed beats the
Wiki map by a paired -0.01133 ± 0.00232 NLL on Wiki, but its C4 contrast with
the Wiki map is inconclusive. Pooling can improve the aggregate objective
without improving the weaker domain.

Consensus produces useful candidate changes in both target domains, but the
second map failed validation: its favorable means were too uncertain in eight
windows per domain. Its later held-out gains belong to the **candidate**, not
the exported baseline. This distinction matters when comparing procedures.

### Broad comparisons and the code counterexamples

| Rule | Candidate cases with ≥0.01 PPL gain | Cases with supported improvement | Cases with supported harm | Accepted maps |
|---|---:|---:|---:|---:|
| Wiki | 13/16 | 12/16 | 1/16 | 4/4 |
| C4 | 15/16 | 14/16 | 0/16 | 4/4 |
| Mixed | 13/16 | 13/16 | 0/16 | 4/4 |
| Consensus | 15/16 | 13/16 | 0/16 | 2/4 |
| Teacher mixed | 7/8 | 6/8 | 0/8 | 1/2 |
| Teacher consensus | 6/8 | 5/8 | 1/8 | 2/2 |

Teacher rows cover only the two smaller models and are not a matched 16-case
comparison. Counts include rejected candidates; an exported fallback has zero
change from baseline, not the candidate's measured gain. Repeated target seeds
also share evaluation data.

No tested procedure established useful improvement everywhere. Two clear
counterexamples are particularly informative:

- The accepted Wiki map on Llama increased code PPL by **0.1078**, with an
  interval of [+0.0404, +0.1760], despite improving Wiki, C4 and math.
- Accepted teacher-consensus on Qwen3-4B increased code PPL by **0.1850**, with
  an interval of [+0.0298, +0.3426], despite passing Wiki/C4 validation.

These are measured failures on an uncalibrated domain. They rule out treating
Wiki/C4 acceptance, or teacher fidelity, as an automatic certificate for code.

![Absolute perplexity changes for all candidates, with paired uncertainty and rejected exports marked by crosses.](results/task_sensitivity_domains/comparison.png)

## 6. What the diagnostics explain—and what they do not

### Useful selections depend on the calibration distribution

The target's Wiki and C4 maps shared only 10 tiles in seed one and 14 in seed
two, giving Jaccard overlaps near 0.009 and 0.012. However, the C4 maps improved
Wiki too. Low overlap therefore demonstrates different useful selections; it
does not prove that the domains require inherently conflicting optimal maps.

We also tested 15 selected tiles per target seed individually on reserved
eight-window probes in each domain. Point-sign agreement between forecast and
measurement was only about half, but almost all effects were inconclusive.
No probe established a stable wrong-sign effect. The small, deliberately
selected probe sample cannot estimate a global tile error rate or settle the
mechanism of domain conflict.

### Score alignment and noise differ across models

Using 64 fitting windows per domain, Wiki/C4 score-vector cosines were about
0.038 and 0.033 for the two target seeds, 0.795 for Qwen3-4B, and 0.276 for
Llama. Within-C4 split-half cosines were approximately 0.219, 0.961 and 0.204
for those model groups. Weak cross-domain alignment on the target coexists
with considerable within-domain sampling variability.

This helps explain why a single scalar summary of score disagreement cannot
decide whether a format change will generalize. It also makes the useful
four-domain transfer on Qwen3-4B less surprising, but it is an association,
not a causal proof or a new validated selector.

### Teacher KL did not provide a universal repair

For observed-token NLL, the logit gradient is $q-\mathrm{onehot}(y)$.
For teacher KL it is $q-p$. Removing the observed-token residual suggested
a hypothesis: teacher probabilities might provide a less noisy measure of
quantization damage.

The result was model-dependent. On Qwen3-4B at matched 32-window budgets,
cross-domain cosine fell from 0.775 for NLL to 0.350 for teacher KL. The C4
ratio $\sum SE_b^2/\sum\mu_b^2$ rose from 0.051 to 0.496. On Llama,
alignment improved and this noise-scale estimate decreased, but accuracy did
not become uniformly better. The ratio is descriptive, not a certified
fraction of noise. Together with the code regression, these measurements do
not support replacing actual-NLL selection with teacher KL as the default.

## 7. What “universal” can reasonably mean

The experiments support reusing the **procedure**: construct candidates at the
deployment baseline, estimate task sensitivity, limit the proposed change,
measure its actual effect, and validate independently. They do not support
reusing the same tile identities, a fixed tile quota, or one final budget.

An unconditional improvement for every possible text/label distribution is
also too strong a requirement. If two normalized next-token distributions
differ, at least one token has lower probability under the changed model.
A distribution concentrated on that context and token would have worse log
loss. This does not prevent broad practical gains on real workloads; it
clarifies why evidence must refer to specified domains.

If a fixed map's **true expected NLL change** is nonpositive separately in
each declared domain, its change is nonpositive on any fixed mixture of those
domains by linearity of expectation. Our finite-sample two-SE checks do not
prove that premise, and the identity does not cover a new domain outside the
mixture. The [guarantee analysis](results/task_sensitivity_four_over_six/GUARANTEE.md)
discusses the assumptions needed for stronger certification.

## 8. Implementation, validation, and reproducibility

The final experiment uses fake quantization to evaluate accuracy with native
model behavior. It does not benchmark a packed mixed-format inference kernel,
latency, memory bandwidth, long-context generation, or task decoding accuracy.
Small perplexity gains should not be converted into throughput or reasoning
claims.

Model revisions, dataset revisions, token fingerprints, quantized module
shapes, maps, validation decisions, and per-example losses are recorded in
the raw reports. Native baseline checks reproduced the required reference
losses. Structural tests covered moment pooling, consensus eligibility and
fallback behavior; teacher-loss checks covered its gradient and normalization.
All summary calculations and plot generation completed on Slurm allocations.

An attempted scoring placement optimization was rejected by an exact score
comparison and disabled. The reported experiments use the original scoring
backend. Intermediate failures are documented rather than mixed into the
final numerical results.

| Artifact | Purpose |
|---|---|
| [Final numerical report](results/task_sensitivity_domains/REPORT.md) | All 80 candidate/domain cells, intervals, contrasts and acceptance decisions |
| [Summary CSV](results/task_sensitivity_domains/summary.csv) | Absolute and relative PPL changes, uncertainty and the 0.01 criterion |
| [Hypotheses](results/task_sensitivity_domains/HYPOTHESES.md) | Motivation and competing explanations declared during experiment design |
| [Execution notes](results/task_sensitivity_domains/RUN_NOTES.md) | Provenance checks, unsuccessful setup attempts and scheduling changes |
| [Target runner](run_domain_sensitivity.py) | Two-seed calibration, finite-step checks and held-out evaluation |
| [Model panel](run_domain_panel.py) | Matched native model transfer experiments |
| [Teacher follow-up](run_domain_teacher.py) | Teacher-KL hypotheses with actual-NLL validation |
| [Summary generator](summarize_domain_sensitivity.py) | Paired tables, CSVs and figure |

The accepted C4 type maps are under `results/task_sensitivity_domains/`:
`seed20260912/c4_export.json`, `seed20260913/c4_export.json`,
`panel/qwen3-4b/c4_export.json`, and
`panel/llama-3.1-8b-local/c4_export.json`. They must be applied to the same
pristine source weights, with the recorded activation configuration and the
appropriate native or generic map loader. Rejected candidates are retained
separately for analysis. Large score checkpoints are gitignored; the committed
JSON reports and scripts preserve the experimental record and regeneration
procedure.

## 9. Practical conclusion and next experiments

For the tested setting, C4 task-gradient selection with backtracking is the
best starting point when broad transfer and acceptance rate matter. It is not
the winner on every individual metric: mixed calibration yields larger target
Wiki gains, and consensus can be stronger on Qwen3-4B. The common improvement
over a raw selector is to respect the actual quantized baseline and verify
the combined finite change.

The next unresolved problem is reliable coverage of domains such as code.
A useful follow-up would add code to independently split calibration and
validation data, preserve fresh held-out tests, and size validation to resolve
0.01 PPL differences. Larger validation budgets may also reduce the rejection
of useful consensus candidates. These are proposed experiments, not fixes
already established by this study.

MixFP4 works here because a small number of carefully selected E0M3 switches
can improve the quantized network's task loss. The evidence favors **adaptive,
measured selection** over choosing a format from weight shape alone or assuming
that gains on one dataset will carry over everywhere.
