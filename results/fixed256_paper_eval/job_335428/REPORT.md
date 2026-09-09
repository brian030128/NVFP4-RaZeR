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
