# Main tables: 8x64 and 16x64

### Perplexity (NativeLinear (c); BF16 as loaded)

| model | corpus | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | WikiText-2 | 6.2403 | 6.9338 | 6.8706 | 6.7877 | 6.7827 |
| Llama-3.1-8B | C4 | 8.9579 | 9.9284 | 9.8257 | 9.6774 | 9.6827 |
| Mistral-7B-v0.3 | WikiText-2 | 5.3182 | 5.5506 | 5.5224 | 5.4868 | 5.4899 |
| Mistral-7B-v0.3 | C4 | 7.8306 | 8.0895 | 8.0623 | 8.0215 | 8.0323 |
| Phi-4 | WikiText-2 | 6.4615 | 6.6934 | 6.6617 | 6.6078 | 6.6133 |
| Phi-4 | C4 | 10.3098 | 10.5795 | 10.5441 | 10.4979 | 10.5047 |
| Qwen3.8-27B | WikiText-2 | 7.0509 | 7.5506 | 7.3215 | 7.1074 | 7.1232 |
| Qwen3.8-27B | C4 | 9.8935 | 10.2185 | 10.1869 | 10.1235 | 10.1276 |

### Paired ΔNLL, nats per token (= Δ log PPL), ± 2 SE over windows; * = |Δ| > 2 SE

| model | corpus | windows | FlipQuant (ours) 8x64 − FourOverSix | FlipQuant (ours) 8x64 − NVFP4 | FlipQuant (ours) 16x64 − FourOverSix | FlipQuant (ours) 16x64 − NVFP4 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | WikiText-2 | 141 | -0.0121 ± 0.0017 * | -0.0213 ± 0.0023 * | -0.0129 ± 0.0018 * | -0.0220 ± 0.0023 * |
| Llama-3.1-8B | C4 | 256 | -0.0152 ± 0.0030 * | -0.0256 ± 0.0045 * | -0.0147 ± 0.0031 * | -0.0251 ± 0.0046 * |
| Mistral-7B-v0.3 | WikiText-2 | 163 | -0.0065 ± 0.0009 * | -0.0116 ± 0.0013 * | -0.0059 ± 0.0009 * | -0.0110 ± 0.0012 * |
| Mistral-7B-v0.3 | C4 | 256 | -0.0051 ± 0.0010 * | -0.0084 ± 0.0009 * | -0.0037 ± 0.0017 * | -0.0071 ± 0.0014 * |
| Phi-4 | WikiText-2 | 141 | -0.0081 ± 0.0015 * | -0.0129 ± 0.0020 * | -0.0073 ± 0.0014 * | -0.0120 ± 0.0020 * |
| Phi-4 | C4 | 256 | -0.0044 ± 0.0009 * | -0.0077 ± 0.0011 * | -0.0037 ± 0.0009 * | -0.0071 ± 0.0011 * |
| Qwen3.8-27B | WikiText-2 | 145 | -0.0297 ± 0.0061 * | -0.0605 ± 0.0092 * | -0.0275 ± 0.0052 * | -0.0583 ± 0.0091 * |
| Qwen3.8-27B | C4 | 256 | -0.0062 ± 0.0010 * | -0.0093 ± 0.0013 * | -0.0058 ± 0.0010 * | -0.0089 ± 0.0013 * |

### Downstream accuracy, % (lm-eval 0.4.11; MMLU 5-shot, the others 0-shot; acc_norm, MMLU acc)

| model | task | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | MMLU (5-shot) | 65.31 | 62.51 | 62.98 | 63.47 | 63.24 |
| Llama-3.1-8B | ARC-C | 54.78 | 52.30 | 53.84 | 53.75 | 53.92 |
| Llama-3.1-8B | ARC-E | 82.53 | 78.32 | 80.43 | 80.89 | 81.10 |
| Llama-3.1-8B | HellaSwag | 79.34 | 77.86 | 78.16 | 78.41 | 78.38 |
| Llama-3.1-8B | PIQA | 81.12 | 79.38 | 80.47 | 80.20 | 80.58 |
| Llama-3.1-8B | mean of 5 | 72.62 | 70.08 | 71.18 | 71.35 | 71.44 |
| Mistral-7B-v0.3 | MMLU (5-shot) | 62.42 | 59.96 | 60.39 | 59.93 | 60.48 |
| Mistral-7B-v0.3 | ARC-C | 54.69 | 51.96 | 53.07 | 52.82 | 52.73 |
| Mistral-7B-v0.3 | ARC-E | 80.13 | 78.28 | 78.96 | 79.04 | 79.17 |
| Mistral-7B-v0.3 | HellaSwag | 80.62 | 80.12 | 80.06 | 80.22 | 80.24 |
| Mistral-7B-v0.3 | PIQA | 81.77 | 81.12 | 81.01 | 81.18 | 81.61 |
| Mistral-7B-v0.3 | mean of 5 | 71.93 | 70.29 | 70.70 | 70.64 | 70.85 |
| Phi-4 | MMLU (5-shot) | 80.29 | 79.11 | 79.23 | 79.30 | 79.15 |
| Phi-4 | ARC-C | 56.14 | 54.10 | 54.86 | 54.52 | 55.12 |
| Phi-4 | ARC-E | 72.77 | 73.23 | 71.89 | 72.31 | 72.26 |
| Phi-4 | HellaSwag | 81.95 | 80.86 | 81.11 | 81.11 | 81.25 |
| Phi-4 | PIQA | 81.12 | 80.85 | 80.69 | 80.79 | 80.69 |
| Phi-4 | mean of 5 | 74.45 | 73.63 | 73.56 | 73.61 | 73.69 |
| Qwen3.8-27B | MMLU (5-shot) | 82.57 | 81.25 | 81.29 | 82.47 | 82.38 |
| Qwen3.8-27B | ARC-C | 58.79 | 60.92 | 59.30 | 59.98 | 60.15 |
| Qwen3.8-27B | ARC-E | 73.02 | 76.47 | 74.58 | 75.72 | 76.64 |
| Qwen3.8-27B | HellaSwag | 82.89 | 82.44 | 82.26 | 82.31 | 82.08 |
| Qwen3.8-27B | PIQA | 81.50 | 81.39 | 81.28 | 81.18 | 81.28 |
| Qwen3.8-27B | mean of 5 | 75.76 | 76.50 | 75.74 | 76.33 | 76.51 |

### Paired accuracy differences, percentage points, ± 2 SE; * = |Δ| > 2 SE

| model | task | examples | FlipQuant (ours) 8x64 − FourOverSix | FlipQuant (ours) 8x64 − NVFP4 | FlipQuant (ours) 16x64 − FourOverSix | FlipQuant (ours) 16x64 − NVFP4 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | MMLU (5-shot) | 14042 | +0.49 ± 0.63 | +0.97 ± 0.64 * | +0.26 ± 0.62 | +0.73 ± 0.65 * |
| Llama-3.1-8B | ARC-C | 1172 | -0.09 ± 1.72 | +1.45 ± 1.88 | +0.09 ± 1.81 | +1.62 ± 1.83 |
| Llama-3.1-8B | ARC-E | 2376 | +0.46 ± 1.05 | +2.57 ± 1.18 * | +0.67 ± 1.00 | +2.78 ± 1.15 * |
| Llama-3.1-8B | HellaSwag | 10042 | +0.25 ± 0.36 | +0.55 ± 0.40 * | +0.22 ± 0.37 | +0.52 ± 0.39 * |
| Llama-3.1-8B | PIQA | 1838 | -0.27 ± 1.00 | +0.82 ± 1.11 | +0.11 ± 1.03 | +1.20 ± 1.13 * |
| Llama-3.1-8B | mean of 5 |  | +0.17 ± 0.47 | +1.27 ± 0.52 * | +0.27 ± 0.48 | +1.37 ± 0.51 * |
| Mistral-7B-v0.3 | MMLU (5-shot) | 14042 | -0.46 ± 0.54 | -0.02 ± 0.60 | +0.09 ± 0.56 | +0.52 ± 0.61 |
| Mistral-7B-v0.3 | ARC-C | 1172 | -0.26 ± 1.40 | +0.85 ± 1.47 | -0.34 ± 1.62 | +0.77 ± 1.57 |
| Mistral-7B-v0.3 | ARC-E | 2376 | +0.08 ± 0.81 | +0.76 ± 0.91 | +0.21 ± 0.88 | +0.88 ± 0.89 |
| Mistral-7B-v0.3 | HellaSwag | 10042 | +0.16 ± 0.29 | +0.10 ± 0.33 | +0.18 ± 0.30 | +0.12 ± 0.33 |
| Mistral-7B-v0.3 | PIQA | 1838 | +0.16 ± 0.84 | +0.05 ± 0.85 | +0.60 ± 0.88 | +0.49 ± 0.79 |
| Mistral-7B-v0.3 | mean of 5 |  | -0.06 ± 0.38 | +0.35 ± 0.41 | +0.15 ± 0.43 | +0.56 ± 0.42 * |
| Phi-4 | MMLU (5-shot) | 14042 | +0.06 ± 0.39 | +0.19 ± 0.42 | -0.09 ± 0.39 | +0.04 ± 0.41 |
| Phi-4 | ARC-C | 1172 | -0.34 ± 1.21 | +0.43 ± 1.42 | +0.26 ± 1.24 | +1.02 ± 1.34 |
| Phi-4 | ARC-E | 2376 | +0.42 ± 0.79 | -0.93 ± 0.94 | +0.38 ± 0.76 | -0.97 ± 0.93 * |
| Phi-4 | HellaSwag | 10042 | +0.00 ± 0.31 | +0.25 ± 0.36 | +0.14 ± 0.31 | +0.39 ± 0.36 * |
| Phi-4 | PIQA | 1838 | +0.11 ± 0.84 | -0.05 ± 0.96 | +0.00 ± 0.87 | -0.16 ± 0.88 |
| Phi-4 | mean of 5 |  | +0.05 ± 0.35 | -0.02 ± 0.41 | +0.14 ± 0.35 | +0.06 ± 0.39 |
| Qwen3.8-27B | MMLU (5-shot) | 14042 | +1.18 ± 0.44 * | +1.22 ± 0.46 * | +1.09 ± 0.44 * | +1.13 ± 0.47 * |
| Qwen3.8-27B | ARC-C | 1172 | +0.68 ± 1.41 | -0.94 ± 1.65 | +0.85 ± 1.41 | -0.77 ± 1.72 |
| Qwen3.8-27B | ARC-E | 2376 | +1.14 ± 0.94 * | -0.76 ± 1.00 | +2.06 ± 0.97 * | +0.17 ± 1.03 |
| Qwen3.8-27B | HellaSwag | 10042 | +0.05 ± 0.30 | -0.13 ± 0.34 | -0.19 ± 0.30 | -0.37 ± 0.34 * |
| Qwen3.8-27B | PIQA | 1838 | -0.11 ± 0.81 | -0.22 ± 0.92 | +0.00 ± 0.84 | -0.11 ± 0.92 |
| Qwen3.8-27B | mean of 5 |  | +0.59 ± 0.39 * | -0.16 ± 0.44 | +0.76 ± 0.40 * | +0.01 ± 0.46 |

MMLU questions whose top two choices tie in log-likelihood, % (lm-eval computes them from BF16 logits and its argmax takes the earlier choice; every policy is scored the same way):

| model | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64 |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 5.8 | 5.6 | 6.7 | 6.1 | 6.2 |
| Mistral-7B-v0.3 | 6.4 | 6.3 | 7.4 | 6.7 | 7.0 |
| Phi-4 | 0.9 | 1.1 | 1.0 | 1.1 | 1.0 |
| Qwen3.8-27B | 1.0 | 1.0 | 1.1 | 1.2 | 1.2 |

### Prefill latency, CUDA graph (primary)

#### Llama-3.1-8B, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 16.41 | 7.55 | 7.46 | 8.97 | 8.88 | 9.30 | 9.21 | 7.67 | 7.58 |
| 1x256 | 22.18 | 9.49 | 9.40 | 10.60 | 10.51 | 11.02 | 10.94 | 9.63 | 9.53 |
| 1x512 | 38.65 | 15.17 | 15.22 | 15.71 | 15.60 | 17.09 | 17.04 | 16.05 | 15.79 |
| 1x1024 | 71.07 | 31.39 | 31.57 | 31.02 | 31.17 | 32.72 | 33.00 | 32.02 | 32.31 |
| 1x2048 | 133.32 | 62.14 | 62.40 | 62.32 | 62.59 | 65.62 | 65.87 | 63.28 | 63.55 |
| 1x4096 | 272.35 | 134.44 | 134.91 | 134.48 | 135.08 | 141.28 | 141.84 | 136.77 | 137.50 |
| 1x8192 | 604.73 | 324.11 | 324.64 | 324.91 | 326.15 | 337.97 | 338.35 | 328.87 | 329.40 |
| 4x2048 | 556.90 | 276.53 | 277.11 | 277.54 | 278.16 | 290.41 | 290.77 | 281.10 | 281.84 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1x128 | +23.5 % [+23.4, +23.5] | +23.1 % [+23.0, +23.2] | +3.8 % [+3.8, +3.9] | +3.7 % [+3.6, +3.7] | +1.5 % [+1.5, +1.7] | +1.6 % [+1.5, +1.6] |
| 1x256 | +16.4 % [+16.3, +16.4] | +16.2 % [+16.1, +16.5] | +4.1 % [+4.0, +4.1] | +4.0 % [+3.9, +4.3] | +1.4 % [+1.4, +1.5] | +1.5 % [+1.4, +1.6] |
| 1x512 | +12.0 % [+7.6, +12.0] | +12.2 % [+9.4, +13.6] | +9.1 % [+5.9, +9.8] | +8.3 % [+7.0, +10.6] | +3.5 % [+1.6, +4.3] | +5.7 % [+1.3, +7.5] |
| 1x1024 | +4.6 % [+4.0, +4.8] | +4.3 % [+4.1, +5.5] | +5.8 % [+5.6, +6.3] | +5.5 % [+5.2, +6.5] | +2.4 % [+1.9, +2.4] | +2.4 % [+1.8, +3.2] |
| 1x2048 | +5.6 % [+5.0, +5.9] | +5.6 % [+5.2, +6.5] | +5.1 % [+4.9, +5.6] | +5.3 % [+5.1, +6.1] | +1.8 % [+1.7, +2.0] | +2.0 % [+1.4, +2.5] |
| 1x4096 | +5.2 % [+4.5, +5.2] | +5.0 % [+4.7, +5.8] | +4.9 % [+4.6, +5.4] | +5.1 % [+4.8, +5.8] | +2.0 % [+1.7, +2.0] | +1.7 % [+1.3, +2.3] |
| 1x8192 | +4.2 % [+3.9, +4.5] | +4.3 % [+3.9, +5.0] | +3.8 % [+3.6, +4.3] | +4.0 % [+3.8, +4.7] | +1.4 % [+1.3, +1.6] | +1.6 % [+1.2, +1.8] |
| 4x2048 | +5.0 % [+4.7, +5.2] | +5.0 % [+4.8, +5.7] | +4.6 % [+4.4, +5.0] | +4.7 % [+4.5, +5.4] | +1.7 % [+1.6, +1.9] | +1.6 % [+1.5, +2.0] |

#### Mistral-7B-v0.3, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 15.92 | 7.08 | 6.99 | 8.50 | 8.41 | 8.84 | 8.75 | 7.20 | 7.10 |
| 1x256 | 21.07 | 8.89 | 8.80 | 10.00 | 9.92 | 10.44 | 10.36 | 9.01 | 8.92 |
| 1x512 | 36.91 | 13.57 | 13.35 | 13.95 | 13.73 | 15.08 | 14.79 | 14.05 | 13.91 |
| 1x1024 | 67.80 | 28.07 | 28.35 | 27.58 | 27.83 | 29.66 | 29.95 | 28.98 | 29.15 |
| 1x2048 | 126.98 | 56.11 | 56.34 | 56.29 | 56.59 | 59.44 | 59.78 | 57.25 | 57.46 |
| 1x4096 | 259.90 | 123.01 | 123.49 | 123.04 | 123.65 | 129.87 | 130.47 | 125.16 | 125.59 |
| 1x8192 | 579.28 | 300.76 | 300.93 | 301.45 | 302.11 | 314.60 | 315.22 | 305.10 | 305.57 |
| 4x2048 | 532.00 | 253.18 | 253.40 | 254.10 | 254.48 | 266.93 | 267.74 | 257.53 | 258.07 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1x128 | +25.2 % [+25.2, +25.2] | +24.8 % [+24.8, +24.9] | +4.0 % [+4.0, +4.1] | +4.0 % [+3.9, +4.0] | +1.6 % [+1.6, +1.7] | +1.6 % [+1.6, +1.7] |
| 1x256 | +17.7 % [+17.6, +17.7] | +17.4 % [+17.4, +17.5] | +4.4 % [+4.4, +4.5] | +4.4 % [+4.4, +4.4] | +1.3 % [+1.3, +1.3] | +1.3 % [+1.3, +1.4] |
| 1x512 | +10.8 % [+10.0, +12.6] | +11.1 % [+9.9, +12.4] | +8.0 % [+7.7, +8.6] | +8.1 % [+7.2, +8.8] | +4.1 % [+1.6, +4.7] | +3.5 % [+2.2, +5.6] |
| 1x1024 | +5.5 % [+5.1, +6.6] | +5.9 % [+5.2, +6.2] | +7.6 % [+6.4, +8.1] | +7.9 % [+7.1, +8.0] | +2.8 % [+2.4, +3.0] | +3.4 % [+1.9, +4.3] |
| 1x2048 | +6.1 % [+5.7, +6.3] | +6.1 % [+5.4, +6.3] | +5.7 % [+5.2, +6.0] | +5.7 % [+5.4, +6.1] | +2.0 % [+1.8, +2.1] | +2.0 % [+1.4, +2.8] |
| 1x4096 | +5.6 % [+5.4, +5.8] | +5.6 % [+5.0, +6.0] | +5.5 % [+5.2, +5.9] | +5.6 % [+5.3, +6.1] | +1.7 % [+1.5, +1.9] | +1.7 % [+1.3, +2.4] |
| 1x8192 | +4.7 % [+4.6, +4.8] | +4.6 % [+4.1, +4.9] | +4.3 % [+4.1, +4.6] | +4.3 % [+4.2, +4.7] | +1.5 % [+1.1, +1.6] | +1.4 % [+1.0, +1.9] |
| 4x2048 | +5.7 % [+5.5, +5.7] | +5.5 % [+5.1, +5.9] | +5.2 % [+4.8, +5.5] | +5.1 % [+5.0, +5.5] | +1.8 % [+1.5, +1.9] | +1.8 % [+1.4, +2.4] |

#### Phi-4, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 28.28 | 11.47 | 11.38 | 12.44 | 12.34 | 12.78 | 12.69 | 11.57 | 11.47 |
| 1x256 | 37.95 | 15.22 | 14.98 | 15.38 | 15.20 | 16.08 | 15.82 | 15.49 | 15.38 |
| 1x512 | 68.38 | 27.84 | 27.60 | 28.07 | 27.85 | 29.51 | 29.58 | 28.56 | 28.26 |
| 1x1024 | 128.25 | 54.83 | 55.28 | 55.10 | 55.60 | 58.22 | 58.71 | 56.06 | 56.40 |
| 1x2048 | 248.21 | 110.90 | 111.50 | 111.31 | 111.94 | 117.65 | 118.25 | 113.08 | 113.57 |
| 1x4096 | 512.87 | 237.26 | 237.59 | 238.23 | 238.55 | 250.56 | 250.87 | 241.52 | 241.52 |
| 1x8192 | 1119.36 | 559.24 | 560.76 | 562.05 | 563.82 | 586.46 | 587.93 | 568.02 | 569.36 |
| 4x2048 | 1043.46 | 486.98 | 488.21 | 489.69 | 491.33 | 514.23 | 515.63 | 495.81 | 496.49 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1x128 | +11.6 % [+11.6, +11.6] | +11.4 % [+11.2, +11.4] | +2.9 % [+2.8, +2.9] | +2.7 % [+2.5, +2.9] | +0.8 % [+0.8, +0.9] | +0.8 % [+0.8, +0.9] |
| 1x256 | +5.9 % [+5.4, +6.3] | +5.3 % [+5.0, +7.8] | +4.2 % [+3.4, +5.3] | +4.4 % [+4.0, +5.0] | +2.7 % [+0.7, +3.2] | +1.2 % [+0.8, +3.9] |
| 1x512 | +7.1 % [+5.5, +7.3] | +6.0 % [+5.1, +8.2] | +6.1 % [+5.1, +6.5] | +5.1 % [+4.8, +7.3] | +2.4 % [+2.2, +2.7] | +2.5 % [+1.6, +2.8] |
| 1x1024 | +6.0 % [+5.5, +6.4] | +6.1 % [+5.4, +6.7] | +5.5 % [+5.1, +5.7] | +5.9 % [+5.5, +5.9] | +2.1 % [+1.8, +2.1] | +2.3 % [+1.4, +2.7] |
| 1x2048 | +6.0 % [+5.5, +6.4] | +6.2 % [+5.4, +6.6] | +5.5 % [+5.4, +5.8] | +5.7 % [+5.7, +6.1] | +1.9 % [+1.7, +2.0] | +2.1 % [+1.2, +2.2] |
| 1x4096 | +5.7 % [+5.2, +5.8] | +5.6 % [+5.1, +6.1] | +5.2 % [+4.9, +5.3] | +5.3 % [+5.1, +5.5] | +1.7 % [+1.6, +1.9] | +1.9 % [+1.2, +2.1] |
| 1x8192 | +4.9 % [+4.5, +5.1] | +4.9 % [+4.3, +5.3] | +4.3 % [+4.1, +4.4] | +4.3 % [+4.2, +4.7] | +1.6 % [+1.4, +1.7] | +1.6 % [+1.0, +1.9] |
| 4x2048 | +5.7 % [+5.2, +5.8] | +5.6 % [+5.0, +6.0] | +4.9 % [+4.7, +5.1] | +5.0 % [+4.9, +5.3] | +1.7 % [+1.6, +1.8] | +1.8 % [+1.2, +2.2] |

#### Qwen3.8-27B, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 86.47 | 57.79 | 57.71 | 60.88 | 60.78 | 61.46 | 61.29 | 58.01 | 57.81 |
| 1x256 | 115.81 | 73.23 | 72.94 | 75.35 | 75.00 | 76.52 | 76.15 | 73.76 | 73.34 |
| 1x512 | 184.01 | 109.14 | 108.64 | 109.71 | 109.21 | 112.51 | 111.92 | 110.30 | 109.86 |
| 1x1024 | 329.66 | 193.72 | 194.12 | 194.45 | 195.10 | 199.86 | 200.60 | 195.63 | 196.29 |
| 1x2048 | 637.21 | 380.52 | 380.75 | 381.33 | 382.08 | 392.84 | 392.77 | 384.50 | 385.23 |
| 1x4096 | 1307.62 | 808.22 | 807.03 | 808.51 | 808.16 | 830.68 | 830.39 | 815.23 | 814.91 |
| 1x8192 | 2815.62 | 1812.21 | 1812.03 | 1811.73 | 1814.77 | 1855.82 | 1857.50 | 1825.57 | 1826.12 |
| 4x2048 | 2701.11 | 1677.29 | 1677.30 | 1679.18 | 1681.01 | 1722.99 | 1724.31 | 1691.30 | 1693.84 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1x128 | +6.2 % [+6.1, +6.2] | +6.3 % [+6.1, +6.6] | +0.9 % [+0.6, +1.0] | +0.9 % [+0.9, +1.0] | +0.2 % [-0.0, +0.2] | +0.4 % [+0.2, +0.4] |
| 1x256 | +4.4 % [+4.2, +4.5] | +4.5 % [+4.5, +4.7] | +1.5 % [+1.5, +1.6] | +1.6 % [+1.5, +1.7] | +0.6 % [+0.4, +0.7] | +0.7 % [+0.6, +0.8] |
| 1x512 | +3.0 % [+2.7, +3.1] | +3.1 % [+2.9, +3.3] | +2.5 % [+2.3, +2.7] | +2.6 % [+2.3, +2.8] | +1.1 % [+0.9, +1.2] | +1.1 % [+0.9, +1.3] |
| 1x1024 | +3.3 % [+2.8, +3.8] | +3.2 % [+3.1, +3.4] | +2.8 % [+2.5, +2.9] | +2.9 % [+2.8, +3.0] | +1.2 % [+0.8, +1.4] | +1.0 % [+0.9, +1.1] |
| 1x2048 | +3.3 % [+3.0, +3.4] | +3.2 % [+3.1, +3.3] | +3.0 % [+2.7, +3.0] | +2.9 % [+2.9, +3.1] | +1.1 % [+1.1, +1.4] | +1.0 % [+1.0, +1.0] |
| 1x4096 | +2.9 % [+2.6, +3.0] | +2.9 % [+2.6, +2.9] | +2.7 % [+2.4, +2.8] | +2.8 % [+2.7, +3.0] | +1.0 % [+0.8, +1.1] | +0.9 % [+0.7, +1.0] |
| 1x8192 | +2.5 % [+2.3, +2.7] | +2.4 % [+2.3, +2.5] | +2.3 % [+2.2, +2.5] | +2.4 % [+2.3, +2.6] | +0.8 % [+0.7, +1.0] | +0.7 % [+0.7, +0.8] |
| 4x2048 | +2.8 % [+2.6, +2.9] | +2.8 % [+2.7, +2.8] | +2.6 % [+2.4, +2.7] | +2.7 % [+2.5, +2.7] | +1.0 % [+0.8, +1.1] | +0.8 % [+0.8, +1.0] |

### GEMM latency (primary; deviation 2: isolated launches, cold weights)

#### Llama-3.1-8B: GEMM kernel time per forward, µs (every quantized text Linear; isolated launches, cold weights (distinct weight copies >= 4x L2 plus a 512 MiB read-flush before every call), CUPTI device time, median of 3 rounds x 30 repetitions in a rotated order)

| T | stock wA | stock wB | FlipQuant (ours) 16x64, typical | FlipQuant (ours) 16x64, worst | FlipQuant (ours) 8x64, typical | FlipQuant (ours) 8x64, worst |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 3530 | 4991 | 3689 | 3691 | 5490 | 5546 |
| 256 | 4542 | 5750 | 4687 | 4711 | 6383 | 6464 |
| 512 | 7006 | 7369 | 7396 | 7474 | 8545 | 8665 |
| 1024 | 13548 | 13590 | 13980 | 13996 | 14773 | 14873 |
| 2048 | 23381 | 23241 | 24226 | 24383 | 25618 | 25866 |
| 4096 | 45939 | 45622 | 47599 | 47894 | 49995 | 50469 |
| 8192 | 87666 | 87212 | 91048 | 91522 | 96509 | 97558 |

Overheads against the same-placement stock kernel (and 8x64 against stock wA):

| T | FlipQuant (ours) 16x64, typical vs stock wA | FlipQuant (ours) 16x64, worst vs stock wA | FlipQuant (ours) 8x64, typical vs stock wB | FlipQuant (ours) 8x64, worst vs stock wB | FlipQuant (ours) 8x64, typical vs stock wA | FlipQuant (ours) 8x64, worst vs stock wA |
|---|---:|---:|---:|---:|---:|---:|
| 128 | +4.5 % | +4.6 % | +10.0 % | +11.1 % | +55.5 % | +57.1 % |
| 256 | +3.2 % | +3.7 % | +11.0 % | +12.4 % | +40.5 % | +42.3 % |
| 512 | +5.6 % | +6.7 % | +16.0 % | +17.6 % | +22.0 % | +23.7 % |
| 1024 | +3.2 % | +3.3 % | +8.7 % | +9.4 % | +9.0 % | +9.8 % |
| 2048 | +3.6 % | +4.3 % | +10.2 % | +11.3 % | +9.6 % | +10.6 % |
| 4096 | +3.6 % | +4.3 % | +9.6 % | +10.6 % | +8.8 % | +9.9 % |
| 8192 | +3.9 % | +4.4 % | +10.7 % | +11.9 % | +10.1 % | +11.3 % |

The activation quantizer per forward, µs (isolated launches; FourOverSix and NVFP4 share the stock GEMM):

| T | NVFP4 quantizer | FourOverSix quantizer | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 799 | 714 | -10.6 % |
| 256 | 960 | 879 | -8.4 % |
| 512 | 1589 | 1446 | -9.0 % |
| 1024 | 2441 | 2760 | +13.1 % |
| 2048 | 4253 | 4597 | +8.1 % |
| 4096 | 8517 | 9320 | +9.4 % |
| 8192 | 17057 | 18108 | +6.2 % |

Quantizer launches net of the reuse measured in step 05.

#### Mistral-7B-v0.3: GEMM kernel time per forward, µs (every quantized text Linear; isolated launches, cold weights (distinct weight copies >= 4x L2 plus a 512 MiB read-flush before every call), CUPTI device time, median of 3 rounds x 30 repetitions in a rotated order)

| T | stock wA | stock wB | FlipQuant (ours) 16x64, typical | FlipQuant (ours) 16x64, worst | FlipQuant (ours) 8x64, typical | FlipQuant (ours) 8x64, worst |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 3532 | 4992 | 3688 | 3696 | 5492 | 5550 |
| 256 | 4544 | 5749 | 4690 | 4718 | 6372 | 6445 |
| 512 | 7017 | 7376 | 7393 | 7456 | 8572 | 8663 |
| 1024 | 13571 | 13647 | 13992 | 13986 | 14807 | 14850 |
| 2048 | 23308 | 23208 | 24339 | 24440 | 25563 | 25774 |
| 4096 | 46132 | 45796 | 47736 | 47933 | 50565 | 50882 |
| 8192 | 88048 | 87598 | 91512 | 92013 | 96956 | 97865 |

Overheads against the same-placement stock kernel (and 8x64 against stock wA):

| T | FlipQuant (ours) 16x64, typical vs stock wA | FlipQuant (ours) 16x64, worst vs stock wA | FlipQuant (ours) 8x64, typical vs stock wB | FlipQuant (ours) 8x64, worst vs stock wB | FlipQuant (ours) 8x64, typical vs stock wA | FlipQuant (ours) 8x64, worst vs stock wA |
|---|---:|---:|---:|---:|---:|---:|
| 128 | +4.4 % | +4.6 % | +10.0 % | +11.2 % | +55.5 % | +57.1 % |
| 256 | +3.2 % | +3.8 % | +10.8 % | +12.1 % | +40.2 % | +41.8 % |
| 512 | +5.3 % | +6.3 % | +16.2 % | +17.5 % | +22.2 % | +23.4 % |
| 1024 | +3.1 % | +3.1 % | +8.5 % | +8.8 % | +9.1 % | +9.4 % |
| 2048 | +4.4 % | +4.9 % | +10.1 % | +11.1 % | +9.7 % | +10.6 % |
| 4096 | +3.5 % | +3.9 % | +10.4 % | +11.1 % | +9.6 % | +10.3 % |
| 8192 | +3.9 % | +4.5 % | +10.7 % | +11.7 % | +10.1 % | +11.2 % |

The activation quantizer per forward, µs (isolated launches; FourOverSix and NVFP4 share the stock GEMM):

| T | NVFP4 quantizer | FourOverSix quantizer | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 799 | 714 | -10.7 % |
| 256 | 959 | 881 | -8.2 % |
| 512 | 1590 | 1449 | -8.9 % |
| 1024 | 2443 | 2762 | +13.1 % |
| 2048 | 4253 | 4605 | +8.3 % |
| 4096 | 8532 | 9363 | +9.7 % |
| 8192 | 17199 | 18259 | +6.2 % |

Quantizer launches net of the reuse measured in step 05.

#### Phi-4: GEMM kernel time per forward, µs (every quantized text Linear; isolated launches, cold weights (distinct weight copies >= 4x L2 plus a 512 MiB read-flush before every call), CUPTI device time, median of 3 rounds x 30 repetitions in a rotated order)

| T | stock wA | stock wB | FlipQuant (ours) 16x64, typical | FlipQuant (ours) 16x64, worst | FlipQuant (ours) 8x64, typical | FlipQuant (ours) 8x64, worst |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 5650 | 6692 | 5739 | 5754 | 7146 | 7162 |
| 256 | 7004 | 7629 | 7284 | 7279 | 8276 | 8294 |
| 512 | 11965 | 11960 | 12431 | 12449 | 13259 | 13275 |
| 1024 | 22951 | 22891 | 23769 | 23816 | 25057 | 25098 |
| 2048 | 45412 | 45164 | 46966 | 46986 | 49318 | 49515 |
| 4096 | 85608 | 85278 | 88480 | 88625 | 93510 | 93745 |
| 8192 | 168158 | 167707 | 173787 | 174258 | 184532 | 185022 |

Overheads against the same-placement stock kernel (and 8x64 against stock wA):

| T | FlipQuant (ours) 16x64, typical vs stock wA | FlipQuant (ours) 16x64, worst vs stock wA | FlipQuant (ours) 8x64, typical vs stock wB | FlipQuant (ours) 8x64, worst vs stock wB | FlipQuant (ours) 8x64, typical vs stock wA | FlipQuant (ours) 8x64, worst vs stock wA |
|---|---:|---:|---:|---:|---:|---:|
| 128 | +1.6 % | +1.8 % | +6.8 % | +7.0 % | +26.5 % | +26.7 % |
| 256 | +4.0 % | +3.9 % | +8.5 % | +8.7 % | +18.2 % | +18.4 % |
| 512 | +3.9 % | +4.0 % | +10.9 % | +11.0 % | +10.8 % | +10.9 % |
| 1024 | +3.6 % | +3.8 % | +9.5 % | +9.6 % | +9.2 % | +9.4 % |
| 2048 | +3.4 % | +3.5 % | +9.2 % | +9.6 % | +8.6 % | +9.0 % |
| 4096 | +3.4 % | +3.5 % | +9.7 % | +9.9 % | +9.2 % | +9.5 % |
| 8192 | +3.3 % | +3.6 % | +10.0 % | +10.3 % | +9.7 % | +10.0 % |

The activation quantizer per forward, µs (isolated launches; FourOverSix and NVFP4 share the stock GEMM):

| T | NVFP4 quantizer | FourOverSix quantizer | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1102 | 1030 | -6.5 % |
| 256 | 1706 | 1519 | -11.0 % |
| 512 | 2688 | 2381 | -11.4 % |
| 1024 | 3908 | 4661 | +19.3 % |
| 2048 | 6906 | 7637 | +10.6 % |
| 4096 | 14267 | 15135 | +6.1 % |
| 8192 | 28636 | 29432 | +2.8 % |

Quantizer launches net of the reuse measured in step 05.

#### Qwen3.8-27B: GEMM kernel time per forward, µs (every quantized text Linear; isolated launches, cold weights (distinct weight copies >= 4x L2 plus a 512 MiB read-flush before every call), CUPTI device time, median of 3 rounds x 30 repetitions in a rotated order)

| T | stock wA | stock wB | FlipQuant (ours) 16x64, typical | FlipQuant (ours) 16x64, worst | FlipQuant (ours) 8x64, typical | FlipQuant (ours) 8x64, worst |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 11121 | 14386 | 11354 | 11427 | 15359 | 15476 |
| 256 | 13904 | 16356 | 14491 | 14629 | 18079 | 18254 |
| 512 | 23417 | 24725 | 24559 | 24687 | 27263 | 27497 |
| 1024 | 43770 | 44681 | 45320 | 45616 | 48848 | 49278 |
| 2048 | 83233 | 83511 | 85739 | 86362 | 90699 | 91457 |
| 4096 | 156793 | 156084 | 162090 | 162826 | 170945 | 172322 |
| 8192 | 302944 | 302163 | 313015 | 314246 | 331739 | 334554 |

Overheads against the same-placement stock kernel (and 8x64 against stock wA):

| T | FlipQuant (ours) 16x64, typical vs stock wA | FlipQuant (ours) 16x64, worst vs stock wA | FlipQuant (ours) 8x64, typical vs stock wB | FlipQuant (ours) 8x64, worst vs stock wB | FlipQuant (ours) 8x64, typical vs stock wA | FlipQuant (ours) 8x64, worst vs stock wA |
|---|---:|---:|---:|---:|---:|---:|
| 128 | +2.1 % | +2.8 % | +6.8 % | +7.6 % | +38.1 % | +39.2 % |
| 256 | +4.2 % | +5.2 % | +10.5 % | +11.6 % | +30.0 % | +31.3 % |
| 512 | +4.9 % | +5.4 % | +10.3 % | +11.2 % | +16.4 % | +17.4 % |
| 1024 | +3.5 % | +4.2 % | +9.3 % | +10.3 % | +11.6 % | +12.6 % |
| 2048 | +3.0 % | +3.8 % | +8.6 % | +9.5 % | +9.0 % | +9.9 % |
| 4096 | +3.4 % | +3.8 % | +9.5 % | +10.4 % | +9.0 % | +9.9 % |
| 8192 | +3.3 % | +3.7 % | +9.8 % | +10.7 % | +9.5 % | +10.4 % |

The activation quantizer per forward, µs (isolated launches; FourOverSix and NVFP4 share the stock GEMM):

| T | NVFP4 quantizer | FourOverSix quantizer | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1755 | 1639 | -6.6 % |
| 256 | 2717 | 2419 | -10.9 % |
| 512 | 4219 | 3779 | -10.4 % |
| 1024 | 6265 | 7413 | +18.3 % |
| 2048 | 11216 | 12310 | +9.8 % |
| 4096 | 22900 | 24412 | +6.6 % |
| 8192 | 46134 | 47871 | +3.8 % |

Quantizer launches net of the reuse measured in step 05.

### GEMM overheads: old method, new method, end to end

#### Llama-3.1-8B: overhead against the same-placement stock kernel, three ways

GEMM per forward, CUPTI, back-to-back calls, L2-warm, densest module (old step 06); GEMM per forward, isolated and cold (new, typical / worst tags); and the end-to-end CUDA-graph prefill (FlipQuant (ours) vs FourOverSix with the same placement, the median over rounds of the per-round ratio).

| batch x prompt | 16x64: GEMM, old | 16x64: GEMM, new typical | 16x64: GEMM, new worst | 16x64: end to end | 8x64 (wB): GEMM, old | 8x64 (wB): GEMM, new typical | 8x64 (wB): GEMM, new worst | 8x64 (wB): end to end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +9.4 % | +4.5 % | +4.6 % | +1.5 % | +10.3 % | +10.0 % | +11.1 % | +3.8 % |
| 1x256 | +5.3 % | +3.2 % | +3.7 % | +1.4 % | +10.5 % | +11.0 % | +12.4 % | +4.1 % |
| 1x512 | +5.2 % | +5.6 % | +6.7 % | +3.5 % | +12.6 % | +16.0 % | +17.6 % | +9.1 % |
| 1x1024 | +4.2 % | +3.2 % | +3.3 % | +2.4 % | +11.3 % | +8.7 % | +9.4 % | +5.8 % |
| 1x2048 | +4.6 % | +3.6 % | +4.3 % | +1.8 % | +11.6 % | +10.2 % | +11.3 % | +5.1 % |
| 1x4096 | +4.4 % | +3.6 % | +4.3 % | +2.0 % | +12.5 % | +9.6 % | +10.6 % | +4.9 % |
| 1x8192 | +4.8 % | +3.9 % | +4.4 % | +1.4 % | +12.9 % | +10.7 % | +11.9 % | +3.8 % |
| 4x2048 | +4.8 % | +3.9 % | +4.4 % | +1.7 % | +12.9 % | +10.7 % | +11.9 % | +4.6 % |

#### Mistral-7B-v0.3: overhead against the same-placement stock kernel, three ways

GEMM per forward, CUPTI, back-to-back calls, L2-warm, densest module (old step 06); GEMM per forward, isolated and cold (new, typical / worst tags); and the end-to-end CUDA-graph prefill (FlipQuant (ours) vs FourOverSix with the same placement, the median over rounds of the per-round ratio).

| batch x prompt | 16x64: GEMM, old | 16x64: GEMM, new typical | 16x64: GEMM, new worst | 16x64: end to end | 8x64 (wB): GEMM, old | 8x64 (wB): GEMM, new typical | 8x64 (wB): GEMM, new worst | 8x64 (wB): end to end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +9.7 % | +4.4 % | +4.6 % | +1.6 % | +10.5 % | +10.0 % | +11.2 % | +4.0 % |
| 1x256 | +5.2 % | +3.2 % | +3.8 % | +1.3 % | +10.4 % | +10.8 % | +12.1 % | +4.4 % |
| 1x512 | +4.2 % | +5.3 % | +6.3 % | +4.1 % | +13.7 % | +16.2 % | +17.5 % | +8.0 % |
| 1x1024 | +4.3 % | +3.1 % | +3.1 % | +2.8 % | +11.3 % | +8.5 % | +8.8 % | +7.6 % |
| 1x2048 | +4.5 % | +4.4 % | +4.9 % | +2.0 % | +11.5 % | +10.1 % | +11.1 % | +5.7 % |
| 1x4096 | +4.3 % | +3.5 % | +3.9 % | +1.7 % | +12.2 % | +10.4 % | +11.1 % | +5.5 % |
| 1x8192 | +4.7 % | +3.9 % | +4.5 % | +1.5 % | +12.8 % | +10.7 % | +11.7 % | +4.3 % |
| 4x2048 | +4.7 % | +3.9 % | +4.5 % | +1.8 % | +12.8 % | +10.7 % | +11.7 % | +5.2 % |

#### Phi-4: overhead against the same-placement stock kernel, three ways

GEMM per forward, CUPTI, back-to-back calls, L2-warm, densest module (old step 06); GEMM per forward, isolated and cold (new, typical / worst tags); and the end-to-end CUDA-graph prefill (FlipQuant (ours) vs FourOverSix with the same placement, the median over rounds of the per-round ratio).

| batch x prompt | 16x64: GEMM, old | 16x64: GEMM, new typical | 16x64: GEMM, new worst | 16x64: end to end | 8x64 (wB): GEMM, old | 8x64 (wB): GEMM, new typical | 8x64 (wB): GEMM, new worst | 8x64 (wB): end to end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +9.6 % | +1.6 % | +1.8 % | +0.8 % | +8.8 % | +6.8 % | +7.0 % | +2.9 % |
| 1x256 | +3.0 % | +4.0 % | +3.9 % | +2.7 % | +10.5 % | +8.5 % | +8.7 % | +4.2 % |
| 1x512 | +4.5 % | +3.9 % | +4.0 % | +2.4 % | +12.4 % | +10.9 % | +11.0 % | +6.1 % |
| 1x1024 | +4.1 % | +3.6 % | +3.8 % | +2.1 % | +12.2 % | +9.5 % | +9.6 % | +5.5 % |
| 1x2048 | +3.6 % | +3.4 % | +3.5 % | +1.9 % | +10.4 % | +9.2 % | +9.6 % | +5.5 % |
| 1x4096 | +3.9 % | +3.4 % | +3.5 % | +1.7 % | +11.4 % | +9.7 % | +9.9 % | +5.2 % |
| 1x8192 | +4.0 % | +3.3 % | +3.6 % | +1.6 % | +12.6 % | +10.0 % | +10.3 % | +4.3 % |
| 4x2048 | +4.0 % | +3.3 % | +3.6 % | +1.7 % | +12.6 % | +10.0 % | +10.3 % | +4.9 % |

#### Qwen3.8-27B: overhead against the same-placement stock kernel, three ways

GEMM per forward, CUPTI, back-to-back calls, L2-warm, densest module (old step 06); GEMM per forward, isolated and cold (new, typical / worst tags); and the end-to-end CUDA-graph prefill (FlipQuant (ours) vs FourOverSix with the same placement, the median over rounds of the per-round ratio).

| batch x prompt | 16x64: GEMM, old | 16x64: GEMM, new typical | 16x64: GEMM, new worst | 16x64: end to end | 8x64 (wB): GEMM, old | 8x64 (wB): GEMM, new typical | 8x64 (wB): GEMM, new worst | 8x64 (wB): end to end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +11.4 % | +2.1 % | +2.8 % | +0.2 % | +9.3 % | +6.8 % | +7.6 % | +0.9 % |
| 1x256 | +4.3 % | +4.2 % | +5.2 % | +0.6 % | +10.5 % | +10.5 % | +11.6 % | +1.5 % |
| 1x512 | +6.2 % | +4.9 % | +5.4 % | +1.1 % | +13.3 % | +10.3 % | +11.2 % | +2.5 % |
| 1x1024 | +3.5 % | +3.5 % | +4.2 % | +1.2 % | +11.5 % | +9.3 % | +10.3 % | +2.8 % |
| 1x2048 | +4.1 % | +3.0 % | +3.8 % | +1.1 % | +10.9 % | +8.6 % | +9.5 % | +3.0 % |
| 1x4096 | +4.0 % | +3.4 % | +3.8 % | +1.0 % | +11.6 % | +9.5 % | +10.4 % | +2.7 % |
| 1x8192 | +4.1 % | +3.3 % | +3.7 % | +0.8 % | +12.1 % | +9.8 % | +10.7 % | +2.3 % |
| 4x2048 | +4.1 % | +3.3 % | +3.7 % | +1.0 % | +12.1 % | +9.8 % | +10.7 % | +2.6 % |

GEMM vs end-to-end consistency (deviation 2; isolated, cold GEMM; tolerance 1 % of the reference prefill): Llama-3.1-8B 16 (typical) / 12 (worst) of 32 rows flagged; Mistral-7B-v0.3 18 (typical) / 16 (worst) of 32 rows flagged; Phi-4 15 (typical) / 15 (worst) of 32 rows flagged; Qwen3.8-27B 6 (typical) / 4 (worst) of 32 rows flagged (appendix).

The GEMM tables of the alternative method (CUPTI, back-to-back calls, L2-warm, densest module; step 06, deviation 1) are in the appendix. ALTERNATIVE METHOD (CUPTI, back-to-back calls, L2-warm, densest module; step 06): GEMM vs end-to-end consistency (deviation 1; tolerance 1 % of the reference prefill): Llama-3.1-8B 16 of 32 rows flagged; Mistral-7B-v0.3 22 of 32 rows flagged; Phi-4 16 of 32 rows flagged; Qwen3.8-27B 5 of 32 rows flagged (appendix).
