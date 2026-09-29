# Appendix: 256x64, eager prefill (every unit), and the per-shape GEMM detail

### Perplexity (NativeLinear (c); BF16 as loaded)

| model | corpus | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | WikiText-2 | 6.2403 | 6.9338 | 6.8706 | 6.8007 |
| Llama-3.1-8B | C4 | 8.9579 | 9.9284 | 9.8257 | 9.7252 |
| Mistral-7B-v0.3 | WikiText-2 | 5.3182 | 5.5506 | 5.5224 | 5.4904 |
| Mistral-7B-v0.3 | C4 | 7.8306 | 8.0895 | 8.0623 | 8.0279 |
| Phi-4 | WikiText-2 | 6.4615 | 6.6934 | 6.6617 | 6.6257 |
| Phi-4 | C4 | 10.3098 | 10.5795 | 10.5441 | 10.5146 |
| Qwen3.8-27B | WikiText-2 | 7.0509 | 7.5506 | 7.3215 | 7.1773 |
| Qwen3.8-27B | C4 | 9.8935 | 10.2185 | 10.1869 | 10.1512 |

### Paired ΔNLL, nats per token (= Δ log PPL), ± 2 SE over windows; * = |Δ| > 2 SE

| model | corpus | windows | FlipQuant (ours) 256x64 − FourOverSix | FlipQuant (ours) 256x64 − NVFP4 |
|---|---|---:|---:|---:|
| Llama-3.1-8B | WikiText-2 | 141 | -0.0102 ± 0.0017 * | -0.0194 ± 0.0021 * |
| Llama-3.1-8B | C4 | 256 | -0.0103 ± 0.0022 * | -0.0207 ± 0.0036 * |
| Mistral-7B-v0.3 | WikiText-2 | 163 | -0.0058 ± 0.0010 * | -0.0109 ± 0.0012 * |
| Mistral-7B-v0.3 | C4 | 256 | -0.0043 ± 0.0012 * | -0.0077 ± 0.0010 * |
| Phi-4 | WikiText-2 | 141 | -0.0054 ± 0.0013 * | -0.0102 ± 0.0018 * |
| Phi-4 | C4 | 256 | -0.0028 ± 0.0009 * | -0.0062 ± 0.0011 * |
| Qwen3.8-27B | WikiText-2 | 145 | -0.0199 ± 0.0056 * | -0.0507 ± 0.0088 * |
| Qwen3.8-27B | C4 | 256 | -0.0035 ± 0.0009 * | -0.0066 ± 0.0013 * |

### Downstream accuracy, % (lm-eval 0.4.11; MMLU 5-shot, the others 0-shot; acc_norm, MMLU acc)

| model | task | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | MMLU (5-shot) | 65.31 | 62.51 | 62.98 | 63.06 |
| Llama-3.1-8B | ARC-C | 54.78 | 52.30 | 53.84 | 53.92 |
| Llama-3.1-8B | ARC-E | 82.53 | 78.32 | 80.43 | 80.01 |
| Llama-3.1-8B | HellaSwag | 79.34 | 77.86 | 78.16 | 78.16 |
| Llama-3.1-8B | PIQA | 81.12 | 79.38 | 80.47 | 80.47 |
| Llama-3.1-8B | mean of 5 | 72.62 | 70.08 | 71.18 | 71.12 |
| Mistral-7B-v0.3 | MMLU (5-shot) | 62.42 | 59.96 | 60.39 | 60.20 |
| Mistral-7B-v0.3 | ARC-C | 54.69 | 51.96 | 53.07 | 52.47 |
| Mistral-7B-v0.3 | ARC-E | 80.13 | 78.28 | 78.96 | 78.62 |
| Mistral-7B-v0.3 | HellaSwag | 80.62 | 80.12 | 80.06 | 80.28 |
| Mistral-7B-v0.3 | PIQA | 81.77 | 81.12 | 81.01 | 81.56 |
| Mistral-7B-v0.3 | mean of 5 | 71.93 | 70.29 | 70.70 | 70.63 |
| Phi-4 | MMLU (5-shot) | 80.29 | 79.11 | 79.23 | 78.95 |
| Phi-4 | ARC-C | 56.14 | 54.10 | 54.86 | 54.52 |
| Phi-4 | ARC-E | 72.77 | 73.23 | 71.89 | 71.97 |
| Phi-4 | HellaSwag | 81.95 | 80.86 | 81.11 | 81.28 |
| Phi-4 | PIQA | 81.12 | 80.85 | 80.69 | 81.39 |
| Phi-4 | mean of 5 | 74.45 | 73.63 | 73.56 | 73.62 |
| Qwen3.8-27B | MMLU (5-shot) | 82.57 | 81.25 | 81.29 | 81.77 |
| Qwen3.8-27B | ARC-C | 58.79 | 60.92 | 59.30 | 59.13 |
| Qwen3.8-27B | ARC-E | 73.02 | 76.47 | 74.58 | 74.37 |
| Qwen3.8-27B | HellaSwag | 82.89 | 82.44 | 82.26 | 82.28 |
| Qwen3.8-27B | PIQA | 81.50 | 81.39 | 81.28 | 81.18 |
| Qwen3.8-27B | mean of 5 | 75.76 | 76.50 | 75.74 | 75.75 |

### Paired accuracy differences, percentage points, ± 2 SE; * = |Δ| > 2 SE

| model | task | examples | FlipQuant (ours) 256x64 − FourOverSix | FlipQuant (ours) 256x64 − NVFP4 |
|---|---|---:|---:|---:|
| Llama-3.1-8B | MMLU (5-shot) | 14042 | +0.08 ± 0.64 | +0.56 ± 0.66 |
| Llama-3.1-8B | ARC-C | 1172 | +0.09 ± 1.85 | +1.62 ± 1.98 |
| Llama-3.1-8B | ARC-E | 2376 | -0.42 ± 1.00 | +1.68 ± 1.16 * |
| Llama-3.1-8B | HellaSwag | 10042 | +0.00 ± 0.37 | +0.30 ± 0.40 |
| Llama-3.1-8B | PIQA | 1838 | +0.00 ± 1.11 | +1.09 ± 1.16 |
| Llama-3.1-8B | mean of 5 |  | -0.05 ± 0.50 | +1.05 ± 0.54 * |
| Mistral-7B-v0.3 | MMLU (5-shot) | 14042 | -0.19 ± 0.55 | +0.24 ± 0.60 |
| Mistral-7B-v0.3 | ARC-C | 1172 | -0.60 ± 1.57 | +0.51 ± 1.55 |
| Mistral-7B-v0.3 | ARC-E | 2376 | -0.34 ± 0.87 | +0.34 ± 0.94 |
| Mistral-7B-v0.3 | HellaSwag | 10042 | +0.22 ± 0.30 | +0.16 ± 0.34 |
| Mistral-7B-v0.3 | PIQA | 1838 | +0.54 ± 0.91 | +0.44 ± 0.91 |
| Mistral-7B-v0.3 | mean of 5 |  | -0.07 ± 0.42 | +0.34 ± 0.43 |
| Phi-4 | MMLU (5-shot) | 14042 | -0.28 ± 0.39 | -0.16 ± 0.43 |
| Phi-4 | ARC-C | 1172 | -0.34 ± 1.32 | +0.43 ± 1.40 |
| Phi-4 | ARC-E | 2376 | +0.08 ± 0.83 | -1.26 ± 0.99 * |
| Phi-4 | HellaSwag | 10042 | +0.17 ± 0.32 | +0.42 ± 0.36 * |
| Phi-4 | PIQA | 1838 | +0.71 ± 0.86 | +0.54 ± 0.91 |
| Phi-4 | mean of 5 |  | +0.07 ± 0.37 | -0.01 ± 0.40 |
| Qwen3.8-27B | MMLU (5-shot) | 14042 | +0.48 ± 0.45 * | +0.52 ± 0.49 * |
| Qwen3.8-27B | ARC-C | 1172 | -0.17 ± 1.47 | -1.79 ± 1.75 * |
| Qwen3.8-27B | ARC-E | 2376 | -0.21 ± 0.95 | -2.10 ± 1.13 * |
| Qwen3.8-27B | HellaSwag | 10042 | +0.02 ± 0.31 | -0.16 ± 0.36 |
| Qwen3.8-27B | PIQA | 1838 | -0.11 ± 0.84 | -0.22 ± 0.92 |
| Qwen3.8-27B | mean of 5 |  | +0.00 ± 0.40 | -0.75 ± 0.47 * |

MMLU questions whose top two choices tie in log-likelihood, % (lm-eval computes them from BF16 logits and its argmax takes the earlier choice; every policy is scored the same way):

| model | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 5.8 | 5.6 | 6.7 | 6.4 |
| Mistral-7B-v0.3 | 6.4 | 6.3 | 7.4 | 7.3 |
| Phi-4 | 0.9 | 1.1 | 1.0 | 1.0 |
| Qwen3.8-27B | 1.0 | 1.0 | 1.1 | 1.1 |

### Prefill latency, CUDA graph (primary)

#### Llama-3.1-8B, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|
| 1x128 | 16.41 | 7.55 | 7.46 | 7.67 | 7.58 |
| 1x256 | 22.18 | 9.49 | 9.40 | 9.63 | 9.54 |
| 1x512 | 38.65 | 15.17 | 15.22 | 15.86 | 15.64 |
| 1x1024 | 71.07 | 31.39 | 31.57 | 32.07 | 32.38 |
| 1x2048 | 133.32 | 62.14 | 62.40 | 63.36 | 63.58 |
| 1x4096 | 272.35 | 134.44 | 134.91 | 136.68 | 137.56 |
| 1x8192 | 604.73 | 324.11 | 324.64 | 328.79 | 329.52 |
| 4x2048 | 556.90 | 276.53 | 277.11 | 281.16 | 281.86 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|
| 1x128 | +1.6 % [+1.6, +1.6] | +1.6 % [+1.5, +1.7] |
| 1x256 | +1.6 % [+1.5, +1.6] | +1.6 % [+1.4, +1.6] |
| 1x512 | +2.4 % [+1.9, +5.1] | +3.6 % [+2.7, +6.4] |
| 1x1024 | +2.7 % [+2.5, +3.1] | +2.2 % [+2.0, +2.7] |
| 1x2048 | +2.1 % [+1.8, +2.2] | +2.1 % [+1.5, +2.1] |
| 1x4096 | +2.1 % [+1.9, +2.2] | +1.7 % [+1.3, +1.9] |
| 1x8192 | +1.5 % [+1.4, +1.7] | +1.5 % [+1.2, +1.6] |
| 4x2048 | +1.8 % [+1.7, +2.0] | +1.6 % [+1.5, +1.8] |

#### Mistral-7B-v0.3, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|
| 1x128 | 15.92 | 7.08 | 6.99 | 7.20 | 7.11 |
| 1x256 | 21.07 | 8.89 | 8.80 | 9.03 | 8.93 |
| 1x512 | 36.91 | 13.57 | 13.35 | 13.98 | 13.89 |
| 1x1024 | 67.80 | 28.07 | 28.35 | 28.97 | 29.23 |
| 1x2048 | 126.98 | 56.11 | 56.34 | 57.34 | 57.58 |
| 1x4096 | 259.90 | 123.01 | 123.49 | 125.21 | 125.77 |
| 1x8192 | 579.28 | 300.76 | 300.93 | 305.32 | 305.89 |
| 4x2048 | 532.00 | 253.18 | 253.40 | 257.90 | 258.52 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|
| 1x128 | +1.7 % [+1.7, +1.7] | +1.7 % [+1.6, +1.7] |
| 1x256 | +1.5 % [+1.4, +1.5] | +1.5 % [+1.4, +1.6] |
| 1x512 | +4.2 % [+3.4, +5.0] | +3.8 % [+2.7, +3.9] |
| 1x1024 | +3.4 % [+2.4, +4.0] | +3.3 % [+1.8, +3.5] |
| 1x2048 | +2.2 % [+2.2, +2.7] | +2.2 % [+1.6, +2.6] |
| 1x4096 | +2.0 % [+1.8, +2.4] | +1.8 % [+1.3, +2.1] |
| 1x8192 | +1.7 % [+1.5, +2.0] | +1.5 % [+1.1, +1.7] |
| 4x2048 | +2.0 % [+1.9, +2.3] | +1.8 % [+1.6, +2.1] |

#### Phi-4, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|
| 1x128 | 28.28 | 11.47 | 11.38 | 11.57 | 11.47 |
| 1x256 | 37.95 | 15.22 | 14.98 | 15.62 | 15.43 |
| 1x512 | 68.38 | 27.84 | 27.60 | 28.48 | 28.26 |
| 1x1024 | 128.25 | 54.83 | 55.28 | 56.05 | 56.44 |
| 1x2048 | 248.21 | 110.90 | 111.50 | 113.15 | 113.62 |
| 1x4096 | 512.87 | 237.26 | 237.59 | 241.35 | 241.70 |
| 1x8192 | 1119.36 | 559.24 | 560.76 | 567.95 | 569.57 |
| 4x2048 | 1043.46 | 486.98 | 488.21 | 495.59 | 497.19 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|
| 1x128 | +0.8 % [+0.8, +0.9] | +0.8 % [+0.8, +0.9] |
| 1x256 | +3.0 % [+2.6, +4.0] | +2.5 % [+1.6, +4.4] |
| 1x512 | +2.4 % [+2.2, +2.8] | +2.3 % [+2.0, +2.7] |
| 1x1024 | +2.1 % [+1.8, +2.5] | +2.2 % [+1.7, +2.6] |
| 1x2048 | +2.0 % [+1.8, +2.2] | +1.9 % [+1.5, +2.2] |
| 1x4096 | +1.9 % [+1.6, +2.1] | +1.8 % [+1.4, +2.0] |
| 1x8192 | +1.7 % [+1.5, +1.9] | +1.6 % [+1.2, +1.7] |
| 4x2048 | +1.9 % [+1.7, +2.1] | +1.7 % [+1.4, +2.0] |

#### Qwen3.8-27B, CUDA-graph prefill, ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|
| 1x128 | 86.47 | 57.79 | 57.71 | 58.00 | 57.83 |
| 1x256 | 115.81 | 73.23 | 72.94 | 73.74 | 73.34 |
| 1x512 | 184.01 | 109.14 | 108.64 | 110.40 | 109.89 |
| 1x1024 | 329.66 | 193.72 | 194.12 | 195.86 | 196.29 |
| 1x2048 | 637.21 | 380.52 | 380.75 | 384.38 | 384.85 |
| 1x4096 | 1307.62 | 808.22 | 807.03 | 815.16 | 814.75 |
| 1x8192 | 2815.62 | 1812.21 | 1812.03 | 1824.57 | 1825.84 |
| 4x2048 | 2701.11 | 1677.29 | 1677.30 | 1691.21 | 1693.40 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|
| 1x128 | +0.2 % [+0.2, +0.5] | +0.4 % [+0.1, +0.4] |
| 1x256 | +0.6 % [+0.4, +0.6] | +0.7 % [+0.6, +0.8] |
| 1x512 | +1.0 % [+1.0, +1.3] | +1.3 % [+1.0, +1.3] |
| 1x1024 | +1.2 % [+0.9, +1.3] | +1.1 % [+0.9, +1.3] |
| 1x2048 | +1.1 % [+0.9, +1.2] | +1.0 % [+1.0, +1.1] |
| 1x4096 | +1.0 % [+0.8, +1.1] | +0.8 % [+0.8, +1.0] |
| 1x8192 | +0.9 % [+0.7, +1.0] | +0.7 % [+0.5, +0.7] |
| 4x2048 | +1.0 % [+0.9, +1.0] | +0.8 % [+0.7, +0.9] |

### GEMM latency

#### Llama-3.1-8B: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, median of 3 rounds in rotated order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|
| 128 | 2864 | 4416 | 3137 | +9.5 % |
| 256 | 4140 | 5377 | 4370 | +5.5 % |
| 512 | 6138 | 6518 | 6435 | +4.8 % |
| 1024 | 11694 | 11872 | 12205 | +4.4 % |
| 2048 | 20523 | 20440 | 21466 | +4.6 % |
| 4096 | 42378 | 42292 | 44326 | +4.6 % |
| 8192 | 82090 | 81712 | 86055 | +4.8 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 796 | 712 | -10.6 % |
| 256 | 968 | 889 | -8.1 % |
| 512 | 1631 | 1493 | -8.5 % |
| 1024 | 2442 | 2800 | +14.7 % |
| 2048 | 4213 | 4621 | +9.7 % |
| 4096 | 8718 | 9135 | +4.8 % |
| 8192 | 16808 | 17939 | +6.7 % |

Quantizer launches net of the reuse measured in step 05.

#### Mistral-7B-v0.3: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, median of 3 rounds in rotated order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|
| 128 | 2863 | 4419 | 3131 | +9.4 % |
| 256 | 4140 | 5380 | 4370 | +5.5 % |
| 512 | 6114 | 6429 | 6402 | +4.7 % |
| 1024 | 11696 | 11900 | 12232 | +4.6 % |
| 2048 | 20578 | 20489 | 21508 | +4.5 % |
| 4096 | 42485 | 42341 | 44308 | +4.3 % |
| 8192 | 82192 | 81770 | 86121 | +4.8 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 796 | 711 | -10.7 % |
| 256 | 968 | 889 | -8.2 % |
| 512 | 1638 | 1494 | -8.8 % |
| 1024 | 2446 | 2801 | +14.5 % |
| 2048 | 4221 | 4625 | +9.6 % |
| 4096 | 8736 | 9154 | +4.8 % |
| 8192 | 16846 | 17961 | +6.6 % |

Quantizer launches net of the reuse measured in step 05.

#### Phi-4: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, median of 3 rounds in rotated order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|
| 128 | 3825 | 5335 | 4188 | +9.5 % |
| 256 | 5885 | 6565 | 6060 | +3.0 % |
| 512 | 10644 | 10636 | 11129 | +4.6 % |
| 1024 | 20642 | 20549 | 21510 | +4.2 % |
| 2048 | 42106 | 41904 | 43692 | +3.8 % |
| 4096 | 80234 | 79773 | 83475 | +4.0 % |
| 8192 | 172136 | 169746 | 178262 | +3.6 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1114 | 1040 | -6.7 % |
| 256 | 1960 | 1719 | -12.3 % |
| 512 | 3174 | 2883 | -9.2 % |
| 1024 | 4368 | 5090 | +16.5 % |
| 2048 | 8227 | 8833 | +7.4 % |
| 4096 | 14365 | 15251 | +6.2 % |
| 8192 | 27330 | 29832 | +9.2 % |

Quantizer launches net of the reuse measured in step 05.

#### Qwen3.8-27B: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, median of 3 rounds in rotated order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|
| 128 | 7843 | 11817 | 8753 | +11.6 % |
| 256 | 12064 | 14595 | 12605 | +4.5 % |
| 512 | 18696 | 20018 | 19940 | +6.7 % |
| 1024 | 37601 | 38308 | 39034 | +3.8 % |
| 2048 | 75558 | 75958 | 78770 | +4.3 % |
| 4096 | 145093 | 144284 | 150961 | +4.0 % |
| 8192 | 288063 | 286921 | 300326 | +4.3 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1756 | 1639 | -6.7 % |
| 256 | 2807 | 2524 | -10.1 % |
| 512 | 4944 | 4490 | -9.2 % |
| 1024 | 6450 | 7585 | +17.6 % |
| 2048 | 12822 | 13770 | +7.4 % |
| 4096 | 23182 | 24592 | +6.1 % |
| 8192 | 43429 | 46694 | +7.5 % |

Quantizer launches net of the reuse measured in step 05.

### Prefill latency, eager (supplementary)

#### Llama-3.1-8B, eager prefill (supplementary; † host-bound: the graph is more than 5 % faster), ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 39.50 † | 35.90 † | 30.91 † | 29.94 † | 32.33 † | 30.00 † | 29.98 † | 30.52 † | 36.06 † | 30.53 † | 34.37 † |
| 1x256 | 39.86 † | 33.52 † | 30.74 † | 30.84 † | 30.66 † | 34.15 † | 31.22 † | 30.98 † | 36.39 † | 31.37 † | 36.10 † |
| 1x512 | 38.28 | 31.55 † | 31.39 † | 31.48 † | 31.13 † | 31.45 † | 31.46 † | 32.06 † | 31.49 † | 32.06 † | 36.59 † |
| 1x1024 | 70.12 | 33.38 † | 33.30 † | 33.32 † | 32.74 † | 33.15 † | 33.41 † | 34.26 † | 33.53 † | 34.21 † | 36.29 † |
| 1x2048 | 131.25 | 61.72 | 61.96 | 61.79 | 62.10 | 65.15 | 65.40 | 62.86 | 63.00 | 62.86 | 63.07 |
| 1x4096 | 267.88 | 133.56 | 133.97 | 133.46 | 134.11 | 140.20 | 140.78 | 135.73 | 136.15 | 135.80 | 136.20 |
| 1x8192 | 596.49 | 321.30 | 321.60 | 321.87 | 322.81 | 334.79 | 335.28 | 325.56 | 326.13 | 325.76 | 326.16 |
| 4x2048 | 551.00 | 274.27 | 274.91 | 275.05 | 275.71 | 288.04 | 288.50 | 278.85 | 279.32 | 279.01 | 279.40 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | -1.6 % [-5.0, +14.2] | -17.4 % [-17.7, +18.2] | -2.8 % [-14.5, +19.5] | +0.1 % [-1.0, +20.4] | +16.9 % [-1.6, +20.8] | -4.3 % [-16.2, +0.2] | +2.9 % [-2.8, +18.9] | -5.5 % [-16.0, -0.2] |
| 1x256 | +1.8 % [-16.0, +18.6] | +8.1 % [-16.5, +18.9] | -0.3 % [-12.5, +17.9] | +0.8 % [-3.0, +18.6] | +1.6 % [-1.9, +19.9] | -1.2 % [-14.3, +4.5] | +0.0 % [-0.9, +19.2] | -1.7 % [-11.8, +3.3] |
| 1x512 | +1.4 % [-16.9, +16.8] | -1.6 % [-14.5, +18.7] | +0.5 % [-13.2, +16.1] | -1.4 % [-15.7, +17.3] | +0.0 % [-15.2, +19.2] | +0.4 % [-3.3, +18.7] | +0.4 % [-1.4, +18.2] | +0.2 % [-2.6, +16.1] |
| 1x1024 | +1.1 % [-14.7, +14.5] | -0.7 % [-13.4, +14.2] | +2.0 % [-1.3, +15.2] | +0.0 % [-13.5, +14.0] | +0.2 % [-13.6, +16.1] | +2.0 % [-1.3, +12.0] | +0.4 % [-6.0, +14.0] | +0.4 % [-6.9, +15.1] |
| 1x2048 | +5.6 % [+4.8, +5.6] | +5.5 % [+5.2, +6.3] | +5.1 % [+4.8, +5.6] | +5.5 % [+5.0, +6.1] | +1.7 % [+1.5, +1.9] | +1.8 % [+1.4, +2.3] | +1.9 % [+1.5, +2.1] | +1.8 % [+1.5, +2.0] |
| 1x4096 | +5.1 % [+4.3, +5.1] | +4.9 % [+4.6, +5.8] | +4.8 % [+4.6, +5.3] | +5.1 % [+4.9, +5.8] | +1.6 % [+1.4, +1.8] | +1.8 % [+1.2, +2.4] | +1.7 % [+1.5, +2.1] | +1.6 % [+1.3, +1.9] |
| 1x8192 | +4.3 % [+3.7, +4.4] | +4.2 % [+3.8, +4.8] | +3.7 % [+3.6, +4.1] | +3.9 % [+3.8, +4.6] | +1.3 % [+1.1, +1.5] | +1.4 % [+1.0, +1.9] | +1.4 % [+1.3, +1.8] | +1.3 % [+1.0, +1.8] |
| 4x2048 | +5.0 % [+4.4, +5.2] | +5.0 % [+4.5, +5.7] | +4.4 % [+4.4, +4.9] | +4.6 % [+4.5, +5.3] | +1.6 % [+1.4, +1.7] | +1.7 % [+1.2, +2.2] | +1.7 % [+1.5, +2.0] | +1.6 % [+1.3, +2.0] |

#### Mistral-7B-v0.3, eager prefill (supplementary; † host-bound: the graph is more than 5 % faster), ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 38.86 † | 31.95 † | 30.09 † | 30.11 † | 34.41 † | 35.07 † | 29.83 † | 31.07 † | 29.91 † | 29.85 † | 30.41 † |
| 1x256 | 39.00 † | 30.16 † | 30.16 † | 34.71 † | 35.03 † | 35.40 † | 29.92 † | 35.80 † | 30.61 † | 30.22 † | 30.94 † |
| 1x512 | 38.31 † | 30.06 † | 30.16 † | 30.45 † | 30.77 † | 34.16 † | 29.74 † | 35.67 † | 30.59 † | 30.19 † | 31.11 † |
| 1x1024 | 66.98 | 30.42 † | 30.60 † | 29.97 † | 31.04 † | 30.15 † | 30.17 | 32.06 † | 30.53 † | 30.62 † | 30.76 † |
| 1x2048 | 125.30 | 55.69 | 55.91 | 55.87 | 56.15 | 59.11 | 59.42 | 56.78 | 57.02 | 56.84 | 57.08 |
| 1x4096 | 256.36 | 121.45 | 121.86 | 121.51 | 122.16 | 128.15 | 128.91 | 123.72 | 124.22 | 123.87 | 124.28 |
| 1x8192 | 571.79 | 297.63 | 298.00 | 298.43 | 299.32 | 311.05 | 311.93 | 302.01 | 302.47 | 302.57 | 302.80 |
| 4x2048 | 526.72 | 251.13 | 251.55 | 251.89 | 252.46 | 264.65 | 265.44 | 255.73 | 256.02 | 255.77 | 256.20 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | -1.8 % [-15.1, +9.2] | +9.8 % [-5.1, +19.0] | -13.1 % [-18.3, +6.9] | +13.2 % [+1.1, +19.6] | -0.9 % [-14.4, +19.1] | -2.7 % [-11.1, +19.6] | +0.8 % [-10.9, +18.4] | +0.4 % [-19.1, +11.6] |
| 1x256 | -2.1 % [-13.1, -0.2] | +17.2 % [-12.4, +17.5] | -11.8 % [-17.2, +0.3] | +1.9 % [-3.6, +18.0] | +0.9 % [-14.2, +18.9] | +19.0 % [-11.5, +20.2] | +1.6 % [-1.6, +20.3] | +0.3 % [-25.3, +19.6] |
| 1x512 | -3.1 % [-16.4, +0.3] | +13.6 % [-16.2, +17.8] | -3.3 % [-17.4, +0.6] | -1.0 % [-7.5, +19.7] | +1.2 % [-14.0, +18.6] | +14.3 % [-0.9, +20.2] | +0.0 % [-2.1, +17.8] | +0.3 % [-16.1, +4.7] |
| 1x1024 | -3.2 % [-5.4, +0.2] | -0.7 % [-9.1, +8.9] | -2.4 % [-15.6, +0.8] | +0.2 % [-7.7, +2.8] | -0.2 % [-4.1, +17.4] | +2.4 % [-3.3, +18.9] | -1.8 % [-3.8, +2.5] | +0.6 % [-7.7, +21.2] |
| 1x2048 | +6.2 % [+6.1, +6.5] | +6.1 % [+5.2, +6.6] | +5.8 % [+5.5, +6.4] | +5.7 % [+5.6, +6.4] | +2.0 % [+1.7, +2.3] | +2.0 % [+1.0, +3.0] | +2.3 % [+2.0, +2.7] | +2.1 % [+1.3, +2.3] |
| 1x4096 | +5.7 % [+5.5, +5.9] | +5.5 % [+4.9, +6.1] | +5.5 % [+5.1, +5.9] | +5.5 % [+5.1, +6.0] | +1.9 % [+1.7, +2.0] | +1.8 % [+1.2, +2.6] | +2.1 % [+1.9, +2.4] | +2.0 % [+1.2, +2.4] |
| 1x8192 | +4.6 % [+4.5, +4.8] | +4.5 % [+3.9, +5.0] | +4.3 % [+3.9, +4.8] | +4.2 % [+3.8, +4.8] | +1.5 % [+1.1, +1.6] | +1.5 % [+0.9, +2.2] | +1.7 % [+1.5, +2.0] | +1.6 % [+1.1, +1.8] |
| 4x2048 | +5.4 % [+5.3, +5.6] | +5.4 % [+4.9, +5.8] | +5.2 % [+4.8, +5.4] | +5.1 % [+4.8, +5.5] | +1.8 % [+1.5, +1.9] | +1.9 % [+1.3, +2.5] | +2.0 % [+1.8, +2.3] | +1.8 % [+1.4, +2.2] |

#### Phi-4, eager prefill (supplementary; † host-bound: the graph is more than 5 % faster), ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 43.20 † | 34.72 † | 35.01 † | 34.73 † | 35.12 † | 34.95 † | 34.48 † | 35.43 † | 35.07 † | 34.77 † | 40.78 † |
| 1x256 | 40.28 † | 35.28 † | 34.95 † | 35.26 † | 40.93 † | 35.12 † | 35.54 † | 35.19 † | 35.48 † | 40.12 † | 38.08 † |
| 1x512 | 67.01 | 35.44 † | 36.08 † | 37.82 † | 40.97 † | 35.35 † | 35.92 † | 35.22 † | 35.40 † | 41.22 † | 36.04 † |
| 1x1024 | 125.62 | 53.99 | 54.49 | 54.29 | 54.77 | 57.38 | 57.91 | 55.05 | 55.61 | 55.27 | 55.69 |
| 1x2048 | 242.99 | 109.02 | 109.69 | 109.52 | 110.13 | 115.58 | 116.22 | 111.17 | 111.73 | 111.42 | 111.88 |
| 1x4096 | 503.13 | 233.49 | 233.81 | 234.36 | 234.97 | 246.25 | 246.59 | 237.62 | 237.79 | 237.95 | 237.79 |
| 1x8192 | 1103.20 | 551.06 | 552.59 | 553.62 | 555.00 | 577.23 | 578.48 | 558.93 | 560.19 | 559.74 | 560.56 |
| 4x2048 | 1035.39 | 480.80 | 482.32 | 483.17 | 484.63 | 507.28 | 508.72 | 489.09 | 490.13 | 489.18 | 490.43 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | -0.2 % [-1.8, +9.1] | -1.3 % [-4.6, +17.5] | +0.2 % [-16.7, +2.5] | +1.2 % [-5.9, +17.5] | +0.9 % [-7.5, +14.6] | +0.9 % [-2.2, +8.6] | +12.4 % [+0.4, +16.6] | +0.2 % [-13.9, +12.1] |
| 1x256 | +1.6 % [+0.9, +4.8] | -1.0 % [-12.4, +16.9] | +0.7 % [-17.7, +2.6] | -1.2 % [-13.3, +14.7] | +0.4 % [-8.8, +4.8] | +0.4 % [-1.0, +2.2] | +5.8 % [-0.5, +18.5] | +13.7 % [-13.7, +19.9] |
| 1x512 | -1.7 % [-11.6, +1.3] | -0.8 % [-13.5, +17.2] | -0.6 % [-13.5, +1.4] | -2.3 % [-14.9, +9.2] | -2.9 % [-16.5, +1.4] | -0.2 % [-13.9, +2.5] | -1.1 % [-13.2, +12.5] | +16.3 % [-13.5, +18.5] |
| 1x1024 | +6.2 % [+5.8, +6.8] | +6.4 % [+5.6, +7.1] | +5.7 % [+5.5, +6.2] | +5.6 % [+5.5, +6.3] | +2.1 % [+1.9, +2.2] | +2.2 % [+1.5, +2.4] | +2.2 % [+1.6, +2.6] | +2.2 % [+2.1, +2.4] |
| 1x2048 | +6.0 % [+5.4, +6.3] | +6.0 % [+5.3, +6.7] | +5.4 % [+5.3, +5.7] | +5.5 % [+5.4, +6.0] | +1.9 % [+1.7, +2.0] | +2.1 % [+1.2, +2.3] | +2.0 % [+1.7, +2.3] | +2.1 % [+1.5, +2.4] |
| 1x4096 | +5.5 % [+4.9, +5.7] | +5.4 % [+4.9, +6.0] | +4.9 % [+4.9, +5.2] | +5.1 % [+5.0, +5.6] | +1.7 % [+1.6, +2.0] | +1.8 % [+1.1, +2.1] | +1.8 % [+1.6, +2.1] | +1.7 % [+1.3, +2.1] |
| 1x8192 | +4.7 % [+4.0, +5.0] | +4.7 % [+4.2, +5.2] | +4.1 % [+4.0, +4.5] | +4.2 % [+4.1, +4.7] | +1.4 % [+1.2, +1.5] | +1.4 % [+0.8, +1.9] | +1.5 % [+1.3, +1.7] | +1.5 % [+1.2, +1.7] |
| 4x2048 | +5.5 % [+4.9, +5.8] | +5.5 % [+5.0, +6.0] | +4.9 % [+4.8, +5.1] | +5.0 % [+4.9, +5.3] | +1.7 % [+1.6, +1.8] | +1.8 % [+1.2, +2.0] | +1.9 % [+1.6, +2.0] | +1.8 % [+1.5, +2.0] |

#### Qwen3.8-27B, eager prefill (supplementary; † host-bound: the graph is more than 5 % faster), ms, median of [5] round(s)

| batch x prompt | BF16 | NVFP4 | FourOverSix | NVFP4 (wB) | FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64, NVFP4 act. (latency only) | FlipQuant (ours) 16x64 | FlipQuant (ours) 256x64, NVFP4 act. (latency only) | FlipQuant (ours) 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 401.96 † | 397.43 † | 391.99 † | 393.35 † | 397.69 † | 396.93 † | 394.72 † | 394.41 † | 396.25 † | 391.17 † | 393.22 † |
| 1x256 | 440.02 † | 443.66 † | 442.83 † | 445.67 † | 452.75 † | 447.95 † | 444.52 † | 437.55 † | 443.28 † | 441.85 † | 448.00 † |
| 1x512 | 540.70 † | 553.15 † | 555.00 † | 551.60 † | 567.93 † | 558.20 † | 546.98 † | 542.42 † | 545.77 † | 543.18 † | 548.12 † |
| 1x1024 | 738.22 † | 746.43 † | 750.05 † | 742.96 † | 731.30 † | 812.26 † | 735.90 † | 744.31 † | 739.61 † | 738.99 † | 739.35 † |
| 1x2048 | 1143.22 † | 1138.24 † | 1134.90 † | 1149.80 † | 1125.29 † | 1134.75 † | 1131.44 † | 1164.57 † | 1149.48 † | 1137.26 † | 1165.59 † |
| 1x4096 | 1947.25 † | 1952.66 † | 1930.33 † | 1935.79 † | 1922.92 † | 1942.54 † | 1938.94 † | 1968.93 † | 1928.51 † | 1923.15 † | 2049.48 † |
| 1x8192 | 4153.08 † | 3629.92 † | 3594.00 † | 3616.51 † | 3630.26 † | 3692.87 † | 3615.79 † | 3602.82 † | 3695.52 † | 3656.28 † | 3723.49 † |
| 4x2048 | 2686.02 | 1669.31 | 1667.41 | 1669.13 | 1671.21 | 1713.25 | 1713.91 | 1682.05 | 1682.61 | 1682.23 | 1682.26 |

FlipQuant (ours) / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:

| batch x prompt | FlipQuant (ours) 8x64 vs FourOverSix | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | FlipQuant (ours) 8x64, NVFP4 act. (latency only) vs NVFP4 (wB) | FlipQuant (ours) 16x64 vs FourOverSix | FlipQuant (ours) 16x64, NVFP4 act. (latency only) vs NVFP4 | FlipQuant (ours) 256x64 vs FourOverSix | FlipQuant (ours) 256x64, NVFP4 act. (latency only) vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +0.0 % [-4.3, +1.1] | -1.8 % [-5.7, +1.0] | -1.6 % [-13.5, +14.8] | -0.8 % [-11.5, +1.5] | +1.1 % [-15.5, +11.9] | -1.1 % [-13.4, +16.6] | -0.1 % [-1.2, +1.8] | -3.3 % [-13.2, -0.0] |
| 1x256 | +0.3 % [-0.9, +16.2] | +0.2 % [-14.2, +15.4] | -0.5 % [-13.9, +14.3] | +0.5 % [-3.8, +15.6] | +1.0 % [-1.8, +12.8] | -1.5 % [-14.0, -0.2] | +0.6 % [-1.4, +2.2] | -0.5 % [-16.2, +1.0] |
| 1x512 | +1.1 % [-12.9, +9.4] | +1.9 % [-12.8, +15.6] | -1.0 % [-4.1, +0.3] | +2.4 % [-3.0, +5.8] | -2.5 % [-11.0, +16.6] | -1.9 % [-14.3, +14.0] | -1.2 % [-12.0, +3.7] | -2.0 % [-16.9, +15.3] |
| 1x1024 | -2.4 % [-10.9, +0.0] | +8.2 % [-6.3, +15.7] | +0.0 % [-2.1, +1.7] | +10.5 % [-15.3, +15.7] | -1.2 % [-11.0, +2.1] | -0.5 % [-2.5, -0.0] | -0.8 % [-11.0, +1.6] | -1.6 % [-5.8, +2.6] |
| 1x2048 | +1.5 % [-14.5, +15.3] | +0.1 % [-4.3, +4.2] | +0.5 % [-3.4, +16.4] | -1.2 % [-12.9, +1.4] | +2.3 % [-13.1, +7.3] | +0.5 % [+0.2, +5.2] | +3.3 % [-13.0, +6.0] | -1.6 % [-4.4, +1.1] |
| 1x4096 | +1.1 % [-10.6, +2.7] | +1.2 % [-3.2, +10.9] | +0.4 % [-3.2, +5.0] | -1.3 % [-9.4, +13.9] | -0.1 % [-5.7, +3.4] | -0.3 % [-1.1, +11.2] | +3.1 % [-7.0, +13.6] | -1.7 % [-2.6, +2.4] |
| 1x8192 | +0.6 % [-2.0, +2.3] | +1.6 % [-4.3, +4.9] | +0.1 % [-2.1, +3.3] | +1.4 % [-5.3, +7.1] | +2.3 % [-2.1, +4.6] | -1.3 % [-2.2, -0.2] | +2.2 % [-0.5, +5.1] | +0.2 % [-3.6, +4.0] |
| 4x2048 | +2.8 % [+2.3, +3.1] | +2.7 % [+2.6, +2.9] | +2.6 % [+2.1, +2.8] | +2.6 % [+2.5, +2.7] | +0.9 % [+0.8, +1.1] | +0.8 % [+0.7, +1.0] | +0.9 % [+0.9, +1.0] | +0.8 % [+0.7, +1.0] |

### GEMM vs end-to-end consistency (deviation 1)

Per-forward GEMM time difference against the end-to-end CUDA-graph prefill difference, both in ms and as % of the reference prefill; FLAG when they differ by more than 1 % of the reference prefill (the activation quantizer and the rest of the forward are the same on both sides).

| model | comparison | batch x prompt | end to end, ms (%) [per-round range, %] | GEMM per forward, ms (%) | gap, pp | check |
|---|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.116 (+1.5 %) [+1.5, +1.7] | +0.270 (+3.6 %) | -2.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.135 (+1.4 %) [+1.4, +1.5] | +0.218 (+2.3 %) | -0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.535 (+3.5 %) [+1.6, +4.3] | +0.321 (+2.1 %) | +1.4 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +0.754 (+2.4 %) [+1.9, +2.4] | +0.488 (+1.5 %) | +0.8 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +1.115 (+1.8 %) [+1.7, +2.0] | +0.945 (+1.5 %) | +0.3 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +2.638 (+2.0 %) [+1.7, +2.0] | +1.865 (+1.4 %) | +0.6 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +4.537 (+1.4 %) [+1.3, +1.6] | +3.946 (+1.2 %) | +0.2 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +4.622 (+1.7 %) [+1.6, +1.9] | +3.946 (+1.4 %) | +0.2 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.118 (+1.6 %) [+1.6, +1.6] | +0.273 (+3.7 %) | -2.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.147 (+1.6 %) [+1.5, +1.6] | +0.229 (+2.4 %) | -0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.369 (+2.4 %) [+1.9, +5.1] | +0.298 (+2.0 %) | +0.5 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +0.852 (+2.7 %) [+2.5, +3.1] | +0.512 (+1.6 %) | +1.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +1.306 (+2.1 %) [+1.8, +2.3] | +0.943 (+1.5 %) | +0.6 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +2.820 (+2.1 %) [+1.9, +2.2] | +1.948 (+1.4 %) | +0.6 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +5.014 (+1.5 %) [+1.4, +1.7] | +3.966 (+1.2 %) | +0.3 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +4.954 (+1.8 %) [+1.7, +2.0] | +3.966 (+1.4 %) | +0.4 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.339 (+3.8 %) [+3.8, +3.9] | +0.453 (+5.1 %) | -1.3 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.429 (+4.1 %) [+4.0, +4.1] | +0.564 (+5.4 %) | -1.3 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.418 (+9.1 %) [+5.9, +9.8] | +0.823 (+5.3 %) | +3.8 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +1.804 (+5.8 %) [+5.6, +6.3] | +1.347 (+4.3 %) | +1.5 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +3.214 (+5.1 %) [+4.9, +5.6] | +2.371 (+3.8 %) | +1.3 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +6.598 (+4.9 %) [+4.6, +5.4] | +5.274 (+3.9 %) | +1.0 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +12.471 (+3.8 %) [+3.6, +4.2] | +10.506 (+3.2 %) | +0.6 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +12.870 (+4.6 %) [+4.4, +5.0] | +10.506 (+3.8 %) | +0.8 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.752 (+23.5 %) [+23.4, +23.5] | +2.004 (+26.9 %) | -3.4 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +1.541 (+16.4 %) [+16.3, +16.4] | +1.801 (+19.2 %) | -2.8 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.824 (+12.0 %) [+7.7, +12.0] | +1.204 (+7.9 %) | +4.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +1.458 (+4.6 %) [+4.0, +4.8] | +1.525 (+4.8 %) | -0.2 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +3.479 (+5.6 %) [+5.0, +5.9] | +2.288 (+3.7 %) | +1.9 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +6.985 (+5.2 %) [+4.5, +5.2] | +5.188 (+3.8 %) | +1.3 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +13.676 (+4.2 %) [+3.9, +4.5] | +10.128 (+3.1 %) | +1.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +13.872 (+5.0 %) [+4.7, +5.2] | +10.128 (+3.7 %) | +1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.114 (+1.6 %) [+1.6, +1.7] | +0.279 (+4.0 %) | -2.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.115 (+1.3 %) [+1.3, +1.3] | +0.217 (+2.5 %) | -1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.553 (+4.1 %) [+1.6, +4.7] | +0.256 (+1.9 %) | +2.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +0.796 (+2.8 %) [+2.4, +3.0] | +0.507 (+1.8 %) | +1.0 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +1.126 (+2.0 %) [+1.8, +2.1] | +0.918 (+1.6 %) | +0.4 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +2.083 (+1.7 %) [+1.5, +1.9] | +1.840 (+1.5 %) | +0.2 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +4.566 (+1.5 %) [+1.1, +1.6] | +3.871 (+1.3 %) | +0.2 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +4.623 (+1.8 %) [+1.5, +1.9] | +3.871 (+1.5 %) | +0.3 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.119 (+1.7 %) [+1.7, +1.7] | +0.268 (+3.8 %) | -2.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.128 (+1.5 %) [+1.4, +1.5] | +0.229 (+2.6 %) | -1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.561 (+4.2 %) [+3.5, +5.0] | +0.288 (+2.2 %) | +2.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +0.971 (+3.4 %) [+2.4, +4.0] | +0.535 (+1.9 %) | +1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +1.253 (+2.2 %) [+2.2, +2.7] | +0.930 (+1.7 %) | +0.6 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +2.436 (+2.0 %) [+1.8, +2.4] | +1.822 (+1.5 %) | +0.5 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +5.045 (+1.7 %) [+1.5, +2.0] | +3.929 (+1.3 %) | +0.4 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +5.120 (+2.0 %) [+1.9, +2.3] | +3.929 (+1.6 %) | +0.5 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.338 (+4.0 %) [+4.0, +4.1] | +0.462 (+5.5 %) | -1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.437 (+4.4 %) [+4.4, +4.5] | +0.558 (+5.6 %) | -1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.102 (+8.0 %) [+7.7, +8.7] | +0.881 (+6.4 %) | +1.6 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +2.113 (+7.6 %) [+6.5, +8.1] | +1.348 (+4.8 %) | +2.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +3.219 (+5.7 %) [+5.2, +6.0] | +2.354 (+4.2 %) | +1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +6.825 (+5.5 %) [+5.2, +5.9] | +5.146 (+4.2 %) | +1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +12.934 (+4.3 %) [+4.1, +4.6] | +10.436 (+3.5 %) | +0.8 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +13.253 (+5.2 %) [+4.8, +5.5] | +10.436 (+4.1 %) | +1.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.761 (+25.2 %) [+25.2, +25.2] | +2.018 (+28.9 %) | -3.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +1.557 (+17.7 %) [+17.6, +17.7] | +1.798 (+20.4 %) | -2.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.441 (+10.8 %) [+10.1, +12.6] | +1.196 (+9.0 %) | +1.8 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +1.558 (+5.5 %) [+5.1, +6.6] | +1.552 (+5.5 %) | +0.0 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +3.411 (+6.1 %) [+5.7, +6.3] | +2.266 (+4.0 %) | +2.0 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +6.945 (+5.6 %) [+5.4, +5.8] | +5.002 (+4.1 %) | +1.6 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +14.147 (+4.7 %) [+4.6, +4.8] | +10.014 (+3.3 %) | +1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +14.376 (+5.7 %) [+5.5, +5.7] | +10.014 (+4.0 %) | +1.7 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.095 (+0.8 %) [+0.8, +0.9] | +0.369 (+3.2 %) | -2.4 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.407 (+2.7 %) [+0.7, +3.1] | +0.175 (+1.2 %) | +1.5 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.654 (+2.4 %) [+2.2, +2.7] | +0.483 (+1.8 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +1.150 (+2.1 %) [+1.8, +2.1] | +0.850 (+1.5 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +2.123 (+1.9 %) [+1.7, +2.0] | +1.530 (+1.4 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +4.050 (+1.7 %) [+1.6, +1.9] | +3.108 (+1.3 %) | +0.4 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +8.708 (+1.6 %) [+1.4, +1.7] | +6.925 (+1.2 %) | +0.3 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +8.375 (+1.7 %) [+1.6, +1.8] | +6.925 (+1.4 %) | +0.3 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.096 (+0.8 %) [+0.8, +0.9] | +0.363 (+3.2 %) | -2.3 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.446 (+3.0 %) [+2.7, +3.9] | +0.175 (+1.2 %) | +1.8 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.661 (+2.4 %) [+2.2, +2.8] | +0.485 (+1.8 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +1.160 (+2.1 %) [+1.8, +2.5] | +0.868 (+1.6 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +2.246 (+2.0 %) [+1.8, +2.2] | +1.586 (+1.4 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +4.444 (+1.9 %) [+1.6, +2.1] | +3.241 (+1.4 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +9.338 (+1.7 %) [+1.5, +1.9] | +6.125 (+1.1 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +9.344 (+1.9 %) [+1.7, +2.1] | +6.125 (+1.3 %) | +0.7 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.355 (+2.9 %) [+2.8, +2.9] | +0.469 (+3.8 %) | -0.9 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.633 (+4.2 %) [+3.4, +5.3] | +0.692 (+4.6 %) | -0.4 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.710 (+6.1 %) [+5.1, +6.5] | +1.314 (+4.7 %) | +1.4 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +3.067 (+5.5 %) [+5.1, +5.7] | +2.502 (+4.5 %) | +1.0 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +6.176 (+5.5 %) [+5.4, +5.8] | +4.348 (+3.9 %) | +1.6 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +12.322 (+5.2 %) [+4.9, +5.3] | +9.107 (+3.8 %) | +1.3 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +24.039 (+4.3 %) [+4.1, +4.4] | +21.357 (+3.8 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +24.228 (+4.9 %) [+4.8, +5.1] | +21.357 (+4.3 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.318 (+11.6 %) [+11.6, +11.6] | +1.979 (+17.4 %) | -5.8 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +0.883 (+5.9 %) [+5.4, +6.3] | +1.372 (+9.2 %) | -3.3 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.964 (+7.1 %) [+5.6, +7.3] | +1.307 (+4.7 %) | +2.4 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +3.321 (+6.0 %) [+5.5, +6.3] | +2.409 (+4.4 %) | +1.7 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +6.701 (+6.0 %) [+5.5, +6.4] | +4.147 (+3.7 %) | +2.3 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +13.521 (+5.7 %) [+5.2, +5.8] | +8.646 (+3.6 %) | +2.1 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +27.502 (+4.9 %) [+4.5, +5.1] | +18.966 (+3.4 %) | +1.5 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +27.749 (+5.7 %) [+5.2, +5.8] | +18.966 (+3.9 %) | +1.8 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.093 (+0.2 %) [-0.0, +0.2] | +0.897 (+1.6 %) | -1.4 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.405 (+0.6 %) [+0.4, +0.7] | +0.513 (+0.7 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +1.225 (+1.1 %) [+0.9, +1.2] | +1.154 (+1.1 %) | +0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +2.361 (+1.2 %) [+0.8, +1.4] | +1.306 (+0.7 %) | +0.5 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +4.218 (+1.1 %) [+1.1, +1.4] | +3.062 (+0.8 %) | +0.3 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +7.745 (+1.0 %) [+0.8, +1.1] | +5.734 (+0.7 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +15.083 (+0.8 %) [+0.7, +1.0] | +11.677 (+0.6 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +16.546 (+1.0 %) [+0.8, +1.1] | +11.677 (+0.7 %) | +0.3 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.121 (+0.2 %) [+0.2, +0.5] | +0.910 (+1.6 %) | -1.4 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.408 (+0.6 %) [+0.4, +0.6] | +0.542 (+0.7 %) | -0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +1.113 (+1.0 %) [+1.0, +1.3] | +1.244 (+1.1 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +2.375 (+1.2 %) [+0.9, +1.3] | +1.433 (+0.7 %) | +0.5 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +4.097 (+1.1 %) [+0.9, +1.2] | +3.212 (+0.8 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +8.369 (+1.0 %) [+0.8, +1.1] | +5.868 (+0.7 %) | +0.3 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +16.112 (+0.9 %) [+0.7, +1.0] | +12.263 (+0.7 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +16.255 (+1.0 %) [+0.9, +1.0] | +12.263 (+0.7 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.531 (+0.9 %) [+0.6, +1.0] | +1.104 (+1.8 %) | -0.9 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +1.146 (+1.5 %) [+1.5, +1.6] | +1.538 (+2.1 %) | -0.5 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +2.774 (+2.5 %) [+2.3, +2.7] | +2.668 (+2.4 %) | +0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +5.500 (+2.8 %) [+2.5, +2.9] | +4.412 (+2.3 %) | +0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +11.283 (+3.0 %) [+2.7, +3.0] | +8.243 (+2.2 %) | +0.8 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +22.082 (+2.7 %) [+2.4, +2.8] | +16.792 (+2.1 %) | +0.7 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +42.001 (+2.3 %) [+2.2, +2.5] | +34.603 (+1.9 %) | +0.4 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +44.064 (+2.6 %) [+2.4, +2.7] | +34.603 (+2.1 %) | +0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +3.565 (+6.2 %) [+6.1, +6.2] | +5.079 (+8.8 %) | -2.6 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +3.232 (+4.4 %) [+4.2, +4.5] | +4.069 (+5.6 %) | -1.1 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +3.288 (+3.0 %) [+2.7, +3.1] | +3.990 (+3.7 %) | -0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +6.484 (+3.3 %) [+2.8, +3.7] | +5.119 (+2.6 %) | +0.7 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +12.607 (+3.3 %) [+3.0, +3.4] | +8.643 (+2.3 %) | +1.0 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +23.356 (+2.9 %) [+2.6, +3.0] | +15.983 (+2.0 %) | +0.9 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +45.931 (+2.5 %) [+2.3, +2.7] | +33.462 (+1.8 %) | +0.7 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +47.068 (+2.8 %) [+2.6, +2.9] | +33.462 (+2.0 %) | +0.8 |  |

### GEMM kernel time per shape, µs (wN = CTA tile width picked by the tile table)

| model | projection | out x in | modules | T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | NVFP4 quantizer | FourOverSix quantizer |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.2 | 10.6 (w32) | 10.6 (w32) | 16.7 | 4.3 | 3.8 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 256 | 11.2 (w64) | 15.4 | 11.6 (w64) | 11.6 (w64) | 16.9 | 4.8 | 4.5 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 512 | 16.2 (w128) | 15.9 | 16.7 (w128) | 16.7 (w128) | 17.5 | 7.9 | 7.4 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 1024 | 30.6 (w128) | 30.4 | 31.6 (w128) | 31.6 (w128) | 34.4 | 12.0 | 13.7 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 2048 | 46.9 (w128) | 46.5 | 49.0 (w128) | 48.7 (w128) | 52.2 | 21.5 | 23.6 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 4096 | 99.1 (w128) | 98.0 | 103.9 (w128) | 103.7 (w128) | 111.8 | 34.7 | 39.3 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 8192 | 193.4 (w128) | 187.7 | 204.4 (w128) | 204.2 (w128) | 217.5 | 80.1 | 85.7 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 128 | 6.6 (w16) | 15.0 | 7.7 (w16) | 7.6 (w16) | 16.5 | 4.3 | 3.8 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.5 (w32) | 8.5 (w32) | 16.7 | 4.7 | 4.4 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.1 | 9.8 (w32) | 9.7 (w32) | 16.8 | 7.9 | 7.4 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.5 (w64) | 11.4 (w64) | 16.8 | 11.9 | 13.7 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 2048 | 16.0 (w128) | 16.1 | 16.8 (w128) | 16.7 (w128) | 17.6 | 19.5 | 21.6 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 4096 | 30.5 (w128) | 30.7 | 31.9 (w128) | 32.2 (w128) | 36.5 | 34.9 | 39.7 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 8192 | 46.7 (w128) | 46.6 | 50.3 (w128) | 50.0 (w128) | 57.8 | 67.7 | 77.8 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 128 | 6.7 (w16) | 15.0 | 7.7 (w16) | 7.7 (w16) | 16.5 | 4.3 | 3.8 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.5 (w32) | 8.6 (w32) | 16.7 | 4.7 | 4.4 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.2 | 9.8 (w32) | 9.8 (w32) | 16.7 | 7.9 | 7.4 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.4 | 11.5 (w64) | 11.5 (w64) | 16.8 | 11.9 | 13.7 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 2048 | 16.1 (w128) | 16.0 | 16.8 (w128) | 16.8 (w128) | 17.5 | 19.5 | 21.6 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 4096 | 30.7 (w128) | 31.1 | 31.8 (w128) | 32.2 (w128) | 37.1 | 34.7 | 39.6 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 8192 | 47.0 (w128) | 46.6 | 49.7 (w128) | 49.6 (w128) | 56.7 | 67.8 | 77.8 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.1 | 10.5 (w32) | 10.5 (w32) | 16.7 | 4.3 | 3.8 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.6 (w64) | 11.5 (w64) | 17.0 | 4.8 | 4.5 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 512 | 16.3 (w128) | 16.1 | 16.8 (w128) | 16.8 (w128) | 17.7 | 7.9 | 7.4 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 1024 | 30.7 (w128) | 30.5 | 31.6 (w128) | 31.7 (w128) | 33.4 | 12.5 | 13.8 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 2048 | 46.9 (w128) | 46.8 | 48.9 (w128) | 48.9 (w128) | 51.9 | 21.9 | 24.1 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 4096 | 100.1 (w128) | 100.1 | 103.8 (w128) | 104.5 (w128) | 112.2 | 34.8 | 39.3 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 8192 | 196.0 (w128) | 193.1 | 205.0 (w128) | 204.8 (w128) | 217.4 | 80.7 | 86.0 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 128 | 16.3 (w128) | 16.1 | 16.9 (w128) | 16.9 (w128) | 17.6 | 4.3 | 3.8 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 256 | 30.7 (w128) | 30.6 | 31.9 (w128) | 32.0 (w128) | 33.6 | 4.8 | 4.6 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 512 | 45.9 (w128) | 46.9 | 49.9 (w128) | 49.0 (w128) | 53.5 | 9.0 | 8.4 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 1024 | 85.4 (w128) | 85.3 | 89.6 (w128) | 89.9 (w128) | 95.2 | 12.9 | 14.8 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 2048 | 175.7 (w128) | 174.7 | 183.6 (w128) | 183.3 (w128) | 194.7 | 21.1 | 23.2 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 4096 | 360.9 (w128) | 358.6 | 376.1 (w128) | 376.0 (w128) | 401.1 | 42.0 | 45.1 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 8192 | 707.9 (w128) | 704.6 | 739.3 (w128) | 738.0 (w128) | 787.8 | 80.8 | 86.2 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 128 | 16.2 (w128) | 16.1 | 16.9 (w128) | 17.0 (w128) | 17.7 | 4.3 | 3.8 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 256 | 30.8 (w128) | 30.7 | 32.2 (w128) | 32.3 (w128) | 34.0 | 4.7 | 4.5 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 512 | 47.0 (w128) | 46.7 | 49.3 (w128) | 49.2 (w128) | 54.7 | 8.9 | 8.3 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 1024 | 85.4 (w128) | 85.7 | 90.1 (w128) | 90.1 (w128) | 95.5 | 12.8 | 14.8 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 2048 | 175.8 (w128) | 174.8 | 183.8 (w128) | 184.0 (w128) | 195.5 | 21.2 | 23.2 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 4096 | 360.1 (w128) | 358.4 | 377.3 (w128) | 377.6 (w128) | 402.2 | 42.1 | 45.2 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 8192 | 706.3 (w128) | 703.1 | 739.0 (w128) | 740.7 (w128) | 791.2 | 80.6 | 86.2 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 128 | 25.2 (w64) | 45.5 | 27.7 (w32) | 27.7 (w32) | 50.4 | 12.0 | 10.8 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 256 | 31.3 (w64) | 45.7 | 31.7 (w64) | 32.1 (w64) | 50.8 | 15.9 | 14.2 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 512 | 48.3 (w128) | 47.8 | 49.5 (w128) | 49.7 (w128) | 52.6 | 26.1 | 23.4 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 1024 | 111.6 (w64) | 108.3 | 114.8 (w64) | 115.3 (w64) | 121.0 | 38.9 | 45.1 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 2048 | 163.9 (w128) | 163.9 | 172.1 (w128) | 172.3 (w128) | 183.4 | 67.1 | 73.5 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 4096 | 343.0 (w128) | 344.7 | 357.8 (w128) | 359.0 (w128) | 385.5 | 160.9 | 161.7 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 8192 | 668.0 (w128) | 671.8 | 701.0 (w128) | 702.0 (w128) | 753.3 | 283.6 | 302.7 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.1 | 10.6 (w32) | 10.5 (w32) | 16.7 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.7 (w64) | 11.6 (w64) | 17.1 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 512 | 16.2 (w128) | 16.0 | 16.7 (w128) | 16.8 (w128) | 17.8 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 1024 | 30.6 (w128) | 30.5 | 31.6 (w128) | 31.7 (w128) | 33.5 | 12.0 | 13.8 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 2048 | 47.1 (w128) | 46.0 | 49.0 (w128) | 48.6 (w128) | 52.0 | 21.7 | 23.6 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 4096 | 98.6 (w128) | 96.6 | 104.0 (w128) | 102.5 (w128) | 111.0 | 34.7 | 39.5 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 8192 | 192.1 (w128) | 184.2 | 204.8 (w128) | 201.4 (w128) | 218.0 | 80.3 | 85.8 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 128 | 6.6 (w16) | 15.0 | 7.9 (w16) | 7.6 (w16) | 16.7 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.8 (w32) | 8.6 (w32) | 17.1 | 4.7 | 4.4 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.1 | 9.8 (w32) | 9.8 (w32) | 17.0 | 7.9 | 7.4 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.6 (w64) | 11.4 (w64) | 17.0 | 11.9 | 13.7 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 2048 | 16.1 (w128) | 16.1 | 16.9 (w128) | 16.7 (w128) | 17.7 | 19.6 | 21.7 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 4096 | 31.0 (w128) | 30.8 | 32.4 (w128) | 32.6 (w128) | 37.5 | 34.9 | 39.4 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 8192 | 46.4 (w128) | 46.1 | 49.3 (w128) | 50.3 (w128) | 57.8 | 67.6 | 77.6 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 128 | 6.7 (w16) | 15.0 | 7.7 (w16) | 7.7 (w16) | 16.6 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.5 (w32) | 8.6 (w32) | 16.8 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.1 | 9.8 (w32) | 9.8 (w32) | 16.8 | 7.9 | 7.4 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.4 (w64) | 11.4 (w64) | 16.9 | 11.9 | 13.7 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 2048 | 16.0 (w128) | 16.1 | 16.7 (w128) | 16.7 (w128) | 17.7 | 19.5 | 21.6 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 4096 | 31.1 (w128) | 31.0 | 33.0 (w128) | 31.8 (w128) | 37.2 | 34.8 | 39.6 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 8192 | 46.6 (w128) | 46.8 | 50.2 (w128) | 49.4 (w128) | 57.8 | 68.1 | 78.0 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.1 | 10.5 (w32) | 10.6 (w32) | 16.8 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 256 | 11.2 (w64) | 15.4 | 11.6 (w64) | 11.7 (w64) | 17.0 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 512 | 16.2 (w128) | 16.1 | 16.7 (w128) | 16.8 (w128) | 17.7 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 1024 | 30.7 (w128) | 30.5 | 31.7 (w128) | 31.9 (w128) | 33.9 | 12.5 | 13.8 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 2048 | 47.3 (w128) | 47.1 | 49.3 (w128) | 49.7 (w128) | 52.8 | 21.9 | 24.1 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 4096 | 100.2 (w128) | 100.4 | 104.7 (w128) | 104.5 (w128) | 111.4 | 34.9 | 39.4 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 8192 | 197.0 (w128) | 193.6 | 205.7 (w128) | 206.4 (w128) | 219.1 | 80.8 | 86.3 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 128 | 16.2 (w128) | 16.1 | 16.9 (w128) | 16.9 (w128) | 17.7 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 256 | 30.8 (w128) | 30.6 | 31.9 (w128) | 32.2 (w128) | 33.5 | 4.8 | 4.6 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 512 | 45.8 (w128) | 45.6 | 47.9 (w128) | 48.0 (w128) | 53.1 | 9.0 | 8.5 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 1024 | 85.7 (w128) | 85.9 | 89.7 (w128) | 89.9 (w128) | 95.0 | 12.9 | 14.8 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 2048 | 176.0 (w128) | 175.5 | 183.6 (w128) | 183.5 (w128) | 194.4 | 21.2 | 23.2 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 4096 | 361.6 (w128) | 359.9 | 375.8 (w128) | 376.6 (w128) | 400.3 | 42.3 | 45.3 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 8192 | 709.2 (w128) | 706.4 | 737.7 (w128) | 739.2 (w128) | 785.9 | 81.1 | 86.4 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 128 | 16.3 (w128) | 16.1 | 16.9 (w128) | 16.9 (w128) | 17.6 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 256 | 30.6 (w128) | 30.6 | 31.9 (w128) | 32.2 (w128) | 33.6 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 512 | 46.9 (w128) | 45.6 | 48.8 (w128) | 49.2 (w128) | 53.8 | 9.0 | 8.3 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 1024 | 85.3 (w128) | 86.0 | 90.2 (w128) | 90.5 (w128) | 95.4 | 12.7 | 14.9 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 2048 | 176.5 (w128) | 175.5 | 184.2 (w128) | 184.6 (w128) | 195.9 | 21.2 | 23.2 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 4096 | 361.0 (w128) | 359.6 | 376.8 (w128) | 377.9 (w128) | 401.8 | 42.3 | 45.2 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 8192 | 708.5 (w128) | 704.8 | 739.7 (w128) | 741.8 (w128) | 789.4 | 81.2 | 86.6 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 128 | 25.2 (w64) | 45.5 | 27.7 (w32) | 27.6 (w32) | 50.4 | 12.0 | 10.8 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 256 | 31.4 (w64) | 45.9 | 31.6 (w64) | 31.7 (w64) | 50.5 | 15.9 | 14.2 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 512 | 47.7 (w128) | 47.4 | 49.4 (w128) | 49.6 (w128) | 52.2 | 26.2 | 23.4 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 1024 | 111.4 (w64) | 108.4 | 115.1 (w64) | 115.4 (w64) | 122.3 | 39.0 | 45.2 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 2048 | 164.1 (w128) | 164.0 | 172.0 (w128) | 172.4 (w128) | 183.3 | 67.1 | 73.6 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 4096 | 344.1 (w128) | 344.9 | 358.5 (w128) | 358.8 (w128) | 384.8 | 161.1 | 162.0 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 8192 | 668.7 (w128) | 673.5 | 702.1 (w128) | 702.7 (w128) | 753.3 | 284.2 | 302.7 |
| Phi-4 | o_proj | 5120x5120 | 40 | 128 | 11.1 (w64) | 18.3 | 12.9 (w64) | 13.0 (w64) | 20.0 | 4.9 | 4.6 |
| Phi-4 | o_proj | 5120x5120 | 40 | 256 | 14.8 (w64) | 18.7 | 15.2 (w64) | 15.2 (w64) | 20.4 | 7.7 | 6.8 |
| Phi-4 | o_proj | 5120x5120 | 40 | 512 | 20.2 (w128) | 20.2 | 20.7 (w128) | 20.6 (w128) | 21.5 | 11.1 | 10.0 |
| Phi-4 | o_proj | 5120x5120 | 40 | 1024 | 37.6 (w128) | 37.5 | 38.8 (w128) | 38.9 (w128) | 41.9 | 17.1 | 20.5 |
| Phi-4 | o_proj | 5120x5120 | 40 | 2048 | 86.2 (w128) | 86.1 | 89.1 (w128) | 89.2 (w128) | 93.9 | 26.2 | 30.1 |
| Phi-4 | o_proj | 5120x5120 | 40 | 4096 | 146.2 (w128) | 146.0 | 153.6 (w128) | 154.2 (w128) | 164.0 | 49.2 | 55.4 |
| Phi-4 | o_proj | 5120x5120 | 40 | 8192 | 301.6 (w128) | 298.8 | 314.6 (w128) | 314.8 (w128) | 335.4 | 105.3 | 116.1 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 128 | 13.2 (w64) | 18.6 | 14.0 (w64) | 13.9 (w64) | 20.3 | 4.9 | 4.6 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 256 | 19.4 (w128) | 19.3 | 19.9 (w128) | 20.0 (w128) | 21.1 | 7.7 | 6.8 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 512 | 37.8 (w128) | 37.7 | 38.8 (w128) | 38.9 (w128) | 42.4 | 11.3 | 10.2 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 1024 | 59.2 (w128) | 57.4 | 61.3 (w128) | 61.9 (w128) | 68.3 | 18.3 | 21.4 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 2048 | 123.0 (w128) | 122.8 | 126.7 (w128) | 127.0 (w128) | 135.4 | 26.2 | 30.2 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 4096 | 237.5 (w128) | 235.3 | 247.1 (w128) | 247.6 (w128) | 262.5 | 49.8 | 55.7 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 8192 | 460.4 (w128) | 457.1 | 477.6 (w128) | 478.9 (w128) | 510.5 | 106.3 | 116.7 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 128 | 40.2 (w128) | 40.2 | 40.8 (w128) | 40.7 (w128) | 43.2 | 5.1 | 4.9 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 256 | 69.6 (w128) | 69.2 | 72.2 (w128) | 72.2 (w128) | 77.6 | 12.1 | 10.6 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 512 | 144.1 (w128) | 144.2 | 150.6 (w128) | 150.6 (w128) | 161.7 | 18.5 | 17.0 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 1024 | 284.6 (w128) | 284.5 | 296.3 (w128) | 296.5 (w128) | 314.6 | 23.9 | 27.0 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 2048 | 558.7 (w128) | 554.5 | 579.9 (w128) | 580.4 (w128) | 616.5 | 36.9 | 38.6 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 4096 | 1102.8 (w128) | 1096.0 | 1144.9 (w128) | 1146.3 (w128) | 1222.4 | 56.5 | 61.4 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 8192 | 2452.7 (w128) | 2416.9 | 2563.7 (w128) | 2559.9 (w128) | 2729.2 | 115.3 | 126.3 |
| Phi-4 | down_proj | 5120x17920 | 40 | 128 | 31.2 (w64) | 56.3 | 37.1 (w64) | 37.1 (w64) | 61.6 | 13.0 | 12.0 |
| Phi-4 | down_proj | 5120x17920 | 40 | 256 | 43.3 (w64) | 57.0 | 44.2 (w64) | 44.1 (w64) | 62.4 | 21.5 | 18.8 |
| Phi-4 | down_proj | 5120x17920 | 40 | 512 | 64.0 (w128) | 63.8 | 68.1 (w128) | 68.1 (w128) | 73.2 | 38.4 | 34.9 |
| Phi-4 | down_proj | 5120x17920 | 40 | 1024 | 134.6 (w128) | 134.4 | 140.8 (w128) | 140.4 (w128) | 151.4 | 49.9 | 58.3 |
| Phi-4 | down_proj | 5120x17920 | 40 | 2048 | 284.8 (w128) | 284.2 | 295.2 (w128) | 295.7 (w128) | 310.5 | 116.4 | 122.0 |
| Phi-4 | down_proj | 5120x17920 | 40 | 4096 | 519.3 (w128) | 517.1 | 537.9 (w128) | 538.8 (w128) | 573.0 | 203.6 | 208.8 |
| Phi-4 | down_proj | 5120x17920 | 40 | 8192 | 1088.7 (w128) | 1070.9 | 1120.6 (w128) | 1103.0 (w128) | 1202.6 | 356.4 | 386.7 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 128 | 12.7 (w64) | 21.3 | 14.9 (w64) | 14.9 (w64) | 23.4 | 4.9 | 4.7 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 256 | 17.5 (w64) | 21.8 | 18.0 (w64) | 18.1 (w64) | 24.0 | 8.0 | 7.2 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 512 | 23.3 (w128) | 23.1 | 23.9 (w128) | 24.0 (w128) | 25.2 | 11.5 | 10.5 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 1024 | 45.5 (w128) | 45.1 | 46.7 (w128) | 46.8 (w128) | 50.3 | 18.8 | 21.5 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 2048 | 98.8 (w128) | 98.5 | 102.3 (w128) | 102.5 (w128) | 108.7 | 28.6 | 31.9 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 4096 | 177.9 (w128) | 176.6 | 184.6 (w128) | 184.9 (w128) | 196.4 | 53.7 | 59.9 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 8192 | 360.5 (w128) | 357.6 | 374.7 (w128) | 374.8 (w128) | 399.9 | 123.2 | 130.8 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 128 | 15.5 (w64) | 18.8 | 16.0 (w64) | 16.0 (w64) | 20.7 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 256 | 20.1 (w128) | 20.0 | 20.7 (w128) | 20.7 (w128) | 21.9 | 7.7 | 6.8 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 512 | 38.0 (w128) | 37.9 | 39.0 (w128) | 39.3 (w128) | 42.7 | 11.7 | 10.7 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 1024 | 88.2 (w128) | 88.2 | 90.8 (w128) | 91.1 (w128) | 94.5 | 16.3 | 19.7 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 2048 | 146.0 (w128) | 146.1 | 153.0 (w128) | 153.2 (w128) | 163.2 | 29.5 | 33.4 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 4096 | 311.1 (w128) | 308.4 | 323.4 (w128) | 322.9 (w128) | 344.2 | 56.3 | 61.4 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 8192 | 618.6 (w128) | 615.1 | 644.0 (w128) | 643.9 (w128) | 685.4 | 104.3 | 113.9 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 128 | 11.6 (w64) | 18.4 | 13.2 (w64) | 13.3 (w64) | 20.1 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 256 | 19.2 (w128) | 18.9 | 19.4 (w128) | 19.6 (w128) | 20.7 | 7.6 | 6.8 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 512 | 34.3 (w64) | 37.5 | 36.6 (w64) | 36.8 (w64) | 42.9 | 11.2 | 10.1 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 1024 | 62.4 (w128) | 57.9 | 63.1 (w128) | 64.6 (w128) | 70.2 | 17.2 | 20.4 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 2048 | 106.2 (w128) | 106.4 | 109.5 (w128) | 109.6 (w128) | 115.6 | 25.8 | 29.8 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 4096 | 189.0 (w128) | 187.6 | 195.8 (w128) | 195.9 (w128) | 208.5 | 46.0 | 52.1 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 8192 | 365.5 (w128) | 363.1 | 380.0 (w128) | 380.3 (w128) | 404.5 | 105.4 | 116.2 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 128 | 7.1 (w16) | 18.1 | 8.3 (w16) | 8.3 (w16) | 19.5 | 4.8 | 4.6 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 256 | 7.1 (w16) | 18.1 | 8.3 (w16) | 8.3 (w16) | 19.4 | 7.7 | 6.8 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 512 | 7.7 (w32) | 18.1 | 9.4 (w32) | 9.4 (w32) | 19.5 | 11.1 | 10.0 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 1024 | 9.9 (w32) | 18.2 | 10.1 (w32) | 10.1 (w32) | 19.5 | 16.3 | 19.6 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 2048 | 10.7 (w64) | 18.4 | 12.3 (w64) | 12.4 (w64) | 20.0 | 25.9 | 29.7 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 4096 | 17.7 (w64) | 18.9 | 17.8 (w64) | 17.9 (w64) | 20.7 | 45.3 | 51.7 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 8192 | 20.2 (w128) | 20.2 | 20.4 (w128) | 20.4 (w128) | 20.7 | 84.7 | 99.5 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 128 | 7.1 (w16) | 18.1 | 8.2 (w16) | 8.3 (w16) | 19.6 | 4.8 | 4.6 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 256 | 7.1 (w16) | 18.1 | 8.3 (w16) | 8.3 (w16) | 19.5 | 7.6 | 6.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 512 | 7.7 (w32) | 18.1 | 9.4 (w32) | 9.4 (w32) | 19.5 | 11.0 | 9.9 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 1024 | 10.0 (w32) | 18.2 | 10.2 (w32) | 10.1 (w32) | 19.6 | 16.5 | 19.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 2048 | 10.6 (w64) | 18.4 | 12.3 (w64) | 12.2 (w64) | 19.8 | 25.8 | 29.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 4096 | 17.4 (w64) | 18.5 | 17.6 (w64) | 17.6 (w64) | 20.1 | 45.3 | 51.5 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 8192 | 20.2 (w128) | 20.2 | 20.3 (w128) | 20.4 (w128) | 20.8 | 84.7 | 99.6 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 128 | 20.0 (w128) | 19.9 | 20.6 (w128) | 20.6 (w128) | 21.6 | 4.9 | 4.6 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 256 | 38.4 (w128) | 38.3 | 39.7 (w128) | 39.6 (w128) | 42.0 | 7.9 | 7.0 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 512 | 61.1 (w128) | 60.8 | 64.3 (w128) | 64.4 (w128) | 68.8 | 14.9 | 13.5 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 1024 | 124.9 (w128) | 124.6 | 129.5 (w128) | 129.2 (w128) | 138.2 | 16.7 | 20.3 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 2048 | 271.2 (w128) | 269.6 | 282.2 (w128) | 282.2 (w128) | 300.1 | 32.9 | 33.6 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 4096 | 534.6 (w128) | 531.6 | 556.2 (w128) | 556.4 (w128) | 592.7 | 55.4 | 60.2 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 8192 | 1058.7 (w128) | 1055.8 | 1100.9 (w128) | 1101.2 (w128) | 1175.8 | 105.8 | 115.0 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 128 | 19.9 (w128) | 19.8 | 20.7 (w128) | 20.7 (w128) | 21.6 | 4.9 | 4.6 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 256 | 38.2 (w128) | 38.1 | 39.2 (w128) | 39.3 (w128) | 43.0 | 8.0 | 7.1 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 512 | 61.5 (w128) | 61.5 | 64.4 (w128) | 64.5 (w128) | 68.8 | 14.4 | 12.6 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 1024 | 124.4 (w128) | 124.5 | 129.7 (w128) | 130.0 (w128) | 138.5 | 16.7 | 20.3 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 2048 | 272.1 (w128) | 270.0 | 283.2 (w128) | 283.5 (w128) | 301.4 | 32.7 | 33.6 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 4096 | 535.4 (w128) | 531.5 | 558.3 (w128) | 559.0 (w128) | 595.0 | 55.6 | 60.2 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 8192 | 1059.9 (w128) | 1055.5 | 1104.1 (w128) | 1105.6 (w128) | 1178.2 | 106.0 | 115.1 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 128 | 30.3 (w64) | 54.7 | 36.6 (w64) | 36.7 (w64) | 60.3 | 12.8 | 11.7 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 256 | 41.6 (w64) | 55.2 | 42.8 (w64) | 43.0 (w64) | 61.2 | 20.1 | 18.3 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 512 | 60.9 (w128) | 60.4 | 65.5 (w128) | 66.0 (w128) | 71.0 | 38.8 | 35.1 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 1024 | 131.1 (w128) | 130.9 | 136.4 (w128) | 136.9 (w128) | 147.9 | 49.0 | 57.1 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 2048 | 275.6 (w128) | 274.4 | 285.6 (w128) | 287.2 (w128) | 302.9 | 110.2 | 117.1 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 4096 | 503.8 (w128) | 501.4 | 522.7 (w128) | 523.6 (w128) | 559.1 | 197.0 | 202.9 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 8192 | 1036.0 (w128) | 1034.1 | 1079.0 (w128) | 1085.6 (w128) | 1179.5 | 344.5 | 369.3 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 128 | 19.2 (w128) | 19.0 | 19.8 (w128) | 19.9 (w128) | 20.8 | 4.9 | 4.6 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 256 | 34.2 (w64) | 37.3 | 36.8 (w64) | 36.7 (w64) | 43.2 | 8.1 | 7.2 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 512 | 57.1 (w128) | 56.6 | 61.3 (w128) | 62.1 (w128) | 69.1 | 12.9 | 11.9 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 1024 | 108.6 (w128) | 108.7 | 112.3 (w128) | 112.2 (w128) | 119.0 | 16.5 | 19.5 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 2048 | 192.1 (w128) | 190.6 | 198.9 (w128) | 199.0 (w128) | 210.7 | 26.4 | 30.4 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 4096 | 379.9 (w128) | 376.1 | 395.0 (w128) | 395.5 (w128) | 421.1 | 55.6 | 60.6 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 8192 | 747.1 (w128) | 741.5 | 775.6 (w128) | 777.0 (w128) | 828.5 | 106.0 | 115.3 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 128 | 7.8 (w16) | 18.1 | 8.8 (w16) | 8.9 (w16) | 19.8 | 4.8 | 4.6 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 256 | 8.1 (w32) | 18.1 | 9.9 (w32) | 10.0 (w32) | 20.0 | 7.7 | 6.8 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 512 | 10.7 (w64) | 18.2 | 11.6 (w32) | 11.7 (w32) | 20.0 | 11.1 | 10.0 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 1024 | 13.0 (w64) | 18.4 | 13.5 (w64) | 13.5 (w64) | 20.0 | 15.9 | 19.4 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 2048 | 19.3 (w128) | 19.4 | 19.9 (w128) | 19.9 (w128) | 20.9 | 25.9 | 29.7 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 4096 | 37.2 (w128) | 37.3 | 38.3 (w128) | 39.7 (w128) | 44.7 | 45.5 | 52.0 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 8192 | 59.3 (w128) | 59.0 | 61.9 (w128) | 62.1 (w128) | 66.7 | 86.3 | 101.1 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 128 | 7.7 (w16) | 18.1 | 8.9 (w16) | 9.0 (w16) | 19.9 | 4.9 | 4.6 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 256 | 8.1 (w32) | 18.1 | 10.0 (w32) | 10.1 (w32) | 20.1 | 7.7 | 6.8 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 512 | 10.7 (w64) | 18.1 | 11.8 (w32) | 11.9 (w32) | 20.2 | 11.0 | 9.9 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 1024 | 13.0 (w64) | 18.3 | 13.6 (w64) | 13.6 (w64) | 20.1 | 15.9 | 19.4 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 2048 | 19.2 (w128) | 19.3 | 20.0 (w128) | 20.1 (w128) | 21.1 | 25.8 | 29.6 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 4096 | 37.2 (w128) | 37.2 | 40.1 (w128) | 38.8 (w128) | 45.7 | 45.4 | 51.8 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 8192 | 59.3 (w128) | 58.9 | 62.4 (w128) | 62.5 (w128) | 67.0 | 86.0 | 101.1 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 128 | 12.6 (w64) | 21.3 | 15.0 (w64) | 15.0 (w64) | 23.4 | 4.9 | 4.6 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 256 | 17.6 (w64) | 21.7 | 18.2 (w64) | 18.2 (w64) | 23.8 | 8.0 | 7.2 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 512 | 23.0 (w128) | 22.9 | 23.9 (w128) | 23.9 (w128) | 25.2 | 11.6 | 10.7 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 1024 | 45.6 (w128) | 45.9 | 47.4 (w128) | 47.4 (w128) | 50.0 | 18.6 | 21.1 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 2048 | 99.2 (w128) | 98.8 | 102.8 (w128) | 102.8 (w128) | 109.9 | 28.6 | 31.8 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 4096 | 179.7 (w128) | 178.6 | 186.8 (w128) | 187.1 (w128) | 198.7 | 54.1 | 60.2 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 8192 | 364.7 (w128) | 363.1 | 379.9 (w128) | 380.1 (w128) | 405.2 | 124.2 | 131.7 |
