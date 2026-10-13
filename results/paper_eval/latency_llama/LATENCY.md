# Table latency, part W: CUDA-graph prefill on the RTX PRO 6000 (Llama Instruct models; Mistral drift check)

PROTOCOL.md part W: A's protocol (58fe644) unchanged except the model list; flipquant paper-sm120-runs 6f5ad27 (its timed path is A's); records under RUN/latency_llama.

Medians over rounds of each process's median (ms); ratios paired within rounds, median [min, max] over rounds. Kernels: build_V. Harness: flipquant evaluation.latency prefill (PROTOCOL.md).

## Llama-3.2-1B-Instruct

| shape | BF16 | NVFP4 | FourOverSix | FourOverSix (wB) | FlipQuant 8x64 | FlipQuant 16x64 | FlipQuant 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 3.65 | 2.98 | 2.93 | 2.93 | 3.00 | 2.94 | 2.94 |
| 1x256 | 3.97 | 3.24 | 3.20 | 3.19 | 3.28 | 3.21 | 3.21 |
| 1x512 | 5.96 | 4.22 | 4.18 | 4.17 | 4.30 | 4.22 | 4.20 |
| 1x1024 | 10.97 | 6.48 | 6.57 | 6.58 | 6.71 | 6.59 | 6.59 |
| 1x2048 | 23.36 | 12.13 | 12.26 | 12.35 | 12.67 | 12.29 | 12.33 |
| 1x4096 | 48.18 | 29.77 | 29.90 | 30.03 | 30.78 | 30.16 | 30.00 |
| 1x8192 | 103.48 | 69.95 | 70.38 | 71.14 | 72.43 | 70.58 | 70.45 |
| 4x2048 | 91.54 | 58.12 | 58.36 | 59.09 | 60.44 | 58.62 | 58.51 |

| shape | FlipQuant 16x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix | FlipQuant 256x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix (wB) | NVFP4 vs FourOverSix | FourOverSix (wB) vs FourOverSix | BF16 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +0.2 % [+0.0, +0.5] | +2.3 % [+2.2, +2.4] | +0.2 % [+0.2, +0.4] | +2.6 % [+2.5, +2.7] | +1.5 % [+1.4, +1.7] | -0.3 % [-0.4, -0.2] | +24.3 % [+24.1, +24.5] |
| 1x256 | +0.4 % [+0.3, +0.6] | +2.3 % [+2.2, +2.4] | +0.2 % [+0.1, +0.3] | +2.7 % [+2.6, +2.8] | +1.1 % [+1.1, +1.3] | -0.4 % [-0.6, -0.3] | +24.0 % [+24.0, +24.2] |
| 1x512 | +0.8 % [+0.8, +1.1] | +2.9 % [+2.6, +2.9] | +0.4 % [+0.4, +0.5] | +3.2 % [+3.2, +3.3] | +0.9 % [+0.8, +1.1] | -0.3 % [-0.6, -0.3] | +41.8 % [+41.6, +42.7] |
| 1x1024 | +0.4 % [+0.4, +1.0] | +2.1 % [+2.0, +2.2] | +0.3 % [+0.1, +0.5] | +2.0 % [+1.9, +2.1] | -1.3 % [-1.4, -0.8] | +0.1 % [+0.0, +0.2] | +67.1 % [+66.8, +67.3] |
| 1x2048 | +0.2 % [-0.5, +2.5] | +3.7 % [+2.2, +4.6] | +0.6 % [-0.8, +1.9] | +3.1 % [+2.1, +3.2] | -1.1 % [-1.9, +0.2] | +0.7 % [-0.6, +1.5] | +90.7 % [+89.9, +93.5] |
| 1x4096 | +0.5 % [+0.4, +2.4] | +2.9 % [+2.7, +3.1] | +0.3 % [+0.2, +1.0] | +2.5 % [+1.7, +3.2] | -0.4 % [-0.7, +0.9] | +0.4 % [-0.1, +1.0] | +61.1 % [+60.6, +62.0] |
| 1x8192 | +0.3 % [+0.3, +1.9] | +2.9 % [+2.7, +3.2] | +0.2 % [+0.1, +0.6] | +1.8 % [+1.5, +2.2] | -0.5 % [-0.6, +0.6] | +1.1 % [+0.7, +1.2] | +47.1 % [+46.8, +47.8] |
| 4x2048 | +0.7 % [+0.4, +2.0] | +3.7 % [+3.3, +3.9] | +0.4 % [+0.3, +0.6] | +2.4 % [+2.2, +2.8] | -0.3 % [-0.4, +0.6] | +1.3 % [+0.8, +1.6] | +57.1 % [+56.9, +57.8] |

## Llama-3.1-8B-Instruct

| shape | BF16 | NVFP4 | FourOverSix | FourOverSix (wB) | FlipQuant 8x64 | FlipQuant 16x64 | FlipQuant 256x64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 16.66 | 7.53 | 7.44 | 8.87 | 7.47 | 7.53 | 7.44 |
| 1x256 | 22.64 | 9.43 | 9.35 | 10.51 | 9.51 | 9.41 | 9.38 |
| 1x512 | 39.32 | 15.59 | 15.29 | 15.84 | 16.77 | 15.79 | 15.86 |
| 1x1024 | 73.56 | 31.04 | 31.10 | 31.55 | 33.31 | 31.50 | 31.49 |
| 1x2048 | 137.60 | 63.07 | 63.22 | 63.84 | 66.10 | 63.20 | 63.25 |
| 1x4096 | 279.16 | 136.82 | 136.99 | 138.31 | 142.61 | 138.29 | 138.02 |
| 1x8192 | 610.18 | 328.53 | 328.36 | 330.99 | 340.15 | 331.17 | 331.42 |
| 4x2048 | 559.08 | 280.15 | 280.08 | 282.60 | 291.84 | 282.57 | 282.81 |

| shape | FlipQuant 16x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix | FlipQuant 256x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix (wB) | NVFP4 vs FourOverSix | FourOverSix (wB) vs FourOverSix | BF16 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | +1.2 % [+1.1, +1.3] | +0.3 % [+0.2, +0.5] | -0.1 % [-0.2, +0.0] | -15.8 % [-15.9, -15.6] | +1.2 % [+1.1, +1.2] | +19.1 % [+19.0, +19.2] | +123.6 % [+123.1, +124.0] |
| 1x256 | +0.7 % [+0.2, +0.8] | +1.8 % [+1.3, +2.1] | +0.4 % [+0.1, +0.4] | -9.5 % [-9.7, -9.2] | +0.8 % [+0.6, +0.9] | +12.4 % [+12.2, +12.7] | +142.2 % [+138.7, +142.9] |
| 1x512 | +2.4 % [+0.2, +6.0] | +9.6 % [+7.7, +11.9] | +2.9 % [+1.8, +6.1] | +6.3 % [+5.3, +7.6] | +1.6 % [+0.6, +3.1] | +3.4 % [+0.1, +4.8] | +157.7 % [+152.9, +164.9] |
| 1x1024 | +1.1 % [-0.4, +1.6] | +7.2 % [+6.4, +7.7] | +1.0 % [+0.6, +1.4] | +5.6 % [+5.0, +6.8] | -0.2 % [-1.1, -0.1] | +1.2 % [+0.8, +1.6] | +135.5 % [+129.5, +143.2] |
| 1x2048 | -0.0 % [-2.1, +0.1] | +4.7 % [+3.8, +4.9] | -0.2 % [-0.5, +0.3] | +3.5 % [+3.1, +4.2] | -0.5 % [-1.4, +0.0] | +0.7 % [+0.5, +1.3] | +117.0 % [+112.3, +117.7] |
| 1x4096 | +0.8 % [-1.3, +1.0] | +4.1 % [+3.3, +4.2] | +0.5 % [+0.1, +0.9] | +2.9 % [+2.6, +3.6] | -0.4 % [-1.3, +0.1] | +0.7 % [+0.5, +1.3] | +103.3 % [+100.0, +103.9] |
| 1x8192 | +0.8 % [-1.1, +1.0] | +3.5 % [+2.9, +3.7] | +0.6 % [+0.4, +1.0] | +2.7 % [+2.3, +3.2] | -0.3 % [-1.0, +0.2] | +0.6 % [+0.4, +1.0] | +85.6 % [+84.7, +86.4] |
| 4x2048 | +0.7 % [-0.9, +1.0] | +4.1 % [+3.6, +4.3] | +0.7 % [+0.4, +1.1] | +3.2 % [+2.9, +3.7] | -0.2 % [-1.0, +0.2] | +0.7 % [+0.5, +1.1] | +99.3 % [+98.8, +100.6] |

## Mistral-7B-Instruct-v0.3 (session-drift check: fo6 and fq-16x64 only)

| shape | FourOverSix | FlipQuant 16x64 |
|---|---:|---:|
| 1x128 | 6.97 | 7.04 |
| 1x256 | 8.73 | 8.77 |
| 1x512 | 13.41 | 13.68 |
| 1x1024 | 27.73 | 27.92 |
| 1x2048 | 56.87 | 56.35 |
| 1x4096 | 124.68 | 124.77 |
| 1x8192 | 303.37 | 303.83 |
| 4x2048 | 255.34 | 256.25 |

| shape | FlipQuant 16x64 vs FourOverSix |
|---|---:|
| 1x128 | +1.0 % [+1.0, +1.2] |
| 1x256 | +0.6 % [+0.4, +0.8] |
| 1x512 | +2.0 % [+1.5, +2.9] |
| 1x1024 | +1.6 % [+0.7, +3.1] |
| 1x2048 | +0.1 % [-0.9, +2.3] |
| 1x4096 | +0.8 % [+0.1, +2.6] |
| 1x8192 | +0.7 % [+0.2, +2.8] |
| 4x2048 | +1.0 % [+0.4, +2.8] |

## Summary: the median over the 8 shapes of the paired ratio

| model | FlipQuant 16x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix | FlipQuant 256x64 vs FourOverSix | FlipQuant 8x64 vs FourOverSix (wB) | NVFP4 vs FourOverSix | FourOverSix (wB) vs FourOverSix | BF16 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | +0.4 % | +2.9 % | +0.3 % | +2.5 % | -0.4 % | +0.3 % | +52.1 % |
| Llama-3.1-8B-Instruct | +0.8 % | +4.1 % | +0.6 % | +3.1 % | -0.2 % | +0.9 % | +120.3 % |
| Mistral-7B-Instruct-v0.3 (session-drift check: fo6 and fq-16x64 only) | +0.9 % | — | — | — | — | — | — |

## Session-drift check: Mistral-7B-Instruct-v0.3 against A (58fe644)

Median over rounds (graph ms), part W against A, and the difference; A's Mistral numbers are not replaced.

| shape | fo6 W | fo6 A | Δ | fq-16x64 W | fq-16x64 A | Δ | fq-16x64 vs fo6, W | the same, A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x128 | 6.97 | 6.98 | -0.04 % | 7.04 | 7.05 | -0.09 % | +1.0 % [+1.0, +1.2] | +1.0 % [+0.9, +1.0] |
| 1x256 | 8.73 | 8.75 | -0.21 % | 8.77 | 8.79 | -0.32 % | +0.6 % [+0.4, +0.8] | +0.6 % [+0.5, +0.7] |
| 1x512 | 13.41 | 13.48 | -0.50 % | 13.68 | 13.69 | -0.06 % | +2.0 % [+1.5, +2.9] | +2.3 % [-0.4, +3.4] |
| 1x1024 | 27.73 | 27.77 | -0.14 % | 27.92 | 28.13 | -0.74 % | +1.6 % [+0.7, +3.1] | +1.5 % [+1.0, +1.9] |
| 1x2048 | 56.87 | 57.07 | -0.35 % | 56.35 | 57.01 | -1.14 % | +0.1 % [-0.9, +2.3] | -0.1 % [-0.2, +0.1] |
| 1x4096 | 124.68 | 125.14 | -0.36 % | 124.77 | 126.09 | -1.04 % | +0.8 % [+0.1, +2.6] | +0.8 % [+0.6, +1.0] |
| 1x8192 | 303.37 | 304.35 | -0.32 % | 303.83 | 306.33 | -0.82 % | +0.7 % [+0.2, +2.8] | +0.8 % [+0.5, +0.8] |
| 4x2048 | 255.34 | 256.27 | -0.36 % | 256.25 | 258.38 | -0.82 % | +1.0 % [+0.4, +2.8] | +0.9 % [+0.7, +1.1] |

Cells against A: median -0.36 %, range [-1.14, -0.04] % (16 cells); the fq-16x64 vs fo6 ratio's median over shapes: W +0.88 %, A +0.82 %.

## Combined 1x2048 table: the six paper models (A's records for Mistral, Nemotron, Phi-4 and Qwen3.8-27B; part W's for the two Llama models)

Raw medians over rounds (graph ms):

| policy | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 23.3582 | 137.5994 | 129.5987 | 154.3428 | 250.9950 | 465.7502 |
| NVFP4 | 12.1270 | 63.0691 | 56.8917 | 81.4361 | 112.4662 | 206.6475 |
| FourOverSix | 12.2636 | 63.2162 | 57.0733 | 82.2633 | 113.0619 | 207.2973 |
| FlipQuant 16x64 | 12.2931 | 63.2011 | 57.0050 | 82.8244 | 114.0531 | 209.4143 |
| FlipQuant 8x64 | 12.6672 | 66.1010 | 59.8418 | 85.4560 | 118.9691 | 217.0089 |
| FlipQuant 256x64 (extra) | 12.3321 | 63.2497 | 56.9905 | 83.9817 | 114.1494 | 209.2176 |

LaTeX, each cell rounded once to one decimal from the raw value above:

```latex
BF16 & 23.4 & 137.6 & 129.6 & 154.3 & 251.0 & 465.8 \\
NVFP4 & 12.1 & 63.1 & 56.9 & 81.4 & 112.5 & 206.6 \\
FourOverSix & 12.3 & 63.2 & 57.1 & 82.3 & 113.1 & 207.3 \\
FlipQuant 16x64 & 12.3 & 63.2 & 57.0 & 82.8 & 114.1 & 209.4 \\
FlipQuant 8x64 & 12.7 & 66.1 & 59.8 & 85.5 & 119.0 & 217.0 \\
FlipQuant 256x64 & 12.3 & 63.2 & 57.0 & 84.0 & 114.1 & 209.2 \\
```

## Six-model summaries

**FlipQuant 16x64 vs FourOverSix** (paired within rounds; the per-shape value is the median over rounds):
- Over all 8 shapes and six models: [-0.24, +2.39] %.
- Per-model median over the 8 shapes: Llama-3.2-1B-Instruct +0.38 %; Llama-3.1-8B-Instruct +0.79 %; Mistral-7B-Instruct-v0.3 +0.82 %; Nemotron-Nano-9B-v2 +0.55 %; Phi-4 +0.82 %; Qwen3.8-27B +0.82 %.
- At 1x2048, the paired ratio: [-0.08, +1.01] % (Llama-3.2-1B-Instruct +0.24; Llama-3.1-8B-Instruct -0.04; Mistral-7B-Instruct-v0.3 -0.08; Nemotron-Nano-9B-v2 +0.73; Phi-4 +1.01; Qwen3.8-27B +0.84).
- At 1x2048, the ratio of the medians: [-0.12, +1.02] % (Llama-3.2-1B-Instruct +0.24; Llama-3.1-8B-Instruct -0.02; Mistral-7B-Instruct-v0.3 -0.12; Nemotron-Nano-9B-v2 +0.68; Phi-4 +0.88; Qwen3.8-27B +1.02).

**FlipQuant 8x64 vs FourOverSix** (paired within rounds; the per-shape value is the median over rounds):
- Over all 8 shapes and six models: [+0.16, +9.63] %.
- Per-model median over the 8 shapes: Llama-3.2-1B-Instruct +2.86 %; Llama-3.1-8B-Instruct +4.07 %; Mistral-7B-Instruct-v0.3 +4.27 %; Nemotron-Nano-9B-v2 +3.33 %; Phi-4 +4.53 %; Qwen3.8-27B +3.99 %.
- At 1x2048, the paired ratio: [+3.69, +5.17] % (Llama-3.2-1B-Instruct +3.69; Llama-3.1-8B-Instruct +4.67; Mistral-7B-Instruct-v0.3 +4.85; Nemotron-Nano-9B-v2 +3.89; Phi-4 +5.17; Qwen3.8-27B +4.73).
- At 1x2048, the ratio of the medians: [+3.29, +5.22] % (Llama-3.2-1B-Instruct +3.29; Llama-3.1-8B-Instruct +4.56; Mistral-7B-Instruct-v0.3 +4.85; Nemotron-Nano-9B-v2 +3.88; Phi-4 +5.22; Qwen3.8-27B +4.68).

**BF16 / FlipQuant 16x64 at 1x2048** (ratio of the medians): [1.86, 2.27] x (Llama-3.2-1B-Instruct 1.90x; Llama-3.1-8B-Instruct 2.18x; Mistral-7B-Instruct-v0.3 2.27x; Nemotron-Nano-9B-v2 1.86x; Phi-4 2.20x; Qwen3.8-27B 2.22x).
