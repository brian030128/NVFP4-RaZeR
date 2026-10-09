# Final table main-ppl (per-window NLL; parts F, I, J, K)

PPL (WikiText-2 / C4). Non-hybrid models in n16k64 (main-ppl 44f8cea + the release FlipQuant records); Nemotron-Nano-9B-v2 and Qwen3.8-27B in n16k64-fast (part F). † = FlipQuant not significantly better than FourOverSix (paired ΔNLL > -2 SE).

| row | Qwen3-1.7B | Qwen3-8B | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | loss recovered (12 pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.7162 / 19.2463 | 9.7251 / 13.3004 | 5.4961 / 8.1351 | 8.0863 / 11.1528 | 6.4615 / 10.3098 | 7.0523 / 9.8931 | 100.0 % |
| NVFP4 | 18.8713 / 21.0349 | 10.0484 / 13.7705 | 5.7304 / 8.4156 | 8.4535 / 11.5437 | 6.6934 / 10.5795 | 7.5740 / 10.2231 | 0.0 % |
| FourOverSix | 19.4745 / 20.9072 | 10.0071 / 13.7575 | 5.7083 / 8.3987 | 8.4260 / 11.5034 | 6.6617 / 10.5441 | 7.3023 / 10.1903 | 6.7 % |
| IF4 | 19.6007 / 20.9536 | 9.8852 / 13.6027 | 5.6703 / 8.3621 | 8.3911 / 11.4502 | 6.6346 / 10.5108 | 7.3343 / 10.1657 | 13.4 % |
| MixFP4 (Zou) | 19.5027 / 21.5547 | 9.8615 / 13.5916 | 5.6629 / 8.3689 | 8.4088 / 11.4578 | 6.6392 / 10.5066 | 7.3476 / 10.1639 | 9.4 % |
| GPTQ‡ | 18.6717 / 21.0696 | 10.1287 / 13.7039 | 5.7351 / 8.4150 | 8.4294 / 11.5005 | 6.6883 / 10.5534 | 7.5278 / 10.1576 | 4.6 % |
| FOCUS | 15.0890 / 19.5345 | 9.5687 / 13.5715 | 5.6784 / 8.3408 | 8.3685 / 11.4734 | 6.6279 / 10.5234 | TBD | 83.3 % (10 pairs) |
| FlipQuant 8x64 | 15.4039 / 19.4491 | 9.6083 / 13.5284 | 5.6501 / 8.3188 | 8.3536 / 11.4442 | 6.6345 / 10.5187 | 7.1399 / 10.1441 | 78.9 % |
| FlipQuant 16x64 | 15.3979 / 19.4097 | 9.5902 / 13.5554 | 5.6532 / 8.3236 | 8.3532 / 11.4452 | 6.6325 / 10.5157 | 7.1859 / 10.1410 | 78.2 % |
| FlipQuant 256x64 | 15.9680 / 19.7617 | 9.6779 / 13.5653 | 5.6622 / 8.3397 | 8.3797 / 11.4677 | 6.6382 / 10.5181 | 7.2195 / 10.1613 | 64.7 % |

Tile granularity: |8x64 − 16x64| paired ΔNLL: max 0.0064 (('qwen3.8-27b', 'wiki')), mean 0.0012, signed mean -0.0004; significant in 3 of 12 pairs. 256x64 keeps 81.1 % of the 16x64 gain over FourOverSix overall; per model qwen3-1.7b 82 %, qwen3-8b 83 %, mistral-7b 81 %, nemotron-nano-9b-v2 63 %, phi4-14b 85 %, qwen3.8-27b 68 %.

FlipQuant vs NVFP4: significantly better in 36 of 36 cells; daggers (vs FourOverSix): 0 of 36 cells.

Loss recovered over the 10 pairs FOCUS has (without Qwen3.8-27B): FourOverSix 0.1 %; IF4 8.6 %; MixFP4 (Zou) 4.0 %; GPTQ‡ 3.0 %; FOCUS 83.3 %; FlipQuant 8x64 82.1 %; FlipQuant 16x64 82.4 %; FlipQuant 256x64 67.4 %.

FOCUS − FlipQuant, paired ΔNLL x 1e-3 ± 2 SE (negative: FOCUS better; * beyond 2 SE), WikiText-2 / C4:

| model | vs 8x64 | vs 16x64 | vs 256x64 |
|---|---:|---:|---:|
| qwen3-1.7b | -20.65 ± 3.57* / +4.39 ± 2.04* | -20.26 ± 3.82* / +6.41 ± 2.12* | -56.62 ± 4.70* / -11.56 ± 2.36* |
| qwen3-8b | -4.12 ± 2.40* / +3.18 ± 1.54* | -2.24 ± 2.40 / +1.18 ± 1.53 | -11.35 ± 2.63* / +0.46 ± 1.75 |
| mistral-7b | +5.01 ± 1.49* / +2.64 ± 1.40* | +4.45 ± 1.46* / +2.06 ± 1.17* | +2.86 ± 1.50* / +0.12 ± 1.21 |
| nemotron-nano-9b-v2 | +1.78 ± 1.52* / +2.55 ± 1.00* | +1.83 ± 1.46* / +2.46 ± 1.00* | -1.34 ± 1.55 / +0.49 ± 1.03 |
| phi4-14b | -0.99 ± 1.64 / +0.45 ± 1.03 | -0.70 ± 1.44 / +0.74 ± 0.93 | -1.55 ± 1.69 / +0.50 ± 1.01 |

GPTQ‡ − NVFP4 and GPTQ‡ − FourOverSix, paired ΔNLL x 1e-3 ± 2 SE, WikiText-2 / C4:

| model | vs NVFP4 | vs FourOverSix |
|---|---:|---:|
| qwen3-1.7b | -10.64 ± 11.15 / +1.65 ± 2.87 | -42.10 ± 11.88* / +7.74 ± 2.97* |
| qwen3-8b | +7.96 ± 2.81* / -4.85 ± 1.50* | +12.08 ± 2.89* / -3.90 ± 1.79* |
| mistral-7b | +0.82 ± 1.36 / -0.06 ± 1.52 | +4.68 ± 1.39* / +1.94 ± 2.08 |
| nemotron-nano-9b-v2 | -2.86 ± 1.52* / -3.75 ± 1.13* | +0.40 ± 1.64 / -0.25 ± 1.12 |
| phi4-14b | -0.76 ± 1.72 / -2.47 ± 1.05* | +3.98 ± 2.01* / +0.88 ± 1.16 |
| qwen3.8-27b | -6.12 ± 7.73 / -6.43 ± 1.15* | +30.41 ± 6.97* / -3.22 ± 1.29* |

Missing (TBD): qwen3.8-27b focus.
