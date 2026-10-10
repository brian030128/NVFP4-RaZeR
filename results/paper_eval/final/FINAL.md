# Final table main-ppl (per-window NLL; parts F, I, J, K)

PPL (WikiText-2 / C4). Non-hybrid models in n16k64 (main-ppl 44f8cea + the release FlipQuant records); Nemotron-Nano-9B-v2 and Qwen3.8-27B in n16k64-fast (part F). † = FlipQuant not significantly better than FourOverSix (paired ΔNLL > -2 SE).

| row | Qwen3-1.7B | Qwen3-8B | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | mean capped NLL recovery (12 pairs) | loss recovered (14 pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.7162 / 19.2463 | 9.7251 / 13.3004 | 5.4961 / 8.1351 | 8.0863 / 11.1528 | 6.4615 / 10.3098 | 7.0523 / 9.8931 | TBD | 7.2171 / 10.3853 | 100.0 % | 100.0 % |
| NVFP4 | 18.8713 / 21.0349 | 10.0484 / 13.7705 | 5.7304 / 8.4156 | 8.4535 / 11.5437 | 6.6934 / 10.5795 | 7.5740 / 10.2231 | TBD | 7.8876 / 11.3409 | 0.0 % | 0.0 % |
| FourOverSix | 19.4745 / 20.9072 | 10.0071 / 13.7575 | 5.7083 / 8.3987 | 8.4260 / 11.5034 | 6.6617 / 10.5441 | 7.3023 / 10.1903 | TBD | 7.8186 / 11.2667 | 6.1 % | 7.2 % |
| IF4 | 19.6007 / 20.9536 | 9.8852 / 13.6027 | 5.6703 / 8.3621 | 8.3911 / 11.4502 | 6.6346 / 10.5108 | 7.3343 / 10.1657 | TBD | 7.7390 / 11.1793 | 19.2 % | 14.7 % |
| MixFP4 (Zou) | 19.5027 / 21.5547 | 9.8615 / 13.5916 | 5.6629 / 8.3689 | 8.4088 / 11.4578 | 6.6392 / 10.5066 | 7.3476 / 10.1639 | TBD | 7.7316 / 11.1592 | 17.5 % | 11.9 % |
| GPTQ‡ | 18.6717 / 21.0696 | 10.1287 / 13.7039 | 5.7351 / 8.4150 | 8.4294 / 11.5005 | 6.6883 / 10.5534 | 7.5278 / 10.1576 | TBD | 7.8292 / 11.3664 | 2.5 % | 4.2 % |
| FOCUS | 15.0890 / 19.5345 | 9.5687 / 13.5715 | 5.6784 / 8.3408 | 8.3685 / 11.4734 | 6.6279 / 10.5234 | TBD | TBD | 7.7113 / 11.1618 | 42.2 % | 67.0 % (12 pairs) |
| FlipQuant 8x64 | 15.4039 / 19.4491 | 9.6083 / 13.5284 | 5.6501 / 8.3188 | 8.3536 / 11.4442 | 6.6345 / 10.5187 | 7.1399 / 10.1441 | TBD | 7.6951 / 11.1384 | 46.2 % | 66.4 % |
| FlipQuant 16x64 | 15.3979 / 19.4097 | 9.5902 / 13.5554 | 5.6532 / 8.3236 | 8.3532 / 11.4452 | 6.6325 / 10.5157 | 7.1859 / 10.1410 | TBD | 7.7106 / 11.1420 | 45.6 % | 65.6 % |
| FlipQuant 256x64 | 15.9680 / 19.7617 | 9.6779 / 13.5653 | 5.6622 / 8.3397 | 8.3797 / 11.4677 | 6.6382 / 10.5181 | 7.2195 / 10.1613 | TBD | 7.7501 / 11.2171 | 40.5 % | 53.6 % |

Tile granularity: |8x64 − 16x64| paired ΔNLL: max 0.0064 (('qwen3.8-27b', 'wiki')), mean 0.0012, signed mean -0.0005; significant in 4 of 14 pairs. 256x64 keeps 79.6 % of the 16x64 gain over FourOverSix overall; per model qwen3-1.7b 82 %, qwen3-8b 83 %, mistral-7b 81 %, nemotron-nano-9b-v2 63 %, phi4-14b 85 %, qwen3.8-27b 68 %, llama3.1-8b-instruct 53 %.

FlipQuant vs NVFP4: significantly better in 42 of 42 cells; daggers (vs FourOverSix): 0 of 42 cells.

Mean capped NLL recovery: the 12 common pairs (the models with FOCUS): qwen3-1.7b/wiki, qwen3-1.7b/c4, qwen3-8b/wiki, qwen3-8b/c4, mistral-7b/wiki, mistral-7b/c4, nemotron-nano-9b-v2/wiki, nemotron-nano-9b-v2/c4, phi4-14b/wiki, phi4-14b/c4, llama3.1-8b-instruct/wiki, llama3.1-8b-instruct/c4. Pairs with NLL_NVFP4 - NLL_BF16 <= 0 (excluded, flagged): none.

Per pair, 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - NLL_BF16)):

| row | qwen3-1.7b/wiki | qwen3-1.7b/c4 | qwen3-8b/wiki | qwen3-8b/c4 | mistral-7b/wiki | mistral-7b/c4 | nemotron-nano-9b-v2/wiki | nemotron-nano-9b-v2/c4 | phi4-14b/wiki | phi4-14b/c4 | llama3.1-8b-instruct/wiki | llama3.1-8b-instruct/c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NVFP4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FourOverSix | -25.9 | 6.8 | 12.6 | 2.7 | 9.3 | 5.9 | 7.3 | 10.2 | 13.4 | 13.0 | 9.9 | 7.5 |
| IF4 | -31.3 | 4.4 | 50.1 | 35.3 | 25.2 | 18.8 | 16.7 | 23.6 | 25.0 | 25.2 | 21.4 | 16.3 |
| MixFP4 (Zou) | -27.1 | -27.5 | 57.4 | 37.6 | 28.4 | 16.4 | 11.9 | 21.7 | 23.0 | 26.8 | 22.5 | 18.3 |
| GPTQ‡ | 8.8 | -1.9 | -24.3 | 14.0 | -2.0 | 0.2 | 6.4 | 10.9 | 2.2 | 9.6 | 8.4 | -2.6 |
| FOCUS | 100.0 | 83.3 | 100.0 | 41.9 | 21.8 | 26.3 | 22.8 | 17.7 | 27.9 | 20.6 | 25.4 | 18.1 |
| FlipQuant 8x64 | 100.0 | 88.2 | 100.0 | 51.1 | 33.8 | 34.1 | 26.8 | 25.1 | 25.1 | 22.3 | 27.8 | 20.5 |
| FlipQuant 16x64 | 100.0 | 90.5 | 100.0 | 45.3 | 32.5 | 32.4 | 26.9 | 24.9 | 25.9 | 23.4 | 25.5 | 20.1 |
| FlipQuant 256x64 | 100.0 | 70.3 | 100.0 | 43.2 | 28.7 | 26.7 | 19.8 | 19.2 | 23.5 | 22.5 | 19.8 | 12.5 |

Loss recovered over the 12 pairs FOCUS has (without Qwen3.8-27B): FourOverSix 2.4 %; IF4 11.3 %; MixFP4 (Zou) 8.3 %; GPTQ‡ 3.0 %; FOCUS 67.0 %; FlipQuant 8x64 66.8 %; FlipQuant 16x64 66.7 %; FlipQuant 256x64 53.9 %.

FOCUS − FlipQuant, paired ΔNLL x 1e-3 ± 2 SE (negative: FOCUS better; * beyond 2 SE), WikiText-2 / C4:

| model | vs 8x64 | vs 16x64 | vs 256x64 |
|---|---:|---:|---:|
| qwen3-1.7b | -20.65 ± 3.57* / +4.39 ± 2.04* | -20.26 ± 3.82* / +6.41 ± 2.12* | -56.62 ± 4.70* / -11.56 ± 2.36* |
| qwen3-8b | -4.12 ± 2.40* / +3.18 ± 1.54* | -2.24 ± 2.40 / +1.18 ± 1.53 | -11.35 ± 2.63* / +0.46 ± 1.75 |
| mistral-7b | +5.01 ± 1.49* / +2.64 ± 1.40* | +4.45 ± 1.46* / +2.06 ± 1.17* | +2.86 ± 1.50* / +0.12 ± 1.21 |
| nemotron-nano-9b-v2 | +1.78 ± 1.52* / +2.55 ± 1.00* | +1.83 ± 1.46* / +2.46 ± 1.00* | -1.34 ± 1.55 / +0.49 ± 1.03 |
| phi4-14b | -0.99 ± 1.64 / +0.45 ± 1.03 | -0.70 ± 1.44 / +0.74 ± 0.93 | -1.55 ± 1.69 / +0.50 ± 1.01 |
| llama3.1-8b-instruct | +2.11 ± 1.97* / +2.11 ± 1.98* | +0.09 ± 1.77 / +1.78 ± 1.89 | -5.02 ± 1.91* / -4.94 ± 2.42* |

GPTQ‡ − NVFP4 and GPTQ‡ − FourOverSix, paired ΔNLL x 1e-3 ± 2 SE, WikiText-2 / C4:

| model | vs NVFP4 | vs FourOverSix |
|---|---:|---:|
| qwen3-1.7b | -10.64 ± 11.15 / +1.65 ± 2.87 | -42.10 ± 11.88* / +7.74 ± 2.97* |
| qwen3-8b | +7.96 ± 2.81* / -4.85 ± 1.50* | +12.08 ± 2.89* / -3.90 ± 1.79* |
| mistral-7b | +0.82 ± 1.36 / -0.06 ± 1.52 | +4.68 ± 1.39* / +1.94 ± 2.08 |
| nemotron-nano-9b-v2 | -2.86 ± 1.52* / -3.75 ± 1.13* | +0.40 ± 1.64 / -0.25 ± 1.12 |
| phi4-14b | -0.76 ± 1.72 / -2.47 ± 1.05* | +3.98 ± 2.01* / +0.88 ± 1.16 |
| qwen3.8-27b | -6.12 ± 7.73 / -6.43 ± 1.15* | +30.41 ± 6.97* / -3.22 ± 1.29* |
| llama3.1-8b-instruct | -7.42 ± 5.16* / +2.25 ± 18.94 | +1.36 ± 5.15 / +8.81 ± 19.19 |

Missing (TBD): qwen3.8-27b focus, llama3.2-1b-instruct (no records yet: a TBD column).
