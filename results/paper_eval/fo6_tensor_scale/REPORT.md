# Part R: FourOverSix's tensor-scale constant, 6 x 448 vs the original 6 x 256

PROTOCOL.md part R. FO6-448 = our FourOverSix (global scale amax / (6 x 448)), FO6-256 = the original FourOverSix (arXiv:2512.02010, Sec. 3: amax / (6 x 256)), both for weights and per-token activations, native sm120 (build_V, `--kernel-set auto`, `--paper-convention`), flipquant paper-sm120-runs d92c032 (`--fo6-tensor-qmax 256`). ΔNLL x 1e-3 ± 2 SE (paired over the same windows), * beyond 2 SE.

## Summary

- **Weights:** only 0.0003-0.0111 % of weight blocks exceed 2/3 of their tensor's amax, the blocks for which 448 cannot select a true 4. Yet the constant changes 92-95 % of blocks, and 18.3-19.2 % flip their 4-vs-6 choice. The cause is that every block scale is E4M3-rounded relative to a tensor scale 7/4 times larger, so the E4M3 grid shifts by a factor of 4/7. The share of blocks choosing 4 is nearly the same (43.8-45.2 % with 448, 43.8-45.2 % with 256).
- **Activations (per token):** the block holding each token's max can now take a true 4: at least 16/K of the blocks (0.09-0.78 % across the models' K).
- **Accuracy:** model-dependent and small. FO6-256 is significantly better than FO6-448 in 5 of 16 cells, worse in 2, and not significantly different in the rest. The mean capped NLL recovery (16 pairs) moves from 9.5 % to 11.7 %.
- **FlipQuant:** its maps were calibrated with 448 and not recalibrated. It is still significantly better than the original FourOverSix in every cell: 16 / 16 at 8x64, 16 / 16 at 16x64, 16 / 16 at 256x64.

## Checks

- The new `fourover6` at its default equals flipquant aa12408:flipquant/formats.py bit for bit: 128 cases (every scope, CPU and GPU, random and real weights): True.
- tests/test_fo6_tensor_qmax.py (18 passed on sm_120 with build_V): 256 equals an independent implementation of the original definition; the native quantizer at fp32(1/1536) gives fourover6(x, 'row', 256)'s scale bytes and global scales bit for bit; the unfused route at fp32(1/2688) equals the fused NativeLinear bit for bit; an install at 256 matches fake mode at 256, not 448.
- The FO6-448 reruns with the new code reproduce the table's FO6 records per window, bit for bit, for the models whose table record came from evaluation.ppl: Nemotron-Nano-9B-v2 True, Qwen3.8-27B True, Llama-3.2-1B-Instruct True, Llama-3.1-8B-Instruct True.
- The other four models' table FO6 records come from the earlier sm120 harness (main-ppl / paper). Per window, the same-code FO6-448 rerun vs that record: Qwen3-1.7B: bit for bit; Qwen3-8B: bit for bit; Mistral-7B-Instruct-v0.3: bit for bit; Phi-4: bit for bit. The paired ΔNLL below uses the same-code FO6-448 (identical to the table's row where bit for bit).

## PPL and paired ΔNLL

| model | FO6-448 (wiki / C4) | FO6-256 (wiki / C4) | FO6-256 − FO6-448 | FO6-256 − NVFP4 | FlipQuant 16x64 − FO6-256 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 19.4745 / 20.9072 | 18.8182 / 21.1941 | -34.28 ± 10.41* / +13.63 ± 2.66* | -2.82 ± 9.55 / +7.54 ± 2.92* | -200.59 ± 9.93* / -87.95 ± 3.91* |
| Qwen3-8B | 10.0071 / 13.7575 | 9.9709 / 13.6991 | -3.62 ± 2.70* / -4.25 ± 1.71* | -7.75 ± 2.68* / -5.20 ± 1.77* | -38.93 ± 3.22* / -10.54 ± 2.02* |
| Mistral-7B-Instruct-v0.3 | 5.7083 / 8.3987 | 5.7062 / 8.3890 | -0.36 ± 1.30 / -1.16 ± 1.17 | -4.23 ± 1.32* / -3.16 ± 1.12* | -9.33 ± 1.45* / -7.82 ± 1.18* |
| Nemotron-Nano-9B-v2 | 8.4260 / 11.5034 | 8.4375 / 11.5366 | +1.37 ± 1.47 / +2.88 ± 0.98* | -1.89 ± 1.43* / -0.62 ± 1.13 | -10.05 ± 1.39* / -7.96 ± 0.97* |
| Phi-4 | 6.6617 / 10.5441 | 6.6622 / 10.5480 | +0.06 ± 1.81 / +0.37 ± 1.01 | -4.68 ± 1.74* / -2.99 ± 1.04* | -4.46 ± 1.76* / -3.07 ± 1.00* |
| Qwen3.8-27B | 7.3023 / 10.1903 | 7.3157 / 10.1834 | +1.83 ± 4.37 / -0.68 ± 1.09 | -34.71 ± 9.06* / -3.90 ± 1.23* | -17.89 ± 3.90* / -4.16 ± 1.12* |
| Llama-3.2-1B-Instruct | 15.3152 / 21.4488 | 15.3088 / 21.3724 | -0.42 ± 2.66 / -3.57 ± 2.14* | -14.68 ± 2.79* / -18.29 ± 2.53* | -57.48 ± 3.29* / -82.29 ± 4.06* |
| Llama-3.1-8B-Instruct | 7.8186 / 11.2667 | 7.7686 / 11.2803 | -6.42 ± 1.87* / +1.21 ± 2.22 | -15.20 ± 2.20* / -5.36 ± 2.68* | -7.49 ± 2.17* / -12.34 ± 3.54* |

FO6-256 is significantly better than FO6-448 in 5 of 16 cells and significantly worse in 2. FO6-256 vs NVFP4: better in 13, not significant in 2, worse in 1 cells. FlipQuant (maps calibrated with 448, not recalibrated) is significantly better than FO6-256 in 16 / 16 cells at 8x64, 16 / 16 cells at 16x64, 16 / 16 cells at 256x64; every cell at 16x64: True.

The same-code FO6-448 reruns equal the table's FourOverSix records per window for all eight models, so FO6-256 − FO6-448 is also FO6-256 − the table's FourOverSix row.

## Mean capped NLL recovery (16 pairs; the table's BF16 and NVFP4)

| row | mean capped NLL recovery |
|---|---:|
| FO6-448, the table's records | 9.5 % |
| FO6-448, the same-code reruns | 9.5 % |
| FO6-256 | 11.7 % |

Per pair (FO6-448 same-code / FO6-256): qwen3-1.7b/wiki -25.9 / 2.3; qwen3-1.7b/c4 6.8 / -8.5; qwen3-8b/wiki 12.6 / 23.7; qwen3-8b/c4 2.7 / 15.0; mistral-7b/wiki 9.3 / 10.1; mistral-7b/c4 5.9 / 9.3; nemotron-nano-9b-v2/wiki 7.3 / 4.2; nemotron-nano-9b-v2/c4 10.2 / 1.8; phi4-14b/wiki 13.4 / 13.3; phi4-14b/c4 13.0 / 11.6; qwen3.8-27b/wiki 51.2 / 48.6; qwen3.8-27b/c4 9.8 / 11.9; llama3.2-1b-instruct/wiki 8.6 / 8.9; llama3.2-1b-instruct/c4 9.1 / 11.3; llama3.1-8b-instruct/wiki 9.9 / 17.1; llama3.1-8b-instruct/c4 7.5 / 6.1. Flagged pairs (non-positive NVFP4 − BF16 gap): none.

## Block statistics of the quantized weights (CPU, formats.fourover6, tensor scope)

Share of 16-element blocks. *> 2/3 amax*: blocks whose max exceeds 2/3 of the tensor amax, i.e. the blocks whose 4-candidate 448 clamps (counted directly: *clamped 448*). *4 chosen*: blocks that take the 4-candidate. *choice flips*: 4-vs-6 decision differs between 448 and 256. *blocks changed*: any dequantized value differs. *subnormal scale*: the chosen E4M3 block scale is below 2^-6 (E4M3's subnormal range).

| model | > 2/3 amax | clamped 448 | 4 chosen 448 | 4 chosen 256 | choice flips | blocks changed | subnormal scale 448 / 256 | activations: ≥ 1 block per token (16/K) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | 0.0111 % | 0.0111 % | 43.88 % | 43.86 % | 18.41 % | 95.18 % | 0.000 / 0.000 % | K=2048: 0.78 %, K=6144: 0.26 % |
| Qwen3-8B | 0.0034 % | 0.0034 % | 43.80 % | 43.77 % | 18.33 % | 95.40 % | 0.000 / 0.000 % | K=4096: 0.39 %, K=12288: 0.13 % |
| Mistral-7B-Instruct-v0.3 | 0.0018 % | 0.0018 % | 44.62 % | 44.61 % | 18.50 % | 94.80 % | 0.000 / 0.000 % | K=4096: 0.39 %, K=14336: 0.11 % |
| Nemotron-Nano-9B-v2 | 0.0004 % | 0.0004 % | 43.80 % | 44.31 % | 19.17 % | 91.93 % | 0.024 / 0.025 % | K=4480: 0.36 %, K=5120: 0.31 %, K=10240: 0.16 %, K=15680: 0.10 % |
| Phi-4 | 0.0003 % | 0.0003 % | 45.21 % | 45.19 % | 18.63 % | 94.58 % | 0.000 / 0.000 % | K=5120: 0.31 %, K=17920: 0.09 % |
| Qwen3.8-27B | 0.0006 % | 0.0006 % | 44.65 % | 44.62 % | 18.51 % | 94.71 % | 0.000 / 0.000 % | K=5120: 0.31 %, K=6144: 0.26 %, K=17408: 0.09 % |
| Llama-3.2-1B-Instruct | 0.0066 % | 0.0066 % | 44.32 % | 44.30 % | 18.44 % | 93.91 % | 0.000 / 0.000 % | K=2048: 0.78 %, K=8192: 0.20 % |
| Llama-3.1-8B-Instruct | 0.0017 % | 0.0017 % | 44.69 % | 44.68 % | 18.51 % | 94.85 % | 0.000 / 0.000 % | K=4096: 0.39 %, K=14336: 0.11 % |

Records: `records/` (the check, the 16 PPL records, the per-model statistics; per-module counts in fo6_tensor_scale.json are omitted, they are in records/<model>/stats.json). No FlipQuant map was recalibrated.
