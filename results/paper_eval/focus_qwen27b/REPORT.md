# Part Q: FOCUS on Qwen3.8-27B (the state trained on the cluster, evaluated as part K)

PROTOCOL.md part Q. ΔNLL x 1e-3 ± 2 SE (paired over the same windows), * beyond 2 SE; negative = FOCUS better.

## Provenance of the state

- Repository `edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4` @ `b84ff372a7ccab5b8541395db26056c3acd14aff`, file `focus_256/focus.pt`: 18,263,340,091 bytes, sha256 `baf02d9c36551191fe7bc216f6408346cc7d34471cc3e9b524eb60e3f81eeb9b` (equal to the Hub's LFS sha256; downloaded into the HF cache on /home in 1238 s).
- Windows: the state's 256 window token sha256 are identical, in order, to the 256 release fit windows for qwen3.8-27b (the record's 128 fit windows, math then code, and the release 16x64 trainer report's 128 extension windows); its fit_extension.json records equal that report's, settings equal.
- Hyperparameters equal to part K's (`run_sm120.FOCUS_COMMON`): epochs 1, batch 32, lr_scale 0.005, lr_sub 0.001, topk 1000, num_sub 2, init_q 6.0, act fourover6, act_scope row, seed 42, seqlen 512, data release, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, max_blocks 0, zeroshot False.
- Recorded deviation: micro-batch 1 on 2 x H200 with device_map balanced (part K: micro 8, one GPU); the global batch (32, 8 steps) is the same, only the summation order differs; --windows 256 = all 256 release windows (part K: 0 = all) (job 519366, NVIDIA H200, NVIDIA H200).
- Training: status complete, 8 steps, 2249 s; KL-top 0.0521 -> 0.0480.

## Evaluation

Part K's command: `evaluation.ppl --weight focus --focus <state> --focus-deploy --act fourover6 --act-scope row --paper-convention`, native (build_V, `--kernel-set auto`), n16k64-fast (transformers 5.16.1, torch 2.9.0+cu128), revision `1d4bf0f2`. One GPU (NVIDIA RTX PRO 6000 Blackwell Workstation Edition): 116 s, torch peak 83.5 GiB; kernel `stock_ko`, 496 modules (496 quantized).

| row | WikiText-2 | C4 |
|---|---:|---:|
| BF16 | 7.0523 | 9.8931 |
| NVFP4 | 7.5740 | 10.2231 |
| FourOverSix | 7.3023 | 10.1903 |
| IF4 | 7.3343 | 10.1657 |
| MixFP4 (Zou) | 7.3476 | 10.1639 |
| GPTQ‡ | 7.5278 | 10.1576 |
| FOCUS | 7.2746 | 10.1572 |
| FlipQuant 8x64 | 7.1399 | 10.1441 |
| FlipQuant 16x64 | 7.1859 | 10.1410 |
| FlipQuant 256x64 | 7.2195 | 10.1613 |

Plausibility (BF16 < FOCUS < NVFP4): wiki True, c4 True.

## FOCUS against the other rows (paired ΔNLL x 1e-3 ± 2 SE)

| FOCUS − | WikiText-2 | C4 |
|---|---:|---:|
| FourOverSix | -3.80 ± 4.78 | -3.25 ± 1.21* |
| NVFP4 | -40.33 ± 7.60* | -6.47 ± 1.30* |
| BF16 | +31.04 ± 4.45* | +26.35 ± 2.24* |
| FlipQuant 8x64 | +18.69 ± 4.62* | +1.29 ± 1.08* |
| FlipQuant 16x64 | +12.27 ± 4.36* | +1.60 ± 1.09* |
| FlipQuant 256x64 | +7.61 ± 4.16* | -0.40 ± 1.12 |

Largest single-window NLL difference vs FourOverSix (nats; window) and the mean ΔNLL x 1e-3 without it: wiki -0.164 (w137; without it -2.68); c4 -0.062 (w108; without it -3.02).

## Mean capped NLL recovery

Over the 14 common pairs (every model with all methods: qwen3-1.7b/wiki, qwen3-1.7b/c4, qwen3-8b/wiki, qwen3-8b/c4, mistral-7b/wiki, mistral-7b/c4, nemotron-nano-9b-v2/wiki, nemotron-nano-9b-v2/c4, phi4-14b/wiki, phi4-14b/c4, qwen3.8-27b/wiki, qwen3.8-27b/c4, llama3.1-8b-instruct/wiki, llama3.1-8b-instruct/c4); per pair 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - NLL_BF16)), no floor, equal weight. Pairs with a non-positive gap (NLL_NVFP4 - NLL_BF16 <= 0): none.

| row | 14 pairs | the previous pairs (without qwen3.8-27b) |
|---|---:|---:|
| BF16 | 100.0 % | 100.0 % |
| NVFP4 | 0.0 % | 0.0 % |
| FourOverSix | 9.5 % | 6.1 % |
| IF4 | 20.9 % | 19.2 % |
| MixFP4 (Zou) | 19.3 % | 17.5 % |
| GPTQ‡ | 4.1 % | 2.5 % |
| FOCUS | 41.6 % | 42.2 % |
| FlipQuant 8x64 | 47.2 % | 46.2 % |
| FlipQuant 16x64 | 46.1 % | 45.6 % |
| FlipQuant 256x64 | 40.8 % | 40.5 % |

Per pair:

| row | qwen3-1.7b/wiki | qwen3-1.7b/c4 | qwen3-8b/wiki | qwen3-8b/c4 | mistral-7b/wiki | mistral-7b/c4 | nemotron-nano-9b-v2/wiki | nemotron-nano-9b-v2/c4 | phi4-14b/wiki | phi4-14b/c4 | qwen3.8-27b/wiki | qwen3.8-27b/c4 | llama3.1-8b-instruct/wiki | llama3.1-8b-instruct/c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NVFP4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FourOverSix | -25.9 | 6.8 | 12.6 | 2.7 | 9.3 | 5.9 | 7.3 | 10.2 | 13.4 | 13.0 | 51.2 | 9.8 | 9.9 | 7.5 |
| IF4 | -31.3 | 4.4 | 50.1 | 35.3 | 25.2 | 18.8 | 16.7 | 23.6 | 25.0 | 25.2 | 45.1 | 17.2 | 21.4 | 16.3 |
| MixFP4 (Zou) | -27.1 | -27.5 | 57.4 | 37.6 | 28.4 | 16.4 | 11.9 | 21.7 | 23.0 | 26.8 | 42.5 | 17.7 | 22.5 | 18.3 |
| GPTQ‡ | 8.8 | -1.9 | -24.3 | 14.0 | -2.0 | 0.2 | 6.4 | 10.9 | 2.2 | 9.6 | 8.6 | 19.6 | 8.4 | -2.6 |
| FOCUS | 100.0 | 83.3 | 100.0 | 41.9 | 21.8 | 26.3 | 22.8 | 17.7 | 27.9 | 20.6 | 56.5 | 19.7 | 25.4 | 18.1 |
| FlipQuant 8x64 | 100.0 | 88.2 | 100.0 | 51.1 | 33.8 | 34.1 | 26.8 | 25.1 | 25.1 | 22.3 | 82.7 | 23.6 | 27.8 | 20.5 |
| FlipQuant 16x64 | 100.0 | 90.5 | 100.0 | 45.3 | 32.5 | 32.4 | 26.9 | 24.9 | 25.9 | 23.4 | 73.7 | 24.6 | 25.5 | 20.1 |
| FlipQuant 256x64 | 100.0 | 70.3 | 100.0 | 43.2 | 28.7 | 26.7 | 19.8 | 19.2 | 23.5 | 22.5 | 67.2 | 18.5 | 19.8 | 12.5 |

The updated table: `results/paper_eval/final/table_main_ppl_final.tex` (FINAL.md, final.json). Records: `records/` (the provenance check, the download, the PPL record, and the state's report.json / fit_extension.json from the Hub).
