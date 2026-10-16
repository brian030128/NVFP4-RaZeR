# GSM8K: six paper models, ten rows (part Y)

lm-eval 0.4.11 `gsm8k_llama` as shipped (8-shot CoT, chat template with few-shot as multi-turn, greedy, max_gen_toks 1024), all 1,319 problems, thinking off (Qwen3.8-27B `enable_thinking=False`; Nemotron `/no_think`; the others have no thinking mode). Part N's protocol, command and env, on flipquant paper-sm120-runs 4bbe0b1 (step 0: STEP0.md). Native rows on the RTX PRO 6000 (build_V, `--kernel-set auto`); IF4 and MixFP4 (Zou) simulated (†). Batch 64; Qwen3.8-27B BF16 at 48 (O0's memory rule). Every cell is a new run; the Nemotron FourOverSix / FlipQuant 16x64 cells are step 0 (c')'s runs.

GSM8K generations depend on the batch composition (part M's pad check), so all rows of a model run the full set at one batch size and see the same length-sorted batches -- except Qwen3.8-27B BF16 (batch 48): its paired comparisons with the batch-64 rows (vs BF16, fidelity) compare different batch compositions and are marked ◊; so is any other pairing of two runs at different batch sizes (amendment 4's fallback).

Accuracy in % ± 2 SE (binomial). Paired columns: strict-match difference in points (row-only correct / reference-only correct, exact McNemar p), * = p < 0.05.

## Llama-3.2-1B-Instruct

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 45.41 ± 2.74 | 45.56 ± 2.74 | +15.69* (274/67, p=7.6e-31) | — |
| NVFP4 | 29.19 ± 2.50 | 29.72 ± 2.52 | -0.53 (149/156, p=0.73) | -16.22* (72/286, p=2.6e-31) |
| FourOverSix | 29.72 ± 2.52 | 31.54 ± 2.56 | — | -15.69* (67/274, p=7.6e-31) |
| FOCUS | 32.15 ± 2.57 | 32.60 ± 2.58 | +2.43 (165/133, p=0.072) | -13.27* (73/248, p=1.9e-23) |
| GPTQ‡ | 30.78 ± 2.54 | 31.08 ± 2.55 | +1.06 (155/141, p=0.45) | -14.63* (84/277, p=3.4e-25) |
| IF4†§ | 33.51 ± 2.60 | 34.72 ± 2.62 | +3.79* (166/116, p=0.0035) | -11.90* (91/248, p=6e-18) |
| MixFP4 (Zou)†§ | 33.59 ± 2.60 | 33.97 ± 2.61 | +3.87* (177/126, p=0.004) | -11.83* (76/232, p=1.6e-19) |
| FlipQuant 8x64 | 30.63 ± 2.54 | 31.84 ± 2.57 | +0.91 (145/133, p=0.51) | -14.78* (74/269, p=3.8e-27) |
| FlipQuant 16x64 | 31.61 ± 2.56 | 32.60 ± 2.58 | +1.90 (142/117, p=0.14) | -13.80* (66/248, p=5.9e-26) |
| FlipQuant 256x64 | 31.08 ± 2.55 | 32.15 ± 2.57 | +1.36 (143/125, p=0.3) | -14.33* (76/265, p=1.2e-25) |

## Llama-3.1-8B-Instruct

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 85.90 ± 1.92 | 86.20 ± 1.90 | +6.44* (148/63, p=4.6e-09) | — |
| NVFP4 | 79.98 ± 2.20 | 80.97 ± 2.16 | +0.53 (111/104, p=0.68) | -5.91* (61/139, p=3.5e-08) |
| FourOverSix | 79.45 ± 2.22 | 80.06 ± 2.20 | — | -6.44* (63/148, p=4.6e-09) |
| FOCUS | 77.86 ± 2.29 | 79.45 ± 2.22 | -1.59 (94/115, p=0.17) | -8.04* (57/163, p=5.1e-13) |
| GPTQ‡ | 81.80 ± 2.12 | 82.64 ± 2.09 | +2.35* (120/89, p=0.038) | -4.09* (61/115, p=5.7e-05) |
| IF4† | 80.44 ± 2.18 | 80.89 ± 2.16 | +0.99 (113/100, p=0.41) | -5.46* (60/132, p=2.2e-07) |
| MixFP4 (Zou)† | 80.67 ± 2.17 | 81.27 ± 2.15 | +1.21 (116/100, p=0.31) | -5.23* (59/128, p=5e-07) |
| FlipQuant 8x64 | 81.43 ± 2.14 | 81.96 ± 2.12 | +1.97 (105/79, p=0.065) | -4.47* (67/126, p=2.6e-05) |
| FlipQuant 16x64 | 80.06 ± 2.20 | 80.44 ± 2.18 | +0.61 (99/91, p=0.61) | -5.84* (58/135, p=3e-08) |
| FlipQuant 256x64 | 81.73 ± 2.13 | 82.64 ± 2.09 | +2.27* (109/79, p=0.034) | -4.17* (60/115, p=3.9e-05) |

## Mistral-7B-Instruct-v0.3

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 47.01 ± 2.75 | 53.60 ± 2.75 | +2.50 (168/135, p=0.066) | — |
| NVFP4 | 44.50 ± 2.74 | 49.36 ± 2.75 | +0.00 (160/160, p=1) | -2.50 (130/163, p=0.061) |
| FourOverSix | 44.50 ± 2.74 | 48.82 ± 2.75 | — | -2.50 (135/168, p=0.066) |
| FOCUS | 43.37 ± 2.73 | 48.07 ± 2.75 | -1.14 (141/156, p=0.42) | -3.64* (113/161, p=0.0044) |
| GPTQ‡ | 45.41 ± 2.74 | 50.34 ± 2.75 | +0.91 (152/140, p=0.52) | -1.59 (133/154, p=0.24) |
| IF4† | 45.19 ± 2.74 | 50.19 ± 2.75 | +0.68 (157/148, p=0.65) | -1.82 (127/151, p=0.17) |
| MixFP4 (Zou)† | 45.56 ± 2.74 | 49.73 ± 2.75 | +1.06 (158/144, p=0.45) | -1.44 (136/155, p=0.29) |
| FlipQuant 8x64 | 45.11 ± 2.74 | 49.28 ± 2.75 | +0.61 (142/134, p=0.67) | -1.90 (131/156, p=0.16) |
| FlipQuant 16x64 | 44.81 ± 2.74 | 48.98 ± 2.75 | +0.30 (143/139, p=0.86) | -2.20 (136/165, p=0.11) |
| FlipQuant 256x64 | 44.73 ± 2.74 | 48.82 ± 2.75 | +0.23 (143/140, p=0.91) | -2.27 (134/164, p=0.093) |

## Nemotron-Nano-9B-v2

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 87.41 ± 1.83 | 90.52 ± 1.61 | -1.52 (44/64, p=0.067) | — |
| NVFP4 | 87.34 ± 1.83 | 88.86 ± 1.73 | -1.59 (53/74, p=0.076) | -0.08 (58/59, p=1) |
| FourOverSix | 88.93 ± 1.73 | 90.67 ± 1.60 | — | +1.52 (64/44, p=0.067) |
| FOCUS | 89.16 ± 1.71 | 89.92 ± 1.66 | +0.23 (59/56, p=0.85) | +1.74* (68/45, p=0.038) |
| GPTQ‡ | 87.19 ± 1.84 | 90.37 ± 1.62 | -1.74 (56/79, p=0.058) | -0.23 (66/69, p=0.86) |
| IF4† | 87.11 ± 1.85 | 88.32 ± 1.77 | -1.82* (45/69, p=0.031) | -0.30 (55/59, p=0.78) |
| MixFP4 (Zou)† | 85.52 ± 1.94 | 87.04 ± 1.85 | -3.41* (41/86, p=8.1e-05) | -1.90* (50/75, p=0.031) |
| FlipQuant 8x64 | 87.11 ± 1.85 | 90.52 ± 1.61 | -1.82* (42/66, p=0.026) | -0.30 (54/58, p=0.78) |
| FlipQuant 16x64 | 89.31 ± 1.70 | 91.28 ± 1.55 | +0.38 (51/46, p=0.68) | +1.90* (64/39, p=0.018) |
| FlipQuant 256x64 | 88.93 ± 1.73 | 90.22 ± 1.64 | +0.00 (49/49, p=1) | +1.52 (64/44, p=0.067) |

## Phi-4

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 93.33 ± 1.37 | 93.25 ± 1.38 | +1.59* (31/10, p=0.0015) | — |
| NVFP4 | 93.03 ± 1.40 | 92.95 ± 1.41 | +1.29* (27/10, p=0.0076) | -0.30 (18/22, p=0.64) |
| FourOverSix | 91.74 ± 1.52 | 91.66 ± 1.52 | — | -1.59* (10/31, p=0.0015) |
| FOCUS | 92.72 ± 1.43 | 92.65 ± 1.44 | +0.99 (32/19, p=0.092) | -0.61 (17/25, p=0.28) |
| GPTQ‡ | 92.19 ± 1.48 | 92.42 ± 1.46 | +0.45 (26/20, p=0.46) | -1.14* (13/28, p=0.028) |
| IF4† | 92.65 ± 1.44 | 92.80 ± 1.42 | +0.91 (27/15, p=0.088) | -0.68 (14/23, p=0.19) |
| MixFP4 (Zou)† | 93.40 ± 1.37 | 93.56 ± 1.35 | +1.67* (33/11, p=0.0013) | +0.08 (23/22, p=1) |
| FlipQuant 8x64 | 92.27 ± 1.47 | 92.19 ± 1.48 | +0.53 (19/12, p=0.28) | -1.06* (13/27, p=0.038) |
| FlipQuant 16x64 | 92.12 ± 1.48 | 92.04 ± 1.49 | +0.38 (21/16, p=0.51) | -1.21* (15/31, p=0.026) |
| FlipQuant 256x64 | 92.42 ± 1.46 | 92.34 ± 1.46 | +0.68 (20/11, p=0.15) | -0.91 (14/26, p=0.081) |

## Qwen3.8-27B

| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |
|---|---:|---:|---|---|
| BF16 | 95.68 ± 1.12 | 95.83 ± 1.10 | +1.06◊ (30/16, p=0.054) | — |
| NVFP4 | 95.38 ± 1.16 | 95.60 ± 1.13 | +0.76 (36/26, p=0.25) | -0.30◊ (21/25, p=0.66) |
| FourOverSix | 94.62 ± 1.24 | 95.15 ± 1.18 | — | -1.06◊ (16/30, p=0.054) |
| FOCUS | 93.56 ± 1.35 | 94.31 ± 1.28 | -1.06 (26/40, p=0.11) | -2.12*◊ (12/40, p=0.00013) |
| GPTQ‡ | 96.06 ± 1.07 | 96.21 ± 1.05 | +1.44* (34/15, p=0.0094) | +0.38◊ (21/16, p=0.51) |
| IF4†¤ | 95.60 ± 1.13 | 95.91 ± 1.09 | +0.99 (33/20, p=0.098) | -0.08◊ (19/20, p=1) |
| MixFP4 (Zou)†¤ | 95.38 ± 1.16 | 95.53 ± 1.14 | +0.76 (32/22, p=0.22) | -0.30◊ (18/22, p=0.64) |
| FlipQuant 8x64 | 95.53 ± 1.14 | 95.83 ± 1.10 | +0.91 (31/19, p=0.12) | -0.15◊ (18/20, p=0.87) |
| FlipQuant 16x64 | 95.15 ± 1.18 | 95.60 ± 1.13 | +0.53 (31/24, p=0.42) | -0.53◊ (15/22, p=0.32) |
| FlipQuant 256x64 | 95.45 ± 1.15 | 95.60 ± 1.13 | +0.83 (28/17, p=0.14) | -0.23◊ (18/21, p=0.75) |

## Summary: strict-match per model

Mean = the unweighted mean over the models with all ten rows (6).

| method | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 45.41 | 85.90 | 47.01 | 87.41 | 93.33 | 95.68 | 75.79 |
| NVFP4 | 29.19 | 79.98 | 44.50 | 87.34 | 93.03 | 95.38 | 71.57 |
| FourOverSix | 29.72 | 79.45 | 44.50 | 88.93 | 91.74 | 94.62 | 71.49 |
| FOCUS | 32.15 | 77.86 | 43.37 | 89.16 | 92.72 | 93.56 | 71.47 |
| GPTQ‡ | 30.78 | 81.80 | 45.41 | 87.19 | 92.19 | 96.06 | 72.24 |
| IF4† | 33.51 | 80.44 | 45.19 | 87.11 | 92.65 | 95.60 | 72.42 |
| MixFP4 (Zou)† | 33.59 | 80.67 | 45.56 | 85.52 | 93.40 | 95.38 | 72.35 |
| FlipQuant 8x64 | 30.63 | 81.43 | 45.11 | 87.11 | 92.27 | 95.53 | 72.01 |
| FlipQuant 16x64 | 31.61 | 80.06 | 44.81 | 89.31 | 92.12 | 95.15 | 72.18 |
| FlipQuant 256x64 | 31.08 | 81.73 | 44.73 | 88.93 | 92.42 | 95.45 | 72.39 |

## Fidelity to BF16

Per problem against the same model's BF16 run: the extracted answer equal (strict, flexible) after the scorer's own normalization (lm-eval's exact_match: `,`, `$`, `#### ` and a trailing `.` removed, case ignored), correctness flips (row-only + BF16-only correct, strict), and byte-identical generations. Pooled = over every problem of the models shown. ◊ = the two runs had different batch sizes (Qwen3.8-27B BF16 ran at batch 48), so their batch compositions differ.

| method | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | pooled |
|---|---|---|---|---|---|---|---|
| NVFP4 | 30.7 / 31.5; 358 (72+286); 1.0 | 78.4 / 79.5; 200 (61+139); 2.3 | 44.4 / 52.3; 293 (130+163); 3.6 | 87.3 / 91.4; 117 (58+59); 13.0 | 96.1 / 96.1; 40 (18+22); 10.3 | 95.6 / 96.0; 46 (21+25); 3.1◊ | 72.1 / 74.5; 1054; 5.5 |
| FourOverSix | 32.8 / 35.1; 341 (67+274); 1.0 | 77.9 / 79.0; 211 (63+148); 1.9 | 45.0 / 52.4; 303 (135+168); 4.5 | 88.5 / 92.9; 108 (64+44); 15.2 | 95.8 / 95.9; 41 (10+31); 9.2 | 95.6 / 96.4; 46 (16+30); 2.5◊ | 72.6 / 75.3; 1050; 5.7 |
| FOCUS | 35.4 / 35.9; 321 (73+248); 1.3 | 76.3 / 78.1; 220 (57+163); 2.0 | 45.1 / 51.8; 274 (113+161); 2.9 | 88.2 / 91.9; 113 (68+45); 12.0 | 95.6 / 95.8; 42 (17+25); 13.0 | 94.6 / 95.8; 52 (12+40); 2.1◊ | 72.5 / 74.9; 1022; 5.5 |
| GPTQ‡ | 31.8 / 32.3; 361 (84+277); 1.1 | 80.9 / 81.9; 176 (61+115); 3.6 | 46.3 / 54.1; 287 (133+154); 4.3 | 86.4 / 92.3; 135 (66+69); 15.9 | 95.7 / 96.2; 41 (13+28); 15.9 | 96.2 / 96.7; 37 (21+16); 3.6◊ | 72.9 / 75.6; 1037; 7.4 |
| IF4† | 34.3 / 35.9; 339 (91+248); 1.3 | 79.1 / 79.9; 192 (60+132); 2.6 | 45.8 / 52.8; 278 (127+151); 3.5 | 87.4 / 91.3; 114 (55+59); 14.6 | 96.2 / 96.7; 37 (14+23); 12.1 | 95.9 / 96.2; 39 (19+20); 3.7◊ | 73.1 / 75.5; 999; 6.3 |
| MixFP4 (Zou)† | 36.8 / 37.4; 308 (76+232); 1.4 | 79.9 / 80.9; 187 (59+128); 3.3 | 46.4 / 53.6; 291 (136+155); 4.4 | 86.7 / 90.6; 125 (50+75); 16.5 | 95.8 / 96.3; 45 (23+22); 13.8 | 96.0 / 96.4; 40 (18+22); 3.9◊ | 73.6 / 75.9; 996; 7.2 |
| FlipQuant 8x64 | 32.8 / 34.0; 343 (74+269); 1.4 | 79.5 / 80.5; 193 (67+126); 2.5 | 45.5 / 52.4; 287 (131+156); 3.8 | 87.1 / 92.8; 112 (54+58); 15.8 | 95.5 / 95.7; 40 (13+27); 10.3 | 96.2 / 96.6; 38 (18+20); 3.0◊ | 72.8 / 75.3; 1013; 6.1 |
| FlipQuant 16x64 | 35.3 / 36.5; 314 (66+248); 1.6 | 79.4 / 80.1; 193 (58+135); 1.7 | 45.4 / 53.1; 301 (136+165); 4.2 | 88.6 / 92.8; 103 (64+39); 16.2 | 95.2 / 95.4; 46 (15+31); 13.3 | 96.1 / 96.7; 37 (15+22); 2.4◊ | 73.3 / 75.8; 994; 6.6 |
| FlipQuant 256x64 | 34.3 / 35.5; 341 (76+265); 0.8 | 80.7 / 82.0; 175 (60+115); 2.7 | 44.8 / 51.6; 298 (134+164); 3.9 | 88.1 / 92.3; 108 (64+44); 16.1 | 95.7 / 95.8; 40 (14+26); 13.0 | 96.1 / 96.4; 39 (18+21); 3.4◊ | 73.3 / 75.6; 1001; 6.7 |

Each cell: answer agreement strict / flexible (%); correctness flips (row-only + BF16-only); byte-identical generations (%).

**Agreement.** Both sides extracted an answer, and the two are equal after the scorer's normalization. A problem where neither side extracted an answer (`[invalid]`) is not agreement; it is counted below. One side without an answer is a disagreement.

### Generations without a strict-match answer

Per row, the problems whose generation has no strict-match answer (`[invalid]`: no "final answer is <number>"). In brackets: the problems where the row and BF16 both have none.

| method | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 25 | 28 | 160 | 54 | 3 | 20 | 290 |
| NVFP4 | 57 (2) | 40 (6) | 150 (83) | 35 (15) | 3 (2) | 19 (5) | 304 |
| FourOverSix | 115 (8) | 19 (1) | 132 (82) | 28 (11) | 4 (3) | 24 (5) | 322 |
| FOCUS | 49 (6) | 49 (6) | 140 (88) | 26 (14) | 4 (1) | 35 (7) | 303 |
| GPTQ‡ | 37 (2) | 28 (9) | 142 (92) | 60 (20) | 11 (3) | 17 (7) | 295 |
| IF4† | 81 (8) | 34 (5) | 172 (95) | 24 (12) | 9 (3) | 19 (9) | 339 |
| MixFP4 (Zou)† | 44 (4) | 36 (6) | 135 (77) | 32 (18) | 7 (1) | 19 (7) | 273 |
| FlipQuant 8x64 | 75 (5) | 26 (4) | 136 (81) | 54 (22) | 1 (1) | 17 (6) | 309 |
| FlipQuant 16x64 | 73 (3) | 24 (4) | 136 (81) | 31 (15) | 1 (1) | 21 (7) | 286 |
| FlipQuant 256x64 | 63 (1) | 25 (4) | 129 (75) | 29 (17) | 2 (0) | 15 (7) | 263 |

### FlipQuant 16x64 vs FOCUS, IF4, Zou and FourOverSix: per-problem agreement with BF16 (exact McNemar)

Strict-extracted answer equal to BF16's: FlipQuant-only agreeing / other-only agreeing, p. Pooled over the models shown.

| vs | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 | Nemotron-Nano-9B-v2 | Phi-4 | Qwen3.8-27B | pooled |
|---|---|---|---|---|---|---|---|
| FOCUS | 169 / 171, p=0.96 | 120 / 79, p=0.0045 | 152 / 148, p=0.86 | 45 / 39, p=0.59 | 26 / 31, p=0.6 | 39 / 20, p=0.018◊ | 551 / 488, p=0.054 |
| IF4† | 162 / 149, p=0.5 | 101 / 97, p=0.83 | 142 / 147, p=0.81 | 55 / 39, p=0.12 | 23 / 36, p=0.12 | 22 / 20, p=0.88◊ | 505 / 488, p=0.61 |
| MixFP4 (Zou)† | 149 / 169, p=0.29 | 88 / 95, p=0.66 | 142 / 155, p=0.49 | 76 / 50, p=0.026 | 25 / 33, p=0.36 | 23 / 22, p=1◊ | 503 / 524, p=0.53 |
| FourOverSix | 161 / 129, p=0.069 | 99 / 80, p=0.18 | 139 / 134, p=0.81 | 42 / 40, p=0.91 | 17 / 24, p=0.35 | 28 / 22, p=0.48◊ | 486 / 429, p=0.064 |

## Old vs new: part N (RTN rows) and O0 (BF16)

Strict-match accuracy new / old, the paired difference (new-only / old-only correct, exact McNemar p) and the share of problems whose generation is byte-identical. Old: part N (flipquant paper-sm120-runs f2a56f1) and O0 (BF16, part O), the same command, env and batch (Qwen3.8-27B BF16 48 in both).

| model | method | new | old | paired | byte-identical |
|---|---|---:|---:|---|---:|
| Nemotron-Nano-9B-v2 | BF16 | 87.41 | 87.41 | +0.00 (0/0, p=1) | 100.0 % |
| Qwen3.8-27B | BF16 | 95.68 | 95.68 | +0.00 (0/0, p=1) | 100.0 % |
| Nemotron-Nano-9B-v2 | NVFP4 | 87.34 | 87.34 | +0.00 (0/0, p=1) | 100.0 % |
| Nemotron-Nano-9B-v2 | FourOverSix | 88.93 | 88.93 | +0.00 (0/0, p=1) | 100.0 % |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 89.31 | 89.31 | +0.00 (0/0, p=1) | 100.0 % |
| Qwen3.8-27B | NVFP4 | 95.38 | 95.38 | +0.00 (0/0, p=1) | 100.0 % |
| Qwen3.8-27B | FourOverSix | 94.62 | 94.62 | +0.00 (0/0, p=1) | 100.0 % |
| Qwen3.8-27B | FlipQuant 16x64 | 95.15 | 95.15 | +0.00 (0/0, p=1) | 100.0 % |

## Deviations and notes

1. **Batch.** Qwen3.8-27B BF16 ran at batch 48: its record's batch_size and its logged command both say so. Every other run used 64. Qwen3.8-27B BF16's paired comparisons are marked ◊.
2. **X2 inside part Y (amendments 1 and 2).** Part X amendment 2's 20 zero-shot runs ran from 2026-10-15T12:26:51Z to 2026-10-15T15:36:32Z, between Llama-3.1-8B-Instruct's NVFP4 and FO6 rows.
   - X2's hold took effect at that boundary, not at the registered one (after part Y's native rows).
   - Part Y's driver waited for an idle GPU, by its own rule. Nothing was stopped, and every job ran alone on the GPU.
   - No part Y result depends on the position.
3. **Date pin (amendment 3), the rows marked §.** Llama-3.2-1B-Instruct's chat template writes today's date into its system header.
   - The model's native rows ran on 2026-10-15. Its IF4 and MixFP4 rows ran with the date pinned to "15 Oct 2026": gsm8k6_date.py pins transformers' strftime_now and nothing else.
   - Both passed the registered check: sample_digest equal to BF16's, so all 2,638 prompts are byte-identical.
   - The first, unpinned IF4 run (2026-10-16 01:59–02:50 UTC, "16 Oct 2026") stopped the panel on that check. Its record is kept as `if4_date-oct16.*` and is not a panel cell.
   - No other model's template writes a date. Llama-3.1-8B-Instruct's header is a fixed "26 Jul 2024".
4. **Old vs new and dates.** The old records compared (part N / O0: Nemotron, Qwen3.8-27B) have no date in their prompts. No older Llama-3.2-1B-Instruct GSM8K record exists, because parts N / O and tab:ptq cover Nemotron and Qwen3.8-27B only, so no "Today Date" line is compared.
5. **Allocator and batch (amendment 4 and its addendum), the rows marked ¤.** Qwen3.8-27B's IF4 and MixFP4 rows ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Under the default allocator, IF4 had run out of memory at batch 64 from fragmentation (if4_oom-b64-default-alloc.log).
   - Before the full rows, a --limit 64 pre-check showed the allocator changes no output: 128 / 128 samples byte-identical. Every other row ran under the default allocator, as registered.
   - The batch each ran at: IF4† 64; MixFP4 (Zou)† 64.
   - A row below 64 is marked ◊ against batch-64 rows. A row at 48 shares the batch composition of Qwen3.8-27B BF16 (48), so that pairing carries no ◊.
6. **Simulated rows' cost** (process seconds, against the model's native FourOverSix row): Llama-3.2-1B-Instruct IF4 6.7x, MixFP4 6.8x; Llama-3.1-8B-Instruct IF4 5.1x, MixFP4 4.5x; Mistral-7B-Instruct-v0.3 IF4 4.5x, MixFP4 3.8x; Nemotron-Nano-9B-v2 IF4 4.6x, MixFP4 4.5x; Phi-4 IF4 2.8x, MixFP4 2.5x; Qwen3.8-27B IF4 6.9x, MixFP4 5.7x.

## Timings and generated lengths

Per run: model load and the task's wall time (s), mean / max generated tokens (to the first end-of-sequence token), and the problems that used the whole 1,024-token budget.

| model | method | batch | load | task | total | mean tokens | max | budget hits | start (UTC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-1B-Instruct | BF16 | 64 | 4 | 322 | 330 | 98 | 1024 | 7 | 2026-10-15T08:20:36Z |
| Llama-3.2-1B-Instruct | NVFP4 | 64 | 5 | 474 | 482 | 112 | 1024 | 24 | 2026-10-15T08:26:12Z |
| Llama-3.2-1B-Instruct | FourOverSix | 64 | 5 | 325 | 333 | 96 | 1024 | 7 | 2026-10-15T08:34:20Z |
| Llama-3.2-1B-Instruct | FOCUS | 64 | 5 | 499 | 507 | 100 | 1024 | 17 | 2026-10-15T08:39:59Z |
| Llama-3.2-1B-Instruct | GPTQ‡ | 64 | 6 | 251 | 260 | 90 | 1024 | 4 | 2026-10-15T08:48:33Z |
| Llama-3.2-1B-Instruct | IF4†§ | 64 | 9 | 2227 | 2239 | 103 | 1024 | 14 | 2026-10-16T02:54:51Z |
| Llama-3.2-1B-Instruct | MixFP4 (Zou)†§ | 64 | 8 | 2254 | 2265 | 104 | 1024 | 14 | 2026-10-16T03:32:16Z |
| Llama-3.2-1B-Instruct | FlipQuant 8x64 | 64 | 6 | 245 | 254 | 91 | 1024 | 7 | 2026-10-15T08:52:59Z |
| Llama-3.2-1B-Instruct | FlipQuant 16x64 | 64 | 6 | 272 | 281 | 88 | 1024 | 5 | 2026-10-15T08:57:19Z |
| Llama-3.2-1B-Instruct | FlipQuant 256x64 | 64 | 6 | 310 | 319 | 93 | 1024 | 7 | 2026-10-15T09:02:06Z |
| Llama-3.1-8B-Instruct | BF16 | 64 | 6 | 1574 | 1583 | 128 | 1024 | 10 | 2026-10-15T11:34:28Z |
| Llama-3.1-8B-Instruct | NVFP4 | 64 | 10 | 1530 | 1543 | 125 | 1024 | 11 | 2026-10-15T12:00:57Z |
| Llama-3.1-8B-Instruct | FourOverSix | 64 | 12 | 1198 | 1213 | 118 | 1024 | 5 | 2026-10-15T15:36:33Z |
| Llama-3.1-8B-Instruct | FOCUS | 64 | 12 | 1876 | 1891 | 127 | 1024 | 16 | 2026-10-15T15:56:57Z |
| Llama-3.1-8B-Instruct | GPTQ‡ | 64 | 16 | 1366 | 1385 | 125 | 1024 | 7 | 2026-10-15T16:28:37Z |
| Llama-3.1-8B-Instruct | IF4† | 64 | 39 | 6154 | 6196 | 130 | 1024 | 12 | 2026-10-16T06:51:13Z |
| Llama-3.1-8B-Instruct | MixFP4 (Zou)† | 64 | 36 | 5439 | 5478 | 127 | 1024 | 14 | 2026-10-16T08:34:35Z |
| Llama-3.1-8B-Instruct | FlipQuant 8x64 | 64 | 14 | 1149 | 1166 | 117 | 1024 | 7 | 2026-10-15T16:51:49Z |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 | 64 | 14 | 1181 | 1197 | 118 | 1024 | 9 | 2026-10-15T17:11:21Z |
| Llama-3.1-8B-Instruct | FlipQuant 256x64 | 64 | 14 | 901 | 917 | 117 | 1024 | 2 | 2026-10-15T17:31:24Z |
| Mistral-7B-Instruct-v0.3 | BF16 | 64 | 4 | 1010 | 1017 | 124 | 1024 | 1 | 2026-10-15T09:07:31Z |
| Mistral-7B-Instruct-v0.3 | NVFP4 | 64 | 8 | 985 | 996 | 128 | 1024 | 1 | 2026-10-15T09:24:33Z |
| Mistral-7B-Instruct-v0.3 | FourOverSix | 64 | 11 | 1145 | 1159 | 126 | 1024 | 2 | 2026-10-15T09:41:15Z |
| Mistral-7B-Instruct-v0.3 | FOCUS | 64 | 12 | 1076 | 1091 | 122 | 1024 | 3 | 2026-10-15T10:00:43Z |
| Mistral-7B-Instruct-v0.3 | GPTQ‡ | 64 | 22 | 1021 | 1047 | 126 | 1024 | 2 | 2026-10-15T10:19:04Z |
| Mistral-7B-Instruct-v0.3 | IF4† | 64 | 35 | 5216 | 5255 | 137 | 1024 | 3 | 2026-10-16T04:10:07Z |
| Mistral-7B-Instruct-v0.3 | MixFP4 (Zou)† | 64 | 35 | 4361 | 4400 | 134 | 1024 | 2 | 2026-10-16T05:37:47Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 | 64 | 13 | 1059 | 1075 | 124 | 1024 | 3 | 2026-10-15T10:36:37Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 | 64 | 13 | 1070 | 1085 | 126 | 1024 | 3 | 2026-10-15T10:54:37Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 | 64 | 15 | 1276 | 1294 | 130 | 1024 | 6 | 2026-10-15T11:12:49Z |
| Nemotron-Nano-9B-v2 | BF16 | 64 | 7 | 439 | 449 | 114 | 503 | 0 | 2026-10-15T17:46:48Z |
| Nemotron-Nano-9B-v2 | NVFP4 | 64 | 12 | 327 | 342 | 112 | 608 | 0 | 2026-10-15T17:54:23Z |
| Nemotron-Nano-9B-v2 | FourOverSix | 64 | 14 | 350 | 367 | 115 | 553 | 0 | 2026-10-15T08:07:41Z |
| Nemotron-Nano-9B-v2 | FOCUS | 64 | 14 | 315 | 332 | 110 | 507 | 0 | 2026-10-15T18:00:16Z |
| Nemotron-Nano-9B-v2 | GPTQ‡ | 64 | 19 | 327 | 349 | 117 | 443 | 0 | 2026-10-15T18:06:00Z |
| Nemotron-Nano-9B-v2 | IF4† | 64 | 51 | 1651 | 1704 | 106 | 481 | 0 | 2026-10-16T10:05:59Z |
| Nemotron-Nano-9B-v2 | MixFP4 (Zou)† | 64 | 40 | 1593 | 1636 | 107 | 505 | 0 | 2026-10-16T10:34:28Z |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 64 | 16 | 363 | 383 | 118 | 582 | 0 | 2026-10-15T18:11:55Z |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 64 | 17 | 357 | 376 | 119 | 574 | 0 | 2026-10-15T08:13:55Z |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 64 | 17 | 349 | 369 | 116 | 573 | 0 | 2026-10-15T18:18:23Z |
| Phi-4 | BF16 | 64 | 7 | 1329 | 1339 | 125 | 493 | 0 | 2026-10-15T18:24:38Z |
| Phi-4 | NVFP4 | 64 | 15 | 1135 | 1154 | 132 | 459 | 0 | 2026-10-15T18:47:03Z |
| Phi-4 | FourOverSix | 64 | 20 | 1204 | 1227 | 138 | 508 | 0 | 2026-10-15T19:06:22Z |
| Phi-4 | FOCUS | 64 | 20 | 1325 | 1349 | 130 | 1024 | 1 | 2026-10-15T19:27:05Z |
| Phi-4 | GPTQ‡ | 64 | 28 | 1088 | 1120 | 120 | 425 | 0 | 2026-10-15T19:49:47Z |
| Phi-4 | IF4† | 64 | 85 | 3355 | 3443 | 136 | 548 | 0 | 2026-10-16T11:01:50Z |
| Phi-4 | MixFP4 (Zou)† | 64 | 66 | 2983 | 3053 | 134 | 530 | 0 | 2026-10-16T11:59:19Z |
| Phi-4 | FlipQuant 8x64 | 64 | 24 | 1158 | 1185 | 135 | 459 | 0 | 2026-10-15T20:08:32Z |
| Phi-4 | FlipQuant 16x64 | 64 | 24 | 1127 | 1154 | 130 | 449 | 0 | 2026-10-15T20:28:23Z |
| Phi-4 | FlipQuant 256x64 | 64 | 24 | 1153 | 1180 | 132 | 481 | 0 | 2026-10-15T20:47:42Z |
| Qwen3.8-27B | BF16 | 48 | 13 | 2955 | 2971 | 232 | 1024 | 20 | 2026-10-15T21:07:28Z |
| Qwen3.8-27B | NVFP4 | 64 | 27 | 1874 | 1905 | 163 | 1024 | 7 | 2026-10-15T21:57:05Z |
| Qwen3.8-27B | FourOverSix | 64 | 35 | 2003 | 2042 | 185 | 1024 | 13 | 2026-10-15T22:28:56Z |
| Qwen3.8-27B | FOCUS | 64 | 34 | 1989 | 2025 | 167 | 1024 | 12 | 2026-10-15T23:03:21Z |
| Qwen3.8-27B | GPTQ‡ | 64 | 59 | 2174 | 2238 | 208 | 1024 | 17 | 2026-10-15T23:37:27Z |
| Qwen3.8-27B | IF4†¤ | 64 | 116 | 13913 | 14032 | 200 | 1024 | 16 | 2026-10-16T13:54:53Z |
| Qwen3.8-27B | MixFP4 (Zou)†¤ | 64 | 116 | 11607 | 11727 | 200 | 1024 | 13 | 2026-10-16T17:48:52Z |
| Qwen3.8-27B | FlipQuant 8x64 | 64 | 80 | 2062 | 2145 | 188 | 1024 | 13 | 2026-10-16T00:14:54Z |
| Qwen3.8-27B | FlipQuant 16x64 | 64 | 40 | 1960 | 2003 | 185 | 1024 | 13 | 2026-10-16T00:50:48Z |
| Qwen3.8-27B | FlipQuant 256x64 | 64 | 42 | 2065 | 2110 | 179 | 1024 | 13 | 2026-10-16T01:24:17Z |

Sum of the process times: 31.7 h.

## LaTeX (strict-match, %)

```latex
Method & Llama-3.2-1B-Instruct & Llama-3.1-8B-Instruct & Mistral-7B-Instruct-v0.3 & Nemotron-Nano-9B-v2 & Phi-4 & Qwen3.8-27B & Mean \\
\midrule
BF16 & 45.41 & 85.90 & 47.01 & 87.41 & 93.33 & 95.68 & 75.79 \\
NVFP4 & 29.19 & 79.98 & 44.50 & 87.34 & 93.03 & 95.38 & 71.57 \\
FourOverSix & 29.72 & 79.45 & 44.50 & 88.93 & 91.74 & 94.62 & 71.49 \\
FOCUS & 32.15 & 77.86 & 43.37 & 89.16 & 92.72 & 93.56 & 71.47 \\
GPTQ$^{\ddagger}$ & 30.78 & 81.80 & 45.41 & 87.19 & 92.19 & 96.06 & 72.24 \\
IF4$^{\dagger}$ & 33.51 & 80.44 & 45.19 & 87.11 & 92.65 & 95.60 & 72.42 \\
MixFP4$^{\dagger}$ & 33.59 & 80.67 & 45.56 & 85.52 & 93.40 & 95.38 & 72.35 \\
\method{} $8{\times}64$ & 30.63 & 81.43 & 45.11 & 87.11 & 92.27 & 95.53 & 72.01 \\
\method{} $16{\times}64$ & 31.61 & 80.06 & 44.81 & 89.31 & 92.12 & 95.15 & 72.18 \\
\method{} $256{\times}64$ & 31.08 & 81.73 & 44.73 & 88.93 & 92.42 & 95.45 & 72.39 \\
```
