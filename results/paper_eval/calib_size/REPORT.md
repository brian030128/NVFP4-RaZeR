# Calibration size: FlipQuant 16x64 and FOCUS at 1,024 and 2,048 windows (part AA)

PROTOCOL.md part AA and its amendment 1. FlipQuant 16x64 (release settings: 5 epochs, top-1000 teacher, seed 0, last-step map) and FOCUS (part K's FOCUS_COMMON: 1 epoch, global batch 32, micro 8) trained on 1,024 or 2,048 windows of 512 tokens instead of the release's 256. The sets are nested, 256 ⊂ 1,024 ⊂ 2,048, and audited against every evaluation text (WINDOWS.md). FlipQuant and FOCUS read the same windows (token sha256 checked). flipquant calib-size f912a30. Evaluation exactly as parts X (zero-shot) and Y (GSM8K), plus PPL (paper convention). Step 0: STEP0.md. 2,048-window memory probes: PROBES.md.

## Summary

Per model: zero-shot Avg. (%), GSM8K strict-match (%) and WikiText-2 PPL of each method and calibration size (TBD: not run yet; —: not in the scope).

**Zero-shot Avg. (%):**

| method | Llama-1B | Phi-4 |
|---|---:|---:|
| FourOverSix | 56.24 | 72.39 |
| FOCUS, 256 | 57.40 | 73.14 |
| FOCUS, 1,024 | 57.90 | 73.05 |
| FOCUS, 2,048 | TBD | TBD |
| FlipQuant 16x64, 256 | 57.25 | 72.58 |
| FlipQuant 16x64, 1,024 | 57.93 | 72.78 |
| FlipQuant 16x64, 2,048 | TBD | TBD |

**GSM8K strict-match (%):**

| method | Llama-1B | Phi-4 |
|---|---:|---:|
| FourOverSix | 29.72 | 91.74 |
| FOCUS, 256 | 32.15 | 92.72 |
| FOCUS, 1,024 | 32.22 | 91.51 |
| FOCUS, 2,048 | TBD | TBD |
| FlipQuant 16x64, 256 | 31.61 | 92.12 |
| FlipQuant 16x64, 1,024 | 35.18 | 93.18 |
| FlipQuant 16x64, 2,048 | TBD | TBD |

**WikiText-2 PPL:**

| method | Llama-1B | Phi-4 |
|---|---:|---:|
| FourOverSix | 15.315 | 6.662 |
| FOCUS, 256 | 14.661 | 6.628 |
| FOCUS, 1,024 | 14.687 | 6.641 |
| FOCUS, 2,048 | TBD | TBD |
| FlipQuant 16x64, 256 | 14.454 | 6.633 |
| FlipQuant 16x64, 1,024 | 14.387 | 6.618 |
| FlipQuant 16x64, 2,048 | TBD | TBD |

**Pooled over the models** (discordant counts summed; difference in points over all pooled questions; exact McNemar p; * = p < 0.05):

| comparison | zero-shot | models | GSM8K | models |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.20 (1622/1711, p=0.13) | Llama-1B, Phi-4 | -0.57 (153/168, p=0.43) | Llama-1B, Phi-4 |
| FQ16-1k vs FourOverSix | +1.04* (1813/1360, p=9.2e-16) | Llama-1B, Phi-4 | +3.45* (202/111, p=3e-07) | Llama-1B, Phi-4 |
| FQ16-1k vs FQ16-256 | +0.21 (1492/1402, p=0.098) | Llama-1B, Phi-4 | +2.31* (191/130, p=0.00079) | Llama-1B, Phi-4 |
| FOCUS-1k vs FourOverSix | +1.20* (2050/1525, p=1.6e-18) | Llama-1B, Phi-4 | +1.14 (190/160, p=0.12) | Llama-1B, Phi-4 |
| FOCUS-1k vs FOCUS-256 | +0.17 (1546/1473, p=0.19) | Llama-1B, Phi-4 | -0.57 (153/168, p=0.43) | Llama-1B, Phi-4 |
| FQ16-1k vs FOCUS-1k | -0.16 (1638/1710, p=0.22) | Llama-1B, Phi-4 | +2.31* (201/140, p=0.0011) | Llama-1B, Phi-4 |
| FQ16-1k vs IF4† | +0.57* (1906/1657, p=3.2e-05) | Llama-1B, Phi-4 | +1.10 (192/163, p=0.14) | Llama-1B, Phi-4 |
| FQ16-1k vs MixFP4 (Zou)† | +0.45* (1883/1685, p=0.00097) | Llama-1B, Phi-4 | +0.68 (166/148, p=0.34) | Llama-1B, Phi-4 |
| FQ16-256 vs IF4† | +0.36* (1849/1690, p=0.0079) | Llama-1B, Phi-4 | -1.21 (149/181, p=0.088) | Llama-1B, Phi-4 |
| FQ16-256 vs MixFP4 (Zou)† | +0.25 (1867/1759, p=0.076) | Llama-1B, Phi-4 | -1.63* (148/191, p=0.022) | Llama-1B, Phi-4 |

## Llama-3.2-1B-Instruct

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 | 59.92 | 45.41 ± 2.74 | 13.163 | 18.519 |
| NVFP4 | 71.65 | 57.72 | 59.91 | 59.97 | 36.26 | 53.10 | 56.43 | 29.19 ± 2.50 | 15.535 | 21.767 |
| FourOverSix | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 | 56.24 | 29.72 ± 2.52 | 15.315 | 21.449 |
| IF4† | 71.93 | 58.31 | 60.77 | 60.52 | 36.18 | 54.86 | 57.09 | 33.51 ± 2.60 | 15.345 | 21.074 |
| MixFP4 (Zou)† | 74.21 | 58.50 | 60.30 | 60.35 | 35.84 | 54.82 | 57.34 | 33.59 ± 2.60 | 15.248 | 21.049 |
| GPTQ‡ | 71.06 | 57.62 | 59.91 | 59.89 | 36.26 | 54.49 | 56.54 | 30.78 ± 2.54 | 15.533 | 21.019 |
| FlipQuant 8x64 | 72.52 | 59.59 | 59.27 | 62.25 | 37.80 | 56.10 | 57.92 | 30.63 ± 2.54 | 14.453 | 19.679 |
| FlipQuant 256x64 | 73.07 | 59.01 | 61.01 | 60.82 | 36.35 | 56.08 | 57.72 | 31.08 ± 2.55 | 14.657 | 20.059 |
| FOCUS, 256 | 72.58 | 59.57 | 59.12 | 60.86 | 37.03 | 55.23 | 57.40 | 32.15 ± 2.57 | 14.661 | 19.885 |
| FOCUS, 1,024 | 72.74 | 59.27 | 59.67 | 61.91 | 36.86 | 56.92 | 57.90 | 32.22 ± 2.57 | 14.687 | 19.817 |
| FOCUS, 2,048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlipQuant 16x64, 256 | 72.85 | 59.31 | 58.09 | 61.15 | 36.35 | 55.75 | 57.25 | 31.61 ± 2.56 | 14.454 | 19.684 |
| FlipQuant 16x64, 1,024 | 73.39 | 59.36 | 60.46 | 62.08 | 36.26 | 56.05 | 57.93 | 35.18 ± 2.63 | 14.387 | 19.605 |
| FlipQuant 16x64, 2,048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FOCUS, 256 (retrained, step 0 d) | 72.14 | 59.05 | 59.75 | 61.91 | 37.71 | 55.66 | 57.70 | 34.04 ± 2.61 | 14.685 | 19.881 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.04 (1133/1141, p=0.88) | -0.53 (132/139, p=0.72) | -0.0143 ± 0.0023* | -0.0101 ± 0.0018* |
| FQ16-1k vs FourOverSix | +1.82* (1336/939, p=8.6e-17) | +5.46* (175/103, p=1.9e-05) | -0.0625 ± 0.0029* | -0.0899 ± 0.0039* |
| FQ16-1k vs FQ16-256 | +0.37 (1046/965, p=0.074) | +3.56* (166/119, p=0.0063) | -0.0046 ± 0.0021* | -0.0040 ± 0.0016* |
| FOCUS-1k vs FourOverSix | +1.89* (1420/1006, p=4.3e-17) | +2.50 (161/128, p=0.06) | -0.0419 ± 0.0030* | -0.0791 ± 0.0043* |
| FOCUS-1k vs FOCUS-256 | +0.41 (1085/995, p=0.051) | +0.08 (134/133, p=1) | +0.0018 ± 0.0023 | -0.0034 ± 0.0019* |
| FQ16-1k vs FOCUS-1k | -0.08 (1135/1152, p=0.74) | +2.96* (164/125, p=0.025) | -0.0207 ± 0.0022* | -0.0107 ± 0.0019* |
| FQ16-1k vs IF4† | +1.04* (1359/1131, p=5.3e-06) | +1.67 (168/146, p=0.24) | -0.0645 ± 0.0034* | -0.0722 ± 0.0037* |
| FQ16-1k vs MixFP4 (Zou)† | +0.83* (1326/1144, p=0.00027) | +1.59 (152/131, p=0.23) | -0.0582 ± 0.0033* | -0.0711 ± 0.0036* |
| FQ16-2k vs FourOverSix | TBD | TBD | TBD | TBD |
| FQ16-2k vs FQ16-256 | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FourOverSix | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FOCUS-256 | TBD | TBD | TBD | TBD |
| FQ16-2k vs FOCUS-2k | TBD | TBD | TBD | TBD |
| FQ16-2k vs IF4† | TBD | TBD | TBD | TBD |
| FQ16-2k vs MixFP4 (Zou)† | TBD | TBD | TBD | TBD |
| FQ16-2k vs FQ16-1k | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FOCUS-1k | TBD | TBD | TBD | TBD |
| FQ16-256 vs IF4† | +0.67* (1272/1125, p=0.0029) | -1.90 (128/153, p=0.15) | -0.0598 ± 0.0035* | -0.0682 ± 0.0038* |
| FQ16-256 vs MixFP4 (Zou)† | +0.46* (1292/1191, p=0.045) | -1.97 (130/156, p=0.14) | -0.0535 ± 0.0034* | -0.0670 ± 0.0037* |
| FOCUS-256 retrained vs FOCUS-256 | +0.01 (967/964, p=0.96) | +1.90 (145/120, p=0.14) | +0.0016 ± 0.0021 | -0.0002 ± 0.0014 |
| FOCUS-1k vs FOCUS-256 retrained | +0.40 (1067/980, p=0.057) | -1.82 (131/155, p=0.17) | +0.0002 ± 0.0025 | -0.0032 ± 0.0019* |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.54 (88/78, p=0.48) | +0.05 (299/294, p=0.87) | +2.37 (164/134, p=0.093) | +0.93 (131/109, p=0.18) | -0.09 (60/61, p=1) | +0.29 (304/289, p=0.57) |
| FOCUS-1k vs FOCUS-256 | +0.16 (89/86, p=0.88) | -0.30 (299/329, p=0.25) | +0.55 (147/140, p=0.72) | +1.05 (124/99, p=0.11) | -0.17 (63/65, p=0.93) | +1.69* (363/276, p=0.00066) |
| FQ16-1k vs FOCUS-1k | +0.65 (97/85, p=0.41) | +0.09 (366/357, p=0.77) | +0.79 (157/147, p=0.61) | +0.17 (134/130, p=0.85) | -0.60 (61/68, p=0.6) | -0.87 (320/365, p=0.093) |
| FQ16-256 vs FOCUS-256 | +0.27 (95/90, p=0.77) | -0.26 (328/354, p=0.34) | -1.03 (156/169, p=0.51) | +0.29 (136/129, p=0.71) | -0.68 (66/74, p=0.55) | +0.52 (352/325, p=0.32) |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | part P | 160 | 0.0281 | 83 | 39 | 14.40 | 15842 | 3.27 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.1484 | 225 | 156 | 16.60 | 18112 | 3.25 | 3,141,677,056 |
| FOCUS | 256 | part P | 8 | — | — | 35 | 13.57 | — | — | 729,889,075 |
| FOCUS | 1,024 | this part | 32 | — | 173 | 139 | 13.57 | 15650 | — | 729,889,075 |

Evaluation wall times (s): gsm8k_focus256_rerun 340, gsm8k_focus_1024 386, gsm8k_fq16_1024 292, gsm8k_fq16_256 285, ppl_fo6 19, ppl_focus256 18, ppl_focus256_rerun 19, ppl_focus_1024 19, ppl_fq16_1024 19, ppl_fq16_256 19, zs_focus256_rerun 186, zs_focus_1024 186, zs_fq16_1024 188, zs_fq16_256 183.

## Phi-4

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 81.12 | 81.95 | 76.95 | 72.77 | 56.14 | 72.27 | 73.53 | 93.33 ± 1.37 | 6.462 | 10.310 |
| NVFP4 | 80.85 | 80.86 | 73.64 | 73.23 | 54.10 | 71.88 | 72.43 | 93.03 ± 1.40 | 6.693 | 10.580 |
| FourOverSix | 80.69 | 81.11 | 74.27 | 71.89 | 54.86 | 71.55 | 72.39 | 91.74 ± 1.52 | 6.662 | 10.544 |
| IF4† | 81.01 | 81.15 | 74.82 | 72.52 | 55.38 | 71.49 | 72.73 | 92.65 ± 1.44 | 6.635 | 10.511 |
| MixFP4 (Zou)† | 81.23 | 81.14 | 73.09 | 73.74 | 55.89 | 71.28 | 72.73 | 93.40 ± 1.37 | 6.639 | 10.507 |
| GPTQ‡ | 80.63 | 80.91 | 73.09 | 72.98 | 56.06 | 72.02 | 72.61 | 92.19 ± 1.48 | 6.688 | 10.553 |
| FlipQuant 8x64 | 80.90 | 81.12 | 74.51 | 71.93 | 55.38 | 71.73 | 72.59 | 92.27 ± 1.47 | 6.634 | 10.519 |
| FlipQuant 256x64 | 81.01 | 81.19 | 74.82 | 71.51 | 54.69 | 72.07 | 72.55 | 92.42 ± 1.46 | 6.638 | 10.518 |
| FOCUS, 256 | 80.52 | 81.48 | 76.32 | 72.81 | 55.38 | 72.33 | 73.14 | 92.72 ± 1.43 | 6.628 | 10.523 |
| FOCUS, 1,024 | 81.28 | 81.32 | 75.77 | 73.19 | 54.52 | 72.19 | 73.05 | 91.51 ± 1.54 | 6.641 | 10.536 |
| FOCUS, 2,048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlipQuant 16x64, 256 | 81.01 | 81.35 | 74.66 | 72.01 | 54.69 | 71.76 | 72.58 | 92.12 ± 1.48 | 6.633 | 10.516 |
| FlipQuant 16x64, 1,024 | 81.39 | 81.18 | 74.43 | 72.69 | 55.20 | 71.76 | 72.78 | 93.18 ± 1.39 | 6.618 | 10.503 |
| FlipQuant 16x64, 2,048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.37* (489/570, p=0.014) | -0.61 (21/29, p=0.32) | +0.0007 ± 0.0014 | -0.0007 ± 0.0009 |
| FQ16-1k vs FourOverSix | +0.26 (477/421, p=0.066) | +1.44* (27/8, p=0.0019) | -0.0066 ± 0.0014* | -0.0039 ± 0.0009* |
| FQ16-1k vs FQ16-256 | +0.04 (446/437, p=0.79) | +1.06* (25/11, p=0.029) | -0.0022 ± 0.0013* | -0.0012 ± 0.0007* |
| FOCUS-1k vs FourOverSix | +0.51* (630/519, p=0.0012) | -0.23 (29/32, p=0.8) | -0.0032 ± 0.0017* | -0.0007 ± 0.0012 |
| FOCUS-1k vs FOCUS-256 | -0.08 (461/478, p=0.6) | -1.21* (19/35, p=0.04) | +0.0019 ± 0.0013* | +0.0012 ± 0.0008* |
| FQ16-1k vs FOCUS-1k | -0.25 (503/558, p=0.097) | +1.67* (37/15, p=0.0032) | -0.0034 ± 0.0016* | -0.0032 ± 0.0009* |
| FQ16-1k vs IF4† | +0.10 (547/526, p=0.54) | +0.53 (24/17, p=0.35) | -0.0025 ± 0.0018* | -0.0007 ± 0.0010 |
| FQ16-1k vs MixFP4 (Zou)† | +0.07 (557/541, p=0.65) | -0.23 (14/17, p=0.72) | -0.0032 ± 0.0019* | -0.0003 ± 0.0009 |
| FQ16-2k vs FourOverSix | TBD | TBD | TBD | TBD |
| FQ16-2k vs FQ16-256 | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FourOverSix | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FOCUS-256 | TBD | TBD | TBD | TBD |
| FQ16-2k vs FOCUS-2k | TBD | TBD | TBD | TBD |
| FQ16-2k vs IF4† | TBD | TBD | TBD | TBD |
| FQ16-2k vs MixFP4 (Zou)† | TBD | TBD | TBD | TBD |
| FQ16-2k vs FQ16-1k | TBD | TBD | TBD | TBD |
| FOCUS-2k vs FOCUS-1k | TBD | TBD | TBD | TBD |
| FQ16-256 vs IF4† | +0.05 (577/565, p=0.74) | -0.53 (21/28, p=0.39) | -0.0003 ± 0.0017 | +0.0005 ± 0.0011 |
| FQ16-256 vs MixFP4 (Zou)† | +0.03 (575/568, p=0.86) | -1.29* (18/35, p=0.027) | -0.0010 ± 0.0018 | +0.0009 ± 0.0010 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.38 (34/27, p=0.44) | -0.17 (106/123, p=0.29) | -0.24 (79/82, p=0.87) | +0.67 (55/39, p=0.12) | +0.51 (29/23, p=0.49) | +0.00 (143/143, p=1) |
| FOCUS-1k vs FOCUS-256 | +0.76 (37/23, p=0.092) | -0.16 (123/139, p=0.35) | -0.55 (73/80, p=0.63) | +0.38 (59/50, p=0.44) | -0.85 (31/41, p=0.29) | -0.14 (138/145, p=0.72) |
| FQ16-1k vs FOCUS-1k | +0.11 (34/32, p=0.9) | -0.14 (135/149, p=0.44) | -1.34 (82/99, p=0.23) | -0.51 (49/61, p=0.29) | +0.68 (37/29, p=0.39) | -0.43 (166/188, p=0.26) |
| FQ16-256 vs FOCUS-256 | +0.49 (40/31, p=0.34) | -0.13 (146/159, p=0.49) | -1.66 (80/101, p=0.14) | -0.80 (42/61, p=0.076) | -0.68 (28/36, p=0.38) | -0.56 (153/182, p=0.13) |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0060 | 502 | 376 | 59.91 | 65242 | 28.23 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0369 | 1688 | 1500 | 62.11 | 67634 | 28.26 | 3,141,677,056 |
| FOCUS | 256 | part K | 8 | — | — | 413 | 91.04 | — | — | 10,223,745,627 |
| FOCUS | 1,024 | this part | 32 | — | 1702 | 1650 | 91.04 | 96148 | — | 10,223,745,627 |

Evaluation wall times (s): gsm8k_focus_1024 1176, gsm8k_fq16_1024 1182, ppl_fo6 69, ppl_focus256 69, ppl_focus_1024 68, ppl_fq16_1024 73, ppl_fq16_256 73, zs_focus_1024 360, zs_fq16_1024 369.

