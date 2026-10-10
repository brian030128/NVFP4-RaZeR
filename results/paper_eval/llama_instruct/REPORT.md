# Part P: Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct in tab:main-ppl

PROTOCOL.md part P. Every row with the existing main-table settings, n16k64, native sm120 (build_V, `--kernel-set auto`) except IF4 / MixFP4 (Zou), simulated as part I. The final table is `results/paper_eval/final/` (tables_final.py). ΔNLL x 1e-3 ± 2 SE, * beyond 2 SE.

## Llama-3.2-1B-Instruct

| unit | wall (s) | trainer total (s) | device peak (MiB) | GPU allocated peak (GiB) | E0M3 share | fit extension |
|---|---:|---:|---:|---:|---:|---|
| 8x64 | 81 | 70 | 15,862 | 14.4 | 38,587 / 1,900,544 (2.030 %) | all checks pass |
| 16x64 | 83 | 71 | 15,842 | 14.4 | 26,656 / 950,272 (2.805 %) | all checks pass |
| 256x64 | 84 | 72 | 15,826 | 14.4 | 6,757 / 59,392 (11.377 %) | all checks pass |

Calibration windows (256 x 512): the 128 of the paper rule (`builder_seed0` with this model's tokenizer; not taken from an archived record) and the 128 of the extension rule (seed 20260930). Compared with Llama-3.1-8B's release windows (not required: this model's fit set is drawn with its own tokenizer and its development records are its own draws): fit equal True, extension equal False (0 of 128 extension windows shared; 128 of the base model's 128 extension documents are among this model's drawn development documents, which the extension rule skips: the development rule takes the next documents of the same streams).

| row | WikiText-2 | C4 | ΔNLL vs FourOverSix (wiki / C4) | FlipQuant vs NVFP4 (wiki / C4) |
|---|---:|---:|---:|---:|
| BF16 | 13.1629 | 18.5189 | -151.44 ± 3.73* / -146.88 ± 5.83* |  |
| NVFP4 | 15.5352 | 21.7669 | +14.26 ± 2.49* / +14.72 ± 2.36* |  |
| FourOverSix | 15.3152 | 21.4488 |  |  |
| IF4 | 15.3448 | 21.0737 | +1.93 ± 3.10 / -17.65 ± 2.64* |  |
| MixFP4 (Zou) | 15.2483 | 21.0489 | -4.38 ± 3.04* / -18.82 ± 2.54* |  |
| GPTQ‡ | 15.5327 | 21.0190 | +14.10 ± 39.30 / -20.25 ± 3.36* |  |
| FOCUS | 14.6611 | 19.8847 | -43.65 ± 2.93* / -75.72 ± 3.87* |  |
| FlipQuant 8x64 | 14.4527 | 19.6791 | -57.97 ± 2.90* / -86.11 ± 3.91* | -72.23 ± 3.62* / -100.83 ± 4.79* |
| FlipQuant 16x64 | 14.4536 | 19.6842 | -57.90 ± 2.85* / -85.86 ± 3.83* | -72.16 ± 3.49* / -100.58 ± 4.65* |
| FlipQuant 256x64 | 14.6573 | 20.0588 | -43.91 ± 2.84* / -67.00 ± 3.33* | -58.17 ± 3.31* / -81.72 ± 4.10* |

Largest single-window NLL difference vs FourOverSix (nats; window index), and the mean ΔNLL x 1e-3 without that window:

- WikiText-2: BF16 -0.21 (w7; without it -151.06); NVFP4 +0.06 (w69; without it +13.95); IF4 +0.06 (w139; without it +1.48); MixFP4 (Zou) -0.06 (w70; without it -3.97); GPTQ‡ +2.73 (w88; without it -5.30); FOCUS -0.09 (w72; without it -43.35); FlipQuant 8x64 -0.09 (w38; without it -57.72); FlipQuant 16x64 -0.10 (w7; without it -57.63); FlipQuant 256x64 -0.10 (w7; without it -43.49).
- C4: BF16 -0.35 (w201; without it -146.09); NVFP4 +0.08 (w243; without it +14.45); IF4 -0.09 (w150; without it -17.35); MixFP4 (Zou) -0.09 (w150; without it -18.53); GPTQ‡ +0.17 (w82; without it -20.98); FOCUS -0.15 (w201; without it -75.42); FlipQuant 8x64 -0.18 (w120; without it -85.76); FlipQuant 16x64 -0.19 (w150; without it -85.46); FlipQuant 256x64 -0.15 (w120; without it -66.69).

## Llama-3.1-8B-Instruct

| unit | wall (s) | trainer total (s) | device peak (MiB) | GPU allocated peak (GiB) | E0M3 share | fit extension |
|---|---:|---:|---:|---:|---:|---|
| 8x64 | 289 | 277 | 44,652 | 41.9 | 83,771 / 13,631,488 (0.615 %) | all checks pass |
| 16x64 | 283 | 271 | 44,468 | 41.7 | 52,719 / 6,815,744 (0.773 %) | all checks pass |
| 256x64 | 284 | 272 | 44,352 | 41.6 | 9,447 / 425,984 (2.218 %) | all checks pass |

Calibration windows (256 x 512): the 128 of the paper rule (`builder_seed0` with this model's tokenizer; not taken from an archived record) and the 128 of the extension rule (seed 20260930). Equal to base Llama-3.1-8B's release windows, token for token: fit True, extension True.

| row | WikiText-2 | C4 | ΔNLL vs FourOverSix (wiki / C4) | FlipQuant vs NVFP4 (wiki / C4) |
|---|---:|---:|---:|---:|
| BF16 | 7.2171 | 10.3853 | -80.05 ± 3.13* / -81.46 ± 15.97* |  |
| NVFP4 | 7.8876 | 11.3409 | +8.78 ± 1.93* / +6.56 ± 2.36* |  |
| FourOverSix | 7.8186 | 11.2667 |  |  |
| IF4 | 7.7390 | 11.1793 | -10.23 ± 1.90* / -7.79 ± 3.26* |  |
| MixFP4 (Zou) | 7.7316 | 11.1592 | -11.19 ± 2.11* / -9.58 ± 3.33* |  |
| GPTQ‡ | 7.8292 | 11.3664 | +1.36 ± 5.15 / +8.81 ± 19.19 |  |
| FOCUS | 7.7113 | 11.1618 | -13.82 ± 1.93* / -9.35 ± 2.88* |  |
| FlipQuant 8x64 | 7.6951 | 11.1384 | -15.93 ± 1.68* / -11.45 ± 2.62* | -24.71 ± 2.16* / -18.02 ± 2.92* |
| FlipQuant 16x64 | 7.7106 | 11.1420 | -13.90 ± 1.77* / -11.13 ± 2.90* | -22.69 ± 2.16* / -17.69 ± 3.07* |
| FlipQuant 256x64 | 7.7501 | 11.2171 | -8.80 ± 1.68* / -4.41 ± 1.56* | -17.58 ± 2.10* / -10.98 ± 2.46* |

Largest single-window NLL difference vs FourOverSix (nats; window index), and the mean ΔNLL x 1e-3 without that window:

- WikiText-2: BF16 -0.14 (w102; without it -79.63); NVFP4 +0.04 (w71; without it +8.59); IF4 -0.04 (w92; without it -10.03); MixFP4 (Zou) -0.05 (w92; without it -10.94); GPTQ‡ +0.24 (w140; without it -0.32); FOCUS -0.05 (w77; without it -13.57); FlipQuant 8x64 -0.04 (w30; without it -15.77); FlipQuant 16x64 -0.04 (w77; without it -13.73); FlipQuant 256x64 -0.03 (w136; without it -8.62).
- C4: BF16 -1.12 (w62; without it -77.38); NVFP4 +0.14 (w48; without it +6.03); IF4 -0.23 (w62; without it -6.93); MixFP4 (Zou) -0.26 (w190; without it -8.59); GPTQ‡ +2.32 (w219; without it -0.23); FOCUS -0.22 (w190; without it -8.53); FlipQuant 8x64 -0.20 (w62; without it -10.72); FlipQuant 16x64 -0.24 (w190; without it -10.23); FlipQuant 256x64 -0.06 (w62; without it -4.18).

## Mean capped NLL recovery

Over the 16 common pairs (the models with FOCUS; the coordinator's definition, per pair 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - NLL_BF16)), no floor, equal weight). Flagged pairs (NLL_NVFP4 - NLL_BF16 <= 0): none.

| row | mean capped NLL recovery |
|---|---:|
| BF16 | 100.0 % (16 pairs) |
| NVFP4 | 0.0 % (16 pairs) |
| FourOverSix | 9.5 % (16 pairs) |
| IF4 | 20.0 % (16 pairs) |
| MixFP4 (Zou) | 18.9 % (16 pairs) |
| GPTQ‡ | 5.0 % (16 pairs) |
| FOCUS | 42.1 % (16 pairs) |
| FlipQuant 8x64 | 47.9 % (16 pairs) |
| FlipQuant 16x64 | 47.0 % (16 pairs) |
| FlipQuant 256x64 | 41.1 % (16 pairs) |

Per-pair values: `results/paper_eval/final/FINAL.md`. Updated table: `results/paper_eval/final/table_main_ppl_final.tex`.
