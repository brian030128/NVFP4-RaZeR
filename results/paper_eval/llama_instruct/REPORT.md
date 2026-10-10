# Part P: Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct in tab:main-ppl

PROTOCOL.md part P. Every row with the existing main-table settings, n16k64, native sm120 (build_V, `--kernel-set auto`) except IF4 / MixFP4 (Zou), simulated as part I. The final table is `results/paper_eval/final/` (tables_final.py). ΔNLL x 1e-3 ± 2 SE, * beyond 2 SE.

## Llama-3.2-1B-Instruct

Not run (the HF license of meta-llama/Llama-3.2-1B-Instruct is not accepted for the edgeai-lab account; the model stays stopped until it is).

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

Largest single-window NLL difference vs FourOverSix (nats; window index), and the mean ΔNLL x 1e-3 without that window, C4: BF16 -1.12 (w62; without it -77.38); NVFP4 +0.14 (w48; without it +6.03); IF4 -0.23 (w62; without it -6.93); MixFP4 (Zou) -0.26 (w190; without it -8.59); GPTQ‡ +2.32 (w219; without it -0.23); FOCUS -0.22 (w190; without it -8.53); FlipQuant 8x64 -0.20 (w62; without it -10.72); FlipQuant 16x64 -0.24 (w190; without it -10.23); FlipQuant 256x64 -0.06 (w62; without it -4.18).

## Mean capped NLL recovery

Over the 12 common pairs (the models with FOCUS; the coordinator's definition, per pair 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - NLL_BF16)), no floor, equal weight). Flagged pairs (NLL_NVFP4 - NLL_BF16 <= 0): none.

| row | mean capped NLL recovery |
|---|---:|
| BF16 | 100.0 % (12 pairs) |
| NVFP4 | 0.0 % (12 pairs) |
| FourOverSix | 6.1 % (12 pairs) |
| IF4 | 19.2 % (12 pairs) |
| MixFP4 (Zou) | 17.5 % (12 pairs) |
| GPTQ‡ | 2.5 % (12 pairs) |
| FOCUS | 42.2 % (12 pairs) |
| FlipQuant 8x64 | 46.2 % (12 pairs) |
| FlipQuant 16x64 | 45.6 % (12 pairs) |
| FlipQuant 256x64 | 40.5 % (12 pairs) |

Per-pair values: `results/paper_eval/final/FINAL.md`. Updated table: `results/paper_eval/final/table_main_ppl_final.tex`.
