# Calibration size: FlipQuant 16x64 and FOCUS at 1,024 and 2,048 windows (part AA)

PROTOCOL.md part AA and its amendment 1. FlipQuant 16x64 (release settings: 5 epochs, top-1000 teacher, seed 0, last-step map) and FOCUS (part K's FOCUS_COMMON: 1 epoch, global batch 32, micro 8) trained on 1,024 or 2,048 windows of 512 tokens instead of the release's 256. The sets are nested, 256 ⊂ 1,024 ⊂ 2,048, and audited against every evaluation text (WINDOWS.md). FlipQuant and FOCUS read the same windows (token sha256 checked). flipquant calib-size f912a30. Evaluation exactly as parts X (zero-shot) and Y (GSM8K), plus PPL (paper convention). Step 0: STEP0.md. 2,048-window memory probes: PROBES.md.

## Findings

Pooled exact McNemar over the eight models (discordant counts summed; * = p < 0.05). The per-model tables below have every pair.

- **FlipQuant 16x64 gains GSM8K from more windows, and the gain saturates at 1,024.** vs 256: GSM8K +1.60* (p=2.2e-06) at 1,024 and +1.30* (p=0.00016) at 2,048; 2,048 vs 1,024 -0.30 (p=0.38). Zero-shot: +0.09 (p=0.14) / +0.14* (p=0.015).
- **FOCUS gains too.** vs 256 (seven models; Qwen3.8-27B's 256 reference is in another environment): GSM8K +1.43* (p=0.00034) at 1,024 and +2.14* (p=5.4e-08) at 2,048; zero-shot +0.16* (p=0.014) / +0.09 (p=0.17).
- **FlipQuant vs FOCUS at the same size.**
  - GSM8K: +0.90* (p=0.013) at 256, +1.90* (p=2.1e-07) at 1,024 (+1.09* (p=0.0063) without Qwen3.8-27B's early-stopping FOCUS-1k), -0.07 (p=0.87) at 2,048.
  - Zero-shot: -0.06 (p=0.36) / -0.16* (p=0.009) / +0.01 (p=0.93).
  - At 2,048 the two are level on both.
- **vs IF4 / MixFP4 (Zou), simulated:** zero-shot +0.57* (p=2.3e-18) / +0.63* (p=7.2e-22) at 2,048 (+0.43* (p=5.9e-11) / +0.49* (p=1.6e-13) at 256); GSM8K +0.58 (p=0.16) / +0.64 (p=0.12) (six models).
- **The motivating losses.** Each model's own table has the pairs.
  - Phi-4 GSM8K vs MixFP4: −1.29* at 256 is gone at 1,024 / 2,048.
  - Qwen3.8-27B zero-shot vs IF4: −0.45* at 256 is not significant at 1,024 / 2,048.
  - Nemotron's zero-shot gap to FOCUS closes at 2,048.
  - Phi-4's zero-shot gap stays against FOCUS-256 (−0.42* at 2,048). Against FOCUS-2k it is −0.11 (n.s.), but only because FOCUS-2k itself fell (−0.30* vs FOCUS-256).
- **Caveats.**
  - Each FOCUS cell is one run, and FOCUS is not reproducible run to run (step 0 (d); amendment 3).
  - Qwen3.8-27B's FOCUS-1k stops early on GSM8K; its 256 control and its 2,048 run do not, so this is a run-specific outcome, not a window-count trend.
  - FlipQuant is deterministic, but a different window draw was not tried.
  - Qwen3-1.7B trades WikiText-2 / C4 NLL for GSM8K with more math / code windows, FlipQuant more than FOCUS.
  - Nemotron's FQ16-2k strict-match loss is format only (flexible-extract unchanged).
  - E0M3 shares grow with the steps (2,048: 6.8–17.2 %, Llama-1B 27.0 %; amendment 4).
- **GPU time of the part:** FlipQuant calibration 20.5 h, zero-shot 4.7 h, GSM8K 11.7 h, PPL 0.9 h, FOCUS calibration 21.2 h, probes 0.5 h, offload test 0.0 h, theta rerun 0.1 h; 59.6 h in all.

## Summary

Per model: zero-shot Avg. (%), GSM8K strict-match (%) and WikiText-2 PPL of each method and calibration size (TBD: not run yet; —: not in the scope).

**Zero-shot Avg. (%):**

| method | Llama-1B | Phi-4 | Mistral | Llama-8B | Nemotron | Qwen27B | Qwen3-1.7B | Qwen3-8B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix | 56.24 | 72.39 | 75.43 | 71.74 | 70.89 | 73.70 | 56.13 | 68.93 |
| FOCUS, 256 | 57.40 | 73.14 | 75.16 | 72.33 | 71.61 | 74.30 | 58.66 | 70.37 |
| FOCUS, 1,024 | 57.90 | 73.05 | 75.37 | 72.89 | 71.66 | 74.72 | 58.25 | 70.88 |
| FOCUS, 2,048 | 57.89 | 73.03 | 75.63 | 72.27 | 71.50 | 74.32 | 58.35 | 70.95 |
| FlipQuant 16x64, 256 | 57.25 | 72.58 | 75.77 | 72.58 | 70.74 | 73.91 | 58.08 | 69.73 |
| FlipQuant 16x64, 1,024 | 57.93 | 72.78 | 75.69 | 72.28 | 71.09 | 74.41 | 58.15 | 70.18 |
| FlipQuant 16x64, 2,048 | 57.72 | 72.69 | 75.82 | 72.73 | 71.38 | 74.55 | 58.16 | 70.30 |

**GSM8K strict-match (%):**

| method | Llama-1B | Phi-4 | Mistral | Llama-8B | Nemotron | Qwen27B | Qwen3-1.7B | Qwen3-8B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix | 29.72 | 91.74 | 44.50 | 79.45 | 88.93 | 94.62 | 61.87 | 88.25 |
| FOCUS, 256 | 32.15 | 92.72 | 43.37 | 77.86 | 89.16 | 93.56 | 61.26 | 86.43 |
| FOCUS, 1,024 | 32.22 | 91.51 | 46.40 | 79.91 | 88.86 | 88.40 | 66.34 | 87.72 |
| FOCUS, 2,048 | 34.04 | 92.27 | 44.66 | 80.82 | 90.45 | 96.66 | 66.57 | 89.16 |
| FlipQuant 16x64, 256 | 31.61 | 92.12 | 44.81 | 80.06 | 89.31 | 95.15 | 61.33 | 89.31 |
| FlipQuant 16x64, 1,024 | 35.18 | 93.18 | 45.79 | 80.36 | 88.63 | 95.91 | 67.25 | 90.22 |
| FlipQuant 16x64, 2,048 | 34.27 | 92.87 | 46.55 | 81.58 | 87.04 | 95.68 | 66.41 | 89.69 |

**WikiText-2 PPL:**

| method | Llama-1B | Phi-4 | Mistral | Llama-8B | Nemotron | Qwen27B | Qwen3-1.7B | Qwen3-8B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FourOverSix | 15.315 | 6.662 | 5.708 | 7.819 | 8.426 | 7.302 | 19.475 | 10.007 |
| FOCUS, 256 | 14.661 | 6.628 | 5.678 | 7.711 | 8.368 | 7.275 | 15.089 | 9.569 |
| FOCUS, 1,024 | 14.687 | 6.641 | 5.663 | 7.676 | 8.344 | 7.190 | 14.968 | 9.523 |
| FOCUS, 2,048 | 14.583 | 6.620 | 5.666 | 7.693 | 8.336 | 7.064 | 15.297 | 9.502 |
| FlipQuant 16x64, 256 | 14.454 | 6.633 | 5.653 | 7.711 | 8.353 | 7.186 | 15.398 | 9.590 |
| FlipQuant 16x64, 1,024 | 14.387 | 6.618 | 5.636 | 7.652 | 8.327 | 7.078 | 15.762 | 9.584 |
| FlipQuant 16x64, 2,048 | 14.390 | 6.605 | 5.628 | 7.630 | 8.321 | 7.070 | 16.024 | 9.620 |

**Pooled over the models** (discordant counts summed; difference in points over all pooled questions; exact McNemar p; * = p < 0.05):

| comparison | zero-shot | models | GSM8K | models |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 — all models | -0.06 (6098/6201, p=0.36) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.90* (769/674, p=0.013) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-256 vs FOCUS-256 — without Qwen27B | -0.07 (5540/5650, p=0.3) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B | +0.80* (728/654, p=0.05) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs FourOverSix | +0.82* (6422/4983, p=2e-41) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +2.18* (768/538, p=2.1e-10) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs FQ16-256 | +0.09 (5163/5014, p=0.14) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.60* (716/547, p=2.2e-06) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FOCUS-1k vs FourOverSix | +0.99* (7422/5697, p=2.4e-51) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.28 (809/779, p=0.47) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FOCUS-1k vs FOCUS-256 | +0.16* (5170/4921, p=0.014) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B | +1.43* (735/603, p=0.00034) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs FOCUS-1k — all models | -0.16* (5817/6103, p=0.009) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.90* (838/638, p=2.1e-07) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs FOCUS-1k — without Qwen27B | -0.16* (5299/5545, p=0.019) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B | +1.09* (722/621, p=0.0063) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs IF4† | +0.51* (7007/6107, p=4e-15) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.76 (538/478, p=0.064) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |
| FQ16-1k vs MixFP4 (Zou)† | +0.57* (7038/6039, p=2.5e-18) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.82* (548/483, p=0.046) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |
| FQ16-2k vs FourOverSix | +0.88* (6680/5140, p=1.3e-45) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.88* (780/582, p=8.9e-08) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs FQ16-256 | +0.14* (5381/5131, p=0.015) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.30* (716/579, p=0.00016) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FOCUS-2k vs FourOverSix | +0.87* (7404/5875, p=3.3e-40) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.94* (836/631, p=9.6e-08) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FOCUS-2k vs FOCUS-256 | +0.09 (5401/5257, p=0.17) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B | +2.14* (758/560, p=5.4e-08) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs FOCUS-2k — all models | +0.01 (6008/5997, p=0.93) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | -0.07 (679/686, p=0.87) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs FOCUS-2k — without Qwen27B | -0.04 (5443/5511, p=0.52) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B | +0.06 (665/659, p=0.89) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs IF4† | +0.57* (7049/6048, p=2.3e-18) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.58 (545/499, p=0.16) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |
| FQ16-2k vs MixFP4 (Zou)† | +0.63* (7096/5996, p=7.2e-22) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.64 (544/493, p=0.12) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |
| FQ16-2k vs FQ16-1k | +0.06 (5143/5042, p=0.32) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | -0.30 (618/650, p=0.38) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FOCUS-2k vs FOCUS-1k | -0.11 (5400/5596, p=0.063) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +1.66* (809/634, p=4.5e-06) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs FOCUS-256 | +0.03 (6193/6147, p=0.69) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +2.50* (839/575, p=2.3e-12) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs FOCUS-256 | +0.08 (6324/6177, p=0.19) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +2.20* (831/599, p=9.3e-10) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-256 vs GPTQ‡ | +0.80* (7559/6154, p=3.6e-33) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | -0.69 (688/761, p=0.059) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-1k vs GPTQ‡ | +0.89* (7606/6052, p=2.2e-40) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.91* (760/664, p=0.012) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-2k vs GPTQ‡ | +0.95* (7694/6039, p=2.4e-45) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | +0.61 (753/689, p=0.097) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B |
| FQ16-256 vs IF4† | +0.43* (6943/6192, p=5.9e-11) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | -0.24 (497/516, p=0.57) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |
| FQ16-256 vs MixFP4 (Zou)† | +0.49* (7050/6200, p=1.6e-13) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B, Qwen3-1.7B, Qwen3-8B | -0.18 (522/536, p=0.69) | Llama-1B, Phi-4, Mistral, Llama-8B, Nemotron, Qwen27B |

"All models" includes Qwen3.8-27B, whose FOCUS-1k stops early on GSM8K (amendment 3); "without Qwen27B" pools the other seven (the coordinator's request). Qwen3.8-27B's environment-matched comparisons against its local control are in its section.

## Llama-3.2-1B-Instruct

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 | 59.92 | 45.41 ± 2.74 | 13.163 | 18.519 |
| NVFP4 | 71.65 | 57.72 | 59.91 | 59.97 | 36.26 | 53.10 | 56.43 | 29.19 ± 2.50 | 15.535 | 21.767 |
| FourOverSix | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 | 56.24 | 29.72 ± 2.52 | 15.315 | 21.449 |
| IF4† | 71.93 | 58.31 | 60.77 | 60.52 | 36.18 | 54.86 | 57.09 | 33.51 ± 2.60 § | 15.345 | 21.074 |
| MixFP4 (Zou)† | 74.21 | 58.50 | 60.30 | 60.35 | 35.84 | 54.82 | 57.34 | 33.59 ± 2.60 § | 15.248 | 21.049 |
| GPTQ‡ | 71.06 | 57.62 | 59.91 | 59.89 | 36.26 | 54.49 | 56.54 | 30.78 ± 2.54 | 15.533 | 21.019 |
| FlipQuant 8x64 | 72.52 | 59.59 | 59.27 | 62.25 | 37.80 | 56.10 | 57.92 | 30.63 ± 2.54 | 14.453 | 19.679 |
| FlipQuant 256x64 | 73.07 | 59.01 | 61.01 | 60.82 | 36.35 | 56.08 | 57.72 | 31.08 ± 2.55 | 14.657 | 20.059 |
| FOCUS, 256 | 72.58 | 59.57 | 59.12 | 60.86 | 37.03 | 55.23 | 57.40 | 32.15 ± 2.57 | 14.661 | 19.885 |
| FOCUS, 1,024 | 72.74 | 59.27 | 59.67 | 61.91 | 36.86 | 56.92 | 57.90 | 32.22 ± 2.57 | 14.687 | 19.817 |
| FOCUS, 2,048 | 73.23 | 59.41 | 59.19 | 61.45 | 38.23 | 55.85 | 57.89 | 34.04 ± 2.61 | 14.583 | 19.848 |
| FlipQuant 16x64, 256 | 72.85 | 59.31 | 58.09 | 61.15 | 36.35 | 55.75 | 57.25 | 31.61 ± 2.56 | 14.454 | 19.684 |
| FlipQuant 16x64, 1,024 | 73.39 | 59.36 | 60.46 | 62.08 | 36.26 | 56.05 | 57.93 | 35.18 ± 2.63 | 14.387 | 19.605 |
| FlipQuant 16x64, 2,048 | 72.69 | 59.41 | 60.06 | 61.66 | 36.18 | 56.32 | 57.72 | 34.27 ± 2.61 | 14.390 | 19.610 |
| FOCUS, 256 (retrained, step 0 d) | 72.14 | 59.05 | 59.75 | 61.91 | 37.71 | 55.66 | 57.70 | 34.04 ± 2.61 | 14.685 | 19.881 |

§ GSM8K run with the chat template's date pinned to 15 Oct 2026 (part Y amendment 3), the date the native rows saw; this part's rows of this model are pinned too. All prompts are equal (equal sample digests).

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
| FQ16-2k vs FourOverSix | +1.77* (1367/980, p=1.4e-15) | +4.55* (189/129, p=0.00091) | -0.0623 ± 0.0031* | -0.0896 ± 0.0037* |
| FQ16-2k vs FQ16-256 | +0.32 (1053/982, p=0.12) | +2.65* (159/124, p=0.043) | -0.0044 ± 0.0022* | -0.0038 ± 0.0016* |
| FOCUS-2k vs FourOverSix | +1.74* (1441/1060, p=2.7e-14) | +4.32* (184/127, p=0.0015) | -0.0490 ± 0.0031* | -0.0776 ± 0.0038* |
| FOCUS-2k vs FOCUS-256 | +0.26 (1132/1075, p=0.23) | +1.90 (146/121, p=0.14) | -0.0053 ± 0.0025* | -0.0019 ± 0.0019 |
| FQ16-2k vs FOCUS-2k | +0.03 (1175/1169, p=0.92) | +0.23 (145/142, p=0.91) | -0.0133 ± 0.0027* | -0.0120 ± 0.0020* |
| FQ16-2k vs IF4† | +1.00* (1337/1119, p=1.2e-05) | +0.76 (164/154, p=0.61) | -0.0642 ± 0.0035* | -0.0720 ± 0.0036* |
| FQ16-2k vs MixFP4 (Zou)† | +0.79* (1312/1140, p=0.00055) | +0.68 (150/141, p=0.64) | -0.0579 ± 0.0034* | -0.0708 ± 0.0035* |
| FQ16-2k vs FQ16-1k | -0.05 (1003/1013, p=0.84) | -0.91 (143/155, p=0.52) | +0.0002 ± 0.0022 | +0.0003 ± 0.0015 |
| FOCUS-2k vs FOCUS-1k | -0.15 (1035/1068, p=0.49) | +1.82 (148/124, p=0.16) | -0.0071 ± 0.0025* | +0.0016 ± 0.0020 |
| FQ16-1k vs FOCUS-256 | +0.33 (1192/1119, p=0.13) | +3.03* (173/133, p=0.026) | -0.0189 ± 0.0024* | -0.0142 ± 0.0018* |
| FQ16-2k vs FOCUS-256 | +0.29 (1213/1150, p=0.2) | +2.12 (154/126, p=0.11) | -0.0187 ± 0.0025* | -0.0139 ± 0.0019* |
| FQ16-256 vs GPTQ‡ | +1.26* (1414/1138, p=5.1e-08) | +0.83 (151/140, p=0.56) | -0.0720 ± 0.0387* | -0.0656 ± 0.0045* |
| FQ16-1k vs GPTQ‡ | +1.63* (1462/1105, p=2e-12) | +4.40* (186/128, p=0.0013) | -0.0766 ± 0.0387* | -0.0696 ± 0.0046* |
| FQ16-2k vs GPTQ‡ | +1.59* (1454/1107, p=7.5e-12) | +3.49* (174/128, p=0.0095) | -0.0764 ± 0.0386* | -0.0694 ± 0.0045* |
| FQ16-256 vs IF4† | +0.67* (1272/1125, p=0.0029) | -1.90 (128/153, p=0.15) | -0.0598 ± 0.0035* | -0.0682 ± 0.0038* |
| FQ16-256 vs MixFP4 (Zou)† | +0.46* (1292/1191, p=0.045) | -1.97 (130/156, p=0.14) | -0.0535 ± 0.0034* | -0.0670 ± 0.0037* |
| FOCUS-256 retrained vs FOCUS-256 | +0.01 (967/964, p=0.96) | +1.90 (145/120, p=0.14) | +0.0016 ± 0.0021 | -0.0002 ± 0.0014 |
| FOCUS-1k vs FOCUS-256 retrained | +0.40 (1067/980, p=0.057) | -1.82 (131/155, p=0.17) | +0.0002 ± 0.0025 | -0.0032 ± 0.0019* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 25 | 0 | 97.6 | 1024 | 7 |
| NVFP4 | 57 | 0 | 112.0 | 1024 | 24 |
| FourOverSix | 115 | 0 | 96.2 | 1024 | 7 |
| IF4† | 81 | 0 | 102.7 | 1024 | 14 |
| MixFP4 (Zou)† | 44 | 0 | 104.0 | 1024 | 14 |
| GPTQ‡ | 37 | 0 | 89.5 | 1024 | 4 |
| FlipQuant 8x64 | 75 | 0 | 91.4 | 1024 | 7 |
| FlipQuant 256x64 | 63 | 0 | 92.7 | 1024 | 7 |
| FOCUS, 256 | 49 | 0 | 99.6 | 1024 | 17 |
| FOCUS, 1,024 | 86 | 0 | 99.6 | 1024 | 15 |
| FOCUS, 2,048 | 67 | 0 | 90.5 | 1024 | 6 |
| FlipQuant 16x64, 256 | 73 | 0 | 88.2 | 1024 | 5 |
| FlipQuant 16x64, 1,024 | 34 | 0 | 94.0 | 1024 | 5 |
| FlipQuant 16x64, 2,048 | 53 | 0 | 99.5 | 1024 | 14 |
| FOCUS, 256 (retrained, step 0 d) | 48 | 0 | 97.3 | 1024 | 12 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.54 (88/78, p=0.48) | +0.05 (299/294, p=0.87) | +2.37 (164/134, p=0.093) | +0.93 (131/109, p=0.18) | -0.09 (60/61, p=1) | +0.29 (304/289, p=0.57) |
| FOCUS-1k vs FOCUS-256 | +0.16 (89/86, p=0.88) | -0.30 (299/329, p=0.25) | +0.55 (147/140, p=0.72) | +1.05 (124/99, p=0.11) | -0.17 (63/65, p=0.93) | +1.69* (363/276, p=0.00066) |
| FQ16-1k vs FOCUS-1k | +0.65 (97/85, p=0.41) | +0.09 (366/357, p=0.77) | +0.79 (157/147, p=0.61) | +0.17 (134/130, p=0.85) | -0.60 (61/68, p=0.6) | -0.87 (320/365, p=0.093) |
| FQ16-2k vs FQ16-256 | -0.16 (79/82, p=0.87) | +0.10 (304/294, p=0.71) | +1.97 (155/130, p=0.16) | +0.51 (137/125, p=0.5) | -0.17 (62/64, p=0.93) | +0.56 (316/287, p=0.25) |
| FOCUS-2k vs FOCUS-256 | +0.65 (91/79, p=0.4) | -0.16 (328/344, p=0.56) | +0.08 (154/153, p=1) | +0.59 (130/116, p=0.41) | +1.19 (78/64, p=0.28) | +0.62 (351/319, p=0.23) |
| FQ16-2k vs FOCUS-2k | -0.54 (81/91, p=0.49) | +0.00 (356/356, p=1) | +0.87 (156/145, p=0.56) | +0.21 (152/147, p=0.82) | -2.05 (74/98, p=0.079) | +0.47 (356/332, p=0.38) |
| FQ16-256 vs FOCUS-256 | +0.27 (95/90, p=0.77) | -0.26 (328/354, p=0.34) | -1.03 (156/169, p=0.51) | +0.29 (136/129, p=0.71) | -0.68 (66/74, p=0.55) | +0.52 (352/325, p=0.32) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 950,272 | KL 0.1539; E0M3 0; flips 0 | KL 0.1431; E0M3 3,576; flips 3,576 | KL 0.1153; E0M3 9,032; flips 8,736 | KL 0.1037; E0M3 16,480; flips 11,720 | KL 0.0964; E0M3 26,656; flips 16,202 | 2.81 % |
| 1,024 | 950,272 | KL 0.1320; E0M3 10,557; flips 17,555 | KL 0.1019; E0M3 36,803; flips 67,462 | KL 0.0918; E0M3 71,470; flips 129,839 | KL 0.0852; E0M3 106,747; flips 188,903 | KL 0.0810; E0M3 141,062; flips 240,011 | 14.84 % |
| 2,048 | 950,272 | KL 0.1183; E0M3 33,627; flips 84,427 | KL 0.0937; E0M3 95,225; flips 310,902 | KL 0.0861; E0M3 158,150; flips 527,185 | KL 0.0821; E0M3 211,676; flips 696,756 | KL 0.0805; E0M3 256,459; flips 827,389 | 26.99 % |

**Theta at the end of the 2,048-window run** (a rerun with `--save-theta` whose map equals the evaluated one: yes):
- E0M3 tiles with theta < 0.02 (within one lr step of the threshold): 7.9% of 256,459;
- E2M1 tiles with theta in (-0.02, 0]: 2.9%; all tiles with |theta| < 0.02: 4.2% of 950,272;
- E0M3 theta quantiles (10/25/50/75/90 %): 0.026, 0.072, 0.184, 0.404, 0.730.

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | part P | 160 | 0.0281 | 83 | 39 | 14.40 | 15842 | 3.27 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.1484 | 225 | 156 | 16.60 | 18112 | 3.25 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.2699 | 414 | 316 | 19.52 | 21094 | 3.25 | 6,283,354,112 |
| FOCUS | 256 | part P | 8 | — | — | 35 | 13.57 | — | — | 729,889,075 |
| FOCUS | 1,024 | this part | 32 | — | 173 | 139 | 13.57 | 15650 | — | 729,889,075 |
| FOCUS | 2,048 | this part | 64 | — | 322 | 278 | 13.57 | 15650 | — | 729,889,075 |

Evaluation wall times (s): gsm8k_focus256_rerun 340, gsm8k_focus_1024 386, gsm8k_focus_2048 291, gsm8k_fq16_1024 292, gsm8k_fq16_2048 470, gsm8k_fq16_256 285, ppl_fo6 19, ppl_focus256 18, ppl_focus256_rerun 19, ppl_focus_1024 19, ppl_focus_2048 18, ppl_fq16_1024 19, ppl_fq16_2048 19, ppl_fq16_256 19, zs_focus256_rerun 186, zs_focus_1024 186, zs_focus_2048 184, zs_fq16_1024 188, zs_fq16_2048 183, zs_fq16_256 183.

## Phi-4

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

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
| FOCUS, 2,048 | 81.18 | 80.95 | 76.32 | 73.06 | 54.78 | 71.86 | 73.03 | 92.27 ± 1.47 | 6.620 | 10.519 |
| FlipQuant 16x64, 256 | 81.01 | 81.35 | 74.66 | 72.01 | 54.69 | 71.76 | 72.58 | 92.12 ± 1.48 | 6.633 | 10.516 |
| FlipQuant 16x64, 1,024 | 81.39 | 81.18 | 74.43 | 72.69 | 55.20 | 71.76 | 72.78 | 93.18 ± 1.39 | 6.618 | 10.503 |
| FlipQuant 16x64, 2,048 | 80.79 | 81.24 | 75.93 | 71.68 | 54.86 | 71.67 | 72.69 | 92.87 ± 1.42 | 6.605 | 10.496 |

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
| FQ16-2k vs FourOverSix | +0.17 (494/457, p=0.24) | +1.14* (24/9, p=0.014) | -0.0086 ± 0.0017* | -0.0046 ± 0.0010* |
| FQ16-2k vs FQ16-256 | -0.05 (435/445, p=0.76) | +0.76 (26/16, p=0.16) | -0.0042 ± 0.0015* | -0.0019 ± 0.0008* |
| FOCUS-2k vs FourOverSix | +0.28 (595/533, p=0.069) | +0.53 (28/21, p=0.39) | -0.0063 ± 0.0018* | -0.0024 ± 0.0012* |
| FOCUS-2k vs FOCUS-256 | -0.30* (461/527, p=0.039) | -0.45 (26/32, p=0.51) | -0.0012 ± 0.0012 | -0.0004 ± 0.0009 |
| FQ16-2k vs FOCUS-2k | -0.11 (532/557, p=0.47) | +0.61 (28/20, p=0.31) | -0.0023 ± 0.0016* | -0.0022 ± 0.0010* |
| FQ16-2k vs IF4† | +0.01 (554/552, p=0.98) | +0.23 (25/22, p=0.77) | -0.0045 ± 0.0019* | -0.0014 ± 0.0011* |
| FQ16-2k vs MixFP4 (Zou)† | -0.01 (567/570, p=0.95) | -0.53 (19/26, p=0.37) | -0.0052 ± 0.0020* | -0.0010 ± 0.0010* |
| FQ16-2k vs FQ16-1k | -0.09 (432/451, p=0.54) | -0.30 (15/19, p=0.61) | -0.0020 ± 0.0014* | -0.0007 ± 0.0007 |
| FOCUS-2k vs FOCUS-1k | -0.22 (463/512, p=0.12) | +0.76 (35/25, p=0.25) | -0.0031 ± 0.0013* | -0.0016 ± 0.0009* |
| FQ16-1k vs FOCUS-256 | -0.33* (479/551, p=0.027) | +0.45 (20/14, p=0.39) | -0.0015 ± 0.0015 | -0.0019 ± 0.0009* |
| FQ16-2k vs FOCUS-256 | -0.42* (480/571, p=0.0055) | +0.15 (25/23, p=0.89) | -0.0035 ± 0.0014* | -0.0026 ± 0.0009* |
| FQ16-256 vs GPTQ‡ | +0.09 (609/590, p=0.6) | -0.08 (27/28, p=1) | -0.0084 ± 0.0020* | -0.0036 ± 0.0011* |
| FQ16-1k vs GPTQ‡ | +0.13 (604/576, p=0.43) | +0.99 (32/19, p=0.092) | -0.0106 ± 0.0022* | -0.0048 ± 0.0011* |
| FQ16-2k vs GPTQ‡ | +0.04 (588/579, p=0.81) | +0.68 (27/18, p=0.23) | -0.0126 ± 0.0020* | -0.0055 ± 0.0011* |
| FQ16-256 vs IF4† | +0.05 (577/565, p=0.74) | -0.53 (21/28, p=0.39) | -0.0003 ± 0.0017 | +0.0005 ± 0.0011 |
| FQ16-256 vs MixFP4 (Zou)† | +0.03 (575/568, p=0.86) | -1.29* (18/35, p=0.027) | -0.0010 ± 0.0018 | +0.0009 ± 0.0010 |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 3 | 0 | 124.6 | 493 | 0 |
| NVFP4 | 3 | 0 | 131.6 | 459 | 0 |
| FourOverSix | 4 | 0 | 138.3 | 508 | 0 |
| IF4† | 9 | 0 | 135.9 | 548 | 0 |
| MixFP4 (Zou)† | 7 | 0 | 133.8 | 530 | 0 |
| GPTQ‡ | 11 | 0 | 119.9 | 425 | 0 |
| FlipQuant 8x64 | 1 | 0 | 135.1 | 459 | 0 |
| FlipQuant 256x64 | 2 | 0 | 132.5 | 481 | 0 |
| FOCUS, 256 | 4 | 0 | 129.7 | 1024 | 1 |
| FOCUS, 1,024 | 10 | 0 | 113.7 | 1024 | 1 |
| FOCUS, 2,048 | 12 | 0 | 124.3 | 448 | 0 |
| FlipQuant 16x64, 256 | 1 | 0 | 129.6 | 449 | 0 |
| FlipQuant 16x64, 1,024 | 1 | 0 | 133.9 | 530 | 0 |
| FlipQuant 16x64, 2,048 | 0 | 0 | 134.1 | 507 | 0 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.38 (34/27, p=0.44) | -0.17 (106/123, p=0.29) | -0.24 (79/82, p=0.87) | +0.67 (55/39, p=0.12) | +0.51 (29/23, p=0.49) | +0.00 (143/143, p=1) |
| FOCUS-1k vs FOCUS-256 | +0.76 (37/23, p=0.092) | -0.16 (123/139, p=0.35) | -0.55 (73/80, p=0.63) | +0.38 (59/50, p=0.44) | -0.85 (31/41, p=0.29) | -0.14 (138/145, p=0.72) |
| FQ16-1k vs FOCUS-1k | +0.11 (34/32, p=0.9) | -0.14 (135/149, p=0.44) | -1.34 (82/99, p=0.23) | -0.51 (49/61, p=0.29) | +0.68 (37/29, p=0.39) | -0.43 (166/188, p=0.26) |
| FQ16-2k vs FQ16-256 | -0.22 (27/31, p=0.69) | -0.11 (107/118, p=0.51) | +1.26 (90/74, p=0.24) | -0.34 (40/48, p=0.46) | +0.17 (31/29, p=0.9) | -0.10 (140/145, p=0.81) |
| FOCUS-2k vs FOCUS-256 | +0.65 (40/28, p=0.18) | -0.53* (121/174, p=0.0024) | +0.00 (83/83, p=1) | +0.25 (51/45, p=0.61) | -0.60 (24/31, p=0.42) | -0.47 (142/166, p=0.19) |
| FQ16-2k vs FOCUS-2k | -0.38 (32/39, p=0.48) | +0.29 (171/142, p=0.11) | -0.39 (90/95, p=0.77) | -1.39* (39/72, p=0.0022) | +0.09 (31/30, p=1) | -0.19 (169/179, p=0.63) |
| FQ16-256 vs FOCUS-256 | +0.49 (40/31, p=0.34) | -0.13 (146/159, p=0.49) | -1.66 (80/101, p=0.14) | -0.80 (42/61, p=0.076) | -0.68 (28/36, p=0.38) | -0.56 (153/182, p=0.13) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 13,312,000 | KL 0.0357; E0M3 0; flips 0 | KL 0.0358; E0M3 142; flips 142 | KL 0.0352; E0M3 5,261; flips 5,299 | KL 0.0323; E0M3 33,031; flips 30,546 | KL 0.0293; E0M3 79,591; flips 58,550 | 0.60 % |
| 1,024 | 13,312,000 | KL 0.0363; E0M3 3,587; flips 5,239 | KL 0.0337; E0M3 80,171; flips 122,744 | KL 0.0295; E0M3 229,344; flips 388,787 | KL 0.0268; E0M3 368,694; flips 583,900 | KL 0.0252; E0M3 491,665; flips 717,973 | 3.69 % |
| 2,048 | 13,312,000 | KL 0.0357; E0M3 39,024; flips 72,868 | KL 0.0311; E0M3 327,216; flips 878,292 | KL 0.0278; E0M3 607,362; flips 1,682,104 | KL 0.0257; E0M3 853,836; flips 2,151,576 | KL 0.0244; E0M3 1,085,106; flips 2,554,710 | 8.15 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0060 | 502 | 376 | 59.91 | 65242 | 28.23 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0369 | 1688 | 1500 | 62.11 | 67634 | 28.26 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.0815 | 3268 | 2999 | 65.04 | 70574 | 28.26 | 6,283,354,112 |
| FOCUS | 256 | part K | 8 | — | — | 413 | 91.04 | — | — | 10,223,745,627 |
| FOCUS | 1,024 | this part | 32 | — | 1702 | 1650 | 91.04 | 96148 | — | 10,223,745,627 |
| FOCUS | 2,048 | this part | 64 | — | 3338 | 3298 | 91.04 | 96148 | — | 10,223,745,627 |

Evaluation wall times (s): gsm8k_focus_1024 1176, gsm8k_focus_2048 1154, gsm8k_fq16_1024 1182, gsm8k_fq16_2048 1211, ppl_fo6 69, ppl_focus256 69, ppl_focus_1024 68, ppl_focus_2048 69, ppl_fq16_1024 73, ppl_fq16_2048 74, ppl_fq16_256 73, zs_focus_1024 360, zs_focus_2048 359, zs_fq16_1024 369, zs_fq16_2048 367, zs_fq16_256 367.

## Mistral-7B-Instruct-v0.3

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 83.51 | 83.32 | 74.59 | 81.73 | 59.90 | 73.53 | 76.10 | 47.01 ± 2.75 | 5.496 | 8.135 |
| NVFP4 | 82.86 | 82.73 | 74.03 | 81.78 | 59.13 | 73.24 | 75.63 | 44.50 ± 2.74 | 5.730 | 8.416 |
| FourOverSix | 82.75 | 82.86 | 74.82 | 80.64 | 58.36 | 73.12 | 75.43 | 44.50 ± 2.74 | 5.708 | 8.399 |
| IF4† | 83.19 | 82.90 | 75.37 | 80.98 | 58.11 | 72.77 | 75.55 | 45.19 ± 2.74 | 5.670 | 8.362 |
| MixFP4 (Zou)† | 83.24 | 83.05 | 75.30 | 81.44 | 58.53 | 73.04 | 75.77 | 45.56 ± 2.74 | 5.663 | 8.369 |
| GPTQ‡ | 83.46 | 82.40 | 75.14 | 81.86 | 59.22 | 72.91 | 75.83 | 45.41 ± 2.74 | 5.735 | 8.415 |
| FlipQuant 8x64 | 82.37 | 82.71 | 75.45 | 80.77 | 58.19 | 73.61 | 75.52 | 45.11 ± 2.74 | 5.650 | 8.319 |
| FlipQuant 256x64 | 83.24 | 82.71 | 73.32 | 81.27 | 58.11 | 73.76 | 75.40 | 44.73 ± 2.74 | 5.662 | 8.340 |
| FOCUS, 256 | 82.70 | 82.39 | 74.90 | 81.82 | 57.34 | 71.78 | 75.16 | 43.37 ± 2.73 | 5.678 | 8.341 |
| FOCUS, 1,024 | 82.75 | 82.75 | 73.88 | 81.78 | 58.28 | 72.79 | 75.37 | 46.40 ± 2.75 | 5.663 | 8.328 |
| FOCUS, 2,048 | 83.19 | 82.87 | 75.30 | 81.27 | 57.51 | 73.65 | 75.63 | 44.66 ± 2.74 | 5.666 | 8.319 |
| FlipQuant 16x64, 256 | 82.86 | 82.94 | 74.66 | 80.64 | 59.56 | 73.96 | 75.77 | 44.81 ± 2.74 | 5.653 | 8.324 |
| FlipQuant 16x64, 1,024 | 83.24 | 82.82 | 74.98 | 80.85 | 58.45 | 73.80 | 75.69 | 45.79 ± 2.74 | 5.636 | 8.304 |
| FlipQuant 16x64, 2,048 | 82.70 | 82.76 | 75.69 | 81.57 | 59.04 | 73.14 | 75.82 | 46.55 ± 2.75 | 5.628 | 8.293 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | +0.76* (775/610, p=1e-05) | +1.44 (164/145, p=0.31) | -0.0044 ± 0.0015* | -0.0021 ± 0.0012* |
| FQ16-1k vs FourOverSix | +0.22 (491/443, p=0.12) | +1.29 (147/130, p=0.34) | -0.0128 ± 0.0012* | -0.0114 ± 0.0011* |
| FQ16-1k vs FQ16-256 | -0.08 (421/438, p=0.59) | +0.99 (148/135, p=0.48) | -0.0031 ± 0.0011* | -0.0024 ± 0.0009* |
| FOCUS-1k vs FourOverSix | -0.06 (539/553, p=0.69) | +1.90 (163/138, p=0.17) | -0.0080 ± 0.0014* | -0.0085 ± 0.0012* |
| FOCUS-1k vs FOCUS-256 | +0.39* (614/528, p=0.012) | +3.03* (169/129, p=0.024) | -0.0027 ± 0.0012* | -0.0015 ± 0.0012* |
| FQ16-1k vs FOCUS-1k | +0.28 (584/522, p=0.067) | -0.61 (158/166, p=0.7) | -0.0048 ± 0.0012* | -0.0029 ± 0.0011* |
| FQ16-1k vs IF4† | +0.19 (577/535, p=0.22) | +0.61 (155/147, p=0.69) | -0.0061 ± 0.0014* | -0.0070 ± 0.0011* |
| FQ16-1k vs MixFP4 (Zou)† | -0.01 (547/550, p=0.95) | +0.23 (167/164, p=0.91) | -0.0048 ± 0.0015* | -0.0078 ± 0.0011* |
| FQ16-2k vs FourOverSix | +0.14 (507/476, p=0.34) | +2.05 (154/127, p=0.12) | -0.0142 ± 0.0012* | -0.0127 ± 0.0013* |
| FQ16-2k vs FQ16-256 | -0.16 (454/488, p=0.28) | +1.74 (148/125, p=0.18) | -0.0045 ± 0.0010* | -0.0037 ± 0.0009* |
| FOCUS-2k vs FourOverSix | +0.22 (587/540, p=0.17) | +0.15 (156/154, p=0.95) | -0.0074 ± 0.0014* | -0.0095 ± 0.0014* |
| FOCUS-2k vs FOCUS-256 | +0.67* (730/583, p=5.5e-05) | +1.29 (158/141, p=0.35) | -0.0021 ± 0.0015* | -0.0026 ± 0.0011* |
| FQ16-2k vs FOCUS-2k | -0.07 (528/544, p=0.65) | +1.90 (159/134, p=0.16) | -0.0068 ± 0.0013* | -0.0032 ± 0.0010* |
| FQ16-2k vs IF4† | +0.11 (553/528, p=0.47) | +1.36 (164/146, p=0.33) | -0.0075 ± 0.0013* | -0.0083 ± 0.0012* |
| FQ16-2k vs MixFP4 (Zou)† | -0.09 (535/555, p=0.56) | +0.99 (160/147, p=0.49) | -0.0062 ± 0.0014* | -0.0092 ± 0.0013* |
| FQ16-2k vs FQ16-1k | -0.08 (456/473, p=0.6) | +0.76 (144/134, p=0.59) | -0.0014 ± 0.0010* | -0.0013 ± 0.0007* |
| FOCUS-2k vs FOCUS-1k | +0.28 (531/470, p=0.058) | -1.74 (137/160, p=0.2) | +0.0006 ± 0.0011 | -0.0010 ± 0.0010 |
| FQ16-1k vs FOCUS-256 | +0.68* (764/616, p=7.5e-05) | +2.43 (162/130, p=0.069) | -0.0076 ± 0.0015* | -0.0045 ± 0.0012* |
| FQ16-2k vs FOCUS-256 | +0.60* (748/617, p=0.00043) | +3.18* (169/127, p=0.017) | -0.0089 ± 0.0015* | -0.0058 ± 0.0012* |
| FQ16-256 vs GPTQ‡ | +0.30 (645/579, p=0.063) | -0.61 (146/154, p=0.69) | -0.0144 ± 0.0015* | -0.0109 ± 0.0014* |
| FQ16-1k vs GPTQ‡ | +0.22 (626/577, p=0.17) | +0.38 (155/150, p=0.82) | -0.0175 ± 0.0016* | -0.0133 ± 0.0017* |
| FQ16-2k vs GPTQ‡ | +0.15 (634/602, p=0.38) | +1.14 (162/147, p=0.43) | -0.0189 ± 0.0015* | -0.0147 ± 0.0016* |
| FQ16-256 vs IF4† | +0.27 (568/509, p=0.077) | -0.38 (151/156, p=0.82) | -0.0030 ± 0.0014* | -0.0046 ± 0.0013* |
| FQ16-256 vs MixFP4 (Zou)† | +0.06 (557/543, p=0.7) | -0.76 (158/168, p=0.62) | -0.0017 ± 0.0014* | -0.0054 ± 0.0013* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 160 | 0 | 124.5 | 1024 | 1 |
| NVFP4 | 150 | 0 | 128.3 | 1024 | 1 |
| FourOverSix | 132 | 0 | 126.1 | 1024 | 2 |
| IF4† | 172 | 0 | 136.8 | 1024 | 3 |
| MixFP4 (Zou)† | 135 | 0 | 134.0 | 1024 | 2 |
| GPTQ‡ | 142 | 0 | 125.6 | 1024 | 2 |
| FlipQuant 8x64 | 136 | 0 | 123.8 | 1024 | 3 |
| FlipQuant 256x64 | 129 | 0 | 129.7 | 1024 | 6 |
| FOCUS, 256 | 140 | 0 | 122.2 | 1024 | 3 |
| FOCUS, 1,024 | 126 | 0 | 120.7 | 1024 | 1 |
| FOCUS, 2,048 | 118 | 0 | 121.7 | 1024 | 1 |
| FlipQuant 16x64, 256 | 136 | 0 | 125.7 | 1024 | 3 |
| FlipQuant 16x64, 1,024 | 145 | 0 | 125.6 | 1024 | 2 |
| FlipQuant 16x64, 2,048 | 108 | 0 | 124.3 | 1024 | 5 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.38 (33/26, p=0.43) | -0.12 (105/117, p=0.46) | +0.32 (56/52, p=0.77) | +0.21 (55/50, p=0.7) | -1.11 (31/44, p=0.17) | -0.16 (141/149, p=0.68) |
| FOCUS-1k vs FOCUS-256 | +0.05 (45/44, p=1) | +0.36 (195/159, p=0.063) | -1.03 (57/70, p=0.29) | -0.04 (61/62, p=1) | +0.94 (53/42, p=0.3) | +1.01* (203/151, p=0.0066) |
| FQ16-1k vs FOCUS-1k | +0.49 (54/45, p=0.42) | +0.07 (155/148, p=0.73) | +1.10 (72/58, p=0.25) | -0.93 (47/69, p=0.051) | +0.17 (51/49, p=0.92) | +1.01* (205/153, p=0.0069) |
| FQ16-2k vs FQ16-256 | -0.16 (33/36, p=0.81) | -0.18 (118/136, p=0.29) | +1.03 (62/49, p=0.25) | +0.93* (61/39, p=0.035) | -0.51 (44/50, p=0.61) | -0.82* (136/178, p=0.021) |
| FOCUS-2k vs FOCUS-256 | +0.49 (54/45, p=0.42) | +0.48* (213/165, p=0.016) | +0.39 (71/66, p=0.73) | -0.55 (64/77, p=0.31) | +0.17 (48/46, p=0.92) | +1.86* (280/184, p=9.6e-06) |
| FQ16-2k vs FOCUS-2k | -0.49 (36/45, p=0.37) | -0.11 (147/158, p=0.57) | +0.39 (68/63, p=0.73) | +0.29 (55/48, p=0.55) | +1.54 (53/35, p=0.069) | -0.50 (169/195, p=0.19) |
| FQ16-256 vs FOCUS-256 | +0.16 (56/53, p=0.85) | +0.55* (230/175, p=0.0072) | -0.24 (64/67, p=0.86) | -1.18* (64/92, p=0.03) | +2.22* (63/37, p=0.012) | +2.17* (298/186, p=4e-07) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 6,815,744 | KL 0.0577; E0M3 0; flips 0 | KL 0.0560; E0M3 2,551; flips 2,551 | KL 0.0468; E0M3 12,925; flips 10,716 | KL 0.0408; E0M3 37,613; flips 27,518 | KL 0.0343; E0M3 75,906; flips 47,991 | 1.11 % |
| 1,024 | 6,815,744 | KL 0.0525; E0M3 14,979; flips 17,057 | KL 0.0381; E0M3 72,722; flips 107,571 | KL 0.0321; E0M3 176,890; flips 267,062 | KL 0.0285; E0M3 272,103; flips 384,053 | KL 0.0264; E0M3 359,840; flips 481,003 | 5.28 % |
| 2,048 | 6,815,744 | KL 0.0467; E0M3 44,714; flips 89,334 | KL 0.0337; E0M3 228,577; flips 564,003 | KL 0.0291; E0M3 409,125; flips 1,020,424 | KL 0.0269; E0M3 584,002; flips 1,348,499 | KL 0.0252; E0M3 727,133; flips 1,534,123 | 10.67 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0111 | 270 | 191 | 36.62 | 39126 | 14.42 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0528 | 876 | 760 | 38.81 | 41356 | 14.45 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.1067 | 1686 | 1523 | 41.74 | 44438 | 14.45 | 6,283,354,112 |
| FOCUS | 256 | part K | 8 | — | — | 230 | 48.68 | — | — | 5,234,657,055 |
| FOCUS | 1,024 | this part | 32 | — | 964 | 918 | 48.68 | 51460 | — | 5,234,657,055 |
| FOCUS | 2,048 | this part | 64 | — | 1888 | 1836 | 48.68 | 51460 | — | 5,234,657,055 |

Evaluation wall times (s): gsm8k_focus_1024 860, gsm8k_focus_2048 885, gsm8k_fq16_1024 1089, gsm8k_fq16_2048 1278, ppl_fo6 38, ppl_focus256 40, ppl_focus_1024 39, ppl_focus_2048 40, ppl_fq16_1024 41, ppl_fq16_2048 41, ppl_fq16_256 41, zs_focus_1024 262, zs_focus_2048 263, zs_fq16_1024 268, zs_fq16_2048 270, zs_fq16_256 268.

## Llama-3.1-8B-Instruct

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 81.34 | 79.58 | 73.80 | 79.71 | 55.72 | 72.13 | 73.71 | 85.90 ± 1.92 | 7.217 | 10.385 |
| NVFP4 | 80.09 | 77.71 | 72.06 | 77.69 | 52.30 | 71.22 | 71.85 | 79.98 ± 2.20 | 7.888 | 11.341 |
| FourOverSix | 80.14 | 78.04 | 71.74 | 77.02 | 52.05 | 71.45 | 71.74 | 79.45 ± 2.22 | 7.819 | 11.267 |
| IF4† | 79.65 | 78.79 | 72.77 | 77.36 | 53.67 | 70.85 | 72.18 | 80.44 ± 2.18 | 7.739 | 11.179 |
| MixFP4 (Zou)† | 79.11 | 78.61 | 72.85 | 77.86 | 53.24 | 70.52 | 72.03 | 80.67 ± 2.17 | 7.732 | 11.159 |
| GPTQ‡ | 79.71 | 77.81 | 72.06 | 77.78 | 52.73 | 71.03 | 71.85 | 81.80 ± 2.12 | 7.829 | 11.366 |
| FlipQuant 8x64 | 80.41 | 78.60 | 73.24 | 78.83 | 52.73 | 71.69 | 72.58 | 81.43 ± 2.14 | 7.695 | 11.138 |
| FlipQuant 256x64 | 80.20 | 78.65 | 71.98 | 78.20 | 54.27 | 70.44 | 72.29 | 81.73 ± 2.13 | 7.750 | 11.217 |
| FOCUS, 256 | 80.41 | 78.47 | 71.74 | 77.53 | 54.01 | 71.84 | 72.33 | 77.86 ± 2.29 | 7.711 | 11.162 |
| FOCUS, 1,024 | 80.58 | 78.61 | 72.30 | 78.87 | 54.52 | 72.48 | 72.89 | 79.91 ± 2.21 | 7.676 | 11.083 |
| FOCUS, 2,048 | 79.87 | 78.10 | 73.56 | 76.85 | 53.07 | 72.17 | 72.27 | 80.82 ± 2.17 | 7.693 | 11.063 |
| FlipQuant 16x64, 256 | 79.60 | 78.56 | 73.64 | 78.96 | 53.24 | 71.49 | 72.58 | 80.06 ± 2.20 | 7.711 | 11.142 |
| FlipQuant 16x64, 1,024 | 79.82 | 78.52 | 73.48 | 78.28 | 52.56 | 71.03 | 72.28 | 80.36 ± 2.19 | 7.652 | 11.040 |
| FlipQuant 16x64, 2,048 | 80.36 | 78.72 | 72.38 | 79.17 | 54.27 | 71.47 | 72.73 | 81.58 ± 2.13 | 7.630 | 11.024 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | +0.11 (705/680, p=0.52) | +2.20 (119/90, p=0.053) | -0.0001 ± 0.0018 | -0.0018 ± 0.0019 |
| FQ16-1k vs FourOverSix | +0.36* (679/601, p=0.031) | +0.91 (100/88, p=0.42) | -0.0216 ± 0.0019* | -0.0203 ± 0.0038* |
| FQ16-1k vs FQ16-256 | -0.23 (565/615, p=0.15) | +0.30 (84/80, p=0.81) | -0.0077 ± 0.0016* | -0.0092 ± 0.0022* |
| FOCUS-1k vs FourOverSix | +0.91* (846/648, p=3.3e-07) | +0.45 (114/108, p=0.74) | -0.0184 ± 0.0020* | -0.0164 ± 0.0048* |
| FOCUS-1k vs FOCUS-256 | +0.43* (663/568, p=0.0074) | +2.05 (118/91, p=0.072) | -0.0046 ± 0.0017* | -0.0071 ± 0.0027* |
| FQ16-1k vs FOCUS-1k | -0.55* (648/768, p=0.0016) | +0.45 (94/88, p=0.71) | -0.0032 ± 0.0018* | -0.0039 ± 0.0021* |
| FQ16-1k vs IF4† | +0.01 (714/711, p=0.96) | -0.08 (100/101, p=1) | -0.0113 ± 0.0022* | -0.0125 ± 0.0020* |
| FQ16-1k vs MixFP4 (Zou)† | +0.18 (741/701, p=0.3) | -0.30 (100/104, p=0.83) | -0.0104 ± 0.0023* | -0.0107 ± 0.0023* |
| FQ16-2k vs FourOverSix | +0.72* (756/598, p=1.9e-05) | +2.12* (106/78, p=0.046) | -0.0245 ± 0.0018* | -0.0218 ± 0.0050* |
| FQ16-2k vs FQ16-256 | +0.14 (602/572, p=0.4) | +1.52 (94/74, p=0.14) | -0.0106 ± 0.0016* | -0.0106 ± 0.0029* |
| FOCUS-2k vs FourOverSix | +0.32 (773/704, p=0.077) | +1.36 (122/104, p=0.26) | -0.0162 ± 0.0020* | -0.0183 ± 0.0052* |
| FOCUS-2k vs FOCUS-256 | -0.16 (600/634, p=0.35) | +2.96* (125/86, p=0.0087) | -0.0024 ± 0.0018* | -0.0089 ± 0.0032* |
| FQ16-2k vs FOCUS-2k | +0.41* (727/638, p=0.017) | +0.76 (103/93, p=0.52) | -0.0083 ± 0.0017* | -0.0035 ± 0.0019* |
| FQ16-2k vs IF4† | +0.38* (763/680, p=0.031) | +1.14 (104/89, p=0.31) | -0.0142 ± 0.0021* | -0.0140 ± 0.0030* |
| FQ16-2k vs MixFP4 (Zou)† | +0.55* (792/672, p=0.0019) | +0.91 (104/92, p=0.43) | -0.0133 ± 0.0023* | -0.0122 ± 0.0029* |
| FQ16-2k vs FQ16-1k | +0.37* (593/513, p=0.017) | +1.21 (92/76, p=0.25) | -0.0029 ± 0.0014* | -0.0014 ± 0.0019 |
| FOCUS-2k vs FOCUS-1k | -0.59* (565/694, p=0.00031) | +0.91 (107/95, p=0.44) | +0.0022 ± 0.0016* | -0.0018 ± 0.0013* |
| FQ16-1k vs FOCUS-256 | -0.11 (663/688, p=0.51) | +2.50* (112/79, p=0.02) | -0.0078 ± 0.0020* | -0.0110 ± 0.0024* |
| FQ16-2k vs FOCUS-256 | +0.25 (721/666, p=0.15) | +3.71* (118/69, p=0.00042) | -0.0107 ± 0.0020* | -0.0124 ± 0.0031* |
| FQ16-256 vs GPTQ‡ | +0.69* (853/702, p=0.00014) | -1.74 (87/110, p=0.12) | -0.0153 ± 0.0048* | -0.0199 ± 0.0189* |
| FQ16-1k vs GPTQ‡ | +0.46* (804/703, p=0.01) | -1.44 (86/105, p=0.19) | -0.0229 ± 0.0051* | -0.0291 ± 0.0189* |
| FQ16-2k vs GPTQ‡ | +0.83* (879/698, p=5.7e-06) | -0.23 (89/92, p=0.88) | -0.0258 ± 0.0049* | -0.0306 ± 0.0191* |
| FQ16-256 vs IF4† | +0.24 (746/693, p=0.17) | -0.38 (106/111, p=0.79) | -0.0037 ± 0.0021* | -0.0033 ± 0.0024* |
| FQ16-256 vs MixFP4 (Zou)† | +0.41* (790/700, p=0.021) | -0.61 (99/107, p=0.63) | -0.0027 ± 0.0023* | -0.0015 ± 0.0023 |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 28 | 0 | 127.7 | 1024 | 10 |
| NVFP4 | 40 | 0 | 124.7 | 1024 | 11 |
| FourOverSix | 19 | 0 | 117.6 | 1024 | 5 |
| IF4† | 34 | 0 | 129.9 | 1024 | 12 |
| MixFP4 (Zou)† | 36 | 0 | 126.8 | 1024 | 14 |
| GPTQ‡ | 28 | 0 | 125.1 | 1024 | 7 |
| FlipQuant 8x64 | 26 | 0 | 117.2 | 1024 | 7 |
| FlipQuant 256x64 | 25 | 0 | 116.8 | 1024 | 2 |
| FOCUS, 256 | 49 | 0 | 126.9 | 1024 | 16 |
| FOCUS, 1,024 | 42 | 0 | 117.8 | 1024 | 9 |
| FOCUS, 2,048 | 36 | 0 | 116.5 | 1024 | 10 |
| FlipQuant 16x64, 256 | 24 | 0 | 118.0 | 1024 | 9 |
| FlipQuant 16x64, 1,024 | 23 | 0 | 119.9 | 1024 | 10 |
| FlipQuant 16x64, 2,048 | 16 | 0 | 117.6 | 1024 | 4 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.22 (48/44, p=0.75) | -0.04 (166/170, p=0.87) | -0.16 (67/69, p=0.93) | -0.67 (68/84, p=0.22) | -0.68 (44/52, p=0.48) | -0.47 (172/196, p=0.23) |
| FOCUS-1k vs FOCUS-256 | +0.16 (40/37, p=0.82) | +0.14 (196/182, p=0.5) | +0.55 (81/74, p=0.63) | +1.35* (97/65, p=0.015) | +0.51 (55/49, p=0.62) | +0.64 (194/161, p=0.089) |
| FQ16-1k vs FOCUS-1k | -0.76 (41/55, p=0.18) | -0.09 (202/211, p=0.69) | +1.18 (101/86, p=0.31) | -0.59 (77/91, p=0.32) | -1.96* (45/68, p=0.038) | -1.46* (182/257, p=0.0004) |
| FQ16-2k vs FQ16-256 | +0.76 (54/40, p=0.18) | +0.16 (164/148, p=0.4) | -1.26 (70/86, p=0.23) | +0.21 (83/78, p=0.75) | +1.02 (57/45, p=0.28) | -0.02 (174/175, p=1) |
| FOCUS-2k vs FOCUS-256 | -0.54 (39/49, p=0.34) | -0.37 (164/201, p=0.059) | +1.82 (84/61, p=0.067) | -0.67 (78/94, p=0.25) | -0.94 (50/61, p=0.34) | +0.33 (185/168, p=0.39) |
| FQ16-2k vs FOCUS-2k | +0.49 (52/43, p=0.41) | +0.62* (223/161, p=0.0018) | -1.18 (77/92, p=0.28) | +2.31* (118/63, p=5.3e-05) | +1.19 (61/47, p=0.21) | -0.70 (196/232, p=0.091) |
| FQ16-256 vs FOCUS-256 | -0.82 (41/56, p=0.15) | +0.09 (204/195, p=0.69) | +1.89 (99/75, p=0.081) | +1.43* (111/77, p=0.016) | -0.77 (58/67, p=0.47) | -0.35 (192/210, p=0.4) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 6,815,744 | KL 0.1024; E0M3 0; flips 0 | KL 0.0972; E0M3 1,154; flips 1,194 | KL 0.0824; E0M3 8,279; flips 7,673 | KL 0.0754; E0M3 26,524; flips 21,213 | KL 0.0682; E0M3 52,719; flips 35,265 | 0.77 % |
| 1,024 | 6,815,744 | KL 0.0929; E0M3 11,553; flips 14,573 | KL 0.0754; E0M3 62,803; flips 96,872 | KL 0.0647; E0M3 168,470; flips 254,775 | KL 0.0563; E0M3 279,675; flips 374,617 | KL 0.0511; E0M3 377,393; flips 445,836 | 5.54 % |
| 2,048 | 6,815,744 | KL 0.0889; E0M3 43,200; flips 87,500 | KL 0.0716; E0M3 215,293; flips 532,521 | KL 0.0599; E0M3 423,036; flips 977,689 | KL 0.0532; E0M3 606,055; flips 1,257,315 | KL 0.0493; E0M3 766,726; flips 1,466,403 | 11.25 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | part P | 160 | 0.0077 | 283 | 201 | 41.71 | 44468 | 15.92 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0554 | 922 | 800 | 43.91 | 46598 | 15.91 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.1125 | 1778 | 1602 | 46.83 | 49760 | 15.90 | 6,283,354,112 |
| FOCUS | 256 | part P | 8 | — | — | 231 | 53.04 | — | — | 5,234,657,055 |
| FOCUS | 1,024 | this part | 32 | — | 972 | 924 | 53.04 | 56036 | — | 5,234,657,055 |
| FOCUS | 2,048 | this part | 64 | — | 1904 | 1847 | 53.04 | 56036 | — | 5,234,657,055 |

Evaluation wall times (s): gsm8k_focus_1024 1393, gsm8k_focus_2048 1266, gsm8k_fq16_1024 1558, gsm8k_fq16_2048 1172, ppl_fo6 41, ppl_focus256 42, ppl_focus_1024 42, ppl_focus_2048 42, ppl_fq16_1024 44, ppl_fq16_2048 44, ppl_fq16_256 43, zs_focus_1024 279, zs_focus_2048 280, zs_fq16_1024 273, zs_fq16_2048 276, zs_fq16_256 277.

## Nemotron-Nano-9B-v2

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 80.85 | 77.49 | 73.01 | 79.84 | 56.14 | 66.78 | 72.35 | 87.41 ± 1.83 | 8.086 | 11.153 |
| NVFP4 | 80.52 | 76.46 | 72.93 | 79.29 | 55.20 | 64.51 | 71.49 | 87.34 ± 1.83 | 8.453 | 11.544 |
| FourOverSix | 79.98 | 76.55 | 71.51 | 79.04 | 53.92 | 64.31 | 70.89 | 88.93 ± 1.73 | 8.426 | 11.503 |
| IF4† | 79.87 | 76.57 | 72.45 | 78.62 | 54.27 | 63.98 | 70.96 | 87.11 ± 1.85 | 8.391 | 11.450 |
| MixFP4 (Zou)† | 80.41 | 76.94 | 72.77 | 78.83 | 53.58 | 63.71 | 71.04 | 85.52 ± 1.94 | 8.409 | 11.458 |
| GPTQ‡ | 79.16 | 76.15 | 72.30 | 79.00 | 53.16 | 65.42 | 70.86 | 87.19 ± 1.84 | 8.429 | 11.500 |
| FlipQuant 8x64 | 80.47 | 76.52 | 72.77 | 78.83 | 52.99 | 64.76 | 71.06 | 87.11 ± 1.85 | 8.354 | 11.444 |
| FlipQuant 256x64 | 80.36 | 76.70 | 72.53 | 78.20 | 53.41 | 63.28 | 70.75 | 88.93 ± 1.73 | 8.380 | 11.468 |
| FOCUS, 256 | 80.85 | 76.82 | 73.88 | 79.55 | 54.01 | 64.54 | 71.61 | 89.16 ± 1.71 | 8.368 | 11.473 |
| FOCUS, 1,024 | 79.98 | 76.76 | 72.93 | 79.59 | 54.86 | 65.83 | 71.66 | 88.86 ± 1.73 | 8.344 | 11.458 |
| FOCUS, 2,048 | 80.52 | 76.65 | 72.38 | 79.59 | 54.35 | 65.52 | 71.50 | 90.45 ± 1.62 | 8.336 | 11.438 |
| FlipQuant 16x64, 256 | 80.79 | 76.69 | 71.74 | 78.58 | 52.90 | 63.73 | 70.74 | 89.31 ± 1.70 | 8.353 | 11.445 |
| FlipQuant 16x64, 1,024 | 80.52 | 76.60 | 72.77 | 78.58 | 53.33 | 64.76 | 71.09 | 88.63 ± 1.75 | 8.327 | 11.440 |
| FlipQuant 16x64, 2,048 | 80.90 | 76.65 | 72.53 | 79.67 | 53.58 | 64.95 | 71.38 | 87.04 ± 1.85 | 8.321 | 11.420 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.54* (482/601, p=0.00033) | +0.15 (61/59, p=0.93) | -0.0018 ± 0.0015* | -0.0025 ± 0.0010* |
| FQ16-1k vs FourOverSix | +0.16 (522/486, p=0.27) | -0.30 (52/56, p=0.77) | -0.0119 ± 0.0013* | -0.0056 ± 0.0008* |
| FQ16-1k vs FQ16-256 | +0.26 (483/426, p=0.063) | -0.68 (46/55, p=0.43) | -0.0032 ± 0.0012* | -0.0005 ± 0.0008 |
| FOCUS-1k vs FourOverSix | +0.65* (679/538, p=5.9e-05) | -0.08 (59/60, p=1) | -0.0098 ± 0.0018* | -0.0039 ± 0.0011* |
| FOCUS-1k vs FOCUS-256 | +0.20 (541/498, p=0.19) | -0.30 (46/50, p=0.76) | -0.0029 ± 0.0015* | -0.0013 ± 0.0009* |
| FQ16-1k vs FOCUS-1k | -0.48* (471/576, p=0.0013) | -0.23 (52/55, p=0.85) | -0.0021 ± 0.0015* | -0.0016 ± 0.0010* |
| FQ16-1k vs IF4† | +0.22 (591/544, p=0.17) | +1.52 (68/48, p=0.077) | -0.0077 ± 0.0016* | -0.0009 ± 0.0010 |
| FQ16-1k vs MixFP4 (Zou)† | +0.06 (574/561, p=0.72) | +3.11* (92/51, p=0.00076) | -0.0098 ± 0.0014* | -0.0016 ± 0.0011* |
| FQ16-2k vs FourOverSix | +0.38* (569/485, p=0.011) | -1.90* (44/69, p=0.024) | -0.0126 ± 0.0012* | -0.0073 ± 0.0009* |
| FQ16-2k vs FQ16-256 | +0.48* (538/433, p=0.00084) | -2.27* (42/72, p=0.0064) | -0.0039 ± 0.0012* | -0.0022 ± 0.0008* |
| FOCUS-2k vs FourOverSix | +0.51* (662/551, p=0.0016) | +1.52 (58/38, p=0.052) | -0.0107 ± 0.0017* | -0.0057 ± 0.0011* |
| FOCUS-2k vs FOCUS-256 | +0.06 (553/540, p=0.72) | +1.29 (61/44, p=0.12) | -0.0039 ± 0.0013* | -0.0031 ± 0.0009* |
| FQ16-2k vs FOCUS-2k | -0.12 (526/553, p=0.43) | -3.41* (35/80, p=3.3e-05) | -0.0019 ± 0.0014* | -0.0016 ± 0.0010* |
| FQ16-2k vs IF4† | +0.43* (620/525, p=0.0054) | -0.08 (65/66, p=1) | -0.0084 ± 0.0014* | -0.0026 ± 0.0010* |
| FQ16-2k vs MixFP4 (Zou)† | +0.28 (610/549, p=0.078) | +1.52 (87/67, p=0.13) | -0.0105 ± 0.0014* | -0.0033 ± 0.0010* |
| FQ16-2k vs FQ16-1k | +0.22 (447/399, p=0.11) | -1.59 (46/67, p=0.059) | -0.0007 ± 0.0012 | -0.0017 ± 0.0008* |
| FOCUS-2k vs FOCUS-1k | -0.14 (479/509, p=0.36) | +1.59* (57/36, p=0.038) | -0.0009 ± 0.0014 | -0.0017 ± 0.0008* |
| FQ16-1k vs FOCUS-256 | -0.28 (519/581, p=0.066) | -0.53 (57/64, p=0.59) | -0.0050 ± 0.0014* | -0.0029 ± 0.0009* |
| FQ16-2k vs FOCUS-256 | -0.06 (552/566, p=0.7) | -2.12* (51/79, p=0.018) | -0.0057 ± 0.0014* | -0.0047 ± 0.0010* |
| FQ16-256 vs GPTQ‡ | -0.11 (590/613, p=0.53) | +2.12* (81/53, p=0.019) | -0.0091 ± 0.0017* | -0.0048 ± 0.0011* |
| FQ16-1k vs GPTQ‡ | +0.16 (610/576, p=0.34) | +1.44 (80/61, p=0.13) | -0.0123 ± 0.0018* | -0.0053 ± 0.0011* |
| FQ16-2k vs GPTQ‡ | +0.38* (666/584, p=0.022) | -0.15 (79/81, p=0.94) | -0.0130 ± 0.0017* | -0.0070 ± 0.0012* |
| FQ16-256 vs IF4† | -0.05 (555/565, p=0.79) | +2.20* (70/41, p=0.0076) | -0.0045 ± 0.0016* | -0.0004 ± 0.0010 |
| FQ16-256 vs MixFP4 (Zou)† | -0.20 (557/601, p=0.21) | +3.79* (93/43, p=2.2e-05) | -0.0066 ± 0.0015* | -0.0011 ± 0.0010* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 54 | 0 | 113.8 | 503 | 0 |
| NVFP4 | 35 | 0 | 111.8 | 608 | 0 |
| FourOverSix | 28 | 0 | 114.7 | 553 | 0 |
| IF4† | 24 | 0 | 105.9 | 481 | 0 |
| MixFP4 (Zou)† | 32 | 0 | 106.6 | 505 | 0 |
| GPTQ‡ | 60 | 0 | 116.8 | 443 | 0 |
| FlipQuant 8x64 | 54 | 0 | 118.4 | 582 | 0 |
| FlipQuant 256x64 | 29 | 0 | 115.9 | 573 | 0 |
| FOCUS, 256 | 26 | 0 | 109.7 | 507 | 0 |
| FOCUS, 1,024 | 30 | 0 | 114.5 | 1024 | 1 |
| FOCUS, 2,048 | 29 | 0 | 116.3 | 1024 | 1 |
| FlipQuant 16x64, 256 | 31 | 0 | 119.0 | 574 | 0 |
| FlipQuant 16x64, 1,024 | 45 | 0 | 116.6 | 525 | 0 |
| FlipQuant 16x64, 2,048 | 67 | 0 | 120.8 | 1024 | 1 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | -0.27 (22/27, p=0.57) | -0.09 (127/136, p=0.62) | +1.03 (70/57, p=0.29) | +0.00 (38/38, p=1) | +0.43 (33/28, p=0.61) | +1.03* (193/140, p=0.0043) |
| FOCUS-1k vs FOCUS-256 | -0.87* (18/34, p=0.036) | -0.06 (155/161, p=0.78) | -0.95 (62/74, p=0.35) | +0.04 (45/44, p=1) | +0.85 (36/26, p=0.25) | +1.28* (225/159, p=0.00089) |
| FQ16-1k vs FOCUS-1k | +0.54 (39/29, p=0.27) | -0.16 (148/164, p=0.4) | -0.16 (65/67, p=0.93) | -1.01* (32/56, p=0.014) | -1.54* (24/42, p=0.036) | -1.07* (163/218, p=0.0056) |
| FQ16-2k vs FQ16-256 | +0.11 (24/22, p=0.88) | -0.04 (146/150, p=0.86) | +0.79 (65/55, p=0.41) | +1.09* (61/35, p=0.01) | +0.68 (36/28, p=0.38) | +1.22* (206/143, p=0.00088) |
| FOCUS-2k vs FOCUS-256 | -0.33 (30/36, p=0.54) | -0.17 (155/172, p=0.38) | -1.50 (62/81, p=0.13) | +0.04 (47/46, p=1) | +0.34 (35/31, p=0.71) | +0.97* (224/174, p=0.014) |
| FQ16-2k vs FOCUS-2k | +0.38 (39/32, p=0.48) | +0.00 (157/157, p=1) | +0.16 (70/68, p=0.93) | +0.08 (45/43, p=0.92) | -0.77 (34/43, p=0.36) | -0.56 (181/210, p=0.16) |
| FQ16-256 vs FOCUS-256 | -0.05 (30/31, p=1) | -0.13 (160/173, p=0.51) | -2.13* (58/85, p=0.029) | -0.97* (33/56, p=0.019) | -1.11 (25/38, p=0.13) | -0.82* (176/218, p=0.039) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 7,530,880 | KL 0.0368; E0M3 0; flips 0 | KL 0.0357; E0M3 2,102; flips 2,150 | KL 0.0315; E0M3 7,934; flips 8,140 | KL 0.0286; E0M3 35,254; flips 30,386 | KL 0.0256; E0M3 80,098; flips 56,086 | 1.06 % |
| 1,024 | 7,530,880 | KL 0.0341; E0M3 7,037; flips 11,213 | KL 0.0286; E0M3 78,730; flips 121,355 | KL 0.0251; E0M3 210,605; flips 338,449 | KL 0.0229; E0M3 318,302; flips 486,979 | KL 0.0215; E0M3 420,667; flips 603,719 | 5.59 % |
| 2,048 | 7,530,880 | KL 0.0321; E0M3 38,771; flips 79,737 | KL 0.0264; E0M3 266,740; flips 689,105 | KL 0.0236; E0M3 476,900; flips 1,281,224 | KL 0.0220; E0M3 671,197; flips 1,690,949 | KL 0.0212; E0M3 845,369; flips 2,014,366 | 11.23 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0106 | 2621 | 2394 | 53.74 | 57232 | 17.48 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0559 | 10223 | 9537 | 55.94 | 59522 | 17.50 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.1123 | 20425 | 19128 | 58.87 | 62542 | 17.50 | 6,283,354,112 |
| FOCUS | 256 | part K | 8 | — | — | 295 | 53.94 | — | — | 5,783,806,155 |
| FOCUS | 1,024 | this part | 32 | — | 1097 | 1034 | 53.94 | 58592 | — | 5,783,806,155 |
| FOCUS | 2,048 | this part | 64 | — | 2117 | 2056 | 53.94 | 58592 | — | 5,783,806,155 |

Evaluation wall times (s): gsm8k_focus_1024 374, gsm8k_focus_2048 375, gsm8k_fq16_1024 358, gsm8k_fq16_2048 384, ppl_fo6 56, ppl_focus256 56, ppl_focus_1024 55, ppl_focus_2048 56, ppl_fq16_1024 58, ppl_fq16_2048 59, ppl_fq16_256 58, zs_focus_1024 353, zs_focus_2048 343, zs_fq16_1024 349, zs_fq16_2048 351, zs_fq16_256 348.

## Qwen3.8-27B

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 81.50 | 82.86 | 75.93 | 73.11 | 58.36 | 72.02 | 73.96 | 95.68 ± 1.12 ◊ | 7.052 | 9.893 |
| NVFP4 | 80.90 | 82.39 | 73.88 | 76.22 | 59.73 | 70.17 | 73.88 | 95.38 ± 1.16 | 7.574 | 10.223 |
| FourOverSix | 81.07 | 82.35 | 75.14 | 74.12 | 58.96 | 70.56 | 73.70 | 94.62 ± 1.24 | 7.302 | 10.190 |
| IF4† | 80.85 | 82.32 | 74.74 | 78.28 | 60.75 | 71.61 | 74.76 | 95.60 ± 1.13 ¤ | 7.334 | 10.166 |
| MixFP4 (Zou)† | 80.63 | 82.48 | 75.45 | 75.04 | 58.53 | 71.67 | 73.97 | 95.38 ± 1.16 ¤ | 7.348 | 10.164 |
| GPTQ‡ | 80.63 | 81.99 | 74.90 | 75.21 | 60.07 | 71.30 | 74.02 | 96.06 ± 1.07 | 7.528 | 10.158 |
| FlipQuant 8x64 | 81.28 | 82.29 | 76.16 | 75.00 | 58.87 | 70.75 | 74.06 | 95.53 ± 1.14 | 7.140 | 10.144 |
| FlipQuant 256x64 | 81.12 | 82.41 | 74.74 | 75.29 | 58.70 | 71.32 | 73.93 | 95.45 ± 1.15 | 7.220 | 10.161 |
| FOCUS, 256 | 81.56 | 82.19 | 75.37 | 76.68 | 59.56 | 70.42 | 74.30 | 93.56 ± 1.35 | 7.275 | 10.157 |
| FOCUS, 1,024 | 81.28 | 82.45 | 76.24 | 77.15 | 60.15 | 71.07 | 74.72 | 88.40 ± 1.76 | 7.190 | 10.136 |
| FOCUS, 2,048 | 81.12 | 82.14 | 75.06 | 76.73 | 60.49 | 70.39 | 74.32 | 96.66 ± 0.99 | 7.064 | 10.135 |
| FlipQuant 16x64, 256 | 81.77 | 82.50 | 74.66 | 75.08 | 58.36 | 71.07 | 73.91 | 95.15 ± 1.18 | 7.186 | 10.141 |
| FlipQuant 16x64, 1,024 | 81.72 | 82.22 | 75.85 | 75.97 | 59.30 | 71.41 | 74.41 | 95.91 ± 1.09 | 7.078 | 10.122 |
| FlipQuant 16x64, 2,048 | 81.61 | 82.35 | 74.66 | 77.23 | 60.15 | 71.26 | 74.55 | 95.68 ± 1.12 | 7.070 | 10.117 |
| FOCUS, 256 (local control, amendment 3) | 81.61 | 82.56 | 75.45 | 76.94 | 59.22 | 70.56 | 74.39 | 95.60 ± 1.13 | 7.233 | 10.150 |

◊ BF16 GSM8K ran at batch 48 (part Y), every other row at 64. Generations depend on the batch composition, so its paired comparisons would compare different batches; it enters none here.

¤ GSM8K run with expandable segments (part Y amendment 4; an allocator setting, which its pre-check showed changes no generation).

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | +0.03 (558/551, p=0.86) | +1.59* (41/20, p=0.0099) | -0.0123 ± 0.0044* | -0.0016 ± 0.0011* |
| FQ16-1k vs FourOverSix | +0.46* (523/423, p=0.0013) | +1.29* (32/15, p=0.019) | -0.0311 ± 0.0052* | -0.0067 ± 0.0011* |
| FQ16-1k vs FQ16-256 | +0.16 (445/409, p=0.23) | +0.76 (26/16, p=0.16) | -0.0151 ± 0.0039* | -0.0018 ± 0.0009* |
| FOCUS-1k vs FourOverSix | +0.64* (651/511, p=4.5e-05) | -6.22* (31/113, p=3.7e-12) | -0.0155 ± 0.0057* | -0.0053 ± 0.0012* |
| FOCUS-1k vs FOCUS-256 (cluster: a different environment) | +0.38* (527/444, p=0.0085) | -5.16* (38/106, p=1.3e-08) | -0.0117 ± 0.0044* | -0.0021 ± 0.0010* |
| FQ16-1k vs FOCUS-1k | -0.18 (518/558, p=0.23) | +7.51* (116/17, p=2.6e-19) | -0.0156 ± 0.0041* | -0.0013 ± 0.0010* |
| FQ16-1k vs IF4† | -0.28 (561/623, p=0.076) | +0.30 (23/19, p=0.64) | -0.0355 ± 0.0051* | -0.0043 ± 0.0011* |
| FQ16-1k vs MixFP4 (Zou)† | +0.08 (592/575, p=0.64) | +0.53 (23/16, p=0.34) | -0.0373 ± 0.0046* | -0.0041 ± 0.0011* |
| FQ16-2k vs FourOverSix | +0.59* (545/417, p=4.1e-05) | +1.06 (33/19, p=0.07) | -0.0323 ± 0.0063* | -0.0072 ± 0.0011* |
| FQ16-2k vs FQ16-256 | +0.29* (491/427, p=0.038) | +0.53 (23/16, p=0.34) | -0.0162 ± 0.0045* | -0.0024 ± 0.0009* |
| FOCUS-2k vs FourOverSix | +0.22 (588/539, p=0.15) | +2.05* (41/14, p=0.00036) | -0.0332 ± 0.0070* | -0.0054 ± 0.0011* |
| FOCUS-2k vs FOCUS-256 (cluster: a different environment) | -0.04 (494/502, p=0.82) | +3.11* (53/12, p=2.8e-07) | -0.0294 ± 0.0056* | -0.0022 ± 0.0009* |
| FQ16-2k vs FOCUS-2k | +0.36* (565/486, p=0.016) | -0.99 (14/27, p=0.06) | +0.0009 ± 0.0032 | -0.0018 ± 0.0011* |
| FQ16-2k vs IF4† | -0.16 (560/594, p=0.33) | +0.08 (23/22, p=1) | -0.0367 ± 0.0057* | -0.0048 ± 0.0012* |
| FQ16-2k vs MixFP4 (Zou)† | +0.21 (575/530, p=0.19) | +0.30 (24/20, p=0.65) | -0.0385 ± 0.0054* | -0.0046 ± 0.0012* |
| FQ16-2k vs FQ16-1k | +0.13 (469/441, p=0.37) | -0.23 (18/21, p=0.75) | -0.0011 ± 0.0029 | -0.0006 ± 0.0008 |
| FOCUS-2k vs FOCUS-1k | -0.42* (468/559, p=0.005) | +8.26* (120/11, p=2.6e-24) | -0.0177 ± 0.0040* | -0.0001 ± 0.0009 |
| FQ16-1k vs FOCUS-256 | +0.20 (559/516, p=0.2) | +2.35* (43/12, p=3.3e-05) | -0.0274 ± 0.0051* | -0.0034 ± 0.0011* |
| FQ16-2k vs FOCUS-256 | +0.32* (574/503, p=0.033) | +2.12* (44/16, p=0.00039) | -0.0285 ± 0.0056* | -0.0040 ± 0.0011* |
| FQ16-256 vs GPTQ‡ | +0.16 (582/547, p=0.31) | -0.91 (17/29, p=0.1) | -0.0465 ± 0.0079* | -0.0016 ± 0.0015* |
| FQ16-1k vs GPTQ‡ | +0.32* (617/546, p=0.04) | -0.15 (16/18, p=0.86) | -0.0616 ± 0.0092* | -0.0035 ± 0.0017* |
| FQ16-2k vs GPTQ‡ | +0.45* (612/513, p=0.0035) | -0.38 (20/25, p=0.55) | -0.0627 ± 0.0096* | -0.0040 ± 0.0017* |
| FQ16-256 vs IF4† | -0.45* (557/655, p=0.0053) | -0.45 (21/27, p=0.47) | -0.0204 ± 0.0039* | -0.0024 ± 0.0011* |
| FQ16-256 vs MixFP4 (Zou)† | -0.09 (539/558, p=0.59) | -0.23 (24/27, p=0.78) | -0.0222 ± 0.0041* | -0.0023 ± 0.0011* |
| FOCUS-256 local vs FOCUS-256 cluster | +0.22 (490/442, p=0.12) | +2.05* (45/18, p=0.0009) | -0.0058 ± 0.0036* | -0.0007 ± 0.0008 |
| FQ16-256 vs FOCUS-256 local (FOCUS in this environment) | -0.19 (520/561, p=0.22) | -0.45 (22/28, p=0.48) | -0.0065 ± 0.0042* | -0.0009 ± 0.0010 |
| FOCUS-1k vs FOCUS-256 local (same environment) | +0.16 (524/489, p=0.29) | -7.20* (22/117, p=7.6e-17) | -0.0060 ± 0.0037* | -0.0014 ± 0.0009* |
| FOCUS-2k vs FOCUS-256 local (same environment) | -0.26 (457/513, p=0.077) | +1.06* (28/14, p=0.044) | -0.0236 ± 0.0051* | -0.0015 ± 0.0009* |
| FQ16-1k vs FOCUS-256 local | -0.02 (542/547, p=0.9) | +0.30 (20/16, p=0.62) | -0.0216 ± 0.0045* | -0.0027 ± 0.0010* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 20 | 0 | 232.2 | 1024 | 20 |
| NVFP4 | 19 | 4 | 163.2 | 1024 | 7 |
| FourOverSix | 24 | 8 | 184.9 | 1024 | 13 |
| IF4† | 19 | 2 | 199.6 | 1024 | 16 |
| MixFP4 (Zou)† | 19 | 3 | 199.7 | 1024 | 13 |
| GPTQ‡ | 17 | 0 | 208.2 | 1024 | 17 |
| FlipQuant 8x64 | 17 | 4 | 188.3 | 1024 | 13 |
| FlipQuant 256x64 | 15 | 2 | 178.7 | 1024 | 13 |
| FOCUS, 256 | 35 | 11 | 166.5 | 1024 | 12 |
| FOCUS, 1,024 | 115 | 34 | 147.6 | 1024 | 7 |
| FOCUS, 2,048 | 7 | 1 | 168.5 | 1024 | 6 |
| FlipQuant 16x64, 256 | 21 | 5 | 184.6 | 1024 | 13 |
| FlipQuant 16x64, 1,024 | 17 | 4 | 193.5 | 1024 | 13 |
| FlipQuant 16x64, 2,048 | 13 | 1 | 180.7 | 1024 | 8 |
| FOCUS, 256 (local control, amendment 3) | 19 | 6 | 167.5 | 1024 | 11 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | -0.05 (26/27, p=1) | -0.28 (94/122, p=0.066) | +1.18 (73/58, p=0.22) | +0.88 (76/55, p=0.08) | +0.94 (41/30, p=0.24) | +0.35 (135/117, p=0.28) |
| FQ16-1k vs FOCUS-1k | +0.44 (41/33, p=0.42) | -0.23 (124/147, p=0.18) | -0.39 (70/75, p=0.74) | -1.18* (73/101, p=0.04) | -0.85 (46/56, p=0.37) | +0.35 (164/146, p=0.33) |
| FQ16-2k vs FQ16-256 | -0.16 (31/34, p=0.8) | -0.15 (103/118, p=0.35) | +0.00 (64/64, p=1) | +2.15* (96/45, p=2.1e-05) | +1.79* (49/28, p=0.022) | +0.19 (148/138, p=0.59) |
| FQ16-2k vs FOCUS-2k | +0.49 (42/33, p=0.36) | +0.22 (155/133, p=0.22) | -0.39 (66/71, p=0.73) | +0.51 (80/68, p=0.37) | -0.34 (43/47, p=0.75) | +0.87* (179/134, p=0.013) |
| FQ16-256 vs FOCUS-256 | +0.22 (42/38, p=0.74) | +0.31 (166/135, p=0.084) | -0.71 (64/73, p=0.49) | -1.60* (60/98, p=0.0031) | -1.19 (45/59, p=0.2) | +0.64 (181/148, p=0.078) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 23,779,840 | KL 0.0490; E0M3 0; flips 0 | KL 0.0492; E0M3 164; flips 164 | KL 0.0473; E0M3 5,225; flips 5,179 | KL 0.0435; E0M3 38,482; flips 35,215 | KL 0.0379; E0M3 105,278; flips 79,760 | 0.44 % |
| 1,024 | 23,779,840 | KL 0.0490; E0M3 5,091; flips 6,041 | KL 0.0445; E0M3 79,927; flips 117,788 | KL 0.0381; E0M3 263,869; flips 398,918 | KL 0.0333; E0M3 458,052; flips 633,763 | KL 0.0304; E0M3 602,694; flips 723,232 | 2.53 % |
| 2,048 | 23,779,840 | KL 0.0492; E0M3 44,298; flips 83,360 | KL 0.0416; E0M3 436,806; flips 1,058,734 | KL 0.0349; E0M3 872,640; flips 2,054,090 | KL 0.0313; E0M3 1,369,646; flips 2,809,590 | KL 0.0290; E0M3 1,608,918; flips 2,619,664 | 6.77 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0044 | 2486 | 2154 | 91.25 | 95770 | 51.89 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0253 | 9478 | 8643 | 90.52 | 95000 | 51.92 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.0677 | 18756 | 17262 | 90.52 | 95020 | 51.92 | 6,283,354,112 |
| FOCUS | 256 | part Q (cluster) | 8 | — | — | 2249 | 78.62 | — | — | 18,263,340,091 |
| FOCUS | 1,024 | this part | 32 | — | 17972 | 17810 | 91.79 | 95388 | — | 18,263,340,091 |
| FOCUS | 2,048 | this part | 64 | — | 35780 | 35627 | 91.79 | 95388 | — | 18,263,340,091 |
| FOCUS (local control) | 256 | this part | 8 | — | 4576 | 4453 | 91.79 | 95388 | — | 18,263,340,091 |

Evaluation wall times (s): gsm8k_focus256_local 2088, gsm8k_focus_1024 1759, gsm8k_focus_2048 1779, gsm8k_fq16_1024 2180, gsm8k_fq16_2048 2049, ppl_fo6 123, ppl_focus256 123, ppl_focus256_local 123, ppl_focus_1024 123, ppl_focus_2048 123, ppl_fq16_1024 131, ppl_fq16_2048 132, ppl_fq16_256 131, zs_focus256_local 1116, zs_focus_1024 1134, zs_focus_2048 1134, zs_fq16_1024 1100, zs_fq16_2048 1112, zs_fq16_256 1125.

## Qwen3-1.7B

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 72.25 | 60.40 | 61.01 | 69.70 | 42.92 | 50.48 | 59.46 | 74.53 ± 2.40 | 16.716 | 19.246 |
| NVFP4 | 70.24 | 57.61 | 58.41 | 62.88 | 37.54 | 42.71 | 54.90 | 61.64 ± 2.68 | 18.871 | 21.035 |
| FourOverSix | 70.57 | 58.03 | 58.48 | 65.49 | 40.36 | 43.86 | 56.13 | 61.87 ± 2.67 | 19.475 | 20.907 |
| IF4† | 68.44 | 58.81 | 58.72 | 66.62 | 38.48 | 43.49 | 55.76 | not run | 19.601 | 20.954 |
| MixFP4 (Zou)† | 68.34 | 58.49 | 57.38 | 65.66 | 39.08 | 43.00 | 55.32 | not run | 19.503 | 21.555 |
| GPTQ‡ | 69.26 | 56.74 | 57.54 | 64.65 | 39.93 | 43.55 | 55.28 | 66.72 ± 2.59 | 18.672 | 21.070 |
| FlipQuant 8x64 | 71.11 | 59.37 | 59.51 | 69.61 | 42.58 | 48.65 | 58.47 | TBD | 15.404 | 19.449 |
| FlipQuant 256x64 | 70.95 | 58.63 | 59.43 | 65.03 | 41.98 | 46.77 | 57.13 | TBD | 15.968 | 19.762 |
| FOCUS, 256 | 71.76 | 59.02 | 60.30 | 69.61 | 41.89 | 49.39 | 58.66 | 61.26 ± 2.68 | 15.089 | 19.535 |
| FOCUS, 1,024 | 71.71 | 58.61 | 60.62 | 68.56 | 41.13 | 48.86 | 58.25 | 66.34 ± 2.60 | 14.968 | 19.294 |
| FOCUS, 2,048 | 70.89 | 59.21 | 61.17 | 69.02 | 41.21 | 48.61 | 58.35 | 66.57 ± 2.60 | 15.297 | 19.483 |
| FlipQuant 16x64, 256 | 70.95 | 59.13 | 58.33 | 69.57 | 41.21 | 49.27 | 58.08 | 61.33 ± 2.68 | 15.398 | 19.410 |
| FlipQuant 16x64, 1,024 | 71.33 | 59.03 | 58.25 | 69.57 | 41.38 | 49.33 | 58.15 | 67.25 ± 2.58 | 15.762 | 19.571 |
| FlipQuant 16x64, 2,048 | 71.44 | 59.43 | 59.75 | 68.81 | 41.72 | 47.84 | 58.16 | 66.41 ± 2.60 | 16.024 | 19.722 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.20 (1193/1237, p=0.38) | +0.08 (149/148, p=1) | +0.0203 ± 0.0038* | -0.0064 ± 0.0021* |
| FQ16-1k vs FourOverSix | +2.30* (1553/1050, p=5.6e-23) | +5.38* (171/100, p=1.9e-05) | -0.2115 ± 0.0129* | -0.0660 ± 0.0038* |
| FQ16-1k vs FQ16-256 | +0.00 (1082/1081, p=1) | +5.91* (171/93, p=1.8e-06) | +0.0234 ± 0.0030* | +0.0083 ± 0.0018* |
| FOCUS-1k vs FourOverSix | +2.05* (1648/1201, p=5.6e-17) | +4.47* (190/131, p=0.0012) | -0.2632 ± 0.0144* | -0.0803 ± 0.0044* |
| FOCUS-1k vs FOCUS-256 | -0.45* (1068/1167, p=0.038) | +5.08* (181/114, p=0.00011) | -0.0080 ± 0.0030* | -0.0124 ± 0.0022* |
| FQ16-1k vs FOCUS-1k | +0.26 (1233/1177, p=0.26) | +0.91 (145/133, p=0.51) | +0.0517 ± 0.0040* | +0.0142 ± 0.0022* |
| FQ16-1k vs IF4† | +2.17* (1705/1231, p=2.2e-18) | TBD | -0.2179 ± 0.0121* | -0.0683 ± 0.0043* |
| FQ16-1k vs MixFP4 (Zou)† | +2.59* (1746/1180, p=1.1e-25) | TBD | -0.2129 ± 0.0112* | -0.0965 ± 0.0049* |
| FQ16-2k vs FourOverSix | +2.16* (1556/1083, p=3.2e-20) | +4.55* (174/114, p=0.00049) | -0.1950 ± 0.0127* | -0.0583 ± 0.0036* |
| FQ16-2k vs FQ16-256 | -0.13 (1084/1113, p=0.55) | +5.08* (177/110, p=9.1e-05) | +0.0399 ± 0.0033* | +0.0160 ± 0.0023* |
| FOCUS-2k vs FourOverSix | +2.28* (1711/1213, p=3.2e-20) | +4.70* (185/123, p=0.00049) | -0.2415 ± 0.0140* | -0.0705 ± 0.0043* |
| FOCUS-2k vs FOCUS-256 | -0.22 (1155/1203, p=0.33) | +5.31* (165/95, p=1.7e-05) | +0.0137 ± 0.0032* | -0.0026 ± 0.0024* |
| FQ16-2k vs FOCUS-2k | -0.11 (1210/1235, p=0.63) | -0.15 (139/141, p=0.95) | +0.0464 ± 0.0038* | +0.0122 ± 0.0027* |
| FQ16-2k vs IF4† | +2.03* (1683/1239, p=2.2e-16) | TBD | -0.2015 ± 0.0118* | -0.0606 ± 0.0040* |
| FQ16-2k vs MixFP4 (Zou)† | +2.45* (1726/1190, p=2.9e-23) | TBD | -0.1965 ± 0.0110* | -0.0888 ± 0.0045* |
| FQ16-2k vs FQ16-1k | -0.14 (1062/1092, p=0.53) | -0.83 (126/137, p=0.54) | +0.0165 ± 0.0032* | +0.0077 ± 0.0020* |
| FOCUS-2k vs FOCUS-1k | +0.23 (1146/1095, p=0.29) | +0.23 (141/138, p=0.9) | +0.0217 ± 0.0030* | +0.0097 ± 0.0019* |
| FQ16-1k vs FOCUS-256 | -0.20 (1244/1287, p=0.4) | +5.99* (183/104, p=3.6e-06) | +0.0437 ± 0.0043* | +0.0019 ± 0.0023 |
| FQ16-2k vs FOCUS-256 | -0.33 (1241/1314, p=0.15) | +5.16* (186/118, p=0.00011) | +0.0601 ± 0.0043* | +0.0096 ± 0.0028* |
| FQ16-256 vs GPTQ‡ | +3.24* (1896/1188, p=1.9e-37) | -5.38* (120/191, p=6.7e-05) | -0.1928 ± 0.0083* | -0.0821 ± 0.0041* |
| FQ16-1k vs GPTQ‡ | +3.25* (1892/1183, p=1.2e-37) | +0.53 (144/137, p=0.72) | -0.1694 ± 0.0085* | -0.0738 ± 0.0040* |
| FQ16-2k vs GPTQ‡ | +3.11* (1866/1187, p=6.9e-35) | -0.30 (144/148, p=0.86) | -0.1529 ± 0.0082* | -0.0661 ± 0.0036* |
| FQ16-256 vs IF4† | +2.16* (1699/1226, p=2.2e-18) | TBD | -0.2413 ± 0.0120* | -0.0765 ± 0.0044* |
| FQ16-256 vs MixFP4 (Zou)† | +2.59* (1765/1200, p=2.8e-25) | TBD | -0.2363 ± 0.0113* | -0.1048 ± 0.0049* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 4 | 0 | 125.2 | 640 | 0 |
| NVFP4 | 22 | 0 | 125.1 | 1024 | 1 |
| FourOverSix | 7 | 0 | 128.4 | 889 | 0 |
| GPTQ‡ | 10 | 0 | 133.8 | 1024 | 2 |
| FOCUS, 256 | 11 | 0 | 120.4 | 557 | 0 |
| FOCUS, 1,024 | 18 | 0 | 128.7 | 1024 | 2 |
| FOCUS, 2,048 | 20 | 0 | 128.8 | 1024 | 3 |
| FlipQuant 16x64, 256 | 44 | 0 | 127.9 | 623 | 0 |
| FlipQuant 16x64, 1,024 | 17 | 0 | 133.4 | 1024 | 2 |
| FlipQuant 16x64, 2,048 | 14 | 0 | 130.5 | 1024 | 4 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +0.38 (83/76, p=0.63) | -0.10 (323/333, p=0.73) | -0.08 (169/170, p=1) | +0.00 (137/137, p=1) | +0.17 (87/85, p=0.94) | +0.06 (283/280, p=0.93) |
| FOCUS-1k vs FOCUS-256 | -0.05 (94/95, p=1) | -0.41 (330/371, p=0.13) | +0.32 (172/168, p=0.87) | -1.05 (128/153, p=0.15) | -0.77 (65/74, p=0.5) | -0.52 (279/306, p=0.28) |
| FQ16-1k vs FOCUS-1k | -0.38 (90/97, p=0.66) | +0.42 (413/371, p=0.14) | -2.37 (162/192, p=0.12) | +1.01 (155/131, p=0.17) | +0.26 (80/77, p=0.87) | +0.47 (333/309, p=0.36) |
| FQ16-2k vs FQ16-256 | +0.49 (86/77, p=0.53) | +0.30 (359/329, p=0.27) | +1.42 (171/153, p=0.34) | -0.76 (128/146, p=0.3) | +0.51 (76/70, p=0.68) | -1.44* (264/338, p=0.0029) |
| FOCUS-2k vs FOCUS-256 | -0.87 (105/121, p=0.32) | +0.19 (384/365, p=0.51) | +0.87 (165/154, p=0.58) | -0.59 (135/149, p=0.44) | -0.68 (79/87, p=0.59) | -0.78 (287/327, p=0.12) |
| FQ16-2k vs FOCUS-2k | +0.54 (107/97, p=0.53) | +0.22 (402/380, p=0.45) | -1.42 (157/175, p=0.35) | -0.21 (155/160, p=0.82) | +0.51 (87/81, p=0.7) | -0.78 (302/342, p=0.12) |
| FQ16-256 vs FOCUS-256 | -0.82 (99/114, p=0.34) | +0.11 (395/384, p=0.72) | -1.97 (166/191, p=0.2) | -0.04 (157/158, p=1) | -0.68 (77/85, p=0.58) | -0.12 (299/305, p=0.84) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 1,376,256 | KL 0.1655; E0M3 0; flips 0 | KL 0.1632; E0M3 926; flips 1,076 | KL 0.1389; E0M3 4,759; flips 4,455 | KL 0.1277; E0M3 9,470; flips 6,387 | KL 0.1189; E0M3 16,351; flips 9,751 | 1.19 % |
| 1,024 | 1,376,256 | KL 0.1596; E0M3 5,110; flips 6,636 | KL 0.1299; E0M3 19,594; flips 28,726 | KL 0.1163; E0M3 46,249; flips 66,963 | KL 0.1046; E0M3 78,236; flips 105,105 | KL 0.0954; E0M3 112,189; flips 138,241 | 8.15 % |
| 2,048 | 1,376,256 | KL 0.1486; E0M3 16,687; flips 32,927 | KL 0.1208; E0M3 63,447; flips 154,324 | KL 0.1068; E0M3 124,792; flips 297,033 | KL 0.0968; E0M3 183,511; flips 405,189 | KL 0.0907; E0M3 237,001; flips 486,236 | 17.22 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0119 | 111 | 61 | 20.99 | 22652 | 4.70 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0815 | 334 | 245 | 23.19 | 24842 | 4.73 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.1722 | 632 | 490 | 26.12 | 27862 | 4.73 | 6,283,354,112 |
| FOCUS | 256 | part K | 8 | — | — | 50 | 15.93 | — | — | 1,057,104,195 |
| FOCUS | 1,024 | this part | 32 | — | 238 | 200 | 15.93 | 18788 | — | 1,057,104,195 |
| FOCUS | 2,048 | this part | 64 | — | 447 | 399 | 15.93 | 18788 | — | 1,057,104,195 |

Evaluation wall times (s): gsm8k_bf16 461, gsm8k_fo6 443, gsm8k_focus 389, gsm8k_focus_1024 566, gsm8k_focus_2048 560, gsm8k_fq-16x64 459, gsm8k_fq16_1024 584, gsm8k_fq16_2048 635, gsm8k_gptq 577, gsm8k_nvfp4 534, ppl_fo6 26, ppl_focus256 27, ppl_focus_1024 27, ppl_focus_2048 27, ppl_fq16_1024 27, ppl_fq16_2048 27, ppl_fq16_256 26, zs_focus_1024 264, zs_focus_2048 260, zs_fq16_1024 260, zs_fq16_2048 256, zs_fq16_256 263.

## Qwen3-8B

Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 (c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 included).
The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL windows); a failed check would show "mismatch" (calib_size.json baseline_check). † simulated with each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.

| method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | GSM8K | Wiki PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 77.80 | 74.93 | 68.03 | 80.85 | 56.91 | 64.25 | 70.46 | 91.36 ± 1.55 | 9.725 | 13.300 |
| NVFP4 | 76.44 | 73.16 | 67.56 | 79.80 | 55.38 | 62.47 | 69.13 | 88.25 ± 1.77 | 10.048 | 13.770 |
| FourOverSix | 75.95 | 73.18 | 67.17 | 79.71 | 54.86 | 62.72 | 68.93 | 88.25 ± 1.77 | 10.007 | 13.757 |
| IF4† | 76.66 | 73.74 | 67.40 | 78.07 | 54.10 | 63.69 | 68.94 | not run | 9.885 | 13.603 |
| MixFP4 (Zou)† | 76.33 | 73.76 | 68.67 | 78.37 | 53.92 | 62.95 | 69.00 | not run | 9.861 | 13.592 |
| GPTQ‡ | 76.44 | 72.84 | 67.25 | 79.84 | 54.69 | 63.48 | 69.09 | 89.08 ± 1.72 | 10.129 | 13.704 |
| FlipQuant 8x64 | 77.64 | 74.12 | 68.82 | 80.56 | 55.80 | 63.24 | 70.03 | TBD | 9.608 | 13.528 |
| FlipQuant 256x64 | 76.50 | 73.78 | 68.98 | 79.84 | 55.03 | 63.57 | 69.62 | TBD | 9.678 | 13.565 |
| FOCUS, 256 | 77.91 | 73.83 | 69.30 | 81.36 | 55.97 | 63.83 | 70.37 | 86.43 ± 1.89 | 9.569 | 13.571 |
| FOCUS, 1,024 | 78.18 | 74.15 | 69.93 | 81.23 | 58.36 | 63.46 | 70.88 | 87.72 ± 1.81 | 9.523 | 13.501 |
| FOCUS, 2,048 | 77.97 | 74.13 | 70.40 | 81.10 | 58.02 | 64.06 | 70.95 | 89.16 ± 1.71 | 9.502 | 13.489 |
| FlipQuant 16x64, 256 | 76.71 | 74.27 | 68.59 | 79.97 | 55.46 | 63.40 | 69.73 | 89.31 ± 1.70 | 9.590 | 13.555 |
| FlipQuant 16x64, 1,024 | 78.18 | 74.23 | 68.11 | 81.40 | 56.31 | 62.84 | 70.18 | 90.22 ± 1.64 | 9.584 | 13.487 |
| FlipQuant 16x64, 2,048 | 77.86 | 74.24 | 69.46 | 80.51 | 56.31 | 63.42 | 70.30 | 89.69 ± 1.67 | 9.620 | 13.568 |

**Paired.**
- Zero-shot: pooled over all questions of the six tasks.
- GSM8K: strict-match.
- Format: difference in points (row-only / reference-only correct, exact McNemar p).
- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).
- \* = p < 0.05, or |Δ| > 2 SE.

| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |
|---|---|---|---|---|
| FQ16-256 vs FOCUS-256 | -0.22 (763/811, p=0.24) | +2.88* (82/44, p=0.00091) | +0.0022 ± 0.0024 | -0.0012 ± 0.0015 |
| FQ16-1k vs FourOverSix | +1.01* (841/620, p=8.1e-09) | +1.97* (64/38, p=0.013) | -0.0432 ± 0.0032* | -0.0198 ± 0.0018* |
| FQ16-1k vs FQ16-256 | +0.15 (675/643, p=0.39) | +0.91 (50/38, p=0.24) | -0.0006 ± 0.0020 | -0.0050 ± 0.0011* |
| FOCUS-1k vs FourOverSix | +1.32* (1009/721, p=4.7e-12) | -0.53 (62/69, p=0.6) | -0.0496 ± 0.0037* | -0.0188 ± 0.0028* |
| FOCUS-1k vs FOCUS-256 | +0.23 (738/687, p=0.19) | +1.29 (68/51, p=0.14) | -0.0048 ± 0.0021* | -0.0052 ± 0.0013* |
| FQ16-1k vs FOCUS-1k | -0.31 (725/792, p=0.09) | +2.50* (72/39, p=0.0022) | +0.0064 ± 0.0023* | -0.0010 ± 0.0017 |
| FQ16-1k vs IF4† | +0.67* (953/806, p=0.0005) | TBD | -0.0309 ± 0.0034* | -0.0085 ± 0.0020* |
| FQ16-1k vs MixFP4 (Zou)† | +0.77* (955/787, p=6.2e-05) | TBD | -0.0285 ± 0.0031* | -0.0077 ± 0.0019* |
| FQ16-2k vs FourOverSix | +1.11* (886/644, p=6.7e-10) | +1.44 (56/37, p=0.061) | -0.0395 ± 0.0032* | -0.0139 ± 0.0021* |
| FQ16-2k vs FQ16-256 | +0.24 (724/671, p=0.16) | +0.38 (47/42, p=0.67) | +0.0031 ± 0.0021* | +0.0009 ± 0.0013 |
| FOCUS-2k vs FourOverSix | +1.43* (1047/735, p=1.5e-13) | +0.91 (62/50, p=0.3) | -0.0518 ± 0.0038* | -0.0197 ± 0.0027* |
| FOCUS-2k vs FOCUS-256 | +0.34 (770/695, p=0.053) | +2.73* (77/41, p=0.0012) | -0.0070 ± 0.0023* | -0.0061 ± 0.0014* |
| FQ16-2k vs FOCUS-2k | -0.32 (745/815, p=0.081) | +0.53 (56/49, p=0.56) | +0.0123 ± 0.0021* | +0.0058 ± 0.0015* |
| FQ16-2k vs IF4† | +0.77* (979/811, p=7.8e-05) | TBD | -0.0272 ± 0.0034* | -0.0025 ± 0.0020* |
| FQ16-2k vs MixFP4 (Zou)† | +0.87* (979/790, p=7.7e-06) | TBD | -0.0248 ± 0.0031* | -0.0017 ± 0.0019 |
| FQ16-2k vs FQ16-1k | +0.10 (681/660, p=0.58) | -0.53 (34/41, p=0.49) | +0.0037 ± 0.0019* | +0.0060 ± 0.0011* |
| FOCUS-2k vs FOCUS-1k | +0.11 (713/689, p=0.54) | +1.44 (64/45, p=0.084) | -0.0022 ± 0.0021* | -0.0009 ± 0.0013 |
| FQ16-1k vs FOCUS-256 | -0.07 (773/789, p=0.7) | +3.79* (89/39, p=1.2e-05) | +0.0016 ± 0.0023 | -0.0062 ± 0.0016* |
| FQ16-2k vs FOCUS-256 | +0.02 (795/790, p=0.92) | +3.26* (84/41, p=0.00015) | +0.0053 ± 0.0021* | -0.0002 ± 0.0016 |
| FQ16-256 vs GPTQ‡ | +0.79* (970/797, p=4.2e-05) | +0.23 (59/56, p=0.85) | -0.0546 ± 0.0038* | -0.0109 ± 0.0023* |
| FQ16-1k vs GPTQ‡ | +0.94* (991/786, p=1.3e-06) | +1.14 (61/46, p=0.18) | -0.0553 ± 0.0038* | -0.0159 ± 0.0021* |
| FQ16-2k vs GPTQ‡ | +1.03* (995/769, p=8.1e-08) | +0.61 (58/50, p=0.5) | -0.0516 ± 0.0037* | -0.0099 ± 0.0024* |
| FQ16-256 vs IF4† | +0.53* (969/854, p=0.0076) | TBD | -0.0303 ± 0.0038* | -0.0035 ± 0.0019* |
| FQ16-256 vs MixFP4 (Zou)† | +0.62* (975/839, p=0.0015) | TBD | -0.0279 ± 0.0034* | -0.0027 ± 0.0019* |

GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at "The final [answer [is]]", mean / max generated tokens and budget (1,024) hits, of 1,319:

| method | strict no-answer | cut at "The final…" | mean tokens | max | budget hits |
|---|---:|---:|---:|---:|---:|
| BF16 | 6 | 0 | 132.7 | 440 | 0 |
| NVFP4 | 39 | 0 | 140.3 | 1024 | 1 |
| FourOverSix | 14 | 0 | 132.6 | 562 | 0 |
| GPTQ‡ | 23 | 0 | 142.6 | 1024 | 3 |
| FOCUS, 256 | 59 | 0 | 127.5 | 1024 | 1 |
| FOCUS, 1,024 | 31 | 0 | 130.7 | 1024 | 1 |
| FOCUS, 2,048 | 18 | 0 | 132.3 | 496 | 0 |
| FlipQuant 16x64, 256 | 17 | 0 | 128.8 | 1024 | 1 |
| FlipQuant 16x64, 1,024 | 10 | 0 | 133.4 | 1024 | 1 |
| FlipQuant 16x64, 2,048 | 11 | 0 | 128.7 | 1024 | 1 |

Zero-shot per task (difference in points, counts, exact p):

| comparison | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA |
|---|---|---|---|---|---|---|
| FQ16-1k vs FQ16-256 | +1.47* (61/34, p=0.0073) | -0.04 (191/195, p=0.88) | -0.47 (99/105, p=0.73) | +1.43* (87/53, p=0.0051) | +0.85 (62/52, p=0.4) | -0.56 (175/204, p=0.15) |
| FOCUS-1k vs FOCUS-256 | +0.27 (60/55, p=0.71) | +0.32 (233/201, p=0.14) | +0.63 (108/100, p=0.63) | -0.13 (73/76, p=0.87) | +2.39* (74/46, p=0.013) | -0.37 (190/209, p=0.37) |
| FQ16-1k vs FOCUS-1k | +0.00 (60/60, p=1) | +0.08 (234/226, p=0.74) | -1.82 (104/127, p=0.15) | +0.17 (83/79, p=0.81) | -2.05* (48/72, p=0.035) | -0.62 (196/228, p=0.13) |
| FQ16-2k vs FQ16-256 | +1.14* (59/38, p=0.042) | -0.03 (215/218, p=0.92) | +0.87 (116/105, p=0.5) | +0.55 (80/67, p=0.32) | +0.85 (56/46, p=0.37) | +0.02 (198/197, p=1) |
| FOCUS-2k vs FOCUS-256 | +0.05 (65/64, p=1) | +0.30 (229/199, p=0.16) | +1.10 (119/105, p=0.39) | -0.25 (71/77, p=0.68) | +2.05* (67/43, p=0.028) | +0.23 (219/207, p=0.59) |
| FQ16-2k vs FOCUS-2k | -0.11 (53/55, p=0.92) | +0.11 (235/224, p=0.64) | -0.95 (107/119, p=0.46) | -0.59 (73/87, p=0.3) | -1.71 (50/70, p=0.082) | -0.64 (227/260, p=0.15) |
| FQ16-256 vs FOCUS-256 | -1.20* (42/64, p=0.041) | +0.44 (267/223, p=0.052) | -0.71 (122/131, p=0.62) | -1.39* (61/94, p=0.0099) | -0.51 (54/60, p=0.64) | -0.43 (217/239, p=0.33) |

**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).

| windows | tiles | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | final E0M3 share |
|---:|---:|---|---|---|---|---|---:|
| 256 | 6,782,976 | KL 0.0885; E0M3 0; flips 0 | KL 0.0873; E0M3 683; flips 693 | KL 0.0802; E0M3 4,071; flips 4,058 | KL 0.0734; E0M3 15,576; flips 13,007 | KL 0.0654; E0M3 37,511; flips 26,785 | 0.55 % |
| 1,024 | 6,782,976 | KL 0.0883; E0M3 4,790; flips 6,458 | KL 0.0748; E0M3 35,920; flips 51,092 | KL 0.0650; E0M3 118,439; flips 168,127 | KL 0.0557; E0M3 210,556; flips 269,605 | KL 0.0489; E0M3 300,918; flips 352,050 | 4.44 % |
| 2,048 | 6,782,976 | KL 0.0840; E0M3 20,903; flips 38,189 | KL 0.0696; E0M3 147,084; flips 331,443 | KL 0.0575; E0M3 315,279; flips 678,909 | KL 0.0502; E0M3 469,912; flips 919,469 | KL 0.0456; E0M3 600,273; flips 1,057,715 | 8.85 % |

**Calibration cost and E0M3 share.**

| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | device peak (MiB) | host RSS (GiB) | teacher / state (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FlipQuant 16x64 | 256 | release | 160 | 0.0055 | 296 | 210 | 46.84 | 49352 | 16.18 | 785,419,264 |
| FlipQuant 16x64 | 1,024 | this part | 640 | 0.0444 | 970 | 836 | 49.03 | 51602 | 16.21 | 3,141,677,056 |
| FlipQuant 16x64 | 2,048 | this part | 1280 | 0.0885 | 1875 | 1674 | 51.96 | 54622 | 16.20 | 6,283,354,112 |
| FOCUS | 256 | part K | 8 | — | — | 236 | 52.62 | — | — | 5,209,511,899 |
| FOCUS | 1,024 | this part | 32 | — | 991 | 941 | 52.62 | 55328 | — | 5,209,511,899 |
| FOCUS | 2,048 | this part | 64 | — | 1939 | 1881 | 52.62 | 55328 | — | 5,209,511,899 |

Evaluation wall times (s): gsm8k_bf16 888, gsm8k_fo6 855, gsm8k_focus 887, gsm8k_focus_1024 913, gsm8k_focus_2048 838, gsm8k_fq-16x64 903, gsm8k_fq16_1024 989, gsm8k_fq16_2048 912, gsm8k_gptq 1170, gsm8k_nvfp4 1011, ppl_fo6 44, ppl_focus256 45, ppl_focus_1024 45, ppl_focus_2048 45, ppl_fq16_1024 47, ppl_fq16_2048 48, ppl_fq16_256 46, zs_focus_1024 321, zs_focus_2048 323, zs_fq16_1024 318, zs_fq16_2048 317, zs_fq16_256 329.

