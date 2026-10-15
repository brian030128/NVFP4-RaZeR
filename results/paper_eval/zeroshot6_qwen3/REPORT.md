# Zero-shot accuracy: six tasks, Qwen3-1.7B and Qwen3-8B, ten rows (part X amendment 2, X2)

Part X's protocol, unchanged, for two more models (PROTOCOL.md part X amendment 2):
- **Harness:** flipquant origin/main `3db0072`, `evaluation.downstream`, in the same fresh detached worktree, unmodified.
- **Settings:** lm-eval 0.4.11, 0-shot, batch 16, per-token activations. Datasets are pinned, LAMBADA at 900124bf.
- **Execution:** native rows on the RTX PRO 6000 (build_V, `--kernel-set auto`). IF4 and MixFP4 (Zou) are simulated with their own rules (†), as in tab:main-ppl.
- **Checks:** part X's check_record on every run. Every input was re-hashed before its run against the registered records.
- Every cell is a new run. Part X's report for the other six models is `results/paper_eval/zeroshot6/`.

**Columns.** Accuracy in %: acc_norm for PIQA, HellaSwag, ARC-e and ARC-c; acc for WinoGrande and LAMBADA. Avg. is the unweighted mean of the six.
- **Paired columns:** they pool every question of the six tasks (21,848 per row). Each gives the difference in points, the row-only / reference-only correct counts and the exact McNemar p.
- `*` = p < 0.05 (acc_panel_tables.py's formula).

## Summary

Avg. of the six tasks (%) per model, with the pooled paired difference from FourOverSix in points (* = exact McNemar p < 0.05; the full table has the counts and p). Mean = the unweighted mean of the two models' Avg.

| method | Qwen3-1.7B | Qwen3-8B | mean |
|---|---:|---:|---:|
| BF16 | 59.46 (+3.53*) | 70.46 (+1.60*) | 64.96 |
| NVFP4 | 54.90 (-0.93*) | 69.13 (+0.03) | 62.02 |
| FourOverSix | 56.13 | 68.93 | 62.53 |
| FOCUS | 58.66 (+2.50*) | 70.37 (+1.08*) | 64.51 |
| GPTQ‡ | 55.28 (-0.94*) | 69.09 (+0.07) | 62.18 |
| IF4† | 55.76 (+0.13) | 68.94 (+0.34) | 62.35 |
| MixFP4 (Zou)† | 55.32 (-0.29) | 69.00 (+0.24) | 62.16 |
| FlipQuant 8x64 | 58.47 (+2.42*) | 70.03 (+0.93*) | 64.25 |
| FlipQuant 16x64 | 58.08 (+2.30*) | 69.73 (+0.87*) | 63.90 |
| FlipQuant 256x64 | 57.13 (+1.09*) | 69.62 (+0.65*) | 63.37 |

## Full table

| model | method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | vs FourOverSix | vs BF16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3-1.7B | BF16 | 72.25 | 60.40 | 61.01 | 69.70 | 42.92 | 50.48 | 59.46 | +3.53* (1766/994, p=2.2e-49) | — |
| Qwen3-1.7B | NVFP4 | 70.24 | 57.61 | 58.41 | 62.88 | 37.54 | 42.71 | 54.90 | -0.93* (1356/1559, p=0.00018) | -4.46* (967/1942, p=2.8e-74) |
| Qwen3-1.7B | FourOverSix | 70.57 | 58.03 | 58.48 | 65.49 | 40.36 | 43.86 | 56.13 | — | -3.53* (994/1766, p=2.2e-49) |
| Qwen3-1.7B | FOCUS | 71.76 | 59.02 | 60.30 | 69.61 | 41.89 | 49.39 | 58.66 | +2.50* (1705/1159, p=1.7e-24) | -1.03* (1114/1340, p=5.5e-06) |
| Qwen3-1.7B | GPTQ‡ | 69.26 | 56.74 | 57.54 | 64.65 | 39.93 | 43.55 | 55.28 | -0.94* (1488/1694, p=0.00028) | -4.48* (964/1942, p=8.1e-75) |
| Qwen3-1.7B | IF4† | 68.44 | 58.81 | 58.72 | 66.62 | 38.48 | 43.49 | 55.76 | +0.13 (1515/1486, p=0.61) | -3.40* (990/1733, p=1.9e-46) |
| Qwen3-1.7B | MixFP4 (Zou)† | 68.34 | 58.49 | 57.38 | 65.66 | 39.08 | 43.00 | 55.32 | -0.29 (1506/1569, p=0.26) | -3.82* (953/1788, p=5.1e-58) |
| Qwen3-1.7B | FlipQuant 8x64 | 71.11 | 59.37 | 59.51 | 69.61 | 42.58 | 48.65 | 58.47 | +2.42* (1539/1010, p=9e-26) | -1.11* (1068/1311, p=6.8e-07) |
| Qwen3-1.7B | FlipQuant 16x64 | 70.95 | 59.13 | 58.33 | 69.57 | 41.21 | 49.27 | 58.08 | +2.30* (1475/973, p=2.9e-24) | -1.24* (1074/1344, p=4.4e-08) |
| Qwen3-1.7B | FlipQuant 256x64 | 70.95 | 58.63 | 59.43 | 65.03 | 41.98 | 46.77 | 57.13 | +1.09* (1345/1107, p=1.7e-06) | -2.44* (1012/1546, p=3.8e-26) |
| Qwen3-8B | BF16 | 77.80 | 74.93 | 68.03 | 80.85 | 56.91 | 64.25 | 70.46 | +1.60* (972/622, p=1.7e-18) | — |
| Qwen3-8B | NVFP4 | 76.44 | 73.16 | 67.56 | 79.80 | 55.38 | 62.47 | 69.13 | +0.03 (842/835, p=0.88) | -1.57* (652/995, p=2.7e-17) |
| Qwen3-8B | FourOverSix | 75.95 | 73.18 | 67.17 | 79.71 | 54.86 | 62.72 | 68.93 | — | -1.60* (622/972, p=1.7e-18) |
| Qwen3-8B | FOCUS | 77.91 | 73.83 | 69.30 | 81.36 | 55.97 | 63.83 | 70.37 | +1.08* (983/746, p=1.3e-08) | -0.52* (700/813, p=0.004) |
| Qwen3-8B | GPTQ‡ | 76.44 | 72.84 | 67.25 | 79.84 | 54.69 | 63.48 | 69.09 | +0.07 (939/923, p=0.73) | -1.53* (613/947, p=2.6e-17) |
| Qwen3-8B | IF4† | 76.66 | 73.74 | 67.40 | 78.07 | 54.10 | 63.69 | 68.94 | +0.34 (962/888, p=0.09) | -1.26* (654/930, p=4.3e-12) |
| Qwen3-8B | MixFP4 (Zou)† | 76.33 | 73.76 | 68.67 | 78.37 | 53.92 | 62.95 | 69.00 | +0.24 (929/876, p=0.22) | -1.36* (640/937, p=7.7e-14) |
| Qwen3-8B | FlipQuant 8x64 | 77.64 | 74.12 | 68.82 | 80.56 | 55.80 | 63.24 | 70.03 | +0.93* (819/615, p=7.9e-08) | -0.67* (680/826, p=0.00018) |
| Qwen3-8B | FlipQuant 16x64 | 76.71 | 74.27 | 68.59 | 79.97 | 55.46 | 63.40 | 69.73 | +0.87* (799/610, p=5.3e-07) | -0.74* (683/844, p=4.2e-05) |
| Qwen3-8B | FlipQuant 256x64 | 76.50 | 73.78 | 68.98 | 79.84 | 55.03 | 63.57 | 69.62 | +0.65* (801/659, p=0.00022) | -0.95* (664/872, p=1.2e-07) |

**Sample digests.** Every row of a model saw the same samples: equal sample_digest per task, checked by the driver.
- Across the two models: PIQA equal, HellaSwag equal, WinoGrande equal, ARC-e equal, ARC-c equal, LAMBADA equal.
- Against part X's six models: PIQA equal, HellaSwag equal, WinoGrande equal, ARC-e equal, ARC-c equal, LAMBADA equal.

## LaTeX

```latex
\midrule\multicolumn{8}{l}{\textit{Qwen3-1.7B}} \\
BF16 & 72.25 & 60.40 & 61.01 & 69.70 & 42.92 & 50.48 & 59.46 \\
NVFP4 & 70.24 & 57.61 & 58.41 & 62.88 & 37.54 & 42.71 & 54.90 \\
FourOverSix & 70.57 & 58.03 & 58.48 & 65.49 & 40.36 & 43.86 & 56.13 \\
FOCUS & 71.76 & 59.02 & 60.30 & 69.61 & 41.89 & 49.39 & 58.66 \\
GPTQ$^{\ddagger}$ & 69.26 & 56.74 & 57.54 & 64.65 & 39.93 & 43.55 & 55.28 \\
IF4$^{\dagger}$ & 68.44 & 58.81 & 58.72 & 66.62 & 38.48 & 43.49 & 55.76 \\
MixFP4$^{\dagger}$ & 68.34 & 58.49 & 57.38 & 65.66 & 39.08 & 43.00 & 55.32 \\
\method{} $8{\times}64$ & 71.11 & 59.37 & 59.51 & 69.61 & 42.58 & 48.65 & 58.47 \\
\method{} $16{\times}64$ & 70.95 & 59.13 & 58.33 & 69.57 & 41.21 & 49.27 & 58.08 \\
\method{} $256{\times}64$ & 70.95 & 58.63 & 59.43 & 65.03 & 41.98 & 46.77 & 57.13 \\
\midrule\multicolumn{8}{l}{\textit{Qwen3-8B}} \\
BF16 & 77.80 & 74.93 & 68.03 & 80.85 & 56.91 & 64.25 & 70.46 \\
NVFP4 & 76.44 & 73.16 & 67.56 & 79.80 & 55.38 & 62.47 & 69.13 \\
FourOverSix & 75.95 & 73.18 & 67.17 & 79.71 & 54.86 & 62.72 & 68.93 \\
FOCUS & 77.91 & 73.83 & 69.30 & 81.36 & 55.97 & 63.83 & 70.37 \\
GPTQ$^{\ddagger}$ & 76.44 & 72.84 & 67.25 & 79.84 & 54.69 & 63.48 & 69.09 \\
IF4$^{\dagger}$ & 76.66 & 73.74 & 67.40 & 78.07 & 54.10 & 63.69 & 68.94 \\
MixFP4$^{\dagger}$ & 76.33 & 73.76 & 68.67 & 78.37 & 53.92 & 62.95 & 69.00 \\
\method{} $8{\times}64$ & 77.64 & 74.12 & 68.82 & 80.56 & 55.80 & 63.24 & 70.03 \\
\method{} $16{\times}64$ & 76.71 & 74.27 & 68.59 & 79.97 & 55.46 & 63.40 & 69.73 \\
\method{} $256{\times}64$ & 76.50 & 73.78 & 68.98 & 79.84 & 55.03 & 63.57 & 69.62 \\
```

The columns: method, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, LAMBADA, Avg. Values are rounded once, from the per-example records.

## Old vs new: the co-author's acc_panel (flipquant main docs/results/acc_panel/acc_panel.md)

- **Old:** fake quant on H200 (scripts/run_main_acc.py, run_small_acc.py). Only the table's rounded percentages exist; there are no per-example records.
- **New:** this run, with native rows on the RTX PRO 6000.
- Δ = new − old in points. 2 SE = 2 sqrt(p1 q1 / n + p2 q2 / n) (binomial, unpaired). ✗ = |Δ| > 2 SE.

| model | method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. Δ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | BF16 | 72.25 / 72.20 (+0.05) | 60.40 / 60.40 (-0.00) | 61.01 / 61.40 (-0.39) | 69.70 / 70.20 (-0.50) | 42.92 / 43.00 (-0.08) | 50.48 / 50.36 (+0.12) | -0.14 |
| Qwen3-1.7B | FourOverSix | 70.57 / 70.62 (-0.05) | 58.03 / 57.83 (+0.20) | 58.48 / 55.88 (+2.60) | 65.49 / 65.40 (+0.09) | 40.36 / 39.16 (+1.20) | 43.86 / 43.76 (+0.10) | +0.69 |
| Qwen3-1.7B | FOCUS | 71.76 / 71.82 (-0.06) | 59.02 / 58.86 (+0.16) | 60.30 / 59.12 (+1.18) | 69.61 / 68.73 (+0.88) | 41.89 / 41.64 (+0.25) | 49.39 / 49.72 (-0.33) | +0.35 |
| Qwen3-1.7B | FlipQuant 8x64 | 71.11 / 70.89 (+0.22) | 59.37 / 59.51 (-0.14) | 59.51 / 60.85 (-1.34) | 69.61 / 69.23 (+0.38) | 42.58 / 41.04 (+1.54) | 48.65 / 48.92 (-0.27) | +0.07 |
| Qwen3-1.7B | FlipQuant 16x64 | 70.95 / 72.14 (-1.19) | 59.13 / 59.17 (-0.04) | 58.33 / 58.33 (-0.00) | 69.57 / 68.69 (+0.88) | 41.21 / 41.98 (-0.77) | 49.27 / 49.33 (-0.06) | -0.20 |
| Qwen3-1.7B | FlipQuant 256x64 | 70.95 / 71.71 (-0.76) | 58.63 / 58.95 (-0.32) | 59.43 / 59.75 (-0.32) | 65.03 / 66.62 (-1.59) | 41.98 / 39.51 (+2.47) | 46.77 / 48.15 (-1.38) | -0.32 |
| Qwen3-8B | BF16 | 77.80 / 77.64 (+0.16) | 74.93 / 74.89 (+0.04) | 68.03 / 68.27 (-0.24) | 80.85 / 80.93 (-0.08) | 56.91 / 56.48 (+0.43) | 64.25 / 64.12 (+0.13) | +0.07 |
| Qwen3-8B | FourOverSix | 75.95 / 76.88 (-0.93) | 73.18 / 73.29 (-0.11) | 67.17 / 68.75 (-1.58) | 79.71 / 79.04 (+0.67) | 54.86 / 54.01 (+0.85) | 62.72 / 62.70 (+0.02) | -0.18 |
| Qwen3-8B | FOCUS | 77.91 / 77.80 (+0.11) | 73.83 / 74.06 (-0.23) | 69.30 / 68.98 (+0.32) | 81.36 / 80.26 (+1.10) | 55.97 / 55.46 (+0.51) | 63.83 / 64.02 (-0.19) | +0.27 |
| Qwen3-8B | FlipQuant 8x64 | 77.64 / 76.88 (+0.76) | 74.12 / 74.13 (-0.01) | 68.82 / 69.06 (-0.24) | 80.56 / 80.51 (+0.05) | 55.80 / 55.63 (+0.17) | 63.24 / 62.72 (+0.52) | +0.21 |
| Qwen3-8B | FlipQuant 16x64 | 76.71 / 76.99 (-0.28) | 74.27 / 74.17 (+0.10) | 68.59 / 69.53 (-0.94) | 79.97 / 80.18 (-0.21) | 55.46 / 56.83 (-1.37) | 63.40 / 63.65 (-0.25) | -0.49 |
| Qwen3-8B | FlipQuant 256x64 | 76.50 / 76.93 (-0.43) | 73.78 / 73.64 (+0.14) | 68.98 / 67.80 (+1.18) | 79.84 / 79.25 (+0.59) | 55.03 / 54.95 (+0.08) | 63.57 / 63.19 (+0.38) | +0.32 |

Cells beyond 2 SE: 0 of 72. Each cell reads new / old (Δ).
- Old maps: run_small_acc.py / run_main_acc.py. The release maps here are the edgeai-lab uploads, with tiles bit-identical (release-maps 0d6e56b).
- Old FOCUS states: the edgeai-lab Qwen3 FOCUS states are this part's files (results/paper_eval/focus_hf, sha256 equal).
- Known difference: the execution (fake on H200 vs native on the RTX PRO 6000).

## Inputs and checks

- **Preflight** (CPU, 2026-10-15T12:14:08Z): every input equals its registered record, and flipquant is clean at 3db0072. Result: passed.
  - FOCUS states: 5fa877acad86... (1.7B), e4689e64c66f... (8B).
  - GPTQ‡ codes files: bb50de2383f7... and 335a69b142c7...; their code digests equal the main-table records.
  - Release maps: the release summary's map_sha256.
  - IF4 / Zou: the main-table rows' installed-weight sha256.
  - Each input was hashed again before its run.
- **Every run passed part X's check_record:**
  - complete, with every example (21,848 per row), 0-shot, batch 16, the main-table revision;
  - lm-eval 0.4.11, and every dataset load pinned;
  - equal sample digests within the model;
  - native coverage and the kernel families from build_V;
  - the GPTQ‡ codes_sha256 equal to the main-table record, and the IF4 / Zou installed-weight sha256 equal to the main-table rows';
  - the FOCUS state and map paths.
- **Interval:** X2's runs, the first START to the last END: 2026-10-15T12:26:51Z to 2026-10-15T15:36:32Z (20 runs). See Timings and part Y amendment 2 for where it falls among part Y's rows.

## Timings

Per run: model load, each task's wall time and the whole process, in seconds; plus the queue START (UTC). Simulated rows are marked †.

| model | method | load | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | total | start |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | BF16 | 6 | 16 | 135 | 11 | 31 | 17 | 41 | 262 | 2026-10-15T12:26:51Z |
| Qwen3-1.7B | NVFP4 | 4 | 17 | 132 | 10 | 32 | 17 | 40 | 256 | 2026-10-15T12:31:18Z |
| Qwen3-1.7B | FourOverSix | 5 | 16 | 131 | 11 | 32 | 16 | 41 | 255 | 2026-10-15T12:35:40Z |
| Qwen3-1.7B | FOCUS | 5 | 16 | 129 | 10 | 31 | 17 | 44 | 257 | 2026-10-15T12:40:02Z |
| Qwen3-1.7B | GPTQ‡ | 6 | 16 | 129 | 10 | 30 | 16 | 47 | 259 | 2026-10-15T12:44:25Z |
| Qwen3-1.7B | IF4† | 9 | 75 | 805 | 51 | 163 | 87 | 126 | 1320 | 2026-10-15T12:48:51Z |
| Qwen3-1.7B | MixFP4 (Zou)† | 8 | 74 | 792 | 50 | 161 | 85 | 123 | 1298 | 2026-10-15T13:10:57Z |
| Qwen3-1.7B | FlipQuant 8x64 | 5 | 16 | 131 | 11 | 31 | 17 | 40 | 256 | 2026-10-15T13:32:41Z |
| Qwen3-1.7B | FlipQuant 16x64 | 5 | 17 | 129 | 11 | 30 | 18 | 40 | 254 | 2026-10-15T13:37:03Z |
| Qwen3-1.7B | FlipQuant 256x64 | 5 | 16 | 127 | 11 | 30 | 17 | 41 | 252 | 2026-10-15T13:41:24Z |
| Qwen3-8B | BF16 | 19 | 20 | 270 | 13 | 39 | 22 | 55 | 443 | 2026-10-15T13:45:42Z |
| Qwen3-8B | NVFP4 | 9 | 19 | 165 | 12 | 36 | 20 | 44 | 308 | 2026-10-15T13:53:11Z |
| Qwen3-8B | FourOverSix | 12 | 18 | 164 | 12 | 36 | 20 | 43 | 310 | 2026-10-15T13:58:26Z |
| Qwen3-8B | FOCUS | 11 | 18 | 165 | 14 | 37 | 21 | 49 | 318 | 2026-10-15T14:03:41Z |
| Qwen3-8B | GPTQ‡ | 15 | 19 | 166 | 12 | 36 | 20 | 57 | 330 | 2026-10-15T14:09:06Z |
| Qwen3-8B | IF4† | 34 | 103 | 1304 | 67 | 219 | 117 | 185 | 2033 | 2026-10-15T14:14:43Z |
| Qwen3-8B | MixFP4 (Zou)† | 35 | 99 | 1214 | 63 | 207 | 111 | 175 | 1906 | 2026-10-15T14:48:42Z |
| Qwen3-8B | FlipQuant 8x64 | 13 | 19 | 166 | 12 | 37 | 19 | 44 | 314 | 2026-10-15T15:20:35Z |
| Qwen3-8B | FlipQuant 16x64 | 13 | 19 | 164 | 12 | 36 | 20 | 44 | 311 | 2026-10-15T15:25:55Z |
| Qwen3-8B | FlipQuant 256x64 | 13 | 18 | 166 | 12 | 36 | 20 | 45 | 314 | 2026-10-15T15:31:12Z |

Sum of the process times: 3.13 h. Wall time, from the first START to the last END: 2026-10-15T12:26:51Z to 2026-10-15T15:36:32Z.
