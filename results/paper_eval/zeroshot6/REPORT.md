# Zero-shot accuracy: six tasks, six models, ten rows (part X)

flipquant origin/main `3db0072` `evaluation.downstream` (fresh detached worktree, unmodified; the llama3.1-8b-instruct registry entry added at run time by `zs6_fq.py`, deviation 1), lm-eval 0.4.11, 0-shot, the paper's settings (batch 16, Qwen3.8-27B 8; per-token activations; datasets pinned, LAMBADA at 900124bf). Native rows on the RTX PRO 6000 (build_V, `--kernel-set auto`); IF4 and MixFP4 (Zou) simulated with their own rules (†), as in tab:main-ppl. Every cell is a new run. Step 0 (code identity against NVFP4-RaZeR): STEP0.md. PROTOCOL.md part X.

Accuracy in %: acc_norm for PIQA, HellaSwag, ARC-e and ARC-c; acc for WinoGrande and LAMBADA. Avg. = the unweighted mean of the six. The paired columns pool every question of the six tasks (21,848 per row): difference in points (row-only correct / reference-only correct, exact McNemar p), * = p < 0.05 (acc_panel_tables.py's formula).

## Summary

Avg. of the six tasks (%) per model, with the pooled paired difference from FourOverSix in points (* = exact McNemar p < 0.05; the full table has the counts and p). Mean = the unweighted mean of the models' Avg.

| method | Llama-1B-Ins | Mistral-7B-Ins | Llama-8B-Ins | Nemotron-9B | Phi-4 | Qwen3.8-27B | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 59.92 (+4.17*) | 76.10 (+0.56*) | 73.71 (+1.57*) | 72.35 (+1.38*) | 73.53 (+0.91*) | 73.96 (+0.52*) | 71.60 |
| NVFP4 | 56.43 (-0.04) | 75.63 (+0.10) | 71.85 (-0.11) | 71.49 (+0.23) | 72.43 (+0.05) | 73.88 (+0.11) | 70.28 |
| FourOverSix | 56.24 | 75.43 | 71.74 | 70.89 | 72.39 | 73.70 | 70.07 |
| FOCUS | 57.40 (+1.48*) | 75.16 (-0.46*) | 72.33 (+0.47*) | 71.61 (+0.45*) | 73.14 (+0.59*) | 74.30 (+0.26) | 70.66 |
| GPTQ‡ | 56.54 (+0.18) | 75.83 (-0.00) | 71.85 (-0.11) | 70.86 (+0.01) | 72.61 (+0.13) | 74.02 (+0.13) | 70.29 |
| IF4† | 57.09 (+0.77*) | 75.55 (+0.03) | 72.18 (+0.34) | 70.96 (-0.05) | 72.73 (+0.16) | 74.76 (+0.74*) | 70.55 |
| MixFP4 (Zou)† | 57.34 (+0.98*) | 75.77 (+0.23) | 72.03 (+0.17) | 71.04 (+0.11) | 72.73 (+0.18) | 73.97 (+0.38*) | 70.48 |
| FlipQuant 8x64 | 57.92 (+1.89*) | 75.52 (+0.05) | 72.58 (+0.65*) | 71.06 (+0.13) | 72.59 (+0.11) | 74.06 (+0.19) | 70.62 |
| FlipQuant 16x64 | 57.25 (+1.45*) | 75.77 (+0.30*) | 72.58 (+0.59*) | 70.74 (-0.10) | 72.58 (+0.22) | 73.91 (+0.29*) | 70.47 |
| FlipQuant 256x64 | 57.72 (+1.54*) | 75.40 (+0.09) | 72.29 (+0.31) | 70.75 (-0.20) | 72.55 (+0.17) | 73.93 (+0.30*) | 70.44 |

Significant pooled differences from FourOverSix over the 6 complete models (up / down): BF16 6 / 0; NVFP4 0 / 0; FOCUS 4 / 1; GPTQ‡ 0 / 0; IF4 2 / 0; MixFP4 (Zou) 2 / 0; FlipQuant 8x64 2 / 0; FlipQuant 16x64 4 / 0; FlipQuant 256x64 2 / 0.

## Full table

| model | method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. | vs FourOverSix | vs BF16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Llama-3.2-1B-Instruct | BF16 | 75.14 | 61.67 | 61.48 | 63.80 | 37.97 | 59.46 | 59.92 | +4.17* (1596/685, p=3.5e-83) | — |
| Llama-3.2-1B-Instruct | NVFP4 | 71.65 | 57.72 | 59.91 | 59.97 | 36.26 | 53.10 | 56.43 | -0.04 (1153/1162, p=0.87) | -4.21* (759/1679, p=3.3e-79) |
| Llama-3.2-1B-Instruct | FourOverSix | 71.38 | 57.74 | 58.17 | 60.19 | 36.35 | 53.64 | 56.24 | — | -4.17* (685/1596, p=3.5e-83) |
| Llama-3.2-1B-Instruct | FOCUS | 72.58 | 59.57 | 59.12 | 60.86 | 37.03 | 55.23 | 57.40 | +1.48* (1398/1074, p=7.7e-11) | -2.69* (733/1320, p=9e-39) |
| Llama-3.2-1B-Instruct | GPTQ‡ | 71.06 | 57.62 | 59.91 | 59.89 | 36.26 | 54.49 | 56.54 | +0.18 (1262/1222, p=0.43) | -3.99* (720/1591, p=5.5e-75) |
| Llama-3.2-1B-Instruct | IF4† | 71.93 | 58.31 | 60.77 | 60.52 | 36.18 | 54.86 | 57.09 | +0.77* (1322/1153, p=0.00073) | -3.40* (712/1454, p=3.3e-58) |
| Llama-3.2-1B-Instruct | MixFP4 (Zou)† | 74.21 | 58.50 | 60.30 | 60.35 | 35.84 | 54.82 | 57.34 | +0.98* (1325/1110, p=1.4e-05) | -3.19* (777/1473, p=2.2e-49) |
| Llama-3.2-1B-Instruct | FlipQuant 8x64 | 72.52 | 59.59 | 59.27 | 62.25 | 37.80 | 56.10 | 57.92 | +1.89* (1272/858, p=2.8e-19) | -2.27* (736/1233, p=2.7e-29) |
| Llama-3.2-1B-Instruct | FlipQuant 16x64 | 72.85 | 59.31 | 58.09 | 61.15 | 36.35 | 55.75 | 57.25 | +1.45* (1214/898, p=6.6e-12) | -2.72* (704/1299, p=8.8e-41) |
| Llama-3.2-1B-Instruct | FlipQuant 256x64 | 73.07 | 59.01 | 61.01 | 60.82 | 36.35 | 56.08 | 57.72 | +1.54* (1215/879, p=2.2e-13) | -2.63* (763/1338, p=2.1e-36) |
| Mistral-7B-Instruct-v0.3 | BF16 | 83.51 | 83.32 | 74.59 | 81.73 | 59.90 | 73.53 | 76.10 | +0.56* (545/423, p=9.9e-05) | — |
| Mistral-7B-Instruct-v0.3 | NVFP4 | 82.86 | 82.73 | 74.03 | 81.78 | 59.13 | 73.24 | 75.63 | +0.10 (562/541, p=0.55) | -0.46* (462/563, p=0.0018) |
| Mistral-7B-Instruct-v0.3 | FourOverSix | 82.75 | 82.86 | 74.82 | 80.64 | 58.36 | 73.12 | 75.43 | — | -0.56* (423/545, p=9.9e-05) |
| Mistral-7B-Instruct-v0.3 | FOCUS | 82.70 | 82.39 | 74.90 | 81.82 | 57.34 | 71.78 | 75.16 | -0.46* (639/739, p=0.0076) | -1.02* (493/715, p=1.8e-10) |
| Mistral-7B-Instruct-v0.3 | GPTQ‡ | 83.46 | 82.40 | 75.14 | 81.86 | 59.22 | 72.91 | 75.83 | -0.00 (612/613, p=1) | -0.56* (468/591, p=0.00017) |
| Mistral-7B-Instruct-v0.3 | IF4† | 83.19 | 82.90 | 75.37 | 80.98 | 58.11 | 72.77 | 75.55 | +0.03 (552/546, p=0.88) | -0.53* (412/528, p=0.00017) |
| Mistral-7B-Instruct-v0.3 | MixFP4 (Zou)† | 83.24 | 83.05 | 75.30 | 81.44 | 58.53 | 73.04 | 75.77 | +0.23 (593/542, p=0.14) | -0.32* (432/503, p=0.022) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 | 82.37 | 82.71 | 75.45 | 80.77 | 58.19 | 73.61 | 75.52 | +0.05 (454/442, p=0.71) | -0.50* (426/536, p=0.00044) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 | 82.86 | 82.94 | 74.66 | 80.64 | 59.56 | 73.96 | 75.77 | +0.30* (490/425, p=0.034) | -0.26 (442/499, p=0.068) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 | 83.24 | 82.71 | 73.32 | 81.27 | 58.11 | 73.76 | 75.40 | +0.09 (447/427, p=0.52) | -0.47* (429/531, p=0.0011) |
| Llama-3.1-8B-Instruct | BF16 | 81.34 | 79.58 | 73.80 | 79.71 | 55.72 | 72.13 | 73.71 | +1.57* (847/503, p=6.2e-21) | — |
| Llama-3.1-8B-Instruct | NVFP4 | 80.09 | 77.71 | 72.06 | 77.69 | 52.30 | 71.22 | 71.85 | -0.11 (717/740, p=0.56) | -1.68* (510/877, p=4.8e-23) |
| Llama-3.1-8B-Instruct | FourOverSix | 80.14 | 78.04 | 71.74 | 77.02 | 52.05 | 71.45 | 71.74 | — | -1.57* (503/847, p=6.2e-21) |
| Llama-3.1-8B-Instruct | FOCUS | 80.41 | 78.47 | 71.74 | 77.53 | 54.01 | 71.84 | 72.33 | +0.47* (778/675, p=0.0074) | -1.10* (499/740, p=8e-12) |
| Llama-3.1-8B-Instruct | GPTQ‡ | 79.71 | 77.81 | 72.06 | 77.78 | 52.73 | 71.03 | 71.85 | -0.11 (787/810, p=0.58) | -1.68* (497/864, p=1.8e-23) |
| Llama-3.1-8B-Instruct | IF4† | 79.65 | 78.79 | 72.77 | 77.36 | 53.67 | 70.85 | 72.18 | +0.34 (794/719, p=0.057) | -1.23* (479/748, p=1.6e-14) |
| Llama-3.1-8B-Instruct | MixFP4 (Zou)† | 79.11 | 78.61 | 72.85 | 77.86 | 53.24 | 70.52 | 72.03 | +0.17 (761/723, p=0.34) | -1.40* (467/773, p=3.2e-18) |
| Llama-3.1-8B-Instruct | FlipQuant 8x64 | 80.41 | 78.60 | 73.24 | 78.83 | 52.73 | 71.69 | 72.58 | +0.65* (680/537, p=4.6e-05) | -0.92* (498/699, p=6.9e-09) |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 | 79.60 | 78.56 | 73.64 | 78.96 | 53.24 | 71.49 | 72.58 | +0.59* (699/571, p=0.00036) | -0.99* (498/714, p=6e-10) |
| Llama-3.1-8B-Instruct | FlipQuant 256x64 | 80.20 | 78.65 | 71.98 | 78.20 | 54.27 | 70.44 | 72.29 | +0.31 (660/593, p=0.062) | -1.27* (461/738, p=1.2e-15) |
| Nemotron-Nano-9B-v2 | BF16 | 80.85 | 77.49 | 73.01 | 79.84 | 56.14 | 66.78 | 72.35 | +1.38* (683/381, p=1.5e-20) | — |
| Nemotron-Nano-9B-v2 | NVFP4 | 80.52 | 76.46 | 72.93 | 79.29 | 55.20 | 64.51 | 71.49 | +0.23 (610/560, p=0.15) | -1.15* (400/652, p=7.5e-15) |
| Nemotron-Nano-9B-v2 | FourOverSix | 79.98 | 76.55 | 71.51 | 79.04 | 53.92 | 64.31 | 70.89 | — | -1.38* (381/683, p=1.5e-20) |
| Nemotron-Nano-9B-v2 | FOCUS | 80.85 | 76.82 | 73.88 | 79.55 | 54.01 | 64.54 | 71.61 | +0.45* (610/512, p=0.0038) | -0.93* (403/607, p=1.5e-10) |
| Nemotron-Nano-9B-v2 | GPTQ‡ | 79.16 | 76.15 | 72.30 | 79.00 | 53.16 | 65.42 | 70.86 | +0.01 (602/600, p=0.98) | -1.37* (390/690, p=5.3e-20) |
| Nemotron-Nano-9B-v2 | IF4† | 79.87 | 76.57 | 72.45 | 78.62 | 54.27 | 63.98 | 70.96 | -0.05 (593/604, p=0.77) | -1.43* (340/653, p=1.8e-23) |
| Nemotron-Nano-9B-v2 | MixFP4 (Zou)† | 80.41 | 76.94 | 72.77 | 78.83 | 53.58 | 63.71 | 71.04 | +0.11 (597/574, p=0.52) | -1.28* (372/651, p=2.2e-18) |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 80.47 | 76.52 | 72.77 | 78.83 | 52.99 | 64.76 | 71.06 | +0.13 (490/461, p=0.36) | -1.25* (364/637, p=5.1e-18) |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 80.79 | 76.69 | 71.74 | 78.58 | 52.90 | 63.73 | 70.74 | -0.10 (455/476, p=0.51) | -1.48* (339/662, p=9.8e-25) |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 80.36 | 76.70 | 72.53 | 78.20 | 53.41 | 63.28 | 70.75 | -0.20 (452/496, p=0.16) | -1.58* (342/688, p=1.9e-27) |
| Phi-4 | BF16 | 81.12 | 81.95 | 76.95 | 72.77 | 56.14 | 72.27 | 73.53 | +0.91* (589/390, p=2.2e-10) | — |
| Phi-4 | NVFP4 | 80.85 | 80.86 | 73.64 | 73.23 | 54.10 | 71.88 | 72.43 | +0.05 (589/579, p=0.79) | -0.87* (438/627, p=7.7e-09) |
| Phi-4 | FourOverSix | 80.69 | 81.11 | 74.27 | 71.89 | 54.86 | 71.55 | 72.39 | — | -0.91* (390/589, p=2.2e-10) |
| Phi-4 | FOCUS | 80.52 | 81.48 | 76.32 | 72.81 | 55.38 | 72.33 | 73.14 | +0.59* (625/497, p=0.00015) | -0.32* (422/493, p=0.021) |
| Phi-4 | GPTQ‡ | 80.63 | 80.91 | 73.09 | 72.98 | 56.06 | 72.02 | 72.61 | +0.13 (605/577, p=0.43) | -0.78* (409/580, p=6e-08) |
| Phi-4 | IF4† | 81.01 | 81.15 | 74.82 | 72.52 | 55.38 | 71.49 | 72.73 | +0.16 (599/564, p=0.32) | -0.75* (358/522, p=3.6e-08) |
| Phi-4 | MixFP4 (Zou)† | 81.23 | 81.14 | 73.09 | 73.74 | 55.89 | 71.28 | 72.73 | +0.18 (592/552, p=0.25) | -0.73* (377/536, p=1.6e-07) |
| Phi-4 | FlipQuant 8x64 | 80.90 | 81.12 | 74.51 | 71.93 | 55.38 | 71.73 | 72.59 | +0.11 (466/442, p=0.45) | -0.80* (381/556, p=1.2e-08) |
| Phi-4 | FlipQuant 16x64 | 81.01 | 81.35 | 74.66 | 72.01 | 54.69 | 71.76 | 72.58 | +0.22 (476/429, p=0.13) | -0.70* (393/545, p=7.8e-07) |
| Phi-4 | FlipQuant 256x64 | 81.01 | 81.19 | 74.82 | 71.51 | 54.69 | 72.07 | 72.55 | +0.17 (487/450, p=0.24) | -0.74* (390/552, p=1.5e-07) |
| Qwen3.8-27B | BF16 | 81.50 | 82.86 | 75.93 | 73.11 | 58.36 | 72.02 | 73.96 | +0.52* (529/416, p=0.00026) | — |
| Qwen3.8-27B | NVFP4 | 80.90 | 82.39 | 73.88 | 76.22 | 59.73 | 70.17 | 73.88 | +0.11 (573/549, p=0.49) | -0.41* (469/558, p=0.006) |
| Qwen3.8-27B | FourOverSix | 81.07 | 82.35 | 75.14 | 74.12 | 58.96 | 70.56 | 73.70 | — | -0.52* (416/529, p=0.00026) |
| Qwen3.8-27B | FOCUS | 81.56 | 82.19 | 75.37 | 76.68 | 59.56 | 70.42 | 74.30 | +0.26 (584/527, p=0.093) | -0.26 (505/561, p=0.092) |
| Qwen3.8-27B | GPTQ‡ | 80.63 | 81.99 | 74.90 | 75.21 | 60.07 | 71.30 | 74.02 | +0.13 (603/574, p=0.41) | -0.38* (447/531, p=0.0079) |
| Qwen3.8-27B | IF4† | 80.85 | 82.32 | 74.74 | 78.28 | 60.75 | 71.61 | 74.76 | +0.74* (700/538, p=4.6e-06) | +0.22 (556/507, p=0.14) |
| Qwen3.8-27B | MixFP4 (Zou)† | 80.63 | 82.48 | 75.45 | 75.04 | 58.53 | 71.67 | 73.97 | +0.38* (631/548, p=0.017) | -0.14 (467/497, p=0.35) |
| Qwen3.8-27B | FlipQuant 8x64 | 81.28 | 82.29 | 76.16 | 75.00 | 58.87 | 70.75 | 74.06 | +0.19 (486/445, p=0.19) | -0.33* (444/516, p=0.022) |
| Qwen3.8-27B | FlipQuant 16x64 | 81.77 | 82.50 | 74.66 | 75.08 | 58.36 | 71.07 | 73.91 | +0.29* (484/420, p=0.036) | -0.22 (432/481, p=0.11) |
| Qwen3.8-27B | FlipQuant 256x64 | 81.12 | 82.41 | 74.74 | 75.29 | 58.70 | 71.32 | 73.93 | +0.30* (487/421, p=0.031) | -0.22 (451/498, p=0.14) |

Every row of a model saw the same samples (equal sample_digest per task; the driver checks it). Across models: PIQA equal, HellaSwag equal, WinoGrande equal, ARC-e equal, ARC-c equal, LAMBADA equal.

## LaTeX

```latex
\midrule\multicolumn{8}{l}{\textit{Llama-3.2-1B-Instruct}} \\
BF16 & 75.14 & 61.67 & 61.48 & 63.80 & 37.97 & 59.46 & 59.92 \\
NVFP4 & 71.65 & 57.72 & 59.91 & 59.97 & 36.26 & 53.10 & 56.43 \\
FourOverSix & 71.38 & 57.74 & 58.17 & 60.19 & 36.35 & 53.64 & 56.24 \\
FOCUS & 72.58 & 59.57 & 59.12 & 60.86 & 37.03 & 55.23 & 57.40 \\
GPTQ$^{\ddagger}$ & 71.06 & 57.62 & 59.91 & 59.89 & 36.26 & 54.49 & 56.54 \\
IF4$^{\dagger}$ & 71.93 & 58.31 & 60.77 & 60.52 & 36.18 & 54.86 & 57.09 \\
MixFP4$^{\dagger}$ & 74.21 & 58.50 & 60.30 & 60.35 & 35.84 & 54.82 & 57.34 \\
\method{} $8{\times}64$ & 72.52 & 59.59 & 59.27 & 62.25 & 37.80 & 56.10 & 57.92 \\
\method{} $16{\times}64$ & 72.85 & 59.31 & 58.09 & 61.15 & 36.35 & 55.75 & 57.25 \\
\method{} $256{\times}64$ & 73.07 & 59.01 & 61.01 & 60.82 & 36.35 & 56.08 & 57.72 \\
\midrule\multicolumn{8}{l}{\textit{Mistral-7B-Instruct-v0.3}} \\
BF16 & 83.51 & 83.32 & 74.59 & 81.73 & 59.90 & 73.53 & 76.10 \\
NVFP4 & 82.86 & 82.73 & 74.03 & 81.78 & 59.13 & 73.24 & 75.63 \\
FourOverSix & 82.75 & 82.86 & 74.82 & 80.64 & 58.36 & 73.12 & 75.43 \\
FOCUS & 82.70 & 82.39 & 74.90 & 81.82 & 57.34 & 71.78 & 75.16 \\
GPTQ$^{\ddagger}$ & 83.46 & 82.40 & 75.14 & 81.86 & 59.22 & 72.91 & 75.83 \\
IF4$^{\dagger}$ & 83.19 & 82.90 & 75.37 & 80.98 & 58.11 & 72.77 & 75.55 \\
MixFP4$^{\dagger}$ & 83.24 & 83.05 & 75.30 & 81.44 & 58.53 & 73.04 & 75.77 \\
\method{} $8{\times}64$ & 82.37 & 82.71 & 75.45 & 80.77 & 58.19 & 73.61 & 75.52 \\
\method{} $16{\times}64$ & 82.86 & 82.94 & 74.66 & 80.64 & 59.56 & 73.96 & 75.77 \\
\method{} $256{\times}64$ & 83.24 & 82.71 & 73.32 & 81.27 & 58.11 & 73.76 & 75.40 \\
\midrule\multicolumn{8}{l}{\textit{Llama-3.1-8B-Instruct}} \\
BF16 & 81.34 & 79.58 & 73.80 & 79.71 & 55.72 & 72.13 & 73.71 \\
NVFP4 & 80.09 & 77.71 & 72.06 & 77.69 & 52.30 & 71.22 & 71.85 \\
FourOverSix & 80.14 & 78.04 & 71.74 & 77.02 & 52.05 & 71.45 & 71.74 \\
FOCUS & 80.41 & 78.47 & 71.74 & 77.53 & 54.01 & 71.84 & 72.33 \\
GPTQ$^{\ddagger}$ & 79.71 & 77.81 & 72.06 & 77.78 & 52.73 & 71.03 & 71.85 \\
IF4$^{\dagger}$ & 79.65 & 78.79 & 72.77 & 77.36 & 53.67 & 70.85 & 72.18 \\
MixFP4$^{\dagger}$ & 79.11 & 78.61 & 72.85 & 77.86 & 53.24 & 70.52 & 72.03 \\
\method{} $8{\times}64$ & 80.41 & 78.60 & 73.24 & 78.83 & 52.73 & 71.69 & 72.58 \\
\method{} $16{\times}64$ & 79.60 & 78.56 & 73.64 & 78.96 & 53.24 & 71.49 & 72.58 \\
\method{} $256{\times}64$ & 80.20 & 78.65 & 71.98 & 78.20 & 54.27 & 70.44 & 72.29 \\
\midrule\multicolumn{8}{l}{\textit{Nemotron-Nano-9B-v2}} \\
BF16 & 80.85 & 77.49 & 73.01 & 79.84 & 56.14 & 66.78 & 72.35 \\
NVFP4 & 80.52 & 76.46 & 72.93 & 79.29 & 55.20 & 64.51 & 71.49 \\
FourOverSix & 79.98 & 76.55 & 71.51 & 79.04 & 53.92 & 64.31 & 70.89 \\
FOCUS & 80.85 & 76.82 & 73.88 & 79.55 & 54.01 & 64.54 & 71.61 \\
GPTQ$^{\ddagger}$ & 79.16 & 76.15 & 72.30 & 79.00 & 53.16 & 65.42 & 70.86 \\
IF4$^{\dagger}$ & 79.87 & 76.57 & 72.45 & 78.62 & 54.27 & 63.98 & 70.96 \\
MixFP4$^{\dagger}$ & 80.41 & 76.94 & 72.77 & 78.83 & 53.58 & 63.71 & 71.04 \\
\method{} $8{\times}64$ & 80.47 & 76.52 & 72.77 & 78.83 & 52.99 & 64.76 & 71.06 \\
\method{} $16{\times}64$ & 80.79 & 76.69 & 71.74 & 78.58 & 52.90 & 63.73 & 70.74 \\
\method{} $256{\times}64$ & 80.36 & 76.70 & 72.53 & 78.20 & 53.41 & 63.28 & 70.75 \\
\midrule\multicolumn{8}{l}{\textit{Phi-4}} \\
BF16 & 81.12 & 81.95 & 76.95 & 72.77 & 56.14 & 72.27 & 73.53 \\
NVFP4 & 80.85 & 80.86 & 73.64 & 73.23 & 54.10 & 71.88 & 72.43 \\
FourOverSix & 80.69 & 81.11 & 74.27 & 71.89 & 54.86 & 71.55 & 72.39 \\
FOCUS & 80.52 & 81.48 & 76.32 & 72.81 & 55.38 & 72.33 & 73.14 \\
GPTQ$^{\ddagger}$ & 80.63 & 80.91 & 73.09 & 72.98 & 56.06 & 72.02 & 72.61 \\
IF4$^{\dagger}$ & 81.01 & 81.15 & 74.82 & 72.52 & 55.38 & 71.49 & 72.73 \\
MixFP4$^{\dagger}$ & 81.23 & 81.14 & 73.09 & 73.74 & 55.89 & 71.28 & 72.73 \\
\method{} $8{\times}64$ & 80.90 & 81.12 & 74.51 & 71.93 & 55.38 & 71.73 & 72.59 \\
\method{} $16{\times}64$ & 81.01 & 81.35 & 74.66 & 72.01 & 54.69 & 71.76 & 72.58 \\
\method{} $256{\times}64$ & 81.01 & 81.19 & 74.82 & 71.51 & 54.69 & 72.07 & 72.55 \\
\midrule\multicolumn{8}{l}{\textit{Qwen3.8-27B}} \\
BF16 & 81.50 & 82.86 & 75.93 & 73.11 & 58.36 & 72.02 & 73.96 \\
NVFP4 & 80.90 & 82.39 & 73.88 & 76.22 & 59.73 & 70.17 & 73.88 \\
FourOverSix & 81.07 & 82.35 & 75.14 & 74.12 & 58.96 & 70.56 & 73.70 \\
FOCUS & 81.56 & 82.19 & 75.37 & 76.68 & 59.56 & 70.42 & 74.30 \\
GPTQ$^{\ddagger}$ & 80.63 & 81.99 & 74.90 & 75.21 & 60.07 & 71.30 & 74.02 \\
IF4$^{\dagger}$ & 80.85 & 82.32 & 74.74 & 78.28 & 60.75 & 71.61 & 74.76 \\
MixFP4$^{\dagger}$ & 80.63 & 82.48 & 75.45 & 75.04 & 58.53 & 71.67 & 73.97 \\
\method{} $8{\times}64$ & 81.28 & 82.29 & 76.16 & 75.00 & 58.87 & 70.75 & 74.06 \\
\method{} $16{\times}64$ & 81.77 & 82.50 & 74.66 & 75.08 & 58.36 & 71.07 & 73.91 \\
\method{} $256{\times}64$ & 81.12 & 82.41 & 74.74 & 75.29 & 58.70 & 71.32 & 73.93 \\
```

The columns: method, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, LAMBADA, Avg.; values rounded once from the per-example records.

## Old vs new: the co-author's acc_panel (flipquant main docs/results/acc_panel/acc_panel.md)

Old: fake quant on H200 (scripts/run_main_acc.py, run_small_acc.py), only the table's rounded percentages (no per-example records here). New: this run (native rows on the RTX PRO 6000). Δ = new − old in points; 2 SE = 2 sqrt(p1 q1 / n + p2 q2 / n) (binomial, unpaired); ✗ = |Δ| > 2 SE. The old Llama-3.1-8B row is the base model and is not compared.

| model | method | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | Avg. Δ |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | BF16 | 83.51 / 83.62 (-0.11) | 83.32 / 83.37 (-0.05) | 74.59 / 74.66 (-0.07) | 81.73 / 81.69 (+0.04) | 59.90 / 60.15 (-0.25) | 73.53 / 73.51 (+0.02) | -0.07 |
| Mistral-7B-Instruct-v0.3 | FourOverSix | 82.75 / 82.48 (+0.27) | 82.86 / 82.82 (+0.04) | 74.82 / 73.88 (+0.94) | 80.64 / 80.81 (-0.17) | 58.36 / 58.36 (+0.00) | 73.12 / 73.37 (-0.25) | +0.14 |
| Mistral-7B-Instruct-v0.3 | FOCUS | 82.70 / 82.75 (-0.05) | 82.39 / 82.26 (+0.13) | 74.90 / 73.88 (+1.02) | 81.82 / 82.28 (-0.46) | 57.34 / 57.85 (-0.51) | 71.78 / 71.65 (+0.13) | +0.04 |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 | 82.37 / 82.48 (-0.11) | 82.71 / 82.89 (-0.18) | 75.45 / 74.51 (+0.94) | 80.77 / 80.85 (-0.08) | 58.19 / 57.85 (+0.34) | 73.61 / 73.80 (-0.19) | +0.12 |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 | 82.86 / 83.03 (-0.17) | 82.94 / 82.82 (+0.12) | 74.66 / 74.43 (+0.23) | 80.64 / 81.40 (-0.76) | 59.56 / 58.19 (+1.37) | 73.96 / 73.41 (+0.55) | +0.22 |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 | 83.24 / 82.32 (+0.92) | 82.71 / 82.61 (+0.10) | 73.32 / 74.98 (-1.66) | 81.27 / 81.48 (-0.21) | 58.11 / 58.28 (-0.17) | 73.76 / 73.61 (+0.15) | -0.14 |
| Nemotron-Nano-9B-v2 | BF16 | 80.85 / 80.69 (+0.16) | 77.49 / 77.56 (-0.07) | 73.01 / 73.56 (-0.55) | 79.84 / 79.97 (-0.13) | 56.14 / 55.80 (+0.34) | 66.78 / 66.60 (+0.18) | -0.01 |
| Nemotron-Nano-9B-v2 | FourOverSix | 79.98 / 79.98 (-0.00) | 76.55 / 76.60 (-0.05) | 71.51 / 72.06 (-0.55) | 79.04 / 78.37 (+0.67) | 53.92 / 52.13 (+1.79) | 64.31 / 64.31 (+0.00) | +0.31 |
| Nemotron-Nano-9B-v2 | FOCUS | 80.85 / 80.14 (+0.71) | 76.82 / 77.03 (-0.21) | 73.88 / 73.01 (+0.87) | 79.55 / 79.63 (-0.08) | 54.01 / 54.95 (-0.94) | 64.54 / 65.85 (-1.31) | -0.16 |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 80.47 / 80.14 (+0.33) | 76.52 / 76.71 (-0.19) | 72.77 / 73.16 (-0.39) | 78.83 / 78.45 (+0.38) | 52.99 / 52.56 (+0.43) | 64.76 / 64.39 (+0.37) | +0.15 |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 80.79 / 80.85 (-0.06) | 76.69 / 76.57 (+0.12) | 71.74 / 71.90 (-0.16) | 78.58 / 79.42 (-0.84) | 52.90 / 53.92 (-1.02) | 63.73 / 63.38 (+0.35) | -0.27 |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 80.36 / 80.36 (-0.00) | 76.70 / 76.84 (-0.14) | 72.53 / 71.27 (+1.26) | 78.20 / 78.70 (-0.50) | 53.41 / 53.16 (+0.25) | 63.28 / 63.89 (-0.61) | +0.04 |
| Phi-4 | BF16 | 81.12 / 81.28 (-0.16) | 81.95 / 81.99 (-0.04) | 76.95 / 76.40 (+0.55) | 72.77 / 72.73 (+0.04) | 56.14 / 55.97 (+0.17) | 72.27 / 72.21 (+0.06) | +0.10 |
| Phi-4 | FourOverSix | 80.69 / 80.41 (+0.28) | 81.11 / 81.33 (-0.22) | 74.27 / 75.14 (-0.87) | 71.89 / 71.46 (+0.43) | 54.86 / 54.35 (+0.51) | 71.55 / 71.16 (+0.39) | +0.09 |
| Phi-4 | FOCUS | 80.52 / 80.69 (-0.17) | 81.48 / 81.49 (-0.01) | 76.32 / 75.61 (+0.71) | 72.81 / 72.47 (+0.34) | 55.38 / 55.20 (+0.18) | 72.33 / 72.23 (+0.10) | +0.19 |
| Phi-4 | FlipQuant 8x64 | 80.90 / 81.01 (-0.11) | 81.12 / 81.37 (-0.25) | 74.51 / 74.51 (-0.00) | 71.93 / 71.51 (+0.42) | 55.38 / 55.03 (+0.35) | 71.73 / 71.67 (+0.06) | +0.08 |
| Phi-4 | FlipQuant 16x64 | 81.01 / 80.52 (+0.49) | 81.35 / 81.32 (+0.03) | 74.66 / 73.95 (+0.71) | 72.01 / 72.56 (-0.55) | 54.69 / 55.20 (-0.51) | 71.76 / 71.82 (-0.06) | +0.02 |
| Phi-4 | FlipQuant 256x64 | 81.01 / 81.01 (+0.00) | 81.19 / 81.44 (-0.25) | 74.82 / 73.72 (+1.10) | 71.51 / 71.72 (-0.21) | 54.69 / 55.63 (-0.94) | 72.07 / 71.92 (+0.15) | -0.02 |
| Qwen3.8-27B | BF16 | 81.50 / 81.61 (-0.11) | 82.86 / 82.86 (+0.00) | 75.93 / 76.01 (-0.08) | 73.11 / 73.02 (+0.09) | 58.36 / 58.62 (-0.26) | 72.02 / 72.06 (-0.04) | -0.07 |
| Qwen3.8-27B | FourOverSix | 81.07 / 81.23 (-0.16) | 82.35 / 82.44 (-0.09) | 75.14 / 74.74 (+0.40) | 74.12 / 72.98 (+1.14) | 58.96 / 58.36 (+0.60) | 70.56 / 70.79 (-0.23) | +0.28 |
| Qwen3.8-27B | FOCUS | 81.56 / 81.56 (-0.00) | 82.19 / 82.35 (-0.16) | 75.37 / 75.85 (-0.48) | 76.68 / 76.60 (+0.08) | 59.56 / 59.04 (+0.52) | 70.42 / 70.35 (+0.07) | +0.01 |
| Qwen3.8-27B | FlipQuant 8x64 | 81.28 / 80.85 (+0.43) | 82.29 / 82.46 (-0.17) | 76.16 / 75.06 (+1.10) | 75.00 / 74.45 (+0.55) | 58.87 / 59.04 (-0.17) | 70.75 / 71.26 (-0.51) | +0.21 |
| Qwen3.8-27B | FlipQuant 16x64 | 81.77 / 81.01 (+0.76) | 82.50 / 82.32 (+0.18) | 74.66 / 74.51 (+0.15) | 75.08 / 75.51 (-0.43) | 58.36 / 59.47 (-1.11) | 71.07 / 70.93 (+0.14) | -0.05 |
| Qwen3.8-27B | FlipQuant 256x64 | 81.12 / 81.07 (+0.05) | 82.41 / 82.28 (+0.13) | 74.74 / 75.69 (-0.95) | 75.29 / 74.66 (+0.63) | 58.70 / 58.53 (+0.17) | 71.32 / 70.70 (+0.62) | +0.11 |

Cells beyond 2 SE: 0 of 144. Each cell reads new / old (Δ).

What the old runs used (from flipquant main's scripts):
- run_small_acc.py (Mistral, Phi-4) downloads the edgeai-lab release maps and FOCUS states. The uploaded maps' tiles are bit-identical to the release maps here (release-maps 0d6e56b), and the FOCUS states for Mistral and Phi-4 are part K's, uploaded unchanged (focus_hf).
- run_main_acc.py (Nemotron, Qwen3.8-27B) reads flipquant's repo maps (`maps/<model>/`). Their tiles equal the release maps here for Nemotron-Nano-9B-v2 (yes), Qwen3.8-27B (yes); its FOCUS states come from `--focus MODEL=PATH`, not recorded here.
- The repo maps of Mistral and Phi-4 are older paper-setting maps with other tiles (Mistral-7B-Instruct-v0.3 equal: no, Phi-4 equal: no; `RUN/zeroshot6/old_panel_maps.json`). The old Mistral / Phi-4 rows used them only if acc_panel_tables.py took those rows from runs/main_acc rather than runs/small_acc (it reads both).
- Known difference between old and new: the execution (fake on H200 vs native on the RTX PRO 6000). Not recorded for the old runs: the Mistral / Phi-4 revisions (main leaves them unpinned, so they resolved at download time) and the Nemotron / Qwen3.8-27B FOCUS states.

## Native vs simulated (fake) cross-check: Mistral-7B-Instruct-v0.3

The same flags with `--mode fake`. Per task: accuracy native / fake, the paired difference (native-only / fake-only correct, exact McNemar p), the share of questions with the same correctness, and the per-choice log-likelihood difference native vs fake (mean |Δ| and max |Δ|, nats), next to native vs BF16 (mean |Δ|) for scale. lm-eval sums each continuation's token log-probabilities from BF16 logits, so a choice's LL carries BF16 resolution (about 0.5 nats at |LL| near 100).

| row | task | native | fake | Δ (points) | native-only / fake-only, p | same correctness | mean / max abs ΔLL, native vs fake | mean abs ΔLL, native vs BF16 |
|---|---|---:|---:|---:|---|---:|---|---:|
| FourOverSix | PIQA | 82.75 | 82.43 | +0.33 | 39 / 33, p=0.56 | 96.08 % | 1.995 / 15.00 | 2.239 |
| FourOverSix | HellaSwag | 82.86 | 82.81 | +0.05 | 124 / 119, p=0.8 | 97.58 % | 1.502 / 17.00 | 1.990 |
| FourOverSix | WinoGrande | 74.82 | 74.43 | +0.39 | 63 / 58, p=0.72 | 90.45 % | 0.540 / 5.00 | 0.648 |
| FourOverSix | ARC-e | 80.64 | 80.98 | -0.34 | 51 / 59, p=0.5 | 95.37 % | 0.999 / 10.25 | 1.154 |
| FourOverSix | ARC-c | 58.36 | 57.42 | +0.94 | 46 / 35, p=0.27 | 93.09 % | 1.151 / 9.38 | 1.350 |
| FourOverSix | LAMBADA | 73.12 | 73.26 | -0.14 | 122 / 129, p=0.7 | 95.13 % | 0.159 / 3.00 | 0.185 |
| FourOverSix | pooled | 75.43 | 75.22 | +0.05 | 445 / 433, p=0.71 | | | |
| FlipQuant 16x64 | PIQA | 82.86 | 82.86 | +0.00 | 33 / 33, p=1 | 96.41 % | 1.737 / 12.00 | 2.421 |
| FlipQuant 16x64 | HellaSwag | 82.94 | 82.98 | -0.04 | 105 / 109, p=0.84 | 97.87 % | 1.428 / 14.00 | 2.208 |
| FlipQuant 16x64 | WinoGrande | 74.66 | 74.43 | +0.24 | 52 / 49, p=0.84 | 92.03 % | 0.517 / 4.50 | 0.617 |
| FlipQuant 16x64 | ARC-e | 80.64 | 81.48 | -0.84 | 40 / 60, p=0.057 | 95.79 % | 0.918 / 7.25 | 1.356 |
| FlipQuant 16x64 | ARC-c | 59.56 | 57.85 | +1.71 | 51 / 31, p=0.035 | 93.00 % | 1.057 / 7.25 | 1.566 |
| FlipQuant 16x64 | LAMBADA | 73.96 | 74.07 | -0.12 | 136 / 142, p=0.76 | 94.61 % | 0.154 / 2.93 | 0.189 |
| FlipQuant 16x64 | pooled | 75.77 | 75.61 | -0.03 | 417 / 424, p=0.84 | | | |

## Deviations and notes

1. **Registry (deviation 1).** flipquant main has no llama3.1-8b-instruct. Its jobs ran through `zs6_fq.py`, which adds paper-sm120-runs 7b472b0's ModelSpec (Meta's Llama-3.1-8B-Instruct @ 0e9e39f2) to `models.registry.MODELS` at run time. Nothing in the flipquant checkout changed (the driver checks a clean 3db0072).
2. **Revisions (deviation 2).** main leaves Mistral and Phi-4 unpinned. Every job passed the main-table revision, and the driver checked each report's source revision.
3. **Parity model (deviation 3, amendment 1).** NVFP4-RaZeR's lm-eval harness knows neither Instruct model, and base Mistral-7B-v0.3's weights are no longer in the HF cache. The step 0 parity ran on Phi-4, a panel model, at the same revision in both harnesses.
4. **Envs.** Nemotron and Qwen3.8-27B ran in n16k64-fast, the others in n16k64, as in the main table (the envs differ only in the hybrid kernels).
5. **GPTQ‡ codes.** The two Llama models' codes were regenerated by main's GPTQ in step 0 (codes_sha256 = part P's: 952850d7..., d3fe9816...); the other four were loaded from the main-table files. Every panel run's codes_sha256 equals the main-table record's.
6. **FOCUS states.** Every state matched its recorded sha256. The 1B's had none; step 0 recorded it (418006a09ce958ad...). Qwen3.8-27B's was downloaded again (part Q's file and revision, sha256 equal).
7. **Native vs fake.** The two agree in accuracy (the cross-check's pooled differences are not significant). Per choice, their log-likelihoods differ about as much as native W4A4 differs from BF16. Interpretation, not measured here: W4A4 rounding is discontinuous, so the small arithmetic differences of the simulated path (BF16 rounding of the dequantized operands) flip FP4 rounding decisions downstream.
8. **Protocol text.** The registration and amendment 1 headers carried wrong times; they were corrected to the commits' (10:33 and 10:44 UTC).

## Timings

Per run: model load, each task's wall time, and the whole process (seconds); queue START (UTC). Simulated rows are marked †.

| model | method | load | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | LAMBADA | total | start |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-1B-Instruct | BF16 | 4 | 13 | 86 | 9 | 22 | 14 | 34 | 186 | 2026-10-14T11:17:52Z |
| Llama-3.2-1B-Instruct | NVFP4 | 7 | 12 | 83 | 8 | 21 | 11 | 35 | 180 | 2026-10-14T11:21:04Z |
| Llama-3.2-1B-Instruct | FourOverSix | 6 | 12 | 84 | 8 | 22 | 12 | 34 | 181 | 2026-10-14T11:24:10Z |
| Llama-3.2-1B-Instruct | FOCUS | 6 | 12 | 84 | 8 | 20 | 12 | 35 | 180 | 2026-10-14T11:27:18Z |
| Llama-3.2-1B-Instruct | GPTQ‡ | 6 | 12 | 83 | 8 | 21 | 12 | 37 | 183 | 2026-10-14T11:30:24Z |
| Llama-3.2-1B-Instruct | IF4† | 9 | 46 | 501 | 32 | 98 | 54 | 89 | 832 | 2026-10-14T11:33:33Z |
| Llama-3.2-1B-Instruct | MixFP4 (Zou)† | 9 | 47 | 490 | 32 | 97 | 52 | 85 | 816 | 2026-10-14T11:47:31Z |
| Llama-3.2-1B-Instruct | FlipQuant 8x64 | 6 | 12 | 87 | 8 | 21 | 12 | 34 | 184 | 2026-10-14T12:01:13Z |
| Llama-3.2-1B-Instruct | FlipQuant 16x64 | 6 | 12 | 84 | 9 | 21 | 12 | 34 | 181 | 2026-10-14T12:04:23Z |
| Llama-3.2-1B-Instruct | FlipQuant 256x64 | 6 | 12 | 81 | 8 | 21 | 11 | 33 | 176 | 2026-10-14T12:07:30Z |
| Mistral-7B-Instruct-v0.3 | BF16 | 4 | 19 | 258 | 11 | 36 | 21 | 55 | 408 | 2026-10-14T12:10:31Z |
| Mistral-7B-Instruct-v0.3 | NVFP4 | 8 | 15 | 134 | 10 | 29 | 16 | 39 | 257 | 2026-10-14T12:17:25Z |
| Mistral-7B-Instruct-v0.3 | FourOverSix | 11 | 16 | 133 | 10 | 29 | 16 | 38 | 256 | 2026-10-14T12:21:48Z |
| Mistral-7B-Instruct-v0.3 | FOCUS | 11 | 15 | 134 | 10 | 31 | 16 | 39 | 261 | 2026-10-14T12:26:14Z |
| Mistral-7B-Instruct-v0.3 | GPTQ‡ | 13 | 16 | 134 | 10 | 30 | 16 | 49 | 271 | 2026-10-14T12:30:45Z |
| Mistral-7B-Instruct-v0.3 | IF4† | 35 | 93 | 1277 | 58 | 206 | 111 | 182 | 1966 | 2026-10-14T12:35:22Z |
| Mistral-7B-Instruct-v0.3 | MixFP4 (Zou)† | 36 | 90 | 1214 | 57 | 199 | 104 | 173 | 1875 | 2026-10-14T13:08:15Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 | 12 | 16 | 140 | 10 | 30 | 16 | 39 | 267 | 2026-10-14T13:39:36Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 | 13 | 16 | 134 | 12 | 31 | 17 | 38 | 264 | 2026-10-14T13:44:09Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 | 13 | 16 | 135 | 12 | 30 | 17 | 39 | 265 | 2026-10-14T13:48:39Z |
| Llama-3.1-8B-Instruct | BF16 | 6 | 20 | 263 | 12 | 37 | 21 | 60 | 425 | 2026-10-14T14:38:21Z |
| Llama-3.1-8B-Instruct | NVFP4 | 12 | 19 | 145 | 12 | 33 | 21 | 42 | 287 | 2026-10-14T14:45:32Z |
| Llama-3.1-8B-Instruct | FourOverSix | 14 | 18 | 146 | 11 | 30 | 17 | 43 | 283 | 2026-10-14T14:50:24Z |
| Llama-3.1-8B-Instruct | FOCUS | 13 | 16 | 143 | 11 | 30 | 17 | 47 | 281 | 2026-10-14T14:55:17Z |
| Llama-3.1-8B-Instruct | GPTQ‡ | 17 | 18 | 144 | 12 | 31 | 17 | 50 | 292 | 2026-10-14T15:00:07Z |
| Llama-3.1-8B-Instruct | IF4† | 40 | 94 | 1236 | 58 | 195 | 105 | 175 | 1906 | 2026-10-14T15:05:07Z |
| Llama-3.1-8B-Instruct | MixFP4 (Zou)† | 36 | 89 | 1167 | 57 | 187 | 103 | 167 | 1809 | 2026-10-14T15:36:59Z |
| Llama-3.1-8B-Instruct | FlipQuant 8x64 | 14 | 16 | 145 | 11 | 30 | 16 | 40 | 277 | 2026-10-14T16:07:14Z |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 | 15 | 17 | 142 | 11 | 31 | 17 | 41 | 277 | 2026-10-14T16:11:57Z |
| Llama-3.1-8B-Instruct | FlipQuant 256x64 | 15 | 16 | 141 | 12 | 31 | 19 | 41 | 279 | 2026-10-14T16:16:40Z |
| Nemotron-Nano-9B-v2 | BF16 | 8 | 28 | 310 | 12 | 41 | 25 | 59 | 491 | 2026-10-14T16:21:25Z |
| Nemotron-Nano-9B-v2 | NVFP4 | 13 | 23 | 191 | 13 | 35 | 20 | 45 | 344 | 2026-10-14T16:29:42Z |
| Nemotron-Nano-9B-v2 | FourOverSix | 15 | 24 | 192 | 12 | 34 | 19 | 45 | 346 | 2026-10-14T16:35:32Z |
| Nemotron-Nano-9B-v2 | FOCUS | 15 | 23 | 190 | 12 | 37 | 20 | 45 | 346 | 2026-10-14T16:41:28Z |
| Nemotron-Nano-9B-v2 | GPTQ‡ | 18 | 23 | 192 | 12 | 34 | 19 | 50 | 352 | 2026-10-14T16:47:26Z |
| Nemotron-Nano-9B-v2 | IF4† | 42 | 68 | 995 | 36 | 123 | 69 | 140 | 1476 | 2026-10-14T16:53:24Z |
| Nemotron-Nano-9B-v2 | MixFP4 (Zou)† | 42 | 64 | 964 | 35 | 115 | 65 | 136 | 1425 | 2026-10-14T17:18:06Z |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 17 | 23 | 199 | 13 | 34 | 19 | 47 | 355 | 2026-10-14T17:41:57Z |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 17 | 23 | 193 | 12 | 35 | 19 | 45 | 347 | 2026-10-14T17:47:58Z |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 17 | 22 | 193 | 12 | 33 | 18 | 46 | 345 | 2026-10-14T17:53:51Z |
| Phi-4 | BF16 | 8 | 26 | 433 | 14 | 49 | 30 | 76 | 640 | 2026-10-14T17:59:42Z |
| Phi-4 | NVFP4 | 17 | 17 | 208 | 11 | 32 | 18 | 48 | 355 | 2026-10-14T18:10:28Z |
| Phi-4 | FourOverSix | 21 | 17 | 207 | 11 | 33 | 19 | 48 | 361 | 2026-10-14T18:16:30Z |
| Phi-4 | FOCUS | 20 | 17 | 206 | 11 | 33 | 18 | 48 | 358 | 2026-10-14T18:22:44Z |
| Phi-4 | GPTQ‡ | 28 | 18 | 207 | 11 | 33 | 19 | 51 | 371 | 2026-10-14T18:28:55Z |
| Phi-4 | IF4† | 68 | 84 | 1379 | 50 | 168 | 94 | 188 | 2036 | 2026-10-14T18:35:11Z |
| Phi-4 | MixFP4 (Zou)† | 67 | 79 | 1342 | 46 | 160 | 91 | 183 | 1972 | 2026-10-14T19:09:14Z |
| Phi-4 | FlipQuant 8x64 | 25 | 18 | 215 | 11 | 33 | 18 | 49 | 373 | 2026-10-14T19:42:12Z |
| Phi-4 | FlipQuant 16x64 | 25 | 18 | 209 | 11 | 32 | 18 | 48 | 365 | 2026-10-14T19:48:31Z |
| Phi-4 | FlipQuant 256x64 | 24 | 18 | 209 | 11 | 33 | 18 | 48 | 366 | 2026-10-14T19:54:43Z |
| Qwen3.8-27B | BF16 | 13 | 69 | 930 | 44 | 142 | 77 | 137 | 1417 | 2026-10-14T20:00:55Z |
| Qwen3.8-27B | NVFP4 | 28 | 62 | 657 | 43 | 135 | 72 | 108 | 1109 | 2026-10-14T20:24:38Z |
| Qwen3.8-27B | FourOverSix | 37 | 62 | 658 | 43 | 136 | 71 | 108 | 1119 | 2026-10-14T20:43:14Z |
| Qwen3.8-27B | FOCUS | 33 | 63 | 654 | 42 | 135 | 74 | 119 | 1125 | 2026-10-14T21:02:10Z |
| Qwen3.8-27B | GPTQ‡ | 47 | 63 | 648 | 42 | 133 | 71 | 148 | 1155 | 2026-10-14T21:21:15Z |
| Qwen3.8-27B | IF4† | 118 | 359 | 4399 | 239 | 782 | 422 | 581 | 6904 | 2026-10-14T21:40:39Z |
| Qwen3.8-27B | MixFP4 (Zou)† | 119 | 347 | 4116 | 232 | 757 | 404 | 549 | 6527 | 2026-10-14T23:35:50Z |
| Qwen3.8-27B | FlipQuant 8x64 | 43 | 62 | 655 | 43 | 134 | 71 | 106 | 1119 | 2026-10-15T01:24:44Z |
| Qwen3.8-27B | FlipQuant 16x64 | 43 | 63 | 653 | 44 | 134 | 70 | 109 | 1120 | 2026-10-15T01:43:30Z |
| Qwen3.8-27B | FlipQuant 256x64 | 43 | 63 | 651 | 43 | 133 | 71 | 106 | 1114 | 2026-10-15T02:02:16Z |
| Mistral-7B-Instruct-v0.3 | FourOverSix (fake, cross-check) | 6 | 57 | 917 | 35 | 118 | 63 | 135 | 1334 | 2026-10-14T13:53:10Z |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 (fake, cross-check) | 7 | 57 | 924 | 37 | 121 | 71 | 144 | 1365 | 2026-10-14T14:15:30Z |

Sum of the process times: 14.9 h.
