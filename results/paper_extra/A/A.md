# Experiment A: IF4 (Cook et al.) and MixFP4 (Zou et al.) per-block selection, coarsened to hardware tiles

One fake (c) simulator for every row: the listed weights with FourOverSix per-token activations (NVFP4 activations for the NVFP4 row only). FlipQuant (ours; the TM-OPT+TC maps) here is simulated (fake (c)) and so differs slightly from the native main-table numbers. ΔNLL in nats per token (= Δ log PPL), paired over windows, ± 2 SE.

## A1 checks (a1_check.json)

- IF4 against the official fouroversix reference (dadfad69): choice equal on every module: True; dequantized values bitwise equal: True.
- IF4's FP candidate vs MixFP4 (Zou et al.)'s E2M1 candidate: 1189004 of 356515840 elements differ (BF16; operation order), so each rule has its own E2M1 base: e2m1 for IF4, e2m1z for MixFP4 (Zou et al.).
- MixFP4 (Zou et al.) against the repo's quant_nvif4: 64019 of 356515840 elements differ.
- MixFP4 (Zou et al.)'s E1M2 candidate against the repo's E0M3 alpha = 1 candidate: 0 elements differ.
- The tile rule at 1x16 equals the per-block rule, every rule and module: True.

## Llama-3.1-8B

### Primary (a): degradation when coarsened, R@g − R@1x16

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | +0.0123 ± 0.0020 * | +0.0112 ± 0.0019 * |
| IF4 (Cook et al.) | 16x64 | +0.0120 ± 0.0019 * | +0.0135 ± 0.0022 * |
| IF4 (Cook et al.) | 256x64 | +0.0124 ± 0.0019 * | +0.0130 ± 0.0018 * |
| MixFP4 (Zou et al.) | 8x64 | +0.0133 ± 0.0019 * | +0.0124 ± 0.0026 * |
| MixFP4 (Zou et al.) | 16x64 | +0.0113 ± 0.0019 * | +0.0124 ± 0.0026 * |
| MixFP4 (Zou et al.) | 256x64 | +0.0117 ± 0.0019 * | +0.0112 ± 0.0023 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | +0.0056 ± 0.0019 * | +0.0084 ± 0.0024 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | +0.0098 ± 0.0019 * | +0.0095 ± 0.0042 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | +0.0082 ± 0.0021 * | +0.0076 ± 0.0025 * |

### Primary (b): gain over the rule's own base, R@g − base

Base: each rule's own E2M1 candidate everywhere with FourOverSix activations (e2m1 for IF4, e2m1z for MixFP4 (Zou et al.): NVFP4 weights); FourOverSix for MixFP4 (Zou et al.) + FourOverSix. The same FourOverSix activations everywhere.

| rule | base | g | WikiText-2 | C4 |
|---|---|---|---:|---:|
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 1x16 | -0.0087 ± 0.0022 * | -0.0109 ± 0.0023 * |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 8x64 | +0.0036 ± 0.0026 * | +0.0003 ± 0.0018 |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 16x64 | +0.0033 ± 0.0026 * | +0.0026 ± 0.0020 * |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 256x64 | +0.0037 ± 0.0025 * | +0.0021 ± 0.0017 * |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 1x16 | -0.0105 ± 0.0021 * | -0.0109 ± 0.0029 * |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 8x64 | +0.0028 ± 0.0022 * | +0.0014 ± 0.0019 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 16x64 | +0.0008 ± 0.0024 | +0.0015 ± 0.0018 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 256x64 | +0.0012 ± 0.0024 | +0.0002 ± 0.0019 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 1x16 | -0.0070 ± 0.0020 * | -0.0056 ± 0.0021 * |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 8x64 | -0.0015 ± 0.0022 | +0.0029 ± 0.0016 * |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 16x64 | +0.0027 ± 0.0021 * | +0.0039 ± 0.0031 * |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 256x64 | +0.0012 ± 0.0020 | +0.0021 ± 0.0015 * |

### Primary (c): FlipQuant (ours; TM-OPT+TC maps) against the rule at the same tile, tc@g − R@g

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | -0.0205 ± 0.0025 * | -0.0212 ± 0.0033 * |
| IF4 (Cook et al.) | 16x64 | -0.0205 ± 0.0027 * | -0.0234 ± 0.0037 * |
| IF4 (Cook et al.) | 256x64 | -0.0181 ± 0.0024 * | -0.0179 ± 0.0024 * |
| MixFP4 (Zou et al.) | 8x64 | -0.0206 ± 0.0023 * | -0.0222 ± 0.0037 * |
| MixFP4 (Zou et al.) | 16x64 | -0.0189 ± 0.0024 * | -0.0221 ± 0.0033 * |
| MixFP4 (Zou et al.) | 256x64 | -0.0164 ± 0.0024 * | -0.0158 ± 0.0025 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | -0.0126 ± 0.0023 * | -0.0170 ± 0.0032 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | -0.0171 ± 0.0023 * | -0.0179 ± 0.0048 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | -0.0127 ± 0.0021 * | -0.0110 ± 0.0022 * |

`*` = |Δ| > 2 SE.

### Primary (d): how much of the 1x16 gain over FourOverSix survives at the tile

gain_g = NLL(FourOverSix) − NLL(R@g), ± 2 SE (positive = better than FourOverSix); retained = gain_g / gain_1x16 with a 95 % paired bootstrap interval over windows. No fraction where R@1x16 is not better than FourOverSix by 2 SE.

| rule | corpus | gain 1x16 | g | gain_g | retained [95 %] |
|---|---|---:|---|---:|---:|
| IF4 (Cook et al.) | WikiText-2 | +0.0058 ± 0.0022 | 8x64 | -0.0064 ± 0.0026 | -111 % [-225, -53] |
| IF4 (Cook et al.) | WikiText-2 | +0.0058 ± 0.0022 | 16x64 | -0.0062 ± 0.0025 | -106 % [-220, -50] |
| IF4 (Cook et al.) | WikiText-2 | +0.0058 ± 0.0022 | 256x64 | -0.0066 ± 0.0025 | -113 % [-229, -55] |
| IF4 (Cook et al.) | C4 | +0.0041 ± 0.0018 | 8x64 | -0.0071 ± 0.0016 | -172 % [-333, -102] |
| IF4 (Cook et al.) | C4 | +0.0041 ± 0.0018 | 16x64 | -0.0094 ± 0.0022 | -229 % [-449, -137] |
| IF4 (Cook et al.) | C4 | +0.0041 ± 0.0018 | 256x64 | -0.0089 ± 0.0017 | -217 % [-416, -131] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0068 ± 0.0022 | 8x64 | -0.0065 ± 0.0025 | -96 % [-177, -48] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0068 ± 0.0022 | 16x64 | -0.0045 ± 0.0025 | -66 % [-136, -25] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0068 ± 0.0022 | 256x64 | -0.0049 ± 0.0026 | -73 % [-148, -29] |
| MixFP4 (Zou et al.) | C4 | +0.0043 ± 0.0023 | 8x64 | -0.0081 ± 0.0019 | -188 % [-413, -110] |
| MixFP4 (Zou et al.) | C4 | +0.0043 ± 0.0023 | 16x64 | -0.0081 ± 0.0018 | -188 % [-400, -110] |
| MixFP4 (Zou et al.) | C4 | +0.0043 ± 0.0023 | 256x64 | -0.0069 ± 0.0018 | -160 % [-361, -90] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0070 ± 0.0020 | 8x64 | +0.0015 ± 0.0022 | 21 % [-11, 45] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0070 ± 0.0020 | 16x64 | -0.0027 ± 0.0021 | -39 % [-85, -8] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0070 ± 0.0020 | 256x64 | -0.0012 ± 0.0020 | -17 % [-53, 9] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0056 ± 0.0021 | 8x64 | -0.0029 ± 0.0016 | -51 % [-100, -21] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0056 ± 0.0021 | 16x64 | -0.0039 ± 0.0031 | -70 % [-130, -27] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0056 ± 0.0021 | 256x64 | -0.0021 ± 0.0015 | -37 % [-74, -11] |

### Mechanism: within-tile mixing of each rule's 1x16 choices

Tiles whose 16-blocks disagree at 1x16 (some prefer E2M1, some the uniform grid), % of tiles; in brackets, the mean minority share inside those tiles, %.

| rule | tile | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) | 8x64 | 100.0 (37.3) | 100.0 (37.7) | 100.0 (38.1) | 100.0 (38.4) | 100.0 (36.7) | 100.0 (37.1) | 100.0 (37.2) | 100.0 (37.5) |
| IF4 (Cook et al.) | 16x64 | 100.0 (37.8) | 100.0 (38.4) | 100.0 (38.9) | 100.0 (39.2) | 100.0 (37.2) | 100.0 (37.6) | 100.0 (37.7) | 100.0 (38.0) |
| IF4 (Cook et al.) | 256x64 | 100.0 (38.0) | 100.0 (38.7) | 100.0 (39.6) | 100.0 (39.5) | 100.0 (37.3) | 100.0 (37.7) | 100.0 (37.8) | 100.0 (38.2) |
| MixFP4 (Zou et al.) | 8x64 | 100.0 (37.2) | 100.0 (37.6) | 100.0 (38.0) | 100.0 (38.3) | 100.0 (36.6) | 100.0 (37.0) | 100.0 (37.1) | 100.0 (37.3) |
| MixFP4 (Zou et al.) | 16x64 | 100.0 (37.7) | 100.0 (38.3) | 100.0 (38.7) | 100.0 (39.1) | 100.0 (37.1) | 100.0 (37.5) | 100.0 (37.6) | 100.0 (37.9) |
| MixFP4 (Zou et al.) | 256x64 | 100.0 (37.8) | 100.0 (38.6) | 100.0 (39.5) | 100.0 (39.4) | 100.0 (37.2) | 100.0 (37.6) | 100.0 (37.7) | 100.0 (38.0) |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | 100.0 (42.3) | 100.0 (42.1) | 100.0 (42.0) | 100.0 (42.4) | 100.0 (42.2) | 100.0 (42.3) | 100.0 (42.4) | 100.0 (42.4) |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | 100.0 (44.1) | 100.0 (43.9) | 100.0 (43.7) | 100.0 (44.2) | 100.0 (43.9) | 100.0 (44.1) | 100.0 (44.2) | 100.0 (44.2) |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | 100.0 (46.3) | 100.0 (46.3) | 100.0 (46.7) | 100.0 (47.0) | 100.0 (45.7) | 100.0 (46.1) | 100.0 (46.3) | 100.0 (46.5) |

### Every policy: PPL; ΔNLL vs FourOverSix and vs NVFP4 (paper row); installed-weight squared error

| policy | uniform share | PPL WikiText-2 | PPL C4 | ΔNLL vs FourOverSix, WikiText-2 | C4 | vs NVFP4, WikiText-2 | C4 | weight error vs FourOverSix | vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | 6.2403 | 8.9579 | -0.0977 ± 0.0071 | -0.0913 ± 0.0155 | -0.1048 ± 0.0076 | -0.1030 ± 0.0172 | -100.0 % | -100.0 % |
| NVFP4 | — | 6.9296 | 9.9302 | +0.0071 ± 0.0022 | +0.0118 ± 0.0024 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +19.2 % | +0.0 % |
| FourOverSix | — | 6.8807 | 9.8141 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.0071 ± 0.0022 | -0.0118 ± 0.0024 | +0.0 % | -16.1 % |
| NVFP4 weights + FourOverSix act. | — | 6.9002 | 9.8809 | +0.0028 ± 0.0021 | +0.0068 ± 0.0015 | -0.0042 ± 0.0018 | -0.0050 ± 0.0015 | +19.2 % | +0.0 % |
| NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | — | 6.9063 | 9.8795 | +0.0037 ± 0.0023 | +0.0066 ± 0.0017 | -0.0034 ± 0.0016 | -0.0051 ± 0.0019 | +19.2 % | +0.0 % |
| FlipQuant (ours) 8x64 | 2.27 % | 6.7846 | 9.6767 | -0.0141 ± 0.0019 | -0.0141 ± 0.0028 | -0.0211 ± 0.0023 | -0.0259 ± 0.0045 | +0.1 % | -16.1 % |
| FlipQuant (ours) 16x64 | 2.96 % | 6.7825 | 9.6778 | -0.0144 ± 0.0021 | -0.0140 ± 0.0028 | -0.0215 ± 0.0023 | -0.0257 ± 0.0045 | +0.1 % | -16.1 % |
| FlipQuant (ours) 256x64 | 8.30 % | 6.8022 | 9.7264 | -0.0115 ± 0.0018 | -0.0090 ± 0.0018 | -0.0186 ± 0.0022 | -0.0207 ± 0.0034 | +0.2 % | -16.0 % |
| IF4 (Cook et al.) 1x16 | 61.98 % | 6.8407 | 9.7739 | -0.0058 ± 0.0022 | -0.0041 ± 0.0018 | -0.0129 ± 0.0022 | -0.0159 ± 0.0032 | -17.7 % | -31.0 % |
| IF4 (Cook et al.) 8x64 | 91.32 % | 6.9252 | 9.8837 | +0.0064 ± 0.0026 | +0.0071 ± 0.0016 | -0.0006 ± 0.0026 | -0.0047 ± 0.0024 | +0.3 % | -15.8 % |
| IF4 (Cook et al.) 16x64 | 96.16 % | 6.9233 | 9.9070 | +0.0062 ± 0.0025 | +0.0094 ± 0.0022 | -0.0009 ± 0.0025 | -0.0023 ± 0.0023 | +0.8 % | -15.4 % |
| IF4 (Cook et al.) 256x64 | 99.06 % | 6.9261 | 9.9020 | +0.0066 ± 0.0025 | +0.0089 ± 0.0017 | -0.0005 ± 0.0025 | -0.0028 ± 0.0023 | +1.2 % | -15.1 % |
| MixFP4 (Zou et al.) 1x16 | 62.11 % | 6.8340 | 9.7720 | -0.0068 ± 0.0022 | -0.0043 ± 0.0023 | -0.0139 ± 0.0022 | -0.0161 ± 0.0033 | -17.6 % | -30.9 % |
| MixFP4 (Zou et al.) 8x64 | 91.40 % | 6.9258 | 9.8938 | +0.0065 ± 0.0025 | +0.0081 ± 0.0019 | -0.0005 ± 0.0024 | -0.0037 ± 0.0020 | +0.3 % | -15.8 % |
| MixFP4 (Zou et al.) 16x64 | 96.20 % | 6.9119 | 9.8939 | +0.0045 ± 0.0025 | +0.0081 ± 0.0018 | -0.0026 ± 0.0024 | -0.0037 ± 0.0024 | +0.8 % | -15.4 % |
| MixFP4 (Zou et al.) 256x64 | 99.06 % | 6.9148 | 9.8817 | +0.0049 ± 0.0026 | +0.0069 ± 0.0018 | -0.0021 ± 0.0026 | -0.0049 ± 0.0021 | +1.3 % | -15.1 % |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 53.54 % | 6.8324 | 9.7595 | -0.0070 ± 0.0020 | -0.0056 ± 0.0021 | -0.0141 ± 0.0023 | -0.0173 ± 0.0038 | -20.2 % | -33.0 % |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 48.80 % | 6.8705 | 9.8422 | -0.0015 ± 0.0022 | +0.0029 ± 0.0016 | -0.0086 ± 0.0022 | -0.0089 ± 0.0023 | -3.8 % | -19.3 % |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 47.67 % | 6.8995 | 9.8523 | +0.0027 ± 0.0021 | +0.0039 ± 0.0031 | -0.0043 ± 0.0023 | -0.0079 ± 0.0022 | -2.7 % | -18.3 % |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 40.67 % | 6.8890 | 9.8343 | +0.0012 ± 0.0020 | +0.0021 ± 0.0015 | -0.0059 ± 0.0024 | -0.0097 ± 0.0022 | -0.6 % | -16.6 % |

### Uniform-format share of the weights by projection, %

| policy | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) 1x16 | 61.98 | 60.89 | 59.95 | 60.03 | 62.73 | 62.30 | 62.17 | 61.83 |
| IF4 (Cook et al.) 8x64 | 91.32 | 86.20 | 80.59 | 84.44 | 93.70 | 91.50 | 92.71 | 91.79 |
| IF4 (Cook et al.) 16x64 | 96.16 | 90.86 | 84.50 | 89.10 | 97.70 | 96.44 | 97.62 | 96.83 |
| IF4 (Cook et al.) 256x64 | 99.06 | 93.89 | 86.79 | 94.26 | 99.30 | 99.97 | 99.99 | 99.84 |
| MixFP4 (Zou et al.) 1x16 | 62.11 | 61.02 | 60.09 | 60.17 | 62.84 | 62.43 | 62.30 | 61.99 |
| MixFP4 (Zou et al.) 8x64 | 91.40 | 86.24 | 80.65 | 84.54 | 93.76 | 91.58 | 92.75 | 91.91 |
| MixFP4 (Zou et al.) 16x64 | 96.20 | 90.88 | 84.47 | 89.12 | 97.72 | 96.50 | 97.64 | 96.91 |
| MixFP4 (Zou et al.) 256x64 | 99.06 | 93.87 | 86.79 | 94.18 | 99.31 | 99.97 | 99.99 | 99.85 |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 53.54 | 52.51 | 51.68 | 51.65 | 54.25 | 53.87 | 53.69 | 53.41 |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 48.80 | 44.29 | 40.35 | 40.00 | 53.42 | 50.00 | 49.44 | 48.18 |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 47.67 | 42.34 | 38.09 | 37.08 | 54.11 | 48.95 | 48.43 | 46.76 |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 40.67 | 30.95 | 18.93 | 18.66 | 63.83 | 41.42 | 41.73 | 38.13 |
| FlipQuant (ours) 8x64 | 2.27 | 2.72 | 2.74 | 3.66 | 3.19 | 1.89 | 2.18 | 2.21 |
| FlipQuant (ours) 16x64 | 2.96 | 3.46 | 3.56 | 4.78 | 4.13 | 2.46 | 2.87 | 2.90 |
| FlipQuant (ours) 256x64 | 8.30 | 9.22 | 9.53 | 12.85 | 11.09 | 6.93 | 8.23 | 8.28 |

Blocks with a zero (underflowed) E4M3 scale, which become all-zero blocks: 0 in every arm.

### Paper table (WikiText-2): `A_table_llama8b_wiki.tex`

```latex
% Llama-3.1-8B, WikiText-2 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 6.84 & 6.93 & 6.92 & 6.93 \\
MixFP4 (Zou et al.) & 6.83 & 6.93 & 6.91 & 6.91 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 6.83 & 6.87 & 6.90 & 6.89 \\
FlipQuant (ours) & -- & 6.78 & 6.78 & 6.80 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 6.93, NVFP4 weights + FourOverSix act. 6.90, FourOverSix 6.88, BF16 6.24} \\
\bottomrule
\end{tabular}
```

### Paper table (C4): `A_table_llama8b_c4.tex`

```latex
% Llama-3.1-8B, C4 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 9.77 & 9.88 & 9.91 & 9.90 \\
MixFP4 (Zou et al.) & 9.77 & 9.89 & 9.89 & 9.88 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 9.76 & 9.84 & 9.85 & 9.83 \\
FlipQuant (ours) & -- & 9.68 & 9.68 & 9.73 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 9.93, NVFP4 weights + FourOverSix act. 9.88, FourOverSix 9.81, BF16 8.96} \\
\bottomrule
\end{tabular}
```

Re-run references equal to the Parts 2-3 fake (c) records, window by window: {'bf16': True, 'nvfp4': True, 'fo6': True, 'tc-8x64': True, 'tc-16x64': True, 'tc-256x64': True}

