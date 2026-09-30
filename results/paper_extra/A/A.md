# Experiment A: IF4 (Cook et al.) and MixFP4 (Zou et al.) per-block selection, coarsened to hardware tiles

One fake (c) simulator for every row: the listed weights with FourOverSix per-token activations (NVFP4 activations for the NVFP4 row only). FlipQuant (ours; the TM-OPT+TC maps) here is simulated (fake (c)) and so differs slightly from the native main-table numbers. ΔNLL in nats per token (= Δ log PPL), paired over windows, ± 2 SE.

## A1 checks (a1_check.json)

- IF4 against the official fouroversix reference (dadfad69): choice equal on every module: True; dequantized values bitwise equal: True.
- IF4's FP candidate vs MixFP4 (Zou et al.)'s E2M1 candidate: 1189004 of 356515840 elements differ (BF16; operation order), so each rule has its own E2M1 base: e2m1 for IF4, e2m1z for MixFP4 (Zou et al.).
- MixFP4 (Zou et al.) against the repo's quant_nvif4: 64019 of 356515840 elements differ.
- MixFP4 (Zou et al.)'s E1M2 candidate against the repo's E0M3 alpha = 1 candidate: 0 elements differ.
- The tile rule at 1x16 equals the per-block rule, every rule and module: True.
- IF4 (Cook et al.) + FourOverSix (amendment 2; if4fo6_check.json, 11 modules): FP candidate equals MixFP4 (Zou et al.) + FourOverSix's bitwise: True; INT4 candidate equals IF4's as installed: True; ties keep FP: True; the existing rules' outputs unchanged: 30 of 30.

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
| IF4 (Cook et al.) + FourOverSix | 8x64 | +0.0089 ± 0.0020 * | +0.0089 ± 0.0034 * |
| IF4 (Cook et al.) + FourOverSix | 16x64 | +0.0127 ± 0.0021 * | +0.0098 ± 0.0026 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | +0.0086 ± 0.0022 * | +0.0070 ± 0.0020 * |

### Primary (b): gain over the rule's own base, R@g − base

Base: each rule's own E2M1 candidate everywhere with FourOverSix activations (e2m1 for IF4, e2m1z for MixFP4 (Zou et al.): NVFP4 weights); FourOverSix for MixFP4 (Zou et al.) + FourOverSix and IF4 (Cook et al.) + FourOverSix. The same FourOverSix activations everywhere.

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
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 1x16 | -0.0071 ± 0.0021 * | -0.0055 ± 0.0018 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 8x64 | +0.0018 ± 0.0021 | +0.0034 ± 0.0025 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 16x64 | +0.0055 ± 0.0023 * | +0.0043 ± 0.0017 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 256x64 | +0.0014 ± 0.0020 | +0.0015 ± 0.0018 |

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
| IF4 (Cook et al.) + FourOverSix | 8x64 | -0.0158 ± 0.0022 * | -0.0175 ± 0.0042 * |
| IF4 (Cook et al.) + FourOverSix | 16x64 | -0.0199 ± 0.0026 * | -0.0183 ± 0.0036 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | -0.0129 ± 0.0022 * | -0.0105 ± 0.0023 * |

### Contrast (amendment 2): IF4 (Cook et al.) + FourOverSix − MixFP4 (Zou et al.) + FourOverSix, at the same g

The two uniform candidates (IF4's INT4 with the shared max/6 scale; Zou's E1M2 with its own max/7 scale) on the same FourOverSix E2M1 base, each with its own tie rule.

| g | WikiText-2 | C4 |
|---|---:|---:|
| 1x16 | -0.0001 ± 0.0018 | +0.0001 ± 0.0015 |
| 8x64 | +0.0033 ± 0.0017 * | +0.0005 ± 0.0023 |
| 16x64 | +0.0028 ± 0.0018 * | +0.0004 ± 0.0024 |
| 256x64 | +0.0002 ± 0.0019 | -0.0006 ± 0.0016 |

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
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0071 ± 0.0021 | 8x64 | -0.0018 ± 0.0021 | -25 % [-67, 3] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0071 ± 0.0021 | 16x64 | -0.0055 ± 0.0023 | -78 % [-141, -40] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0071 ± 0.0021 | 256x64 | -0.0014 ± 0.0020 | -20 % [-57, 7] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0055 ± 0.0018 | 8x64 | -0.0034 ± 0.0025 | -62 % [-112, -22] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0055 ± 0.0018 | 16x64 | -0.0043 ± 0.0017 | -78 % [-125, -46] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0055 ± 0.0018 | 256x64 | -0.0015 ± 0.0018 | -27 % [-71, 3] |

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
| IF4 (Cook et al.) + FourOverSix | 8x64 | 100.0 (42.4) | 100.0 (42.2) | 100.0 (42.0) | 100.0 (42.4) | 100.0 (42.2) | 100.0 (42.3) | 100.0 (42.4) | 100.0 (42.5) |
| IF4 (Cook et al.) + FourOverSix | 16x64 | 100.0 (44.1) | 100.0 (43.9) | 100.0 (43.8) | 100.0 (44.2) | 100.0 (43.9) | 100.0 (44.1) | 100.0 (44.2) | 100.0 (44.3) |
| IF4 (Cook et al.) + FourOverSix | 256x64 | 100.0 (46.3) | 100.0 (46.4) | 100.0 (46.7) | 100.0 (47.1) | 100.0 (45.7) | 100.0 (46.2) | 100.0 (46.3) | 100.0 (46.6) |

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
| IF4 (Cook et al.) + FourOverSix 1x16 | 53.46 % | 6.8319 | 9.7601 | -0.0071 ± 0.0021 | -0.0055 ± 0.0018 | -0.0142 ± 0.0023 | -0.0173 ± 0.0035 | -20.3 % | -33.1 % |
| IF4 (Cook et al.) + FourOverSix 8x64 | 48.86 % | 6.8929 | 9.8476 | +0.0018 ± 0.0021 | +0.0034 ± 0.0025 | -0.0053 ± 0.0022 | -0.0084 ± 0.0020 | -3.8 % | -19.3 % |
| IF4 (Cook et al.) + FourOverSix 16x64 | 47.76 % | 6.9189 | 9.8564 | +0.0055 ± 0.0023 | +0.0043 ± 0.0017 | -0.0015 ± 0.0022 | -0.0075 ± 0.0019 | -2.7 % | -18.4 % |
| IF4 (Cook et al.) + FourOverSix 256x64 | 41.19 % | 6.8906 | 9.8288 | +0.0014 ± 0.0020 | +0.0015 ± 0.0018 | -0.0056 ± 0.0023 | -0.0103 ± 0.0027 | -0.6 % | -16.6 % |

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
| IF4 (Cook et al.) + FourOverSix 1x16 | 53.46 | 52.45 | 51.59 | 51.56 | 54.22 | 53.80 | 53.64 | 53.30 |
| IF4 (Cook et al.) + FourOverSix 8x64 | 48.86 | 44.46 | 40.32 | 39.92 | 53.56 | 50.06 | 49.67 | 48.01 |
| IF4 (Cook et al.) + FourOverSix 16x64 | 47.76 | 42.62 | 38.01 | 37.05 | 54.44 | 49.03 | 48.75 | 46.53 |
| IF4 (Cook et al.) + FourOverSix 256x64 | 41.19 | 31.64 | 18.96 | 18.76 | 64.96 | 41.86 | 43.04 | 37.79 |
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
IF4 (Cook et al.) + FourOverSix (our variant) & 6.83 & 6.89 & 6.92 & 6.89 \\
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
IF4 (Cook et al.) + FourOverSix (our variant) & 9.76 & 9.85 & 9.86 & 9.83 \\
FlipQuant (ours) & -- & 9.68 & 9.68 & 9.73 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 9.93, NVFP4 weights + FourOverSix act. 9.88, FourOverSix 9.81, BF16 8.96} \\
\bottomrule
\end{tabular}
```

Re-run references equal to the Parts 2-3 fake (c) records, window by window: {'bf16': True, 'nvfp4': True, 'fo6': True, 'tc-8x64': True, 'tc-16x64': True, 'tc-256x64': True}

## Mistral-7B-v0.3

### Primary (a): degradation when coarsened, R@g − R@1x16

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | +0.0025 ± 0.0011 * | +0.0017 ± 0.0013 * |
| IF4 (Cook et al.) | 16x64 | +0.0040 ± 0.0012 * | +0.0023 ± 0.0012 * |
| IF4 (Cook et al.) | 256x64 | +0.0035 ± 0.0011 * | +0.0029 ± 0.0009 * |
| MixFP4 (Zou et al.) | 8x64 | +0.0041 ± 0.0011 * | +0.0025 ± 0.0009 * |
| MixFP4 (Zou et al.) | 16x64 | +0.0050 ± 0.0011 * | +0.0031 ± 0.0009 * |
| MixFP4 (Zou et al.) | 256x64 | +0.0038 ± 0.0012 * | +0.0028 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | +0.0017 ± 0.0011 * | +0.0016 ± 0.0011 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | +0.0024 ± 0.0012 * | +0.0021 ± 0.0008 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | +0.0040 ± 0.0011 * | +0.0023 ± 0.0012 * |
| IF4 (Cook et al.) + FourOverSix | 8x64 | +0.0054 ± 0.0012 * | +0.0006 ± 0.0010 |
| IF4 (Cook et al.) + FourOverSix | 16x64 | +0.0042 ± 0.0011 * | +0.0017 ± 0.0008 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | +0.0035 ± 0.0011 * | +0.0009 ± 0.0009 |

### Primary (b): gain over the rule's own base, R@g − base

Base: each rule's own E2M1 candidate everywhere with FourOverSix activations (e2m1 for IF4, e2m1z for MixFP4 (Zou et al.): NVFP4 weights); FourOverSix for MixFP4 (Zou et al.) + FourOverSix and IF4 (Cook et al.) + FourOverSix. The same FourOverSix activations everywhere.

| rule | base | g | WikiText-2 | C4 |
|---|---|---|---:|---:|
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 1x16 | -0.0055 ± 0.0012 * | -0.0031 ± 0.0011 * |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 8x64 | -0.0029 ± 0.0014 * | -0.0014 ± 0.0016 |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 16x64 | -0.0015 ± 0.0015 * | -0.0008 ± 0.0015 |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 256x64 | -0.0020 ± 0.0014 * | -0.0002 ± 0.0012 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 1x16 | -0.0057 ± 0.0014 * | -0.0033 ± 0.0011 * |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 8x64 | -0.0016 ± 0.0013 * | -0.0008 ± 0.0011 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 16x64 | -0.0008 ± 0.0014 | -0.0002 ± 0.0012 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 256x64 | -0.0020 ± 0.0013 * | -0.0004 ± 0.0012 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 1x16 | -0.0033 ± 0.0011 * | -0.0025 ± 0.0012 * |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 8x64 | -0.0016 ± 0.0009 * | -0.0009 ± 0.0009 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 16x64 | -0.0009 ± 0.0010 | -0.0004 ± 0.0010 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 256x64 | +0.0007 ± 0.0009 | -0.0001 ± 0.0008 |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 1x16 | -0.0039 ± 0.0011 * | -0.0015 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 8x64 | +0.0015 ± 0.0012 * | -0.0009 ± 0.0009 |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 16x64 | +0.0003 ± 0.0010 | +0.0002 ± 0.0010 |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 256x64 | -0.0003 ± 0.0011 | -0.0006 ± 0.0007 |

### Primary (c): FlipQuant (ours; TM-OPT+TC maps) against the rule at the same tile, tc@g − R@g

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | -0.0063 ± 0.0014 * | -0.0056 ± 0.0015 * |
| IF4 (Cook et al.) | 16x64 | -0.0078 ± 0.0013 * | -0.0061 ± 0.0012 * |
| IF4 (Cook et al.) | 256x64 | -0.0073 ± 0.0013 * | -0.0055 ± 0.0011 * |
| MixFP4 (Zou et al.) | 8x64 | -0.0076 ± 0.0012 * | -0.0065 ± 0.0012 * |
| MixFP4 (Zou et al.) | 16x64 | -0.0085 ± 0.0013 * | -0.0071 ± 0.0011 * |
| MixFP4 (Zou et al.) | 256x64 | -0.0074 ± 0.0012 * | -0.0055 ± 0.0011 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | -0.0043 ± 0.0011 * | -0.0049 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | -0.0051 ± 0.0011 * | -0.0053 ± 0.0009 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | -0.0067 ± 0.0010 * | -0.0043 ± 0.0011 * |
| IF4 (Cook et al.) + FourOverSix | 8x64 | -0.0074 ± 0.0012 * | -0.0049 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 16x64 | -0.0063 ± 0.0011 * | -0.0059 ± 0.0009 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | -0.0057 ± 0.0012 * | -0.0038 ± 0.0010 * |

### Contrast (amendment 2): IF4 (Cook et al.) + FourOverSix − MixFP4 (Zou et al.) + FourOverSix, at the same g

The two uniform candidates (IF4's INT4 with the shared max/6 scale; Zou's E1M2 with its own max/7 scale) on the same FourOverSix E2M1 base, each with its own tie rule.

| g | WikiText-2 | C4 |
|---|---:|---:|
| 1x16 | -0.0006 ± 0.0010 | +0.0009 ± 0.0009 * |
| 8x64 | +0.0032 ± 0.0012 * | -0.0000 ± 0.0008 |
| 16x64 | +0.0012 ± 0.0010 * | +0.0006 ± 0.0007 |
| 256x64 | -0.0010 ± 0.0009 * | -0.0005 ± 0.0008 |

`*` = |Δ| > 2 SE.

### Primary (d): how much of the 1x16 gain over FourOverSix survives at the tile

gain_g = NLL(FourOverSix) − NLL(R@g), ± 2 SE (positive = better than FourOverSix); retained = gain_g / gain_1x16 with a 95 % paired bootstrap interval over windows. No fraction where R@1x16 is not better than FourOverSix by 2 SE.

| rule | corpus | gain 1x16 | g | gain_g | retained [95 %] |
|---|---|---:|---|---:|---:|
| IF4 (Cook et al.) | WikiText-2 | +0.0021 ± 0.0011 | 8x64 | -0.0004 ± 0.0014 | -19 % [-142, 36] |
| IF4 (Cook et al.) | WikiText-2 | +0.0021 ± 0.0011 | 16x64 | -0.0019 ± 0.0014 | -87 % [-268, -18] |
| IF4 (Cook et al.) | WikiText-2 | +0.0021 ± 0.0011 | 256x64 | -0.0013 ± 0.0013 | -62 % [-214, -2] |
| IF4 (Cook et al.) | C4 | +0.0019 ± 0.0012 | 8x64 | +0.0002 ± 0.0019 | 9 % [-159, 86] |
| IF4 (Cook et al.) | C4 | +0.0019 ± 0.0012 | 16x64 | -0.0004 ± 0.0018 | -22 % [-237, 57] |
| IF4 (Cook et al.) | C4 | +0.0019 ± 0.0012 | 256x64 | -0.0010 ± 0.0013 | -56 % [-294, 8] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0024 ± 0.0011 | 8x64 | -0.0018 ± 0.0012 | -73 % [-199, -19] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0024 ± 0.0011 | 16x64 | -0.0026 ± 0.0013 | -107 % [-267, -43] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0024 ± 0.0011 | 256x64 | -0.0014 ± 0.0013 | -58 % [-185, -3] |
| MixFP4 (Zou et al.) | C4 | +0.0017 ± 0.0014 | 8x64 | -0.0007 ± 0.0013 | -43 % [-315, 21] |
| MixFP4 (Zou et al.) | C4 | +0.0017 ± 0.0014 | 16x64 | -0.0013 ± 0.0014 | -77 % [-447, 4] |
| MixFP4 (Zou et al.) | C4 | +0.0017 ± 0.0014 | 256x64 | -0.0011 ± 0.0014 | -63 % [-399, 13] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0033 ± 0.0011 | 8x64 | +0.0016 ± 0.0009 | 49 % [23, 79] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0033 ± 0.0011 | 16x64 | +0.0009 ± 0.0010 | 27 % [-4, 55] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0033 ± 0.0011 | 256x64 | -0.0007 ± 0.0009 | -22 % [-62, 5] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0012 | 8x64 | +0.0009 ± 0.0009 | 35 % [-1, 68] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0012 | 16x64 | +0.0004 ± 0.0010 | 17 % [-31, 46] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0012 | 256x64 | +0.0001 ± 0.0008 | 6 % [-29, 35] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0039 ± 0.0011 | 8x64 | -0.0015 ± 0.0012 | -40 % [-88, -10] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0039 ± 0.0011 | 16x64 | -0.0003 ± 0.0010 | -9 % [-43, 14] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0039 ± 0.0011 | 256x64 | +0.0003 ± 0.0011 | 8 % [-24, 33] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0015 ± 0.0010 | 8x64 | +0.0009 ± 0.0009 | 58 % [0, 134] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0015 ± 0.0010 | 16x64 | -0.0002 ± 0.0010 | -11 % [-147, 38] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0015 ± 0.0010 | 256x64 | +0.0006 ± 0.0007 | 41 % [-6, 98] |

### Mechanism: within-tile mixing of each rule's 1x16 choices

Tiles whose 16-blocks disagree at 1x16 (some prefer E2M1, some the uniform grid), % of tiles; in brackets, the mean minority share inside those tiles, %.

| rule | tile | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) | 8x64 | 100.0 (37.4) | 100.0 (37.8) | 100.0 (38.3) | 100.0 (38.5) | 100.0 (36.8) | 100.0 (37.3) | 100.0 (37.4) | 100.0 (37.6) |
| IF4 (Cook et al.) | 16x64 | 100.0 (38.0) | 100.0 (38.4) | 100.0 (39.1) | 100.0 (39.3) | 100.0 (37.2) | 100.0 (37.8) | 100.0 (37.9) | 100.0 (38.1) |
| IF4 (Cook et al.) | 256x64 | 100.0 (38.1) | 100.0 (38.8) | 100.0 (39.7) | 100.0 (39.7) | 100.0 (37.3) | 100.0 (37.9) | 100.0 (38.0) | 100.0 (38.3) |
| MixFP4 (Zou et al.) | 8x64 | 100.0 (37.3) | 100.0 (37.7) | 100.0 (38.2) | 100.0 (38.4) | 100.0 (36.7) | 100.0 (37.1) | 100.0 (37.3) | 100.0 (37.4) |
| MixFP4 (Zou et al.) | 16x64 | 100.0 (37.8) | 100.0 (38.3) | 100.0 (39.0) | 100.0 (39.2) | 100.0 (37.1) | 100.0 (37.7) | 100.0 (37.8) | 100.0 (38.0) |
| MixFP4 (Zou et al.) | 256x64 | 100.0 (38.0) | 100.0 (38.7) | 100.0 (39.6) | 100.0 (39.5) | 100.0 (37.2) | 100.0 (37.8) | 100.0 (37.9) | 100.0 (38.1) |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | 100.0 (42.4) | 100.0 (42.2) | 100.0 (42.2) | 100.0 (42.3) | 100.0 (42.2) | 100.0 (42.4) | 100.0 (42.4) | 100.0 (42.5) |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | 100.0 (44.2) | 100.0 (44.0) | 100.0 (44.0) | 100.0 (44.1) | 100.0 (43.9) | 100.0 (44.2) | 100.0 (44.2) | 100.0 (44.3) |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | 100.0 (46.4) | 100.0 (46.6) | 100.0 (47.1) | 100.0 (47.0) | 100.0 (45.7) | 100.0 (46.3) | 100.0 (46.5) | 100.0 (46.6) |
| IF4 (Cook et al.) + FourOverSix | 8x64 | 100.0 (42.4) | 100.0 (42.2) | 100.0 (42.2) | 100.0 (42.3) | 100.0 (42.2) | 100.0 (42.4) | 100.0 (42.4) | 100.0 (42.5) |
| IF4 (Cook et al.) + FourOverSix | 16x64 | 100.0 (44.2) | 100.0 (44.0) | 100.0 (44.0) | 100.0 (44.1) | 100.0 (43.9) | 100.0 (44.2) | 100.0 (44.3) | 100.0 (44.3) |
| IF4 (Cook et al.) + FourOverSix | 256x64 | 100.0 (46.5) | 100.0 (46.6) | 100.0 (47.1) | 100.0 (47.0) | 100.0 (45.7) | 100.0 (46.4) | 100.0 (46.5) | 100.0 (46.7) |

### Every policy: PPL; ΔNLL vs FourOverSix and vs NVFP4 (paper row); installed-weight squared error

| policy | uniform share | PPL WikiText-2 | PPL C4 | ΔNLL vs FourOverSix, WikiText-2 | C4 | vs NVFP4, WikiText-2 | C4 | weight error vs FourOverSix | vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | 5.3182 | 7.8306 | -0.0374 ± 0.0013 | -0.0299 ± 0.0025 | -0.0428 ± 0.0015 | -0.0332 ± 0.0021 | -100.0 % | -100.0 % |
| NVFP4 | — | 5.5508 | 8.0949 | +0.0054 ± 0.0012 | +0.0033 ± 0.0011 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +19.2 % | +0.0 % |
| FourOverSix | — | 5.5210 | 8.0681 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.0054 ± 0.0012 | -0.0033 ± 0.0011 | +0.0 % | -16.1 % |
| NVFP4 weights + FourOverSix act. | — | 5.5395 | 8.0780 | +0.0034 ± 0.0011 | +0.0012 ± 0.0009 | -0.0020 ± 0.0009 | -0.0021 ± 0.0007 | +19.2 % | +0.0 % |
| NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | — | 5.5395 | 8.0805 | +0.0033 ± 0.0012 | +0.0015 ± 0.0009 | -0.0020 ± 0.0009 | -0.0018 ± 0.0006 | +19.2 % | +0.0 % |
| FlipQuant (ours) 8x64 | 2.49 % | 5.4886 | 8.0216 | -0.0059 ± 0.0009 | -0.0058 ± 0.0009 | -0.0113 ± 0.0013 | -0.0091 ± 0.0010 | +0.1 % | -16.1 % |
| FlipQuant (ours) 16x64 | 3.29 % | 5.4882 | 8.0221 | -0.0060 ± 0.0010 | -0.0057 ± 0.0011 | -0.0113 ± 0.0014 | -0.0090 ± 0.0009 | +0.1 % | -16.1 % |
| FlipQuant (ours) 256x64 | 9.98 % | 5.4881 | 8.0325 | -0.0060 ± 0.0010 | -0.0044 ± 0.0009 | -0.0114 ± 0.0012 | -0.0077 ± 0.0010 | +0.2 % | -16.0 % |
| IF4 (Cook et al.) 1x16 | 61.83 % | 5.5092 | 8.0531 | -0.0021 ± 0.0011 | -0.0019 ± 0.0012 | -0.0075 ± 0.0013 | -0.0052 ± 0.0010 | -17.7 % | -31.0 % |
| IF4 (Cook et al.) 8x64 | 90.96 % | 5.5232 | 8.0667 | +0.0004 ± 0.0014 | -0.0002 ± 0.0019 | -0.0050 ± 0.0014 | -0.0035 ± 0.0015 | +0.5 % | -15.7 % |
| IF4 (Cook et al.) 16x64 | 96.01 % | 5.5312 | 8.0714 | +0.0019 ± 0.0014 | +0.0004 ± 0.0018 | -0.0035 ± 0.0015 | -0.0029 ± 0.0014 | +1.0 % | -15.3 % |
| IF4 (Cook et al.) 256x64 | 99.25 % | 5.5283 | 8.0765 | +0.0013 ± 0.0013 | +0.0010 ± 0.0013 | -0.0041 ± 0.0015 | -0.0023 ± 0.0013 | +1.4 % | -15.0 % |
| MixFP4 (Zou et al.) 1x16 | 61.97 % | 5.5078 | 8.0541 | -0.0024 ± 0.0011 | -0.0017 ± 0.0014 | -0.0078 ± 0.0013 | -0.0051 ± 0.0011 | -17.6 % | -30.9 % |
| MixFP4 (Zou et al.) 8x64 | 91.06 % | 5.5307 | 8.0741 | +0.0018 ± 0.0012 | +0.0007 ± 0.0013 | -0.0036 ± 0.0014 | -0.0026 ± 0.0011 | +0.5 % | -15.7 % |
| MixFP4 (Zou et al.) 16x64 | 96.05 % | 5.5351 | 8.0789 | +0.0026 ± 0.0013 | +0.0013 ± 0.0014 | -0.0028 ± 0.0014 | -0.0020 ± 0.0011 | +1.0 % | -15.3 % |
| MixFP4 (Zou et al.) 256x64 | 99.26 % | 5.5286 | 8.0769 | +0.0014 ± 0.0013 | +0.0011 ± 0.0014 | -0.0040 ± 0.0013 | -0.0022 ± 0.0013 | +1.4 % | -15.0 % |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 53.41 % | 5.5028 | 8.0482 | -0.0033 ± 0.0011 | -0.0025 ± 0.0012 | -0.0087 ± 0.0013 | -0.0058 ± 0.0010 | -20.2 % | -33.0 % |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 47.69 % | 5.5120 | 8.0612 | -0.0016 ± 0.0009 | -0.0009 ± 0.0009 | -0.0070 ± 0.0012 | -0.0042 ± 0.0012 | -3.7 % | -19.2 % |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 46.06 % | 5.5161 | 8.0648 | -0.0009 ± 0.0010 | -0.0004 ± 0.0010 | -0.0063 ± 0.0012 | -0.0037 ± 0.0010 | -2.5 % | -18.3 % |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 35.52 % | 5.5249 | 8.0669 | +0.0007 ± 0.0009 | -0.0001 ± 0.0008 | -0.0047 ± 0.0012 | -0.0035 ± 0.0012 | -0.5 % | -16.6 % |
| IF4 (Cook et al.) + FourOverSix 1x16 | 53.33 % | 5.4998 | 8.0557 | -0.0039 ± 0.0011 | -0.0015 ± 0.0010 | -0.0092 ± 0.0013 | -0.0049 ± 0.0010 | -20.2 % | -33.1 % |
| IF4 (Cook et al.) + FourOverSix 8x64 | 47.67 % | 5.5295 | 8.0610 | +0.0015 ± 0.0012 | -0.0009 ± 0.0009 | -0.0038 ± 0.0013 | -0.0042 ± 0.0010 | -3.7 % | -19.2 % |
| IF4 (Cook et al.) + FourOverSix 16x64 | 46.06 % | 5.5229 | 8.0695 | +0.0003 ± 0.0010 | +0.0002 ± 0.0010 | -0.0050 ± 0.0014 | -0.0031 ± 0.0010 | -2.5 % | -18.3 % |
| IF4 (Cook et al.) + FourOverSix 256x64 | 35.57 % | 5.5192 | 8.0630 | -0.0003 ± 0.0011 | -0.0006 ± 0.0007 | -0.0057 ± 0.0013 | -0.0039 ± 0.0010 | -0.5 % | -16.6 % |

### Uniform-format share of the weights by projection, %

| policy | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) 1x16 | 61.83 | 60.99 | 60.12 | 59.75 | 62.68 | 62.08 | 61.97 | 61.71 |
| IF4 (Cook et al.) 8x64 | 90.96 | 86.80 | 81.97 | 85.06 | 93.11 | 91.12 | 92.09 | 91.31 |
| IF4 (Cook et al.) 16x64 | 96.01 | 91.62 | 86.31 | 90.61 | 97.07 | 96.18 | 97.27 | 96.60 |
| IF4 (Cook et al.) 256x64 | 99.25 | 95.82 | 90.20 | 95.51 | 98.49 | 99.96 | 100.00 | 99.92 |
| MixFP4 (Zou et al.) 1x16 | 61.97 | 61.11 | 60.23 | 59.88 | 62.79 | 62.23 | 62.10 | 61.87 |
| MixFP4 (Zou et al.) 8x64 | 91.06 | 86.88 | 82.06 | 85.18 | 93.17 | 91.21 | 92.17 | 91.45 |
| MixFP4 (Zou et al.) 16x64 | 96.05 | 91.64 | 86.38 | 90.68 | 97.08 | 96.23 | 97.30 | 96.68 |
| MixFP4 (Zou et al.) 256x64 | 99.26 | 95.85 | 90.22 | 95.47 | 98.50 | 99.96 | 100.00 | 99.91 |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 53.41 | 52.55 | 51.74 | 51.30 | 54.21 | 53.68 | 53.52 | 53.30 |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 47.69 | 44.11 | 39.66 | 38.89 | 53.00 | 48.42 | 48.19 | 47.16 |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 46.06 | 41.88 | 36.50 | 35.26 | 53.66 | 46.72 | 46.64 | 45.30 |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 35.52 | 26.37 | 12.63 | 16.35 | 63.25 | 34.73 | 36.33 | 33.19 |
| IF4 (Cook et al.) + FourOverSix 1x16 | 53.33 | 52.48 | 51.68 | 51.23 | 54.18 | 53.59 | 53.45 | 53.20 |
| IF4 (Cook et al.) + FourOverSix 8x64 | 47.67 | 44.13 | 39.69 | 38.90 | 53.27 | 48.32 | 48.28 | 47.02 |
| IF4 (Cook et al.) + FourOverSix 16x64 | 46.06 | 41.92 | 36.62 | 35.26 | 54.09 | 46.62 | 46.80 | 45.10 |
| IF4 (Cook et al.) + FourOverSix 256x64 | 35.57 | 26.66 | 12.52 | 16.36 | 64.39 | 34.36 | 36.81 | 32.86 |
| FlipQuant (ours) 8x64 | 2.49 | 3.02 | 3.22 | 4.15 | 3.34 | 2.15 | 2.32 | 2.43 |
| FlipQuant (ours) 16x64 | 3.29 | 3.87 | 4.26 | 5.59 | 4.33 | 2.82 | 3.09 | 3.27 |
| FlipQuant (ours) 256x64 | 9.98 | 10.50 | 12.96 | 17.03 | 13.23 | 8.49 | 9.51 | 10.15 |

Blocks with a zero (underflowed) E4M3 scale, which become all-zero blocks: 0 in every arm.

### Paper table (WikiText-2): `A_table_mistral7b_wiki.tex`

```latex
% Mistral-7B-v0.3, WikiText-2 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 5.51 & 5.52 & 5.53 & 5.53 \\
MixFP4 (Zou et al.) & 5.51 & 5.53 & 5.54 & 5.53 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 5.50 & 5.51 & 5.52 & 5.52 \\
IF4 (Cook et al.) + FourOverSix (our variant) & 5.50 & 5.53 & 5.52 & 5.52 \\
FlipQuant (ours) & -- & 5.49 & 5.49 & 5.49 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 5.55, NVFP4 weights + FourOverSix act. 5.54, FourOverSix 5.52, BF16 5.32} \\
\bottomrule
\end{tabular}
```

### Paper table (C4): `A_table_mistral7b_c4.tex`

```latex
% Mistral-7B-v0.3, C4 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 8.05 & 8.07 & 8.07 & 8.08 \\
MixFP4 (Zou et al.) & 8.05 & 8.07 & 8.08 & 8.08 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 8.05 & 8.06 & 8.06 & 8.07 \\
IF4 (Cook et al.) + FourOverSix (our variant) & 8.06 & 8.06 & 8.07 & 8.06 \\
FlipQuant (ours) & -- & 8.02 & 8.02 & 8.03 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 8.09, NVFP4 weights + FourOverSix act. 8.08, FourOverSix 8.07, BF16 7.83} \\
\bottomrule
\end{tabular}
```

Re-run references equal to the Parts 2-3 fake (c) records, window by window: {'bf16': True, 'nvfp4': True, 'fo6': True, 'tc-8x64': True, 'tc-16x64': True, 'tc-256x64': True}

## Phi-4

### Primary (a): degradation when coarsened, R@g − R@1x16

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | +0.0036 ± 0.0015 * | +0.0039 ± 0.0010 * |
| IF4 (Cook et al.) | 16x64 | +0.0031 ± 0.0017 * | +0.0035 ± 0.0010 * |
| IF4 (Cook et al.) | 256x64 | +0.0041 ± 0.0017 * | +0.0038 ± 0.0009 * |
| MixFP4 (Zou et al.) | 8x64 | +0.0048 ± 0.0017 * | +0.0042 ± 0.0009 * |
| MixFP4 (Zou et al.) | 16x64 | +0.0042 ± 0.0016 * | +0.0043 ± 0.0009 * |
| MixFP4 (Zou et al.) | 256x64 | +0.0057 ± 0.0017 * | +0.0042 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | +0.0038 ± 0.0017 * | +0.0021 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | +0.0042 ± 0.0017 * | +0.0018 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | +0.0048 ± 0.0017 * | +0.0030 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 8x64 | +0.0038 ± 0.0018 * | +0.0015 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 16x64 | +0.0050 ± 0.0016 * | +0.0023 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | +0.0048 ± 0.0017 * | +0.0028 ± 0.0010 * |

### Primary (b): gain over the rule's own base, R@g − base

Base: each rule's own E2M1 candidate everywhere with FourOverSix activations (e2m1 for IF4, e2m1z for MixFP4 (Zou et al.): NVFP4 weights); FourOverSix for MixFP4 (Zou et al.) + FourOverSix and IF4 (Cook et al.) + FourOverSix. The same FourOverSix activations everywhere.

| rule | base | g | WikiText-2 | C4 |
|---|---|---|---:|---:|
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 1x16 | -0.0049 ± 0.0016 * | -0.0040 ± 0.0011 * |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 8x64 | -0.0013 ± 0.0017 | -0.0002 ± 0.0012 |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 16x64 | -0.0019 ± 0.0020 | -0.0005 ± 0.0012 |
| IF4 (Cook et al.) | NVFP4 weights + FourOverSix act. | 256x64 | -0.0009 ± 0.0019 | -0.0002 ± 0.0012 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 1x16 | -0.0043 ± 0.0016 * | -0.0043 ± 0.0011 * |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 8x64 | +0.0005 ± 0.0019 | -0.0001 ± 0.0012 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 16x64 | -0.0001 ± 0.0018 | -0.0000 ± 0.0012 |
| MixFP4 (Zou et al.) | NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | 256x64 | +0.0014 ± 0.0019 | -0.0001 ± 0.0012 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 1x16 | -0.0019 ± 0.0015 * | -0.0025 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 8x64 | +0.0019 ± 0.0017 * | -0.0004 ± 0.0010 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 16x64 | +0.0023 ± 0.0017 * | -0.0007 ± 0.0010 |
| MixFP4 (Zou et al.) + FourOverSix | FourOverSix | 256x64 | +0.0029 ± 0.0017 * | +0.0005 ± 0.0010 |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 1x16 | -0.0036 ± 0.0017 * | -0.0026 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 8x64 | +0.0002 ± 0.0016 | -0.0011 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 16x64 | +0.0014 ± 0.0017 | -0.0003 ± 0.0010 |
| IF4 (Cook et al.) + FourOverSix | FourOverSix | 256x64 | +0.0012 ± 0.0017 | +0.0002 ± 0.0011 |

### Primary (c): FlipQuant (ours; TM-OPT+TC maps) against the rule at the same tile, tc@g − R@g

| rule | g | WikiText-2 | C4 |
|---|---|---:|---:|
| IF4 (Cook et al.) | 8x64 | -0.0087 ± 0.0019 * | -0.0061 ± 0.0012 * |
| IF4 (Cook et al.) | 16x64 | -0.0081 ± 0.0019 * | -0.0048 ± 0.0012 * |
| IF4 (Cook et al.) | 256x64 | -0.0064 ± 0.0022 * | -0.0041 ± 0.0012 * |
| MixFP4 (Zou et al.) | 8x64 | -0.0114 ± 0.0021 * | -0.0064 ± 0.0011 * |
| MixFP4 (Zou et al.) | 16x64 | -0.0107 ± 0.0021 * | -0.0055 ± 0.0012 * |
| MixFP4 (Zou et al.) | 256x64 | -0.0096 ± 0.0021 * | -0.0044 ± 0.0011 * |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | -0.0096 ± 0.0017 * | -0.0045 ± 0.0010 * |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | -0.0099 ± 0.0020 * | -0.0033 ± 0.0009 * |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | -0.0079 ± 0.0020 * | -0.0034 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 8x64 | -0.0079 ± 0.0016 * | -0.0039 ± 0.0011 * |
| IF4 (Cook et al.) + FourOverSix | 16x64 | -0.0091 ± 0.0016 * | -0.0037 ± 0.0010 * |
| IF4 (Cook et al.) + FourOverSix | 256x64 | -0.0062 ± 0.0019 * | -0.0032 ± 0.0010 * |

### Contrast (amendment 2): IF4 (Cook et al.) + FourOverSix − MixFP4 (Zou et al.) + FourOverSix, at the same g

The two uniform candidates (IF4's INT4 with the shared max/6 scale; Zou's E1M2 with its own max/7 scale) on the same FourOverSix E2M1 base, each with its own tie rule.

| g | WikiText-2 | C4 |
|---|---:|---:|
| 1x16 | -0.0017 ± 0.0016 * | -0.0001 ± 0.0009 |
| 8x64 | -0.0017 ± 0.0015 * | -0.0006 ± 0.0008 |
| 16x64 | -0.0009 ± 0.0017 | +0.0004 ± 0.0009 |
| 256x64 | -0.0017 ± 0.0017 | -0.0003 ± 0.0009 |

`*` = |Δ| > 2 SE.

### Primary (d): how much of the 1x16 gain over FourOverSix survives at the tile

gain_g = NLL(FourOverSix) − NLL(R@g), ± 2 SE (positive = better than FourOverSix); retained = gain_g / gain_1x16 with a 95 % paired bootstrap interval over windows. No fraction where R@1x16 is not better than FourOverSix by 2 SE.

| rule | corpus | gain 1x16 | g | gain_g | retained [95 %] |
|---|---|---:|---|---:|---:|
| IF4 (Cook et al.) | WikiText-2 | +0.0026 ± 0.0017 | 8x64 | -0.0010 ± 0.0018 | -37 % [-237, 22] |
| IF4 (Cook et al.) | WikiText-2 | +0.0026 ± 0.0017 | 16x64 | -0.0004 ± 0.0019 | -17 % [-173, 43] |
| IF4 (Cook et al.) | WikiText-2 | +0.0026 ± 0.0017 | 256x64 | -0.0014 ± 0.0019 | -54 % [-268, 13] |
| IF4 (Cook et al.) | C4 | +0.0026 ± 0.0010 | 8x64 | -0.0012 ± 0.0012 | -46 % [-128, -3] |
| IF4 (Cook et al.) | C4 | +0.0026 ± 0.0010 | 16x64 | -0.0009 ± 0.0012 | -33 % [-112, 8] |
| IF4 (Cook et al.) | C4 | +0.0026 ± 0.0010 | 256x64 | -0.0012 ± 0.0013 | -44 % [-134, 0] |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0011 ± 0.0016 | 8x64 | -0.0037 ± 0.0019 | no fraction: the 1x16 gain is not significant |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0011 ± 0.0016 | 16x64 | -0.0031 ± 0.0019 | no fraction: the 1x16 gain is not significant |
| MixFP4 (Zou et al.) | WikiText-2 | +0.0011 ± 0.0016 | 256x64 | -0.0045 ± 0.0019 | no fraction: the 1x16 gain is not significant |
| MixFP4 (Zou et al.) | C4 | +0.0027 ± 0.0010 | 8x64 | -0.0015 ± 0.0011 | -54 % [-136, -12] |
| MixFP4 (Zou et al.) | C4 | +0.0027 ± 0.0010 | 16x64 | -0.0016 ± 0.0012 | -58 % [-150, -12] |
| MixFP4 (Zou et al.) | C4 | +0.0027 ± 0.0010 | 256x64 | -0.0015 ± 0.0012 | -53 % [-139, -9] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0019 ± 0.0015 | 8x64 | -0.0019 ± 0.0017 | -99 % [-576, -6] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0019 ± 0.0015 | 16x64 | -0.0023 ± 0.0017 | -118 % [-662, -17] |
| MixFP4 (Zou et al.) + FourOverSix | WikiText-2 | +0.0019 ± 0.0015 | 256x64 | -0.0029 ± 0.0017 | -150 % [-777, -42] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0010 | 8x64 | +0.0004 ± 0.0010 | 17 % [-31, 51] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0010 | 16x64 | +0.0007 ± 0.0010 | 28 % [-12, 62] |
| MixFP4 (Zou et al.) + FourOverSix | C4 | +0.0025 ± 0.0010 | 256x64 | -0.0005 ± 0.0010 | -20 % [-82, 15] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0036 ± 0.0017 | 8x64 | -0.0002 ± 0.0016 | -4 % [-64, 34] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0036 ± 0.0017 | 16x64 | -0.0014 ± 0.0017 | -38 % [-133, 6] |
| IF4 (Cook et al.) + FourOverSix | WikiText-2 | +0.0036 ± 0.0017 | 256x64 | -0.0012 ± 0.0017 | -34 % [-122, 10] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0026 ± 0.0010 | 8x64 | +0.0011 ± 0.0010 | 41 % [3, 74] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0026 ± 0.0010 | 16x64 | +0.0003 ± 0.0010 | 11 % [-35, 41] |
| IF4 (Cook et al.) + FourOverSix | C4 | +0.0026 ± 0.0010 | 256x64 | -0.0002 ± 0.0011 | -8 % [-68, 27] |

### Mechanism: within-tile mixing of each rule's 1x16 choices

Tiles whose 16-blocks disagree at 1x16 (some prefer E2M1, some the uniform grid), % of tiles; in brackets, the mean minority share inside those tiles, %.

| rule | tile | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) | 8x64 | 100.0 (36.6) | — | — | — | 100.0 (36.5) | — | — | 100.0 (36.9) |
| IF4 (Cook et al.) | 16x64 | 100.0 (37.1) | — | — | — | 100.0 (36.9) | — | — | 100.0 (37.4) |
| IF4 (Cook et al.) | 256x64 | 100.0 (37.2) | — | — | — | 100.0 (37.0) | — | — | 100.0 (37.5) |
| MixFP4 (Zou et al.) | 8x64 | 100.0 (36.5) | — | — | — | 100.0 (36.3) | — | — | 100.0 (36.7) |
| MixFP4 (Zou et al.) | 16x64 | 100.0 (36.9) | — | — | — | 100.0 (36.7) | — | — | 100.0 (37.2) |
| MixFP4 (Zou et al.) | 256x64 | 100.0 (37.0) | — | — | — | 100.0 (36.8) | — | — | 100.0 (37.3) |
| MixFP4 (Zou et al.) + FourOverSix | 8x64 | 100.0 (42.1) | — | — | — | 100.0 (42.0) | — | — | 100.0 (42.2) |
| MixFP4 (Zou et al.) + FourOverSix | 16x64 | 100.0 (43.8) | — | — | — | 100.0 (43.7) | — | — | 100.0 (44.0) |
| MixFP4 (Zou et al.) + FourOverSix | 256x64 | 100.0 (45.5) | — | — | — | 100.0 (45.3) | — | — | 100.0 (45.8) |
| IF4 (Cook et al.) + FourOverSix | 8x64 | 100.0 (42.2) | — | — | — | 100.0 (42.1) | — | — | 100.0 (42.3) |
| IF4 (Cook et al.) + FourOverSix | 16x64 | 100.0 (43.9) | — | — | — | 100.0 (43.8) | — | — | 100.0 (44.0) |
| IF4 (Cook et al.) + FourOverSix | 256x64 | 100.0 (45.7) | — | — | — | 100.0 (45.5) | — | — | 100.0 (46.0) |

### Every policy: PPL; ΔNLL vs FourOverSix and vs NVFP4 (paper row); installed-weight squared error

| policy | uniform share | PPL WikiText-2 | PPL C4 | ΔNLL vs FourOverSix, WikiText-2 | C4 | vs NVFP4, WikiText-2 | C4 | weight error vs FourOverSix | vs NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | 6.4615 | 10.3098 | -0.0307 ± 0.0020 | -0.0228 ± 0.0014 | -0.0362 ± 0.0026 | -0.0262 ± 0.0016 | -100.0 % | -100.0 % |
| NVFP4 | — | 6.6998 | 10.5834 | +0.0056 ± 0.0017 | +0.0034 ± 0.0010 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +19.5 % | +0.0 % |
| FourOverSix | — | 6.6627 | 10.5473 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | -0.0056 ± 0.0017 | -0.0034 ± 0.0010 | +0.0 % | -16.4 % |
| NVFP4 weights + FourOverSix act. | — | 6.6780 | 10.5618 | +0.0023 ± 0.0016 | +0.0014 ± 0.0009 | -0.0033 ± 0.0013 | -0.0020 ± 0.0007 | +19.6 % | +0.0 % |
| NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act. | — | 6.6839 | 10.5642 | +0.0032 ± 0.0015 | +0.0016 ± 0.0010 | -0.0024 ± 0.0014 | -0.0018 ± 0.0008 | +19.5 % | +0.0 % |
| FlipQuant (ours) 8x64 | 1.57 % | 6.6115 | 10.4955 | -0.0077 ± 0.0014 | -0.0049 ± 0.0009 | -0.0133 ± 0.0019 | -0.0083 ± 0.0012 | -0.0 % | -16.4 % |
| FlipQuant (ours) 16x64 | 2.16 % | 6.6118 | 10.5056 | -0.0077 ± 0.0015 | -0.0040 ± 0.0008 | -0.0132 ± 0.0020 | -0.0074 ± 0.0011 | -0.0 % | -16.4 % |
| FlipQuant (ours) 256x64 | 6.35 % | 6.6293 | 10.5162 | -0.0050 ± 0.0015 | -0.0030 ± 0.0008 | -0.0106 ± 0.0020 | -0.0064 ± 0.0010 | -0.0 % | -16.4 % |
| IF4 (Cook et al.) 1x16 | 62.84 % | 6.6451 | 10.5195 | -0.0026 ± 0.0017 | -0.0026 ± 0.0010 | -0.0082 ± 0.0018 | -0.0061 ± 0.0011 | -18.3 % | -31.6 % |
| IF4 (Cook et al.) 8x64 | 94.35 % | 6.6692 | 10.5602 | +0.0010 ± 0.0018 | +0.0012 ± 0.0012 | -0.0046 ± 0.0019 | -0.0022 ± 0.0012 | -1.1 % | -17.3 % |
| IF4 (Cook et al.) 16x64 | 98.34 % | 6.6656 | 10.5565 | +0.0004 ± 0.0019 | +0.0009 ± 0.0012 | -0.0051 ± 0.0021 | -0.0025 ± 0.0011 | -0.8 % | -17.0 % |
| IF4 (Cook et al.) 256x64 | 99.65 % | 6.6721 | 10.5597 | +0.0014 ± 0.0019 | +0.0012 ± 0.0013 | -0.0041 ± 0.0021 | -0.0022 ± 0.0012 | -0.7 % | -16.9 % |
| MixFP4 (Zou et al.) 1x16 | 63.01 % | 6.6551 | 10.5184 | -0.0011 ± 0.0016 | -0.0027 ± 0.0010 | -0.0067 ± 0.0018 | -0.0062 ± 0.0011 | -18.2 % | -31.5 % |
| MixFP4 (Zou et al.) 8x64 | 94.50 % | 6.6872 | 10.5630 | +0.0037 ± 0.0019 | +0.0015 ± 0.0011 | -0.0019 ± 0.0021 | -0.0019 ± 0.0012 | -1.2 % | -17.3 % |
| MixFP4 (Zou et al.) 16x64 | 98.39 % | 6.6830 | 10.5641 | +0.0031 ± 0.0019 | +0.0016 ± 0.0012 | -0.0025 ± 0.0021 | -0.0018 ± 0.0013 | -0.9 % | -17.1 % |
| MixFP4 (Zou et al.) 256x64 | 99.65 % | 6.6930 | 10.5627 | +0.0045 ± 0.0019 | +0.0015 ± 0.0012 | -0.0010 ± 0.0022 | -0.0020 ± 0.0012 | -0.7 % | -17.0 % |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 54.44 % | 6.6499 | 10.5211 | -0.0019 ± 0.0015 | -0.0025 ± 0.0010 | -0.0075 ± 0.0018 | -0.0059 ± 0.0012 | -20.7 % | -33.7 % |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 54.67 % | 6.6753 | 10.5427 | +0.0019 ± 0.0017 | -0.0004 ± 0.0010 | -0.0037 ± 0.0020 | -0.0038 ± 0.0011 | -4.4 % | -20.1 % |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 55.95 % | 6.6778 | 10.5399 | +0.0023 ± 0.0017 | -0.0007 ± 0.0010 | -0.0033 ± 0.0019 | -0.0041 ± 0.0011 | -3.3 % | -19.1 % |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 68.59 % | 6.6819 | 10.5525 | +0.0029 ± 0.0017 | +0.0005 ± 0.0010 | -0.0027 ± 0.0021 | -0.0029 ± 0.0011 | -1.4 % | -17.5 % |
| IF4 (Cook et al.) + FourOverSix 1x16 | 54.30 % | 6.6386 | 10.5203 | -0.0036 ± 0.0017 | -0.0026 ± 0.0010 | -0.0092 ± 0.0019 | -0.0060 ± 0.0012 | -20.8 % | -33.7 % |
| IF4 (Cook et al.) + FourOverSix 8x64 | 54.34 % | 6.6637 | 10.5362 | +0.0002 ± 0.0016 | -0.0011 ± 0.0010 | -0.0054 ± 0.0019 | -0.0045 ± 0.0011 | -4.4 % | -20.1 % |
| IF4 (Cook et al.) + FourOverSix 16x64 | 55.51 % | 6.6719 | 10.5444 | +0.0014 ± 0.0017 | -0.0003 ± 0.0010 | -0.0042 ± 0.0019 | -0.0037 ± 0.0010 | -3.3 % | -19.1 % |
| IF4 (Cook et al.) + FourOverSix 256x64 | 67.46 % | 6.6708 | 10.5496 | +0.0012 ± 0.0017 | +0.0002 ± 0.0011 | -0.0043 ± 0.0020 | -0.0032 ± 0.0011 | -1.3 % | -17.4 % |

### Uniform-format share of the weights by projection, %

| policy | all | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF4 (Cook et al.) 1x16 | 62.84 | — | — | — | 63.01 | — | — | 62.54 |
| IF4 (Cook et al.) 8x64 | 94.35 | — | — | — | 95.07 | — | — | 94.02 |
| IF4 (Cook et al.) 16x64 | 98.34 | — | — | — | 98.82 | — | — | 98.22 |
| IF4 (Cook et al.) 256x64 | 99.65 | — | — | — | 99.87 | — | — | 99.79 |
| MixFP4 (Zou et al.) 1x16 | 63.01 | — | — | — | 63.23 | — | — | 62.74 |
| MixFP4 (Zou et al.) 8x64 | 94.50 | — | — | — | 95.31 | — | — | 94.15 |
| MixFP4 (Zou et al.) 16x64 | 98.39 | — | — | — | 98.90 | — | — | 98.26 |
| MixFP4 (Zou et al.) 256x64 | 99.65 | — | — | — | 99.87 | — | — | 99.79 |
| MixFP4 (Zou et al.) + FourOverSix 1x16 | 54.44 | — | — | — | 54.69 | — | — | 54.14 |
| MixFP4 (Zou et al.) + FourOverSix 8x64 | 54.67 | — | — | — | 56.28 | — | — | 52.70 |
| MixFP4 (Zou et al.) + FourOverSix 16x64 | 55.95 | — | — | — | 58.23 | — | — | 53.11 |
| MixFP4 (Zou et al.) + FourOverSix 256x64 | 68.59 | — | — | — | 76.10 | — | — | 58.47 |
| IF4 (Cook et al.) + FourOverSix 1x16 | 54.30 | — | — | — | 54.48 | — | — | 53.98 |
| IF4 (Cook et al.) + FourOverSix 8x64 | 54.34 | — | — | — | 55.58 | — | — | 52.34 |
| IF4 (Cook et al.) + FourOverSix 16x64 | 55.51 | — | — | — | 57.29 | — | — | 52.65 |
| IF4 (Cook et al.) + FourOverSix 256x64 | 67.46 | — | — | — | 73.44 | — | — | 57.06 |
| FlipQuant (ours) 8x64 | 1.57 | — | — | — | 1.89 | — | — | 1.54 |
| FlipQuant (ours) 16x64 | 2.16 | — | — | — | 2.61 | — | — | 2.13 |
| FlipQuant (ours) 256x64 | 6.35 | — | — | — | 7.68 | — | — | 6.48 |

Blocks with a zero (underflowed) E4M3 scale, which become all-zero blocks: 0 in every arm.

### Paper table (WikiText-2): `A_table_phi4_wiki.tex`

```latex
% Phi-4, WikiText-2 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 6.65 & 6.67 & 6.67 & 6.67 \\
MixFP4 (Zou et al.) & 6.66 & 6.69 & 6.68 & 6.69 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 6.65 & 6.68 & 6.68 & 6.68 \\
IF4 (Cook et al.) + FourOverSix (our variant) & 6.64 & 6.66 & 6.67 & 6.67 \\
FlipQuant (ours) & -- & 6.61 & 6.61 & 6.63 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 6.70, NVFP4 weights + FourOverSix act. 6.68, FourOverSix 6.66, BF16 6.46} \\
\bottomrule
\end{tabular}
```

### Paper table (C4): `A_table_phi4_c4.tex`

```latex
% Phi-4, C4 perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here is simulated (fake (c)) and so differs slightly from the native main-table numbers.
\begin{tabular}{lcccc}
\toprule
Method & 1x16 & 8x64 & 16x64 & 256x64 \\
\midrule
IF4 (Cook et al.) & 10.52 & 10.56 & 10.56 & 10.56 \\
MixFP4 (Zou et al.) & 10.52 & 10.56 & 10.56 & 10.56 \\
MixFP4 (Zou et al.) + FourOverSix (our variant) & 10.52 & 10.54 & 10.54 & 10.55 \\
IF4 (Cook et al.) + FourOverSix (our variant) & 10.52 & 10.54 & 10.54 & 10.55 \\
FlipQuant (ours) & -- & 10.50 & 10.51 & 10.52 \\
\midrule
\multicolumn{5}{l}{\footnotesize Reference: NVFP4 10.58, NVFP4 weights + FourOverSix act. 10.56, FourOverSix 10.55, BF16 10.31} \\
\bottomrule
\end{tabular}
```

Re-run references equal to the Parts 2-3 fake (c) records, window by window: {'bf16': True, 'nvfp4': True, 'fo6': True, 'tc-8x64': True, 'tc-16x64': True, 'tc-256x64': True}

