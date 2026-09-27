# B2/B3 and the deployment-path verification — report

Protocol: `PROTOCOL.md`, registered 2026-09-27T13:07:11Z (sha256 6d89d6dd740d…, `registration.json`), before any
Part 2–3 run and pushed with Part 1. Deviation 1: the analysis script gained the diagnostics after the runs; the
registered analyses and criterion are unchanged. The final method is TM-OPT+TC; the maps are the committed seed-0
maps. Nothing was selected or tuned on WikiText-2, C4 or zero-shot.

## Summary

**B2: implemented** (documented in `docs/BUILD_AND_USE.md`).
- **`export_map_artifact.py`:** a trained `map.pt` becomes a MIXFP4MAP/1 map and then an SM120 artifact.
  - Kernels: 8x64 maps run on `n8k64_wB`, 16x64 maps on `n16k64_wA`, and **256x64 maps on `n16k64_wA` as
    16x64 granules** (element mask checked unchanged).
  - Before export it checks, per module and bitwise, that the calibration's candidates (FourOverSix E2M1, E0M3
    alpha=1) equal SM120's fake-quant candidates: 0 mismatches on all 20 artifacts. It also checks that the
    model's weights equal the calibration record's.
  - The exporter then checks every packed weight bit for bit against the fake-quant weight.
- **`run_ppl_deploy.py`:** convention (c) on the released protocol windows, batch 1.
  - It runs NativeLinear, the fake (c) reference and BF16.
  - Its BF16 window NLLs equal the committed BF16 evaluations **bitwise** on all four models. So its windows,
    model loading and NLL are those of the (a) evaluations, and any difference is the quantization's.

**B3: every exported artifact passes the ownership check.**
- 20 artifacts: FourOverSix, NVFP4 and three maps per model.
- 238.7 billion weight elements checked; all exact, 0 format mismatches.
- The tags equal the map on all 12 map artifacts, and the E0M3 elements execute as E0M3 on the chosen kernel.

**(i) Kernel numerics, NativeLinear (c) vs fake (c): the registered criterion holds in 37 of 40 cells, and fails in
3.**
- **The criterion:** "negligible" means |mean| ≤ 2 SE in each cell.
- **The three cells**, each beyond 2 SE by 3–5 % (|z| = 2.06–2.09):

  | model | policy | corpus | ΔNLL ± 2 SE | ΔPPL |
  |---|---|---|---|---:|
  | Mistral-7B | NVFP4 | C4 | −0.00066 ± 0.00064 | −0.005 (native better) |
  | Mistral-7B | TM-OPT+TC 16x64 | C4 | +0.00128 ± 0.00122 | +0.010 |
  | Qwen3.8-27B | TM-OPT+TC 16x64 | WikiText-2 | +0.00297 ± 0.00288 | +0.021 |

- **Bound over all 40 cells:** |ΔPPL| ≤ 0.021 (0.3 %), and |mean ΔNLL| ≤ 0.003.
- **Diagnostics: no kernel or packing defect.** On the most divergent windows and on a typical window:
  - the fused activation quantizer reproduces the reference codes bitwise in every layer;
  - in **every** layer (0 of 224 Mistral and 0 of 496 Qwen layers otherwise), the native output is closer to the
    exact FP64 product of the quantized operands than fake quant is: mean relative error 0.166 % vs 0.24 %.
  - What the differences are: tiny per-layer rounding differences compounding through the network.
  - Where the significance comes from: a few sensitive windows, or a small shift.
    - Qwen: without its largest window (+0.122) the cell is not significant.
    - Mistral TM-OPT+TC 16x64: one window (+0.135) carries half the PPL shift.
    - Mistral NVFP4: a small shift in the native path's favour.
- **Context, not a change of criterion:** with 40 cells, about 1.8 exceedances of 2 SE are expected from chance
  alone (≈ 27 % chance of 3 or more if the cells were independent). The signs are mixed.

**(ii) Effect preservation: every TM-OPT+TC map stays significantly better than FourOverSix and than NVFP4 under
NativeLinear.**
- All 48 comparisons (4 models × 3 maps × 2 baselines × 2 corpora) are significant.
- The effects are close to fake (c)'s.

**(iii) Convention effect, NativeLinear (c) vs our native (a) evaluator** (appendix):
- |mean ΔNLL| ≤ 0.0035; 38 of 40 cells are not significant.
- The two significant cells are NVFP4, where (c) is slightly better. Its per-token activation scales are finer
  than (a)'s per-window scale.

**Wall time per evaluated map:**
- NativeLinear (c) is 25–28 s on Llama and Mistral, 45–48 s on Phi-4 and 468–520 s on Qwen3.8-27B. Our native (a)
  evaluator takes ≈ 45–47 s, 69 s and 541 s.
- NativeLinear is also faster than BF16 on the three smaller models.
- Qwen's time is dominated by the torch implementation of its linear attention, which runs in BF16 in every policy.
- Fake (c) is 5–11× slower than NativeLinear on the three smaller models, and 1.7–2.7× slower on Qwen.

### PPL under NativeLinear (c)

| model | BF16 | NVFP4 | FourOverSix | TM-OPT+TC 8x64 | TM-OPT+TC 16x64 | TM-OPT+TC 256x64 (appendix) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 6.2403 / 8.9579 | 6.9338 / 9.9284 | 6.8706 / 9.8257 | 6.7877 / 9.6774 | 6.7827 / 9.6827 | 6.8007 / 9.7252 |
| Mistral-7B-v0.3 | 5.3182 / 7.8306 | 5.5506 / 8.0895 | 5.5224 / 8.0623 | 5.4868 / 8.0215 | 5.4899 / 8.0323 | 5.4904 / 8.0279 |
| Phi-4 | 6.4615 / 10.3098 | 6.6934 / 10.5795 | 6.6617 / 10.5441 | 6.6078 / 10.4979 | 6.6133 / 10.5047 | 6.6257 / 10.5146 |
| Qwen3.8-27B | 7.0509 / 9.8935 | 7.5506 / 10.2185 | 7.3215 / 10.1869 | 7.1074 / 10.1235 | 7.1232 / 10.1276 | 7.1773 / 10.1512 |

### (i) Kernel numerics: NativeLinear (c) − fake (c)

| model | policy | corpus | PPL native (c) | PPL fake (c) | ΔPPL | ΔNLL mean ± 2 SE | max \|ΔNLL\| | negligible |
|---|---|---|---:|---:|---:|---|---:|---|
| Llama-3.1-8B | FourOverSix | WikiText-2 | 6.8706 | 6.8807 | -0.0101 | -0.001472 ± 0.001949 | 0.0331 | yes |
| Llama-3.1-8B | FourOverSix | C4 | 9.8257 | 9.8141 | +0.0116 | +0.001180 ± 0.001508 | 0.0694 | yes |
| Llama-3.1-8B | NVFP4 | WikiText-2 | 6.9338 | 6.9296 | +0.0042 | +0.000606 ± 0.001767 | 0.0303 | yes |
| Llama-3.1-8B | NVFP4 | C4 | 9.9284 | 9.9302 | -0.0018 | -0.000182 ± 0.001123 | 0.0405 | yes |
| Llama-3.1-8B | TM-OPT+TC 8x64 | WikiText-2 | 6.7877 | 6.7846 | +0.0031 | +0.000454 ± 0.001341 | 0.0241 | yes |
| Llama-3.1-8B | TM-OPT+TC 8x64 | C4 | 9.6774 | 9.6767 | +0.0006 | +0.000063 ± 0.001194 | 0.0606 | yes |
| Llama-3.1-8B | TM-OPT+TC 16x64 | WikiText-2 | 6.7827 | 6.7825 | +0.0001 | +0.000018 ± 0.001482 | 0.0248 | yes |
| Llama-3.1-8B | TM-OPT+TC 16x64 | C4 | 9.6827 | 9.6778 | +0.0049 | +0.000505 ± 0.001036 | 0.0529 | yes |
| Llama-3.1-8B | TM-OPT+TC 256x64 (appendix) | WikiText-2 | 6.8007 | 6.8022 | -0.0015 | -0.000219 ± 0.001450 | 0.0370 | yes |
| Llama-3.1-8B | TM-OPT+TC 256x64 (appendix) | C4 | 9.7252 | 9.7264 | -0.0012 | -0.000126 ± 0.001290 | 0.0650 | yes |
| Mistral-7B-v0.3 | FourOverSix | WikiText-2 | 5.5224 | 5.5210 | +0.0014 | +0.000251 ± 0.000733 | 0.0146 | yes |
| Mistral-7B-v0.3 | FourOverSix | C4 | 8.0623 | 8.0681 | -0.0058 | -0.000725 ± 0.001321 | 0.1479 | yes |
| Mistral-7B-v0.3 | NVFP4 | WikiText-2 | 5.5506 | 5.5508 | -0.0002 | -0.000028 ± 0.000925 | 0.0172 | yes |
| Mistral-7B-v0.3 | NVFP4 | C4 | 8.0895 | 8.0949 | -0.0054 | -0.000664 ± 0.000641 | 0.0154 | **no** |
| Mistral-7B-v0.3 | TM-OPT+TC 8x64 | WikiText-2 | 5.4868 | 5.4886 | -0.0018 | -0.000336 ± 0.000829 | 0.0132 | yes |
| Mistral-7B-v0.3 | TM-OPT+TC 8x64 | C4 | 8.0215 | 8.0216 | -0.0001 | -0.000015 ± 0.000610 | 0.0154 | yes |
| Mistral-7B-v0.3 | TM-OPT+TC 16x64 | WikiText-2 | 5.4899 | 5.4882 | +0.0017 | +0.000303 ± 0.000801 | 0.0156 | yes |
| Mistral-7B-v0.3 | TM-OPT+TC 16x64 | C4 | 8.0323 | 8.0221 | +0.0102 | +0.001276 ± 0.001219 | 0.1346 | **no** |
| Mistral-7B-v0.3 | TM-OPT+TC 256x64 (appendix) | WikiText-2 | 5.4904 | 5.4881 | +0.0023 | +0.000420 ± 0.000841 | 0.0154 | yes |
| Mistral-7B-v0.3 | TM-OPT+TC 256x64 (appendix) | C4 | 8.0279 | 8.0325 | -0.0046 | -0.000578 ± 0.000678 | 0.0390 | yes |
| Phi-4 | FourOverSix | WikiText-2 | 6.6617 | 6.6627 | -0.0009 | -0.000141 ± 0.001297 | 0.0278 | yes |
| Phi-4 | FourOverSix | C4 | 10.5441 | 10.5473 | -0.0032 | -0.000303 ± 0.000888 | 0.0583 | yes |
| Phi-4 | NVFP4 | WikiText-2 | 6.6934 | 6.6998 | -0.0064 | -0.000959 ± 0.001466 | 0.0281 | yes |
| Phi-4 | NVFP4 | C4 | 10.5795 | 10.5834 | -0.0038 | -0.000364 ± 0.000731 | 0.0196 | yes |
| Phi-4 | TM-OPT+TC 8x64 | WikiText-2 | 6.6078 | 6.6115 | -0.0036 | -0.000546 ± 0.001102 | 0.0209 | yes |
| Phi-4 | TM-OPT+TC 8x64 | C4 | 10.4979 | 10.4955 | +0.0025 | +0.000235 ± 0.000689 | 0.0182 | yes |
| Phi-4 | TM-OPT+TC 16x64 | WikiText-2 | 6.6133 | 6.6118 | +0.0015 | +0.000224 ± 0.001139 | 0.0228 | yes |
| Phi-4 | TM-OPT+TC 16x64 | C4 | 10.5047 | 10.5056 | -0.0010 | -0.000091 ± 0.000703 | 0.0183 | yes |
| Phi-4 | TM-OPT+TC 256x64 (appendix) | WikiText-2 | 6.6257 | 6.6293 | -0.0036 | -0.000544 ± 0.001194 | 0.0237 | yes |
| Phi-4 | TM-OPT+TC 256x64 (appendix) | C4 | 10.5146 | 10.5162 | -0.0016 | -0.000155 ± 0.000724 | 0.0217 | yes |
| Qwen3.8-27B | FourOverSix | WikiText-2 | 7.3215 | 7.3016 | +0.0199 | +0.002718 ± 0.004927 | 0.2585 | yes |
| Qwen3.8-27B | FourOverSix | C4 | 10.1869 | 10.1882 | -0.0013 | -0.000124 ± 0.000776 | 0.0263 | yes |
| Qwen3.8-27B | NVFP4 | WikiText-2 | 7.5506 | 7.5446 | +0.0060 | +0.000795 ± 0.005080 | 0.1205 | yes |
| Qwen3.8-27B | NVFP4 | C4 | 10.2185 | 10.2222 | -0.0037 | -0.000359 ± 0.000844 | 0.0221 | yes |
| Qwen3.8-27B | TM-OPT+TC 8x64 | WikiText-2 | 7.1074 | 7.0934 | +0.0140 | +0.001973 ± 0.002584 | 0.0721 | yes |
| Qwen3.8-27B | TM-OPT+TC 8x64 | C4 | 10.1235 | 10.1235 | +0.0000 | +0.000004 ± 0.000739 | 0.0241 | yes |
| Qwen3.8-27B | TM-OPT+TC 16x64 | WikiText-2 | 7.1232 | 7.1021 | +0.0211 | +0.002966 ± 0.002881 | 0.1220 | **no** |
| Qwen3.8-27B | TM-OPT+TC 16x64 | C4 | 10.1276 | 10.1262 | +0.0015 | +0.000145 ± 0.000716 | 0.0233 | yes |
| Qwen3.8-27B | TM-OPT+TC 256x64 (appendix) | WikiText-2 | 7.1773 | 7.1981 | -0.0209 | -0.002905 ± 0.003167 | 0.1003 | yes |
| Qwen3.8-27B | TM-OPT+TC 256x64 (appendix) | C4 | 10.1512 | 10.1451 | +0.0061 | +0.000604 ± 0.000718 | 0.0263 | yes |

### (ii) Effect preservation: TM-OPT+TC minus each baseline, native (c) [fake (c)]

| model | map | vs FourOverSix, WikiText-2 | vs FourOverSix, C4 | vs NVFP4, WikiText-2 | vs NVFP4, C4 |
|---|---|---|---|---|---|
| Llama-3.1-8B | 8x64 | -0.01213 ± 0.00170 (better) [-0.01406] | -0.01521 ± 0.00301 (better) [-0.01410] | -0.02129 ± 0.00226 (better) [-0.02114] | -0.02561 ± 0.00452 (better) [-0.02586] |
| Llama-3.1-8B | 16x64 | -0.01288 ± 0.00177 (better) [-0.01437] | -0.01466 ± 0.00307 (better) [-0.01399] | -0.02204 ± 0.00231 (better) [-0.02145] | -0.02506 ± 0.00460 (better) [-0.02575] |
| Llama-3.1-8B | 256x64 (appendix) | -0.01023 ± 0.00167 (better) [-0.01148] | -0.01029 ± 0.00215 (better) [-0.00898] | -0.01939 ± 0.00207 (better) [-0.01856] | -0.02068 ± 0.00359 (better) [-0.02074] |
| Mistral-7B-v0.3 | 8x64 | -0.00647 ± 0.00094 (better) [-0.00588] | -0.00507 ± 0.00097 (better) [-0.00578] | -0.01157 ± 0.00129 (better) [-0.01126] | -0.00845 ± 0.00092 (better) [-0.00910] |
| Mistral-7B-v0.3 | 16x64 | -0.00590 ± 0.00088 (better) [-0.00596] | -0.00372 ± 0.00166 (better) [-0.00572] | -0.01101 ± 0.00121 (better) [-0.01134] | -0.00710 ± 0.00135 (better) [-0.00904] |
| Mistral-7B-v0.3 | 256x64 (appendix) | -0.00581 ± 0.00096 (better) [-0.00598] | -0.00427 ± 0.00118 (better) [-0.00442] | -0.01091 ± 0.00117 (better) [-0.01136] | -0.00765 ± 0.00100 (better) [-0.00774] |
| Phi-4 | 8x64 | -0.00812 ± 0.00149 (better) [-0.00772] | -0.00439 ± 0.00095 (better) [-0.00493] | -0.01286 ± 0.00198 (better) [-0.01327] | -0.00774 ± 0.00114 (better) [-0.00834] |
| Phi-4 | 16x64 | -0.00730 ± 0.00143 (better) [-0.00767] | -0.00375 ± 0.00091 (better) [-0.00396] | -0.01204 ± 0.00204 (better) [-0.01322] | -0.00710 ± 0.00108 (better) [-0.00737] |
| Phi-4 | 256x64 (appendix) | -0.00543 ± 0.00125 (better) [-0.00502] | -0.00281 ± 0.00085 (better) [-0.00295] | -0.01016 ± 0.00179 (better) [-0.01058] | -0.00616 ± 0.00105 (better) [-0.00637] |
| Qwen3.8-27B | 8x64 | -0.02968 ± 0.00605 (better) [-0.02893] | -0.00624 ± 0.00100 (better) [-0.00637] | -0.06049 ± 0.00924 (better) [-0.06167] | -0.00934 ± 0.00134 (better) [-0.00970] |
| Qwen3.8-27B | 16x64 | -0.02745 ± 0.00520 (better) [-0.02770] | -0.00584 ± 0.00096 (better) [-0.00611] | -0.05827 ± 0.00909 (better) [-0.06044] | -0.00893 ± 0.00132 (better) [-0.00944] |
| Qwen3.8-27B | 256x64 (appendix) | -0.01989 ± 0.00558 (better) [-0.01427] | -0.00351 ± 0.00087 (better) [-0.00424] | -0.05071 ± 0.00879 (better) [-0.04701] | -0.00660 ± 0.00130 (better) [-0.00757] |

### (iii) Convention effect: NativeLinear (c) − native (a) (appendix, descriptive)

| model | FourOverSix (Wiki; C4) | NVFP4 | TM-OPT+TC 8x64 | TM-OPT+TC 16x64 | TM-OPT+TC 256x64 |
|---|---|---|---|---|---|
| Llama-3.1-8B | -0.00074 ± 0.00158; +0.00003 ± 0.00146 | -0.00033 ± 0.00152; +0.00005 ± 0.00143 | +0.00074 ± 0.00137; -0.00022 ± 0.00103 | -0.00057 ± 0.00139; -0.00037 ± 0.00098 | -0.00007 ± 0.00145; +0.00025 ± 0.00129 |
| Mistral-7B-v0.3 | -0.00001 ± 0.00087; -0.00046 ± 0.00118 | -0.00083 ± 0.00087; -0.00076 ± 0.00066* | +0.00010 ± 0.00090; -0.00058 ± 0.00070 | -0.00095 ± 0.00238; +0.00074 ± 0.00159 | +0.00008 ± 0.00095; -0.00059 ± 0.00065 |
| Phi-4 | -0.00047 ± 0.00128; -0.00014 ± 0.00084 | -0.00167 ± 0.00144*; -0.00067 ± 0.00079 | -0.00075 ± 0.00116; +0.00024 ± 0.00073 | -0.00042 ± 0.00105; +0.00013 ± 0.00072 | -0.00077 ± 0.00119; +0.00008 ± 0.00075 |
| Qwen3.8-27B | +0.00168 ± 0.00489; +0.00029 ± 0.00078 | -0.00346 ± 0.00495; -0.00020 ± 0.00082 | +0.00017 ± 0.00273; -0.00026 ± 0.00067 | +0.00124 ± 0.00263; -0.00058 ± 0.00079 | -0.00248 ± 0.00273; +0.00030 ± 0.00075 |

(* significant)

### Wall time per evaluated map (one process per model, batch 1, all windows)

| model | NativeLinear (c) | fake (c) | BF16 | native (a), committed (≈ per map) | export + checks + B3, per artifact |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 27–28 s | 167–284 s | 52 s | 47 s (330 s / 7 maps; packing 41 s) | 41–59 s |
| Mistral-7B-v0.3 | 25–26 s | 174–297 s | 53 s | 45 s (318 s / 7 maps; packing 42 s) | 42–59 s |
| Phi-4 | 45–48 s | 237–376 s | 97 s | 69 s (485 s / 7 maps; packing 84 s) | 86–120 s |
| Qwen3.8-27B | 468–520 s | 883–1249 s | 458 s | 541 s (4327 s / 8 maps; packing 163 s) | 153–213 s |

## Diagnostics of the three significant cells

| cell | z | median ΔNLL | windows positive | largest windows (index, ΔNLL) | without the largest window |
|---|---:|---:|---:|---|---|
| Mistral NVFP4 C4 | −2.07 | −0.00067 | 118/256 | (2, +0.015), (72, +0.014), (196, −0.014) | −0.00073 ± 0.00063 |
| Mistral TM-OPT+TC 16x64 C4 | +2.09 | +0.00046 | 139/256 | (124, +0.135), (22, +0.018), (162, +0.018) | +0.00075 ± 0.00063 |
| Qwen TM-OPT+TC 16x64 WikiText-2 | +2.06 | +0.00170 | 79/145 | (137, +0.122), (106, +0.050), (39, +0.047) | +0.00214 ± 0.00238 (n.s.) |

- **Layerwise runs** (`sm120/eval/layerwise.py`): the window with the largest |ΔNLL| and window 0 of the two map
  cells (`diagnostics/`).
- **What each run compares:** the fake (c) model runs, and at every quantized Linear the native output and the
  fake output are compared on the same input with the exact FP64 product of the quantized operands.

| run | layers | activation codes bitwise | native vs exact, mean (max) | fake vs exact, mean (max) | layers where native is worse |
|---|---:|---|---|---|---:|
| Mistral TM-OPT+TC 16x64, C4 window 124 | 224 | yes | 0.00166 (0.00198) | 0.00248 (0.00361) | 0 |
| Mistral TM-OPT+TC 16x64, C4 window 0 | 224 | yes | 0.00166 (0.00248) | 0.00249 (0.00362) | 0 |
| Qwen TM-OPT+TC 16x64, WikiText window 137 | 496 | yes | 0.00166 (0.00173) | 0.00236 (0.00341) | 0 |
| Qwen TM-OPT+TC 16x64, WikiText window 0 | 496 | yes | 0.00166 (0.00173) | 0.00238 (0.00343) | 0 |

- **The per-layer error profile** is the same on the divergent windows as on a typical window.
- **The native kernel is the more accurate path in every layer:** the fake reference rounds the dequantized operands
  to BF16 before its GEMM, while the kernel rounds once, after an FP32 epilogue.
- **For NVFP4** there is no map, so `layerwise.py` does not apply. Its cell is a small shift (native better), not
  an outlier.

## B3: ownership check (every exported artifact)

- **Map artifacts:** `export_map_artifact.py --ownership` runs `sm120/eval/ownership_check.py` on them.
- **FourOverSix and NVFP4:** the same per-module check with an all-E2M1 map on `stock_wA`.
- **What is checked:** every weight element is executed through an identity GEMM. It must equal the decode of the
  stored nibble and scale bitwise, and it must run in the map's format wherever the formats differ.

| model | artifact | kernel | elements checked | exact | format mismatches | E0M3 elements observed | tags = map |
|---|---|---|---:|---|---:|---:|---|
| Llama-3.1-8B | FourOverSix | stock_wA | 6,394,657,250 | True | 0 | 0 | — (no map) |
| Llama-3.1-8B | NVFP4 | stock_wA | 6,486,930,441 | True | 0 | 0 | — (no map) |
| Llama-3.1-8B | TM-OPT+TC 8x64 | n8k64_wB | 6,388,855,037 | True | 0 | 139,269,451 | True |
| Llama-3.1-8B | TM-OPT+TC 16x64 | n16k64_wA | 6,387,110,751 | True | 0 | 181,499,205 | True |
| Llama-3.1-8B | TM-OPT+TC 256x64 (appendix) | n16k64_wA | 6,373,518,405 | True | 0 | 509,477,942 | True |
| Mistral-7B-v0.3 | FourOverSix | stock_wA | 6,385,698,929 | True | 0 | 0 | — (no map) |
| Mistral-7B-v0.3 | NVFP4 | stock_wA | 6,477,660,751 | True | 0 | 0 | — (no map) |
| Mistral-7B-v0.3 | TM-OPT+TC 8x64 | n8k64_wB | 6,379,346,656 | True | 0 | 152,327,676 | True |
| Mistral-7B-v0.3 | TM-OPT+TC 16x64 | n16k64_wA | 6,377,300,990 | True | 0 | 201,485,678 | True |
| Mistral-7B-v0.3 | TM-OPT+TC 256x64 (appendix) | n16k64_wA | 6,360,294,757 | True | 0 | 611,348,756 | True |
| Phi-4 | FourOverSix | stock_wA | 12,521,897,790 | True | 0 | 0 | — (no map) |
| Phi-4 | NVFP4 | stock_wA | 12,701,721,416 | True | 0 | 0 | — (no map) |
| Phi-4 | TM-OPT+TC 8x64 | n8k64_wB | 12,514,298,135 | True | 0 | 188,973,165 | True |
| Phi-4 | TM-OPT+TC 16x64 | n16k64_wA | 12,511,466,938 | True | 0 | 259,554,136 | True |
| Phi-4 | TM-OPT+TC 256x64 (appendix) | n16k64_wA | 12,491,208,632 | True | 0 | 764,042,577 | True |
| Qwen3.8-27B | FourOverSix | stock_wA | 22,332,224,751 | True | 0 | 0 | — (no map) |
| Qwen3.8-27B | NVFP4 | stock_wA | 22,653,423,249 | True | 0 | 0 | — (no map) |
| Qwen3.8-27B | TM-OPT+TC 8x64 | n8k64_wB | 22,320,716,396 | True | 0 | 276,966,155 | True |
| Qwen3.8-27B | TM-OPT+TC 16x64 | n16k64_wA | 22,318,142,129 | True | 0 | 338,708,021 | True |
| Qwen3.8-27B | TM-OPT+TC 256x64 (appendix) | n16k64_wA | 22,295,613,750 | True | 0 | 880,694,305 | True |

In all: 238,672,087,153 elements, 0 mismatches.

## Checks

- **Windows:**
  - equal to the SM120 reference (token hashes) on all four models;
  - validated against the published records (Llama, Qwen);
  - equal to the windows of the committed (a) evaluations.
- **Weights:** every artifact's model weights equal the calibration record (sha256 per matrix), and its
  candidates equal SM120's.
- **Coverage:** every native policy ran every quantized Linear natively in every forward (calls = modules × windows).
- **Kernels:** 8x64 maps on `n8k64_wB`; 16x64 and 256x64 (as 16x64 granules) on `auto`, the width-selecting
  weights-on-A mixed set; FourOverSix and NVFP4 on `auto_stock`. All builds are in `sm120/build` from this
  repository. The mixed builds pass their self-tests, and the SASS equals the R2 builds.

## Files

- **Tools:** `export_map_artifact.py`, `run_ppl_deploy.py`, `sm120/eval/ownership_check.py`,
  `sm120/eval/layerwise.py`.
- **Tables:** `verification.{json,md}` (`analyze.py`).
- **Records:**
  - `runs/<model>/report.json`: per-window NLLs of every policy, coverage, GEMM audit, times;
  - `runs/queue.sh` and `runs/commands.txt`;
  - `artifacts/<name>/{artifact.json, ownership.json}` and `artifacts/*.provenance.json` (no weights);
  - `diagnostics/` (the layerwise runs).
- **The artifacts themselves:** `/home/dev/n16k64_campaign/deploy_eval/artifacts` (4–15 GB each). They can be
  regenerated with `export_map_artifact.py`.
