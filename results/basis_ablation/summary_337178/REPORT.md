# Basis ablation: E0M3, or task-gradient selection over any legal knob?

Identical data, teacher, CE/KL two-SE rule, counts and held-out sets across arms;
only the scored direction differs (`quantize/basis.py`). `e0m3` is the reported
direction, re-run through the same code path so the arms are comparable.

## Direction length

A shorter direction is a smaller step, not necessarily a worse one.

| model | basis | baseline | alternative | \|d\|/\|W\| | moved tiles | base err | alt err |
|---|---|---|---|---:|---:|---:|---:|
| llama8b | e0m3 | FourOverSix | E0M3 alpha1 | 0.127008 | 13,631,488/13,631,488 | 0.086903 | 0.087555 |
| llama8b | alpha | FourOverSix | E2M1 dense9 alpha | 0.067516 | 13,631,321/13,631,488 | 0.086903 | 0.084001 |
| llama8b | type_pure | NVFP4 alpha1 | E0M3 alpha1 | 0.136485 | 13,631,488/13,631,488 | 0.094883 | 0.087555 |
| qwen4b | e0m3 | FourOverSix | E0M3 alpha1 | 0.126968 | 7,096,320/7,096,320 | 0.086951 | 0.087482 |
| qwen4b | alpha | FourOverSix | E2M1 dense9 alpha | 0.067507 | 7,096,103/7,096,320 | 0.086951 | 0.084040 |
| qwen4b | type_pure | NVFP4 alpha1 | E0M3 alpha1 | 0.136486 | 7,096,320/7,096,320 | 0.094937 | 0.087482 |

## llama8b

`Δ` is against **that arm's own baseline**; paired ΔNLL ± 2 SE is descriptive.

| basis | policy | blocks | wiki PPL | Δ wiki | wiki ΔNLL ±2SE | c4 PPL | Δ c4 | c4 ΔNLL ±2SE |
|---|---|---:|---:|---:|---|---:|---:|---|
| e0m3 | four_over_six | 0 | 9.092934 | — | — | 11.600109 | — | — |
| e0m3 | fixed256_math_code128 | 256 | 9.028599 | -0.064335 | -0.007100 ±0.001722 | 11.513102 | -0.087006 | -0.007529 ±0.002139 |
| e0m3 | adaptive_math_code128 | 2 | 9.088794 | -0.004140 | -0.000455 ±0.001657 | 11.587583 | -0.012526 | -0.001080 ±0.002199 |
| e0m3 | weight_mse | 6,652,753 | 9.088662 | -0.004272 | -0.000470 ±0.002130 | 11.635797 | +0.035688 | +0.003072 ±0.003394 |
| alpha | four_over_six | 0 | 9.092934 | — | — | 11.600109 | — | — |
| alpha | fixed256_math_code128 | 256 | 9.074827 | -0.018107 | -0.001993 ±0.001735 | 11.564209 | -0.035899 | -0.003100 ±0.002239 |
| alpha | adaptive_math_code128 | 15 | 9.086697 | -0.006237 | -0.000686 ±0.001587 | 11.598396 | -0.001713 | -0.000148 ±0.002117 |
| alpha | weight_mse | 13,631,150 | 9.080619 | -0.012315 | -0.001355 ±0.002064 | 11.568762 | -0.031347 | -0.002706 ±0.002611 |
| type_pure | four_over_six | 0 | 9.102980 | — | — | 11.641787 | — | — |
| type_pure | fixed256_math_code128 | 256 | 9.028807 | -0.074173 | -0.008182 ±0.001809 | 11.609019 | -0.032768 | -0.002819 ±0.003000 |
| type_pure | adaptive_math_code128 | 3 | 9.081192 | -0.021788 | -0.002396 ±0.001661 | 11.650161 | +0.008374 | +0.000719 ±0.002770 |
| type_pure | weight_mse | 12,455,917 | 9.163559 | +0.060579 | +0.006633 ±0.002714 | 11.724969 | +0.083182 | +0.007120 ±0.003341 |

### Same number, absolute PPL (baselines differ across arms)

| policy | domain | e0m3 | alpha | type_pure |
|---|---|---:|---:|---:|
| four_over_six | wiki | 9.092934 | 9.092934 | 9.102980 |
| four_over_six | c4 | 11.600109 | 11.600109 | 11.641787 |
| fixed256_math_code128 | wiki | 9.028599 | 9.074827 | 9.028807 |
| fixed256_math_code128 | c4 | 11.513102 | 11.564209 | 11.609019 |
| adaptive_math_code128 | wiki | 9.088794 | 9.086697 | 9.081192 |
| adaptive_math_code128 | c4 | 11.587583 | 11.598396 | 11.650161 |
| weight_mse | wiki | 9.088662 | 9.080619 | 9.163559 |
| weight_mse | c4 | 11.635797 | 11.568762 | 11.724969 |

## qwen4b

`Δ` is against **that arm's own baseline**; paired ΔNLL ± 2 SE is descriptive.

| basis | policy | blocks | wiki PPL | Δ wiki | wiki ΔNLL ±2SE | c4 PPL | Δ c4 | c4 ΔNLL ±2SE |
|---|---|---:|---:|---:|---|---:|---:|---|
| e0m3 | four_over_six | 0 | 19.414897 | — | — | 21.928243 | — | — |
| e0m3 | fixed256_math_code128 | 256 | 17.426627 | -1.988270 | -0.108041 ±0.004074 | 20.940578 | -0.987665 | -0.046087 ±0.003575 |
| e0m3 | adaptive_math_code128 | 1 | 19.397059 | -0.017838 | -0.000919 ±0.000114 | 21.909817 | -0.018426 | -0.000841 ±0.000159 |
| e0m3 | weight_mse | 3,407,142 | 20.108009 | +0.693113 | +0.035078 ±0.004814 | 22.082958 | +0.154715 | +0.007031 ±0.004017 |
| alpha | four_over_six | 0 | 19.414897 | — | — | 21.928243 | — | — |
| alpha | fixed256_math_code128 | 256 | 18.256618 | -1.158278 | -0.061513 ±0.003812 | 21.398842 | -0.529401 | -0.024439 ±0.002938 |
| alpha | adaptive_math_code128 | 1 | 19.398985 | -0.015912 | -0.000820 ±0.000201 | 21.913185 | -0.015058 | -0.000687 ±0.000293 |
| alpha | weight_mse | 7,095,825 | 19.076245 | -0.338651 | -0.017597 ±0.004122 | 21.870663 | -0.057580 | -0.002629 ±0.003609 |
| type_pure | four_over_six | 0 | 18.758341 | — | — | 21.813711 | — | — |
| type_pure | fixed256_math_code128 | 256 | 17.335883 | -1.422458 | -0.078860 ±0.003590 | 20.947186 | -0.866525 | -0.040534 ±0.003201 |
| type_pure | adaptive_math_code128 | 0 | 18.758341 | +0.000000 | +0.000000 ±0.000000 | 21.813711 | +0.000000 | +0.000000 ±0.000000 |
| type_pure | weight_mse | 6,317,891 | 20.254720 | +1.496379 | +0.076749 ±0.006107 | 22.319720 | +0.506009 | +0.022932 ±0.005394 |

### Same number, absolute PPL (baselines differ across arms)

| policy | domain | e0m3 | alpha | type_pure |
|---|---|---:|---:|---:|
| four_over_six | wiki | 19.414897 | 19.414897 | 18.758341 |
| four_over_six | c4 | 21.928243 | 21.928243 | 21.813711 |
| fixed256_math_code128 | wiki | 17.426627 | 18.256618 | 17.335883 |
| fixed256_math_code128 | c4 | 20.940578 | 21.398842 | 20.947186 |
| adaptive_math_code128 | wiki | 19.397059 | 19.398985 | 18.758341 |
| adaptive_math_code128 | c4 | 21.909817 | 21.913185 | 21.813711 |
| weight_mse | wiki | 20.108009 | 19.076245 | 20.254720 |
| weight_mse | c4 | 22.082958 | 21.870663 | 22.319720 |

