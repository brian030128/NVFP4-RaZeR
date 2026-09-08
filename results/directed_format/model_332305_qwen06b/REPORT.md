# Analytic teacher-error Fisher component: qwen06b

One shared score pass; frozen type-only maps. No candidate-loss backtracking.

| Domain | FourOverSix | Fixed budget | Diagonal Fisher | Standard Fisher | Directed Fisher | Directed − baseline ΔNLL ±2SE |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 30.836604 | 27.711027 | 51.107887 | 32.694396 | 31.796201 | +0.030644 ±0.025248 |
| math | 6.814897 | 6.570258 | 9.780673 | 7.047576 | 6.979904 | +0.023924 ±0.039735 |
| code | 12.650615 | 11.860891 | 12.150308 | 11.240001 | 11.179572 | -0.123618 ±0.043271 |
