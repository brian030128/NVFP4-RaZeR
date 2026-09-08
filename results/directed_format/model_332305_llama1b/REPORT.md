# Analytic teacher-error Fisher component: llama1b

One shared score pass; frozen type-only maps. No candidate-loss backtracking.

| Domain | FourOverSix | Fixed budget | Diagonal Fisher | Standard Fisher | Directed Fisher | Directed − baseline ΔNLL ±2SE |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 19.489043 | 18.344613 | 34.895749 | 19.522458 | 19.359486 | -0.006670 ±0.013727 |
| math | 5.415171 | 5.240353 | 7.691687 | 5.491225 | 5.351973 | -0.011739 ±0.013618 |
| code | 10.428661 | 10.238523 | 15.126533 | 11.235485 | 11.032171 | +0.056258 ±0.033829 |
