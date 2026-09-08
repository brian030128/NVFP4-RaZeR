# Interaction-aware tile formats: llama1b

All candidates fixed at original weights. One shared fit pass; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | Weight-MSE PPL | Independent PPL | Interacting PPL | Interacting − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 21.280689 | 21.764492 | 21.848671 | 21.090116 | -0.008996 ± 0.012975 |
| math | 5.808788 | 5.989449 | 5.930176 | 5.880040 | +0.012192 ± 0.022490 |
| code | 7.629597 | 7.299111 | 7.461202 | 7.261058 | -0.049509 ± 0.033672 |
