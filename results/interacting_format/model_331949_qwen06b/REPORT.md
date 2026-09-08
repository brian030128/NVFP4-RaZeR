# Interaction-aware tile formats: qwen06b

All candidates fixed at original weights. One shared fit pass; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | Weight-MSE PPL | Independent PPL | Interacting PPL | Interacting − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 38.094636 | 37.354972 | 37.325292 | 36.168485 | -0.051885 ± 0.018146 |
| math | 7.364156 | 7.175079 | 6.609421 | 6.629307 | -0.105124 ± 0.029776 |
| code | 8.792863 | 9.224444 | 8.804501 | 9.300680 | +0.056147 ± 0.039907 |
