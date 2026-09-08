# Teacher-Fisher tile formats: qwen06b

All candidates fixed at original weights. One shared fit pass; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | Uniform PPL | Diagonal-Fisher PPL | Block-Fisher PPL | Block-Fisher − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 25.290595 | 23.885393 | 23.953120 | 23.843995 | -0.058900 ± 0.018898 |
| math | 6.754815 | 6.230156 | 6.264550 | 6.341050 | -0.063211 ± 0.033491 |
| code | 10.384050 | 10.804020 | 11.009136 | 10.643549 | +0.024683 ± 0.034134 |
