# Online correlated activation formats: qwen06b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 32.089531 | 32.144137 | 32.206824 | 32.228513 | +0.004322 ± 0.013202 |
| math | 6.137240 | 6.050990 | 6.121047 | 6.146964 | +0.001583 ± 0.027904 |
| code | 10.172839 | 10.194327 | 10.430406 | 10.560305 | +0.037381 ± 0.039917 |
