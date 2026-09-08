# Consumer-aware activation formats: qwen06b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Consumer PPL | Consumer − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|
| wiki | 36.464851 | 36.719040 | 36.632353 | +0.004583 ± 0.012682 |
| math | 6.726260 | 6.583038 | 6.650178 | -0.011376 ± 0.033451 |
| code | 11.499311 | 11.989473 | 11.824377 | +0.027876 ± 0.044688 |
