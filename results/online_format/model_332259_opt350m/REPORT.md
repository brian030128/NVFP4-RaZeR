# Online correlated activation formats: opt350m

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 29.861772 | 29.617030 | 29.641390 | 29.493875 | -0.012397 ± 0.009594 |
| math | 21.345903 | 21.070504 | 20.999291 | 21.238339 | -0.005052 ± 0.013751 |
| code | 29.951755 | 30.253229 | 29.799576 | 30.015236 | +0.002117 ± 0.024702 |
