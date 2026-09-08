# Online correlated activation formats: llama1b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 17.678584 | 17.608151 | 17.685972 | 17.661394 | -0.000973 ± 0.010823 |
| math | 5.147764 | 5.082358 | 5.195080 | 5.091625 | -0.010965 ± 0.020545 |
| code | 8.662200 | 8.720176 | 8.614652 | 8.866435 | +0.023304 ± 0.026131 |
