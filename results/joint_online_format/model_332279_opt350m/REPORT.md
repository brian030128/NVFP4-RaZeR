# Online joint weight/activation error: opt350m

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 33.965217 | 33.768375 | 33.574234 | 33.897167 | -0.002006 ± 0.007981 |
| math | 22.398254 | 22.115499 | 22.090023 | 22.019267 | -0.017065 ± 0.014656 |
| code | 47.694444 | 47.353819 | 47.702991 | 47.302640 | -0.008249 ± 0.024592 |
