# Consumer-aware activation formats: llama1b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Consumer PPL | Consumer − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|
| wiki | 22.450639 | 22.485991 | 22.529903 | +0.003524 ± 0.007596 |
| math | 5.165500 | 5.140004 | 5.176462 | +0.002120 ± 0.011268 |
| code | 10.357450 | 10.294096 | 10.285412 | -0.006979 ± 0.026220 |
