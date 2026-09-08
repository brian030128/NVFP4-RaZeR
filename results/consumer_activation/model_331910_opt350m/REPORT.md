# Consumer-aware activation formats: opt350m

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Consumer PPL | Consumer − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|
| wiki | 40.908755 | 40.823613 | 41.015390 | +0.002603 ± 0.006841 |
| math | 22.488153 | 22.687345 | 22.363571 | -0.005555 ± 0.016073 |
| code | 29.730208 | 30.258608 | 30.105614 | +0.012548 ± 0.019703 |
