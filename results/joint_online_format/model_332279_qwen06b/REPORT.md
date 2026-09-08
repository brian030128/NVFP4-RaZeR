# Online joint weight/activation error: qwen06b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 34.277518 | 34.219095 | 34.013255 | 34.492931 | +0.006265 ± 0.014464 |
| math | 6.200327 | 6.170410 | 6.105415 | 6.112708 | -0.014232 ± 0.027105 |
| code | 16.097043 | 16.308591 | 15.820565 | 16.145537 | +0.003008 ± 0.057754 |
