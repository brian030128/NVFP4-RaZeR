# Online joint weight/activation error: llama1b

All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.
Reference-text losses only; runtime timings use unfused fake quantization.

| Domain | Baseline PPL | Activation-MSE PPL | Exact PPL | Compact PPL | Compact − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 20.308930 | 20.220650 | 20.154722 | 20.173142 | -0.006709 ± 0.007680 |
| math | 4.985502 | 4.984850 | 5.041502 | 5.008852 | +0.004673 ± 0.018446 |
| code | 13.224567 | 12.865097 | 12.739751 | 13.079528 | -0.011028 ± 0.027825 |
