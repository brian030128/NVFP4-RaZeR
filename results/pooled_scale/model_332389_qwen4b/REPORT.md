# Pooled-source causal scale transfer: qwen4b

Window-factor scoring, frozen maps, per-token-factor evaluation. No recalibration or acceptance gate.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 21.779775 | 21.969228 | 20.726838 | 20.620582 | 20.537961 | -0.058707 ±0.007401 |
| science | 13.656461 | 13.850560 | 13.244517 | 13.021772 | 13.018419 | -0.047848 ±0.005846 |
| government | 12.757487 | 12.829839 | 12.183027 | 12.095631 | 12.081077 | -0.054478 ±0.006661 |

Fit audits do not change any map.

```json
{
  "four_over_six": {
    "web": {
      "ce": 2.948136603459716,
      "kl": 0.09921195427887142
    },
    "math": {
      "ce": 2.1786333452910185,
      "kl": 0.1102486940799281
    },
    "code": {
      "ce": 1.4129606438800693,
      "kl": 0.10992531111696735
    }
  },
  "pooled192": {
    "web": {
      "ce": 2.9010900165885687,
      "kl": 0.09592187660746276
    },
    "math": {
      "ce": 2.137887017801404,
      "kl": 0.10428709862753749
    },
    "code": {
      "ce": 1.375781495589763,
      "kl": 0.10332473009475507
    }
  }
}
```
