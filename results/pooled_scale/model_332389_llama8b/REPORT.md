# Pooled-source causal scale transfer: llama8b

Window-factor scoring, frozen maps, per-token-factor evaluation. No recalibration or acceptance gate.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 11.675033 | 11.678533 | 11.630268 | 11.603090 | 11.591275 | -0.007200 ±0.004854 |
| science | 10.804980 | 10.750623 | 10.747964 | 10.739562 | 10.701076 | -0.009663 ±0.004517 |
| government | 8.251998 | 8.247904 | 8.208483 | 8.175724 | 8.181400 | -0.008592 ±0.003871 |

Fit audits do not change any map.

```json
{
  "four_over_six": {
    "web": {
      "ce": 2.3944133864715695,
      "kl": 0.11626673967111856
    },
    "math": {
      "ce": 1.919922836124897,
      "kl": 0.10707106476183981
    },
    "code": {
      "ce": 1.1544683251995593,
      "kl": 0.08694364821712952
    }
  },
  "pooled192": {
    "web": {
      "ce": 2.3868165221065283,
      "kl": 0.11006618750980124
    },
    "math": {
      "ce": 1.9073908356949687,
      "kl": 0.09767313982592896
    },
    "code": {
      "ce": 1.1494777972111478,
      "kl": 0.08052783957100473
    }
  }
}
```
