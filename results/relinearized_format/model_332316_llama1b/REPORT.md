# Relinearized format updates: llama1b

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Relinearized | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 31.957872 | 32.577889 | 30.813041 | 30.458388 | 30.905398 | -0.033488 ±0.010213 |
| math | 5.626792 | 5.867444 | 5.563909 | 5.549948 | 5.574183 | -0.009394 ±0.018445 |
| code | 7.614446 | 7.412671 | 7.402233 | 7.353626 | 7.375151 | -0.031931 ±0.020728 |

Fit audits do not change the final map.

```json
{
  "four_over_six": {
    "ce": 3.136334463953972,
    "kl": 0.14514377585146576
  },
  "stale256": {
    "ce": 3.095052868127823,
    "kl": 0.12609942012932152
  },
  "relinearized": {
    "ce": 3.0991753898561,
    "kl": 0.12408990110270679
  }
}
```
