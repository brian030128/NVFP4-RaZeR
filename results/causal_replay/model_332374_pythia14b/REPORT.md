# Frozen-map causal replay: pythia14b

All policies use one FP32 activation factor per token; maps and inputs replay unchanged.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 17.355897 | 17.565628 | 17.143354 | 17.167397 | 17.085960 | -0.015675 ±0.005223 |
| science | 19.709933 | 20.054052 | 19.447926 | 19.591546 | 19.468234 | -0.012339 ±0.006259 |
| government | 14.187066 | 14.372136 | 13.976362 | 13.990906 | 13.978634 | -0.014801 ±0.005609 |

Suffix intervention:

```json
{
  "four_over_six_window": {
    "equal": false,
    "max_logit_difference": 8.6875,
    "prefix_tokens": 128,
    "replacement_token_id": 0
  },
  "four_over_six_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 0
  },
  "pooled192_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 0
  }
}
```
