# Frozen-map causal replay: opt350m

All policies use one FP32 activation factor per token; maps and inputs replay unchanged.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 26.780318 | 27.291743 | 26.498300 | 26.438152 | 26.318007 | -0.017414 ±0.005225 |
| science | 39.988506 | 41.864586 | 39.349138 | 38.937945 | 38.978981 | -0.025570 ±0.007068 |
| government | 19.305159 | 19.579188 | 18.983213 | 19.084383 | 19.041959 | -0.013727 ±0.005354 |

Suffix intervention:

```json
{
  "four_over_six_window": {
    "equal": false,
    "max_logit_difference": 5.71875,
    "prefix_tokens": 128,
    "replacement_token_id": 2
  },
  "four_over_six_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 2
  },
  "pooled192_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 2
  }
}
```
