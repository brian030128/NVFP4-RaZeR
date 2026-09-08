# Frozen-map causal replay: llama1b

All policies use one FP32 activation factor per token; maps and inputs replay unchanged.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 27.428086 | 27.563328 | 26.379952 | 26.272713 | 26.170161 | -0.046948 ±0.006983 |
| science | 22.997938 | 23.512534 | 22.219271 | 22.114276 | 22.191191 | -0.035709 ±0.005738 |
| government | 15.939049 | 16.060804 | 15.388088 | 15.416728 | 15.437538 | -0.031970 ±0.006389 |

Suffix intervention:

```json
{
  "four_over_six_window": {
    "equal": false,
    "max_logit_difference": 4.796875,
    "prefix_tokens": 128,
    "replacement_token_id": 128009
  },
  "four_over_six_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 128009
  },
  "pooled192_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 128009
  }
}
```
