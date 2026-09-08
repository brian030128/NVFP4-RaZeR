# Frozen-map causal replay: qwen06b

All policies use one FP32 activation factor per token; maps and inputs replay unchanged.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 42.286833 | 41.739308 | 38.223699 | 38.276060 | 38.234706 | -0.100732 ±0.010663 |
| science | 22.977852 | 22.997106 | 21.447180 | 21.411143 | 21.385315 | -0.071826 ±0.009366 |
| government | 20.965000 | 21.118993 | 19.356694 | 19.291834 | 19.293438 | -0.083089 ±0.008691 |

Suffix intervention:

```json
{
  "four_over_six_window": {
    "equal": false,
    "max_logit_difference": 9.3203125,
    "prefix_tokens": 128,
    "replacement_token_id": 151645
  },
  "four_over_six_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 151645
  },
  "pooled192_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 151645
  }
}
```
