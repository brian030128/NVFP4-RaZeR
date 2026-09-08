# Frozen-map causal replay: olmo1b

All policies use one FP32 activation factor per token; maps and inputs replay unchanged.

| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |
|---|---:|---:|---:|---:|---:|---:|
| literature | 18.361552 | 18.382374 | 18.297960 | 18.286123 | 18.272085 | -0.004884 ±0.004963 |
| science | 21.505466 | 21.862918 | 21.157830 | 21.036540 | 21.111510 | -0.018489 ±0.013632 |
| government | 12.176172 | 12.145086 | 12.097583 | 12.102692 | 12.085915 | -0.007440 ±0.003889 |

Suffix intervention:

```json
{
  "four_over_six_window": {
    "equal": false,
    "max_logit_difference": 5.125,
    "prefix_tokens": 128,
    "replacement_token_id": 50279
  },
  "four_over_six_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 50279
  },
  "pooled192_row": {
    "equal": true,
    "max_logit_difference": 0.0,
    "prefix_tokens": 128,
    "replacement_token_id": 50279
  }
}
```
