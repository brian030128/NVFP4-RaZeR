# Description-cost tile election: opt350m

```json
{
  "tiles": 591872,
  "tokens": 32704,
  "penalty": 0.00040640428309294976,
  "prior_probability": 1.6895516436803167e-06,
  "selected": 4,
  "predicted_upper": -0.0018965151393786073,
  "relative_description_nats": 53.16418269708732
}
```

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Description | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 32.808285 | 35.469627 | 32.861153 | 33.541415 | 32.717641 | -0.002767 ±0.004910 |
| math | 23.045412 | 24.815565 | 23.519085 | 24.260634 | 23.042420 | -0.000130 ±0.009097 |
| code | 29.806597 | 31.932998 | 29.188449 | 29.243306 | 29.558734 | -0.008350 ±0.015673 |

Fitting audit:

```json
{
  "four_over_six": {
    "ce": 3.2390636168420315,
    "kl": 0.14932967722415924
  },
  "description": {
    "ce": 3.2362570017576218,
    "kl": 0.14675874006934464
  }
}
```
