# Description-cost tile election: llama1b

```json
{
  "tiles": 1900544,
  "tokens": 32704,
  "penalty": 0.0004420759148405577,
  "prior_probability": 5.26164863236598e-07,
  "selected": 7,
  "predicted_upper": -0.005507063120603561,
  "relative_description_nats": 101.20355503261919
}
```

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Description | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 19.338963 | 19.771430 | 18.805402 | 18.482783 | 19.206605 | -0.006868 ±0.008274 |
| math | 5.380593 | 5.554113 | 5.182820 | 5.201706 | 5.301492 | -0.014810 ±0.017414 |
| code | 9.006729 | 8.960552 | 8.881586 | 9.048693 | 9.063867 | +0.006324 ±0.018110 |

Fitting audit:

```json
{
  "four_over_six": {
    "ce": 3.136334463953972,
    "kl": 0.14514377585146576
  },
  "description": {
    "ce": 3.1291199401021004,
    "kl": 0.13930559426080436
  }
}
```
