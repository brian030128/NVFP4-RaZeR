# Description-cost tile election: qwen06b

```json
{
  "tiles": 860160,
  "tokens": 32704,
  "penalty": 0.0004178349344862039,
  "prior_probability": 1.1625730531842295e-06,
  "selected": 16,
  "predicted_upper": -0.014495699666440487,
  "relative_description_nats": 218.637979158989
}
```

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Description | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 33.625020 | 33.588173 | 30.387148 | 31.417902 | 33.071347 | -0.016603 ±0.012309 |
| math | 6.839486 | 6.566429 | 6.376903 | 6.302515 | 6.544409 | -0.044101 ±0.023127 |
| code | 11.351887 | 11.353100 | 10.467990 | 10.779866 | 10.845869 | -0.045600 ±0.027755 |

Fitting audit:

```json
{
  "four_over_six": {
    "ce": 3.4820074811577797,
    "kl": 0.23626128933392465
  },
  "description": {
    "ce": 3.4621789529919624,
    "kl": 0.22984575014561415
  }
}
```
