# Relinearized format updates: qwen06b

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Relinearized | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 54.319533 | 55.013682 | 48.619943 | 49.565979 | 49.372150 | -0.095497 ±0.012305 |
| math | 6.646559 | 6.505477 | 6.333372 | 6.385816 | 6.180158 | -0.072755 ±0.033804 |
| code | 9.053609 | 9.377604 | 8.671262 | 8.730965 | 8.442356 | -0.069902 ±0.035668 |

Fit audits do not change the final map.

```json
{
  "four_over_six": {
    "ce": 3.4820074811577797,
    "kl": 0.23626128933392465
  },
  "stale256": {
    "ce": 3.403717018663883,
    "kl": 0.2257964895106852
  },
  "relinearized": {
    "ce": 3.409776981920004,
    "kl": 0.21228108066134155
  }
}
```
