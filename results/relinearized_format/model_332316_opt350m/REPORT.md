# Relinearized format updates: opt350m

| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Relinearized | ΔNLL ±2SE vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 46.070083 | 48.593232 | 45.889291 | 46.356368 | 45.304715 | -0.016753 ±0.010158 |
| math | 24.865662 | 27.084608 | 25.539327 | 25.961796 | 25.865252 | +0.039413 ±0.021171 |
| code | 22.822859 | 24.274790 | 22.257883 | 22.787557 | 22.282037 | -0.023982 ±0.021396 |

Fit audits do not change the final map.

```json
{
  "four_over_six": {
    "ce": 3.2390636168420315,
    "kl": 0.14932967722415924
  },
  "stale256": {
    "ce": 3.218919552862644,
    "kl": 0.13422120129689574
  },
  "relinearized": {
    "ce": 3.221851732581854,
    "kl": 0.12754930299706757
  }
}
```
