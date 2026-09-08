# Frozen-map held-out C4: opt350m

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 26.743945 |
| pooled192 | 26.469613 |
| c4_64 | 26.353516 |
| mixed64 | 26.475330 |
| weight_mse | 26.993510 |

Pooled192 minus FourOverSix: ΔPPL -0.274332; ΔNLL -0.010311 ±0.002727 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
