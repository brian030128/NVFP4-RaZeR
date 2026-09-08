# Frozen-map held-out C4: olmo1b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 14.984432 |
| pooled192 | 14.883008 |
| c4_64 | 14.897080 |
| mixed64 | 14.884338 |
| weight_mse | 15.078928 |

Pooled192 minus FourOverSix: ΔPPL -0.101424; ΔNLL -0.006792 ±0.002574 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
