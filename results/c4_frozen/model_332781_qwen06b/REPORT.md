# Frozen-map held-out C4: qwen06b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 37.900671 |
| pooled192 | 34.792433 |
| c4_64 | 34.894622 |
| mixed64 | 34.836806 |
| weight_mse | 37.668164 |

Pooled192 minus FourOverSix: ΔPPL -3.108238; ΔNLL -0.085569 ±0.005095 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
