# Frozen-map held-out C4: qwen4b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 21.928243 |
| pooled192 | 20.902742 |
| c4_64 | 21.046875 |
| mixed64 | 20.880267 |
| weight_mse | 22.082958 |

Pooled192 minus FourOverSix: ΔPPL -1.025501; ΔNLL -0.047895 ±0.003392 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
