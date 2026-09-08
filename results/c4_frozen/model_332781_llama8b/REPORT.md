# Frozen-map held-out C4: llama8b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 11.600109 |
| pooled192 | 11.509863 |
| c4_64 | 11.545653 |
| mixed64 | 11.524583 |
| weight_mse | 11.635797 |

Pooled192 minus FourOverSix: ΔPPL -0.090246; ΔNLL -0.007810 ±0.002113 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
