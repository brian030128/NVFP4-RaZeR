# Frozen-map held-out C4: llama1b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 24.898634 |
| pooled192 | 24.124422 |
| c4_64 | 24.095924 |
| mixed64 | 24.159976 |
| weight_mse | 25.239483 |

Pooled192 minus FourOverSix: ΔPPL -0.774213; ΔNLL -0.031588 ±0.002935 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
