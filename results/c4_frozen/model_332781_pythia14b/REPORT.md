# Frozen-map held-out C4: pythia14b

256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 20.471159 |
| pooled192 | 20.162666 |
| c4_64 | 20.130890 |
| mixed64 | 20.211494 |
| weight_mse | 20.504948 |

Pooled192 minus FourOverSix: ΔPPL -0.308493; ΔNLL -0.015184 ±0.003167 (descriptive 2SE).

Source weights verified; frozen map unchanged; calibration hash overlap zero; baseline and selected prefix independence exact.
