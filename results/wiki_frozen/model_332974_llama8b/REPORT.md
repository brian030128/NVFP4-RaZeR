# Frozen-map causal WikiText-2: llama8b

564 nonoverlapping 512-token test windows; 309 trailing tokens omitted. No calibration or map selection. All policies use causal per-token activation factors.

| Policy | WikiText-2 PPL |
|---|---:|
| four_over_six | 9.092934 |
| pooled192 | 9.011329 |
| c4_64 | 9.063319 |
| mixed64 | 9.033074 |
| weight_mse | 9.088662 |

Pooled192 minus FourOverSix: ΔPPL -0.081605; ΔNLL -0.009015 ±0.001816 (descriptive paired 2SE).

Pinned source weights, quantizer code and frozen maps verified. Exact prefix independence passes for baseline and pooled192. No exact overlap between nonempty WikiText rows and calibration document hashes. Adjacent windows may share articles; two-SE does not account for that dependence. These are simulated nonhead-text-linear W4A4 reference-text scores.
