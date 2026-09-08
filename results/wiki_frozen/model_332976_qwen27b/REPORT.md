# Frozen-map causal WikiText-2: qwen27b

580 nonoverlapping 512-token test windows; 233 trailing tokens omitted. No calibration or map selection. All policies use causal per-token activation factors.

| Policy | WikiText-2 PPL |
|---|---:|
| four_over_six | 9.040908 |
| pooled192 | 8.997140 |
| c4_64 | 9.024685 |
| mixed64 | 9.010103 |
| weight_mse | 9.026329 |

Pooled192 minus FourOverSix: ΔPPL -0.043768; ΔNLL -0.004853 ±0.001499 (descriptive paired 2SE).

Pinned source weights, quantizer code and frozen maps verified. Exact prefix independence passes for baseline and pooled192. No exact overlap between nonempty WikiText rows and calibration document hashes. Adjacent windows may share articles; two-SE does not account for that dependence. These are simulated nonhead-text-linear W4A4 reference-text scores.
