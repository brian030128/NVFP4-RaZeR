# Frozen-map causal WikiText-2: qwen4b

584 nonoverlapping 512-token test windows; 70 trailing tokens omitted. No calibration or map selection. All policies use causal per-token activation factors.

| Policy | WikiText-2 PPL |
|---|---:|
| four_over_six | 19.414897 |
| pooled192 | 17.521211 |
| c4_64 | 18.095543 |
| mixed64 | 17.536654 |
| weight_mse | 20.108009 |

Pooled192 minus FourOverSix: ΔPPL -1.893686; ΔNLL -0.102628 ±0.004027 (descriptive paired 2SE).

Pinned source weights, quantizer code and frozen maps verified. Exact prefix independence passes for baseline and pooled192. No exact overlap between nonempty WikiText rows and calibration document hashes. Adjacent windows may share articles; two-SE does not account for that dependence. These are simulated nonhead-text-linear W4A4 reference-text scores.
