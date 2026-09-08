# Frozen full-model conditional selection: llama1b

One calibration pass shared by all quantizers. No per-domain tuning or acceptance gate.
Math/code are reference-text NLL, not generated task accuracy. Intervals are descriptive.

| Activations / domain | Conditional PPL | Dynamic MSE PPL | E2M1 compensated PPL | FourOverSix PPL | Conditional − dynamic ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| a4 / wiki | 24.259884 | 24.790092 | 24.315274 | 24.354559 | -0.021620 ± 0.016235 |
| a4 / math | 5.784034 | 5.877863 | 5.788992 | 5.383435 | -0.016092 ± 0.017358 |
| a4 / code | 10.184309 | 10.429528 | 10.832263 | 9.448651 | -0.023793 ± 0.037227 |
