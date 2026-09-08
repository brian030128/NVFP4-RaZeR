# Frozen full-model conditional selection: llama1b

One calibration pass shared by all quantizers. No per-domain tuning or acceptance gate.
Math/code are reference-text NLL, not generated task accuracy. Intervals are descriptive.

| Activations / domain | Conditional PPL | Dynamic MSE PPL | E2M1 compensated PPL | FourOverSix PPL | Conditional − dynamic ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| a16 / wiki | 19.130984 | 19.363079 | 19.390358 | 20.161726 | -0.012059 ± 0.007845 |
| a16 / math | 5.309929 | 5.606184 | 5.211527 | 4.990859 | -0.054292 ± 0.016101 |
| a16 / code | 8.194822 | 8.247864 | 8.251354 | 8.186662 | -0.006452 ± 0.021230 |
| a4 / wiki | 20.597919 | 21.440242 | 21.000416 | 21.765197 | -0.040080 ± 0.042872 |
| a4 / math | 5.409108 | 5.666008 | 5.416318 | 5.224391 | -0.046401 ± 0.022494 |
| a4 / code | 8.487273 | 8.460322 | 8.408932 | 8.532105 | +0.003181 ± 0.026119 |
