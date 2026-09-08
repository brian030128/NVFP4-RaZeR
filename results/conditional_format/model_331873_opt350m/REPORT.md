# Frozen full-model conditional selection: opt350m

One calibration pass shared by all quantizers. No per-domain tuning or acceptance gate.
Math/code are reference-text NLL, not generated task accuracy. Intervals are descriptive.

| Activations / domain | Conditional PPL | Dynamic MSE PPL | E2M1 compensated PPL | FourOverSix PPL | Conditional − dynamic ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| a16 / wiki | 35.319785 | 36.354754 | 35.156772 | 36.230708 | -0.028882 ± 0.005480 |
| a16 / math | 21.699690 | 22.039955 | 21.097927 | 21.203915 | -0.015559 ± 0.012423 |
| a16 / code | 23.789729 | 23.873186 | 23.892281 | 24.827947 | -0.003502 ± 0.018197 |
| a4 / wiki | 37.107616 | 38.027027 | 36.589425 | 38.098345 | -0.024475 ± 0.010369 |
| a4 / math | 22.566879 | 22.457867 | 21.939927 | 22.062668 | +0.004842 ± 0.014602 |
| a4 / code | 25.837252 | 25.177804 | 25.915663 | 26.538466 | +0.025855 ± 0.031353 |
