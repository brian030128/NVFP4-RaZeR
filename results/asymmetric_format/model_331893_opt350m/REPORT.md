# Frozen full-model conditional selection: opt350m

One calibration pass shared by all quantizers. No per-domain tuning or acceptance gate.
Math/code are reference-text NLL, not generated task accuracy. Intervals are descriptive.

| Activations / domain | Conditional PPL | Dynamic MSE PPL | E2M1 compensated PPL | FourOverSix PPL | Conditional − dynamic ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| a4 / wiki | 38.761635 | 39.380679 | 39.144036 | 41.159114 | -0.015844 ± 0.008430 |
| a4 / math | 24.796660 | 25.089983 | 24.899197 | 24.528279 | -0.011760 ± 0.016430 |
| a4 / code | 32.801166 | 33.524843 | 35.487397 | 34.458601 | -0.021823 ± 0.022978 |
