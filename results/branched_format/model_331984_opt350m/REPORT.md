# Columnwise compensation with legal tile branches: opt350m

Shared calibration and quantization backend. Original global scales; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | E2M1 GPTQ PPL | Dynamic-MSE PPL | Conditional PPL | Conditional − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 47.083270 | 45.211222 | 46.296495 | 44.847354 | -0.048653 ± 0.011033 |
| math | 23.454313 | 24.170157 | 24.838080 | 24.321913 | +0.036323 ± 0.023283 |
| code | 26.482410 | 24.989017 | 24.791575 | 25.879119 | -0.023044 ± 0.024408 |
