# Columnwise compensation with legal tile branches: qwen06b

Shared calibration and quantization backend. Original global scales; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | E2M1 GPTQ PPL | Dynamic-MSE PPL | Conditional PPL | Conditional − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 46.046110 | 42.965890 | 42.679809 | 42.783470 | -0.073491 ± 0.024242 |
| math | 6.130682 | 5.745733 | 5.516949 | 5.515821 | -0.105685 ± 0.032731 |
| code | 10.417516 | 11.088232 | 10.449348 | 10.339067 | -0.007559 ± 0.037919 |
