# Columnwise compensation with legal tile branches: llama1b

Shared calibration and quantization backend. Original global scales; fresh evaluation.
Reference-text loss only; descriptive paired intervals.

| Domain | FourOverSix PPL | E2M1 GPTQ PPL | Dynamic-MSE PPL | Conditional PPL | Conditional − baseline ΔNLL ± 2SE |
|---|---:|---:|---:|---:|---:|
| wiki | 29.349060 | 27.724723 | 27.728468 | 27.955643 | -0.048641 ± 0.010666 |
| math | 5.528809 | 5.768449 | 5.879270 | 5.816338 | +0.050698 ± 0.024825 |
| code | 9.132288 | 9.348577 | 9.223245 | 8.757167 | -0.041944 ± 0.031410 |
