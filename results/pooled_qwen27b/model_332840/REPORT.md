# Qwen3.8-27B: unchanged pooled rule, held-out C4

One shared 192-sequence score table; same 256-tile cap; no backtracking or acceptance gate.

| Policy | C4 PPL |
|---|---:|
| four_over_six | 12.644977 |
| pooled192 | 12.635323 |
| c4_64 | 12.630390 |
| mixed64 | 12.627442 |
| weight_mse | 12.671271 |

Pooled192 minus FourOverSix: ΔPPL -0.009654; ΔNLL -0.000764 ±0.001538 (descriptive paired 2SE).

C4 is a calibration source; this is held-out within-source evaluation. 256 validation documents, 512 tokens each; zero calibration hash overlap. Both policies pass exact prefix independence with causal activation factors. Native Transformers 5.16.1 hybrid text pathway; simulated nonhead-linear W4A4. One calibration pool; no claim of simultaneous confidence, native speed or a universal guarantee.
