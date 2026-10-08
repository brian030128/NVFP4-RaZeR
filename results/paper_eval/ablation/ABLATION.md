# tab:ablation: how the map is chosen (Nemotron-Nano-9B-v2, Qwen3.8-27B)

Native (build_V, --kernel-set auto, paper convention, n16k64-fast), flipquant paper-sm120-runs. Every baseline flips exactly the release map's k_l tiles per layer. Cells: paired ΔNLL vs FourOverSix, x 1e-3 nats/token, ± 2 SE (* = beyond 2 SE); Random = the per-window mean of seeds 0 / 1 / 2. FourOverSix and FlipQuant are part F's records (check G0: paper-sm120-runs reproduces them bit for bit).

## WikiText-2

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | Random | -0.22 ± 0.80 | -0.61 ± 0.90 | -0.56 ± 0.89 |
| Nemotron-Nano-9B-v2 | Activation-weighted | -2.02 ± 1.04* | -0.22 ± 1.10 | +2.41 ± 0.96* |
| Nemotron-Nano-9B-v2 | One-shot gradient | +21.95 ± 2.35* | +12.60 ± 2.05* | -5.95 ± 1.45* |
| Nemotron-Nano-9B-v2 | FlipQuant | -8.63 ± 1.18* | -8.68 ± 1.13* | -5.51 ± 1.22* |
| Qwen3.8-27B | Random | +0.27 ± 2.48 | +0.42 ± 2.57 | +3.34 ± 2.61* |
| Qwen3.8-27B | Activation-weighted | -3.58 ± 4.41 | +0.57 ± 3.67 | -2.16 ± 4.05 |
| Qwen3.8-27B | One-shot gradient | -10.56 ± 5.49* | -8.09 ± 4.94* | -13.80 ± 4.94* |
| Qwen3.8-27B | FlipQuant | -22.49 ± 4.37* | -16.07 ± 4.12* | -11.40 ± 4.08* |

## C4

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | Random | +0.17 ± 0.58 | +0.28 ± 0.57 | +0.41 ± 0.60 |
| Nemotron-Nano-9B-v2 | Activation-weighted | +0.46 ± 0.77 | +0.29 ± 0.78 | +1.58 ± 0.74* |
| Nemotron-Nano-9B-v2 | One-shot gradient | +21.21 ± 1.77* | +13.37 ± 1.45* | -1.81 ± 1.00* |
| Nemotron-Nano-9B-v2 | FlipQuant | -5.16 ± 0.78* | -5.07 ± 0.86* | -3.11 ± 0.81* |
| Qwen3.8-27B | Random | -0.68 ± 0.74 | -0.14 ± 0.74 | -0.06 ± 0.68 |
| Qwen3.8-27B | Activation-weighted | -1.10 ± 0.80* | -0.28 ± 0.88 | -0.39 ± 0.76 |
| Qwen3.8-27B | One-shot gradient | +10.18 ± 1.75* | +4.76 ± 1.52* | -1.21 ± 1.06* |
| Qwen3.8-27B | FlipQuant | -4.54 ± 0.99* | -4.85 ± 0.98* | -2.85 ± 0.85* |

## PPL per cell (WikiText-2 / C4)

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | FourOverSix | 8.4260 / 11.5034 | 8.4260 / 11.5034 | 8.4260 / 11.5034 |
| Nemotron-Nano-9B-v2 | Random | 8.4242 / 11.5054 | 8.4208 / 11.5066 | 8.4212 / 11.5082 |
| Nemotron-Nano-9B-v2 | Activation-weighted | 8.4090 / 11.5087 | 8.4242 / 11.5067 | 8.4463 / 11.5216 |
| Nemotron-Nano-9B-v2 | One-shot gradient | 8.6130 / 11.7500 | 8.5328 / 11.6583 | 8.3760 / 11.4826 |
| Nemotron-Nano-9B-v2 | FlipQuant | 8.3536 / 11.4442 | 8.3532 / 11.4452 | 8.3797 / 11.4677 |
| Qwen3.8-27B | FourOverSix | 7.3023 / 10.1903 | 7.3023 / 10.1903 | 7.3023 / 10.1903 |
| Qwen3.8-27B | Random | 7.3043 / 10.1834 | 7.3054 / 10.1889 | 7.3268 / 10.1896 |
| Qwen3.8-27B | Activation-weighted | 7.2762 / 10.1791 | 7.3065 / 10.1874 | 7.2865 / 10.1864 |
| Qwen3.8-27B | One-shot gradient | 7.2256 / 10.2946 | 7.2435 / 10.2389 | 7.2023 / 10.1780 |
| Qwen3.8-27B | FlipQuant | 7.1399 / 10.1441 | 7.1859 / 10.1410 | 7.2195 / 10.1613 |

## FlipQuant vs each baseline (paired ΔNLL FlipQuant − baseline, x 1e-3, ± 2 SE; negative = FlipQuant better)

| model | unit | vs Random | vs Activation-weighted | vs One-shot (WikiText-2; C4) |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | 8x64 | -8.42 ± 1.10*; -5.34 ± 0.72* | -6.62 ± 1.29*; -5.63 ± 0.84* | -30.59 ± 2.11*; -26.37 ± 1.61* |
| Nemotron-Nano-9B-v2 | 16x64 | -8.07 ± 1.04*; -5.35 ± 0.70* | -8.46 ± 1.25*; -5.36 ± 0.87* | -21.28 ± 1.86*; -18.45 ± 1.36* |
| Nemotron-Nano-9B-v2 | 256x64 | -4.95 ± 1.10*; -3.52 ± 0.67* | -7.93 ± 1.27*; -4.69 ± 0.84* | +0.44 ± 1.26; -1.30 ± 0.90* |
| Qwen3.8-27B | 8x64 | -22.76 ± 3.67*; -3.86 ± 0.70* | -18.91 ± 4.04*; -3.44 ± 0.91* | -11.93 ± 2.95*; -14.73 ± 1.46* |
| Qwen3.8-27B | 16x64 | -16.49 ± 3.71*; -4.70 ± 0.79* | -16.64 ± 3.61*; -4.56 ± 0.97* | -7.98 ± 4.30*; -9.60 ± 1.17* |
| Qwen3.8-27B | 256x64 | -14.74 ± 3.44*; -2.78 ± 0.68* | -9.24 ± 4.27*; -2.46 ± 0.84* | +2.39 ± 3.49; -1.63 ± 0.90* |

Maps: {"nemotron-nano-9b-v2": {"8x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 0}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}, "16x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 0}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}, "256x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 4}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}}, "qwen3.8-27b": {"8x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 0}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}, "16x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 0}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}, "256x64": {"act": {"status": "complete", "windows": 256, "layers_short_of_positive": 11}, "oneshot": {"status": "complete", "steps": 32, "layers_short_of_negative": 0, "settings": {"fit_windows": 256, "teacher_topk": 1000, "epochs": 1, "lr": 0.02, "init_logit": -1.0}}}}}

