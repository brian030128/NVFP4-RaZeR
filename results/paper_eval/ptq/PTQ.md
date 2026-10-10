# tab:ptq: FlipQuant with PTQ methods (16x64; PPL and GSM8K)

Native (build_V, --kernel-set auto), n16k64-fast, flipquant paper-sm120-runs. RTN: part F's records. GPTQ: fixed RTN grid, block 128, damp 0.01, no act-order, the release 256 x 512 fit set, BF16 propagation (the user's decision). Hadamard: block 16, applied in PyTorch before the native kernel (not fused); FlipQuant's map retrained in the rotated basis with the release settings. NVFP4 rows: NVFP4 activations (per-token scales); the others: per-token FourOverSix. Cells: WikiText-2 / C4 PPL; ΔNLL vs RTN FourOverSix x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: lm-eval gsm8k_llama (8-shot CoT, chat template, greedy, 1024 tokens), thinking off, strict-match accuracy (%) ± 2 SE, 1,319 problems (part N).

| model | method | format | WikiText-2 | C4 | ΔNLL wiki | ΔNLL C4 | GSM8K (%) |
|---|---|---|---:|---:|---:|---:|---:|
| Nemotron-Nano-9B-v2 | rtn | NVFP4 | 8.4535 | 11.5437 | +3.26 ± 1.51* | +3.50 ± 1.00* | 87.34 ± 1.83 |
| Nemotron-Nano-9B-v2 | rtn | FourOverSix | 8.4260 | 11.5034 | +0.00 ± 0.00 | +0.00 ± 0.00 | 88.93 ± 1.73 |
| Nemotron-Nano-9B-v2 | rtn | FlipQuant (16$\times$64) | 8.3532 | 11.4452 | -8.68 ± 1.13* | -5.07 ± 0.86* | 89.31 ± 1.70 |
| Nemotron-Nano-9B-v2 | gptq | NVFP4 | 8.4463 | 11.5301 | +2.41 ± 1.66* | +2.32 ± 1.13* | 84.00 ± 2.02 |
| Nemotron-Nano-9B-v2 | gptq | FourOverSix | 8.3949 | 11.4847 | -3.70 ± 1.44* | -1.63 ± 0.99* | 86.13 ± 1.90 |
| Nemotron-Nano-9B-v2 | gptq | FlipQuant (16$\times$64) | 8.3962 | 11.4868 | -3.54 ± 1.44* | -1.45 ± 1.00* | 85.60 ± 1.93 |
| Nemotron-Nano-9B-v2 | hadamard | NVFP4 | 8.5178 | 11.6449 | +10.84 ± 2.04* | +12.23 ± 1.32* | 84.91 ± 1.97 |
| Nemotron-Nano-9B-v2 | hadamard | FourOverSix | 8.4535 | 11.5580 | +3.26 ± 1.91* | +4.73 ± 1.17* | 88.17 ± 1.78 |
| Nemotron-Nano-9B-v2 | hadamard | FlipQuant (16$\times$64) | 8.3727 | 11.4953 | -6.34 ± 1.84* | -0.70 ± 1.13 | 88.93 ± 1.73 |
| Qwen3.8-27B | rtn | NVFP4 | 7.5740 | 10.2231 | +36.53 ± 8.12* | +3.22 ± 1.08* | 95.38 ± 1.16 |
| Qwen3.8-27B | rtn | FourOverSix | 7.3023 | 10.1903 | +0.00 ± 0.00 | +0.00 ± 0.00 | 94.62 ± 1.24 |
| Qwen3.8-27B | rtn | FlipQuant (16$\times$64) | 7.1859 | 10.1410 | -16.07 ± 4.12* | -4.85 ± 0.98* | 95.15 ± 1.18 |
| Qwen3.8-27B | gptq | NVFP4 | 7.5248 | 10.1727 | +30.01 ± 7.02* | -1.73 ± 1.33* | 95.83 ± 1.10 |
| Qwen3.8-27B | gptq | FourOverSix | 7.3296 | 10.1424 | +3.73 ± 5.34 | -4.71 ± 1.08* | 96.59 ± 1.00 |
| Qwen3.8-27B | gptq | FlipQuant (16$\times$64) | 7.3120 | 10.1480 | +1.32 ± 5.11 | -4.16 ± 1.08* | 96.59 ± 1.00 |
| Qwen3.8-27B | hadamard | NVFP4 | 7.6126 | 10.3425 | +41.61 ± 11.36* | +14.83 ± 2.14* | 95.75 ± 1.11 |
| Qwen3.8-27B | hadamard | FourOverSix | 7.7335 | 10.2574 | +57.36 ± 12.87* | +6.57 ± 1.46* | 96.21 ± 1.05 |
| Qwen3.8-27B | hadamard | FlipQuant (16$\times$64) | 7.3684 | 10.1923 | +9.01 ± 6.60* | +0.20 ± 1.30 | 95.75 ± 1.11 |

Check H1 (NVFP4 GPTQ codes under NVFP4 / FourOverSix activations): {"nemotron-nano-9b-v2": {"nvfp4_act": "22854fc2fac91247772fec16656384a0854eff405f2ff188e4153e0643c2f51b", "fo6_act": "22854fc2fac91247772fec16656384a0854eff405f2ff188e4153e0643c2f51b", "equal": true}, "qwen3.8-27b": {"nvfp4_act": "3c1bc3b2cfd80a11d9585c03f9c798f47fd4194813d47c659f9d6862e10e8f69", "fo6_act": "3c1bc3b2cfd80a11d9585c03f9c798f47fd4194813d47c659f9d6862e10e8f69", "equal": true}}

