# K-tile dispatch ablation at 4096³ on the RTX PRO 6000 (kernel-opt amendment 20; the paper's Figure 2(a))

What one format dispatch per K-tile buys over choosing the format per MMA. GEMM only, M = N = K = 4096 (weights N(0, 0.02), 4096 tokens N(0, 1) with per-token FourOverSix, as C2U). Kernel time by CUPTI; median of 3 rotated rounds (each the median of its calls). No clock locking and no ncu (neither is available).

## isolated

| kernel | µs | TFLOP/s | vs stock_ko | per round (µs) |
|---|---:|---:|---:|---|
| stock NVFP4 (stock_ko) | 119.74 | 1147.8 | +0.0 % | 119.74 / 120.11 / 119.70 |
| FP8 (cuBLAS, torch._scaled_mm) | 239.13 | 574.7 | +99.7 % | 239.13 / 239.12 / 239.42 |
| 16x64 per-MMA branch, real tags | 179.01 | 767.8 | +49.5 % | 179.17 / 179.01 / 178.82 |
| 16x64 per-K-tile (build_V), real tags | 121.92 | 1127.3 | +1.8 % | 121.92 / 122.62 / 121.73 |
| 16x64 per-MMA branch, e2m1 tags | 178.97 | 767.9 | +49.5 % | 178.86 / 178.97 / 179.44 |
| 16x64 per-K-tile (build_V), e2m1 tags | 121.09 | 1135.0 | +1.1 % | 121.06 / 121.09 / 121.14 |
| 8x64 per-MMA branch, real tags | 185.73 | 740.0 | +55.1 % | 185.66 / 185.74 / 185.73 |
| 8x64 per-K-tile (build_V), real tags | 127.98 | 1073.9 | +6.9 % | 127.98 / 128.16 / 127.84 |
| 8x64 per-MMA branch, e2m1 tags | 185.97 | 739.0 | +55.3 % | 185.97 / 186.13 / 185.76 |
| 8x64 per-K-tile (build_V), e2m1 tags | 126.91 | 1083.0 | +6.0 % | 126.91 / 127.09 / 126.66 |

## b2b

| kernel | µs | TFLOP/s | vs stock_ko | per round (µs) |
|---|---:|---:|---:|---|
| stock NVFP4 (stock_ko) | 102.08 | 1346.4 | +0.0 % | 102.17 / 102.03 / 102.08 |
| FP8 (cuBLAS, torch._scaled_mm) | 212.86 | 645.7 | +108.5 % | 212.54 / 212.91 / 212.86 |
| 16x64 per-MMA branch, real tags | 167.79 | 819.1 | +64.4 % | 167.92 / 167.60 / 167.79 |
| 16x64 per-K-tile (build_V), real tags | 105.89 | 1298.0 | +3.7 % | 105.87 / 105.89 / 106.32 |
| 16x64 per-MMA branch, e2m1 tags | 167.89 | 818.6 | +64.5 % | 167.82 / 168.16 / 167.89 |
| 16x64 per-K-tile (build_V), e2m1 tags | 105.65 | 1300.9 | +3.5 % | 105.65 / 105.52 / 105.76 |
| 8x64 per-MMA branch, real tags | 173.21 | 793.5 | +69.7 % | 173.21 / 173.49 / 173.17 |
| 8x64 per-K-tile (build_V), real tags | 110.17 | 1247.5 | +7.9 % | 110.17 / 110.22 / 110.14 |
| 8x64 per-MMA branch, e2m1 tags | 173.02 | 794.3 | +69.5 % | 172.89 / 173.02 / 173.17 |
| 8x64 per-K-tile (build_V), e2m1 tags | 108.99 | 1261.0 | +6.8 % | 108.96 / 109.34 / 108.99 |

## Static SASS census (C2 style; static counts, not measured counters)

| kernel | build | instructions | OMMA | predicated OMMA | BRX | WARPSYNC | OMMA per steady k-iteration | tensor-pipe estimate at 4096³ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stock_ko | stock_wA_e64 | 1528 | 64 | 0 | 0 | 3 | 32–32 | 8,388,608 |
| ktile16 | n16k64_wA_e64_t0 | 3984 | 1024 | 0 | 0 | 1 | 32–32 | 8,388,608 |
| permma16 | n16k64_wA_e64_t0_permma | 1816 | 128 | 128 | 0 | 133 | 64–64 | 16,777,216 |
| ktile8 | n8k64_wB_t0 | 4736 | 1024 | 0 | 0 | 51 | 32–32 | 8,388,608 |
| permma8 | n8k64_wB_t0_permma | 1904 | 128 | 128 | 0 | 134 | 64–64 | 16,777,216 |

Checks: {"bitwise": "pass", "timed_operands": {"permma16_real == ktile16_real": true, "permma16_e2m1 == ktile16_e2m1": true, "permma8_real == ktile8_real": true, "permma8_e2m1 == ktile8_e2m1": true}, "sass_gates": {"build_KT_ref/n16k64_wA_e64_t0 == build_V": true, "build_KT_ref/n8k64_wB_t0 == build_V": true, "build_KT_ref/stock_wA_e64 == build_V": true, "build_KT_nodef/n16k64_wA_e64_t0_permma == build_KT": true, "build_KT_nodef/n8k64_wB_t0_permma == build_KT": true}}

