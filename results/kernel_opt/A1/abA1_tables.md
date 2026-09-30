## M1: per-forward GEMM time of the 256x64 maps, A' (4-arm g32) vs today's path, deviation-2 method

### Llama-3.1-8B

Checks: 588 bitwise ok=True, counts ok=True, other processes 0; every call at the same width before and after: True.

| T | after vs before, typical [rounds] | after vs before, worst | 256x64 vs stock_wA, typical: before → after | worst: before → after |
|---|---|---|---|---|
| 1 | -0.99 % [-1.21, -0.92] | -0.85 % | +1.1 → +0.1 % | +1.1 → +0.3 % |
| 4 | -0.80 % [-1.05, -0.69] | -1.02 % | +1.2 → +0.4 % | +1.1 → +0.1 % |
| 16 | -0.96 % [-1.07, -0.71] | -1.08 % | +1.2 → +0.3 % | +1.4 → +0.3 % |
| 32 | -0.85 % [-1.00, -0.73] | -1.20 % | +1.2 → +0.4 % | +1.5 → +0.3 % |
| 64 | -1.13 % [-1.28, -0.96] | -1.44 % | +2.0 → +0.8 % | +2.2 → +0.7 % |
| 128 | -0.03 % [-0.13, +0.06] | -0.05 % | +3.8 → +3.7 % | +3.8 → +3.7 % |
| 256 | -1.56 % [-1.63, -1.29] | -2.27 % | +2.8 → +1.2 % | +3.6 → +1.3 % |
| 512 | -1.26 % [-1.37, -1.10] | -2.39 % | +5.6 → +4.2 % | +6.9 → +4.4 % |
| 1024 | -1.30 % [-1.33, -1.07] | -2.06 % | +3.1 → +1.7 % | +3.6 → +1.4 % |
| 2048 | -1.49 % [-1.62, -1.24] | -2.05 % | +4.4 → +2.9 % | +4.8 → +2.7 % |
| 4096 | -1.66 % [-1.93, -0.99] | -2.18 % | +3.5 → +1.8 % | +4.1 → +1.8 % |
| 8192 | -1.37 % [-1.44, -1.27] | -1.68 % | +4.0 → +2.6 % | +4.4 → +2.7 % |

### Mistral-7B-v0.3

Checks: 588 bitwise ok=True, counts ok=True, other processes 0; every call at the same width before and after: True.

| T | after vs before, typical [rounds] | after vs before, worst | 256x64 vs stock_wA, typical: before → after | worst: before → after |
|---|---|---|---|---|
| 1 | -0.91 % [-1.03, -0.74] | -1.05 % | +1.2 → +0.3 % | +1.3 → +0.3 % |
| 4 | -0.91 % [-1.06, -0.82] | -1.20 % | +1.3 → +0.4 % | +1.4 → +0.2 % |
| 16 | -1.17 % [-1.50, -0.91] | -1.01 % | +1.4 → +0.2 % | +1.3 → +0.3 % |
| 32 | -1.02 % [-1.11, -0.73] | -1.12 % | +1.4 → +0.4 % | +1.3 → +0.2 % |
| 64 | -1.14 % [-1.28, -1.14] | -1.41 % | +1.9 → +0.8 % | +2.3 → +0.8 % |
| 128 | -0.07 % [-0.17, +0.06] | +0.03 % | +3.7 → +3.6 % | +3.6 → +3.7 % |
| 256 | -1.84 % [-1.89, -1.71] | -2.64 % | +3.1 → +1.2 % | +3.8 → +1.1 % |
| 512 | -1.43 % [-1.55, -1.26] | -2.21 % | +5.8 → +4.2 % | +6.7 → +4.3 % |
| 1024 | -1.26 % [-1.63, -1.04] | -1.90 % | +3.0 → +1.7 % | +3.5 → +1.5 % |
| 2048 | -1.69 % [-1.61, -1.49] | -1.90 % | +4.4 → +2.6 % | +4.5 → +2.5 % |
| 4096 | -1.63 % [-1.73, -1.63] | -2.07 % | +3.5 → +1.8 % | +3.9 → +1.8 % |
| 8192 | -1.34 % [-1.36, -1.28] | -1.73 % | +4.1 → +2.8 % | +4.5 → +2.7 % |

### Phi-4

Checks: 336 bitwise ok=True, counts ok=True, other processes 0; every call at the same width before and after: True.

| T | after vs before, typical [rounds] | after vs before, worst | 256x64 vs stock_wA, typical: before → after | worst: before → after |
|---|---|---|---|---|
| 1 | -0.12 % [-0.30, -0.10] | -0.87 % | +0.0 → -0.1 % | +0.4 → -0.5 % |
| 4 | -0.17 % [-0.28, -0.11] | -0.63 % | -0.1 → -0.3 % | +0.1 → -0.5 % |
| 16 | -0.24 % [-0.41, -0.18] | -0.80 % | +0.2 → -0.0 % | +0.5 → -0.3 % |
| 32 | -0.48 % [-0.51, -0.36] | -0.76 % | -0.1 → -0.6 % | +0.1 → -0.7 % |
| 64 | -0.42 % [-0.58, -0.06] | -0.85 % | +0.6 → +0.1 % | +0.8 → -0.1 % |
| 128 | -1.09 % [-1.19, -0.65] | -1.62 % | +1.6 → +0.5 % | +1.9 → +0.2 % |
| 256 | -1.70 % [-1.81, -1.52] | -1.69 % | +4.0 → +2.3 % | +4.2 → +2.4 % |
| 512 | -1.65 % [-1.75, -1.54] | -2.07 % | +3.9 → +2.2 % | +4.3 → +2.2 % |
| 1024 | -1.52 % [-1.77, -1.46] | -1.62 % | +3.9 → +2.4 % | +4.1 → +2.4 % |
| 2048 | -1.52 % [-1.63, -1.29] | -1.41 % | +3.4 → +1.9 % | +3.4 → +2.0 % |
| 4096 | -1.47 % [-1.65, -1.25] | -1.38 % | +3.8 → +2.2 % | +3.7 → +2.3 % |
| 8192 | -1.26 % [-1.33, -1.04] | -1.25 % | +3.5 → +2.2 % | +3.7 → +2.4 % |

### Qwen3.8-27B

Checks: 1008 bitwise ok=True, counts ok=True, other processes 0; every call at the same width before and after: True.

| T | after vs before, typical [rounds] | after vs before, worst | 256x64 vs stock_wA, typical: before → after | worst: before → after |
|---|---|---|---|---|
| 1 | -0.86 % [-0.90, -0.79] | -0.75 % | +1.2 → +0.3 % | +1.1 → +0.3 % |
| 4 | -0.74 % [-0.82, -0.57] | -0.71 % | +1.2 → +0.4 % | +1.1 → +0.4 % |
| 16 | -0.89 % [-0.96, -0.80] | -0.84 % | +1.3 → +0.4 % | +1.2 → +0.3 % |
| 32 | -0.88 % [-1.04, -0.72] | -0.74 % | +1.0 → +0.1 % | +0.9 → +0.2 % |
| 64 | -0.83 % [-0.90, -0.82] | -0.64 % | +1.6 → +0.8 % | +1.5 → +0.8 % |
| 128 | -1.31 % [-1.41, -1.17] | -1.61 % | +2.2 → +0.8 % | +2.5 → +0.8 % |
| 256 | -1.94 % [-2.07, -1.81] | -2.36 % | +4.0 → +2.0 % | +4.8 → +2.3 % |
| 512 | -1.62 % [-1.78, -1.45] | -2.36 % | +4.4 → +2.7 % | +5.2 → +2.7 % |
| 1024 | -1.50 % [-1.76, -1.26] | -2.14 % | +3.9 → +2.3 % | +4.6 → +2.4 % |
| 2048 | -1.26 % [-1.33, -1.21] | -1.78 % | +3.3 → +2.0 % | +3.8 → +1.9 % |
| 4096 | -1.31 % [-1.36, -1.19] | -1.82 % | +3.5 → +2.1 % | +4.0 → +2.1 % |
| 8192 | -1.07 % [-1.10, -1.02] | -1.43 % | +3.4 → +2.3 % | +3.9 → +2.4 % |

## C2: 4096³ breakdown (µs, medians of 3 rotated rounds)

Checks: {'g32_e2m1 == default_e2m1': True, 'g32_real == default_real': True, 'g32_e0m3 == default_e0m3': True}

| configuration | b2b | isolated | sustained |
|---|---:|---:|---:|
| stock_wA | 103.0 | 115.1 | 137.9 |
| nodisp | 103.9 | 115.9 | 138.9 |
| default_e2m1 | 106.3 | 118.4 | 140.8 |
| g32_e2m1 | 105.3 | 116.8 | 139.4 |
| default_real | 106.9 | 118.8 | 142.4 |
| g32_real | 105.4 | 117.1 | 140.4 |
| default_e0m3 | 110.9 | 124.6 | 147.1 |
| g32_e0m3 | 104.8 | 116.5 | 140.4 |

- **b2b:** g32 vs default, e2m1 -0.99 %; g32 vs default, real -1.39 %; g32 vs default, e0m3 -5.57 %; default_e2m1 vs nodisp +2.34 %; default_real vs nodisp +2.86 %; default_e0m3 vs nodisp +6.78 %; g32_e2m1 vs nodisp +1.32 %; g32_real vs nodisp +1.43 %; g32_e0m3 vs nodisp +0.83 %; default_e2m1 vs stock_wA +3.26 %; default_real vs stock_wA +3.79 %; default_e0m3 vs stock_wA +7.74 %; g32_e2m1 vs stock_wA +2.24 %; g32_real vs stock_wA +2.35 %; g32_e0m3 vs stock_wA +1.74 %; nodisp vs stock_wA +0.90 %
- **isolated:** g32 vs default, e2m1 -1.33 %; g32 vs default, real -1.46 %; g32 vs default, e0m3 -6.50 %; default_e2m1 vs nodisp +2.18 %; default_real vs nodisp +2.54 %; default_e0m3 vs nodisp +7.54 %; g32_e2m1 vs nodisp +0.83 %; g32_real vs nodisp +1.05 %; g32_e0m3 vs nodisp +0.55 %; default_e2m1 vs stock_wA +2.84 %; default_real vs stock_wA +3.20 %; default_e0m3 vs stock_wA +8.23 %; g32_e2m1 vs stock_wA +1.47 %; g32_real vs stock_wA +1.70 %; g32_e0m3 vs stock_wA +1.20 %; nodisp vs stock_wA +0.64 %
- **sustained:** g32 vs default, e2m1 -1.05 %; g32 vs default, real -1.42 %; g32 vs default, e0m3 -4.53 %; default_e2m1 vs nodisp +1.41 %; default_real vs nodisp +2.56 %; default_e0m3 vs nodisp +5.90 %; g32_e2m1 vs nodisp +0.35 %; g32_real vs nodisp +1.11 %; g32_e0m3 vs nodisp +1.11 %; default_e2m1 vs stock_wA +2.11 %; default_real vs stock_wA +3.27 %; default_e0m3 vs stock_wA +6.64 %; g32_e2m1 vs stock_wA +1.04 %; g32_real vs stock_wA +1.81 %; g32_e0m3 vs stock_wA +1.81 %; nodisp vs stock_wA +0.70 %

