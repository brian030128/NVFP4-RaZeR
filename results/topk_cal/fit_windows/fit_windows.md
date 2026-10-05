## Llama-3.1-8B 16x64, TM-OPT+TC (deterministic, --no-dev --no-eval)

| | A | C | E128 | E256 | E512 |
|---|---:|---:|---:|---:|---:|
| fit windows | 128 | 128 | 128 | 256 | 512 |
| teacher top-K (0 = full vocabulary) | 0 | 256 | 1,000 | 1,000 | 1,000 |
| epochs | 20 | 5 | 5 | 5 | 5 |
| optimizer steps | 320 | 80 | 80 | 160 | 320 |
| model load (s) | 1.8 | 1.8 | 1.8 | 1.8 | 1.8 |
| data load (s) | 11.9 | 12.2 | 12.2 | 20.8 | 22.6 |
| teacher precompute (s) | 7.9 | 6.4 | 6.2 | 12.0 | 23.2 |
| lean packing (s) | 36.7 | 36.1 | 37.1 | 37.0 | 36.4 |
| **setup** (s) | 58.2 | 56.5 | 57.3 | 71.7 | 83.8 |
| **per epoch** (s) | 23.01 | 20.14 | 20.11 | 40.35 | 80.76 |
| **training** (s) | 460.1 | 100.7 | 100.5 | 201.8 | 403.8 |
| **total** (s) | 518.3 | 157.2 | 157.9 | 273.5 | 487.7 |
| **peak GPU allocated** (GiB) | 40.48 | 41.07 | 41.35 | 41.71 | 42.44 |
| peak GPU reserved (GiB) | 41.45 | 42.09 | 42.39 | 42.69 | 43.42 |
| **peak host RSS**, ru_maxrss (GiB) | 18.13 | 15.88 | 15.88 | 15.88 | 15.88 |
| peak host RSS after the model load (GiB) | 18.19 | 2.55 | 2.61 | 2.74 | 2.67 |
| **teacher storage** (MB) | 16,777.9 | 100.7 | 392.7 | 785.4 | 1,570.8 |
| teacher device | cpu | cuda:0 | cuda:0 | cuda:0 | cuda:0 |
| tail mass, mean / max window | — | 0.0184 / 0.0557 | 0.0073 / 0.0214 | 0.0073 / 0.0344 | 0.0076 / 0.1030 |
| final E0M3 tiles | 201,648 | 3,699 | 3,736 | 32,125 | 106,404 |
| final-epoch training KL | 0.04239 (full) | 0.08259 (top-K) | 0.08504 (top-K) | 0.07883 (top-K) | 0.06685 (top-K) |

## Ratios to A (full vocabulary, 128 windows, 20 epochs)

| | C | E128 | E256 | E512 |
|---|---:|---:|---:|---:|
| setup | 0.971 | 0.985 | 1.232 | 1.441 |
| per epoch | 0.876 | 0.874 | 1.754 | 3.510 |
| training | 0.219 | 0.219 | 0.439 | 0.878 |
| total | 0.303 | 0.305 | 0.528 | 0.941 |
| optimizer steps | 0.250 | 0.250 | 0.500 | 1.000 |
| peak GPU allocated | 1.014 | 1.021 | 1.030 | 1.048 |
| peak GPU reserved | 1.016 | 1.023 | 1.030 | 1.048 |
| peak host RSS, ru_maxrss | 0.876 | 0.876 | 0.876 | 0.876 |
| host RSS after the model load | 0.140 | 0.143 | 0.151 | 0.147 |
| teacher storage | 0.006 | 0.023 | 0.047 | 0.094 |
