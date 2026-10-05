## The runs (Llama-3.1-8B 16x64, TM-OPT+TC, deterministic, --no-dev --no-eval)

A: full-vocabulary teacher, 20 epochs; B: full-vocabulary teacher, 5 epochs; C: top-256 teacher, 5 epochs. "nodev_cost A": results/nodev_cost's run of A's configuration (2026-09-28).

| | A | B | C | nodev_cost A |
|---|---:|---:|---:|---:|
| model load (s) | 1.8 | 1.8 | 1.8 | 1.8 |
| data load (s) | 11.9 | 14.1 | 12.2 | 12.2 |
| teacher precompute (s) | 7.9 | 7.3 | 6.4 | 7.4 |
| lean packing (s) | 36.7 | 36.4 | 36.1 | 36.1 |
| **setup** (s) | 58.2 | 59.6 | 56.5 | 57.6 |
| per epoch (s) | 23.01 | 23.02 | 20.14 | 22.85 |
| epochs | 20 | 5 | 5 | 20 |
| **training** (s) | 460.1 | 115.1 | 100.7 | 457.0 |
| write output (s) | 0.0 | 0.0 | 0.0 | 0.0 |
| **total** (s) | 518.3 | 174.7 | 157.2 | 514.6 |
| **peak GPU allocated** (GiB) | 40.48 | 40.48 | 41.07 | 40.48 |
| peak GPU reserved (GiB) | 41.45 | 41.51 | 42.09 | 41.47 |
| peak GPU allocated, training (GiB) | 40.48 | 40.48 | 41.07 | 40.48 |
| peak GPU allocated, teacher precompute (GiB) | 15.60 | 15.60 | 15.94 | 15.60 |
| **peak host RSS**, sampled (GiB) | 18.19 | 18.24 | 15.72 | 18.60 |
| peak host RSS, ru_maxrss (GiB) | 18.13 | 18.18 | 15.88 | 18.54 |
| peak host RSS, model load (GiB) | 15.69 | 15.09 | 15.72 | 15.36 |
| peak host RSS after the model load (GiB) | 18.19 | 18.24 | 2.55 | 18.60 |
| peak host RSS, training phase (GiB) | 18.19 | 18.24 | 2.55 | 18.60 |
| **teacher storage** (MB) | 16,777.9 | 16,777.9 | 100.7 | 16,777.9 |
| teacher device | cpu | cpu | cuda:0 | cpu |
| tail mass, mean / max window | — | — | 0.0184 / 0.0557 | — |
| final E0M3 tiles | 201,648 | 4,055 | 3,699 | 201,648 |
| final epoch training KL | 0.04239 | 0.08597 | 0.08259 | 0.04239 |

## C as ratios to A and to B

| | C / A | C / B | B / A |
|---|---:|---:|---:|
| setup | 0.9705 | 0.9476 | 1.0242 |
| per epoch | 0.8755 | 0.8748 | 1.0008 |
| training | 0.2189 | 0.8748 | 0.2502 |
| total | 0.3033 | 0.8997 | 0.3371 |
| peak GPU allocated | 1.0145 | 1.0145 | 1.0000 |
| peak GPU reserved | 1.0155 | 1.0141 | 1.0014 |
| peak host RSS, sampled | 0.8645 | 0.8620 | 1.0028 |
| peak host RSS, ru_maxrss | 0.8759 | 0.8731 | 1.0032 |
| peak host RSS after the model load | 0.1403 | 0.1399 | 1.0028 |
| peak host RSS, training phase | 0.1399 | 0.1395 | 1.0028 |
| teacher storage | 0.0060 | 0.0060 | 1.0000 |

## Consistency: today's A against the results/nodev_cost run

| | today | nodev_cost | ratio |
|---|---:|---:|---:|
| total (min) | 8.64 | 8.58 | 1.0073 |
| per epoch (s) | 23.01 | 22.85 | 1.0068 |
| setup (s) | 58.17 | 57.56 | 1.0107 |
| peak GPU allocated (GiB) | 40.48 | 40.48 | 1.0000 |
| peak host RSS, ru_maxrss (GiB) | 18.13 | 18.54 | 0.9778 |
| map sha256 | 54070819a774fed6… | 54070819a774fed6… | equal |

