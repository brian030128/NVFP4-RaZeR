| comparison (round 0) | tiles | legacy candidates | B1 candidates | both | legacy only | B1 only | max normwise rel. error mu / SE | max |bound|/SE of a disagreeing tile |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| llama8b 256x64 | 425,984 | 15,136 | 15,136 | 15,136 | 0 | 0 | 1.1e-06 / 5.5e-07 | 0.00e+00 |
| llama8b 8x64 | 13,631,488 | 379,759 | 379,759 | 379,759 | 0 | 0 | 6.0e-07 / 8.0e-07 | 0.00e+00 |
| phi4 8x64 | 26,624,000 | 700,279 | 700,279 | 700,279 | 0 | 0 | 2.2e-07 / 2.3e-07 | 0.00e+00 |

| unit | run | rounds / dev evaluations | tiles | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | peak host RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256x64 | items 1+3 | 8 / 64 | 8,385 | 18.5 min | 48.4 s | 11.44 s | 48.1 / 54.3 GiB | 43.8 GiB |
| 256x64 | items 1+3 + B1 | 8 / 64 | 8,385 | 16.7 min | 35.6 s | 11.40 s | 48.1 / 55.8 GiB | 44.0 GiB |
| 8x64 | items 1+3 | 9 / 115 | 3,801 | 29.0 min | 48.7 s | 11.43 s | 49.2 / 56.1 GiB | 44.1 GiB |
| 8x64 | items 1+3 + B1 | 9 / 115 | 3,801 | 26.8 min | 34.1 s | 11.40 s | 49.2 / 55.4 GiB | 44.2 GiB |

| unit | committed tiles | B1 tiles | both | only committed | only B1 |
|---|---:|---:|---:|---:|---:|
| 256x64 | 8,385 | 8,385 | 8,385 | 0 | 0 |
| 8x64 | 3,801 | 3,801 | 3,801 | 0 | 0 |

| backend | map | tiles | WikiText-2 | C4 |
|---|---|---:|---:|---:|
| native | FourOverSix | 0 | 6.875681 | 9.825390 |
| native | COMMITTED-256x64 | 268320 | 6.836861 | 9.759441 |
| native | COMMITTED-8x64 | 3801 | 6.813365 | 9.764371 |
| native | B1-256x64 | 268320 | 6.836861 | 9.759441 |
| native | B1-8x64 | 3801 | 6.813365 | 9.764371 |
| fake | FourOverSix | 0 | 6.887180 | 9.829412 |
| fake | COMMITTED-256x64 | 268320 | 6.831542 | 9.773669 |
| fake | COMMITTED-8x64 | 3801 | 6.817782 | 9.760603 |
| fake | B1-256x64 | 268320 | 6.831542 | 9.773669 |
| fake | B1-8x64 | 3801 | 6.817782 | 9.760603 |

| backend | B1 minus committed | ΔWiki | ΔC4 |
|---|---|---|---|
| native (criterion) | 256x64 | +0.00000 ± 0.00000 (inconclusive) | +0.00000 ± 0.00000 (inconclusive) |
| native (criterion) | 8x64 | +0.00000 ± 0.00000 (inconclusive) | +0.00000 ± 0.00000 (inconclusive) |
| fake | 256x64 | +0.00000 ± 0.00000 (inconclusive) | +0.00000 ± 0.00000 (inconclusive) |
| fake | 8x64 | +0.00000 ± 0.00000 (inconclusive) | +0.00000 ± 0.00000 (inconclusive) |
