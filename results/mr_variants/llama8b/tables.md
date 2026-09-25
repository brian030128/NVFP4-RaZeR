| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS | stop |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 256x64 | MR-OPT | 8 / 64 | 8,385 | 0.10811 → 0.09399 | 1.7 min | **17.5 min** | 35.7 s | 12.1 s | 40.7 / 41.6 GiB | 42.3 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG | 4 / 37 | 3,846 | 0.10811 → 0.09671 | 1.6 min | **9.6 min** | 35.9 s | 12.0 s | 40.7 / 41.8 GiB | 42.2 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+WS | 6 / 25 | 7,620 | 0.10811 → 0.09489 | 1.6 min | **8.4 min** | 35.9 s | 12.0 s | 40.7 / 41.7 GiB | 42.3 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG+WS | 4 / 21 | 3,873 | 0.10811 → 0.09701 | 1.7 min | **6.4 min** | 35.9 s | 12.0 s | 40.7 / 41.6 GiB | 42.2 GiB | no step lowers the development objective |
| 8x64 | MR-OPT | 9 / 115 | 3,801 | 0.10811 → 0.09441 | 1.6 min | **28.0 min** | 34.3 s | 12.0 s | 41.8 / 43.3 GiB | 42.2 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG | 3 / 45 | 1,475 | 0.10811 → 0.09827 | 1.7 min | **10.5 min** | 34.3 s | 12.0 s | 41.8 / 43.3 GiB | 42.1 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+WS | 9 / 36 | 3,353 | 0.10811 → 0.09370 | 1.7 min | **12.2 min** | 34.4 s | 12.0 s | 41.8 / 43.3 GiB | 42.1 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG+WS | 3 / 24 | 1,476 | 0.10811 → 0.09830 | 1.6 min | **6.3 min** | 34.4 s | 11.9 s | 41.8 / 43.2 GiB | 42.2 GiB | no step lowers the development objective |

| unit | configuration | tiles shared with MR-OPT | only MR-OPT | only variant | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | — | — | 6.8369 | 9.7594 | -0.00566 ± 0.00171 (better) | -0.00673 ± 0.00169 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 3,659 | 4,726 | 187 | 6.8552 | 9.7838 | -0.00298 ± 0.00166 (better) | -0.00425 ± 0.00154 (better) | +0.00268 ± 0.00155 (worse) | +0.00249 ± 0.00123 (worse) | **NO** / no |
| 256x64 | MR-OPT+WS | 7,423 | 962 | 197 | 6.8409 | 9.7781 | -0.00508 ± 0.00163 (better) | -0.00482 ± 0.00159 (better) | +0.00058 ± 0.00151 | +0.00191 ± 0.00128 (worse) | **NO** / no |
| 256x64 | MR-OPT+SIG+WS | 3,697 | 4,688 | 176 | 6.8506 | 9.7700 | -0.00366 ± 0.00148 (better) | -0.00566 ± 0.00167 (better) | +0.00200 ± 0.00158 (worse) | +0.00108 ± 0.00152 | **NO** / no |
| 8x64 | MR-OPT | — | — | — | 6.8134 | 9.7644 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 1,247 | 2,554 | 228 | 6.8469 | 9.7679 | -0.00420 ± 0.00158 (better) | -0.00587 ± 0.00191 (better) | +0.00490 ± 0.00160 (worse) | +0.00036 ± 0.00228 | **NO** / no |
| 8x64 | MR-OPT+WS | 2,843 | 958 | 510 | 6.8359 | 9.7522 | -0.00581 ± 0.00172 (better) | -0.00747 ± 0.00200 (better) | +0.00330 ± 0.00152 (worse) | -0.00124 ± 0.00248 | **NO** / no |
| 8x64 | MR-OPT+SIG+WS | 1,249 | 2,552 | 227 | 6.8453 | 9.7687 | -0.00443 ± 0.00159 (better) | -0.00579 ± 0.00193 (better) | +0.00468 ± 0.00159 (worse) | +0.00044 ± 0.00227 | **NO** / no |

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 6.8757 | 9.8254 | — | — |
| native | NVFP4 | 6.9361 | 9.9279 | +0.00875 ± 0.00209 (worse) | +0.01038 ± 0.00203 (worse) |
| native | mropt-256x64 | 6.8369 | 9.7594 | -0.00566 ± 0.00171 (better) | -0.00673 ± 0.00169 (better) |
| native | mropt-8x64 | 6.8134 | 9.7644 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) |
| fake | FourOverSix | 6.8872 | 9.8294 | — | — |
| fake | NVFP4 | 6.9437 | 9.9302 | +0.00817 ± 0.00208 (worse) | +0.01020 ± 0.00207 (worse) |
| fake | mropt-256x64 | 6.8315 | 9.7737 | -0.00811 ± 0.00172 (better) | -0.00569 ± 0.00148 (better) |
| fake | mropt-8x64 | 6.8178 | 9.7606 | -0.01013 ± 0.00176 (better) | -0.00703 ± 0.00160 (better) |
| fake | BF16 | 6.2403 | 8.9579 | | |
