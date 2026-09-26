| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS | stop |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 256x64 | MR-OPT | 14 / 85 | 59,953 | 0.03765 → 0.03492 | 2.6 min | **35.3 min** | 67.0 s | 14.0 s | 59.0 / 61.7 GiB | 33.6 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG | 3 / 27 | 13,273 | 0.03765 → 0.03568 | 2.6 min | **9.4 min** | 67.0 s | 14.0 s | 59.0 / 62.0 GiB | 33.6 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+WS | 8 / 30 | 28,561 | 0.03765 → 0.03537 | 2.5 min | **15.7 min** | 67.1 s | 14.0 s | 59.0 / 61.9 GiB | 33.6 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG+WS | 2 / 18 | 13,237 | 0.03765 → 0.03633 | 2.6 min | **6.2 min** | 67.1 s | 14.0 s | 59.0 / 61.8 GiB | 33.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT | 10 / 108 | 57,098 | 0.03765 → 0.03481 | 2.6 min | **36.0 min** | 65.3 s | 14.1 s | 61.3 / 64.2 GiB | 33.6 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG | 2 / 29 | 5,470 | 0.03765 → 0.03675 | 2.6 min | **8.7 min** | 65.4 s | 14.0 s | 61.3 / 64.1 GiB | 33.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+WS | 8 / 35 | 12,478 | 0.03765 → 0.03517 | 2.6 min | **16.7 min** | 65.5 s | 14.0 s | 61.3 / 64.3 GiB | 33.6 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG+WS | 3 / 25 | 5,653 | 0.03765 → 0.03577 | 2.6 min | **8.9 min** | 65.3 s | 14.1 s | 61.3 / 64.2 GiB | 33.7 GiB | no step lowers the development objective |

| unit | configuration | tiles shared with MR-OPT | only MR-OPT | only variant | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | — | — | 6.6519 | 10.5237 | -0.00195 ± 0.00147 (better) | -0.00208 ± 0.00083 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 12,539 | 47,414 | 734 | 6.6420 | 10.5218 | -0.00344 ± 0.00131 (better) | -0.00226 ± 0.00080 (better) | -0.00149 ± 0.00130 (better) | -0.00018 ± 0.00071 | **yes** / no |
| 256x64 | MR-OPT+WS | 27,059 | 32,894 | 1,502 | 6.6525 | 10.5321 | -0.00185 ± 0.00137 (better) | -0.00128 ± 0.00083 (better) | +0.00010 ± 0.00149 | +0.00079 ± 0.00078 (worse) | **NO** / no |
| 256x64 | MR-OPT+SIG+WS | 12,482 | 47,471 | 755 | 6.6225 | 10.5081 | -0.00638 ± 0.00152 (better) | -0.00357 ± 0.00082 (better) | -0.00443 ± 0.00138 (better) | -0.00149 ± 0.00082 (better) | **yes** / yes |
| 8x64 | MR-OPT | — | — | — | 6.6373 | 10.5216 | -0.00415 ± 0.00131 (better) | -0.00228 ± 0.00080 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 5,058 | 52,040 | 412 | 6.6036 | 10.4893 | -0.00923 ± 0.00161 (better) | -0.00535 ± 0.00089 (better) | -0.00508 ± 0.00146 (better) | -0.00307 ± 0.00085 (better) | **yes** / yes |
| 8x64 | MR-OPT+WS | 11,424 | 45,674 | 1,054 | 6.6314 | 10.5145 | -0.00504 ± 0.00137 (better) | -0.00296 ± 0.00074 (better) | -0.00089 ± 0.00121 | -0.00068 ± 0.00073 | **yes** / no |
| 8x64 | MR-OPT+SIG+WS | 5,299 | 51,799 | 354 | 6.6376 | 10.5239 | -0.00410 ± 0.00112 (better) | -0.00206 ± 0.00077 (better) | +0.00005 ± 0.00123 | +0.00023 ± 0.00075 | **yes** / no |

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 6.6649 | 10.5456 | — | — |
| native | NVFP4 | 6.7046 | 10.5866 | +0.00594 ± 0.00182 (worse) | +0.00388 ± 0.00104 (worse) |
| native | mropt-256x64 | 6.6519 | 10.5237 | -0.00195 ± 0.00147 (better) | -0.00208 ± 0.00083 (better) |
| native | mropt-8x64 | 6.6373 | 10.5216 | -0.00415 ± 0.00131 (better) | -0.00228 ± 0.00080 (better) |
| fake | FourOverSix | 6.6667 | 10.5437 | — | — |
| fake | NVFP4 | 6.7029 | 10.5882 | +0.00543 ± 0.00181 (worse) | +0.00421 ± 0.00106 (worse) |
| fake | mropt-256x64 | 6.6458 | 10.5293 | -0.00314 ± 0.00129 (better) | -0.00136 ± 0.00083 (better) |
| fake | mropt-8x64 | 6.6369 | 10.5149 | -0.00448 ± 0.00129 (better) | -0.00273 ± 0.00078 (better) |
| fake | BF16 | 6.4615 | 10.3098 | | |
