| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS | stop |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 256x64 | MR-OPT | 14 / 132 | 22,082 | 0.04218 → 0.02819 | 1.4 min | **22.4 min** | 32.9 s | 6.7 s | 35.8 / 36.8 GiB | 14.7 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG | 3 / 21 | 21,696 | 0.04218 → 0.03133 | 1.4 min | **4.2 min** | 33.0 s | 7.6 s | 35.8 / 36.8 GiB | 14.7 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+WS | 16 / 45 | 21,681 | 0.04218 → 0.02906 | 1.4 min | **13.7 min** | 32.8 s | 6.7 s | 35.8 / 36.8 GiB | 14.7 GiB | no step lowers the development objective |
| 256x64 | MR-OPT+SIG+WS | 3 / 19 | 21,696 | 0.04218 → 0.03133 | 1.4 min | **3.9 min** | 33.1 s | 7.6 s | 35.8 / 36.7 GiB | 14.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT | 20 / 237 | 123,180 | 0.04218 → 0.02764 | 1.5 min | **40.5 min** | 31.6 s | 7.6 s | 37.0 / 38.2 GiB | 14.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG | 9 / 100 | 121,965 | 0.04218 → 0.02928 | 1.5 min | **17.1 min** | 31.6 s | 7.5 s | 37.0 / 38.2 GiB | 14.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+WS | 9 / 36 | 120,750 | 0.04218 → 0.02841 | 1.4 min | **9.1 min** | 31.5 s | 7.6 s | 37.0 / 38.3 GiB | 14.7 GiB | no step lowers the development objective |
| 8x64 | MR-OPT+SIG+WS | 8 / 34 | 121,759 | 0.04218 → 0.02906 | 1.4 min | **8.3 min** | 31.5 s | 7.4 s | 37.0 / 38.3 GiB | 14.7 GiB | no step lowers the development objective |

| unit | configuration | tiles shared with MR-OPT | only MR-OPT | only variant | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | — | — | 5.4950 | 8.0354 | -0.00498 ± 0.00098 (better) | -0.00380 ± 0.00074 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 21,228 | 854 | 468 | 5.5085 | 8.0443 | -0.00252 ± 0.00094 (better) | -0.00269 ± 0.00079 (better) | +0.00245 ± 0.00085 (worse) | +0.00110 ± 0.00070 (worse) | **NO** / no |
| 256x64 | MR-OPT+WS | 21,259 | 823 | 422 | 5.4969 | 8.0359 | -0.00464 ± 0.00102 (better) | -0.00374 ± 0.00080 (better) | +0.00034 ± 0.00087 | +0.00006 ± 0.00072 | **yes** / no |
| 256x64 | MR-OPT+SIG+WS | 21,228 | 854 | 468 | 5.5085 | 8.0443 | -0.00252 ± 0.00094 (better) | -0.00269 ± 0.00079 (better) | +0.00245 ± 0.00085 (worse) | +0.00110 ± 0.00070 (worse) | **NO** / no |
| 8x64 | MR-OPT | — | — | — | 5.4837 | 8.0334 | -0.00704 ± 0.00097 (better) | -0.00405 ± 0.00075 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 116,129 | 7,051 | 5,836 | 5.4960 | 8.0374 | -0.00481 ± 0.00100 (better) | -0.00356 ± 0.00077 (better) | +0.00223 ± 0.00086 (worse) | +0.00049 ± 0.00069 | **NO** / no |
| 8x64 | MR-OPT+WS | 118,339 | 4,841 | 2,411 | 5.4936 | 8.0371 | -0.00525 ± 0.00096 (better) | -0.00359 ± 0.00089 (better) | +0.00179 ± 0.00087 (worse) | +0.00046 ± 0.00071 | **NO** / no |
| 8x64 | MR-OPT+SIG+WS | 116,279 | 6,901 | 5,480 | 5.4934 | 8.0324 | -0.00527 ± 0.00091 (better) | -0.00418 ± 0.00079 (better) | +0.00176 ± 0.00083 (worse) | -0.00013 ± 0.00075 | **NO** / no |

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 5.5225 | 8.0660 | — | — |
| native | NVFP4 | 5.5552 | 8.0957 | +0.00591 ± 0.00113 (worse) | +0.00367 ± 0.00092 (worse) |
| native | mropt-256x64 | 5.4950 | 8.0354 | -0.00498 ± 0.00098 (better) | -0.00380 ± 0.00074 (better) |
| native | mropt-8x64 | 5.4837 | 8.0334 | -0.00704 ± 0.00097 (better) | -0.00405 ± 0.00075 (better) |
| fake | FourOverSix | 5.5260 | 8.0665 | — | — |
| fake | NVFP4 | 5.5531 | 8.0958 | +0.00488 ± 0.00118 (worse) | +0.00363 ± 0.00094 (worse) |
| fake | mropt-256x64 | 5.4937 | 8.0365 | -0.00586 ± 0.00095 (better) | -0.00372 ± 0.00125 (better) |
| fake | mropt-8x64 | 5.4865 | 8.0309 | -0.00717 ± 0.00100 (better) | -0.00442 ± 0.00116 (better) |
| fake | BF16 | 5.3182 | 7.8306 | | |
