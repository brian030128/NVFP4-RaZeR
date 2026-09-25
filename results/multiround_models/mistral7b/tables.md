| unit | E0M3 tiles | rounds / dev evaluations | batch (eval / score) | setup | optimization | scoring per pass | dev evaluation per try | native build per evaluation | peak GPU allocated / reserved | peak host RSS | stop |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8x64 | 120,931 | 14 / 191 | 1 / 1 | 2.1 min | 78.6 min | 99.9 s | 17.5 s | 0.15 s | 16.9 / 18.4 GiB | 14.7 GiB | no step lowers the development objective |
| 256x64 | 22,264 | 13 / 126 | 1 / 1 | 2.0 min | 57.1 min | 98.9 s | 17.1 s | 0.15 s | 16.9 / 18.3 GiB | 14.7 GiB | no step lowers the development objective |

| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| native | FourOverSix | 0 | 5.5225 | 8.0660 | — | — |
| native | NVFP4 | — | 5.5552 | 8.0957 | +0.00591 ± 0.00113 (worse) | +0.00367 ± 0.00092 (worse) |
| native | MixFP4-8x64 | 120,931 | 5.5004 | 8.0334 | -0.00400 ± 0.00097 (better) | -0.00405 ± 0.00083 (better) |
| native | MixFP4-256x64 | 22,264 (256x64) | 5.4956 | 8.0397 | -0.00487 ± 0.00095 (better) | -0.00327 ± 0.00097 (better) |
| fake | FourOverSix | 0 | 5.5260 | 8.0665 | — | — |
| fake | NVFP4 | — | 5.5531 | 8.0958 | +0.00488 ± 0.00118 (worse) | +0.00363 ± 0.00094 (worse) |
| fake | MixFP4-8x64 | 120,931 | 5.4987 | 8.0271 | -0.00495 ± 0.00093 (better) | -0.00490 ± 0.00081 (better) |
| fake | MixFP4-256x64 | 22,264 (256x64) | 5.4996 | 8.0357 | -0.00480 ± 0.00098 (better) | -0.00383 ± 0.00085 (better) |
| fake | BF16 | — | 5.3182 | 7.8306 | -0.03833 ± 0.00128 (better) | -0.02969 ± 0.00177 (better) |
