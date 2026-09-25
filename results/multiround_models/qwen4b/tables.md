| unit | E0M3 tiles | rounds / dev evaluations | batch (eval / score) | setup | optimization | scoring per pass | dev evaluation per try | native build per evaluation | peak GPU allocated / reserved | peak host RSS | stop |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8x64 | 10,786 | 11 / 130 | 1 / 1 | 2.2 min | 73.4 min | 72.1 s | 28.0 s | 0.12 s | 10.2 / 14.0 GiB | 49.4 GiB | no step lowers the development objective |
| 256x64 | 6,673 | 5 / 39 | 1 / 1 | 2.3 min | 23.6 min | 71.7 s | 27.7 s | 0.12 s | 9.6 / 13.1 GiB | 49.4 GiB | no step lowers the development objective |

| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| native | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| native | NVFP4 | — | 13.9418 | 17.2685 | -0.01933 ± 0.00490 (better) | -0.00214 ± 0.00221 |
| native | MixFP4-8x64 | 10,786 | 13.0453 | 17.0581 | -0.08579 ± 0.00557 (better) | -0.01440 ± 0.00258 (better) |
| native | MixFP4-256x64 | 6,673 (256x64) | 13.9906 | 17.4348 | -0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | NVFP4 | — | 13.9488 | 17.2785 | -0.02096 ± 0.00536 (better) | -0.00251 ± 0.00218 (better) |
| fake | MixFP4-8x64 | 10,786 | 13.0396 | 17.0706 | -0.08836 ± 0.00647 (better) | -0.01461 ± 0.00255 (better) |
| fake | MixFP4-256x64 | 6,673 (256x64) | 14.0873 | 17.4644 | -0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| fake | BF16 | — | 13.6588 | 16.6409 | -0.04197 ± 0.00543 (better) | -0.04010 ± 0.00281 (better) |
