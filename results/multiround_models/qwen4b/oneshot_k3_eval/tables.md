| backend | map | E0M3 8x64 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| native | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| native | N16K64-n8-k3 | 8,149 | 11.7192 | 15.7169 | -0.19299 ± 0.00709 (better) | -0.09629 ± 0.00400 (better) |
| native | N16K64-n16-k3 | 8,798 | 12.1275 | 15.9630 | -0.15875 ± 0.00650 (better) | -0.08075 ± 0.00343 (better) |
| native | DET-NATIVE-8x64 | 10,786 | 13.0453 | 17.0581 | -0.08579 ± 0.00557 (better) | -0.01440 ± 0.00258 (better) |
| native | DET-NATIVE-256x64 | 213,536 | 13.9906 | 17.4348 | -0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| native | DET-FAKE-8x64 | 187,147 | 12.6275 | 16.7179 | -0.11835 ± 0.00616 (better) | -0.03454 ± 0.00311 (better) |
| native | DET-FAKE-256x64 | 215,040 | 14.0251 | 17.4551 | -0.01337 ± 0.00442 (better) | +0.00861 ± 0.00201 (worse) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | N16K64-n8-k3 | 8,149 | 11.6899 | 15.7152 | -0.19763 ± 0.00823 (better) | -0.09734 ± 0.00399 (better) |
| fake | N16K64-n16-k3 | 8,798 | 12.1184 | 15.9781 | -0.16162 ± 0.00744 (better) | -0.08075 ± 0.00331 (better) |
| fake | DET-NATIVE-8x64 | 10,786 | 13.0396 | 17.0706 | -0.08836 ± 0.00647 (better) | -0.01461 ± 0.00255 (better) |
| fake | DET-NATIVE-256x64 | 213,536 | 14.0873 | 17.4644 | -0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| fake | DET-FAKE-8x64 | 187,147 | 12.6095 | 16.7192 | -0.12190 ± 0.00713 (better) | -0.03541 ± 0.00279 (better) |
| fake | DET-FAKE-256x64 | 215,040 | 13.9997 | 17.4526 | -0.01731 ± 0.00500 (better) | +0.00752 ± 0.00188 (worse) |

| backend | one-shot map | minus | ΔWiki | ΔC4 |
|---|---|---|---|---|
| native | N16K64-n8-k3 | DET-NATIVE-8x64 | -0.10720 ± 0.00383 (better) | -0.08188 ± 0.00329 (better) |
| native | N16K64-n8-k3 | DET-NATIVE-256x64 | -0.17715 ± 0.00616 (better) | -0.10373 ± 0.00413 (better) |
| native | N16K64-n8-k3 | DET-FAKE-8x64 | -0.07464 ± 0.00293 (better) | -0.06174 ± 0.00273 (better) |
| native | N16K64-n8-k3 | DET-FAKE-256x64 | -0.17962 ± 0.00633 (better) | -0.10490 ± 0.00405 (better) |
| native | N16K64-n16-k3 | DET-NATIVE-8x64 | -0.07296 ± 0.00326 (better) | -0.06635 ± 0.00287 (better) |
| native | N16K64-n16-k3 | DET-NATIVE-256x64 | -0.14291 ± 0.00527 (better) | -0.08820 ± 0.00360 (better) |
| native | N16K64-n16-k3 | DET-FAKE-8x64 | -0.04040 ± 0.00262 (better) | -0.04621 ± 0.00231 (better) |
| native | N16K64-n16-k3 | DET-FAKE-256x64 | -0.14538 ± 0.00557 (better) | -0.08936 ± 0.00350 (better) |
| fake | N16K64-n8-k3 | DET-NATIVE-8x64 | -0.10927 ± 0.00385 (better) | -0.08273 ± 0.00323 (better) |
| fake | N16K64-n8-k3 | DET-NATIVE-256x64 | -0.18655 ± 0.00687 (better) | -0.10554 ± 0.00418 (better) |
| fake | N16K64-n8-k3 | DET-FAKE-8x64 | -0.07573 ± 0.00298 (better) | -0.06193 ± 0.00266 (better) |
| fake | N16K64-n8-k3 | DET-FAKE-256x64 | -0.18031 ± 0.00668 (better) | -0.10486 ± 0.00410 (better) |
| fake | N16K64-n16-k3 | DET-NATIVE-8x64 | -0.07326 ± 0.00315 (better) | -0.06614 ± 0.00277 (better) |
| fake | N16K64-n16-k3 | DET-NATIVE-256x64 | -0.15054 ± 0.00602 (better) | -0.08894 ± 0.00360 (better) |
| fake | N16K64-n16-k3 | DET-FAKE-8x64 | -0.03972 ± 0.00252 (better) | -0.04534 ± 0.00228 (better) |
| fake | N16K64-n16-k3 | DET-FAKE-256x64 | -0.14431 ± 0.00558 (better) | -0.08826 ± 0.00351 (better) |

| map (campaign record, per-token activations: a different protocol) | fake WikiText-2 / C4 | native WikiText-2 / C4 |
|---|---|---|
| N16K64-n8-k3 | 11.7066 / 15.7092 | 11.6925 / 15.6958 |
| N16K64-n16-k3 | 12.1095 / 15.9714 | 12.0888 / 15.9623 (after the weights-on-A fix: 12.0933 / 15.9515) |
| FourOverSix | 14.2183 / 17.3175 | 14.2106 / 17.3062 |
