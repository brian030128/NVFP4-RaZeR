### 256x64

| run | tiles | rounds / dev evaluations | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | peak host RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DET-FAKE-256x64 | 6,720 | 7 / 53 | 1.7 min | 52.3 min | 72.8 s | 50.6 s | 9.6 / 13.1 GiB | 49.4 GiB |
| DET-NATIVE-256x64 | 6,673 | 5 / 39 | 2.3 min | 23.6 min | 71.7 s | 27.7 s | 9.6 / 13.1 GiB | 49.4 GiB |

| backend | map | tiles | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| native | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| native | DET-NATIVE-256x64 | 6,673 | 13.9906 | 17.4348 | -0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| native | DET-FAKE-256x64 | 6,720 | 14.0251 | 17.4551 | -0.01337 ± 0.00442 (better) | +0.00861 ± 0.00201 (worse) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | DET-NATIVE-256x64 | 6,673 | 14.0873 | 17.4644 | -0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| fake | DET-FAKE-256x64 | 6,720 | 13.9997 | 17.4526 | -0.01731 ± 0.00500 (better) | +0.00752 ± 0.00188 (worse) |

| DET-NATIVE minus DET-FAKE | ΔWiki | ΔC4 |
|---|---|---|
| native evaluation (criterion) | -0.00247 ± 0.00328 (inconclusive) | -0.00116 ± 0.00156 (inconclusive) |
| fake evaluation | +0.00623 ± 0.00395 (worse) | +0.00068 ± 0.00147 (inconclusive) |

First divergence: {"round": 1, "try_index": 5, "size": 248, "what": "decision", "fake": {"accepts": true, "dev_delta_kl": -0.0005454571801237762}, "native": {"accepts": false, "dev_delta_kl": 0.001342802686849609}}

### 8x64

| run | tiles | rounds / dev evaluations | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | peak host RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DET-FAKE-8x64 | 187,147 | 19 / 225 | 1.8 min | 209.8 min | 72.5 s | 50.0 s | 10.2 / 13.7 GiB | 49.4 GiB |
| DET-NATIVE-8x64 | 10,786 | 11 / 130 | 2.2 min | 73.4 min | 72.1 s | 28.0 s | 10.2 / 14.0 GiB | 49.4 GiB |

| backend | map | tiles | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| native | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| native | DET-NATIVE-8x64 | 10,786 | 13.0453 | 17.0581 | -0.08579 ± 0.00557 (better) | -0.01440 ± 0.00258 (better) |
| native | DET-FAKE-8x64 | 187,147 | 12.6275 | 16.7179 | -0.11835 ± 0.00616 (better) | -0.03454 ± 0.00311 (better) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | DET-NATIVE-8x64 | 10,786 | 13.0396 | 17.0706 | -0.08836 ± 0.00647 (better) | -0.01461 ± 0.00255 (better) |
| fake | DET-FAKE-8x64 | 187,147 | 12.6095 | 16.7192 | -0.12190 ± 0.00713 (better) | -0.03541 ± 0.00279 (better) |

| DET-NATIVE minus DET-FAKE | ΔWiki | ΔC4 |
|---|---|---|
| native evaluation (criterion) | +0.03255 ± 0.00293 (worse) | +0.02014 ± 0.00183 (worse) |
| fake evaluation | +0.03354 ± 0.00275 (worse) | +0.02080 ± 0.00163 (worse) |

First divergence: {"round": 6, "try_index": 10, "size": 191, "what": "decision", "fake": {"accepts": false, "dev_delta_kl": 0.0006894369726069272}, "native": {"accepts": true, "dev_delta_kl": -0.00019748008344322443}}

