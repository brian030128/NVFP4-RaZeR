| run | configuration | epochs | training s / epoch | selection time | peak GPU allocated / reserved | training peak GPU allocated | host RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| legacy (group 1 STE, 3 epochs) | custom | 3 | 63.6 | 5.4 min | 59.0 / 61.6 GiB | 59.0 GiB | 43.6 GiB |
| TM-OPT, fake monitor (group 1 STE, 3 epochs) | custom | 3 | 34.9 | 2.7 min | 40.6 / 41.6 GiB | 40.6 GiB | 42.0 GiB |
| TM-OPT (group 2, 20 epochs, native monitor) | TM-OPT | 20 | 35.2 | 13.9 min | 40.8 / 42.0 GiB | 40.8 GiB | 42.2 GiB |
| legacy, full-run estimate | legacy | 20 | 63.6 | 27.4 min | | | |

| map | E0M3 tiles | native WikiText-2 | native C4 | native ΔWiki vs FourOverSix | native ΔC4 vs FourOverSix | fake WikiText-2 | fake C4 | fake ΔWiki vs FourOverSix | fake ΔC4 vs FourOverSix |
|---|---:|---:|---:|---|---|---:|---:|---|---|
| FourOverSix | 0 | 6.8757 | 9.8254 | — | — | 6.8872 | 9.8294 | — | — |
| TM-OPT STE 8x64 | 306,968 | 6.7822 | 9.6754 | -0.01369 ± 0.00192 (better) | -0.01539 ± 0.00290 (better) | 6.7792 | 9.6734 | -0.01581 ± 0.00199 (better) | -0.01599 ± 0.00359 (better) |

H200 reference (main, fake evaluation, context only): STE 8x64 6.7847 / 9.6754, 329,837 E0M3 tiles.
Run's own native final evaluation repeated in the joint process: True.
