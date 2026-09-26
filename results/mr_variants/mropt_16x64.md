### Llama-3.1-8B

| unit | E0M3 tiles (own unit) | E0M3 share of weights | rounds / dev evaluations | optimization | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---|---|
| FourOverSix | 0 | 0 % | — | — | 6.8757 | 9.8254 | — | — |
| 8x64 | 3,801 | 0.03 % | 9 / 115 | 28.0 min | 6.8134 | 9.7644 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) |
| 16x64 | 4,995 | 0.07 % | 15 / 184 | 45.4 min | 6.8259 | 9.7412 | -0.00727 ± 0.00163 (better) | -0.00861 ± 0.00219 (better) |
| 256x64 | 8,385 | 1.97 % | 8 / 64 | 17.5 min | 6.8369 | 9.7594 | -0.00566 ± 0.00171 (better) | -0.00673 ± 0.00169 (better) |

| MR-OPT 16x64 minus | ΔWiki | ΔC4 |
|---|---|---|
| MR-OPT 8x64 | +0.00184 ± 0.00166 (worse) | -0.00238 ± 0.00248 |
| MR-OPT 256x64 | -0.00161 ± 0.00157 (better) | -0.00187 ± 0.00210 |

Fake: FourOverSix 6.8872 / 9.8294; MR-OPT 16x64 6.8166 / 9.7446, ΔWiki -0.01030 ± 0.00189 (better), ΔC4 -0.00866 ± 0.00271 (better) vs FourOverSix.

Checks: {"same_windows": true, "native_repeats_earlier_evaluation": {"FourOverSix": true, "mropt-8x64": true, "mropt-256x64": true}, "fake_fourover6_repeats_earlier": true, "map_mismatches": [], "converted": "16x64"}

MR-OPT 16x64 calibration: setup 1.7 min, scoring 34.6 s per pass, dev evaluation 12.1 s per try, dev KL 0.10811 → 0.09264, peak GPU 41.2 / 42.4 GiB, host 42.2 GiB, stop: no step lowers the development objective.

### Mistral-7B-v0.3

| unit | E0M3 tiles (own unit) | E0M3 share of weights | rounds / dev evaluations | optimization | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---|---|
| FourOverSix | 0 | 0 % | — | — | 5.5225 | 8.0660 | — | — |
| 8x64 | 123,180 | 0.90 % | 20 / 237 | 40.5 min | 5.4837 | 8.0334 | -0.00704 ± 0.00097 (better) | -0.00405 ± 0.00075 (better) |
| 16x64 | 127,060 | 1.86 % | 10 / 105 | 18.5 min | 5.4992 | 8.0338 | -0.00423 ± 0.00098 (better) | -0.00400 ± 0.00081 (better) |
| 256x64 | 22,082 | 5.18 % | 14 / 132 | 22.4 min | 5.4950 | 8.0354 | -0.00498 ± 0.00098 (better) | -0.00380 ± 0.00074 (better) |

| MR-OPT 16x64 minus | ΔWiki | ΔC4 |
|---|---|---|
| MR-OPT 8x64 | +0.00281 ± 0.00090 (worse) | +0.00005 ± 0.00071 |
| MR-OPT 256x64 | +0.00075 ± 0.00087 | -0.00020 ± 0.00076 |

Fake: FourOverSix 5.5260 / 8.0665; MR-OPT 16x64 5.5002 / 8.0353, ΔWiki -0.00468 ± 0.00098 (better), ΔC4 -0.00388 ± 0.00152 (better) vs FourOverSix.

Checks: {"same_windows": true, "native_repeats_earlier_evaluation": {"FourOverSix": true, "mropt-8x64": true, "mropt-256x64": true}, "fake_fourover6_repeats_earlier": true, "map_mismatches": [], "converted": "16x64"}

MR-OPT 16x64 calibration: setup 1.5 min, scoring 31.9 s per pass, dev evaluation 7.6 s per try, dev KL 0.04218 → 0.02863, peak GPU 36.4 / 37.5 GiB, host 14.7 GiB, stop: no step lowers the development objective.

### Phi-4

| unit | E0M3 tiles (own unit) | E0M3 share of weights | rounds / dev evaluations | optimization | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|---:|---:|---|---|
| FourOverSix | 0 | 0 % | — | — | 6.6649 | 10.5456 | — | — |
| 8x64 | 57,098 | 0.21 % | 10 / 108 | 36.0 min | 6.6373 | 10.5216 | -0.00415 ± 0.00131 (better) | -0.00228 ± 0.00080 (better) |
| 16x64 | 117,122 | 0.88 % | 13 / 126 | 43.5 min | 6.6288 | 10.5113 | -0.00542 ± 0.00137 (better) | -0.00326 ± 0.00087 (better) |
| 256x64 | 59,953 | 7.21 % | 14 / 85 | 35.3 min | 6.6519 | 10.5237 | -0.00195 ± 0.00147 (better) | -0.00208 ± 0.00083 (better) |

| MR-OPT 16x64 minus | ΔWiki | ΔC4 |
|---|---|---|
| MR-OPT 8x64 | -0.00127 ± 0.00129 | -0.00098 ± 0.00076 (better) |
| MR-OPT 256x64 | -0.00347 ± 0.00137 (better) | -0.00118 ± 0.00085 (better) |

Fake: FourOverSix 6.6667 / 10.5437; MR-OPT 16x64 6.6317 / 10.5158, ΔWiki -0.00526 ± 0.00134 (better), ΔC4 -0.00264 ± 0.00089 (better) vs FourOverSix.

Checks: {"same_windows": true, "native_repeats_earlier_evaluation": {"FourOverSix": true, "mropt-8x64": true, "mropt-256x64": true}, "fake_fourover6_repeats_earlier": true, "map_mismatches": [], "converted": "16x64"}

MR-OPT 16x64 calibration: setup 2.6 min, scoring 66.0 s per pass, dev evaluation 14.0 s per try, dev KL 0.03765 → 0.03473, peak GPU 60.1 / 63.0 GiB, host 33.7 GiB, stop: no step lowers the development objective.

