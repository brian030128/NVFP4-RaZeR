### Llama-3.1-8B

| unit | method | E0M3 tiles | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | final dev KL | selection | setup | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 8x64 | MR-OPT | 3,801 | 6.8134 | 9.7644 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) | 0.09441 | 28.0 min | 1.6 min | 41.8 / 43.3 GiB | 42.2 GiB |
| 8x64 | TM-OPT | 306,968 | 6.7822 | 9.6754 | -0.01369 ± 0.00192 (better) | -0.01539 ± 0.00290 (better) | 0.08288 | 13.9 min | 2.0 min | 40.8 / 42.0 GiB | 42.2 GiB |
| 16x64 | MR-OPT | 4,995 | 6.8259 | 9.7412 | -0.00727 ± 0.00163 (better) | -0.00861 ± 0.00219 (better) | 0.09264 | 45.4 min | 1.7 min | 41.2 / 42.4 GiB | 42.2 GiB |
| 16x64 | TM-OPT | 200,450 | 6.7907 | 9.6813 | -0.01243 ± 0.00194 (better) | -0.01477 ± 0.00282 (better) | 0.08450 | 14.0 min | 1.4 min | 40.6 / 41.8 GiB | 42.3 GiB |
| 256x64 | MR-OPT | 8,385 | 6.8369 | 9.7594 | -0.00566 ± 0.00171 (better) | -0.00673 ± 0.00169 (better) | 0.09399 | 17.5 min | 1.7 min | 40.7 / 41.6 GiB | 42.3 GiB |
| 256x64 | TM-OPT | 35,346 | 6.7957 | 9.7254 | -0.01170 ± 0.00184 (better) | -0.01023 ± 0.00235 (better) | 0.08828 | 14.0 min | 1.4 min | 40.5 / 41.7 GiB | 42.2 GiB |

| unit | TM-OPT − MR-OPT native ΔWiki | native ΔC4 | fake ΔWiki | fake ΔC4 | shared / only TM-OPT / only MR-OPT | Jaccard |
|---|---|---|---|---|---|---:|
| 8x64 | -0.00458 ± 0.00158 (better) | -0.00916 ± 0.00323 (better) | -0.00568 ± 0.00156 (better) | -0.00897 ± 0.00325 (better) | 1,178 / 305,790 / 2,623 | 0.004 |
| 16x64 | -0.00516 ± 0.00168 (better) | -0.00616 ± 0.00189 (better) | -0.00544 ± 0.00154 (better) | -0.00692 ± 0.00185 (better) | 1,997 / 198,453 / 2,998 | 0.010 |
| 256x64 | -0.00604 ± 0.00164 (better) | -0.00349 ± 0.00166 (better) | -0.00439 ± 0.00163 (better) | -0.00588 ± 0.00247 (better) | 4,975 / 30,371 / 3,410 | 0.128 |

Checks: {"fourover6_native_repeats": true, "mropt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

### Mistral-7B-v0.3

| unit | method | E0M3 tiles | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | final dev KL | selection | setup | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 8x64 | MR-OPT | 123,180 | 5.4837 | 8.0334 | -0.00704 ± 0.00097 (better) | -0.00405 ± 0.00075 (better) | 0.02764 | 40.5 min | 1.5 min | 37.0 / 38.2 GiB | 14.7 GiB |
| 8x64 | TM-OPT | 340,045 | 5.4860 | 8.0274 | -0.00662 ± 0.00101 (better) | -0.00480 ± 0.00081 (better) | 0.02680 | 12.1 min | 1.2 min | 36.2 / 37.0 GiB | 14.7 GiB |
| 16x64 | MR-OPT | 127,060 | 5.4992 | 8.0338 | -0.00423 ± 0.00098 (better) | -0.00400 ± 0.00081 (better) | 0.02863 | 18.5 min | 1.5 min | 36.4 / 37.5 GiB | 14.7 GiB |
| 16x64 | TM-OPT | 223,845 | 5.4915 | 8.0285 | -0.00562 ± 0.00094 (better) | -0.00466 ± 0.00091 (better) | 0.02689 | 12.0 min | 1.2 min | 36.0 / 36.9 GiB | 14.7 GiB |
| 256x64 | MR-OPT | 22,082 | 5.4950 | 8.0354 | -0.00498 ± 0.00098 (better) | -0.00380 ± 0.00074 (better) | 0.02819 | 22.4 min | 1.4 min | 35.8 / 36.8 GiB | 14.7 GiB |
| 256x64 | TM-OPT | 42,615 | 5.4957 | 8.0317 | -0.00486 ± 0.00104 (better) | -0.00426 ± 0.00080 (better) | 0.02810 | 12.0 min | 1.2 min | 36.0 / 36.7 GiB | 14.7 GiB |

| unit | TM-OPT − MR-OPT native ΔWiki | native ΔC4 | fake ΔWiki | fake ΔC4 | shared / only TM-OPT / only MR-OPT | Jaccard |
|---|---|---|---|---|---|---:|
| 8x64 | +0.00041 ± 0.00092 (n.s.) | -0.00075 ± 0.00071 (better) | -0.00034 ± 0.00091 (n.s.) | -0.00024 ± 0.00076 (n.s.) | 41,356 / 298,689 / 81,824 | 0.098 |
| 16x64 | -0.00140 ± 0.00089 (better) | -0.00066 ± 0.00085 (n.s.) | -0.00217 ± 0.00092 (better) | -0.00113 ± 0.00087 (better) | 45,621 / 178,224 / 81,439 | 0.149 |
| 256x64 | +0.00012 ± 0.00089 (n.s.) | -0.00046 ± 0.00075 (n.s.) | +0.00001 ± 0.00098 (n.s.) | -0.00044 ± 0.00069 (n.s.) | 12,301 / 30,314 / 9,781 | 0.235 |

Checks: {"fourover6_native_repeats": true, "mropt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

### Phi-4

| unit | method | E0M3 tiles | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | final dev KL | selection | setup | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 8x64 | MR-OPT | 57,098 | 6.6373 | 10.5216 | -0.00415 ± 0.00131 (better) | -0.00228 ± 0.00080 (better) | 0.03481 | 36.0 min | 2.6 min | 61.3 / 64.2 GiB | 33.6 GiB |
| 8x64 | TM-OPT | 423,066 | 6.6089 | 10.4999 | -0.00843 ± 0.00142 (better) | -0.00434 ± 0.00080 (better) | 0.03469 | 23.6 min | 2.1 min | 59.7 / 63.4 GiB | 33.7 GiB |
| 16x64 | MR-OPT | 117,122 | 6.6288 | 10.5113 | -0.00542 ± 0.00137 (better) | -0.00326 ± 0.00087 (better) | 0.03473 | 43.5 min | 2.6 min | 60.1 / 63.0 GiB | 33.7 GiB |
| 16x64 | TM-OPT | 281,702 | 6.6108 | 10.5024 | -0.00815 ± 0.00144 (better) | -0.00410 ± 0.00087 (better) | 0.03520 | 23.6 min | 2.1 min | 59.4 / 63.4 GiB | 33.7 GiB |
| 256x64 | MR-OPT | 59,953 | 6.6519 | 10.5237 | -0.00195 ± 0.00147 (better) | -0.00208 ± 0.00083 (better) | 0.03492 | 35.3 min | 2.6 min | 59.0 / 61.7 GiB | 33.6 GiB |
| 256x64 | TM-OPT | 52,772 | 6.6282 | 10.5131 | -0.00551 ± 0.00142 (better) | -0.00309 ± 0.00084 (better) | 0.03499 | 23.6 min | 2.1 min | 59.2 / 63.2 GiB | 33.7 GiB |

| unit | TM-OPT − MR-OPT native ΔWiki | native ΔC4 | fake ΔWiki | fake ΔC4 | shared / only TM-OPT / only MR-OPT | Jaccard |
|---|---|---|---|---|---|---:|
| 8x64 | -0.00428 ± 0.00126 (better) | -0.00206 ± 0.00072 (better) | -0.00371 ± 0.00125 (better) | -0.00120 ± 0.00076 (better) | 12,556 / 410,510 / 44,542 | 0.027 |
| 16x64 | -0.00273 ± 0.00131 (better) | -0.00084 ± 0.00076 (better) | -0.00316 ± 0.00159 (better) | -0.00110 ± 0.00077 (better) | 23,637 / 258,065 / 93,485 | 0.063 |
| 256x64 | -0.00356 ± 0.00131 (better) | -0.00101 ± 0.00078 (better) | -0.00288 ± 0.00131 (better) | -0.00175 ± 0.00078 (better) | 18,967 / 33,805 / 40,986 | 0.202 |

Checks: {"fourover6_native_repeats": true, "mropt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

