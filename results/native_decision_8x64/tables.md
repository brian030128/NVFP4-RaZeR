| map | E0M3 tiles | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---|---|
| **native evaluation** | | | | | |
| FourOverSix | 0 | 6.875681 | 9.825390 | — | — |
| DET-FAKE-8x64 | 3,454 | 6.822587 | 9.752643 | -0.00775 ± 0.00171 (better) | -0.00743 ± 0.00156 (better) |
| DET-NATIVE-8x64 | 3,801 | 6.813365 | 9.764371 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) |
| **fake evaluation** | | | | | |
| FourOverSix | 0 | 6.887180 | 9.829412 | — | — |
| DET-FAKE-8x64 | 3,454 | 6.818290 | 9.753089 | -0.01005 ± 0.00191 (better) | -0.00780 ± 0.00188 (better) |
| DET-NATIVE-8x64 | 3,801 | 6.817782 | 9.760603 | -0.01013 ± 0.00176 (better) | -0.00703 ± 0.00160 (better) |

| DET-NATIVE-8x64 minus DET-FAKE-8x64 | ΔWiki | ΔC4 |
|---|---|---|
| native evaluation | -0.00135 ± 0.00143 (inconclusive) | +0.00120 ± 0.00147 (inconclusive) |
| fake evaluation | -0.00007 ± 0.00148 (inconclusive) | +0.00077 ± 0.00176 (inconclusive) |

| context (native evaluation) | ΔWiki | ΔC4 |
|---|---|---|
| DET-FAKE-8x64 minus DET-FAKE (256x64) | -0.00134 ± 0.00164 (inconclusive) | -0.00279 ± 0.00108 (better) |
| DET-FAKE-8x64 minus DET-NATIVE (256x64) | -0.00209 ± 0.00157 (better) | -0.00070 ± 0.00111 (inconclusive) |
| DET-NATIVE-8x64 minus DET-FAKE (256x64) | -0.00269 ± 0.00171 (better) | -0.00159 ± 0.00174 (inconclusive) |
| DET-NATIVE-8x64 minus DET-NATIVE (256x64) | -0.00344 ± 0.00172 (better) | +0.00051 ± 0.00157 (inconclusive) |
