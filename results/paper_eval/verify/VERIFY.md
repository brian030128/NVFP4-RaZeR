# C: verification (Section 4.3)

## C1: ownership check of the 18 release artifacts

Every weight element decoded by the kernel itself (an identity activation through the GEMM) must equal its stored value under the map's format (flipquant `--ownership-check`; the install raises on any mismatch).

| model | unit | modules | weight elements checked | informative elements | E0M3 tiles | E0M3 elements observed | exact | format mismatches |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Qwen3-1.7B | 8x64 | 196 of 196 | 1,409,286,144 | 1,287,374,000 | 24,675 | 11,037,956 | True | 0 |
| Qwen3-1.7B | 16x64 | 196 of 196 | 1,409,286,144 | 1,287,216,149 | 16,351 | 14,627,455 | True | 0 |
| Qwen3-1.7B | 256x64 | 196 of 196 | 1,409,286,144 | 1,285,757,889 | 53,440 | 47,825,492 | True | 0 |
| Qwen3-8B | 8x64 | 252 of 252 | 6,945,767,424 | 6,302,722,075 | 67,105 | 29,693,711 | True | 0 |
| Qwen3-8B | 16x64 | 252 of 252 | 6,945,767,424 | 6,302,600,195 | 37,511 | 33,372,609 | True | 0 |
| Qwen3-8B | 256x64 | 252 of 252 | 6,945,767,424 | 6,300,644,482 | 86,176 | 76,672,261 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 8x64 | 224 of 224 | 6,979,321,856 | 6,383,511,757 | 116,977 | 52,333,106 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 16x64 | 224 of 224 | 6,979,321,856 | 6,382,853,652 | 75,906 | 67,942,508 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 256x64 | 224 of 224 | 6,979,321,856 | 6,378,794,042 | 183,856 | 164,259,903 | True | 0 |
| Phi-4 | 8x64 | 160 of 160 | 13,631,488,000 | 12,519,536,650 | 129,730 | 58,635,010 | True | 0 |
| Phi-4 | 16x64 | 160 of 160 | 13,631,488,000 | 12,518,999,412 | 79,591 | 71,945,587 | True | 0 |
| Phi-4 | 256x64 | 160 of 160 | 13,631,488,000 | 12,515,456,916 | 176,944 | 159,943,086 | True | 0 |

## C2: native vs simulated (fake), per-window NLL, paper convention

Paired ΔNLL native − simulated (nats/token, ± 2 SE) on the same windows; a cell agrees when |Δ| ≤ 2 SE. Both sides in one env: n16k64 (the release PPL records) for the four non-hybrid models, n16k64-fast (part F's records) for Nemotron-Nano-9B-v2.

| model | unit | WikiText-2 | C4 |
|---|---|---:|---:|
| Qwen3-1.7B | 8x64 | +0.00033 ± 0.00255 | +0.00126 ± 0.00146 |
| Qwen3-1.7B | 16x64 | +0.00040 ± 0.00228 | +0.00131 ± 0.00134 |
| Qwen3-1.7B | 256x64 | -0.00050 ± 0.00283 | -0.00079 ± 0.00154 |
| Qwen3-8B | 8x64 | +0.00263 ± 0.00173 ✗ | -0.00106 ± 0.00107 |
| Qwen3-8B | 16x64 | -0.00068 ± 0.00192 | +0.00054 ± 0.00102 |
| Qwen3-8B | 256x64 | -0.00082 ± 0.00174 | +0.00055 ± 0.00102 |
| Mistral-7B-Instruct-v0.3 | 8x64 | -0.00024 ± 0.00099 | +0.00012 ± 0.00066 |
| Mistral-7B-Instruct-v0.3 | 16x64 | -0.00016 ± 0.00094 | +0.00066 ± 0.00124 |
| Mistral-7B-Instruct-v0.3 | 256x64 | +0.00026 ± 0.00095 | -0.00001 ± 0.00071 |
| Phi-4 | 8x64 | +0.00008 ± 0.00127 | +0.00028 ± 0.00073 |
| Phi-4 | 16x64 | -0.00050 ± 0.00129 | +0.00025 ± 0.00072 |
| Phi-4 | 256x64 | -0.00045 ± 0.00129 | -0.00057 ± 0.00071 |

**23 of 24 cells agree within 2 SE.**
