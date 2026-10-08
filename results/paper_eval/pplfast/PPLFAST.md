# F: the hybrid models' main-table PPL in n16k64-fast vs the fallback env

flipquant `evaluation.ppl --paper-convention` (NVFP4: `--act-scope row`, per-token NVFP4 scales, amendment 11), the same windows; fallback = the records of the CPU tables (main-ppl 44f8cea for BF16 / NVFP4 / FourOverSix, the release records for FlipQuant); fast = n16k64-fast (amendments 3 and 5). The maps are as calibrated (under the fallback kernels). ✗ = |ΔNLL| beyond 2 SE.

| model | policy | WikiText-2 fallback → fast | ΔNLL fast − fallback | C4 fallback → fast | ΔNLL fast − fallback |
|---|---|---:|---:|---:|---:|
| Nemotron-Nano-9B-v2 | BF16 | 8.0863 → 8.0863 | +0.00000 ± 0.00016 | 11.1534 → 11.1528 | -0.00006 ± 0.00011 |
| Nemotron-Nano-9B-v2 | NVFP4 | 8.4596 → 8.4535 | -0.00072 ± 0.00112 | 11.5420 → 11.5437 | +0.00015 ± 0.00075 |
| Nemotron-Nano-9B-v2 | FourOverSix | 8.4227 → 8.4260 | +0.00039 ± 0.00110 | 11.5093 → 11.5034 | -0.00051 ± 0.00075 |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 8.3477 → 8.3536 | +0.00070 ± 0.00111 | 11.4532 → 11.4442 | -0.00079 ± 0.00069 ✗ |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 8.3570 → 8.3532 | -0.00045 ± 0.00099 | 11.4468 → 11.4452 | -0.00014 ± 0.00071 |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 8.3706 → 8.3797 | +0.00108 ± 0.00101 ✗ | 11.4719 → 11.4677 | -0.00036 ± 0.00068 |
| Qwen3.8-27B | BF16 | 7.0509 → 7.0523 | +0.00020 ± 0.00048 | 9.8935 → 9.8931 | -0.00004 ± 0.00010 |
| Qwen3.8-27B | NVFP4 | 7.5506 → 7.5740 | +0.00310 ± 0.00496 | 10.2185 → 10.2231 | +0.00045 ± 0.00085 |
| Qwen3.8-27B | FourOverSix | 7.3215 → 7.3023 | -0.00262 ± 0.00344 | 10.1869 → 10.1903 | +0.00033 ± 0.00078 |
| Qwen3.8-27B | FlipQuant 8x64 | 7.1430 → 7.1399 | -0.00043 ± 0.00286 | 10.1358 → 10.1441 | +0.00082 ± 0.00070 ✗ |
| Qwen3.8-27B | FlipQuant 16x64 | 7.1705 → 7.1859 | +0.00215 ± 0.00329 | 10.1444 → 10.1410 | -0.00033 ± 0.00078 |
| Qwen3.8-27B | FlipQuant 256x64 | 7.2369 → 7.2195 | -0.00241 ± 0.00395 | 10.1620 → 10.1613 | -0.00006 ± 0.00078 |

**21 of 24 fast − fallback differences are within 2 SE.**

The table's comparisons in each env (ΔNLL vs FourOverSix, ± 2 SE; * = beyond 2 SE):

| model | policy | WikiText-2 fallback | fast | C4 fallback | fast |
|---|---|---:|---:|---:|---:|
| Nemotron-Nano-9B-v2 | NVFP4 | +0.0044 ± 0.0016* | +0.0033 ± 0.0015* | +0.0028 ± 0.0010* | +0.0035 ± 0.0010* |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | -0.0089 ± 0.0012* | -0.0086 ± 0.0012* | -0.0049 ± 0.0008* | -0.0052 ± 0.0008* |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | -0.0078 ± 0.0012* | -0.0087 ± 0.0011* | -0.0054 ± 0.0008* | -0.0051 ± 0.0009* |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | -0.0062 ± 0.0012* | -0.0055 ± 0.0012* | -0.0033 ± 0.0008* | -0.0031 ± 0.0008* |
| Qwen3.8-27B | NVFP4 | +0.0308 ± 0.0081* | +0.0365 ± 0.0081* | +0.0031 ± 0.0010* | +0.0032 ± 0.0011* |
| Qwen3.8-27B | FlipQuant 8x64 | -0.0247 ± 0.0059* | -0.0225 ± 0.0044* | -0.0050 ± 0.0009* | -0.0045 ± 0.0010* |
| Qwen3.8-27B | FlipQuant 16x64 | -0.0208 ± 0.0046* | -0.0161 ± 0.0041* | -0.0042 ± 0.0010* | -0.0048 ± 0.0010* |
| Qwen3.8-27B | FlipQuant 256x64 | -0.0116 ± 0.0037* | -0.0114 ± 0.0041* | -0.0024 ± 0.0008* | -0.0028 ± 0.0008* |

Loss recovered over these 4 model–corpus pairs (Σ(NVFP4 − policy) / Σ(NVFP4 − BF16) in NLL), fallback → fast: NVFP4 0.0 % → 0.0 %; FourOverSix 22.8 % → 25.4 %; FlipQuant 8x64 47.0 % → 47.7 %; FlipQuant 16x64 44.1 % → 44.3 %; FlipQuant 256x64 35.9 % → 37.9 %.
