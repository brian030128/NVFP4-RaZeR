# O1: E0M3 share of the Hadamard-basis FlipQuant 16x64 maps vs the unrotated release maps

Hadamard: part H's map (`RUN/ptq/<model>/fq-16x64_hadamard.pt`; TM-OPT+TC with the release settings in the block-16 Hadamard basis, n16k64-fast). Release: `flipquant_release/<model>/flipquant_16x64.pt` (n16k64). A block-16 Hadamard acts within each 16-column scale group, so a 16x64 tile covers the same weights in both bases and the tiles can be compared position by position. **Env caveat:** part H trained the Hadamard maps in n16k64-fast and the release maps were trained in n16k64, so the rotated vs unrotated shares mix the basis change with the env. Both = E0M3 in both maps; by chance = the expected count if the two maps chose their E0M3 tiles independently within each module (sum of a_l b_l / tiles_l); Jaccard = both / E0M3 in either.

## Nemotron-Nano-9B-v2

| projection | modules | tiles | E0M3 Hadamard | E0M3 release | Hadamard − release (pp) | both | both by chance | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mamba / in_proj | 27 | 2,676,240 | 25,741 (0.962 %) | 25,592 (0.956 %) | +0.006 | 498 | 256 | 0.010 |
| Mamba / out_proj | 27 | 1,209,600 | 16,545 (1.368 %) | 16,003 (1.323 %) | +0.045 | 390 | 245 | 0.012 |
| **Mamba** | 54 | 3,885,840 | 42,286 (1.088 %) | 41,595 (1.070 %) | +0.018 | 888 | 501 | 0.011 |
| attention / q_proj | 4 | 89,600 | 946 (1.056 %) | 934 (1.042 %) | +0.013 | 25 | 10 | 0.013 |
| attention / k_proj | 4 | 17,920 | 253 (1.412 %) | 293 (1.635 %) | -0.223 | 12 | 4 | 0.022 |
| attention / v_proj | 4 | 17,920 | 450 (2.511 %) | 350 (1.953 %) | +0.558 | 18 | 9 | 0.023 |
| attention / o_proj | 4 | 89,600 | 1,970 (2.199 %) | 1,519 (1.695 %) | +0.503 | 52 | 35 | 0.015 |
| **attention** | 16 | 215,040 | 3,619 (1.683 %) | 3,096 (1.440 %) | +0.243 | 107 | 58 | 0.016 |
| MLP / up_proj | 25 | 1,715,000 | 15,291 (0.892 %) | 15,704 (0.916 %) | -0.024 | 256 | 156 | 0.008 |
| MLP / down_proj | 25 | 1,715,000 | 19,084 (1.113 %) | 19,703 (1.149 %) | -0.036 | 311 | 241 | 0.008 |
| **MLP** | 50 | 3,430,000 | 34,375 (1.002 %) | 35,407 (1.032 %) | -0.030 | 567 | 397 | 0.008 |
| **all** | 120 | 7,530,880 | 80,280 (1.066 %) | 80,098 (1.064 %) | +0.002 | 1,562 | 956 | 0.010 |

## Qwen3.8-27B

| projection | modules | tiles | E0M3 Hadamard | E0M3 release | Hadamard − release (pp) | both | both by chance | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gated DeltaNet / in_proj_qkv | 48 | 2,457,600 | 11,235 (0.457 %) | 11,524 (0.469 %) | -0.012 | 232 | 62 | 0.010 |
| Gated DeltaNet / in_proj_z | 48 | 1,474,560 | 5,912 (0.401 %) | 6,396 (0.434 %) | -0.033 | 135 | 29 | 0.011 |
| Gated DeltaNet / in_proj_b | 48 | 11,520 | 56 (0.486 %) | 60 (0.521 %) | -0.035 | 1 | 0 | 0.009 |
| Gated DeltaNet / in_proj_a | 48 | 11,520 | 75 (0.651 %) | 64 (0.556 %) | +0.095 | 0 | 0 | 0.000 |
| Gated DeltaNet / out_proj | 48 | 1,474,560 | 7,440 (0.505 %) | 7,504 (0.509 %) | -0.004 | 83 | 45 | 0.006 |
| **Gated DeltaNet** | 240 | 5,429,760 | 24,718 (0.455 %) | 25,548 (0.471 %) | -0.015 | 451 | 136 | 0.009 |
| attention / q_proj | 16 | 983,040 | 4,690 (0.477 %) | 4,519 (0.460 %) | +0.017 | 62 | 29 | 0.007 |
| attention / k_proj | 16 | 81,920 | 341 (0.416 %) | 358 (0.437 %) | -0.021 | 3 | 2 | 0.004 |
| attention / v_proj | 16 | 81,920 | 838 (1.023 %) | 734 (0.896 %) | +0.127 | 26 | 11 | 0.017 |
| attention / o_proj | 16 | 491,520 | 4,427 (0.901 %) | 3,804 (0.774 %) | +0.127 | 62 | 44 | 0.008 |
| **attention** | 64 | 1,638,400 | 10,296 (0.628 %) | 9,415 (0.575 %) | +0.054 | 153 | 86 | 0.008 |
| MLP / gate_proj | 64 | 5,570,560 | 19,092 (0.343 %) | 20,907 (0.375 %) | -0.033 | 94 | 82 | 0.002 |
| MLP / up_proj | 64 | 5,570,560 | 23,124 (0.415 %) | 24,041 (0.432 %) | -0.016 | 190 | 132 | 0.004 |
| MLP / down_proj | 64 | 5,570,560 | 24,241 (0.435 %) | 25,367 (0.455 %) | -0.020 | 205 | 179 | 0.004 |
| **MLP** | 192 | 16,711,680 | 66,457 (0.398 %) | 70,315 (0.421 %) | -0.023 | 489 | 393 | 0.004 |
| **all** | 496 | 23,779,840 | 101,471 (0.427 %) | 105,278 (0.443 %) | -0.016 | 1,093 | 615 | 0.005 |

