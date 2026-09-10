# Where the k=3 rule puts its E0M3 tiles

Rebuilt from the frozen calibration scores in `results/kse_selection/job_337344`; every model's per-k counts reproduce the published election exactly.

| Model | Tiles | Elected | Rate | Matrices holding one | Largest single matrix |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 13,631,488 | 3,345 | 1 in 4,075 | 216/224 | 836 |
| Qwen3-4B | 7,096,320 | 7,912 | 1 in 897 | 241/252 | 1,307 |
| Qwen3.8-27B | 47,559,680 | 3,785 | 1 in 12,565 | 379/496 | 177 |

| Projection | Llama-3.1-8B /1M | share | Qwen3-4B /1M | share | Qwen3.8-27B /1M | share |
|---|---|---|---|---|---|---|
| `q_proj` | 218.4 | 6.8% | 588.7 | 5.5% | 95.1 | 4.9% |
| `k_proj` | 423.4 | 3.3% | 721.6 | 1.7% | 18.3 | 0.1% |
| `v_proj` | 671.4 | 5.3% | 2728.9 | 6.4% | 518.8 | 2.2% |
| `o_proj` | 309.9 | 9.7% | 2482.1 | 23.1% | 241.1 | 6.3% |
| `gate_proj` | 124.0 | 13.6% | 338.1 | 7.5% | 34.0 | 10.0% |
| `up_proj` | 146.3 | 16.1% | 635.6 | 14.1% | 48.2 | 14.2% |
| `down_proj` | 412.0 | 45.2% | 1888.6 | 41.8% | 41.0 | 12.1% |
| `out_proj` | — | — | — | — | 105.1 | 8.2% |
| `in_proj_qkv` | — | — | — | — | 240.1 | 31.2% |
| `in_proj_z` | — | — | — | — | 135.6 | 10.6% |
| `in_proj_b` | — | — | — | — | 260.4 | 0.2% |
| `in_proj_a` | — | — | — | — | 173.6 | 0.1% |

| Depth quarter | Llama-3.1-8B | Qwen3-4B | Qwen3.8-27B |
|---|---|---|---|
| first | 634 (19.0%) | 1,777 (22.5%) | 409 (10.8%) |
| second | 535 (16.0%) | 1,238 (15.6%) | 632 (16.7%) |
| third | 490 (14.6%) | 1,087 (13.7%) | 958 (25.3%) |
| last | 1,686 (50.4%) | 3,810 (48.2%) | 1,786 (47.2%) |

## Llama-3.1-8B

3,345 elected tiles in 216 of 224 matrices; the top five matrices hold 32.0% and 10 matrices hold exactly one.

| Matrix | Grid (rows x cols) | Tiles | Elected | Share |
|---|---|---:|---:|---:|
| `L31.down_proj` | 512 x 224 | 114,688 | 836 | 25.0% |
| `L29.down_proj` | 512 x 224 | 114,688 | 70 | 2.1% |
| `L30.down_proj` | 512 x 224 | 114,688 | 66 | 2.0% |
| `L31.gate_proj` | 1792 x 64 | 114,688 | 51 | 1.5% |
| `L1.down_proj` | 512 x 224 | 114,688 | 47 | 1.4% |
| `L28.down_proj` | 512 x 224 | 114,688 | 45 | 1.3% |
| `L30.up_proj` | 1792 x 64 | 114,688 | 40 | 1.2% |
| `L31.up_proj` | 1792 x 64 | 114,688 | 39 | 1.2% |
| `L30.gate_proj` | 1792 x 64 | 114,688 | 35 | 1.0% |
| `L29.up_proj` | 1792 x 64 | 114,688 | 32 | 1.0% |

| Geometry statistic | Observed | Uniform placement |
|---|---:|---:|
| Distinct row bands touched | 2,612 | 2881 |
| Distinct column bands touched | 2,231 | 2479 |
| Most tiles in one row band | 19 | 6.7 |
| Most tiles in one column band | 122 | 10.0 |
| Most tiles on one column index, one kind | 125 | 16.4 |
| Most matrices sharing a column index, one kind | 15 | 13.1 |
| Most tiles on one residual channel band | 79 | 35.3 |
| Most residual-reading matrices sharing that band | 44 | 30.6 |

1,508 of the elected tiles sit in the 152 matrices that read the residual stream, where a column index is the same hidden channel band in every layer:

| Hidden channels | Tiles | Matrices |
|---|---:|---:|
| 4032–4095 | 79 | 44 |
| 768–831 | 70 | 39 |
| 1344–1407 | 69 | 41 |
| 2880–2943 | 49 | 30 |
| 1792–1855 | 41 | 33 |

Binding objective: CE for 1,863 tiles (55.7%), KL for 1,482. Median t at elected tiles: CE -3.41, KL -3.58.

| Kind | Column index | Tiles |
|---|---:|---:|
| `down_proj` (of 224) | 198 | 125 |
| `down_proj` (of 224) | 136 | 122 |
| `down_proj` (of 224) | 64 | 88 |
| `down_proj` (of 224) | 189 | 84 |
| `down_proj` (of 224) | 3 | 71 |

## Qwen3-4B

7,912 elected tiles in 241 of 252 matrices; the top five matrices hold 38.5% and 11 matrices hold exactly one.

| Matrix | Grid (rows x cols) | Tiles | Elected | Share |
|---|---|---:|---:|---:|
| `L35.down_proj` | 320 x 152 | 48,640 | 1,307 | 16.5% |
| `L34.down_proj` | 320 x 152 | 48,640 | 813 | 10.3% |
| `L1.down_proj` | 320 x 152 | 48,640 | 564 | 7.1% |
| `L34.o_proj` | 320 x 64 | 20,480 | 187 | 2.4% |
| `L35.o_proj` | 320 x 64 | 20,480 | 172 | 2.2% |
| `L35.up_proj` | 1216 x 40 | 48,640 | 136 | 1.7% |
| `L34.q_proj` | 512 x 40 | 20,480 | 114 | 1.4% |
| `L18.o_proj` | 320 x 64 | 20,480 | 97 | 1.2% |
| `L17.o_proj` | 320 x 64 | 20,480 | 94 | 1.2% |
| `L21.o_proj` | 320 x 64 | 20,480 | 93 | 1.2% |

| Geometry statistic | Observed | Uniform placement |
|---|---:|---:|
| Distinct row bands touched | 4,947 | 5780 |
| Distinct column bands touched | 2,977 | 4012 |
| Most tiles in one row band | 25 | 10.9 |
| Most tiles in one column band | 56 | 17.1 |
| Most tiles on one column index, one kind | 269 | 42.6 |
| Most matrices sharing a column index, one kind | 34 | 24.2 |
| Most tiles on one residual channel band | 644 | 88.2 |
| Most residual-reading matrices sharing that band | 146 | 57.6 |

2,775 of the elected tiles sit in the 169 matrices that read the residual stream, where a column index is the same hidden channel band in every layer:

| Hidden channels | Tiles | Matrices |
|---|---:|---:|
| 0–63 | 644 | 146 |
| 64–127 | 465 | 114 |
| 384–447 | 244 | 80 |
| 320–383 | 128 | 47 |
| 192–255 | 99 | 54 |

Binding objective: CE for 1,005 tiles (12.7%), KL for 6,907. Median t at elected tiles: CE -4.68, KL -3.53.

| Kind | Column index | Tiles |
|---|---:|---:|
| `up_proj` (of 40) | 0 | 269 |
| `up_proj` (of 40) | 1 | 226 |
| `gate_proj` (of 40) | 0 | 120 |
| `v_proj` (of 40) | 0 | 115 |
| `gate_proj` (of 40) | 1 | 107 |

## Qwen3.8-27B

3,785 elected tiles in 379 of 496 matrices; the top five matrices hold 19.0% and 42 matrices hold exactly one.

| Matrix | Grid (rows x cols) | Tiles | Elected | Share |
|---|---|---:|---:|---:|
| `L48.in_proj_qkv` | 1280 x 80 | 102,400 | 177 | 4.7% |
| `L49.in_proj_qkv` | 1280 x 80 | 102,400 | 164 | 4.3% |
| `L50.in_proj_qkv` | 1280 x 80 | 102,400 | 147 | 3.9% |
| `L46.in_proj_qkv` | 1280 x 80 | 102,400 | 117 | 3.1% |
| `L63.q_proj` | 1536 x 80 | 122,880 | 113 | 3.0% |
| `L34.in_proj_qkv` | 1280 x 80 | 102,400 | 75 | 2.0% |
| `L30.in_proj_qkv` | 1280 x 80 | 102,400 | 64 | 1.7% |
| `L63.o_proj` | 640 x 96 | 61,440 | 62 | 1.6% |
| `L33.in_proj_qkv` | 1280 x 80 | 102,400 | 56 | 1.5% |
| `L63.down_proj` | 640 x 272 | 174,080 | 56 | 1.5% |

| Geometry statistic | Observed | Uniform placement |
|---|---:|---:|
| Distinct row bands touched | 2,690 | 3708 |
| Distinct column bands touched | 2,648 | 3088 |
| Most tiles in one row band | 21 | 3.0 |
| Most tiles in one column band | 28 | 7.2 |
| Most tiles on one column index, one kind | 137 | 25.0 |
| Most matrices sharing a column index, one kind | 25 | 14.8 |
| Most tiles on one residual channel band | 358 | 50.0 |
| Most residual-reading matrices sharing that band | 102 | 37.8 |

2,781 of the elected tiles sit in the 257 matrices that read the residual stream, where a column index is the same hidden channel band in every layer:

| Hidden channels | Tiles | Matrices |
|---|---:|---:|
| 3968–4031 | 358 | 102 |
| 256–319 | 206 | 72 |
| 1024–1087 | 153 | 51 |
| 3456–3519 | 111 | 62 |
| 4352–4415 | 75 | 46 |

Binding objective: CE for 560 tiles (14.8%), KL for 3,225. Median t at elected tiles: CE -3.69, KL -3.42.

| Kind | Column index | Tiles |
|---|---:|---:|
| `in_proj_qkv` (of 80) | 62 | 137 |
| `in_proj_qkv` (of 80) | 4 | 100 |
| `in_proj_qkv` (of 80) | 16 | 74 |
| `up_proj` (of 80) | 62 | 72 |
| `in_proj_qkv` (of 80) | 54 | 60 |

