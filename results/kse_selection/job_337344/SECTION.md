The rule is a threshold, so the map it produces is an outcome rather than a design: nothing in it constrains which matrices, which layers, or which part of a matrix the elected tiles come from. Rebuilt from the frozen calibration scores, every model's per-k count reproduces the published election exactly.

### The elected tiles are spread over matrices and concentrated inside them

| Model | Tiles | Elected | Rate | Matrices holding at least one | Largest single matrix |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 13,631,488 | 3,345 | 1 in 4,075 | 216/224 | 836 (25%) |
| Qwen3-4B | 7,096,320 | 7,912 | 1 in 897 | 241/252 | 1,307 (17%) |
| Qwen3.8-27B | 47,559,680 | 3,785 | 1 in 12,565 | 379/496 | 177 (5%) |

Almost every weight matrix contributes something, so this is not a rule that fires on a handful of layers. The mass is nevertheless very uneven: the five largest contributors hold 32% (Llama-3.1-8B), 38% (Qwen3-4B), 19% (Qwen3.8-27B) of all elected tiles, and the single largest is `L31.down_proj` with 836, `L35.down_proj` with 1,307, `L48.in_proj_qkv` with 177.

### Which projections, and how deep

| Projection | Llama-3.1-8B per 1M | share | Qwen3-4B per 1M | share | Qwen3.8-27B per 1M | share |
|---|---|---|---|---|---|---|
| `q_proj` | 218 | 6.8% | 589 | 5.5% | 95 | 4.9% |
| `k_proj` | 423 | 3.3% | 722 | 1.7% | 18 | 0.1% |
| `v_proj` | 671 | 5.3% | 2729 | 6.4% | 519 | 2.2% |
| `o_proj` | 310 | 9.7% | 2482 | 23.1% | 241 | 6.3% |
| `gate_proj` | 124 | 13.6% | 338 | 7.5% | 34 | 10.0% |
| `up_proj` | 146 | 16.1% | 636 | 14.1% | 48 | 14.2% |
| `down_proj` | 412 | 45.2% | 1889 | 41.8% | 41 | 12.1% |
| `out_proj` | — | — | — | — | 105 | 8.2% |
| `in_proj_qkv` | — | — | — | — | 240 | 31.2% |
| `in_proj_z` | — | — | — | — | 136 | 10.6% |
| `in_proj_b` | — | — | — | — | 260 | 0.2% |
| `in_proj_a` | — | — | — | — | 174 | 0.1% |

Rates are elected tiles per million tiles of that kind, so they are comparable across matrices of very different sizes; shares are of the model's elected set. By rate, `v_proj` carries the highest rate on all three models (671, 2729, 519 per 1M), while the largest share of the elected set goes to the MLP `down_proj` on the two dense models and to the hybrid model's linear-attention `in_proj_qkv`. No kind is exempt and none is anywhere near saturated: the most enriched projection in the panel is `v_proj` on Qwen3-4B at 2729 per million, one tile in 366.

| Depth quarter | Llama-3.1-8B | Qwen3-4B | Qwen3.8-27B |
|---|---|---|---|
| first | 634 (19%) | 1,777 (22%) | 409 (11%) |
| second | 535 (16%) | 1,238 (16%) | 632 (17%) |
| third | 490 (15%) | 1,087 (14%) | 958 (25%) |
| last | 1,686 (50%) | 3,810 (48%) | 1,786 (47%) |

Depth is the strongest single predictor: about half of every model's elected tiles are in its last quarter of layers (50%, 48%, 47%), and the last layer alone holds 976 of 3,345 on Llama-3.1-8B, 1,854 of 7,912 on Qwen3-4B, 328 of 3,785 on Qwen3.8-27B. The first layers are the secondary peak, so the profile is a shallow U with a much heavier top end, not a monotone trend.

### Position inside a matrix

The reference for every row below is a seeded uniform placement that keeps each matrix's elected count and its tile grid and randomises only the positions, so a departure is structure the rule found rather than an artefact of where the tiles are.

| Statistic | Llama-3.1-8B obs. | unif. | Qwen3-4B obs. | unif. | Qwen3.8-27B obs. | unif. |
|---|---|---|---|---|---|---|
| Most tiles in one row band (output channels) | 19 | 7 | 25 | 11 | 21 | 3 |
| Most tiles in one column band (input channels) | 122 | 10 | 56 | 17 | 28 | 7 |
| Most tiles on one column index, one projection kind | 125 | 16 | 269 | 43 | 137 | 25 |
| Most matrices sharing a column index, one kind | 15 | 13 | 34 | 24 | 25 | 15 |
| Most tiles on one residual channel band | 79 | 35 | 644 | 88 | 358 | 50 |
| Most residual-reading matrices sharing that band | 44 | 31 | 146 | 58 | 102 | 38 |

Both axes are clustered well beyond chance on every model, in two different shapes. Along the input axis, one 64-channel band collects tiles from many output rows: 122 of the 836 tiles elected in `L31.down_proj` on Llama-3.1-8B share the single input band 12672–12735 while spreading over 391 of 512 output bands. Along the output axis the pattern is the mirror image — one row band, meaning one group of eight output channels, elected across much of its own row: 19 of the 64 column bands in row 1491 of `L29.up_proj` on Llama-3.1-8B are elected, 59% of everything that matrix contributes.

The clustering also runs across layers, not only inside a matrix. For the matrices that read the residual stream directly (`q/k/v_proj`, `gate/up_proj`, the hybrid model's `in_proj_*`) a column index means the same hidden channels in every layer, and the elected tiles pile onto a few such bands:

| Model | Hidden channels | Elected tiles there | Residual-reading matrices hit | Uniform placement |
|---|---|---:|---:|---:|
| Llama-3.1-8B | 4032–4095 | 79 | 44/152 | 31 matrices |
| Qwen3-4B | 0–63 | 644 | 146/169 | 58 matrices |
| Qwen3.8-27B | 3968–4031 | 358 | 102/257 | 38 matrices |

On the two Qwen models this is the clearest structure in the whole map: one 64-channel band of the residual stream is elected in 146 of 169 matrices on Qwen3-4B and 102 of 257 matrices on Qwen3.8-27B, against 58 and 38 under uniform placement. On Llama-3.1-8B the cross-layer version is much weaker (44 against 31) and the concentration lives inside single matrices instead. Either way the elected set is a property of particular input channels rather than of particular matrices. This report does not establish the mechanism; it is consistent with the known concentration of activation magnitude in a small number of channels, which is exactly where a uniform grid and a log-spaced grid differ most.

### Which objective binds

Both objectives must clear the bar, but the one that decides is model dependent: Llama-3.1-8B 56% CE, Qwen3-4B 13% CE, Qwen3.8-27B 15% CE. Median t statistics at elected tiles are around -3.4/-3.6, -4.7/-3.5, -3.7/-3.4 (CE/KL), so the elected set is not sitting on the threshold — it clears it comfortably on both. Neither objective is redundant: each is the binding constraint for a substantial share of the elected tiles on all three models.

[Per-model tables, top matrices and hot column indices](results/kse_selection/job_337344/REPORT.md)

