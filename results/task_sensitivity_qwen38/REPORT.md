# Qwen3.8-27B: frozen 8x64 MixFP4 calibration

Two independent seeds use the predeclared [protocol](PROTOCOL.md).
Changes below compare the proposed map with corrected E2M1 W4A4,
including proposals rejected by validation.

| Seed | Validation decision | E0M3 tiles | Dataset | Baseline PPL | Proposal PPL | Delta PPL | Delta NLL ± 2 SE |
|---|---|---:|---|---:|---:|---:|---:|
| 20260909 | accepted | 69 | wikitext | 7.555223 | 7.133067 | -0.422156 | -0.057498 ± 0.011278 |
| 20260909 | accepted | 69 | c4 | 9.141328 | 9.137327 | -0.004001 | -0.000438 ± 0.002055 |
| 20260910 | accepted | 62 | wikitext | 7.555223 | 7.156911 | -0.398312 | -0.054161 ± 0.010020 |
| 20260910 | accepted | 62 | c4 | 9.141328 | 9.135967 | -0.005361 | -0.000587 ± 0.001852 |

Native BF16 reference: wikitext PPL 7.050639, c4 PPL 8.857866


Forecasts made from independent WikiText calibration validation:

| Seed | Forecast relative PPL change | Validation mean ± 2 SE, transformed to PPL | Observed WikiText change |
|---|---:|---:|---:|
| 20260909 | -9.59% | [-13.95%, -5.00%] | -5.59% |
| 20260910 | -4.70% | [-6.73%, -2.62%] | -5.27% |

Both WikiText test changes lie inside these descriptive validation intervals.
The validation point estimates are imperfect; the first overpredicts the gain.
Neither interval describes C4 transfer: C4 gains are tiny and inconclusive.
A final workload needs representative calibration and independent validation.

Seed 20260909: 0.000145% of text weight tiles selected.
Largest selections: `model.language_model.layers.33.linear_attn.in_proj_qkv` (5 tiles); `model.language_model.layers.24.linear_attn.in_proj_z` (5 tiles); `model.language_model.layers.24.linear_attn.in_proj_qkv` (4 tiles); `model.language_model.layers.28.mlp.up_proj` (3 tiles); `model.language_model.layers.7.self_attn.v_proj` (3 tiles).

Seed 20260910: 0.000130% of text weight tiles selected.
Largest selections: `model.language_model.layers.0.linear_attn.out_proj` (8 tiles); `model.language_model.layers.24.linear_attn.in_proj_z` (6 tiles); `model.language_model.layers.3.self_attn.q_proj` (6 tiles); `model.language_model.layers.24.linear_attn.in_proj_qkv` (4 tiles); `model.language_model.layers.3.self_attn.v_proj` (3 tiles).

The two maps share 44 tiles out of 87 in their union.

Diagnostic controls (registered after the first primary seed):

| Control | Tiles | Dataset | PPL | Delta PPL | Delta NLL ± 2 SE |
|---|---:|---|---:|---:|---:|
| matched_random | 69 | wikitext | 7.530938 | -0.024285 | -0.003220 ± 0.005657 |
| matched_random | 69 | c4 | 9.130084 | -0.011244 | -0.001231 ± 0.001732 |
| weight_mse | 43487115 | wikitext | 7.154471 | -0.400753 | -0.054502 ± 0.011371 |
| weight_mse | 43487115 | c4 | 9.139035 | -0.002293 | -0.000251 ± 0.002874 |

The exported native map and baseline replayed 16 held-out windows exactly
on a fresh model load (maximum NLL difference zero).

Direct paired MSE-minus-calibrated comparisons (negative favors MSE):

| Calibration seed | Dataset | Delta PPL | Delta NLL ± 2 SE |
|---|---|---:|---:|
| 20260909 | wikitext | +0.021403 | +0.002996 ± 0.004203 |
| 20260909 | c4 | +0.001708 | +0.000187 ± 0.002407 |
| 20260910 | wikitext | -0.002441 | -0.000341 ± 0.004081 |
| 20260910 | c4 | +0.003068 | +0.000336 ± 0.002473 |

The uncertainty column is a descriptive paired window standard error,
not a guarantee across domains. WikiText windows may be correlated.
These measurements cover text linear weights and inputs; recurrent state,
convolution, norms, embeddings, head, and vision retain native precision.
Fake quantization measures accuracy, not hardware speed.
