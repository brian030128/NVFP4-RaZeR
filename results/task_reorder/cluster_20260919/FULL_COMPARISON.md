# Full MixFP4 reordering comparison

All Δ values are relative to the model’s FourOverSix baseline; lower is better. Reference raw-map deltas retain the user-supplied values. Other deltas use full-precision report values. Type tiles are 256×64 unless labeled 8×64/fine. Tile counts are not directly comparable across sizes.

## Qwen3.8-27B references

The supplied 198-tile and local 195-tile raw maps are distinct.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 7.287076 | — | 10.188365 | — |
| Published MixFP4 k=3, 8×64 | 3785 | 7.214750 | -0.072327 | 10.149866 | -0.038499 |
| Raw 256×64 — user supplied | 198 | 7.275704 | -0.011373 | 10.177685 | -0.010680 |
| Raw 256×64 — regenerated locally | 195 | 7.266300 | -0.020776 | 10.176030 | -0.012335 |
| Full 8×64 — regenerated locally | 3787 | 7.212709 | -0.074368 | 10.150963 | -0.037402 |

## Qwen: final MLP, FourOverSix background

Only layer 63 gate/up/down can use E0M3; all other weights remain FourOverSix.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| Identity 256×64 | 1 | 7.286880 | -0.000197 | 10.188222 | -0.000143 |
| Identity 8×64 control | 20 | 7.287013 | -0.000064 | 10.187984 | -0.000381 |
| Row-only reorder | 15 | 7.280038 | -0.007039 | 10.180131 | -0.008234 |
| Column-only reorder | 1 | 7.286978 | -0.000099 | 10.188266 | -0.000099 |
| Both-axis reorder | 20 | 7.275291 | -0.011786 | 10.178476 | -0.009889 |

## Qwen: final MLP, local raw-256 background

Only layer 63 gate/up/down change. Elsewhere, the local raw map stays fixed. Pilot maps use 64 election sequences.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| Matched identity 256×64 | 193 | 7.266998 | -0.020079 | 10.176431 | -0.011934 |
| Identity 8×64 control | 192 coarse + 20 fine | 7.266977 | -0.020100 | 10.175884 | -0.012481 |
| Row-only reorder | 207 | 7.259481 | -0.027595 | 10.169099 | -0.019266 |
| Column-only reorder | 193 | 7.267141 | -0.019935 | 10.176552 | -0.011813 |
| Both-axis reorder | 212 | 7.255834 | -0.031243 | 10.167336 | -0.021029 |
| Shared MLP permutation — pooled | 194 | 7.267261 | -0.019815 | 10.176588 | -0.011777 |
| Shared MLP permutation — domain robust | 193 | 7.266998 | -0.020079 | 10.176431 | -0.011934 |
| Finite refinement: baseline fallback | 192 | 7.267115 | -0.019962 | 10.176598 | -0.011767 |

## Qwen: expanded scope, layers 56–63

All 24 MLP matrices use matched 64-sequence maps; weights outside this scope retain the local raw background. Last-MLP-only arms use identity maps in layers 56–62, so their background differs from the three-matrix pilot above.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| Matched identity 256×64 | 184 | 7.267151 | -0.019926 | 10.180216 | -0.008149 |
| Identity 8×64 control | 182 coarse + 92 fine | 7.266588 | -0.020488 | 10.177219 | -0.011146 |
| Eight-row groups: last MLP only | 192 | 7.263466 | -0.023611 | 10.177821 | -0.010544 |
| Eight-row groups: all eight MLPs | 201 | 7.263998 | -0.023079 | 10.175365 | -0.013000 |
| Individual rows: last MLP only | 198 | 7.259854 | -0.027222 | 10.174454 | -0.013911 |
| Both axes: last MLP only | 203 | 7.255811 | -0.031265 | 10.171021 | -0.017344 |

## Llama-3.1-8B references

No new Llama reordering measurements. Credential access is now verified; no Llama compute has been launched.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 6.875525 | — | 9.823733 | — |
| Published MixFP4 k=3, 8×64 | 3345 | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| Raw 256×64 — user supplied | 187 | 6.866879 | -0.008645 | 9.801361 | -0.022372 |

The aggressive rows_k1 candidate failed separate teacher-KL confirmation. The fallback row has zero E0M3 pilot tiles; it is not the rejected candidate’s PPL. Unrestricted and foldable refinement branches produced identical fallback weights and are combined.

All runs listed above completed. Duplicate controls from repeated jobs are deduplicated. Quality uses fake-quantized weights; native permutation overhead has not been measured here. Source report paths are in the CSV and JSON exports.
<!-- confirmed-compact-rows -->

## Qwen: independently confirmed compacted final-MLP rows

9 E0M3 tiles in the last MLP; 192 raw256 tiles elsewhere. Row-only; no column gather. Finite CE/KL selection on development documents, then one frozen 64-window independent confirmation. The new identity map differs from the earlier identity despite equal tile counts.

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| New matched identity 256×64 | 193 | 7.267063 | -0.020014 | 10.176316 | -0.012049 |
| Confirmed finite-pruned compacted rows | 201 | 7.262465 | -0.024611 | 10.172032 | -0.016333 |

Moved rows: 39,933 → 3,914 (−90.2%), with bitwise-identical quantized weights. This is a permutation-structure improvement; native B200 latency is not measured.

| Candidate comparison | Wiki ΔNLL ±2SE | C4 ΔNLL ±2SE |
|---|---:|---:|
| vs raw256 | -0.0005281 ±0.0001174 | -0.0003925 ±0.0000968 |
| vs identity | -0.0006329 ±0.0001205 | -0.0004212 ±0.0000927 |

Intervals are descriptive paired-window intervals, not adjusted for the research process. Raw/base controls were reused only after verifying raw maps, model/quantizer/activation settings, and exact token windows. Both new policies shared the full-model prefix with exact-logit audits.

Source: `/work/u4320956/task_reorder/pilot_20260919/qwen27b/fine_rows_v2/suffix_ce_restore_ppl/report.json`.

<!-- exact-both-compaction -->

## Qwen: exact compaction of the best both-axis layout

| Policy | E0M3 tiles | WikiText | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| Exact compacted both-axis | 212 | 7.255834 | -0.031243 | 10.167336 | -0.021029 |

These are reused parent PPL values, justified by bitwise equality of all three
actual quantized matrices and unchanged weights elsewhere. No new PPL run was
needed. Rows moved:39,934→5,001 (−87.5%); columns moved:27,216→1,392 (−94.9%).
Columns retain intact16-element scale groups. The gate projection becomes identity
on both axes. Native B200 latency is unmeasured. This equivalent model retains the
parent's math teacher-KL trade-off; it did not pass the stricter development gate.

Source: `/work/u4320956/task_reorder/pilot_20260919/qwen27b/fine_rows_v2/both_exact_compaction/report.json`.
