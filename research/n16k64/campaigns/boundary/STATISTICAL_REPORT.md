# Statistical report

## Frozen design

The outcome-blind coverage gate tested B=8, 6, then 4 and selected B=4 with 4 construction partitions. All 32 boundary maps per model use group-only maps against the same FourOverSix baseline. Corruption uses four disjoint near-boundary pools and nested p=(0, 0.1, 0.25, 0.5, 0.75, 1.0) maps with exact module quotas.

Natural inference units are C4 documents and WikiText articles. Every result uses 10,000 deterministic paired-cluster bootstrap draws. P-values use the finite plus-one rule and primary endpoints are Holm-adjusted in the four frozen 12/6/12/6 families. Correlation intervals are bounded to the valid parameter domain.

## Power qualification

Power was computed before new outcomes from centered, verified prior paired arrays. Existing effect locations were removed; only their paired covariance/noise was reused. Endpoint status did not select arms.

| Model | Corpus | Boundary slope | Selected−rejected | Weak boundary | Corruption slope | p=.50 | p=1 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | C4 | limited_inference | limited_inference | descriptive | limited_inference | descriptive | descriptive |
| Llama-3.1-8B | WikiText | limited_inference | limited_inference | descriptive | limited_inference | descriptive | descriptive |
| Qwen3-4B | C4 | descriptive | descriptive | limited_inference | limited_inference | descriptive | descriptive |
| Qwen3-4B | WikiText | descriptive | descriptive | descriptive | descriptive | descriptive | descriptive |
| Mistral-7B-v0.3 | C4 | limited_inference | limited_inference | limited_inference | limited_inference | descriptive | descriptive |
| Mistral-7B-v0.3 | WikiText | limited_inference | limited_inference | limited_inference | limited_inference | limited_inference | limited_inference |

### Numeric power and MDE

| Model | Corpus | Endpoint | Power @ .0010 | Power @ .0025 | MDE80 | MDE90 | Status |
|---|---|---|---:|---:|---:|---:|---|
| Llama-3.1-8B | C4 | `boundary_trend_slope_end_to_end` | 0.247 | 0.991 | 0.001768 | 0.001981 | limited_inference |
| Llama-3.1-8B | C4 | `corruption_slope_end_to_end` | 0.291 | 0.992 | 0.001708 | 0.001932 | limited_inference |
| Llama-3.1-8B | C4 | `p050_minus_p0` | 0.058 | 0.622 | 0.002881 | 0.003179 | descriptive |
| Llama-3.1-8B | C4 | `p100_minus_p0` | 0.058 | 0.622 | 0.002881 | 0.003179 | descriptive |
| Llama-3.1-8B | C4 | `selected_minus_rejected_aggregate` | 0.158 | 0.870 | 0.002277 | 0.002630 | limited_inference |
| Llama-3.1-8B | C4 | `weakest_selected_minus_nearest_rejected` | 0.074 | 0.752 | 0.002603 | 0.002875 | descriptive |
| Llama-3.1-8B | WikiText | `boundary_trend_slope_end_to_end` | 0.225 | 0.989 | 0.001795 | 0.002013 | limited_inference |
| Llama-3.1-8B | WikiText | `corruption_slope_end_to_end` | 0.140 | 0.898 | 0.002226 | 0.002510 | limited_inference |
| Llama-3.1-8B | WikiText | `p050_minus_p0` | 0.076 | 0.644 | 0.002917 | 0.003309 | descriptive |
| Llama-3.1-8B | WikiText | `p100_minus_p0` | 0.076 | 0.644 | 0.002917 | 0.003309 | descriptive |
| Llama-3.1-8B | WikiText | `selected_minus_rejected_aggregate` | 0.122 | 0.827 | 0.002418 | 0.002755 | limited_inference |
| Llama-3.1-8B | WikiText | `weakest_selected_minus_nearest_rejected` | 0.107 | 0.794 | 0.002515 | 0.002819 | descriptive |
| Qwen3-4B | C4 | `boundary_trend_slope_end_to_end` | 0.095 | 0.738 | 0.002652 | 0.003003 | descriptive |
| Qwen3-4B | C4 | `corruption_slope_end_to_end` | 0.141 | 0.888 | 0.002255 | 0.002543 | limited_inference |
| Qwen3-4B | C4 | `p050_minus_p0` | 0.025 | 0.148 | 0.005549 | 0.006273 | descriptive |
| Qwen3-4B | C4 | `p100_minus_p0` | 0.025 | 0.148 | 0.005549 | 0.006273 | descriptive |
| Qwen3-4B | C4 | `selected_minus_rejected_aggregate` | 0.046 | 0.414 | 0.003599 | 0.004025 | descriptive |
| Qwen3-4B | C4 | `weakest_selected_minus_nearest_rejected` | 0.126 | 0.851 | 0.002358 | 0.002661 | limited_inference |
| Qwen3-4B | WikiText | `boundary_trend_slope_end_to_end` | 0.028 | 0.180 | 0.005233 | 0.005950 | descriptive |
| Qwen3-4B | WikiText | `corruption_slope_end_to_end` | 0.023 | 0.151 | 0.005566 | 0.006260 | descriptive |
| Qwen3-4B | WikiText | `p050_minus_p0` | 0.013 | 0.040 | 0.009129 | 0.010201 | descriptive |
| Qwen3-4B | WikiText | `p100_minus_p0` | 0.013 | 0.040 | 0.009129 | 0.010201 | descriptive |
| Qwen3-4B | WikiText | `selected_minus_rejected_aggregate` | 0.013 | 0.067 | 0.007235 | 0.008106 | descriptive |
| Qwen3-4B | WikiText | `weakest_selected_minus_nearest_rejected` | 0.020 | 0.099 | 0.006844 | 0.007811 | descriptive |
| Mistral-7B-v0.3 | C4 | `boundary_trend_slope_end_to_end` | 0.434 | 1.000 | 0.001364 | 0.001495 | limited_inference |
| Mistral-7B-v0.3 | C4 | `corruption_slope_end_to_end` | 0.413 | 1.000 | 0.001394 | 0.001545 | limited_inference |
| Mistral-7B-v0.3 | C4 | `p050_minus_p0` | 0.062 | 0.541 | 0.003016 | 0.003257 | descriptive |
| Mistral-7B-v0.3 | C4 | `p100_minus_p0` | 0.062 | 0.541 | 0.003016 | 0.003257 | descriptive |
| Mistral-7B-v0.3 | C4 | `selected_minus_rejected_aggregate` | 0.527 | 0.999 | 0.001364 | 0.001580 | limited_inference |
| Mistral-7B-v0.3 | C4 | `weakest_selected_minus_nearest_rejected` | 0.673 | 1.000 | 0.001144 | 0.001307 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `boundary_trend_slope_end_to_end` | 0.692 | 1.000 | 0.001123 | 0.001282 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `corruption_slope_end_to_end` | 0.486 | 1.000 | 0.001348 | 0.001532 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `p050_minus_p0` | 0.352 | 0.997 | 0.001591 | 0.001808 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `p100_minus_p0` | 0.352 | 0.997 | 0.001591 | 0.001808 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `selected_minus_rejected_aggregate` | 0.485 | 1.000 | 0.001378 | 0.001573 | limited_inference |
| Mistral-7B-v0.3 | WikiText | `weakest_selected_minus_nearest_rejected` | 0.398 | 0.999 | 0.001502 | 0.001706 | limited_inference |

## Frozen multiplicity families

- `A_development_primary`: 12 endpoints; 9 Holm rejections.
- `A_validation_primary`: 6 endpoints; 4 Holm rejections.
- `B_development_primary`: 12 endpoints; 12 Holm rejections.
- `B_validation_primary`: 6 endpoints; 6 Holm rejections.

## Interpretation

Construction partitions and corruption pools are sensitivity/replicate factors, not independent model replications. Delta NLL is delta log PPL; relative PPL change is exp(delta NLL)-1. Mistral is a one-shot analysis validation, and Qwen3-4B remains the predeclared high-response case.

The continuous boundary slopes and selected-versus-rejected aggregate contrasts favor the frozen ranking, but the weakest-selected versus nearest-rejected contrast is heterogeneous: it is positive for Llama on both corpora, negative for Qwen on both, negative for Mistral C4, and near zero for Mistral WikiText. The evidence therefore supports an aggregate ordering signal, not a sharp causal discontinuity at kappa=3.

All six corruption slopes and p=1 contrasts are positive. At p=.50, score-near rejected replacements are less harmful than matched-random rejected replacements in all six point estimates, although two intervals include zero. This is ranking evidence under imposed stress, not an estimate of a real selector error rate.

## Operational protocol conformance

The 24-hour target and hour-20 GPU-launch cutoff were not met. Required Llama and Qwen retries began after the frozen cutoff because their original runs were invalidated on foreign co-tenancy and workflow continuation occurred later. `PROTOCOL_DEVIATIONS.json` records exact timestamps and safeguards. No outcome-driven scientific definition changed, but the late launch remains a submission-risk limitation.

Two CPU analysis attempts also failed before successful inference: attempt1 used invalid relative bind mounts, and attempt2 stopped before reading outcome values because Qwen's prior report carried extra screening-provenance metadata that changed an overbroad container manifest hash. `ANALYSIS_CORRECTIONS.json` freezes the outcome-blind correction: exact model, revision, token-window hashes, natural-cluster assignments, and token accounting must match directly. The original frozen script is preserved byte-identical, and no map, endpoint, seed, family, or classification rule changed.
