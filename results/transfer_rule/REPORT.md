# A fixed FP4 tile rule with measured transfer

The unchanged procedure improves **21/21** matched causal perplexity comparisons across **7 models**, from OPT350M to Llama8B. **20** gains have supporting descriptive paired2SE intervals; **0** comparisons have supported harm. It beats weight-MSE election in **21/21** point comparisons.

All rows below use per-token FP32 activation factors for both the selected map and its baseline. There is one fixed map per model across literature, science and government text. No confirmation loss chooses a map, tile budget, calibration source, seed or checkpoint.

## The rule

1. Form fixed FourOverSix and E0M3-alpha1 candidates for every legal8x64 weight tile.
2. Build one shared table of CE and teacher-KL tile derivatives on a fixed pool of192 sequences (64 C4,64 OpenWebMath,64 CodeParrot;512tokens each). All controls reuse that table.
3. For each tile, take the maximum of the CE and KL directional mean+2SE. Choose up to256 tiles with the most negative scores.
4. Freeze the map. Evaluate all domains with that same map, including failures.

The rule is the exact top-k solution of an additive surrogate. It is not an exact optimizer of nonlinear network loss. The cap256 and factor2 are common empirical constants, not parameters derived from a universal theorem. They remain unchanged across all seven models. [Full method](../pooled_confirmation/METHOD.md).

## Matched causal results

[Figure (PDF)](causal_transfer.pdf) · [SVG](causal_transfer.svg)

| Model / domain | FourOverSix | Selected | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| llama1b / literature | 27.428086 | 26.170161 | -1.257925 | -0.046948 ±0.006983 |
| llama1b / science | 22.997938 | 22.191191 | -0.806747 | -0.035709 ±0.005738 |
| llama1b / government | 15.939049 | 15.437538 | -0.501511 | -0.031970 ±0.006389 |
| opt350m / literature | 26.780318 | 26.318007 | -0.462311 | -0.017414 ±0.005225 |
| opt350m / science | 39.988506 | 38.978981 | -1.009525 | -0.025570 ±0.007068 |
| opt350m / government | 19.305159 | 19.041959 | -0.263200 | -0.013727 ±0.005354 |
| qwen06b / literature | 42.286833 | 38.234706 | -4.052128 | -0.100732 ±0.010663 |
| qwen06b / science | 22.977852 | 21.385315 | -1.592537 | -0.071826 ±0.009366 |
| qwen06b / government | 20.965000 | 19.293438 | -1.671562 | -0.083089 ±0.008691 |
| pythia14b / literature | 17.355897 | 17.085960 | -0.269937 | -0.015675 ±0.005223 |
| pythia14b / science | 19.709933 | 19.468234 | -0.241699 | -0.012339 ±0.006259 |
| pythia14b / government | 14.187066 | 13.978634 | -0.208432 | -0.014801 ±0.005609 |
| olmo1b / literature | 18.361552 | 18.272085 | -0.089467 | -0.004884 ±0.004963 |
| olmo1b / science | 21.505466 | 21.111510 | -0.393957 | -0.018489 ±0.013632 |
| olmo1b / government | 12.176172 | 12.085915 | -0.090257 | -0.007440 ±0.003889 |
| qwen4b / literature | 21.779775 | 20.537961 | -1.241814 | -0.058707 ±0.007401 |
| qwen4b / science | 13.656461 | 13.018419 | -0.638042 | -0.047848 ±0.005846 |
| qwen4b / government | 12.757487 | 12.081077 | -0.676410 | -0.054478 ±0.006661 |
| llama8b / literature | 11.675033 | 11.591275 | -0.083758 | -0.007200 ±0.004854 |
| llama8b / science | 10.804980 | 10.701076 | -0.103904 | -0.009663 ±0.004517 |
| llama8b / government | 8.251998 | 8.181400 | -0.070598 | -0.008592 ±0.003871 |

## Held-out C4 follow-up

The same frozen maps improve 7/7 C4 comparisons, with 7 descriptive paired two-SE supported gains and 0 supported harms. Each model uses 256 distinct validation documents with 512-token crops, excluding exact calibration document hashes. All policies use causal per-token activation factors. No map is recalibrated or selected. C4 is a calibration source, so these seven measurements are held-out within-source evaluation, separate from the 21 transfer comparisons above.

| Model | FourOverSix C4 PPL | Selected C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| opt350m | 26.743945 | 26.469613 | -0.274332 | -0.010311 ±0.002727 |
| qwen06b | 37.900671 | 34.792433 | -3.108238 | -0.085569 ±0.005095 |
| llama1b | 24.898634 | 24.124422 | -0.774213 | -0.031588 ±0.002935 |
| olmo1b | 14.984432 | 14.883008 | -0.101424 | -0.006792 ±0.002574 |
| pythia14b | 20.471159 | 20.162666 | -0.308493 | -0.015184 ±0.003167 |
| qwen4b | 21.928243 | 20.902742 | -1.025501 | -0.047895 ±0.003392 |
| llama8b | 11.600109 | 11.509863 | -0.090246 | -0.007810 ±0.002113 |

[Full C4 report and fixed controls](../c4_frozen/REPORT_332781.md). Pooled192 beats C4-only selection in 4/7 point comparisons. This does not change the earlier failed source-diversity screen.

## Qwen3.8-27B C4 extension

This separately measured target uses the same 192 calibration sequences per recipe, CE/KL two-SE rule and 256-tile cap, with native Transformers 5.16.1 hybrid text support. Evaluation uses 256 held-out C4 validation documents and causal per-token activation factors. No loss backtracking or test-based map selection is used.

FourOverSix C4 PPL is 12.644977; selected PPL is 12.635323, ΔPPL -0.009654. Paired ΔNLL is -0.000764 ±0.001538 (descriptive 2SE). Supported gain: False; gain of at least 0.01 PPL: False. This result limits the seven-model evidence above: the current rule does not establish a meaningful supported C4 gain on the 27B target.

[Full 27B controls](../pooled_qwen27b/model_332840/REPORT.md) · [Eight-model C4 comparison and overlap audit](../pooled_qwen27b/C4_COMPARISON_332840.md).

## What the experiments establish

The initial confirmation used five model families and three data families not inspected during method development; Pythia and OLMo supplied two new architecture families. A subsequent causal audit replayed the exact same maps and inputs. The4B/8B follow-up tested larger models on these now-inspected data families. The21 rows above combine the15 causal replays and6 size-transfer results. They do not count the earlier window-scaling evaluations again as independent evidence.

The old window-wide activation factor changed earlier logits when only future tokens were replaced in all five models checked under both conventions. Per-token factors removed that dependence, with bitwise-identical prefix logits for both baseline and selected maps in all seven models. The frozen maps retained useful gains. [Causal audit](../causal_replay/REPORT_332374.md), [scale-transfer report](../pooled_scale/REPORT_332389.md).

## Claims that remain limited

The original confirmation passed its baseline-transfer components but **failed its complete prespecified screen**: it beat C4-only selection in8/15 point comparisons, short of the required9. That criterion has not been changed. The later causal audit and size-transfer test answer separately declared questions; their passes do not retroactively rescue the earlier failed comparator criterion.

Under the combined causal convention, pooled192 beats C4-only64 in16/21 point comparisons. The equal-token mixed64 comparison improves14/21, with5 supported gains and2 supported harms. Thus source diversity is not established as uniformly preferable or as the sole cause of the gains.

This supports a transferable calibrated procedure, not a universal weight-only rule. Gradient ranking, distillation and sparse optimization are established tools. Paper-level novelty requires a precise comparison with prior work; it does not follow from a favorable table. The present evidence is reference-text loss from simulated nonhead-linear W4A4, not generation accuracy or native FP4-kernel throughput. Per-token factors change the activation representation and are not claimed as a free change to a single-tensor-factor CUDA interface. Only one fixed calibration pool per model is used here; multi-pool seed replication remains unmeasured.

## Research record and artifacts

[Full continuation log](../format_directions/CONTINUATION_20260908.md) preserves unsuccessful directions. All heavy computation, tests, fitting audits and summaries ran in Slurm. Model revisions, source-weight hashes, input hashes, raw paired losses and maps are retained alongside each source report. The three older primary maps replayed exactly.

```json
{
  "models": 7,
  "cells": 21,
  "gains": 21,
  "supported_gains": 20,
  "supported_harms": 0,
  "beats_weight_mse": 21,
  "beats_c4_64": 16,
  "equal_token_diversity_point_gains": 14,
  "equal_token_diversity_supported_gains": 5,
  "equal_token_diversity_supported_harms": 2,
  "original_confirmation_full_screen_passes": false,
  "causal_audit_passes": true,
  "scale_transfer_passes": true,
  "source_reports": [
    "results/causal_replay/model_332374_llama1b/report.json",
    "results/causal_replay/model_332374_opt350m/report.json",
    "results/causal_replay/model_332374_qwen06b/report.json",
    "results/causal_replay/model_332374_pythia14b/report.json",
    "results/causal_replay/model_332374_olmo1b/report.json",
    "results/pooled_scale/model_332389_qwen4b/report.json",
    "results/pooled_scale/model_332389_llama8b/report.json"
  ],
  "heldout_c4_followup": {
    "summary_path": "results/c4_frozen/summary_332781.json",
    "models": 7,
    "documents_per_model": 256,
    "tokens_per_document": 512,
    "point_gains": 7,
    "gains_at_least_0p01_ppl": 7,
    "supported_gains": 7,
    "supported_harms": 0,
    "beats_c4_only": 4,
    "beats_weight_mse": 7,
    "exact_prefix_independence_models": 7,
    "source_reports": [
      "results/c4_frozen/model_332781_opt350m/report.json",
      "results/c4_frozen/model_332781_qwen06b/report.json",
      "results/c4_frozen/model_332781_llama1b/report.json",
      "results/c4_frozen/model_332781_olmo1b/report.json",
      "results/c4_frozen/model_332781_pythia14b/report.json",
      "results/c4_frozen/model_332781_qwen4b/report.json",
      "results/c4_frozen/model_332781_llama8b/report.json"
    ]
  },
  "qwen27b_c4_followup": {
    "source_report": "results/pooled_qwen27b/model_332840/report.json",
    "selected_tiles": 256,
    "supported_gain": false,
    "gain_at_least_0p01_ppl": false,
    "mean_nll": -0.0007637564558535814,
    "two_se": 0.00153823923543746,
    "ppl_delta": -0.009653996049848956
  }
}
```
