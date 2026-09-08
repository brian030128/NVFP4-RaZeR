# Frozen-rule C4 results including Qwen3.8-27B

Same 192-sequence calibration recipe, CE/KL two-SE score, and 256-tile cap. One map per model; no candidate-loss backtracking or evaluation-based selection. All rows use 256 held-out C4 validation documents with 512-token crops and causal per-token activation factors. C4 is a calibration source; these are within-source results.

| Model | FourOverSix C4 PPL | Selected C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| opt350m | 26.743945 | 26.469613 | -0.274332 | -0.010311 ±0.002727 |
| qwen06b | 37.900671 | 34.792433 | -3.108238 | -0.085569 ±0.005095 |
| llama1b | 24.898634 | 24.124422 | -0.774213 | -0.031588 ±0.002935 |
| olmo1b | 14.984432 | 14.883008 | -0.101424 | -0.006792 ±0.002574 |
| pythia14b | 20.471159 | 20.162666 | -0.308493 | -0.015184 ±0.003167 |
| qwen4b | 21.928243 | 20.902742 | -1.025501 | -0.047895 ±0.003392 |
| llama8b | 11.600109 | 11.509863 | -0.090246 | -0.007810 ±0.002113 |
| qwen27b | 12.644977 | 12.635323 | -0.009654 | -0.000764 ±0.001538 |

The 27B map selects 256 E0M3 tiles. It uses native Transformers 5.16.1 for the hybrid Qwen3.5-family text architecture; the preceding seven models used 4.57.3. Scope is nonhead text linear weights and inputs; vision, recurrent state, convolution and normalization remain native. These are simulated W4A4 results, not native throughput or generation accuracy.

[27B controls and detailed result](model_332840/REPORT.md). [Recomputed eight-model calibration overlap audit](disjointness_332840.json). All 27B model-loading, map, label-count, hash-exclusion and prefix-independence checks pass. One calibration pool per model and descriptive two-SE intervals do not establish a universal guarantee. The earlier source-diversity screen remains failed.

Of the eight point comparisons, 7 improve by at least 0.01 PPL, the practical reference threshold used in the preceding record. The 27B result meets that reference: False. The follow-up protocol specifies measurement without a new pass/fail screen; the reference threshold is reported separately from the sign and paired uncertainty.

```json
{
  "models": 8,
  "point_gains": 8,
  "gains_at_least_0p01_ppl": 7,
  "supported_gains": 7,
  "supported_harms": 0,
  "qwen27b_selected_tiles": 256,
  "qwen27b_gain_at_least_0p01_ppl": false,
  "qwen27b_contrasts": {
    "four_over_six": {
      "mean_nll": -0.0007637564558535814,
      "two_se": 0.00153823923543746,
      "ppl_delta": -0.009653996049848956
    },
    "c4_64": {
      "mean_nll": 0.00039052474312484264,
      "two_se": 0.0015252462956894656,
      "ppl_delta": 0.004933443071992727
    },
    "mixed64": {
      "mean_nll": 0.000623943516984582,
      "two_se": 0.001532902460918862,
      "ppl_delta": 0.007881269171930327
    },
    "weight_mse": {
      "mean_nll": -0.002840957371518016,
      "two_se": 0.0018044049756420235,
      "ppl_delta": -0.035947453753632175
    }
  },
  "qwen27b_report": "results/pooled_qwen27b/model_332840/report.json",
  "previous_summary": "results/c4_frozen/summary_332781.json",
  "calibration_hash_overlap": 0
}
```
