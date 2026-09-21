# N16K64 research decision (V91)

Protocol freeze: `freeze/PROTOCOL_FREEZE.json` sha256 `df78f1fbbd034cb8d2bc8e6f6a264e7e221e1b5f2b2ff99a3fe821369c5ac88f` (created 2026-09-11T09:09:16Z).

## 1. Frozen primary configuration

```
{
 "alternative": "E0M3 signed uniform alpha=1",
 "calibration": "seed0: 64 OpenWebMath + 64 CodeParrot sequences, 512 tokens",
 "count_cap": null,
 "k": 3,
 "name": "n16_k3",
 "rule": "elect tile iff max(mean_CE + 3 SE_CE, mean_KL + 3 SE_KL) < 0",
 "scale_block": 16,
 "type_block": [
  16,
  64
 ],
 "weight_baseline": "FourOverSix E2M1 (quant_nvfp4_4over6)"
}
```

## 2a. Development panel: exact-map perplexity

Source: `runs/V90_analyze_ppl_attempt5/analysis_ppl/LEGACY_PANEL_PPL.json` (sha256 `ea5f6e743d720e5b…`)

| model | dataset | PPL FourOverSix | PPL N8 k3 | PPL N16 k3 | dlogPPL N16-4/6 [95% CI] | dlogPPL N8-4/6 [95% CI] | dlogPPL N16-N8 [95% CI] | N8 tiles | N16 tiles | map sha256 (N16) |
|---|---|---:|---:|---:|---|---|---|---:|---:|---|
| llama8b | wiki | 6.8754 | 6.8414 | 6.8432 | -0.0047 [-0.0063, -0.0031] | -0.0050 [-0.0064, -0.0036] | +0.0003 [-0.0009, +0.0016] | 3130 | 1781 | `0920f55ddc053a5f…` |
| llama8b | c4 | 9.8254 | 9.7763 | 9.7762 | -0.0050 [-0.0067, -0.0035] | -0.0050 [-0.0072, -0.0031] | -0.0000 [-0.0015, +0.0016] | 3130 | 1781 | `0920f55ddc053a5f…` |
| qwen27b | wiki | 7.2839 | 7.2254 | 7.2456 | -0.0053 [-0.0089, -0.0003] | -0.0081 [-0.0115, -0.0044] | +0.0028 [-0.0001, +0.0060] | 3946 | 2168 | `90e1689b3a7e9251…` |
| qwen27b | c4 | 10.1894 | 10.1491 | 10.1564 | -0.0032 [-0.0040, -0.0025] | -0.0040 [-0.0047, -0.0032] | +0.0007 [+0.0000, +0.0014] | 3946 | 2168 | `90e1689b3a7e9251…` |
| qwen4b | wiki | 14.2062 | 11.8841 | 12.1968 | -0.1525 [-0.1617, -0.1441] | -0.1785 [-0.1882, -0.1695] | +0.0260 [+0.0232, +0.0286] | 7349 | 4077 | `188bf0e51c372cd8…` |
| qwen4b | c4 | 17.3024 | 15.8483 | 16.0485 | -0.0752 [-0.0784, -0.0721] | -0.0878 [-0.0915, -0.0841] | +0.0126 [+0.0112, +0.0139] | 7349 | 4077 | `188bf0e51c372cd8…` |

## 2b. Confirmatory panel: exact-map perplexity

Source: `runs/V90_analyze_ppl_attempt5/analysis_ppl/CONFIRMATORY_PPL.json` (sha256 `f68cba72f8632910…`)

| model | dataset | PPL FourOverSix | PPL N8 k3 | PPL N16 k3 | dlogPPL N16-4/6 [95% CI] | dlogPPL N8-4/6 [95% CI] | dlogPPL N16-N8 [95% CI] | N8 tiles | N16 tiles | map sha256 (N16) |
|---|---|---:|---:|---:|---|---|---|---:|---:|---|
| mistral7b | wiki | 5.5233 | 5.4987 | 5.5019 | -0.0039 [-0.0049, -0.0030] | -0.0045 [-0.0055, -0.0035] | +0.0006 [-0.0002, +0.0014] | 7501 | 4179 | `0c3d822a18d0480c…` |
| mistral7b | c4 | 8.0674 | 8.0456 | 8.0475 | -0.0025 [-0.0036, -0.0015] | -0.0027 [-0.0035, -0.0020] | +0.0002 [-0.0006, +0.0010] | 7501 | 4179 | `0c3d822a18d0480c…` |
| olmo2_13b | wiki | 5.3462 | 5.3300 | 5.3371 | -0.0017 [-0.0030, -0.0004] | -0.0030 [-0.0044, -0.0017] | +0.0013 [+0.0000, +0.0027] | 3826 | 2181 | `cb95d5eeede609f0…` |
| olmo2_13b | c4 | 10.0892 | 10.0773 | 10.0797 | -0.0010 [-0.0016, -0.0003] | -0.0012 [-0.0019, -0.0005] | +0.0002 [-0.0005, +0.0009] | 3826 | 2181 | `cb95d5eeede609f0…` |
| phi4 | wiki | 6.6641 | 6.6152 | 6.6288 | -0.0053 [-0.0066, -0.0041] | -0.0074 [-0.0090, -0.0059] | +0.0021 [+0.0010, +0.0031] | 4201 | 2184 | `39214dc272ac9671…` |
| phi4 | c4 | 10.5485 | 10.5000 | 10.5073 | -0.0039 [-0.0047, -0.0031] | -0.0046 [-0.0055, -0.0038] | +0.0007 [+0.0000, +0.0014] | 4201 | 2184 | `39214dc272ac9671…` |

Primary endpoints (one-sided non-inferiority, margin log(1.005), Holm across 6):

| model | dataset | estimate | 95% CI | p (NI) | Holm p | non-inferior (Holm) |
|---|---|---:|---|---:|---:|---|
| mistral7b | wiki | -0.0039 | [-0.0049, -0.0030] | 0 | 0 | True |
| mistral7b | c4 | -0.0025 | [-0.0036, -0.0015] | 0 | 0 | True |
| phi4 | wiki | -0.0053 | [-0.0066, -0.0041] | 0 | 0 | True |
| phi4 | c4 | -0.0039 | [-0.0047, -0.0031] | 0 | 0 | True |
| olmo2_13b | wiki | -0.0017 | [-0.0030, -0.0004] | 0 | 0 | True |
| olmo2_13b | c4 | -0.0010 | [-0.0016, -0.0003] | 0 | 0 | True |

## 3. Development panel accuracy (exact maps, 8 tasks, paired example bootstrap)

Source: `runs/V90_analyze_accuracy_attempt11/analysis_accuracy/LEGACY_PANEL_ACCURACY.json` (sha256 `db6bb68eb50f690f…`)

| model | contrast | macro diff [95% CI] (pp) | tasks CI<0 | tasks CI>0 |
|---|---|---|---|---|
| llama8b | four_over_six-bf16 | -1.65 [-2.14, -1.17] | arc_easy, arc_challenge, hellaswag, boolq, winogrande, piqa, mmlu | – |
| llama8b | n16_k3-bf16 | -1.76 [-2.24, -1.29] | arc_easy, hellaswag, boolq, piqa, mmlu | – |
| llama8b | n16_k3-four_over_six | -0.11 [-0.60, +0.37] | – | – |
| llama8b | n16_k3-n8_k3 | +0.12 [-0.36, +0.60] | – | – |
| llama8b | n8_k3-four_over_six | -0.22 [-0.70, +0.26] | – | – |
| qwen27b | four_over_six-bf16 | -1.35 [-1.80, -0.91] | hellaswag, boolq, piqa, mmlu | – |
| qwen27b | n16_k3-bf16 | -0.73 [-1.15, -0.32] | hellaswag, boolq, piqa, mmlu | arc_easy |
| qwen27b | n16_k3-four_over_six | +0.61 [+0.19, +1.04] | – | – |
| qwen27b | n16_k3-n8_k3 | +0.23 [-0.18, +0.63] | – | – |
| qwen27b | n8_k3-four_over_six | +0.38 [-0.03, +0.81] | – | mmlu |
| qwen4b | four_over_six-bf16 | -2.70 [-3.26, -2.14] | arc_easy, arc_challenge, hellaswag, boolq, winogrande, piqa, mmlu | – |
| qwen4b | n16_k3-bf16 | -2.05 [-2.59, -1.51] | arc_easy, arc_challenge, hellaswag, boolq, piqa, mmlu | – |
| qwen4b | n16_k3-four_over_six | +0.65 [+0.10, +1.19] | – | hellaswag |
| qwen4b | n16_k3-n8_k3 | -0.28 [-0.83, +0.28] | – | – |
| qwen4b | n8_k3-four_over_six | +0.93 [+0.36, +1.50] | – | arc_challenge, hellaswag, boolq |

## 3. Confirmatory panel accuracy (exact maps, 8 tasks, paired example bootstrap)

Source: `runs/V90_analyze_accuracy_attempt11/analysis_accuracy/CONFIRMATORY_ACCURACY.json` (sha256 `82fc2d241685a192…`)

| model | contrast | macro diff [95% CI] (pp) | tasks CI<0 | tasks CI>0 |
|---|---|---|---|---|
| mistral7b | four_over_six-bf16 | -1.42 [-1.84, -1.01] | arc_easy, arc_challenge, boolq, mmlu | – |
| mistral7b | n16_k3-bf16 | -1.19 [-1.59, -0.79] | arc_easy, arc_challenge, boolq, mmlu | – |
| mistral7b | n16_k3-four_over_six | +0.24 [-0.17, +0.64] | – | – |
| mistral7b | n16_k3-n8_k3 | +0.10 [-0.29, +0.49] | – | – |
| mistral7b | n8_k3-four_over_six | +0.14 [-0.28, +0.55] | – | – |
| olmo2_13b | four_over_six-bf16 | -0.98 [-1.40, -0.56] | hellaswag, boolq, winogrande, mmlu | – |
| olmo2_13b | n16_k3-bf16 | -0.90 [-1.32, -0.47] | arc_challenge, hellaswag, mmlu | – |
| olmo2_13b | n16_k3-four_over_six | +0.08 [-0.33, +0.52] | – | – |
| olmo2_13b | n16_k3-n8_k3 | -0.12 [-0.53, +0.30] | piqa | – |
| olmo2_13b | n8_k3-four_over_six | +0.20 [-0.18, +0.59] | – | – |
| phi4 | four_over_six-bf16 | -0.92 [-1.34, -0.48] | arc_easy, hellaswag, boolq, winogrande, mmlu | – |
| phi4 | n16_k3-bf16 | -0.60 [-1.03, -0.18] | arc_easy, hellaswag, boolq, mmlu | – |
| phi4 | n16_k3-four_over_six | +0.31 [-0.11, +0.74] | – | – |
| phi4 | n16_k3-n8_k3 | -0.05 [-0.47, +0.37] | – | – |
| phi4 | n8_k3-four_over_six | +0.36 [-0.07, +0.78] | – | winogrande |

## 4. Retained fraction and gate classification

Frozen gates: `{"minimum_pass": "all 6 primary endpoints have upper 95% CI < margin AND each confirmatory model: 8-task macro accuracy difference lower 95% CI > -0.5 pp AND fewer than 3 tasks with a paired-bootstrap CI entirely below 0", "strong_pass_quality_only": "minimum_pass AND R >= 0.5 AND pooled N8 k3 dlogppl < 0 (overhead component out of scope)", "investigate": "pooled mean of primary endpoints < 0 but minimum_pass fails", "stop_redirect": "R <= 0 on at least 2 of 3 confirmatory models, or pooled primary upper 95% CI >= margin"}`

**Classification: strong pass (quality component only; overhead out of scope).**

```
{
 "all_primary_upper_ci_below_margin": true,
 "holm_noninferior": [
  true,
  true,
  true,
  true,
  true,
  true
 ],
 "pooled_mean_primary_dlogppl": -0.0030391535282211796,
 "accuracy": {
  "mistral7b": {
   "macro_diff": 0.002378833644065171,
   "macro_ci95": [
    -0.0016544477843997874,
    0.00643733462379693
   ],
   "tasks_ci_below_zero": [],
   "ok": true
  },
  "phi4": {
   "macro_diff": 0.0031374779816266732,
   "macro_ci95": [
    -0.0010908788620540358,
    0.007401267275546142
   ],
   "tasks_ci_below_zero": [],
   "ok": true
  },
  "olmo2_13b": {
   "macro_diff": 0.0008462983927443098,
   "macro_ci95": [
    -0.0033268592763874803,
    0.005213335231969365
   ],
   "tasks_ci_below_zero": [],
   "ok": true
  }
 },
 "retained_fraction_by_model": {
  "mistral7b": 0.8864139252889791,
  "olmo2_13b": 0.6265611732710377,
  "phi4": 0.770065282347019
 },
 "pooled_retained_fraction": 0.7798406578406878,
 "pooled_n8_dlogppl_sum": -0.023382880830832824
}
```

## 5. Evidence that qualifies the classification above

The classification in section 4 grades the frozen quality criteria and nothing else. The results below were produced
by this same campaign and materially qualify it.

**5a. k=3 is a pre-registered conservative threshold, not a tuned optimum.** Source: `runs/V90_assemble_deliverables_attempt4/deliverables/K_SENSITIVITY.json` (sha256 `1c50e2737a01d2d6…`).
Frozen rule: `k=3 is primary; k in {2,4,5,6} is sensitivity only and may not replace k=3`

| model | corpus | dlogPPL k=2 vs FourOverSix [95% CI] | dlogPPL k=3 vs FourOverSix [95% CI] | k=2 tiles | k=3 tiles |
|---|---|---|---|---:|---:|
| llama8b | wiki | -0.0052 [-0.0073, -0.0029] | -0.0047 [-0.0063, -0.0031] | 50119 | 1781 |
| llama8b | c4 | -0.0054 [-0.0082, -0.0030] | -0.0050 [-0.0066, -0.0036] | 50119 | 1781 |
| mistral7b | wiki | -0.0048 [-0.0057, -0.0039] | -0.0039 [-0.0049, -0.0030] | 30611 | 4179 |
| mistral7b | c4 | -0.0028 [-0.0036, -0.0021] | -0.0025 [-0.0036, -0.0015] | 30611 | 4179 |
| qwen4b | wiki | -0.2395 [-0.2510, -0.2286] | -0.1525 [-0.1619, -0.1445] | 32532 | 4077 |
| qwen4b | c4 | -0.1188 [-0.1236, -0.1138] | -0.0752 [-0.0784, -0.0721] | 32532 | 4077 |

Lower k selects more tiles and gives better perplexity in every model x corpus cell measured, so k=3 must be
justified by the pre-registration and by per-tile reliability, never as an optimum.

Accuracy, macro over the representative task suite against four_over_six (V40 representative run). Source: `runs/V90_analyze_accuracy_attempt11/analysis_accuracy/K_SENSITIVITY_ACCURACY.json` (sha256 `f84a986de592b50d…`).

| model | policy | macro pp | 95% CI (pp) | tasks CI<0 | tasks CI>0 |
|---|---|---:|---|---:|---:|
| llama8b | n16_k2 | +0.48 | [-0.25, +1.23] | 0 | 0 |
| llama8b | n16_k3 | +0.36 | [-0.36, +1.10] | 0 | 0 |
| llama8b | n16_k4 | +0.26 | [-0.50, +1.01] | 0 | 0 |
| llama8b | n16_k5 | +0.20 | [-0.51, +0.93] | 0 | 0 |
| llama8b | n16_k6 | +0.43 | [-0.29, +1.12] | 0 | 1 |
| llama8b | n8_k2 | +0.28 | [-0.47, +1.01] | 0 | 0 |
| llama8b | n8_k4 | +0.23 | [-0.53, +0.97] | 0 | 0 |
| llama8b | n8_k5 | -0.04 | [-0.73, +0.68] | 0 | 0 |
| llama8b | n8_k6 | +0.09 | [-0.64, +0.81] | 0 | 0 |
| mistral7b | n16_k2 | +0.36 | [-0.28, +0.99] | 0 | 1 |
| mistral7b | n16_k3 | +0.05 | [-0.55, +0.72] | 0 | 0 |
| mistral7b | n16_k4 | +0.32 | [-0.33, +0.97] | 1 | 1 |
| mistral7b | n16_k5 | +0.35 | [-0.27, +1.01] | 0 | 0 |
| mistral7b | n16_k6 | +0.03 | [-0.54, +0.63] | 0 | 0 |
| mistral7b | n8_k2 | +0.05 | [-0.60, +0.72] | 0 | 0 |
| mistral7b | n8_k4 | +0.04 | [-0.55, +0.67] | 0 | 0 |
| mistral7b | n8_k5 | +0.26 | [-0.35, +0.88] | 0 | 0 |
| mistral7b | n8_k6 | -0.14 | [-0.76, +0.47] | 0 | 0 |
| qwen4b | n16_k2 | +1.92 | [+0.93, +2.82] | 0 | 2 |
| qwen4b | n16_k3 | +1.30 | [+0.36, +2.20] | 0 | 0 |
| qwen4b | n16_k4 | +0.55 | [-0.36, +1.47] | 0 | 0 |
| qwen4b | n16_k5 | +0.02 | [-0.92, +1.00] | 0 | 0 |
| qwen4b | n16_k6 | +0.19 | [-0.74, +1.10] | 0 | 0 |
| qwen4b | n8_k2 | +2.01 | [+1.05, +2.90] | 0 | 2 |
| qwen4b | n8_k4 | +0.63 | [-0.31, +1.58] | 0 | 1 |
| qwen4b | n8_k5 | +0.10 | [-0.81, +1.01] | 0 | 0 |
| qwen4b | n8_k6 | +0.45 | [-0.54, +1.36] | 0 | 0 |

3 of 27 model x policy cells have a macro accuracy CI excluding zero (qwen4b n16_k2, qwen4b n16_k3, qwen4b n8_k2).

**5b. A published prior-art format outperforms N16K64 on most models measured.** Source: `runs/V90_assemble_deliverables_attempt4/deliverables/ADDITIONAL_BASELINES.json` (sha256 `9a6686b6c7997de3…`).

| model | corpus | RaZeR minus N16 k3 [95% CI] | nover6 minus N16 k3 [95% CI] |
|---|---|---|---|
| llama8b | wiki | -0.0122 [-0.0143, -0.0101] | +0.0016 [-0.0003, +0.0033] |
| llama8b | c4 | -0.0110 [-0.0146, -0.0081] | +0.0004 [-0.0017, +0.0023] |
| mistral7b | wiki | -0.0012 [-0.0023, -0.0001] | +0.0034 [+0.0023, +0.0044] |
| mistral7b | c4 | -0.0024 [-0.0033, -0.0015] | +0.0010 [+0.0002, +0.0019] |
| qwen4b | wiki | +0.1447 [+0.1375, +0.1525] | +0.1279 [+0.1202, +0.1358] |
| qwen4b | c4 | +0.0749 [+0.0714, +0.0784] | +0.0722 [+0.0691, +0.0754] |

A negative entry means the baseline is **better** than N16 k3. RaZeR reaches its numbers with a different element
format (e3m3 weights, e4m3 activations) and therefore a different datapath, so this is not a matched-hardware
comparison - but it must be shown rather than omitted, and no state-of-the-art quality claim is available from this
campaign.

Accuracy, macro over the representative task suite against four_over_six (V40 representative run). Source: `runs/V90_analyze_accuracy_attempt11/analysis_accuracy/ADDITIONAL_BASELINES_ACCURACY.json` (sha256 `7fa6c8bf0c467829…`).

| model | policy | macro pp | 95% CI (pp) | tasks CI<0 | tasks CI>0 |
|---|---|---:|---|---:|---:|
| llama8b | n16_k3 | +0.36 | [-0.36, +1.10] | 0 | 0 |
| llama8b | nover6_native_rows | +0.15 | [-0.63, +0.93] | 0 | 0 |
| llama8b | nover6_wonly_shared_act | +0.49 | [-0.28, +1.28] | 1 | 0 |
| llama8b | razer_native_rows | +0.90 | [+0.07, +1.71] | 0 | 0 |
| llama8b | razer_wonly_shared_act | +0.16 | [-0.64, +0.96] | 0 | 0 |
| mistral7b | n16_k3 | +0.05 | [-0.55, +0.72] | 0 | 0 |
| mistral7b | nover6_native_rows | +0.27 | [-0.37, +0.93] | 0 | 0 |
| mistral7b | nover6_wonly_shared_act | +0.22 | [-0.37, +0.87] | 0 | 0 |
| mistral7b | razer_native_rows | +0.33 | [-0.32, +1.01] | 0 | 0 |
| mistral7b | razer_wonly_shared_act | +0.59 | [-0.11, +1.29] | 0 | 0 |
| qwen4b | n16_k3 | +1.30 | [+0.36, +2.20] | 0 | 0 |
| qwen4b | nover6_native_rows | +0.61 | [-0.40, +1.56] | 0 | 0 |
| qwen4b | nover6_wonly_shared_act | +0.54 | [-0.48, +1.53] | 0 | 0 |
| qwen4b | razer_native_rows | -0.04 | [-1.10, +0.92] | 0 | 0 |
| qwen4b | razer_wonly_shared_act | +0.71 | [-0.24, +1.65] | 0 | 1 |

2 of 15 model x policy cells have a macro accuracy CI excluding zero (llama8b razer_native_rows, qwen4b n16_k3).

**5c. The first-order score does not predict individual tile effects.** Source: `runs/V90_analyze_misc_attempt7/analysis_misc/FIRST_ORDER_FIDELITY.json` (sha256 `3e28f8f8fd387201…`).

| model | type block | Spearman(pred, actual) CE / KL | selected-stratum FPR | full-map actual/predicted CE |
|---|---|---|---:|---:|
| llama8b | n8 | +0.17 / +0.01 | 0.650 | 0.36 |
| llama8b | n16 | +0.04 / +0.03 | 0.725 | 0.74 |
| mistral7b | n8 | -0.07 / -0.16 | 0.500 | 0.63 |
| mistral7b | n16 | -0.13 / -0.20 | 0.675 | 0.79 |
| qwen4b | n8 | +0.18 / -0.02 | 0.675 | 0.58 |
| qwen4b | n16 | +0.24 / -0.02 | 0.700 | 0.66 |

The aligned evaluation is bitwise deterministic (V43 noise controls: 24 identical baseline evaluations per model on
three different A6000 cards, end-of-run drift exactly 0.0, with two independent cross-card reproductions of the same
measured tile effect), so a single-tile measurement carries no measurement error whatsoever. Under replication the
predicted CE sign was correct in only **3 of the 9** tiles re-measured (three per model; KL 7 of 9), and the tile the
k=3 rule *rejects* bears no relation to its prediction on any model - helping on Llama-3.1-8B (-8.24e-4) and Mistral-7B
(-5.31e-4), hurting on Qwen3-4B (+1.12e-3), always by two to three orders of magnitude more than predicted. Decisively,
the model carrying the highest single-tile rank correlation of all six cells (Qwen3-4B, Spearman +0.239) still
mispredicts its median and rejected tiles by -146x and -1,657x, so this is not an artifact of the models where the score
looks worst. The reproducible object is the aggregate effect of a hash-verified map, never an individual tile selection,
and no per-tile or per-layer claim is supported.

**5d. The archived multiplicity argument does not hold.** Source: `runs/V90_analyze_selection_attempt2/analysis_selection/SELECTION_STATISTICS.json` (sha256 `255cea8aa73c44e4…`).

| model | median per-tile CE/KL correlation | tiles at k=3 | BH q=0.05 retains | sign-flip FDP |
|---|---:|---:|---:|---:|
| llama8b | +0.65 | 1781 | 203 | 0.317 |
| mistral7b | +0.32 | 4179 | 1545 | 0.027 |
| olmo2_13b | +0.33 | 2181 | 20 | 0.111 |
| phi4 | +0.32 | 2184 | 12 | 0.102 |
| qwen27b | +0.43 | 2168 | 108 | 0.206 |
| qwen4b | +0.45 | 4077 | 705 | 0.031 |

CE and KL scores are strongly positively correlated per tile, so the archived independence-based bound is invalid.
The defensible statement is the empirical sign-flip false-discovery proportion above.

## 6. Overhead

Only the user-supplied estimates exist (N8K64 ~13%, N16K64 ~1.5%; `KNOWN_RESULTS.json`). They were not measured in this campaign and
cannot be measured on RTX A6000 / RTX 6000 Ada; any quality-overhead Pareto statement must label them as unverified external estimates.

