# N16K64 claim-to-evidence matrix

This matrix distinguishes what the evidence supports from what it merely
suggests. `delta-log-PPL` is a change in mean token negative log-likelihood
(NLL); relative PPL change is `exp(delta_NLL) - 1`. A confidence interval and a
Holm-adjusted hypothesis-test decision are reported separately.

## Frozen method and data lineage

The primary method uses N16K64 weight type tiles, K16 UE4M3 block scales,
FourOverSix E2M1 as the weight baseline, E0M3 with alpha 1 as the alternative,
and causal per-token FourOverSix activations. Non-head linear weights are in
scope; embeddings, norms, `lm_head`, and KV cache are not. At the all-FourOverSix
student reference, a tile is selected when both
`mean_CE + 3*SE_CE < 0` and `mean_KL + 3*SE_KL < 0`. The calibration is seed 0,
64 OpenWebMath plus 64 CodeParrot sequences of 512 tokens. C4 and WikiText are
evaluation data, not selector calibration data.

Model and tokenizer revisions are frozen together:

| Model | Revision |
|---|---|
| Llama-3.1-8B | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Mistral-7B-v0.3 | `caa1feb0e54d415e2df31207e5f4e273e33509b1` |
| OLMo-2-1124-13B | `3fefddc1bf18a30e1d9b91000271630718f2aa8b` |
| Phi-4 | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |

Authoritative definition: [`PROTOCOL_FREEZE.json`](campaigns/primary/PROTOCOL_FREEZE.json),
whose original SHA-256 is
`df78f1fbbd034cb8d2bc8e6f6a264e7e221e1b5f2b2ff99a3fe821369c5ac88f`.

## Claims

| Claim or question | Classification | Evidence | Essential qualification |
|---|---|---|---|
| Frozen N16K64 k=3 improves model-level PPL over FourOverSix on the confirmatory panel. | **Supported for fake-quantized quality.** All six Mistral/OLMo2/Phi-4 endpoints are negative and pass the frozen Holm non-inferiority gate. Estimates range from -0.0010 to -0.0053 delta-NLL. | [`N16_DECISION.md`](campaigns/primary/N16_DECISION.md), [`CONFIRMATORY_PPL.json`](campaigns/primary/analysis_ppl/CONFIRMATORY_PPL.json), [`decision_gate.json`](campaigns/primary/decision_gate.json) | Quality-only result. It is not native execution or performance evidence. |
| N16K64 retains part of the measured N8 log-PPL gain. | **Observed quality trade-off; not N16 superiority.** Ratio of summed N16 versus N8 delta-NLL gains is 0.7798406578 over Mistral/OLMo2/Phi-4 × C4/Wiki. N8 has the better point estimate in all six cells. | [`N16_DECISION.md`](campaigns/primary/N16_DECISION.md), [`CURRENT_EVIDENCE_AUDIT.json`](validation/CURRENT_EVIDENCE_AUDIT.json) | The denominator is summed N8-minus-FourOverSix delta NLL, not selected count. It is a point ratio, not a confidence bound or universal preservation claim. |
| N16K64 preserves eight-task macro accuracy on the confirmatory panel under the frozen margin. | **Frozen gate passed.** Mistral +0.24 pp CI [-0.17,+0.64], OLMo2 +0.08 [-0.33,+0.52], Phi-4 +0.31 [-0.11,+0.74] versus FourOverSix. | [`CONFIRMATORY_ACCURACY.json`](campaigns/primary/analysis_accuracy/CONFIRMATORY_ACCURACY.json), [`N16_DECISION.md`](campaigns/primary/N16_DECISION.md) | The pointwise intervals include zero. This is a predeclared non-inferiority gate, not proof that accuracy improves. Llama's development CI [-0.60,+0.37] does not establish the same -0.5 pp margin. |
| k=3 is the empirically best threshold. | **Unsupported.** | [`K_SENSITIVITY_PPL.json`](campaigns/primary/analysis_ppl/K_SENSITIVITY_PPL.json), [`N16_DECISION.md`](campaigns/primary/N16_DECISION.md) | k=3 was preregistered as a conservative primary. Lower k improved PPL in every measured k=2 versus k=3 cell; it cannot be relabeled as the tuned optimum. |
| k=2 broadly improves PPL in the extension. | **Supported post hoc.** | [`CLAIM_REGISTER.json`](campaigns/ppl_improvement/CLAIM_REGISTER.json), [`FINAL_PPL_TABLE.csv`](campaigns/ppl_improvement/FINAL_PPL_TABLE.csv) | k=2 was selected using earlier results. The extension is a post-hoc robustness study, not a new confirmatory primary. Material k=2 superiority over k=3 was classified practically equivalent/unsupported. |
| The k=2 gain is selector-specific rather than only selected-weight density. | **Supported under the frozen global and per-module matched-density controls.** | [`MATCHED_BUDGET_CONTROLS.csv`](campaigns/ppl_improvement/MATCHED_BUDGET_CONTROLS.csv), claim C03 in [`CLAIM_REGISTER.json`](campaigns/ppl_improvement/CLAIM_REGISTER.json) | Does not validate individual tiles or native deployment value. |
| The PPL extension establishes a replacement selector, held-out-family generalization, downstream non-inferiority, SOTA, or A6000/Ada portability. | **Unsupported or stopped by gate.** | Claims C05-C09 in [`CLAIM_REGISTER.json`](campaigns/ppl_improvement/CLAIM_REGISTER.json), [`GATE_REGISTER.json`](campaigns/ppl_improvement/GATE_REGISTER.json), [`FINAL_SUBMISSION_RISK_AUDIT.md`](campaigns/ppl_improvement/FINAL_SUBMISSION_RISK_AUDIT.md) | Accuracy/GSM8K/PG19 were stopped by the frozen gate. Portability and broad held-out claims failed. Negative and invalid attempts are retained. |
| Aggregate score ranking can be useful even when individual-tile finite-effect signs are unreliable. | **Supported at group/model level.** Strongest-versus-weakest and later eight-bin dose response pass their frozen ranking gates. | [`MECHANISM_VERDICT.md`](campaigns/mechanism/MECHANISM_VERDICT.md), [`FOLLOWUP_VERDICT.md`](campaigns/followup/FOLLOWUP_VERDICT.md), [`SCORE_MARGIN_ANALYSIS.json`](campaigns/mechanism/SCORE_MARGIN_ANALYSIS.json), [`DOSE_RESPONSE_RESULTS.csv`](campaigns/followup/DOSE_RESPONSE_RESULTS.csv) | This is ranking evidence, not individual-tile causal calibration. Primary finite-effect sign precision/recall and rank correlations remain poor. |
| The CE+KL conjunction universally reduces veto tail risk. | **Not supported under the strict mechanism gate.** One veto class passed and the other failed; the symmetric Mistral completion also failed. | [`CE_KL_VETO_ANALYSIS.json`](campaigns/mechanism/CE_KL_VETO_ANALYSIS.json), [`MISTRAL_VETO_COMPLETION.json`](campaigns/followup/MISTRAL_VETO_COMPLETION.json), both verdicts | CE and KL are dependent. No independence or simultaneous-confidence interpretation is available. |
| The full-map effect is provably nonadditive across attention and MLP. | **Not supported under the strict gate.** | [`MODULE_INTERACTION_ANALYSIS.json`](campaigns/mechanism/MODULE_INTERACTION_ANALYSIS.json), [`MECHANISM_VERDICT.md`](campaigns/mechanism/MECHANISM_VERDICT.md) | Some model/corpus residuals are nonzero, but the preregistered cross-endpoint condition did not pass. |
| The conjunction is universally better than CE-only and KL-only at matched k=3 budgets. | **No universal conjunction superiority.** All three informed maps beat matched random in the four development endpoints, but pairwise conjunction gates fail. | [`OBJECTIVE_ABLATION_RESULTS.json`](campaigns/followup/OBJECTIVE_ABLATION_RESULTS.json), [`FOLLOWUP_VERDICT.md`](campaigns/followup/FOLLOWUP_VERDICT.md) | Natural-threshold maps have different tile counts and are descriptive. Matched-K and natural-threshold interpretations must not be mixed. |
| Critical-k bands and full-map corruption support aggregate score/risk information. | **Power-limited support.** Both frozen pattern gates pass; final classification remains `power_limited_support`. | [`BOUNDARY_CORRUPTION_VERDICT.md`](campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md), [`BOUNDARY_RESULTS.csv`](campaigns/boundary/BOUNDARY_RESULTS.csv), [`CORRUPTION_RESULTS.csv`](campaigns/boundary/CORRUPTION_RESULTS.csv), [`POWER_ANALYSIS.json`](campaigns/boundary/POWER_ANALYSIS.json) | Qwen power status is endpoint-specific, not globally “descriptive.” Mistral remains one-shot analysis validation. CI and Holm conclusions remain separate. |
| There is a consistent local causal jump at kappa=3. | **Unsupported.** | Boundary contrasts in [`PRIMARY_RESULTS_TABLES.md`](campaigns/boundary/PRIMARY_RESULTS_TABLES.md) | The weakest-selected versus nearest-rejected contrast reverses for Llama and is heterogeneous. Kappa is a ranking/confidence variable, not calibrated finite effect. |
| Replacing all selected tiles with near-boundary rejected tiles degrades NLL. | **Power-limited support as an imposed stress test.** p=1 minus p=0 is +0.003012/+0.003772 for Llama, +0.053230/+0.110018 for Qwen, and +0.002009/+0.002895 for Mistral on C4/Wiki. | [`CORRUPTION_RESULTS.json`](campaigns/boundary/CORRUPTION_RESULTS.json), [`BOUNDARY_CORRUPTION_VERDICT.md`](campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md) | Corruption `p` is imposed, not an estimate of the selector's real error rate. p=1 minus p=0 is not the selector-versus-FourOverSix benefit. |
| Standardized pooled boundary/corruption sensitivities are raw NLL effect sizes, accuracy, or explained variance. | **False interpretation.** | Pooled section in [`BOUNDARY_CORRUPTION_VERDICT.md`](campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md) | All-model / leave-Qwen values (-0.876/-0.822 boundary; +0.959/+0.940 corruption) are standardized sensitivities only. Qwen remains in primary per-model tables. |
| The campaigns establish reliable individual-tile causal signs, magnitudes, or finite-effect calibration. | **Unsupported.** | [`SCORE_MARGIN_ANALYSIS.json`](campaigns/mechanism/SCORE_MARGIN_ANALYSIS.json), primary risk audit and all later verdicts | Tiles/tokens/maps are not independent experimental replications. Group ranking does not repair the individual-tile counterevidence. |
| The N16K64 campaigns establish native FP4/E0M3 execution or performance. | **Out of scope and prohibited.** | Every campaign protocol/risk report | They use BF16-dequantized software/fake quantization. No Tensor Core execution, latency, speedup, N8/N16 overhead, area, power, or Blackwell-performance claim follows. |

## Boundary/corruption construction checks and deviations

The outcome-blind coverage gate selected four bands because the six- and
eight-band candidates failed Llama coverage. Retained coverage/tile count per
band is 0.830994/370 for Llama, 0.913417/931 for Qwen, and 0.936109/978 for
Mistral. Four corruption pools per model are disjoint and all four p=1 hashes
are distinct. The exact kappa-greater-than-3 anchor matches every stored primary
map tile-for-tile. See [`COVERAGE_GATE.json`](campaigns/boundary/COVERAGE_GATE.json),
[`CORRUPTION_UNIQUENESS_GATE.json`](campaigns/boundary/CORRUPTION_UNIQUENESS_GATE.json),
and [`CRITICAL_K_SUMMARY.json`](campaigns/boundary/CRITICAL_K_SUMMARY.json).

The campaign completed all 168 required evaluations but took about 61.4 hours.
It missed both the hour-20 launch cutoff and the 24-hour target. Late launches
were frozen retries following invalid co-tenancy/monitoring attempts, not
outcome-selected arms. The hour-20 requirement is an explicit stop rule. Late
Llama/Qwen arrays are protocol-deviating required-retry evidence, with unresolved
fully protocol-conformant admissibility. They cannot be called time-conformant
or upgraded to confirmatory evidence. See [`PROTOCOL_DEVIATIONS.json`](campaigns/boundary/PROTOCOL_DEVIATIONS.json)
and [`ANALYSIS_CORRECTIONS.json`](campaigns/boundary/ANALYSIS_CORRECTIONS.json).

## Secondary completeness and review corrections

The five-draw N16 conjunction has 30/30 favorable PPL point estimates across
Llama/Qwen/Mistral × five draws × two corpora. Its four-task macro accuracy is
not uniformly favorable: Mistral spans -0.149 to +0.729 percentage points.
No population-tail or universal accuracy-preservation claim follows. The eight-task
confirmatory margin applies only to Mistral/OLMo2/Phi-4 and the frozen seed0 panel.
V50 secondary PPL output uses B=2000 and includes historical zero p-values; this
review does not relabel them as finite plus-one tests or multiplicity-controlled
confirmation. Exact draw hashes and observed ranges are in the current audit.

Primary GSM8K and PG19 results exist, while extension downstream promotion was
stopped. The completed near-versus-random p=.50 corruption comparison is a
predeclared secondary control. Neither those controls nor new TODO proposals
validate reorder efficacy, tile-level causality, or a broad connected optimum.
Corrections affect review interpretation only; sealed reports are untouched.

## Latest-main research outside this snapshot

The current `main` also contains the later 256x64 reordering and native
diagnostic work. The accepted Llama/Qwen reordering results, failed 90% recovery
target, incomplete native quality equivalence, and scope-specific timing claims
remain governed by [`MIXFP4_REPORT.md`](../../MIXFP4_REPORT.md),
[`REORDERING_STUDY_STATUS.md`](../../REORDERING_STUDY_STATUS.md), and
[`TASK_REORDER_HANDOFF.md`](../../TASK_REORDER_HANDOFF.md). This N16K64 review
does not overwrite or broaden those distinct claims.

## Homepage execution and supplementary claims (2026-09-21)

The [homepage](../../README.md) now leads with MixFP4 rather than the upstream
RaZeR guide. Its quality tables are rounded from the primary JSON, not from a
native experiment. The 77.98% figure is the ratio of summed six-endpoint ΔNLL
gains; five-draw favorable PPL does not imply all accuracy endpoints improve.

| Claim | Evidence | Status / boundary |
|---|---|---|
| Primary map affects actual quality forward | [Installer](software/primary/campaign/policies.py), [activation hooks](software/primary/campaign/quant.py) | Map selects BF16 dequantized weights; activation quantize–dequantize precedes floating-point Linear. Not native packed-FP4 MMA. |
| A separate native mixed prototype exists | [Native runtime](../../native/model_runtime.cu), [build override](../../scripts/build_native_model_runtime.py), [operator report](../../results/task_reorder/full_model_20260920/kernel_406633/report.json) | Experimental GB200/SM100 path, N256K64 ownership and global activation amax. Recorded checks, not rerun in this review. |
| Prototype establishes native model quality / primary N16 parity | [Full-model report](../../results/task_reorder/full_model_20260920/llama_406828/report.json), [implementation](../../results/task_reorder/full_model_20260920/IMPLEMENTATION.md) | **Not established**: full-output gate false, native accuracy false, native PPL unmeasured. |
| E0M3 descriptor value 0 is officially portable | [PTX ISA 8.8 Table 45](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#tcgen05-instruction-descriptor) | **Not established**: the relevant documented format is E2M1=1. Distinguish saved experimental behavior from official guarantees; absence of documentation is not proof of impossibility. |
| Later reorder diagnoses or accuracy validate N16 | [Repository-wide index](README.md#repository-wide-research-context) | **Not supported**: different ownership/layout and protocols; Llama accuracy is inconclusive, Qwen partial/cancelled. |

The current overview's power/timing qualifications govern interpretation of the
sealed boundary statistical report; its stored label is not permission to call
late launches strictly protocol-conformant confirmation. Historical reports
remain unchanged and are not duplicated as new evidence.
