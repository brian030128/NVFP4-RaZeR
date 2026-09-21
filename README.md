# MixFP4: task-aware format selection at coarse granularity

MixFP4 asks whether a small number of task-selected weight-format changes can
improve low-precision language-model quality without requiring a different format
for every scale block. The frozen primary method keeps FourOverSix-E2M1 as the
baseline and elects E0M3 at **N16K64** granularity using joint cross-entropy (CE)
and teacher-KL directional scores.

**Execution boundary:** the primary PPL and accuracy results below use
fake-quantized/dequantized BF16 weights and activations followed by floating-point
linear operations—not native mixed-format FP4 MMA. A separate Blackwell native
prototype exists, but uses a different configuration and has not established
full-output equivalence or native model quality.

[Detailed research overview](research/n16k64/README.md) ·
[Claims and evidence](research/n16k64/CLAIM_EVIDENCE_MATRIX.md) ·
[Software entry points](research/n16k64/software/README.md) ·
[Next experiments](research/n16k64/TODO_EXPERIMENTS.md)

## In one minute

- **Quality:** N16 has lower PPL point estimates than FourOverSix in all 12
  model/corpus cells below. The largest response is Qwen3-4B, a development
  model; the separate Mistral/OLMo2/Phi-4 confirmation panel has smaller gains.
- **Granularity:** across six confirmation endpoints, N16 retains **77.98%**
  of N8's summed log-PPL improvement over FourOverSix. N8 has the better PPL
  point estimate in those six cells. This is not an accuracy-retention ratio.
- **Downstream:** the three confirmation models pass the frozen **−0.5 percentage
  point** eight-task macro-accuracy non-inferiority gate. Their improvement CIs
  include zero; Llama's development interval does not establish the same margin.
- **Mechanism:** composition-matched group ranking and eight-bin dose response
  are supported; individual-tile signs/magnitudes remain poorly calibrated.
  Universal CE+KL superiority and the strict interaction/veto claims are not supported.
- **Boundary stress tests are complete:** 171 maps and 168 new policy evaluations.
  Both pattern gates pass, but the formal verdict remains
  **`power_limited_support`**, with an unresolved timing-compliance qualification.

These are scoped quality findings, not claims of speedup, memory savings, power
reduction, native N16 execution, or superiority to every quantization baseline.

## Motivation: scale precision is not format ownership

A scale can remain local while format ownership becomes coarse. Here a **1×16
scale block** contains 16 consecutive input-channel weights in one output row.
An **N16K64 format block** covers 16 output rows × 64 input columns: 1,024 weights
and 64 such scale blocks share one format decision. N8K64 halves the output-row
extent, not the scale-block size.

FourOverSix improves E2M1 scale fitting by comparing scale candidates associated
with peaks four and six. MixFP4 adds a different choice: retain that fitted E2M1
block or use E0M3. Local reconstruction improvement alone need not improve
end-to-end loss. The relevant test is therefore a frozen whole-map evaluation,
not just weight reconstruction error. Comparing N16 with FourOverSix isolates
the added format election more directly than comparing it with plain NVFP4.

## Frozen primary method

1. Load the pinned model/tokenizer and quantize non-head Linear weights with
   FourOverSix-E2M1. Embeddings, norms, the vocabulary head and KV cache are
   outside weight-format election.
2. Build the E0M3 alpha=1 candidate. The unscaled E2M1 magnitudes are
   `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`; E0M3 uses `{0, 1, 2, 3, 4, 5, 6, 7}`,
   both with a sign bit. K16 blocks use UE4M3 scales.
3. At the all-FourOverSix student reference, compute each sequence's
   gradient–candidate-direction inner product for CE and teacher KL. Primary
   seed0 calibration is **64 OpenWebMath + 64 CodeParrot sequences, 512 tokens
   each**. Activations use causal **per-token** FourOverSix.
4. Compute objective-wise `U = mean + 3 × SE` across sequences and select a
   block only when **both** upper scores are negative. Where N8 child scores
   form N16 scores, sum children **within each sequence before computing SE**:
   their covariance is retained, rather than adding independent variances.
   CE/KL covariance is recorded; the conjunction does not assume independence.
5. Freeze the map and evaluate its full model. Use CE means for predicted NLL
   changes; KL is a veto signal, not another NLL prediction.

The map is a bit-packed **format mask**, not packed FP4 weights or an executable
kernel. Primary election has no reorder step. The threshold `k=3` is a frozen
operating rule, not a simultaneous-confidence guarantee or a universal optimum.
The boundary analysis uses `κ = min(−mean_CE/SE_CE, −mean_KL/SE_KL)`;
larger κ favors selection. κ, k and the upper-score margin are different quantities.
See the [implementation](research/n16k64/software/primary/campaign/tiles.py),
[protocols and zero-SE rules](research/n16k64/README.md#latest-evidence-and-remaining-work),
and [statistical qualifications](research/n16k64/campaigns/primary/STATISTICAL_VALIDITY_REPORT.md).

## Main quality results

### Perplexity: a common evaluation protocol

Lower is better. All four columns come from the primary campaign's matched
evaluation panel; N8 and N16 use the frozen CE+KL k=3 rule. NVFP4 and FourOverSix
also differ in scale fitting/activation policy, so the NVFP4-to-N16 difference
must not be attributed solely to format selection.

| Role | Model | Corpus | NVFP4 | FourOverSix | N8K64 | N16K64 |
|---|---|---|---:|---:|---:|---:|
| Development | Llama-3.1-8B | WikiText | 6.9317 | 6.8754 | 6.8414 | 6.8432 |
| Development | Llama-3.1-8B | C4 | 9.9379 | 9.8254 | 9.7763 | 9.7762 |
| Development | Qwen3-4B | WikiText | 13.9557 | 14.2062 | 11.8841 | 12.1968 |
| Development | Qwen3-4B | C4 | 17.2621 | 17.3024 | 15.8483 | 16.0485 |
| Development | Qwen3-27B | WikiText | 7.5769 | 7.2839 | 7.2254 | 7.2456 |
| Development | Qwen3-27B | C4 | 10.2237 | 10.1894 | 10.1491 | 10.1564 |
| Confirmation | Mistral-7B-v0.3 | WikiText | 5.5487 | 5.5233 | 5.4987 | 5.5019 |
| Confirmation | Mistral-7B-v0.3 | C4 | 8.0938 | 8.0674 | 8.0456 | 8.0475 |
| Confirmation | OLMo-2-1124-13B | WikiText | 5.4027 | 5.3462 | 5.3300 | 5.3371 |
| Confirmation | OLMo-2-1124-13B | C4 | 10.1226 | 10.0892 | 10.0773 | 10.0797 |
| Confirmation | Phi-4 | WikiText | 6.7013 | 6.6641 | 6.6152 | 6.6288 |
| Confirmation | Phi-4 | C4 | 10.5838 | 10.5485 | 10.5000 | 10.5073 |

Sources: [development PPL JSON](research/n16k64/campaigns/primary/analysis_ppl/LEGACY_PANEL_PPL.json),
[confirmation PPL JSON](research/n16k64/campaigns/primary/analysis_ppl/CONFIRMATORY_PPL.json).
Unrounded paired ΔNLL estimates and CIs are in these files and the
[research overview](research/n16k64/README.md#primary-quality-results).

Δlog-PPL means **ΔNLL**, not absolute ΔPPL; relative PPL change is
`exp(ΔNLL) − 1`. All six confirmation PPL non-inferiority gates passed after
the frozen Holm correction, using a margin of `log(1.005)`. This does not prove
N16 superior to N8. The six pointwise N16-minus-FourOverSix intervals exclude zero;
pointwise intervals and adjusted hypothesis tests remain distinct.

### Eight-task macro accuracy

Unweighted macro over ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, MMLU,
OpenBookQA, PIQA and WinoGrande; values are percentages. Differences and 95% CIs
are in **percentage points** (pp), relative to FourOverSix.

| Role | Model | FourOverSix | N8K64 | N16K64 | N16 − FourOverSix [95% CI], pp |
|---|---|---:|---:|---:|---:|
| Development | Llama-3.1-8B | 68.82 | 68.60 | 68.71 | −0.11 [−0.60, +0.37] |
| Development | Qwen3-4B | 64.26 | 65.19 | 64.91 | +0.65 [+0.10, +1.19] |
| Development | Qwen3-27B | 72.09 | 72.47 | 72.70 | +0.61 [+0.19, +1.04] |
| Confirmation | Mistral-7B-v0.3 | 68.80 | 68.93 | 69.03 | +0.24 [−0.17, +0.64] |
| Confirmation | OLMo-2-1124-13B | 69.96 | 70.16 | 70.04 | +0.08 [−0.33, +0.52] |
| Confirmation | Phi-4 | 71.17 | 71.53 | 71.48 | +0.31 [−0.11, +0.74] |

Sources: [development accuracy](research/n16k64/campaigns/primary/analysis_accuracy/LEGACY_PANEL_ACCURACY.json),
[confirmation accuracy](research/n16k64/campaigns/primary/analysis_accuracy/CONFIRMATORY_ACCURACY.json).
The confirmation **−0.5 pp non-inferiority margin** passes; superiority is not
established. Llama's interval extends below that margin. Macro preservation does
not imply every task improves, nor preservation relative to BF16. A matching
NVFP4 eight-task table is not supplied here.

### N8 → N16 tradeoff

Across **Mistral, OLMo2 and Phi-4 × WikiText/C4**, sum the six N16-minus-FourOverSix
ΔNLL estimates and divide by the corresponding N8 sum:

`(−0.018234921169327074) / (−0.023382880830832824) = 0.7798406578`.

Thus **77.98%** is a summed log-PPL improvement ratio on that panel: not the
mean of six ratios, a selected-tile percentage, accuracy retention, or a
hardware-overhead measurement. Coarser format ownership is the design motivation;
its runtime benefit has not been demonstrated for this primary method.

## Why the selector is useful—and what it does not predict

The [mechanism campaign](research/n16k64/campaigns/mechanism/MECHANISM_VERDICT.md)
supports aggregate ranking despite poor individual-tile finite-effect prediction.
The [eight-bin follow-up](research/n16k64/campaigns/followup/FOLLOWUP_VERDICT.md)
supports group-level dose response, including standardized leave-Qwen-out
sensitivity. Neither establishes reliable tile-level signs or magnitude calibration.
Strict CE/KL-veto, attention/MLP-interaction and Mistral veto-symmetry gates failed;
conjunction is not universally superior to either single objective.

The [boundary/corruption campaign](research/n16k64/campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md)
is **complete**, not a pending experiment:

- Three models (Llama-3.1-8B, Qwen3-4B, Mistral-7B-v0.3), two corpora,
  **171 maps / 168 new policy evaluations** (336 model/corpus cells).
  Each model/corpus analysis has 58 policies including baseline and original map.
- Four bands per side of κ=3, four construction partitions, hence 32 group-only
  policies per model. Common-support coverage is **83.10% / 91.34% / 93.61%**
  for Llama/Qwen/Mistral, with **370 / 931 / 978** tiles per band.
- Four disjoint, rank-interleaved near-boundary replacement pools; nested
  corruption at p=0.1, 0.25, 0.5, 0.75, 1, plus the p=0 anchor.
  Four matched-random p=0.5 controls were **predeclared and completed**.
  Near-boundary replacements are less harmful than random replacements in all
  six point estimates; the pointwise intervals do not all exclude zero. See
  the [full control table](research/n16k64/campaigns/boundary/PRIMARY_RESULTS_TABLES.md).
- Selected groups are more favorable than matched rejected groups overall;
  this is not a claim that every selected group improves over baseline.
  There is **no consistent sharp local jump at κ=3**.
- Full replacement worsens NLL in all six endpoints. This is
  `NLL(corrupted map) − NLL(original map)`, **not** the selector's gain over
  FourOverSix; p is an imposed stress level, not the selector's true error rate.

Both frozen pattern gates pass, but both formal classifications remain
**`power_limited_support`**. Endpoint-specific `descriptive` /
`limited_inference` labels, 10,000 paired document/article-cluster bootstrap
replicates and Holm families are retained. Pointwise 95% CIs do not override
multiplicity or power qualifications; standardized pooled sensitivity is not
raw NLL effect size, accuracy or explained variance. Mistral here is one-shot
analysis validation, not a new independent confirmatory family.

**Timing deviation:** the explicit hour-20 rule prohibited new GPU launches.
Late required retries violated it, and completion took about 61.4 hours rather
than the 24-hour target. These are protocol-deviating supportive/sensitivity
results; strict protocol-conformant eligibility remains **unresolved**, not waived.
See [definitions, endpoint power and timing audit](research/n16k64/README.md#latest-evidence-and-remaining-work).
Nothing in this campaign establishes reorder efficacy.

## Robustness and calibration

Five calibration draws for Llama/Qwen3-4B/Mistral give **30/30 favorable PPL point
estimates** for N16 k=3 versus FourOverSix. This is not 30 independent replications
or universal accuracy preservation: the existing four-task draw analysis includes
negative Mistral differences. See [PPL](research/n16k64/campaigns/primary/analysis_ppl/CALIBRATION_SEED_STABILITY_PPL.json)
and [accuracy](research/n16k64/campaigns/primary/analysis_accuracy/CALIBRATION_SEED_STABILITY_ACCURACY.json).

Calibration size/domain comparisons exist, but composition, sequence budget and
sampling must be separated; the old Math64/Code64/Math64+Code64 comparison does
not hold total budget fixed. Seed0 objective ablation is complete and model
dependent: it does not establish cross-draw universal CE+KL robustness.
Sparse-map overlap alone cannot establish a broad connected flat optimum.
The [next-step specifications](research/n16k64/TODO_EXPERIMENTS.md) prioritize
existing-five-map CPU analysis, cross-draw objective evaluation, a proposed
matched-budget composition design, and only conditionally good-map interpolation.

## Execution status: quality simulation and native prototype

| Path | Storage and computation | Connection to primary N16 | Evidence / limitation |
|---|---|---|---|
| Primary PPL / accuracy | Format mask selects dequantized BF16 weights; per-token activation quantize–dequantize; floating-point Linear | Actual quality path | A6000 / RTX 6000 Ada evidence; no native FP4 dispatch, regardless of policy names containing “native” |
| Separate GB200 mixed prototype | Packed uint8 nibbles, FP8 scales and type map; native activation encoding and SM100 CUTLASS launch | **Not the same method execution**: N256K64 ownership, tensor-global activation amax, different rounding order | Operator/encoding checks recorded; full-output equivalence failed; native accuracy not established and native PPL unmeasured |
| Upstream inference backends | Weight-only dequantization kernels and separate Blackwell W4A4 implementations | Not called by the primary evaluator | Their RaZeR/NVFP4 support is not evidence of N16 E2M1/E0M3 support |

The prototype writes descriptor value 0 for its E0M3 path. NVIDIA's
[PTX ISA 8.8 / CUDA 12.9 Table 45](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)
documents E2M1=1 for that instruction family, not E0M3=0. The saved GB200
observations are experimental evidence, **not official portable support**.
Absence of a public encoding definition is not a proof of hardware impossibility.

See [code-path and backend details](research/n16k64/software/README.md#quality-and-native-execution-paths),
[operator checks](results/task_reorder/full_model_20260920/kernel_406633/report.json),
[full-model gate](results/task_reorder/full_model_20260920/llama_406828/report.json)
and [implementation limitations](results/task_reorder/full_model_20260920/IMPLEMENTATION.md).
Separate diagnostic timing does not validate native N16 quality, speedup,
memory savings, overhead or power benefits.

## Limitations and remaining questions

The primary evidence supports scoped quantization **quality**, not universal
accuracy improvement, individually calibrated causal effects, simultaneous
confidence, native execution or a new SOTA claim. Additional baseline results
also limit a blanket superiority claim. Negative, invalid, OOM and stopped
attempts remain in the [campaign audits](research/n16k64/README.md#where-the-work-stands).

Next work is specified—not launched—in [TODO_EXPERIMENTS.md](research/n16k64/TODO_EXPERIMENTS.md):
P1-A analyzes existing maps/evaluations; P1-B audits/reuses seed0 and plans
45 natural-rule maps / 90 cells across five draws; P2 proposes 30 matched-budget
composition maps / 60 PPL cells; P3 is optional count-matched interpolation
whose new endpoints need evaluation. Boundary/corruption and already completed
GSM8K/PG19 are not relabeled as unfinished work.

## Reproducing and navigating the results

The branch is a review archive with executable historical components, **not a
self-contained replay of every campaign**. Start with the CPU-only integrity
and compact-result checks (no model or GPU required):

```bash
python3 research/n16k64/tools/verify_public_snapshot.py
```

This checks the public snapshot, not every excluded raw artifact or a fresh
bootstrap. The [software guide](research/n16k64/software/README.md) identifies
the exact historical scoring, map installation, PPL, accuracy and analysis
entry points. Generic upstream evaluator flags do **not** reconstruct the
frozen gradient selector.

Exact replay additionally requires pinned external model/tokenizer revisions,
dataset revisions and token-window manifests, original maps/moments and
campaign layout. Most raw arrays, model weights and reviewer ZIPs are
**local-only / not distributed**; no public download is promised.
Six compact boundary paired-cluster arrays are included for CPU recomputation.
Historical outcomes retain original source/config hashes, not this documentation
commit's identity. See [inventory](research/n16k64/ARTIFACT_INVENTORY.md) and
[source lineage](research/n16k64/SNAPSHOT_SOURCE_INDEX.json).

| Research question | Compact evidence |
|---|---|
| Primary quality, non-inferiority and downstream scope | [N16 decision](research/n16k64/campaigns/primary/N16_DECISION.md), [statistical report](research/n16k64/campaigns/primary/STATISTICAL_VALIDITY_REPORT.md) |
| Individual effects versus group ranking | [Mechanism verdict](research/n16k64/campaigns/mechanism/MECHANISM_VERDICT.md) |
| Dose response and objective dependence | [Follow-up verdict](research/n16k64/campaigns/followup/FOLLOWUP_VERDICT.md) |
| Boundary/corruption, including power and timing qualifications | [Current overview](research/n16k64/README.md), [boundary statistical report](research/n16k64/campaigns/boundary/STATISTICAL_REPORT.md) |
| Post-hoc k=2 extension and failed promotions | [Extension risk audit](research/n16k64/campaigns/ppl_improvement/FINAL_SUBMISSION_RISK_AUDIT.md) |
| Other research questions and superseded results | [Repository-wide evidence index](research/n16k64/README.md#repository-wide-research-context) |

## Supplementary and historical work

The k=2 extension is **post hoc**, not the frozen primary rule; held-out,
replacement-selector and downstream promotion failures remain failures.
Primary GSM8K and PG19 analyses are complete for Llama/Qwen3-4B/Mistral;
PG19's four-book scope cannot establish broad long-context robustness.
Scale/selector controls and [additional baselines](research/n16k64/campaigns/primary/analysis_ppl/ADDITIONAL_BASELINES_PPL.json)
remain part of the evidence, rather than attributing all gains to selection.

Separate N256K64 reorder/transfer work is indexed in
[REORDERING_STUDY_STATUS.md](REORDERING_STUDY_STATUS.md) and
[MIXFP4_REPORT.md](MIXFP4_REPORT.md). Its fresh-data diagnoses and inconclusive
accuracy findings do not transfer to N16 or rescue rejected gates.
Earlier local-error, GPTQ/CD2 and tensor-wide activation studies are historical
context, not additional primary confirmations. Details and supersession are
in the repository-wide index above.

## Upstream attribution

This research builds on [NVFP4-RaZeR](https://github.com/abdelfattah-lab/NVFP4-RaZeR).
The prior generic usage/format guide and citation are preserved in
[UPSTREAM_RAZER.md](UPSTREAM_RAZER.md); the released inference artifact remains
in [inference/README.md](inference/README.md). Upstream quality/performance
results are not MixFP4 results.

The [MIT license](LICENSE), third-party notices and upstream patent notice
remain applicable as provided. Please retain the RaZeR citation when using its
implementation; this README does not create a new paper citation.
