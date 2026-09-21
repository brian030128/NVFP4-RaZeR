# MixFP4 N16K64 research snapshot

This directory consolidates the append-only N16K64 research completed between
2026-09-11 and 2026-09-20. It is based on repository `main` commit
`2d3e8f397d57009ba843c83b2a75a176262b026c` and was frozen for review on
2026-09-21. The original campaigns remain immutable outside Git; this is a
small, redacted review snapshot with per-file source hashes.

The [root MixFP4 homepage](../../README.md) contains the six-model PPL and
eight-task accuracy tables. This directory supplies detailed qualifications and
traceability; the [software guide](software/README.md#quality-and-native-execution-paths)
separates the actual BF16 quality path from the distinct GB200 prototype.

Start with:

- [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md) for what is and is not supported;
- [`ARTIFACT_INVENTORY.md`](ARTIFACT_INVENTORY.md) for included/excluded evidence;
- [`COMPATIBILITY_AND_VALIDATION.md`](COMPATIBILITY_AND_VALIDATION.md) for integration, tests, and hard gates;
- [`SNAPSHOT_SOURCE_INDEX.json`](SNAPSHOT_SOURCE_INDEX.json) for original/public SHA-256 lineage.

## Research question and frozen primary method

MixFP4 chooses E2M1 or E0M3 for each weight tile while retaining NVFP4-style
4-bit values and K16 UE4M3 block scales. The frozen N16K64 method uses 16x64
weight type tiles and selects E0M3 only when the upper first-order score is
negative for both next-token cross-entropy (CE) and KL divergence:

```text
mean_CE + 3 * SE_CE < 0
mean_KL + 3 * SE_KL < 0
```

Scores are computed at the all-FourOverSix student reference from seed-0
calibration: 64 OpenWebMath and 64 CodeParrot sequences, each 512 tokens.
Activations use causal per-token FourOverSix. Only non-head linear weights are
eligible; embeddings, norms, `lm_head`, and KV cache are excluded. Every N16K64
campaign here evaluates software/fake-quantized, BF16-dequantized quality.

The k=3 conjunction is a conservative frozen selector, not a simultaneous
confidence interval and not a proof that each selected tile is beneficial. CE
and KL are dependent. The later k=2 campaign is explicitly post hoc.

## Where the work stands

| Campaign | Purpose | Outcome | Attempts and compute | Review entry point |
|---|---|---|---|---|
| Primary full validation | Freeze N16K64 k=3; measure PPL, accuracy, controls, calibration sensitivity, held-out models, and artifact integrity. | **Strong pass for the frozen quality component only.** Six confirmatory PPL non-inferiority endpoints passed; individual-tile fidelity did not. | 338 run directories: 310 complete, 17 failed, 7 invalid, 4 not started. 173.30 leased GPU-hours over all attempts. | [`N16_DECISION.md`](campaigns/primary/N16_DECISION.md) |
| PPL-improvement extension | Broad k=2 robustness, matched-density controls, five draws, calibration size, selector/scale/activation alternatives, held-out families, downstream and portability gates. | k=2 breadth and density-specific information supported **post hoc**. Replacement selector, held-out generalization, downstream, SOTA, and portability claims failed or stopped by gate. | 169 attempts: 152 complete, 13 failed, 3 invalid, 1 not started. 89.02 GPU-hours. | [`FINAL_SUBMISSION_RISK_AUDIT.md`](campaigns/ppl_improvement/FINAL_SUBMISSION_RISK_AUDIT.md) |
| Mechanism | Ask why aggregate selection improves PPL despite weak individual-tile prediction. | Ranking-not-calibration supported. Strict CE/KL-veto and attention/MLP-interaction gates not supported. | 10 run directories; 7.92 valid GPU-hours. | [`MECHANISM_VERDICT.md`](campaigns/mechanism/MECHANISM_VERDICT.md) |
| Follow-up | Eight-bin dose response, matched-budget objective ablation, and missing Mistral veto arm. | Group-level dose response supported. No universal conjunction superiority. Mistral veto symmetry not supported. | 10 run directories; 9.07 valid / 9.11 all-attempt GPU-hours. | [`FOLLOWUP_VERDICT.md`](campaigns/followup/FOLLOWUP_VERDICT.md) |
| Boundary/corruption | Span the kappa=3 boundary and corrupt full maps with four distinct near-boundary pools. | Both pattern gates passed, but both final classifications are **power_limited_support**. No uniform local jump at kappa=3. | 17 run directories; all 168 required evaluations; 23.26 valid / 45.03 all-attempt GPU-hours. | [`BOUNDARY_CORRUPTION_VERDICT.md`](campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md) |

The boundary/corruption campaign did not meet its hour-20 launch cutoff or
24-hour target; it completed in about 61.4 hours after fail-closed co-tenancy
and monitoring retries. Hour 20 was a mandatory launch-stop rule, not an
optional target. Late Llama/Qwen outcomes are protocol-deviating required-retry
evidence; their eligibility as fully protocol-conformant evidence is unresolved.
The stored `power_limited_support` labels are preserved, without upgrading the
late evidence. See [current audit and corrections](INTEGRATION_AUDIT.md).

## Primary quality results

Delta-log-PPL is delta NLL. Negative values favor N16K64 over FourOverSix.
Natural-cluster paired bootstrap intervals are shown.

| Confirmatory model | WikiText delta-NLL [95% CI] | C4 delta-NLL [95% CI] |
|---|---:|---:|
| Mistral-7B-v0.3 | -0.0039 [-0.0049, -0.0030] | -0.0025 [-0.0036, -0.0015] |
| OLMo-2-1124-13B | -0.0017 [-0.0030, -0.0004] | -0.0010 [-0.0016, -0.0003] |
| Phi-4 | -0.0053 [-0.0066, -0.0041] | -0.0039 [-0.0047, -0.0031] |

All six frozen Holm-adjusted non-inferiority gates passed. N8 nevertheless has
the better PPL point estimate in all six cells, so the result does not establish
N16 superiority. The pooled N16/N8 **log-PPL gain ratio** is 0.77984:
sum of six N16-minus-FourOverSix delta-NLL estimates (-0.0182349212), divided
by the corresponding N8 sum (-0.0233828808). This is a point estimate across
Mistral/OLMo2/Phi-4 and C4/Wiki, not a tile-count fraction, mean of six ratios,
or universal 77.98% guarantee. Native overhead was not measured.

The confirmatory eight-task macro-accuracy differences versus FourOverSix are
+0.24 pp for Mistral, +0.08 pp for OLMo2, and +0.31 pp for Phi-4; every 95% CI
includes zero, while the frozen -0.5 pp non-inferiority gate passes. These are
not accuracy-improvement claims.

## Mechanism result in one paragraph

Composition-matched tile groups show consistent aggregate ranking: stronger
score margins tend to produce better group-level marginal effects, and the
eight-bin dose response remains after a leave-Qwen sensitivity. This coexists
with poor individual-tile sign/magnitude calibration. At the critical boundary,
four bands were feasible: common-support coverage is 83.10% for Llama, 91.34%
for Qwen, and 93.61% for Mistral, with 370/931/978 tiles per band. All four
corruption pools are truly distinct. Full replacement worsens NLL across all
three models and both corpora, but power qualification downgrades the final
claim to `power_limited_support`. Corruption proportion is an imposed stress
level, not a measured selector error rate.

## Baseline, group-only, marginal, and corruption comparisons

- **FourOverSix baseline:** no E0M3 weight tiles. `NLL(policy)-NLL(FourOverSix)`
  answers whether an entire policy changes model loss.
- **Group-only:** only the named tile group is changed from FourOverSix. This
  gives a common context for selected and rejected boundary bands.
- **Full minus group:** `NLL(full)-NLL(full minus G)` is the preferred observed
  marginal contribution of group `G` in the selected-map context. It is not
  interchangeable with group-only evidence.
- **Full-map corruption:** keeps total and per-module tile counts fixed while
  replacing selected tiles with near-boundary rejected tiles. `p=1 - p=0`
  measures imposed map damage, not original-selector improvement over baseline.
- **Natural threshold vs matched-K:** natural CE-only/KL-only/conjunction maps
  answer what each rule selects at k=3. Matched-K maps isolate objective
  information at a common per-module budget. Their interpretations are separate.

## Completed, stopped, and superseded directions

Completed work includes the primary k=3 quality campaign, the post-hoc k=2
extension, mechanism ranking/veto/interaction experiments, eight-bin and
objective-ablation follow-up, and boundary/corruption stress tests. All relevant
failed, invalid, OOM, co-tenancy, and negative outcomes remain summarized.

Stopped or unsupported work includes the extension's 256-sequence continuation,
replacement-selector promotion, downstream accuracy/GSM8K/PG19 promotion,
held-out broad-generalization claim, SOTA framing, and A6000/Ada portability
claim. Qwen3.8-27B was not promoted in the follow-up. Historical tensor-wide N8
objective work must not be mixed with the causal-per-token N16 protocol.

The later 256x64 reordering/native-diagnostic program is already in current
`main` and is indexed from the claim matrix rather than duplicated here. Its
historical 90% recovery target was not met, native PPL is unmeasured, and its
timing statements have narrower hardware/operation scopes than this N16K64
quality work.

## Minimal review and reproduction

No GPU is needed to validate this public snapshot:

```bash
python3 research/n16k64/tools/verify_public_snapshot.py
```

The command parses JSON/JSONL/CSV/SVG, checks PNG signatures and internal links,
verifies source/public hashes, checks the manifest, scans for private identifiers
and common credential patterns, and independently spot-checks reported
transformations. See [`validation/VALIDATION_COMMANDS.md`](validation/VALIDATION_COMMANDS.md)
for the exact environment and test commands used for this branch.

Historical campaign code is under [`software/`](software/). It must be run as a
versioned snapshot, for example by putting the selected snapshot before the
repository on `PYTHONPATH`; do not combine modules from different snapshots.
Machine-specific defaults were replaced by environment variables such as
`MIXFP4_WORKSPACE_ROOT`, `MIXFP4_PRIMARY_CAMPAIGN`, and
`NVFP4_RAZER_STAGE_ROOT`. This portability patch does not change quantization,
selector, data, or metric semantics.

A full GPU rerun additionally requires the pinned model/tokenizer revisions,
dataset revisions, exact maps/moments, and paired arrays recorded by the internal
manifests. Those large artifacts are not in Git and are not publicly downloadable
from this branch. Six small boundary paired-cluster NPZ files are included
byte-identically under `campaigns/boundary/arrays/`; other large raw arrays remain
external. The branch supports audit and compact-result recomputation, with
versioned executable historical code, but is not self-contained for GPU replay.

## Latest evidence and remaining work

The [current evidence audit](validation/CURRENT_EVIDENCE_AUDIT.json) rechecked
source hashes, all 171 boundary map headers/hashes and all six 58-policy paired
arrays, and independently recomputed the p=1 effects and retained-gain ratio.
It resolves the exact 15 calibration manifests and 45 existing natural-rule maps.
The original snapshot cutoff remains 2026-09-21T05:24:15Z; completion additions
restore files omitted at that cutoff, not newly generated experimental outcomes.
Its `research_status` records execution separately from terminal classification,
with protocol/outcome hashes, source/map provenance and missing external inputs.

For exact boundary construction, see
[critical-k and coverage definitions](CLAIM_EVIDENCE_MATRIX.md),
[band definitions](campaigns/boundary/BAND_DEFINITIONS.json) and
[pool definitions](campaigns/boundary/CORRUPTION_POOL_DEFINITIONS.json).
The intervention unit is an N16K64 format block, not a 1×16 scale block.
The four partitions are construction sensitivities, not independent replications.
Corruption p counts replacements among all original selected units in this
campaign (corruption coverage is 1); p=1 replaces all of them. Four rank-interleaved
disjoint pools share the nearest-4K rejected reservoir within each stratum;
they are not progressively more distant reservoirs.

Endpoint power must be read individually. For example, Qwen C4's corruption
slope is `limited_inference`, while its boundary slope is `descriptive`;
calling every Qwen endpoint descriptive is incorrect. The
[results table](campaigns/boundary/PRIMARY_RESULTS_TABLES.md) shows pointwise
intervals, and the result JSON retains adjusted tests. Near-minus-random p=.50
is negative in all six point estimates but not uniformly resolved by pointwise
intervals; it was a completed planned control, not a newly proposed study.

Five-draw N16 conjunction PPL is complete for Llama/Qwen/Mistral: all 30
model/draw/corpus point estimates favor FourOverSix-relative improvement. This
does not mean every task or endpoint is favorable: four-task macro accuracy
differences range from +0.124 to +0.442 pp (Llama), +0.565 to +1.300 pp (Qwen),
and -0.149 to +0.729 pp (Mistral). These are observed draw ranges, not confidence
bounds. The representative tasks are ARC-Challenge, BoolQ, PIQA and WinoGrande.
Some historical secondary PPL intervals used 2,000 bootstrap draws and zero
Monte Carlo p-values; they remain historical output and are not upgraded to
10,000-replicate plus-one inference by this review.

| Priority | Remaining question | Existing evidence reused | New evaluation |
|---|---|---|---|
| P0 | Evidence/document/workspace synchronization | Five completed campaigns | None; completed in this integration |
| P1-A | Exact multi-map frequency and density-aware stability | 15 conjunction maps, 30 PPL cells, four-task accuracy | None initially |
| P1-B | Objective robustness across draws | 45 natural maps; seed0 objective follow-up; all conjunction draws | Missing cross-draw objective cells only; reuse audit first |
| P2 | Composition versus sampling at a fixed budget | Existing one-draw size/domain controls | Proposed 30 maps / 60 PPL cells, less exact reuse |
| P3 | Good-map interpolation feasibility | Existing maps, subject to count matching | Optional; endpoints must be revalidated |

Full execution specifications and launch blockers are in
[TODO_EXPERIMENTS.md](TODO_EXPERIMENTS.md). Completed boundary/corruption,
including the predeclared near-versus-random control, is not a pending task.
Primary GSM8K is complete for Llama/Qwen/Mistral; PG19 4K/8K is complete for
the same three models, with only four book clusters. Extension downstream
promotion stopped by its own gate; it must not erase those primary results.

## Claims that remain prohibited

Do not infer reliable individual-tile causal signs or magnitudes, calibrated
finite effects, a simultaneous k=3 confidence guarantee, a true selector error
rate, universal CE/KL/conjunction optimality, all-model generalization, or SOTA.
The N16K64 campaigns do not measure native FP4/E0M3 Tensor Core execution,
latency, throughput, speedup, N8/N16 overhead, area, power, or Blackwell
performance.

## Repository-wide research context

The five campaign rows above are the paper-facing N16 core. The following
question-level index also covers tracked work outside this subtree. These are
different protocols, not extra independent confirmations of the same method.
Later reports supersede earlier *status* entries without converting failures
into passes. The 2026-09-21 homepage refresh reuses existing results only.

| Question / workstream | Evidence and status | Interpretation / paper role |
|---|---|---|
| Does local format/reconstruction fitting suffice? | [Early report](../../results/MIXFP4_REPORT.md), [task sensitivity](../../results/task_sensitivity/REPORT.md) | Historical fixed-alpha / N8, small-calibration, tensor-wide activation experiments; seed failures motivate whole-map validation. Not the frozen causal-per-token N16 protocol. |
| Are baseline numbers comparable? | [Baseline protocol audit](../../results/baseline_protocol_audit/REPORT.md) | Window length, BF16/FP16 labeling, activation convention and C4 shard differences matter. Use within-protocol comparisons; do not pool historical PPL. |
| Does calibration transfer across domains? | [Domain observations](../../results/task_sensitivity_domains/CALIBRATION_OBSERVATIONS.md), [pooled confirmation](../../results/pooled_confirmation/REPORT_332349.md) | Earlier pooled gains coexist with a failed equal-token diversity gate. Not proof that more diverse calibration universally helps. |
| Did the old calibration sweep finish? | [Cancelled/superseded run](../../results/calibration_sensitivity/RUN.md) | Partial results are historical; its replacement design must be read separately, not counted as a completed primary sweep. |
| Are five maps stable and accurate? | Primary V50/V51 PPL/accuracy JSON; [TODO P1-A/P2](TODO_EXPERIMENTS.md) | Completed evaluations, but joint density-aware map analysis and matched-budget composition inference remain distinct tasks. Thirty favorable PPL points do not establish all-task preservation. |
| Do scale/selector controls explain the gain? | [Selector controls](campaigns/primary/analysis_ppl/SELECTOR_CONTROLS_PPL.json), [extension audit](campaigns/ppl_improvement/FINAL_SUBMISSION_RISK_AUDIT.md) | Retain scale/density controls and failed alternative-selector/held-out promotions. k=2 is post hoc; no universal replacement winner. |
| Do supplementary tasks transfer? | [Generation](campaigns/primary/analysis_accuracy/CONFIRMATORY_GENERATION.json), [long context](campaigns/primary/analysis_ppl/LONG_CONTEXT.json) | Completed GSM8K/PG19 is supplementary, with four PG19 book clusters. Extension stopped downstream promotion is a separate outcome. |
| Can reordering improve N256K64 ownership? | [Reordering status](../../REORDERING_STUDY_STATUS.md), [current report](../../MIXFP4_REPORT.md) | Separate accepted Llama/Qwen both-axis results; the 90% recovery target was not met. Failed fresh-transfer gates remain visible. N16 boundary evidence does not validate reorder efficacy. |
| Does the accepted Llama reorder improve broader accuracy? | [Accuracy report](../../results/task_reorder/llama_accuracy_20260920/REPORT.md) | Completed Llama comparison is inconclusive for improvement/equivalence; Qwen was cancelled with partial artifacts, not a completed accuracy confirmation. |
| Why do reorder scores mispredict finite effects? | [Fresh-data/depth diagnoses](../../results/task_reorder/llama_diagnosis_20260920/REPORT.md) | Quantization-dependent repair and early activation-quantization sensitivity are diagnostic support. Fixed layouts transferred across depth do not prove early-layer impossibility; no established projection synergy. |
| Is native mixed execution equivalent? | [Native implementation](../../results/task_reorder/full_model_20260920/IMPLEMENTATION.md), [terminal report](../../results/task_reorder/full_model_20260920/llama_406828/report.json) | GB200 operator/encoding checks and diagnostic timing exist. Full-output equivalence failed, native accuracy is not established, native PPL unmeasured; N256/global-amax is not primary N16/per-token. |
| What does upstream supply? | [Preserved generic guide](../../UPSTREAM_RAZER.md), [inference artifact](../../inference/README.md) | RaZeR/NVFP4 kernels and historical GPTQ/CD2 components do not imply frozen N16 selector dispatch or inherit its quality results. |

Current untracked handoff archives and full campaign directories remain local
provenance, not GitHub links or additional public raw-data releases.
The [inventory](ARTIFACT_INVENTORY.md) explains these exclusions. This index
does not silently combine outcome-selected historical studies with frozen
confirmation.
