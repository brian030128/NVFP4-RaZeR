# Mechanism diagnoses completed

Llama diagnosis full408543 and summary408550 complete, all audits passed, no
active jobs. See results/task_reorder/llama_diagnosis_20260920/REPORT.md for
quantization-context, fresh-transfer, projection-interaction and depth-control
results. These support quantization-dependent repair and substantial earlier
layer activation-quantization prediction error; they do not prove no task-fitting
component or earlier-layer impossibility. No new search/promotion was performed.
Earlier Qwen diagnostic scripts remain unrun under the user's Llama-first scope.

# Completed Llama accuracy result

Four H200 workers408131_[0-3] and summary408134 completed successfully. All
source-weight, question matching and bitwise restoration audits passed.
Accepted joint192147 tiles versus raw187: non-STEM MMLU62.715%→62.935%,
+0.220pp (24correct), Holm p.2848; ARC acc_norm51.280%→51.195%,
-0.085pp (-1correct), Holm p1. Neither change is conclusive.
No candidate or gate changed. No remaining jobs. Qwen experiments deferred per
user priority. See results/task_reorder/llama_accuracy_20260920/REPORT.md.

# User steering: Llama first, outside math/code

Qwen full evaluation cancelled before a completed comparison, per user request.
Use frozen accepted Llama joint192147 versus raw187. Full ARC-Challenge plus
38 MMLU non-STEM subjects (humanities/social-sciences/other); all19 STEM subjects
excluded before any Llama outcome. Report as a non-STEM subset, not full MMLU.
Primary subset and secondary ARC endpoints use paired tests and Holm adjustment.
No candidate fitting, subject selection from outcomes, or map modification.
Plan: results/task_reorder/llama_accuracy_20260920/plan.json.

# Does reordered 256×64 improve task accuracy?

User authorization on 2026-09-20 reopens bounded diagnostics after the earlier
research stop. Their priority is actual task accuracy: task-aware fitting is
acceptable if it produces transferable accuracy gains. The four-GPU cap remains.

The method is discrete, task-aware calibration: floating-point source weights
stay fixed, but calibration CE and teacher KL select permutations and format
masks. It can both repair quantization damage and adapt the effective quantized
model to its calibration distribution. Lower perplexity alone cannot distinguish
these effects or establish better answer accuracy.

Existing six-task accuracy results apply to **8×64**, not the new reordered
256×64 models. Qwen's unweighted task mean was 68.41% FourOverSix versus 69.19%
MixFP4, with pooled paired McNemar p=0.117. Llama's corresponding means were
67.20% versus 67.14%, with p=0.617. Neither establishes an improvement over
FourOverSix at conventional significance. Do not transfer those numbers to the
212-tile Qwen or 147-tile Llama arrangements.

## First experiment: frozen Qwen answer accuracy

`results/task_reorder/accuracy_20260920/plan.json` freezes the exact 212-tile
compacted both-axis Qwen map and local 195-tile raw map before task evaluation.
The compacted map has the same effective quantized weights as the accepted
7.255834 / 10.167336 PPL model. No new fitting, map election, threshold changes,
or benchmark-based selection is allowed in this experiment.

Use lm-eval-harness 0.4.5, full zero-shot MMLU and full zero-shot ARC-Challenge,
no chat template, 2048 maximum context, fixed batch size eight. MMLU uses all 57
subjects and its document-weighted accuracy; ARC uses acc_norm. Four H200
workers cover ARC and three disjoint sets of MMLU subjects. Each worker evaluates
both policies serially on the same physical GPU with identical data and batching.
Preserve model hashes, task dataset fingerprints, question/prompt/target hashes,
and every per-question correct/incorrect outcome. Restore raw weights afterward
and require bitwise-identical audit logits.

Report correct-answer gains and losses, paired accuracy differences and ±2SE,
and exact two-sided McNemar tests. Apply Holm adjustment across the two endpoints;
per-subject results are descriptive. A positive but nonsignificant difference is
not an established improvement. This is fake-quantized-model accuracy, not
accuracy validation of the native backend.

Environment setup 407955 completed with torch/transformers versions unchanged.
Smoke 407961_0 passed in 3m38s, evaluating two documents from each of ARC and one
MMLU subject. Its outcomes are correctness checks, never accuracy evidence.
Full accuracy 407966_[0-3] is running on four H200s, with summary 407970 dependent
on the full array. Attached monitors handle completion directly because
the previous same-session queue notifications failed; a detached logger is not
being treated as a mechanism to resume the agent.

## Subsequent mechanism diagnostics

`run_taskfit_diagnosis.py` is prepared but has not run. It will compare the frozen
both-axis and independently confirmed row-only Qwen policies on reused fit/election
windows, new math/code documents, and new general-text documents, reporting actual
CE and teacher KL. It also tests the counterfactual BF16 weight changes
`W_BF16 ± (W_arranged - W_raw)`, without quantizing those counterfactual models.
Benefit from the positive change in BF16 would suggest a task-directed component;
benefit only in the quantized context would be consistent with context-dependent
quantization repair. Neither outcome alone is a proof of a unique cause.

The earlier authorized layer-depth diagnosis remains pending: isolated frozen
tile changes at several depths, predicted versus actual losses on fit/fresh data,
then isolated versus combined changes. Preserve the raw map outside the intervention.
Do not launch a new broad permutation search as a substitute for diagnosis.
