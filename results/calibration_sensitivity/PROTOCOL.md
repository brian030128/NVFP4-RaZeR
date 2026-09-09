# Calibration source and sample-count sensitivity with frozen causal evaluation

Declared before this study's map derivation and evaluation. User requested no
seed replication. Models: Qwen3-4B, Llama-3.1-8B base, Qwen3.8-27B, using the
same pinned revisions and frameworks as the preceding causal studies.

## Fixed design

Reuse each model's existing shared CE/KL score table: 64 C4, 64 OpenWebMath,
64 CodeParrot sequences of 512 tokens. No new scoring, per-configuration
calibration, fitting-loss backtracking, held-out acceptance, or winner election.
The fixed algorithm ranks max(mean CE + 2 SE, mean teacher KL + 2 SE) and
selects at most 256 negative-score 8x64 E0M3-alpha1 type blocks globally.
All other blocks use canonical FourOverSix E2M1. The score pass used the
original window-wide activation factor; evaluation uses causal per-token
FP32 activation factors. This convention is retained, not silently changed.

Evaluate all 17 calibration settings per model:

* C4, OpenWebMath, CodeParrot individually at 16, 32, 64 sequences (9).
* C4+math, C4+code, math+code at 64 total, 32 from each source (3).
* C4+math+code at 16, 32, 64, 128, 192 total (5).

Subsets are nested prefixes of the existing source order. Pooled counts are
6/5/5, 11/11/10, 22/21/21, 43/43/42, 64/64/64 in web/math/code order.
Include FourOverSix and weight-MSE controls (19 total policies). Reproduce
the three existing calibrated maps exactly before evaluating any new map.
Freeze every setting regardless of its later result. No additional seeds.

## Evaluation and integrity

Every policy is evaluated on the same five datasets within a model:
held-out C4 (256 documents, 512-token crops), full WikiText-2 raw test
(nonoverlapping 512-token windows, final incomplete window omitted), and
literature/PG19, science/arXiv, government/GovReport (64 distinct test
documents each, one 512-token crop each). Reuse the pinned data builders
and source revisions from the current causal evaluations. These previously
inspected dataset families make this a sensitivity study, not untouched
confirmatory evidence. C4 is within-source only for settings containing C4;
all five targets are out-of-source for math-only, code-only, and math+code.
Broad web sources can contain similar topics; source transfer is not proof
of semantic-domain or pretraining independence.

Exclude exact hashes of all 192 calibration documents for every setting so
evaluation membership cannot depend on which calibration subset is used.
WikiText checks exact nonempty row versus calibration-document hashes.
Preserve model weights, quantizer hashes, input hashes, maps, per-window NLLs,
and exact 128-token prefix independence checks for every policy.

Existing losses may be reused only when model revision, framework,
quantizer source, map equality, and every evaluation-window token hash match.
Recompute the first window of each reused policy/dataset cell and assert its
NLL agrees within 1e-6. Record provenance per reused cell. Existing results
are not counted as independent replications. New 27B transfer cells have no
prior losses and must be evaluated. All computation and tests use Slurm.

## Required reporting

Publish all model x calibration-setting x destination cells, including harms.
Report separate dataset PPL, baseline-relative PPL and paired NLL differences
with descriptive two-SE intervals. Report the unweighted arithmetic average
over the five dataset PPLs separately; it is not pooled-corpus perplexity.
Also report mean log-PPL difference over the four non-C4 destinations, giving
a common out-of-source comparison for every calibration setting.

For each setting report actual selected E0M3 8x64 type-block count, fraction
of all quantized type blocks, negative-score eligible count before the cap,
fraction of eligible blocks selected, whether the cap binds, and per-module
selected counts. One selected type block contains 512 weights and 32 distinct
16-element scale blocks; those are not interchangeable counting units.
Report pairwise Jaccard overlap and the intersection count across settings.

Compare source at equal sequence budgets, pair mixtures at 64, and the
pooled nested sample-count curve. Describe sensitivity without selecting a
best configuration or revising the algorithm. No inferential claims about
calibration draws are possible without seed replication. Per-window two-SE
intervals describe evaluation variation only, do not adjust for multiple
comparisons, and do not account for adjacent WikiText windows sharing articles.
The 256 cap and factor 2 remain empirical constants; no universal guarantee
or new source-diversity confirmation screen is claimed. This is simulated
nonhead-text-linear W4A4 reference-text PPL, not generation or kernel speed.
