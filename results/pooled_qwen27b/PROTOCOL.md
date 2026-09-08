# Frozen pooled rule on Qwen3.8-27B

Declared before submission; no result-dependent configuration choice.
Use Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
with native Qwen3.5-family Transformers 5.16.1, as verified in the prior
integration probe. Quantize exactly the 496 text linear projections, excluding
head, embeddings, vision, recurrent state, depthwise convolution and norms.
These are simulated W4A4 reference-text losses, not native kernel measurements.

## Unchanged method

Same FourOverSix E2M1 versus E0M3-alpha1 candidates, 8x64 weight tiles,
per-16-element E4M3 scale blocks. Use 64 C4, 64 OpenWebMath, 64 CodeParrot
training documents with 512-token crops; same stream order and crop seeds
as the seven-model recipe, with source revisions and file paths pinned to
the previous Qwen4B record. Exclude duplicate text hashes between sources.

At the unchanged FourOverSix student collect all 192 per-sequence directional
CE and BF16-teacher KL scores using identity activation STE. Scoring uses
the same window-wide activation factors as the preceding recipe. Define
u_j=max(mean(CE_j)+2SE(CE_j), mean(KL_j)+2SE(KL_j)). Choose at most 256
most negative u_j, with the original module/row-major order breaking ties.
No backtracking, measured-loss gate, per-domain election, new threshold,
calibration seed search or retuning. C4-only64 and mixed64 (22/21/21) controls
reuse this one table; the weight-MSE control uses the same candidates.

Memory-only adaptation: compute pristine teacher log probabilities first
and store them losslessly as FP32 CPU tensors. Then reuse the same model
as the student, holding weight candidates and per-module score tables in
CPU memory. Score each module separately before globally ranking tiles;
verify this produces exactly the same maps as the original full-table
implementation on a synthetic fixture. No quantization of teacher logits,
score subsampling or approximation of the uncertainty formula is introduced.

## Frozen-map C4 evaluation

Save maps and their hashes before loading evaluation examples. Use the
identical held-out C4 recipe from results/c4_frozen/PROTOCOL.md: pinned
validation shard00001, first 256 distinct eligible documents after excluding
all 192 calibration text hashes, one 512-token crop using seed20260928.
This is held-out within-source measurement, not new-domain confirmation.

Every policy uses causal per-token FP32 activation factors. Verify exact
prefix independence for FourOverSix and pooled192 with the same 128-token
prefix intervention. Evaluate all saved maps on exactly the same tokens.
Report absolute PPL, paired pooled192-minus-control NLL mean +/-2SE and
relative PPL change regardless of outcome. No loss result changes a map.
Two-SE is descriptive, not simultaneous inference. Older backtracking-based
27B results are historical and are not comparisons under this protocol.

One two-H100 Slurm job, maximum two GPUs at a time; all heavy work and tests
run on compute nodes. Dependencies and downloads use job-local scratch.

## Implementation retry before any held-out evaluation

Initial normal-queue job332828 was cancelled while pending because of its
QOS job limit. Development-queue attempt332829 passed all prerequisite tests,
loaded the pinned model, and began scoring at about25 seconds per sequence.
It was stopped before map election or held-out evaluation to avoid approaching
the queue time limit. Its runner snapshot and partial report are retained.
The retry uses pinned CPU candidate buffers and caches at most12GiB of exact
FP32 candidate differences per GPU. These are memory-placement changes only;
all candidates, gradients, calibration examples, scoring and election rules
are unchanged. No held-out outcome motivated the retry.
