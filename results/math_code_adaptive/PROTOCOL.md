# Math/code-only calibration; adaptive type count; WikiText and C4 evaluation

This supersedes the unfinished five-dataset, three-source sensitivity study.
No seed replication. Models: Qwen3-4B, Llama-3.1-8B base, Qwen3.8-27B, at their
previously pinned revisions. The 256-block cap is a comparison, not the new
method's count rule. Preserve all outcomes, including zero switches and harms.

## Data separation

Calibration uses only 64 OpenWebMath and 64 CodeParrot documents, 512 tokens
each. Reconstruct their recorded source revisions, document hashes and offsets
from the preceding study. Do not load C4 or WikiText calibration data and do
not use their score rows. Fresh scoring uses causal per-token activation
factors throughout, rather than inheriting the older window-wide score pass.
One fixed pseudo-label RNG is used to reproduce the curvature estimate;
there are no repeated seed experiments or seed selection.

From the single shared 128-sequence pass, use math-only and code-only nested
prefixes at 16/32/64 sequences, and balanced math+code prefixes at total
16/32/64/128 (10 settings). No per-setting model-scoring pass is performed.
The same source subsets feed the adaptive and fixed-256 comparison methods.

Only after all maps freeze, evaluate on WikiText-2 raw test and held-out C4.
WikiText uses every full nonoverlapping 512-token window of the concatenated
test split; C4 uses 256 validation documents with 512-token crops. Reuse the
pinned evaluation builders. Exact hash exclusions use the 128 calibration
documents, independent of setting. Record any difference in C4 membership
from earlier studies rather than assuming identical evaluation samples.
Source separation does not establish semantic-domain, near-duplicate, or
pretraining independence. Both target families were previously inspected.

## Shared baseline derivatives and curvature

Candidates: canonical FourOverSix E2M1 and E0M3-alpha1 per 8x64 type block.
Pristine BF16 teacher; simulated nonhead-text-linear W4A4 student; other
operations native. At the unchanged baseline, one student forward per input
supports three backward passes: CE, teacher KL, and CE against independently
sampled student labels at each of the 511 prediction positions. Activation
quantization uses an identity straight-through derivative during scoring.

For block j, record per-sequence directional derivatives g_CE, g_KL and b
(sampled-label mean-NLL derivative). With n calibration sequences define

    u_j = max(mean(g_CE_j) + 2 SE(g_CE_j), mean(g_KL_j) + 2 SE(g_KL_j))
    h_j = 511 * mean(b_j^2).

The sampled predictive-GGN estimate H = (511/n) B^T B is PSD with diagonal h.
For any PSD H and binary s, s^T H s <= (sum_j s_j sqrt(h_j))^2 by Cauchy–Schwarz.
Unlike the earlier noisy off-diagonal Fisher optimization, this uses a
worst-case alignment penalty for the estimated curvature directions.

Sort negative-u blocks in increasing u, breaking ties by module/flat order.
For every prefix k, including k=0, calculate from the shared statistics

    U(k) = sum_{j in prefix(k)} u_j + 0.5 * (sum_{j in prefix(k)} sqrt(h_j))^2.

Choose the smallest k minimizing U(k) over all eligible prefixes. There is
no count cap or iterative update limit. This is exact optimization along a
fixed ranking path, not global optimization over all subsets. It requires
no candidate-map forward passes, calibration-loss backtracking, or evaluation
feedback. The fixed comparison selects the first at-most-256 negative-u
blocks using the same freshly computed causal derivatives.

The PSD inequality is exact for H, but H is a sampled, local GGN surrogate.
It does not bound the true network Hessian or finite-switch remainder;
straight-through gradients are approximate. Two-SE selection scores are
heuristics, not simultaneous guarantees. This is an unvalidated adaptive
method, not a claim that curvature-based selection is new or always works.

## Reporting and execution

Report separate WikiText and C4 PPL, their unweighted arithmetic average,
paired baseline-relative NLL differences with descriptive two-SE intervals,
and actual E0M3 8x64 type-block counts/fractions for every model and setting.
One type block contains 512 weights and 32 distinct 16-element scale blocks.
Report selected identities/overlap and predicted linear/curvature terms.
Include baseline and weight-MSE controls. No winning setting is elected.

No C4-calibrated pooled192 map belongs to the new experiment; historical
tables stay separately labeled. Prior losses may be reused only with full
weight/map/quantizer/input identity checks and a freshly matched first-window
loss. Otherwise recompute them on this study's inputs. Adjacent WikiText
windows can share articles; two-SE values do not account for that dependence,
multiple comparisons, or unmeasured calibration-seed variability.

All tests, scoring, selection, evaluation and aggregation run through Slurm.
Dataset/policy shards are fixed for queue wall limits and merged only with
identical model, map and input provenance. Partial/cancelled jobs are not
complete results and must not overwrite the primary report.
