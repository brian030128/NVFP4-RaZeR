# Continued research, September 8

The user requested continuing until a defensible success. No success is assumed
from a new objective, a fitting guarantee, or one favorable configuration.

## Calibration-free online correlation

The previous channel-energy activation rule discarded cross-channel terms.
The new exact rule updates the full current-layer activation output residual
after each type switch. Its compact counterpart uses a frozen weight-only
low-rank-plus-diagonal representation under a storage budget.

Job332256 caught a zero-input edge case in correctness tests before evaluation.
After the fix, job332259 completed all three models. Protocol, implementation,
raw losses and paired summary are in results/online_format. Neither exact nor
compact succeeded on the fixed transfer screen. These are objective-specific
results, not evidence that every calibration-free method must fail.

## Joint weight/activation error

The next exact objective includes X(Wq-W)^T, allowing an activation type switch
to compensate weight error. The compact model retains both projected factors
and per-channel residual cross terms, with a fixed metadata budget. Job332279
tested exact, compact and activation-only ablations on fresh examples. Its
declared screen is unchanged after inspecting results. No calibration corpus
or evaluation labels enter these runtime decisions.

Exact joint election improved seven of nine cells over FourOverSix, with no
supported harm, but beat activation-only election in only five of nine cells.
The compact version improved six of nine over FourOverSix and five of nine
over activation MSE. Neither passed its prespecified comparative screen.
See [paired results](../joint_online_format/REPORT_332279.md).

## Whole-network task interactions

Local layer objectives can improve while downstream task loss worsens. A
separate protocol in results/global_format instead collects one shared
matrix of per-sequence tile task derivatives and optimizes a binary quadratic
model with cross-layer interactions. It uses the sequence empirical Fisher,
not the exact Hessian, and an identity activation STE for scoring. It has no
candidate-loss backtracking or selected predicted-NLL budget. Implementation
tests passed before model experiments. Job332294 improved four of nine cells
over FourOverSix and failed the screen. See
[paired results](../global_format/REPORT_332294.md).

## Teacher-KL and predictive Fisher

Job332300 replaced observed-label sequence curvature with independently sampled
student labels, and the scoring objective with KL from a pristine BF16 teacher.
An analytic diagonal-shrinkage estimate regularized noisy off-diagonal terms.
The primary map improved six of nine cells, but had three supported harms and
beat the fixed-budget control in only two. The selector hit its4096-step cap
on Llama and OPT. See [paired results](../predictive_format/REPORT_332300.md).

## Analytic teacher-error curvature component

Job332305 retained the exact rank-one teacher-error component of predictive
Fisher and shrank only its orthogonal residual. Tests verify the categorical
identity, PSD decomposition, and discrete surrogate objective. This does not
make the surrogate exact for a network after finite tile flips. With a16384
step cap, the method still regressed badly on OPT. All results are retained in
the [protocol directory](../directed_format/PROTOCOL.md).

The subsequent read-only audit332309 replayed the frozen maps on exactly the
original64 fitting examples, checking source-weight and data hashes. It changed
no maps and made no acceptance decisions:

| Model | Baseline fitting KL | Fixed-budget fitting KL | Standard-Fisher fitting KL | Directed fitting KL |
|---|---:|---:|---:|---:|
| Llama1B | 0.145144 | 0.129318 | 0.242128 | 0.224108 |
| OPT350M | 0.149330 | 0.136421 | 0.475945 | 0.389208 |

These maps fail their own actual fitting objective. This is stronger evidence
of surrogate failure than a transfer regression alone. It does not isolate
finite-step nonlinearities, identity-STE error, and curvature-estimation error
from one another. The fixed-budget map improves fitting KL on both models,
but its OPT fitting NLL is essentially flat and its domain transfer is poor.
Teacher agreement alone therefore remains an incomplete success criterion.

## Current-model common descent

Job332316 completed a new frozen diagnostic: one legal bit per update,
recomputing CE and teacher-KL derivatives at the current hard model; both
must favor the flip.256 updates use four documents each, cycling over the
same shared64-document training set. No candidate-loss search or checkpoint
selection is performed. Controls include a stale256-bit map from the original
gradients. Actual fitting losses are audited only after all maps freeze.
See the [frozen protocol](../relinearized_format/PROTOCOL.md).

All three models completed. On OPT the final map reduces fitting CE from
3.239064 to3.221852 and KL from0.149330 to0.127549, but math PPL rises from
24.865662 to25.865252 (paired ΔNLL +0.039413 ±0.021171). This is a supported
transfer regression despite improving both actual fitting objectives. The
frozen method therefore fails, independently of its other improvements.
The reserved confirmation plan is not executed to rescue this failure.

This is an optimization hypothesis, not a claimed universal rule or novel
binary optimizer. Any favorable result must survive a new-model/new-domain
confirmation; repeatedly inspected Wiki/math/code domains are development
domains, even when the exact test rows are fresh.

## Source consensus with a withheld source family

Job332332 tests a different transfer hypothesis. One shared baseline score
table contains64 C4,64 OpenWebMath and64 CodeParrot documents. All maps reuse
that table; no configurations are calibrated separately. Consensus requires
negative CE and KL directional estimates for every included source.

The primary map for Wiki excludes web scores, for math excludes math scores,
and for code excludes code scores. Matched pooled-gradient maps distinguish
the value of source agreement from simply adding data. All-source maps are
secondary and cannot rescue a failed leave-source-out test. Actual fitting
losses on included sources are audited only after all maps freeze. The
[protocol](../consensus_format/PROTOCOL.md) and
[related-work boundaries](../consensus_format/RELATED_WORK.md) state the
limited convex-mixture derivative argument and the prior work explicitly.

Completed results:8/9 baseline improvements, all8 supported by descriptive
paired2SE intervals; no supported harm;9/9 lower point loss than weight-MSE;
all18 included-source fitting CE/KL audits improve. However, consensus beats
the matched pooled-source map in only3/9 cells, failing its prespecified
comparative criterion. See [results](../consensus_format/REPORT_332332.md).
The all-source pooled CONTROL improves9/9 development cells; that is an
exploratory secondary finding, not a retroactive pass for source consensus.

## Description-cost sparsity

Job332344 reused the saved C4 score tables. An independent Bernoulli prior
with switch probability1/(D+1) gives a per-tile relative description cost
log(D), replacing a chosen256-tile cap by a log(D)/(64*511) threshold on the
directional score. This is a particular sparse-prior assumption, not a new
generalization theorem or a uniquely justified prior.

The method improves7/9 cells over FourOverSix, with no supported harm, and
passes all three fitting audits. It nevertheless fails its prespecified
gain-retention requirement in three of five cells where the larger map has
supported baseline gains. Its retained NLL gains are only24.5% on Llama wiki,
39.5% on Llama math, and16.4% on Qwen wiki. See the
[full summary](../description_format/REPORT_332344.md). Harmlessness by severe
under-selection is not treated as success.

## Independent test of the pooled-source control

Job332349 advances the existing pooled-source control under a new frozen
confirmation protocol. It does not alter or rescue the failed consensus
primary study. The original three pooled maps replay exactly; two new model
families, Pythia1.4B and OLMo1B, use the same scoring recipe. Five models are
evaluated on literature, scientific articles, and government reports, whose
losses had not been inspected during method development.

An equal-token control compares64 mixed-source sequences with64 C4 sequences;
the192-sequence primary separately measures the effect of additional data.
All maps use subsets of one shared score table, never a separate calibration
per configuration. See the [frozen protocol](../pooled_confirmation/PROTOCOL.md).
The completed confirmation improves15/15 comparisons over FourOverSix, all15
supported by descriptive paired2SE intervals, and beats weight-MSE in15/15.
Both new-model fitting audits improve. However, it beats C4-only64 in8/15
point estimates, short of the prespecified9; the full screen fails. The
equal-token diversity control improves6/15, with two supported gains and two
supported harms. Transfer of the frozen procedure is supported; superiority
of source diversity is not established. See the
[full report](../pooled_confirmation/REPORT_332349.md) and
[figure](../pooled_confirmation/confirmation_332349.pdf).

## Frozen-map causal activation audit

Job332374 replays the same maps and exact recorded inputs, changing only the
activation tensor factor to one FP32 value per token row. This eliminates
dependence of earlier activation quantization on later tokens. Both baseline
and selected policies use the matched row convention; no map is recalibrated
or chosen. Synthetic tests verify exact equality to independent canonical
FourOverSix calls per row, prefix/batch independence, and zero input handling.
Real-model suffix interventions check prefix logits under both conventions.

This is a causal robustness audit on already-inspected inputs, not another
untouched-domain confirmation or a rescue of the failed C4 comparison. Its
[protocol](../causal_replay/PROTOCOL.md) freezes the gain-retention and prefix
independence requirements before its results are inspected.

Completed audit:15/15 improvements over matched per-token-factor FourOverSix,
14 supported, no supported harm;15/15 lower point loss than weight-MSE;
14/15 retain at least half of the previous convention's NLL gain. All five
models pass exact prefix independence for both baseline and selected maps.
The older window-factor version fails the same future-token intervention
in all five. The declared causal-audit screen passes. See the
[report](../causal_replay/REPORT_332374.md). This does not change the earlier
failed C4-only comparator criterion.

## Fixed-recipe scale transfer

Job332389 tests Qwen3-4B and pinned base Llama3.1-8B, with the same shared192
observations, scores, candidates,2SE filter and256 cap. Scoring remains under
the original window convention; final evaluation uses per-token factors with
no new map or row-specific recalibration. All6 baseline comparisons improve,
all6 have supporting paired2SE intervals, and all6 beat weight-MSE and C4-only
selection in point estimates. Both fitting audits and prefix-independence
checks pass. The declared size-transfer screen passes. See the
[report](../pooled_scale/REPORT_332389.md).

The [consolidated causal result](../transfer_rule/REPORT.md) comprises21 unique
model/domain cells, across seven models from350M to8B. All21 improve over the
matched baseline;20 have supporting paired2SE intervals. The earlier window
evaluations are not counted again as independent evidence. This supports a
common calibrated procedure with no candidate-loss search. It is not a
weight-only universal theorem, a new-invention claim for gradient ranking,
a seed-replication result, or a demonstrated native FP4 kernel.
