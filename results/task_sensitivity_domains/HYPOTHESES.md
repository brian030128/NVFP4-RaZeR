# Hypotheses and a candidate common rule

## Evidence used to choose the experiments

The recent FourOverSix report shows large WikiText gains but inconclusive C4
changes on Qwen3.8-27B. The earlier task-sensitivity report contains a Llama
seed regression even after expanding calibration; finite-step fit backtracking
repairs those development failures. The mechanism report shows nonadditive
tile effects, baseline dependence, and useful information in both input-column
location and output-row identity. The historical DECIDE_SUMMARY and rounds
18a/b concern additional scale searches; they do not establish a type-only
solution. Their reconstruction-MSE results cannot be substituted for the
current task-loss experiment.

Relevant external evidence also argues against assuming domain matching alone
solves calibration: a benchmark across more than 40 datasets found that matching
calibration and test distributions was not always optimal ([Liu et al., 2024](https://arxiv.org/abs/2406.12928)).
Outlier/calibration sensitivity also varied substantially across tested model
families ([Paglieri et al., 2024](https://arxiv.org/abs/2405.20835)).
These papers motivate controls; they are not measurements of our E0M3 rule.

## Competing explanations

1. **Different useful corrections across domains.** A tile's task-gradient
   score depends on both internal inputs and downstream prediction errors.
   Stable opposite signs across WikiText and C4, confirmed by isolated
   interventions, would support this explanation. C4-only maps should then
   outperform transferred WikiText maps on independent C4 examples.
2. **The score is an inaccurate finite-switch predictor.** Identity STE
   derivatives through discrete activation rounding need not predict a full
   E2M1-to-E0M3 switch. Even without domain conflict, summed effects can fail
   because switches interact. Wrong-sign isolated interventions implicate the
   surrogate; poor joint fit actual/predicted ratios establish finite-proposal
   misprediction without uniquely identifying its cause.
3. **Complementary corrections survive pooling.** Equal-domain calibration
   may find useful switches missing from a WikiText-only map. Mixed uses the
   same total fit budget, so any improvement is not from doubling examples.
4. **Worst-domain checks improve transfer.** Requiring stable negative scores
   and actual joint improvements separately in each calibration domain may
   reduce observed worst-domain regressions. It could also be too restrictive:
   individually conflicting tiles can jointly complement one another. An
   empty/fallback map is not a useful-gain result.

## Candidate selection rule, stated precisely

Keep fixed E2M1 and E0M3 candidate weights for each legal 8x64 tile. For every
declared calibration domain d, estimate

    s[d,b] = E_d[ <gradient_Wb NLL(W_baseline), Q0[b]-Qbaseline[b]> ].

Consensus eligibility is `max_d(mean[d,b] + 2*SE[d,b]) < 0`. Rank eligible
tiles by `max_d mean[d,b]`, most negative first. Take a prefix with summed
negative worst-domain scores no greater than 0.1. Evaluate its ACTUAL joint
NLL change on each fitting domain; require mean+2SE < 0 and at least 25%
of that domain's predicted gain. Halve the budget on failure, up to eight
attempts. One separate per-domain validation check accepts the frozen map;
otherwise export all-E2M1 with the baseline scaling.

The worst-domain score budget does not bound the magnitude of summed scores
in every other domain, nor the actual NLL. Per-domain finite-step checks are
therefore essential. All constants remain the same across the model panel.

## What could generalize mathematically

If a fixed map's TRUE expected loss change is nonpositive on each declared
domain D_d, it is nonpositive on every fixed mixture of those domains:

    Delta(sum_d pi_d D_d) = sum_d pi_d Delta(D_d) <= 0.

This is an exact expectation identity for pi_d >= 0 and sum pi_d = 1. It
does not require each individual tile to be beneficial; the joint map is
what matters. It does not cover a new domain outside that mixture family.
Our finite sample two-SE gates do not prove the true expectations satisfy
the premise. The pre-existing GUARANTEE.md explains why arbitrary-domain
log-loss dominance cannot be guaranteed by any changed fixed predictor.

The empirical goal is a common, useful calibration procedure that repeats
across declared workloads. The experiments must establish whether this
candidate achieves that, rather than assuming the word “consensus” makes it
safe or universal.

Fitting-score diagnostics additionally report global score-vector cosine,
point-sign agreement, stable negative intersections and stable opposite-sign
counts, plus per-projection cosine. These are descriptive comparisons, not
simultaneous statistical tests over millions of tiles. They do not update maps.

Exploratory diagnostic added after inspecting the first 32+32 score comparison:
a low cross-domain cosine may reflect noisy gradients rather than stable domain
conflict. Compare the two disjoint C4 32-window halves with each other, and
report sum(SE squared)/sum(mean squared) per domain. The latter uses the
estimated variance of each sample mean as a noise scale; it is not a certified
signal fraction for correlated windows. No selection rule changes in response.
