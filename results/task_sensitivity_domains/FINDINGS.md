# Cross-domain MixFP4 findings — completed 2026-09-07

**Best tested practical rule: C4 task-gradient selection with measured finite-step
backtracking and independent validation. A universally improving rule was not
established.** The same procedure produced accepted maps for all four model/seed
runs, with at least 0.01 absolute PPL improvement in 15 of 16 held-out cells.
Fourteen cells had paired two-SE intervals supporting improvement; none supported
harm. The remaining cell, Llama code, was +0.0103 PPL with a wide interval
[-0.0565, +0.0778], so it cannot be claimed as an improvement or confirmed harm.

All six experiment reports and the target math/code stress report are complete.
The final [REPORT.md](REPORT.md), [summary.csv](summary.csv),
[contrasts.csv](contrasts.csv), and [comparison.png](comparison.png) contain all
20 candidate maps × 4 evaluation domains, acceptance decisions and uncertainty.
All GPU work and numerical analysis ran through Slurm on H100 allocations.

## Most useful completed result

Absolute candidate PPL change relative to matched native FourOverSix W4A4.
Negative is better; a 0.01 reduction is worthwhile under the user's criterion.
These are C4-calibrated maps, all independently accepted before held-out testing.

| Model / calibration seed | E0M3 tiles | Wiki | C4 | Math text | Code text |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B / 20260912 | 905 | -0.1094 | -0.0500 | -0.0343 | -0.0462 |
| Qwen3.8-27B / 20260913 | 864 | -0.0775 | -0.0532 | -0.0199 | -0.0352 |
| Qwen3-4B / 20260918 | 125 | -1.9695 | -1.1781 | -0.4542 | -0.5216 |
| Llama-3.1-8B / 20260918 | 2,924 | -0.0452 | -0.0546 | -0.0342 | +0.0103 |

The two target seeds use identical held-out examples; they test calibration
repeatability, not independent test populations. Math/code measure full-reference
text LM loss on GSM8K/MBPP, not answer accuracy or pass@k. The Llama math mean
also has an interval including zero. Full intervals are in REPORT.md.

## Rule to use from this evidence

1. Start from FourOverSix E2M1 weights and fixed NVFP4 FourOverSix activations.
   Compare each legal 8×64 weight tile with its fixed alpha=1 E0M3 candidate.
2. On 64 C4 fitting windows, estimate the sequence mean and SE of
   `s_b = <gradient_Wb NLL(W_baseline), Q_E0M3[b] - Q_baseline[b]>`.
3. Keep tiles with `mean(s_b) + 2 SE(s_b) < 0`, ranked by most negative mean.
   Choose the largest ranked prefix whose summed predicted reduction is at most
   0.1 nats/token.
4. Measure the **combined map's actual fitting loss**. Require a negative
   paired mean plus two SE and at least 25% of the predicted reduction. On
   failure, halve the budget, up to eight attempts. This is an extrapolation
   check, not a mathematical loss bound.
5. Accept the resulting frozen map only if a separate 16-window C4 validation
   set passes the paired two-SE improvement gate. Otherwise export the baseline.

The algorithm's constants stay fixed; tile count and final budget adapt to the
model. C4 budgets were 0.0125 for both Qwen3.8 seeds, 0.1 for Qwen3-4B, and
0.025 for Llama. **Do not replace backtracking with a universal 0.0125 budget.**
This is a common calibration procedure, not one fixed map or a weight-only
shape threshold. C4 validation does not certify arbitrary deployment domains.

Accepted files: `seed20260912/c4_export.json`, `seed20260913/c4_export.json`,
`panel/qwen3-4b/c4_export.json`, and
`panel/llama-3.1-8b-local/c4_export.json`. Apply to pristine weights with the
recorded model revision and activation configuration. Use the native Qwen
loader for the target and the generic type-map loader for the panel.

## Hypotheses tested

**Finite-switch forecasts fail even with domain matching — supported.**
On Qwen3.8, initial C4 proposals predicted -0.1 NLL but measured +0.0995 and
+0.0943 on their own fitting sets. Backtracking repaired both, reducing roughly
53,000 switches to about 900. Stable per-tile signs and a fixed summed-gradient
budget are insufficient. This establishes finite-proposal error, without
uniquely separating STE error, curvature and tile interactions.

**Calibration distribution changes useful selections — supported, but inherent
tile conflicts remain unresolved.** Target Wiki/C4 maps shared only 10 and 14
tiles (Jaccard 0.009 and 0.012). Wiki maps improved Wiki by 0.3684/0.4088 PPL,
but changed C4 by +0.0028/+0.0064. C4 maps improved both domains. Thus different
selections are useful, but low overlap is not proof that their optimal effects
must conflict. Thirty isolated selected tiles were tested on reserved probes;
almost all effects were inconclusive and no stable wrong-sign effect was found.

**Pooling domains is enough — not supported on the target.** Mixed calibration
improved Wiki by 0.4509/0.4183 PPL but C4 by only -0.0026/+0.0004. Equal sample
counts do not imply equal influence on the objective. Pooling helped Qwen3-4B,
so this is a model-dependent failure, not a claim that pooling never helps.

**Consensus improves coverage — partly supported, with a substantial acceptance
cost.** Candidate consensus maps gained at least 0.01 PPL in 15/16 cells, but
only 2/4 maps passed independent validation. The rejected second target map
later improved all four held-out means; those gains cannot be credited to its
baseline export. Eight validation windows per domain can miss useful gains.
Consensus does not dominate C4: on the target it trades a larger Wiki gain for
a smaller C4 gain; on Qwen3-4B it gives strong four-domain gains.

**Teacher KL makes selection more stable and universal — not supported.**
At matched 32-window budgets, Qwen3-4B's Wiki/C4 score cosine fell from 0.775
(observed NLL) to 0.350 (teacher KL), while its C4 noise-scale ratio rose from
0.051 to 0.496. Llama alignment improved, but performance did not become
uniformly better. Accepted teacher-consensus on Qwen3-4B increased code PPL by
0.1850, with interval [+0.0298, +0.3426], despite passing Wiki/C4 validation.
Teacher fitting objectives are not substitutes for deployment-domain checks.

**Noise is architecture-dependent — supported descriptively.** Full-budget
Wiki/C4 score cosines were about 0.03–0.04 for Qwen3.8, 0.795 for Qwen3-4B,
and 0.276 for Llama. Within-C4 split cosines were about 0.219, 0.961 and 0.204.
The ratio sum(SE²)/sum(mean²) is only a descriptive noise scale, not a certified
signal fraction or simultaneous test over millions of tiles.

## What remains unproven

There is no demonstrated scalar weight-statistic rule or fixed tile map that
improves every domain. C4 plus backtracking is the best broadly useful tested
default by coverage and export acceptance, not the winner on every metric.
Wiki-only Llama and teacher-consensus Qwen3-4B have clear code regressions;
Wiki/C4 calibration checks do not cover unseen code distributions.

A next experiment would include code in independently split fitting/validation
domains and size validation to resolve 0.01 PPL changes. That is a new hypothesis,
not a verified fix; current test examples must not be reused for its calibration.
The current study is complete, and no thresholds were tuned on held-out results.
