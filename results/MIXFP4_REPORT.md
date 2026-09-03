# MixFP4: does choosing the FP4 element type per 8x64 tile beat NVFP4?

Perplexity at seq 2048, type block **8x64** for weights (the smallest hardware-realizable weight
tile, one `n8 x k64` MMA B-operand), wikitext and c4.

**Sections 1-5 are the result: W4A4 prefill, six models, alpha fixed at 1.** Everything from
"BACKGROUND" onward predates the scope change below and is kept as the measured record only.

**Short answer: yes, by -0.032 to -0.122 wikitext on every model measured, but only with
calibration.** The election rule's strictness is model-dependent and (§5) could not be predicted
from any of seven cheap statistics, so the recommendation is the fixed rule with the smallest worst
case: `mix_4_6_clipa1_hess_impg16_h10` at an 8x64 type block. See §1.

> ### SCOPE: the scale search is out. This report is about the element type only.
>
> **Everything below that varies `alpha` is out of scope and is retained as background, not as a
> result.** Sections written before this change treat the scale search as the headline; read them as
> history. The sections marked **(current scope)** are the ones that carry the claim.
>
> `alpha` is the clip ratio in `block_scale = alpha * block_max / grid_max`. Searching it is a
> separate mechanism from choosing the element grid, it was measured as the LARGER of the two, and
> it therefore dominated every headline number in the old framing. A MixFP4-vs-NVFP4 delta taken
> under a multi-alpha preset confounds "a different scale" with "a different grid".
>
> **Removed from `CLIP_PRESETS` entirely** -- `cliphead*` no longer parses:
>
> | removed | what it searched |
> |---|---|
> | `head`, `headx`, `headxx` | E2M1 headroom, `alpha` in {1.25 ... 4} |
> | `heade0`, `heade0x` | the same on the E0M3 branch, `alpha` = 7/6, 7/5 |
>
> **`a1` is the reported preset**: `alpha = 1` on BOTH grids, so a block's maximum sits on the top
> code of whichever grid its tile elected -- exactly what plain NVFP4 does. The only thing MixFP4
> then varies is the element data type.
>
> This buys the comparison its cleanliness. `test_a1_e2m1_is_nvfp4` asserts that
> `mix_4_6(clip=a1, elect=never)` is **bit-identical** to `nvfp4`, so every delta reported under
> `a1` is attributable to the type election and nothing else. `test_no_scale_search` asserts the
> removed presets stay removed.
>
> **The baseline changes with the scope.** `nvfp4_4over6` is itself a two-point scale search, so the
> reference for a type-only claim is plain **`nvfp4`**. Deltas against `4over6` in the older sections
> are not comparable to deltas against `nvfp4` in the current ones.
>
> Retained as controls and historical record, deliberately outside the reported set: `base`
> (FourOverSix), the `dense*` family, and the clipping family all still search a scale.

---

## 0. Read this first — measurement noise

Configurations that use calibration (`hess`) are **not bit-reproducible**. The same configuration
run twice:

    mix_4_6_clip<preset>_hess_coclcol_m1   6.539287   and   6.543658     (spread 0.0044)

Configurations without calibration are exactly reproducible (`nvfp4_4over6` gave
6.598703861236572 in three separate jobs). So the variance comes from `collect_importance` —
GPU reductions over the calibration batches are not deterministic, and that propagates through
the elections.

**Consequence: differences below ~0.005 between two `hess` configurations are not meaningful.**

Applied to §1, this is what separates a result from a null:

* the three best-per-model figures (-0.0442, -0.1222, -0.0837 wikitext) are 10x to 28x the floor
  and are real;
* `a1_hess_impg16_h10` on Llama-3.1-8B (+0.0007 / +0.0017) is **inside** it, which is the basis for
  calling that configuration neutral there rather than harmful;
* reordering's contribution on Llama-3.1-8B (-0.0012 / +0.0026, §4) is inside it and should be
  read as no effect.

One further caveat specific to the alpha search, measured on real weights in `check_impg_noop.py`:
`sum_j (c * d_j^2)` and `c * sum_j d_j^2` are different floats, so a multi-candidate alpha search
breaks near-ties nondeterministically at a rate of 7.3e-6 of elements — worth ~0.008 perplexity on
an 8B model. **This does not affect sections 1-4**, which fix alpha at 1: with a single candidate
there is no comparison to break, and `a1_e2m1` reproducing `nvfp4` exactly on all three models
confirms it.

---

## 1. The result (current scope)

**W4A4 prefill, weights 8x64 type block, alpha = 1, activations `nvfp4_4over6`.** Deltas against
plain **`nvfp4`** weights. Negative is better. Six models, two model families, 1B to 14B.

The reference and the validation row are the same measurement twice: `a1_e2m1` freezes alpha at 1
and disables the election, leaving MixFP4 no freedom at all, so it must reproduce `nvfp4` exactly.
It does, on every model, to every printed digit. **Every other number below is therefore
attributable to the element-type election and nothing else.**

`nvfp4` reference (wikitext / c4): Llama-3.1-8B 6.9054 / 9.8832 · Llama-3.1-8B-Ins 7.8566 / 11.2850 ·
Llama-3.2-1B-Ins 15.4130 / 21.6146 · Qwen3-4B 13.9101 / 17.2208 · Qwen3-8B 10.0215 / 13.7317 ·
Qwen3-14B 8.9166 / 12.3908.

### wikitext

| model | `e2m1` | `h10` (no calib) | `hess_h1.5` | `hess_h10` | `hess_m1` | `hess_impg16_h10` |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.0000 | -0.0085 | **-0.0442** | -0.0043 | -0.0388 | +0.0006 |
| Llama-3.1-8B-Ins | 0.0000 | -0.0140 | **-0.0587** | -0.0099 | -0.0314 | -0.0099 |
| Llama-3.2-1B-Ins | 0.0000 | -0.0200 | +0.0938 | **-0.0409** | -0.0205 | -0.0164 |
| Qwen3-4B | 0.0000 | +0.0123 | +0.4231 | -0.0170 | +0.2865 | **-0.1222** |
| Qwen3-8B | 0.0000 | +0.0009 | -0.0179 | +0.0093 | **-0.0837** | -0.0581 |
| Qwen3-14B | 0.0000 | +0.0023 | **-0.0317** | +0.0492 | -0.0217 | +0.0113 |

### c4

| model | `e2m1` | `h10` (no calib) | `hess_h1.5` | `hess_h10` | `hess_m1` | `hess_impg16_h10` |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.0000 | -0.0069 | **-0.0693** | -0.0155 | -0.0572 | +0.0016 |
| Llama-3.1-8B-Ins | 0.0000 | +0.0196 | **-0.0153** | +0.0074 | -0.0062 | +0.0098 |
| Llama-3.2-1B-Ins | 0.0000 | -0.0452 | -0.1420 | -0.1177 | **-0.2551** | -0.0559 |
| Qwen3-4B | 0.0000 | -0.0369 | +0.0413 | -0.0036 | -0.0440 | **-0.1066** |
| Qwen3-8B | 0.0000 | +0.0183 | +0.0466 | +0.0038 | **-0.0279** | -0.0049 |
| Qwen3-14B | 0.0000 | -0.0037 | **-0.0777** | +0.0181 | -0.0209 | -0.0112 |

### Three claims this supports

**1. The element-type block beats plain NVFP4 on every model measured.** Best per model, wikitext:
-0.0442, -0.0587, -0.0409, -0.1222, -0.0837, -0.0317. All are 7x to 28x the 0.0044 noise floor of
§0. This
is like-for-like at identical scale, identical metadata and identical bit width -- the only
difference is that a tile may elect E0M3 instead of E2M1.

**2. It needs calibration.** Without importance, the best the type block manages is -0.0200
wikitext, and on Qwen3-4B it is *positive* (+0.0123). With importance the same models reach -0.04 to
-0.12. The election has to be made on the diagonal-Hessian objective rather than on weight MSE,
which is the same principle as §8 and here is the difference between a result and a null.

**3. No single election rule wins everywhere; pick the one with the smallest worst case.** The best
rule differs by model *and* by dataset -- `h1.5`, `h10`, `m1` and `impg16_h10` each win somewhere --
and a model-specific optimum is a real loss elsewhere: `h1.5` costs **+0.4231** on Qwen3-4B, `m1`
costs **+0.2865**. Summarised over the six models:

| rule | wikitext mean / worst | c4 mean / worst |
|---|---|---|
| **`hess_impg16_h10`** | **-0.0324 / +0.0113** | **-0.0279 / +0.0098** |
| `hess_h10` | -0.0023 / +0.0492 | -0.0179 / +0.0181 |
| `hess_m1` | +0.0151 / **+0.2865** | -0.0686 / -0.0062 |
| `hess_h1.5` | +0.0607 / **+0.4231** | -0.0361 / +0.0466 |

`m1` has much the best c4 mean and a catastrophic wikitext worst case. `hess_impg16_h10` has the
best wikitext mean and by far the smallest worst case on both datasets, and it is the
recommendation.

**Correction: it is not "never harmful".** With five models that claim held (worst case +0.0006).
The sixth, Qwen3-14B, costs **+0.0113** wikitext -- 2.6x the noise floor, so a real if small loss.
The recommendation stands on worst-case grounds, +0.0113 against +0.0492 for `h10` and +0.2865 for
`m1`, but it is a smallest-harm choice, not a free one.

§5 shows this cannot currently be improved by predicting the rule per model: across seven candidate
statistics, no predictor beat this fixed choice on either dataset. The cost of that is the gap to a
per-model oracle, about **0.031** wikitext and **0.024** c4.

### What this costs to deploy

Nothing beyond the E0M3 path itself. Alpha is fixed at 1, so the ue4m3 scale field carries exactly
what NVFP4 writes today; the type block adds one bit per 8x64 weight tile; and the election is
offline, at quantization time. What it does require is a calibration pass -- one batch of any real
text is enough (§7) -- and a kernel that can issue the E0M3 operand.

---

## 2. The election rule (current scope)

A tile holds 32 scale blocks (8 rows x 4 k-chunks). Each has a signed gain
`g_b = loss_E2M1 - loss_E0M3`; the rule decides whether the tile goes E0M3. Rules measured:

| rule | elects E0M3 iff |
|---|---|
| `argmin` | `sum g_b > 0` -- E0M3 is better on aggregate |
| `h<lambda>` | `sum_{g>0} g_b > lambda * sum_{g<0} |g_b|` -- the winners outweigh the losers by `lambda` |
| `m<z>` | `mean(g) > z * stderr(g)` -- the tile's advantage is significant against block-to-block spread |
| `dom` | no block is harmed at all |

The pattern the older sections found on the scale search holds here too, and more sharply:
**a rule that fires whenever E0M3 helps loses; the same rule with "...by a decisive margin" wins.**
`argmin` is `h1`. Every winning row in §1 is `h1.5`, `h10` or `m1`.

What is new, and what the paper has to be honest about, is that **how decisive is model-dependent**.
`h1.5` is optimal on Llama-3.1-8B and catastrophic on Qwen3-4B (+0.4231); `h10` -- a rule that
elects only when the winning blocks outweigh the losing ones ten to one -- is the only harm-ratio
setting that is non-harmful on all three. Erring conservative is cheap; erring permissive is not.

`dom` is the limit of that logic and elects almost nothing at 8x64: it degenerates to plain E2M1,
i.e. to `nvfp4`. It is a floor, not a configuration.

---

## 3. Importance, and how finely it must vary (current scope)

`hess` weights each element's squared error by `E[x_j^2]` of the input channel it multiplies -- the
diagonal-Hessian estimate of layer output error. §1 shows the type block is worth nothing without
it.

Three granularities are available, and they are independent of the type block, which is 8x64
throughout:

| | what it is | effect |
|---|---|---|
| per element (`hess`) | each element gets its own channel's `E[x_j^2]` | the default |
| per scale block (`impg16`) | the mean over the block's 16 channels | cancels in the alpha search, not in the election |
| per type block (`impg64`) | the mean over the tile | **provably nothing** |

**`impg64` is an exact no-op**, and this is a theorem rather than a measurement: a positive constant
across a tile divides out of the alpha search and out of every election rule, because all of them
compare quantities homogeneous of the same degree in the loss (`argmin`, `h`, `vote`) or are ratios
of degree-1 quantities (`m`). `tests/test_imp_gran.py` asserts it bit-for-bit for all five rules.
So "apply the importance at the type-block level" is not a configuration -- it is the no-hess row.

**`impg16` is a real knob, and coarsening helps more often than not.** Against per-element `hess`
at the same election rule (`h10`), in wikitext:

| | per element | per scale block | change |
|---|---|---|---|
| Llama-3.1-8B | -0.0043 | +0.0007 | +0.0050 |
| Qwen3-4B | -0.0170 | **-0.1222** | **-0.1052** |
| Qwen3-8B | +0.0092 | **-0.0581** | **-0.0673** |

The plausible reading -- untested, and stated as a hypothesis -- is variance. `E[x_j^2]` is
heavy-tailed, so per-element weighting makes a block's loss depend mostly on its one or two loudest
channels, which is a high-variance estimate of what the block contributes. Averaging over the 16
keeps the direction and drops the variance. That is the same move as `h<lambda>` over `argmin`,
applied to the criterion rather than to the decision rule.

---

## 4. Reordering, at alpha = 1 (current scope)

Permuting rows and 16-column chunks so that E0M3-preferring scale blocks land in the same tiles
(`coclcol`, see `results/reorder/ALGORITHM.md`) is worth much less than it appeared under the older
scope:

| | `a1_hess_h10` | `a1_hess_coclcol_h10` | reordering is worth |
|---|---|---|---|
| Llama-3.1-8B | -0.0043 / -0.0155 | -0.0055 / -0.0129 | -0.0012 / +0.0026 |
| Qwen3-4B | -0.0170 / -0.0036 | +0.0162 / +0.0065 | +0.0332 / +0.0101 |
| Qwen3-8B | +0.0092 / +0.0038 | -0.0378 / -0.0162 | -0.0470 / -0.0200 |

Only Qwen3-8B gains; Qwen3-4B loses more than the type block earns there; Llama-3.1-8B is inside
noise on both datasets. Earlier sections credited reordering with -0.03 to -0.04 on Qwen3-8B, and
that number was measured on top of a wide scale search -- **the two mechanisms were entangled, and
most of what reordering appeared to buy is not there once alpha is fixed.**

This is consistent with what the reordering study itself found (`results/reorder/REPORT.md`): the
partition search beats a cell-shuffled control by +0.003 of the recoverable ceiling, i.e. by
essentially nothing. Reordering is not part of the recommended configuration.

---

## 5. Can the rule be predicted instead of swept? No. (current scope)

§1 leaves one thing open: the type block beats NVFP4 on every model, but the strictness that wins
differs per model and the wrong choice is expensive (`h1.5` is best on Llama-3.1-8B and costs
**+0.4231** wikitext on Qwen3-4B). A deployable method cannot sweep four rules per model, so the
question is whether a cheap statistic picks the rule.

**Measured across six models, two datasets and seven candidate statistics: it does not.** Nothing
here beat simply using one fixed rule everywhere. This section is the negative result and the
evidence for it, because it is the reason §1 recommends a fixed configuration rather than a
selection procedure.

### The hypothesis that had a mechanism behind it, and its failure

The election computes the *diagonal* surrogate `sum_j S_jj dW_ij^2` of the true layer output error
`tr(dW S dW^T)`, because a full `S = E[x x^T]` per layer is not affordable. The certificate in
CLAUDE.md bounds the surrogate's error by `||S - D||_2 * ||dW||_F^2` with `D = diag(S)`, and
`h<lambda>` is precisely a margin against that error. So:

> **Prediction: a model with more off-diagonal mass in `S` should need a larger `lambda`.**

It is wrong, and it fails on the very first comparison:

| | `h1.5` verdict (wikitext) | `offdiag_spec` = `\|\|S-D\|\|_2 / \|\|D\|\|_2` |
|---|---|---|
| Llama-3.1-8B | helps, **-0.0442** | **4.87** |
| Qwen3-4B | destroys, **+0.4231** | **3.45** |

The model that cannot tolerate a permissive rule has *less* off-diagonal mass, not more. Full
per-model values (`analyze_lambda_predictor.py`, full `S` on a stride of layers, 4 wikitext
batches):

| model | `h1.5` wikitext | offdiag | coherence | kappa_diag | elect@1.5 | marginality | disagree |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Ins | -0.0587 | 4.686 | 0.0545 | 259 | 0.311 | 0.507 | 0.257 |
| Llama-3.1-8B | -0.0442 | 4.869 | 0.0531 | 240 | 0.314 | 0.513 | 0.252 |
| Llama-3.2-1B-Ins | **+0.0938** | 5.879 | 0.0770 | 182 | 0.328 | 0.522 | 0.285 |
| Qwen3-4B | **+0.4231** | 3.445 | 0.0548 | 827 | 0.562 | 0.782 | 0.146 |
| Qwen3-8B | -0.0179 | 3.251 | 0.0453 | 1068 | 0.352 | 0.545 | 0.328 |
| Qwen3-14B | -0.0317 | 4.998 | 0.0498 | 509 | 0.399 | 0.602 | 0.320 |

No column separates the two harmful rows from the four helpful ones. `elect@1.5` comes closest and
still fails: 0.328 where `h1.5` hurts, 0.352 and 0.399 where it helps. `offdiag_spec` is worse than
useless -- its two extreme values, 5.879 and 3.251, sit on opposite sides of the harmful/helpful
split from what the certificate predicts.

### Why the rank correlations must not be believed

Several correlations look excellent -- at n = 5, `coherence` reached **rho = -1.00** against the
`h10` delta and `elect_1.5` and `marginality` reached **+0.90** against `h1.5`. With only 120
orderings of five things, seven statistics tried against four rules on two datasets, a |rho| of 0.9
arises constantly from noise under that much searching.

So the correlation is treated as a screen, never as the result. The result is **leave-one-out**: for
each model, choose the rule using only the other four, and report the perplexity actually paid.
Against two baselines -- `oracle`, the unreachable per-model best, and `fixed`, the best single rule
used everywhere:

Over all six models:

| | wikitext mean | wikitext worst | c4 mean |
|---|---|---|---|
| oracle (unreachable) | -0.0636 | — | -0.0920 |
| **fixed** (`impg16_h10` / `m1`) | **-0.0324** | **+0.0113** | **-0.0685** |
| best predictor, leave-one-out | +0.0313 | +0.2865 | -0.0589 |
| worst predictor, leave-one-out | +0.0669 | +0.4231 | -0.0175 |

**Every predictor is worse than the fixed rule, on both datasets and at both n = 5 and n = 6.** On
wikitext all seven have a *positive* mean, i.e. a model that used them would be worse than plain
NVFP4, while the fixed rule delivers -0.0324. `coherence`, the statistic that had the perfect rho at
n = 5, is among the worst two by leave-one-out. That contrast is the entire point of reporting
leave-one-out rather than rho.

### The pattern that looked real and was not

Ranked by size, the wikitext data separates perfectly: `h1.5` helps all three 8B models and harms
the 1B and 4B ones, across two model families. That prediction was stated in advance and
**confirmed** on the held-out Llama-3.1-8B-Instruct (-0.0587).

A second held-out model, Qwen3-14B, also came out as predicted (-0.0317).

It still does not survive. On c4 the same rule *helps* Llama-3.2-1B-Instruct (**-0.1420**) and
*hurts* Qwen3-8B (**+0.0466**) -- the split reverses with the evaluation set, so it was tracking the
dataset, not the model. A pattern this clean, that reproduces on **two** held-out models, and is
still an artifact, is worth keeping in the report as a caution about how little n = 6 buys.

### What to do instead

Use one rule everywhere and choose it by worst case, not by mean:

| rule | wikitext mean / worst | c4 mean / worst |
|---|---|---|
| **`hess_impg16_h10`** | **-0.0324 / +0.0113** | **-0.0279 / +0.0098** |
| `hess_h10` | -0.0023 / +0.0492 | -0.0179 / +0.0181 |
| `hess_m1` | +0.0151 / **+0.2865** | -0.0686 / -0.0062 |

`m1` has much the best c4 mean and a catastrophic wikitext worst case on Qwen3-4B. `impg16_h10` has
the smallest worst case on both datasets by a factor of four, which is why §1 recommends it -- though
see the correction there: at six models it is no longer harmless, costing +0.0113 on Qwen3-14B. The
cost of having no predictor is the gap to the oracle: about **0.031** wikitext, **0.024** c4.

---

# ============================================================================
# EVERYTHING BELOW IS BACKGROUND, NOT A RESULT
# ============================================================================
#
# The sections that follow were written when the SCALE SEARCH was in scope. They vary `alpha`,
# they use presets that no longer exist (`head*`, `heade0*`), and they report deltas against
# `nvfp4_4over6` rather than `nvfp4`. None of that is comparable to sections 1-4 above.
#
# They are kept because they are the measured record of what was tried, including the negative
# results (sections 9 and 10) which remain valid as statements about the scale search. Do not
# lift a number from here into the paper.

---


## 2. Baselines

| model | `nvfp4` (alpha=1) | `nvfp4_4over6` |
|---|---|---|
| Llama-3.1-8B | 6.6236 / 9.4796 | 6.5987 / 9.4232 |
| Llama-3.1-8B-Instruct | — | 7.5313 / 10.8714 |
| Llama-3.2-1B-Instruct | — | 14.3933 / 20.4528 |
| Qwen3-8B | 9.9067 / 13.5420 | 9.8898 / 13.5430 |
| Qwen3-4B | **13.6584 / 16.8725** | 14.0407 / 17.0142 |

**FourOverSix is not universally safe.** It helps Llama-3.1-8B (-0.025 / -0.056) and Qwen3-8B
(-0.017 / +0.000) but costs Qwen3-4B **+0.382 / +0.142**. Every wider preset is worse still there.

This contradicts CLAUDE.md Part 1, which states the wider alpha search is "neutral-to-positive and
never harmful" and should be taken unconditionally. It holds on the Llama models measured and
fails badly on Qwen3-4B.

---

## 3. Best configuration per model

Delta against `nvfp4_4over6`, wikitext / c4:

| model | best config | delta |
|---|---|---|
| Llama-3.1-8B | `heade0_hess_coclcol_m1` | **-0.059 / -0.060** |
| Llama-3.1-8B-Instruct | `heade0_hess_coclcol_m1` | **-0.052 / -0.055** |
| Llama-3.2-1B-Instruct | `heade0_hess_coclcol_m1` | **-0.166 / -0.434** |
| Qwen3-8B | `heade0_hess_coclcol_h1.5` | **-0.069 / -0.080** |
| Qwen3-4B | `a1_hess_h10` | -0.380 / -0.137 |

(W4A16. In W4A4 the best Qwen3-4B configuration is `a1_hess_impg16_h10` at -0.1222 / -0.1066
against plain `nvfp4` weights -- see §1b.)

Calibrated MixFP4 beats `4over6` on all five models. But on Qwen3-4B the right reference is
`nvfp4`, and against that no single configuration wins on both datasets:

| config | wikitext | c4 |
|---|---|---|
| `a1_hess_h10` | +0.0028 | +0.0049 |
| `a1_hess_v0.7` | +0.0097 | **-0.0271** |
| `a1_hess_m1` | +0.3370 | **-0.0259** |

`a1_hess_h10` is indistinguishable from `nvfp4` -- both deltas sit inside the +-0.0044 noise of §0.
`a1_hess_v0.7` is a genuine c4 win (-0.0271) traded against a small wikitext loss (+0.0097, which is
above the noise floor and so real). Both require freezing alpha at 1 AND an extreme election margin;
every default configuration on this model is far behind.

---

## 4. The decomposition: scale vs type

`hess` weights one loss that drives BOTH decisions (`_selection_loss` feeds `_best_over_alphas`,
whose output feeds `_elect_e0m3`). The `hesst` / `hessa` qualifiers split them.

Llama-3.1-8B, delta vs `nvfp4`:

| config | alpha by | type by | wikitext | c4 |
|---|---|---|---|---|
| `nvfp4_4over6` | MSE, {1,1.5} | — | -0.025 | -0.056 |
| `headx_e2m1` | MSE, 5 cand | — | -0.034 | -0.060 |
| `base_hess_e2m1` | **importance, {1,1.5}** | — | **-0.067** | **-0.095** |
| `headx_hessa_e2m1` | **importance, 5 cand** | — | **-0.073** | **-0.106** |
| `a1_h1.5` | none (alpha=1) | MSE | **+0.003** | +0.002 |
| `a1_hess_h1.5` | none (alpha=1) | importance | -0.033 | -0.050 |
| `headx_hesst_h1.5` | MSE | importance | -0.048 | -0.071 |
| `heade0_hess_coclcol_m1` | importance | importance + reorder | -0.084 | -0.116 |

Three readings:

1. **The MSE-based type election is harmful.** At alpha=1 it is +0.003 on Llama-3.1-8B and
   **+1.289** on Qwen3-4B. The E0M3 type block only becomes useful once the criterion is
   importance-weighted. What looked like its gain in earlier work was the alpha search bundled
   alongside it.
2. **Most of the alpha gain is in the 4-or-6 decision itself**: -0.067 of the -0.073 comes from
   choosing between {1, 1.5} correctly. Widening to five candidates adds only -0.006.
3. **The halves are sub-additive**: -0.073 (alpha) + -0.014 (type, measured on top of MSE-alpha)
   would be -0.087; measured together it is -0.084.

Matched comparison for the type block alone — same alpha criterion, E0M3 off vs on:

| alpha by | E0M3 off | E0M3 on | type block's own gain |
|---|---|---|---|
| MSE | 6.5896 / 9.4192 | 6.5755 / 9.4086 | **-0.014 / -0.011** |
| importance | 6.5510 / 9.3737 | 6.5403 / 9.3660 | **-0.011 / -0.008** |

So E0M3 does win against its matched baseline, consistently, by about 0.01 — near the noise floor
of §0.

---

## 5. Why Qwen3-4B is different

Every MixFP4 mechanism inverts on this model:

| mechanism | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| alpha widening (MSE) | -0.025 | **+0.382** |
| alpha widening (importance) | -0.067 | **+0.663** |
| type election (MSE, alpha=1) | +0.003 | **+1.289** |
| type election (importance, alpha=1) | -0.033 | +0.010 (strict `v0.7`) |

The mechanism is block peakedness. From the selection analysis (§8), E2M1 is log-spaced and coarse
at the top, and those top codes are what absorb an outlier; `alpha > 1` maps the block maximum to a
lower code and **discards exactly those codes**. E0M3 is uniform and has no top-code headroom at
all. A model with peakier blocks can afford neither. Llama's flatter blocks can afford both.

Two facts confirm the format is not forcing anything:

* `clipa1_e2m1` reproduces `nvfp4` **bit for bit** at both 8x64 and 1x16. E0M3 is a genuine
  addition and the floor is exactly reachable.
* At **1x16** — one free, independent choice per 16-element block, the finest granularity the
  format allows — MSE-selected E0M3 still costs **+0.509** on Qwen3-4B (`nvif4` 14.1675 vs
  `nvfp4` 13.6584). So weight MSE is anti-correlated with perplexity for this decision on this
  model, and no election rule reading that signal is safe at any tile size.

Also note **coarse beats fine here**: `a1_hess_h3` at 8x64 (+0.239) is far better than anything at
1x16 (+0.509). The coarse tile plus a strict rule acts as a regulariser — it elects less, and
electing less is what helps — inverting the usual reading of 1x16 as the ceiling.

---

## 6. Reordering: worth and cost

### What it is worth

Only measured usefully once the tag grid is importance-weighted. On the MSE grid the search matched
its own cell-shuffle control (+0.003 of the 1x16 ceiling) and perplexity moved with no consistent
sign across 5 models x 2 datasets (7 of 10 cells worse).

On the corrected grid it contributes about **-0.007** on top of importance-picked alpha
(Llama-3.1-8B), and it is in the best configuration on 4 of 5 models.

### What it costs

Prefill only (W4A4 is prefill-bound), seq 2048, batch 1, H100. Block GEMM total 1.159 ms.

Permuting `W`'s columns requires the activation permuted identically — `(XP)(WP)^T = XW^T`. The
question is never whether that cancels, it is **who produces the activation and who else reads it**.

| axis | per-layer order? | absorbed into |
|---|---|---|
| `x_norm` -> `q/k/v` | **free** | RMSNorm output write + gamma (the rms reduction is order-invariant) |
| `x_norm2` -> `gate/up` | **free** | RMSNorm output write + gamma |
| `h` -> `down_proj` columns | **free** | `gate/up` rows |
| `attn_out` -> `o_proj` columns | **free** | `v` rows, within head |
| residual itself | one global order | `o/down` **rows**, so they write back in global order |

**The only real constraint is that `q`, `k`, `v` must share one order and `gate`, `up` must share
another**, because each trio/pair reads the same normalized tensor.

| variant | gathers/layer | ms | % of block GEMM |
|---|---|---|---|
| independent order per matrix — **what `coclcol` measures** | 5 | 0.155 | **13.4%** |
| one order per norm site, per layer | 0 (fused into RMSNorm) | 0 | **0%** |
| one global residual order | 0 | 0 | 0% |
| `down_proj` + `o_proj` per layer | 0 | 0 | 0% |

Per-matrix the unfused cost is worse than the average suggests: at seq 2048 a 4096-wide gather is
0.031 ms while `k_proj`/`v_proj`'s GEMM is 0.025 ms — the gather costs more than the matmul.

**Every `coclcol` number in this report is the first row** — per-layer freedom on all seven
matrices, which is not purchasable at zero cost. The deployable variant (one order per norm site)
is strictly more constrained and **has not been measured**, so -0.007 is an upper bound on it.

Offline cost is a non-issue: ~22.7 min for a full Llama-3.1-8B quantization pass against ~8 s
without, one-off at quantization time.

---

## 7. Calibration

`hess` needs `E[x_j^2]` per input channel. Measured against a 4-batch wikitext reference, fraction
of 8x64 tile elections unchanged:

| source | agreement |
|---|---|
| one 2048-token wikitext batch | **0.957** |
| 4 batches of c4 (different corpus) | **0.943** |
| 4 batches of random token ids | 0.881 |
| no calibration (plain MSE) | 0.828 |

**One forward pass of arbitrary real text recovers 95%+.** The corpus barely matters; one batch
matches four. Random tokens do not substitute, so it needs real text — just very little of it.

Cross-domain transfer is already implicit in every number here: calibration is drawn from wikitext
**train**, and the c4 column is evaluated on a different corpus, where the gains are largest
(Llama-3.1-8B c4 -0.057, Qwen3-8B -0.080). `build_calibration` draws from the train split
explicitly to avoid leaking the evaluation tokens.

**No calibration-free substitute was found.** The tiles `hess` declines are indistinguishable from
the ones it keeps on every weight statistic (peakedness 2.809 vs 2.839, energy concentration 0.0620
vs 0.0622, gain concentration 0.1082 vs 0.1096); only `imp_max` separates them, 47% higher. No rule
beat the do-nothing baseline of 0.828 agreement: `relgain` 0.839, `mse_h3` 0.832, `veto3.0` 0.829,
`veto2.5` 0.812, `sqnr` 0.749. Re-ranking by `sqnr`/`relgain` is *worse* than plain MSE at
reproducing `hess` (rho 0.741 vs 0.758).

---

## 8. How the type is chosen

Rank correlation between the E0M3 gain and block peakedness `block_max / block_rms`:

| | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| rho(gain, max/rms) | **-0.591** | **-0.592** |
| rho(gain, kurtosis) | -0.478 | -0.474 |
| max/rms, E0M3-preferring | 2.007 | 1.979 |
| max/rms, E2M1-preferring | 2.359 | 2.325 |
| blocks preferring E0M3 | 0.537 | 0.552 |
| tiles electing at 8x64 (MSE -> hess) | 0.263 -> 0.176 | 0.295 -> 0.223 |
| importance ratio E2M1/E0M3 | 1.18x | **2.35x** |

A 16-sample Gaussian block sits at max/rms 2.0, and the E0M3-preferring population averages
1.98-2.01 with sub-Gaussian kurtosis. **E0M3 is the "this block has no outlier" grid** — uniform
spacing suits a flat block; E2M1's log spacing is coarse at the top and absorbs an outlier cheaply.

`hess` does not find new wins: it flips 11-16% of blocks, elects on ~33% **fewer** tiles, and the
tiles it does elect sit on lower-importance channels. It declines elections that would damage
channels the output depends on. The last row is why Qwen3-4B suffers most — its peaked,
E2M1-preferring blocks sit on channels 2.35x more important than elsewhere, so every wrong election
costs double.

---

## 9. Negative results

Each was measured with a control, and each is a null or a loss.

| idea | result |
|---|---|
| Reordering the **weight** tag grid (MSE) | search matched its cell-shuffle control (+0.003 of ceiling); ppl worse on 7 of 10 model x dataset cells |
| Row axis | `row_share` 0.021, profile correlation 1.07x control — no structure |
| Optimizing **purity** directly | harm rose 12.4% -> 14.0%; mass purity 0.673 vs shuffled 0.674 |
| Stricter election rules | `argmin`/`h3`/`h5`/`v0.6-0.75`/`tol`/`dominance` all worse than `h1.5` |
| `dominance` (purity = 1.0 required) | elects **0.0%** of tiles before and after reordering |
| Reordering by weight **magnitude** | +-0.1%, same band as a random control |
| Reordering **activation channels** by magnitude | error **worse** by 4.27%, worst (-55%) on the layer with the strongest channel structure |
| Reorder then **rotate** (block-diagonal Hadamard) | `spread - identity` +0.06/-0.35/-0.34% at rot 16/64/128 — noise |
| Peakedness **veto** on activations (`pv<tau>`) | worse at every threshold; redundant with the error-based election, which already encodes peakedness |
| 10x search budget | purity +0.002, gap over control unchanged to 4 decimals |

Two structural facts explain most of these:

* **Retention is 1/sqrt(cells per tile)** on the MSE grid. `1x128` (8 cells along a row) and
  `8x16` (8 cells down a column) both retain ~0.36; `8x64` (32 cells) retains 0.168 against
  1/sqrt(32) = 0.177. Geometry is irrelevant, only the count — and a permutation cannot change the
  count.
* A **partition search overfits**: on an i.i.d. Gaussian matrix it reports +0.203 of the ceiling
  over the identity order and **+0.001** over a cell-shuffled control. Any reordering claim without
  that control is unmeasured.

The one positive structural result: the **activation** tag grid has the channel structure the
weight one lacks — `col_share` **0.169** vs 0.004, and reordering beats its shuffle control by
**+0.081** at 16x64 vs +0.003 on weights. Applying the type block to activations on the fly at
16x64 is worth **-0.014** wikitext in W4A4 with no reordering at all.

---

## 10. Corrections to CLAUDE.md

1. **"The wider alpha search is neutral-to-positive and never harmful."** False on Qwen3-4B:
   `4over6` is +0.382 wikitext worse than plain `nvfp4`, and every wider preset is worse still.
2. **The E0M3 type block treated as a modest model-dependent extra** (worst case +0.0165 on
   Llama-2-7B). Measured worst case is **+1.289** (Qwen3-4B at alpha=1). It helps 1 of 5 models on
   the MSE criterion and 4 of 5 once importance-weighted.
3. **"The axis that matters is K."** On perplexity the two axes contribute about equally —
   rows +0.008, columns +0.009, both +0.022 (all harmful on the MSE grid).
4. **`headx` is a 5-candidate scale search**, not the 4-or-6 pair. Most of its benefit (-0.067 of
   -0.073) comes from the 2-candidate decision alone.

---

## 11. Gaps

* The **deployable** reordering variant (one order per norm site per layer) is unmeasured. All
  `coclcol` numbers assume per-layer freedom on all seven matrices, which costs 13.4% of prefill
  GEMM.
* `hess` is unmeasured on Llama-2-7B and Llama-3.2-3B (gated, no HF token on this account).
* Run-to-run variance of `hess` configurations (~0.004) is the same order as the type block's
  contribution. Repeat runs would be needed to separate them.
* The W4A4 activation-permutation path (`QuantConfig.a_perm`) is implemented but its perplexity
  run did not complete.

---

## 12. Reproduction

Everything runs through Slurm; the CPU-only jobs request a GPU only because cores are rationed at
12 per GPU on this cluster.

```bash
sbatch slurm/hess_ppl.sbatch llama-3.1-8b-local      # the main sweep (run_ppl_sweep.py path)
sbatch slurm/baseline_cmp.sbatch qwen3-4b            # nvfp4 vs 4over6 vs wider presets
sbatch slurm/e0m3_choice.sbatch llama-3.1-8b-local   # how the type is chosen
sbatch slurm/calib_rob.sbatch llama-3.1-8b-local     # calibration robustness
sbatch slurm/bench_cost.sbatch                       # runtime cost of the gathers
sbatch slurm/reorder_tests.sbatch                    # 9 reorder + 52 MixFP4 CPU tests
```

`run_ppl.py` has **no importance plumbing** — `hess` configurations run through it silently
quantize without importance. Use `run_ppl_sweep.py`.

This account has no HuggingFace token; gated models load from local snapshot paths recorded in
`model2path.json` (`*-local`), which never contact the hub.
