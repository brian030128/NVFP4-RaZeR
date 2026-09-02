# MixFP4: what helps, what doesn't, and what it costs

Perplexity at seq 2048, type block **8x64** for weights (the smallest hardware-realizable weight
tile, one `n8 x k64` MMA B-operand). Five models, wikitext and c4.

**Sections 1-11 are W4A16 (weight-only).** Section 1a gives the W4A4 results, which are the
deployment-relevant ones and where several conclusions differ. W4A4 activations are pinned at
`nvfp4_4over6` throughout so that only the weight side varies.

> ### Scope change: E0M3 headroom is out
>
> **`heade0` / `heade0x` have been removed from `CLIP_PRESETS`.** Those presets gave the E0M3
> branch its own scale candidates (`alpha` = 7/6, 7/5), which entangles the element-type decision
> with a second scale search on that branch. That is not a factor this work claims, so it is gone
> from the code, not merely unused: `test_no_e0m3_headroom` fails if it returns.
>
> **`headx` is now the wide-search preset** -- the same E2M1 candidates `{1, 1.25, 1.5, 2, 3}`, with
> E0M3 pinned at `alpha = 1`. Every `clipheade0_*` configuration has been re-measured as
> `clipheadx_*`, and the tables below report the `headx` numbers.
>
> What this costs, measured rather than assumed: on Llama-3.1-8B W4A4 the E0M3 headroom was worth
> about **-0.009 / -0.011** (§1a). On Qwen3-8B it was worth nothing. It is a real but small effect,
> and removing it does not change any qualitative conclusion in this report.
>
> Unaffected: every result built on `a1`, `base` or `headx` -- which includes the headline Qwen3-4B
> win (§1a) and the whole of §1b -- because none of those presets ever gave E0M3 headroom.
>
> Kept deliberately: `dense9e0`, `dense9sym`, `dense5sym`, `basesym`, `wide` and `full` still give
> E0M3 `alpha > 1`. They exist as confound controls for "does E0M3 stop contributing once alpha is
> searched", and none appears in a reported result. Do not promote one into a headline row.

---

## 0. Read this first — measurement noise

Configurations that use calibration (`hess`) are **not bit-reproducible**. The same configuration
run twice:

    mix_4_6_clipheade0_hess_coclcol_m1   6.539287   and   6.543658     (spread 0.0044)

Configurations without calibration are exactly reproducible (`nvfp4_4over6` gave
6.598703861236572 in three separate jobs). So the variance comes from `collect_importance` —
GPU reductions over the calibration batches are not deterministic, and that propagates through
the elections.

**Consequence: differences below ~0.005 between two `hess` configurations are not meaningful.**
This matters for two headline numbers below — the E0M3 type block's marginal contribution
(-0.011) and reordering's contribution (-0.007) are only 2.5x and 1.5x this noise floor. They
are reported as measured, but a single run should not be trusted to that precision.

---

## 1. Headline

| lever | worth (Llama-3.1-8B wikitext) | needs |
|---|---|---|
| **alpha chosen by importance instead of MSE** | **-0.042** | nothing — stock NVFP4 kernel |
| widening alpha from {1,1.5} to 5 candidates | -0.006 | nothing |
| E0M3 type block (election by importance) | -0.011 | E0M3 hardware path + type block |
| reordering on top | -0.007 | offline search; free only if constrained (§6) |

**The scale search dominates.** Choosing between FourOverSix's existing two candidates with an
importance-weighted criterion is worth about four times the entire E0M3 type block, and it
requires no format change at all: alpha only changes the value written into the ue4m3 scale
field that already exists.

---

## 1a. W4A4 (prefill) — activations fixed at `nvfp4_4over6`

The section that matters for deployment. Only the WEIGHT configuration varies; activations are
`nvfp4_4over6` in every row so the comparison is like for like.

Note `quant_act` never receives `importance` -- the `use_importance` field is discarded in its
dispatch -- so **calibration applies to weights only**. A `hess` qualifier on an activation dtype is
a silent no-op.

### Llama-3.1-8B

| weight config | wikitext / c4 | vs `4over6` | vs `nvfp4` |
|---|---|---|---|
| `nvfp4` (both operands plain) | 6.9454 / 9.9339 | +0.0699 / +0.1077 | — |
| `nvfp4_4over6` (both) | 6.8755 / 9.8262 | — | -0.0699 / -0.1077 |
| `base_hess_e2m1` (importance-alpha, **no type block**) | 6.8323 / 9.7791 | **-0.0432 / -0.0471** | -0.1131 / -0.1548 |
| `heade0_hess_m1` | 6.8084 / 9.7569 | **-0.0671 / -0.0693** | -0.1369 / -0.1770 |
| `heade0_hess_coclcol_m1` | 6.8133 / 9.7518 | -0.0623 / -0.0743 | -0.1321 / -0.1821 |

Two differences from the W4A16 picture:

* **The weight-side alpha gain survives activation quantization unchanged**: -0.0432 / -0.0471 in
  W4A4 against -0.0419 / -0.0391 in W4A16.
* **The type block is worth about twice as much here.** It adds -0.0239 on top of importance-alpha
  (W4A4) against -0.011 (W4A16). So §1's ranking -- "the scale search dominates, the type block is a
  small increment" -- weakens under W4A4: the type block roughly doubles in relative value.
* Reordering is mixed and inside the noise floor of §0: worse on wikitext (-0.0623 vs -0.0671),
  better on c4.

### Qwen3-8B

| weight config | wikitext / c4 | vs `4over6` |
|---|---|---|
| `nvfp4` (both plain) | 10.0654 / 13.7727 | +0.0311 / +0.0156 |
| `nvfp4_4over6` (both) | 10.0342 / 13.7571 | — |
| `base_hess_e2m1` (importance-alpha, no type block) | 9.9719 / 13.7244 | -0.0623 / -0.0327 |
| `heade0_hess_m1` | 9.9918 / 13.7151 | -0.0424 / -0.0420 |
| **`heade0_hess_coclcol_m1`** | **9.9634 / 13.6740** | **-0.0709 / -0.0831** |

Reordering adds **-0.0285 / -0.0411** on top of `heade0_hess_m1` here -- far more than on
Llama-3.1-8B, and well clear of the noise floor.

**Correction (§1b).** The table above has no no-importance control, and one measured later shows
`heade0_m1` without importance at 9.9592 / 13.7108, i.e. **-0.0750 / -0.0463** -- better than
`heade0_hess_m1`. So on Qwen3-8B in W4A4, calibration *costs* +0.0326 wikitext rather than earning
the -0.0424 this table implies, and the honest reading of the reordering row is that it recovers
roughly what `hess` gave away. `hess` remains a clear gain on Llama-3.1-8B (§1b).

### Qwen3-4B

With the `heade0` presets, the same inversion as W4A16 -- plain `nvfp4` on both operands beats every
added mechanism.

| weight config | wikitext / c4 | vs `4over6` | vs `nvfp4` |
|---|---|---|---|
| `nvfp4` (both plain) | **13.9488 / 17.2751** | -0.3202 / -0.0409 | — |
| `nvfp4_4over6` (both) | 14.2691 / 17.3160 | — | +0.3202 / +0.0409 |
| `base_hess_e2m1` | 14.4448 / 17.3682 | +0.1757 / +0.0522 | +0.4960 / +0.0931 |
| `heade0_hess_m1` | 14.8122 / 17.5409 | +0.5431 / +0.2249 | +0.8634 / +0.2659 |
| `heade0_hess_coclcol_m1` | 14.8634 / 17.5491 | +0.5944 / +0.2331 | +0.9146 / +0.2740 |

`4over6` costs +0.3202 wikitext here, closely matching the +0.3823 it costs in W4A16, so the
alpha-widening defect is a property of the weights and is not changed by quantizing activations.

### Qwen3-4B with alpha = 1 -- MixFP4 BEATS nvfp4 here

The `heade0` rows above use a wide alpha search, which is itself the defect on this model. Freezing
alpha at 1 and electing with a strict rule, activations still `nvfp4_4over6`:

| weight config | wikitext / c4 | vs `nvfp4` weights |
|---|---|---|
| `nvfp4` weights | 13.9101 / 17.2208 | — |
| `a1_e2m1` | 13.9101 / 17.2208 | 0.0000 / 0.0000 (validation) |
| `a1_h10` (no importance) | 13.9223 / 17.1839 | +0.0122 / -0.0369 |
| `a1_hess_impg64_h10` | 13.9223 / 17.1839 | +0.0122 / -0.0369 (no-op, see §1b) |
| `a1_hess_h10` (importance per element) | 13.8931 / 17.2172 | -0.0170 / -0.0036 |
| **`a1_hess_impg16_h10`** (importance per scale block) | **13.7879 / 17.1142** | **-0.1222 / -0.1066** |
| `a1_hess_v0.7` | 13.9129 / 17.1849 | +0.0028 / **-0.0359** |

**MixFP4 beats plain NVFP4 on Qwen3-4B in W4A4**, where the best available in W4A16 was parity.
The margin depends strongly on the importance granularity: per-element importance gives -0.0170,
and **coarsening it to one weight per 1x16 scale block gives -0.1222 / -0.1066** -- 7x more, and
~28x the §0 noise floor. See §1b.

### Not yet measured in W4A4

* W4A4 runs are far slower than W4A16 because `quant_act` executes on every forward pass; one
  7-config sweep took over 3.5 hours, and a 2-hour `dev` allocation timed out.

### The alpha-vs-type decomposition under W4A4 (Llama-3.1-8B)

The ladder of §4, re-measured in W4A4 rather than W4A16. All rows are 8x64 weights with
`nvfp4_4over6` activations, against `nvfp4_4over6` weights (6.8755 / 9.8262):

| weight config | what it isolates | wikitext / c4 | delta |
|---|---|---|---|
| `nvfp4` | no alpha search, no type block | 6.9054 / 9.8832 | +0.0299 / +0.0570 |
| `clipheade0_h1.5` | wide alpha + type block, MSE criterion | 6.8499 / 9.8121 | -0.0256 / -0.0141 |
| `clipheadx_hesst_h1.5` | importance on the TYPE election only | 6.8509 / 9.8023 | -0.0246 / -0.0239 |
| `clipheadx_hessa_e2m1` | importance on the ALPHA search only, **no type block at all** | 6.8189 / 9.7743 | **-0.0566 / -0.0519** |
| `clipheade0_hess_m1` | both | 6.8084 / 9.7569 | **-0.0671 / -0.0693** |

**The W4A16 conclusion survives in direction but not in magnitude.** The alpha search alone, with
E2M1 only and no type block, is -0.0566 / -0.0519 -- 84% / 75% of the full configuration. The E0M3
type block adds the remaining **-0.0105 / -0.0174**.

In W4A16 the type block was worth about a quarter of the alpha search (-0.011 vs -0.042); in W4A4
it is worth about a fifth on wikitext but a third on c4. So the earlier statement that the type
block is "a small increment" holds, but it is a larger increment once activations are quantized --
which is consistent with what the Qwen3-4B alpha=1 result shows from the other direction.

### One W4A4 measurement on the ACTIVATION type block

From a separate sweep with weights fixed at `mix_4_6_clipheade0_h1.5@8x64`, varying only the
activation dtype at a 16x64 tile (the A-operand `m16 x k64`):

| activations | wikitext |
|---|---|
| `nvfp4_4over6` | 6.8499 |
| `mix_4_6_clipheade0_e2m1` (type block, never elects) | 6.8455 |
| `mix_4_6_clipheade0_h1.5` | **6.8358** |

The activation type block is worth **-0.0141** with no calibration, no reordering and no search --
decided per tile at runtime. A peakedness veto (`pv<tau>`) on top was worse at every threshold
tried, because the error-based election already encodes peakedness (rho = -0.59, §8).

---

## 1b. How finely does the importance need to vary? (W4A4)

`hess` weights each element's squared error by `E[x_j^2]` of the input channel it multiplies. That
weight is applied **per element** -- 16 different values inside one 1x16 scale block. Coarsening it
(`impg<N>`: replace each weight by its mean over a run of N channels) asks how much of `hess` is
per-element detail and how much is a coarse envelope.

**Note the type block is 8x64 in every row below.** `impg<N>` changes the granularity of the
IMPORTANCE VECTOR, not the type block. The three granularities are independent:

| | size | configurable? |
|---|---|---|
| scale block | 16 | no -- fixed by NVFP4 |
| type block | **8x64** | yes, but held at 8x64 throughout this report |
| importance | element / 16 / 64 | this is what `impg` varies |

### Per-type-block importance is provably a no-op

A positive constant across a type block divides out of **both** decisions: the alpha search compares
candidates within a scale block, and every rule in `_elect_e0m3` compares quantities homogeneous of
the same degree in the loss (`argmin`, `harm`, `vote`) or is a ratio of degree-1 quantities
(`margin`). So `impg64` at 8x64 must reproduce the unweighted quantizer exactly.

`tests/test_imp_gran.py` asserts this bit-for-bit for all five election rules, and the perplexity
run confirms it end to end on Qwen3-4B (13.9223 / 17.1839, identical to `a1_h10` in every digit).
**"Apply hess at the type-block level" is therefore not a configuration -- it is the no-hess row.**

Two related invariances fall out of the same argument and are worth knowing:

* A per-**scale-block** constant cancels in the alpha search but not in the election, so `impg16`
  isolates a pure election effect. Verified: `impg16` equals the unweighted run exactly when the
  election is disabled (`_e2m1`).
* `dominance` is invariant to per-scale-block importance entirely -- it reads only the sign of each
  block's gain. `hess` can reach a dominance election only through the alpha search.

### The measurement, on all three models

8x64 weights, `nvfp4_4over6` activations, no reordering. Deltas against `nvfp4_4over6` weights,
except Qwen3-4B which is against `nvfp4` weights (its 4over6 baseline is broken, §5):

| importance granularity | Qwen3-4B (`a1_h10`) | Llama-3.1-8B (`heade0_m1`) | Qwen3-8B (`heade0_m1`) |
|---|---|---|---|
| none | +0.0122 / -0.0369 | -0.0204 / -0.0104 | **-0.0750 / -0.0463** |
| per type block (`impg64`, = none) | +0.0122 / -0.0369 | -0.0179 / -0.0105 | -0.0673 / -0.0458 |
| per scale block (`impg16`) | **-0.1222 / -0.1066** | -0.0231 / -0.0027 | -0.0653 / -0.0473 |
| per element (`hess`) | -0.0170 / -0.0036 | **-0.0671 / -0.0693** | -0.0424 / -0.0420 |

**Three models, three different answers, and no ordering survives across them:**

* **Qwen3-4B** -- the block mean is far better than per-element (-0.1222 vs -0.0170, a 7x gap).
  This is the best Qwen3-4B configuration measured anywhere in this report.
* **Llama-3.1-8B** -- per-element wins decisively (-0.0671); the block mean is indistinguishable
  from no importance at all.
* **Qwen3-8B** -- **importance HURTS here.** No importance is the best row (-0.0750), per-element
  the worst (-0.0424). This is a correction: earlier sections quote `heade0_hess_m1` on Qwen3-8B as
  a gain, but that was measured without a no-hess control at the same settings. With the control in
  place, `hess` costs +0.0326 wikitext on this model.

The plausible reading of the Qwen3-4B result is variance, and it is the report's recurring theme in
a new place. `E[x_j^2]` is heavy-tailed, so per-element weighting makes a block's loss depend
mostly on its one or two highest-importance channels -- a high-variance estimate of what the block
actually contributes. Averaging over the 16 channels keeps the direction and drops the variance.
That is the same move as `h<lambda>` over `argmin`: not a different criterion, a less noisy one.
But it does not generalize -- on Llama-3.1-8B the per-element detail is exactly what carries the
gain -- so **importance granularity is a per-model knob, not a default to change.**

### Why `impg64` is exact on one model and not the others

`impg64` must equal the no-importance row exactly, and on Qwen3-4B it does -- every digit. On
Llama-3.1-8B it drifts 0.0025 wikitext and on Qwen3-8B 0.0077. The cause is now measured
(`check_impg_noop.py`, real Llama-3.1-8B weights):

    6357 of 872,415,232 elements differ     (7.3e-6, max |delta| 4.9e-3)

`sum_j (c * d_j^2)` and `c * sum_j d_j^2` are **not the same float**, so a per-tile constant
rescales the loss only up to rounding and the alpha search's `err < best_err` can break a near-tie
the other way. The rate is far too low to see in a synthetic check (5e5 elements found 0), and far
too high to ignore across a 7e9-element model -- roughly 5e4 flipped elements per config.

This also explains the split across models: Qwen3-4B runs `clipa1`, a **single-candidate** alpha
preset. With one candidate there is no comparison to flip, so the invariance is exact. The other two
run `clipheade0` with five.

**Consequence: ~0.008 is the noise floor for the Llama-3.1-8B and Qwen3-8B rows in the table above**
-- a floor set by the alpha search's tie-breaking, not by calibration variance (§0). That makes
every `impg16` result on those two models a null. It does not touch the Qwen3-4B result, which is
exact by construction and 28x the §0 floor, nor the per-element `hess` gaps on Llama-3.1-8B
(-0.0671) and Qwen3-8B (+0.0326), which are 4-8x it.

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
