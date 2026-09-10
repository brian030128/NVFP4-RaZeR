# Why Qwen3-4B MixFP4 scores 10.86 WikiText against a 13.66 BF16 reference

The paper-aligned adaptive run (job 335993) reports, for `Qwen/Qwen3-4B` under
the released 2048-token protocol:

| Policy | E0M3 tiles | WikiText-2 | C4 |
|---|---:|---:|---:|
| BF16 reference (no weight or activation quantization) | — | 13.662473 | 16.643564 |
| FourOverSix W4A4 baseline | 0 | 14.269062 | 17.326633 |
| MixFP4 fixed | 256 | 13.040957 | 16.615953 |
| MixFP4 adaptive | 65,536 | **10.864750** | **15.161474** |

A 4-bit model beating its own unquantized weights by 2.80 WikiText perplexity is
not a numerical result — nothing about approximating a network more coarsely can
make it a better predictor than the network being approximated. The explanation
is that the map is not chosen for numerical fidelity at all.

**Summary of the finding.** The map is a gradient-descent step on cross-entropy
with the element-type choice as a binary parameter (§3), so BF16 is its
initialization rather than its target. `Qwen/Qwen3-4B` is a post-trained release
whose predictive distribution is about 35% too sharp for raw-text next-token
prediction, which leaves a large amount of cross-entropy recoverable (§4, §5a);
base models in the same panel have almost none and are harmed at large counts.
Of the 2.812 PPL headline gain, **95% is recalibration of that over-sharpness and
5% is a genuine likelihood improvement** (§5a). Accuracy never reaches BF16.

## 1. The effect is real, and it is not evaluation leakage

The 2048-token protocol scales activations with a tensor-wide factor taken over
the whole window, so it can depend on future tokens, while BF16 has no activation
quantization and gets no such help. That asymmetry would be a sufficient
explanation on its own, so it has to go first.

It is not the cause. The 512-token tile-count sweep (job 335887) uses row-wise
per-token activation factors, which are causal. Its FourOverSix baseline
reproduces the audit's `four_over_six_row` C4 value to the digit
(21.928243), which confirms the convention, and against the audit's matching
causal BF16 reference on the same sample:

| | C4 PPL |
|---|---:|
| BF16, causal protocol | 20.887771 |
| FourOverSix baseline | 21.928243 |
| Adaptive, all 80,978 eligible tiles | **19.064606** (−1.823165 vs BF16) |

Below BF16 under a causal convention too. Whatever is happening survives the
removal of future-token information.

## 2. The gain is not attributable to E0M3 being a better number format

To be clear about what is and is not in dispute: flipping a selected subset of
blocks to E0M3 **does** improve over matched FourOverSix and NVFP4, at equal bit
width, the same 8×64 tile, and the same kernel cost. That result stands. What
does not follow is the causal reading — that the improvement comes from E0M3
representing those blocks' values more faithfully.

The direct control is already in the record. A pure weight-MSE criterion —
elect E0M3 wherever it reduces the reconstruction error of the weights — selects
**3,407,142 of 7,096,320 tiles (48%)** on this model, 52× more than the adaptive
map, and the result is *worse* than the baseline it started from:

| Selection | E0M3 tiles | WikiText (512 protocol) |
|---|---:|---:|
| FourOverSix baseline | 0 | 19.414897 |
| Weight-MSE | 3,407,142 | 20.108009 (+0.693) |

So on this model E0M3 is not more accurate in any sense perplexity rewards:
selecting it on numerical grounds, aggressively, hurts. The information is in the
criterion used to choose tiles, not in the grid being chosen. §5b bounds the
representational contribution independently, and notes where the argument stops
being decisive; §5c gives the reason a fidelity criterion could not have produced
a below-BF16 number at all.

## 3. What the criterion actually is

`run_math_code_calibration.py:148-158` scores each 8×64 tile as

```
value = Σ_tile (∂L/∂W) ⊙ (W_E0M3 − W_E2M1)
```

the first-order directional derivative of a loss along the weight change that
flipping the tile would produce, accumulated per sequence for two losses: cross
entropy on OpenWebMath/CodeParrot text, and KL to the pristine BF16 teacher.
`common_descent_scores` takes a two-standard-error bound over the 128 sequences
for each and keeps tiles where both descend.

That is a gradient-descent step. The E2M1/E0M3 choice is being used as a binary
trainable parameter, and 66,856 of them are fitted to lower cross-entropy on
natural text. The procedure is not choosing a number format; it is training the
model through the only degrees of freedom the format offers.

Once the objective is cross-entropy, BF16 has no special status: it is the
initialization, not the optimum. There is no reason a descent step should stop
when it reaches the starting point.

Two observations follow directly and both hold:

- **The gain grows with the number of fitted parameters.** Selection loss falls
  monotonically with count (1.441248 at 256 → 1.369785 at 65,536), and so does
  held-out perplexity. It then ticks back up at all 66,856 eligible tiles
  (1.370014), which is why 65,536 was the chosen count — a fit that has started
  to overfit. That whole shape is the signature of a fit, not of a format
  decision.
- **Only a model that is off the natural-text optimum has anything to gain.**

## 4. Why this model and not the others

The four-model sweep contains three base models and one post-trained model:

| Model | Release | Best ΔPPL vs baseline | Behaviour at large counts |
|---|---|---:|---|
| OLMo-1B | base | −0.140 | saturates |
| Pythia-1.4B | base | −0.621 | **harmed** beyond 4,096 tiles |
| Llama-3.1-8B | base | −0.091 | **harmed** at all eligible |
| Qwen3-4B | post-trained | **−2.864** | still improving at every eligible tile |

`Qwen/Qwen3-4B` is the post-trained instruct/thinking release, not
`Qwen3-4B-Base`. A base model was trained to minimize exactly this loss on
exactly this kind of text, so it sits near a local optimum and a descent step
finds almost nothing — and past a point starts overfitting 128 calibration
sequences, which is what the two harmed rows are. A post-trained model has been
moved off the raw-text likelihood optimum by instruction and reasoning training,
so a large amount of cross-entropy is recoverable, and the descent step
recovers it.

This also explains the otherwise odd BF16 number itself: 13.66 WikiText for a
4B model is high, and it is high for the same reason the headroom exists.

## 5. The gain is likelihood, not capability

Perplexity is a proper scoring rule on the whole predictive distribution, so it
pays for calibration as well as for prediction. Three independent readings all
say the recovered cross-entropy is mostly the former.

- **It is a uniform shift.** Matching evaluation windows by token hash against
  the released BF16 reproduction, the adaptive policy beats BF16 on **146/146**
  WikiText windows and 251/256 C4 windows, with mean ≈ median (−0.2291 nats on
  WikiText). Not a few catastrophic windows being repaired.
- **It scales with how surprised the model already was.** The per-window gain
  correlates +0.471 with the window's own BF16 NLL on WikiText (+0.263 on C4),
  slope +0.087. Ordinary quantization damage does not behave this way: the
  FourOverSix loss against BF16 is flat in window difficulty (correlation
  +0.003, slope +0.0003). The two are structurally different effects.
- **Accuracy does not follow.** The repo's own zero-shot check (job 336108) puts
  the adaptive map at 0.6395 mean accuracy over six multiple-choice tasks
  against BF16's 0.6509 — still 0.0114 *below* the unquantized model while its
  perplexity is 2.797723 *better*.

Among the quantized policies accuracy does corroborate the method, in the order
perplexity gives (FourOverSix 0.6249 → 256 tiles 0.6286 → 65,536 tiles 0.6395),
so the map is genuinely repairing quantization damage as well. Both things are
happening; only the first one crosses the BF16 line.

## 5a. How much of the gap is sharpness: the temperature-matched measurement

Job 336584 measures this directly on the same 146 WikiText windows. Rescaling
logits by a scalar changes only sharpness and leaves every argmax — hence every
accuracy — untouched, so a per-policy optimal temperature separates "predicts
better" from "is calibrated better". `T*` is fitted on the evaluation set over a
15-point grid, which makes it an oracle upper bound on what recalibration can
buy, not a deployable method.

| Policy | PPL | Entropy | top-1 prob | top-1 acc | BF16 argmax agreement | T* | PPL @ T* |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 13.659909 | 1.2047 | 0.6911 | 0.5159 | 1.0000 | 1.35 | 10.782109 |
| FourOverSix | 14.212161 | 1.3287 | 0.6715 | 0.5031 | 0.8550 | 1.30 | 11.683453 |
| MixFP4 256 | 13.035763 | 1.4497 | 0.6536 | 0.5053 | 0.8561 | 1.25 | 11.371508 |
| MixFP4 65,536 | 10.847653 | 1.9383 | 0.5910 | 0.5107 | 0.8366 | 1.10 | 10.642044 |

(These use `use_cache=False` uniformly, so they sit about 0.057 below the
published table's WikiText values, which use the released per-window cache. All
four rows share the setting, so the contrasts are unaffected.)

Three things are visible at once.

**BF16 is over-sharp for this text, and that is the bulk of the story.** Its own
optimal temperature is 1.35 — its logits are about 35% too confident for raw-text
next-token prediction — and recalibrating it alone takes 13.659909 to 10.782109,
a 2.878 PPL drop. The entire headline gain of the map is 2.812 PPL. A single
scalar on the unquantized model recovers more than the whole thing.

**The map's mechanism is visibly the consumption of that headroom.** Entropy
rises monotonically with tile count (1.2047 → 1.3287 → 1.4497 → 1.9383, a 61%
increase over BF16) while the residual optimal temperature falls monotonically
toward 1 (1.35 → 1.30 → 1.25 → 1.10). By 65,536 tiles almost nothing is left to
recalibrate, because the map has already spent it.

**A small residual is genuine.** At matched calibration the map is still ahead of
BF16, 10.642044 against 10.782109. Decomposing the 2.812 PPL headline gain:

| Component | PPL |
|---|---:|
| Sharpness / calibration | 2.672 (95.0%) |
| Genuine likelihood improvement | 0.140 (5.0%) |

So the honest statement is not "it is entirely an artefact". It is that **95% of
the improvement is recalibration of an over-sharp post-trained model, and 5% is a
real gain in next-token likelihood** — which is exactly what a cross-entropy
gradient step should produce, and exactly why the accuracy numbers barely move.

The elected tiles are not concentrated in one high-leverage place, which rules
out the simplest version of "it just rescales the logits". All 252 targeted
linear modules are touched, across every projection type:

| projection | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiles | 5,043 | 1,380 | 3,196 | 13,547 | 9,705 | 13,142 | 19,523 |

`lm_head` is not quantized at all (the scoring and evaluation both exclude the
output embedding), so the sharpness change is produced inside the network rather
than by scaling the output layer.

The argmax agreement column supports the same reading from the other side. The
65,536-tile map agrees with BF16's argmax *least* of all policies (0.8366, below
even plain FourOverSix at 0.8550) while scoring *higher* accuracy than
FourOverSix (0.5107 against 0.5031). It has moved the model further from BF16 than
plain quantization did, and in a slightly better direction. That is a training
step, not noise — but it is a very small one, and it does not reach BF16's own
accuracy of 0.5159.

## 5b. Independent of §5a: the representational ceiling is ~0.06 PPL

The temperature decomposition is one line of argument. This one is separate and
survives even if that analysis is rejected entirely.

`1x16` is the finest possible type block, and because the MSE selection is
error-minimizing and `1x16` divides every coarser shape, it is a **strict upper
bound** on what any fidelity-driven E0M3 mixing can achieve at any granularity.
From `results/mixfp4_sweep/REPORT.md`, against NVFP4:

| model / setting | `1x16` (MSE ceiling) | `8x64` (deployable shape) |
|---|---:|---:|
| Llama-2-7B W4A16 | −0.0089 | **+0.0282** |
| Llama-2-7B W4A4 | −0.0504 | **+0.0030** |
| Llama-3.1-8B W4A16 | −0.0643 | **+0.0142** |

So the entire representational value of making E0M3 available, measured where it
is maximal, is at most about 0.06 PPL — and at the 8×64 shape this study
deploys, it is *negative*. The Qwen3-4B adaptive gain over FourOverSix is 3.40
PPL, roughly fifty times that ceiling. It cannot be the same phenomenon.

Two further separations point the same way:

- **Same format, fidelity criterion, opposite sign.** On Qwen3-4B, weight-MSE
  elects 3,407,142 tiles (48%, 52× the adaptive map) and WikiText goes
  20.108009 against a 19.414897 baseline — worse. More E0M3, chosen precisely
  for representational accuracy, moves backwards.
- **The sign depends on the checkpoint.** Qwen3-4B −2.864, OLMo-1B −0.140,
  Llama-3.1-8B −0.091, Pythia-1.4B *harmed* beyond 16,384 tiles. A grid that
  represents values better does not damage Pythia.

**Where this argument does not reach.** For Llama-3.1-8B the adaptive gain
(−0.03 to −0.09) is the same order as the representational ceiling (~0.06), so
there the two accounts are not separable by magnitude. The separation is
decisive only where the gain is large, which is Qwen3-4B.

The supportable claim is therefore *"a per-tile type map fitted to task loss
beats FourOverSix at equal bits and equal kernel cost"* — not *"E0M3 is a better
element type"*. E0M3 is the actuator, and a necessary one: without a second
element type there is no free parameter to fit. The information came from the
calibration data, not from the shape of the grid.

## 5c. Why a fidelity method could never have produced this number

"Uses calibration data" does not distinguish this from GPTQ or AWQ, which are
calibrated too. What distinguishes it is what the objective targets:

| method | objective | its optimum | BF16 is |
|---|---|---|---|
| GPTQ | layer output reconstruction | the original weights | a ceiling, by construction |
| AWQ / SmoothQuant | weighted fidelity to W | the original weights | a ceiling |
| distillation | KL to the BF16 teacher | the teacher | a ceiling |
| this rule's CE term | CE on ground-truth labels | **not the original weights** | **not a ceiling** |

A method whose objective is a distance to the original model can approach BF16
and not pass it — the target *is* BF16. This rule scores on two losses, and the
KL-to-teacher half is exactly such a fidelity term, bounded by BF16. The CE half
is not: its minimizer is whatever weights predict text best, and there is no
reason that is the released checkpoint.

This makes a large below-BF16 result a **one-way diagnostic**. A fidelity-objective
method cannot produce one, so observing one is positive evidence that the
objective included ground-truth labels, and the size of the overshoot measures
how much training occurred.

None of which makes the method illegitimate. It is quantization-aware training
with an unusual parameterization: one gradient step, 66,856 binary parameters,
zero inference cost, no metadata change, and an output that is still a legal
4-bit format on the existing `mma.sync` path. Naming it that way changes the
baseline set, though — as a format the baselines are NVFP4 / FourOverSix / RaZeR
at equal bits, which it beats; as QAT a reader will also want round-to-nearest
plus a fitted temperature, and a matched-budget QAT comparison.

**Counterexample to keep in view.** On Qwen3.8-27B the repository records
weight-MSE selection at WikiText 7.1545 with 43.49 million switches, statistically
indistinguishable from the calibrated sparse maps (7.1331 / 7.1569), with an
explicit instruction not to claim calibration beats MSE there. So fidelity
criteria do not always lose. The margin by which calibration beats MSE tracks the
available cross-entropy headroom: large on the over-sharp Qwen3-4B, absent on the
27B. That is the same variable as §4 and §5a, and it is not a property of E0M3.

## 6. Consequences

- The below-BF16 perplexities are not a quality claim and cannot be reported as
  one. `MIXFP4_REPORT.md` §5 already says this; this document supplies the
  mechanism.
- The "share of the BF16 gap recovered" statistic breaks down when the share
  exceeds 100%. It already does for the *fixed 256* map in `MIXFP4_REPORT.md`
  (202.5% on WikiText, 104.0% on C4), and far more so at 65,536 tiles. The
  quantity being divided by is not the quantity being moved.
- Comparisons against the matched FourOverSix baseline are unaffected — that
  baseline shares the model, the data, the windows and the activation
  convention.
- The calibration is done on OpenWebMath and CodeParrot and never on WikiText or
  C4, so this is not test-set contamination. It is a fit to natural-text
  cross-entropy that transfers, which is exactly what a fit to natural-text
  cross-entropy should do.
