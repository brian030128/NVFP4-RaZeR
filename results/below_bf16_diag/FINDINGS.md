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

## 2. It is not that E0M3 is a better number format

The direct control is already in the record. A pure weight-MSE criterion —
elect E0M3 wherever it reduces the reconstruction error of the weights — selects
**3,407,142 of 7,096,320 tiles (48%)** on this model, 52× more than the adaptive
map, and the result is *worse* than the baseline it started from:

| Selection | E0M3 tiles | WikiText (512 protocol) |
|---|---:|---:|
| FourOverSix baseline | 0 | 19.414897 |
| Weight-MSE | 3,407,142 | 20.108009 (+0.693) |

So E0M3 is not more accurate in any sense that perplexity rewards. Selecting it
on numerical grounds, aggressively, hurts. The gain comes entirely from the
criterion used to choose tiles, not from the format being chosen.

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
