# Deciding E2M1 vs E0M3, calibration-free

Everything here is measured with `run_ppl_sweep.py` on wikitext-2 and C4 at seq_len 2048, against
`nvfp4_4over6`. No calibration data is used by any method reported as a recommendation; calibration
data appears only as a *diagnostic*, to explain why something failed.

Per-round detail is in `decide_r*/REPORT.md`. `python summarize_all_decide.py` merges them and
checks that shared baselines agree across rounds.

---

## The answer, in one line

**Use `mix_4_6_clipbothx_clipmin0.3_h3` at an 8x64 type block.** Clipping candidates on both grids,
gated behind a 30% minimum error reduction, with the conservative election. It is the only
configuration measured here that is negative on **every model and both datasets**:

| model / setting | dwikitext | dc4 | mean |
|---|---|---|---|
| Llama-3.1-8B W4A4 | -0.0297 | -0.0195 | **-0.0246** |
| Llama-3.1-8B W4A16 | -0.0179 | -0.0159 | **-0.0169** |
| Llama-3.2-3B W4A16 | -0.0053 | -0.0095 | **-0.0074** |
| Llama-2-7B W4A16 | -0.0007 | -0.0057 | **-0.0032** |

(The Llama-3.1-8B rows use `h1.5`; the other two use `h3`, which is the setting that is safe
everywhere.) This is the direction **round 1 rejected** — `alpha < 1`, i.e. clipping — at +0.006 to
+0.033 wikitext. It becomes the best generalizing idea in the study once it is gated behind a
minimum gain.

Two caveats that keep it honest. `clipfull` (headroom *and* clipping in one candidate set) is worse
than `clipbothx` on both hard models, so the two alpha directions are not additive. And `h1.5`
remains harmful on Llama-2-7B and Llama-3.2-3B even with clipping, so the conservative `kappa^2 = 3`
is load-bearing rather than a formality.

---

## The answer

### Part 1 — widen the block-scale search. Unconditional.

The block scale is `alpha * block_max / grid_max`. FourOverSix is the two-point search
`alpha in {1, 1.5}` on E2M1. Widen it to `{1, 1.25, 1.5, 2, 3}`.

| | wikitext | c4 | mean |
|---|---|---|---|
| Llama-3.1-8B W4A16 | -0.0082 | -0.0050 | **-0.0066** |
| Llama-2-7B W4A16 | -0.0044 | -0.0050 | **-0.0047** |
| Llama-3.2-3B W4A16 | +0.0036 | -0.0030 | +0.0003 |

**Neutral-to-positive, never harmful — but not a guaranteed win.** Worth about -0.005 mean on two of
the three models and nothing on the third. Take it because it is free and cannot hurt, not because
it always pays.

Free: `alpha` only changes the value written into the ue4m3 scale field NVFP4 already stores. Needs
no type block, no E0M3 operand, no extra metadata — it runs on the existing kernel. Shipped as the
data type **`nvfp4_nover6`** (`quant_nvfp4_nover6`), which is pinned by test to be bit-identical to
`quant_mix_4_6(clip="headx", elect="never")`, the configuration actually measured.

Why `alpha > 1` and not `alpha < 1` — the usable code values, in units of the block maximum:

| alpha | block max lands on | grid |
|---|---|---|
| 1 | code 6 | `{0,.083,.167,.25,.333,.5,.667,1}` log-spaced |
| 1.5 | code 4 | `{0,.125,.25,.375,.5,.75,1}` — FourOverSix |
| 2 | code 3 | `{0,.167,.333,.5,.667,1}` uniform, 6 levels |
| 3 | code 2 | `{0,.25,.5,.75,1}` uniform, 4 levels |

Headroom walks E2M1 from log-spaced-at-full-range to uniform-with-few-levels, spending the sparse
top of the grid rather than the resolution near zero. `alpha < 1` is clipping — it saturates the
block maximum — and costs +0.006 to +0.033 wikitext (round 1). Five candidates is where the search
saturates; eight is worse.

**Which alphas actually get chosen.** Measured over the first seven linear layers of Llama-2-7B,
the share of 16-element scale blocks selecting each candidate:

| alpha | 1 | 1.25 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|
| share | 58.3% | **3.6%** | 38.0% | **0.0%** | **0.0%** |

So the whole gain over FourOverSix comes from **one extra candidate, `alpha = 1.25`**, taken by
under 4% of blocks. The coarse uniform grids at `alpha = 2` and `3` are never selected on real
weights, which is why `headxx` (eight candidates, reaching out to `alpha = 4`) is no better and
measurably worse on wikitext — the extra candidates only add ways for MSE to pick something that
does not help perplexity. The action is entirely in the interval `[1, 1.5]`.

### Part 2 — the E0M3 type block is a model-dependent extra.

Add E0M3 headroom (`alpha in {1, 7/6, 7/5}`) and elect per type block with the robust rule at
`kappa^2 = 1.5`:

| | Llama-3.1-8B | Llama-2-7B |
|---|---|---|
| `mix_4_6_clipheade0_h1.5` @ 8x64 | **-0.0265 / -0.0081** | **+0.0165 / +0.0061** |

Worth a further -0.018 wikitext where it works, and a loss of comparable size where it does not --
and it does not work on **two models out of three**:

| `h1.5` @ 8x64 | Llama-3.1-8B | Llama-2-7B | Llama-3.2-3B |
|---|---|---|---|
| dwikitext | **-0.0117** | +0.0128 | +0.0208 |

`kappa^2 = 3` is the only election setting non-harmful on all three, and it is worth about -0.002.
This is not the type block's fault: on Llama-2-7B the same configuration is +0.0051 at a `1x16` type
block, the finest election possible, so E0M3 hurts that model at every granularity. Round 6 tried to
predict the regime from the weights and failed — the per-tensor E0M3 gain fraction is 0.205 on
Llama-2-7B and 0.199 on Llama-3.1-8B, identical. Round 9 rules out the next obvious predictor too:
Llama-3.2-3B has a *large* `1x16` gain (-0.0416 / -0.0943, comparable to Llama-3.1-8B) and yet no
realizable tile rule captures any of it. A large per-block gain is necessary but not sufficient.

**The election rule.** `h<lambda>`: elect E0M3 only when

    sum_{gain>0} gain_b  >  lambda * sum_{gain<0} |gain_b|

This is the exact decision that survives any per-block importance `w_b` in `[1/kappa, kappa]` with
`lambda = kappa^2` — the robust-optimization form, derived in `_elect_e0m3` and checked against the
explicit worst case in the tests. `lambda = 1` is plain argmin, `lambda -> inf` is dominance.
`lambda` in [1.5, 2] is optimal on Llama-3.1-8B; `lambda = 3` is the only value measured that is
non-harmful on both models, and there it is worth almost nothing.

### Part 3 — which direction of alpha helps depends on the OPERAND

Weights and activations want opposite things from the same knob, and the reason is the shape of a
16-element block in each.

| Llama-3.1-8B, 8x64, vs `nvfp4_4over6` | W4A16 (weights only) | W4A4 (weights + activations) |
|---|---|---|
| `h1.5` — election alone, 4over6 scales | -0.0117 / -0.0044 | -0.0079 / -0.0085 |
| `clipheade0_h1.5` — **headroom**, alpha > 1 | **-0.0265 / -0.0081** | -0.0316 / -0.0032 |
| `clipbothx_clipmin0.3_h1.5` — **gated clipping**, alpha < 1 | -0.0179 / -0.0159 | **-0.0297 / -0.0195** |

On weights, headroom wins: a 16-element slice of a weight row has a light upper tail, so the block
maximum is worth keeping and the payoff is in spending the sparse top of the E2M1 grid. On
activations, gated clipping wins by mean delta: an activation block frequently contains a genuine
outlier that the other fifteen elements are paying for, and saturating it is cheap **provided the
gain is decisive** — ungated clipping is a loss in both settings.

Note also how much larger the prize is on activations. The unrealizable `1x16` upper bound is
-0.0810 / -0.1006 at W4A4 against -0.0468 / -0.0623 at W4A16, so the E0M3 decision is worth roughly
twice as much on the A operand as on the B operand. `nvif4` is -0.0689 / -0.0868 there. That is the
opposite of where the study started -- the type block was introduced for weights.

### Why E0M3 exists at all

E0M3 with `alpha = 7/n` is exactly a **uniform n-level grid** (verified against the quantizer):

| alpha | 7/7 | 7/6 | 7/5 | 7/4 | 7/3 |
|---|---|---|---|---|---|
| levels | 7 | 6 | 5 | 4 | 3 |

E2M1 cannot reach these above n = 4 — its codes `{0,.5,1,1.5,2,3,4,6}` are uniform only up to code
2. So the two element types are not "log grid vs uniform grid"; they are one family of block
quantizers that the free alpha search spans jointly, with E0M3 supplying the fine uniform grids.
That is also why the two halves are synergistic: headroom with the E0M3 branch switched off is
-0.0082, always-E0M3 is +0.0400, and the two together are -0.0265.

---

## The principle that organises all of it

> **A rule of the form "do X when it lowers the quantization error" loses. The same rule with
> "...by a decisive margin" wins.**

Measured independently on three unrelated mechanisms (Llama-3.1-8B W4A16, 8x64):

| mechanism | "when it helps" | "when it decisively helps" |
|---|---|---|
| elect E0M3 for a type block | `argmin` +0.0021 / +0.0185 | `h1.5` **-0.0117 / -0.0044** |
| rotate a column chunk (Hadamard) | `rotcol` +0.0946 / +0.1431 | `rotmin0.1` **-0.0149 / -0.0025** |
| clip the block scale | `alpha < 1` +0.006 … +0.033 | `clipmin0.3` **-0.0179 / -0.0159** |

The reason is measured, not assumed. Weight MSE and the true layer output error agree on the sign of
a *large* change and decouple on small ones. Round 3 shows this directly: Hadamard rotation lowers
weight MSE on every layer of Llama-2-7B, but lowers the true output error `||X dW^T||^2` only where
the MSE gain is big, and *raises* it where the gain is small.

| layer | weight MSE | true output error |
|---|---|---|
| q_proj | -24.9% | **-62.2%** |
| k_proj | -18.0% | **-54.7%** |
| o_proj | -5.7% | **-13.2%** |
| v_proj | -2.0% | **+72.7%** |
| gate / up / down_proj | -0.9 … -0.1% | **+2.5 … +8.0%** |

---

## What does not work

| idea | result |
|---|---|
| **Clipping** (`alpha < 1`) ungated | +0.006 to +0.033 wikitext. Works only behind `clipmin<t>`. |
| **MAE / Lp selection loss** | `mae` within 0.0005 of `mse`; `l0.5`, `l1.5` worse. |
| **Unconditional Hadamard rotation** | +0.09 to +0.15 wikitext despite -25% NMSE. |
| **Rotation, selective at threshold 0** | +0.0946 — a zero threshold is not selectivity. |
| **Row permutation** (`_perm`) | Worse than the same election rule without it, every time. An 8x64 tile is 8 rows x 4 scale blocks and the disagreement is mostly *within* a row across k-blocks, so straddling stays near 100% however rows are ordered. |
| **`corr<r>`**, the equicorrelated-input loss | Inert. `(sum dW)^2 / sum dW^2` measures 0.998–1.005 per scale block, so the rank-one term it prices is ~0.3% of the loss. |
| **Calibration-free `diag(S)` proxies** | RMSNorm `gamma^2` correlates +0.63 on q/k/v_proj but **-0.50** on gate/up_proj and is undefined for o_proj/down_proj. Weight column energy has no consistent sign. |
| **Predicting the E0M3 regime from weights** | The per-tensor gain fraction is identical across the two models (0.205 vs 0.199). |

---

## Context

RaZeR remains far ahead of this entire family: `-0.0982 / -0.1292` on Llama-3.1-8B W4A16, against
`-0.0265 / -0.0081` for the best hardware-realizable configuration here. The unrealizable `1x16`
upper bound is `-0.0468 / -0.0623`, so the best realizable rule captures roughly a third of what
per-scale-block choice would give.

The type-block ordering never changed across ten rounds: **8x64 > 16x64 > 32x64 > 32x128**. The
smallest hardware-realizable tile is always the right one, because a smaller tile overrules fewer
scale blocks.
