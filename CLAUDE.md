# CLAUDE.md

Guidance for working in this repository.

**Read `../CLAUDE.md` (i.e. `/home/u4320956/CLAUDE.md`) first.** It sets the cluster-wide rules that
override defaults everywhere on this machine — most importantly that all heavy compute (including
CPU-only work) must go through `sbatch`/`srun` rather than the login node, where the HuggingFace
cache belongs, which account/partitions to use, and the `uv` workflow. Nothing here needs GPUs (see
below), but if a task in this repo ever does grow into something GPU- or CPU-heavy, follow that file
for how to submit it as a job.

## Repository overview

RaZeR is a research codebase for **simulated (fake) 4-bit LLM quantization**. Nothing here needs
FP4 hardware: every format is emulated in FP32/BF16 and the result is written back into the model's
BF16 weights or activations. All quantizers therefore run on the CPU.

- `quantize/quantizer.py` — every fake quantizer (`quant_mxfp4`, `quant_nvfp4`, `quant_nvif4`,
  `quant_mixfp4`, the RaZeR variants, ...) plus the two dispatch functions `quant_weight` (walks the
  model and rewrites `nn.Linear.weight` in place) and `quant_act` (called inline from the quantized
  model modules).
- `quantize/quant_config.py` — `QuantConfig`, the single object threaded through the model.
- `quantize/utils.py` — scale-quantization and type-block helpers.
- `utils.py` — CLI argument definitions, `QuantConfig` construction, model loading, result-file naming.
- `models/qmodule_*.py` — HuggingFace model copies with `quant_act` calls inserted at the
  activation, KV-cache and attention-output boundaries.
- `run_ppl.py`, `run_zeroshot.py`, `run_llama_cot.py` — evaluation entry points.
- `run_mixfp4_sim.py` — CPU-only MixFP4 error sweep (no model download required).
- `tests/test_mixfp4.py` — CPU tests for MixFP4.
- `inference/` — the published CUDA/kernel artifact. Unrelated to the simulation path above; do not
  change it when adding a simulated format.

### Adding a new simulated format

1. Write `quant_<name>(w_fp, n_bits, groupsize, ...)` in `quantize/quantizer.py`. It takes a tensor
   of any shape, quantizes along the last (reduction) dimension, and returns a **dequantized**
   tensor of the same shape in `torch.bfloat16`.
2. Register the name in the `if/elif` chains of both `quant_weight` and `quant_act`.
3. If the format needs extra parameters, add them to `QuantConfig` and to `add_quant_args` in
   `utils.py`, and make sure they end up in the result file name (`get_output_file_tag`) so that
   sweeps do not overwrite each other.

---

## MixFP4

MixFP4 is NVFP4 plus a second, coarser block granularity that selects the FP4 **element data type**.
Everything else — the FP32 per-tensor global scale, the E4M3 block scale, the 16-element scale block
— is inherited from NVFP4 unchanged.

### There are TWO kinds of block

| | **Scale block** | **Type block** |
|---|---|---|
| Size | **Always 16 elements** along K (the NVFP4 block) | **Configurable `<M>x<K>`**, e.g. `1x16`, `16x16`, `256x16`, `32x64`, `32x128` |
| Shape | 1-D, `1 x 16` along the reduction dimension | 2-D tile: `M` rows x `K` columns |
| What it owns | one **E4M3 block scale** | one **element data type**: either **E2M1** or **E0M3** |
| Relationship | — | **A type block contains multiple scale blocks** |

A type block of shape `M x K` contains `M * (K / 16)` scale blocks. Consequently:

- **The K dimension of a type block is always a multiple of 16.** `32x24` is invalid; `32x64` holds
  `32 * 4 = 128` scale blocks. This is asserted in `parse_type_block`.
- `1x16` is the degenerate case where the type block **is** the scale block, i.e. the data type is
  chosen per NVFP4 block. In that configuration MixFP4 is numerically identical to the existing
  `nvif4` quantizer (there is a test asserting exactly this).
- Every scale block inside a type block uses the **same** element data type. Each of them still has
  its **own** E4M3 scale.

```
type block 32x64  (M = 32 rows, K = 64 columns) -> ONE data type for the whole tile
+---------------------------------------------------------------+
| row 0  [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |  each [ scale blk ] = 16 elements
| row 1  [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |  with its own E4M3 scale
|  ...                                                          |
| row 31 [ scale blk ][ scale blk ][ scale blk ][ scale blk ]    |
+---------------------------------------------------------------+
   = 32 * 4 = 128 scale blocks, all E2M1 or all E0M3
```

### The two element data types

- **E2M1** — the standard FP4 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, max magnitude 6. This is
  what plain NVFP4 always uses.
- **E0M3** — the *evenly spaced* signed 4-bit grid `{0, ±1, ..., ±7}`, max magnitude 7. Numerically
  equivalent to signed INT4. The block scale is `block_max / 7`.

Both grids encode 15 distinct values in 16 codes — the redundant zero that RaZeR exploits — and both
use the same ue4m3 block scale. Only the spacing differs: E2M1 has finer resolution near zero and
coarser resolution near the block maximum, E0M3 is uniform. Which one wins depends on the
distribution inside the tile, which is why the choice is data driven.

### Hardware grounding and the minimum type-block size

MixFP4 is not a paper format: the public NVFP4 path already issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read **A, B, or both** as E0M3 instead of E2M1. That fixes three things:

- **`scale_vec::4X` + `ue4m3`** is exactly the NVFP4 scale block: four ue4m3 scales across `k64`,
  i.e. one scale per 16 elements. The scale block is 16 for both element types, and the E0M3 branch
  reuses the same ue4m3 scale — which is what `quant_mixfp4` does.
- **The data type is selected per operand, not per element.** Weights and activations choose
  independently, hence the separate `--w_type_block` and `--a_type_block`.
- **A single instruction cannot subdivide its operand tile**, so the *smallest hardware-realizable*
  type block is one MMA operand tile. For the usual `Y = X · Wᵀ` mapping (`X` = A, `Wᵀ` = B):

  | operand | tile | minimum type block |
  |---|---|---|
  | A — activations | `m16 x k64` | **16x64** (16 tokens x 64 K) |
  | B — weights | `n8 x k64` | **8x64** (8 output channels x 64 K) |

  Confirm this mapping against the actual kernel before quoting it — a kernel that puts weights in
  A swaps the two rows.

Anything coarser (`32x64`, `32x128`, `256x64`, ...) is a union of whole operand tiles and is
realizable. Anything with **K < 64** — including `1x16`, `16x16` and `256x16` — is *not* expressible
with this instruction, because one MMA consumes 64 contiguous K elements under a single declared
element type. Those configurations are still worth sweeping as **accuracy upper bounds**: `1x16` is
the finest possible selection and bounds what any coarser scheme can achieve. Just do not report
them as deployable without a different instruction or a 4x-cost K-splitting trick.

`quant_mixfp4` deliberately does not enforce the K >= 64 rule, so that these reference points stay
measurable.

### Quantization procedure

For a tensor reshaped to `(M, K)` (rows = output channels for weights, tokens for activations):

1. **Global scale** (per tensor, FP32), the NVFP4 convention: `amax / (6 * 448)`.
2. Tile into type blocks, then into 16-element scale blocks.
3. For **each scale block**, compute both candidate E4M3 block scales: `block_max / 6` for E2M1 and
   `block_max / 7` for E0M3, each clamped to `[2^-9, 448]` and rounded to `float8_e4m3fn`.
4. Quantize every scale block **both ways**.
5. For **each type block**, sum the squared error over all of its scale blocks for each data type,
   and keep the data type with the smaller total. Ties go to E2M1.

### Configuration

```
--w_dtype mixfp4 --w_groupsize 16 --w_type_block 32x128
--a_dtype mixfp4 --a_groupsize 16 --a_type_block 1x16
```

- `--w_type_block` / `--a_type_block` accept `"<M>x<K>"` (default `1x16`) and are ignored by every
  other data type.
- `--w_groupsize` / `--a_groupsize` **must be 16** for MixFP4 — the scale block is the NVFP4 block
  and is not configurable.
- The type-block shape is appended to the result file name (`..._mixfp4-32x128__...`) so a sweep
  over type-block shapes does not collide.

### Implementation notes / gotchas

- `quant_mixfp4` reshapes to `(-1, last_dim)`, so 3-D activations `(batch, seq, hidden)` and 4-D
  KV tensors `(batch, heads, seq, head_dim)` are all handled; `M` spans everything except the last
  dimension.
- **Outer-dimension padding**: if `M` is not divisible by the type block's `M`, the tensor is
  zero-padded at the bottom. The padded rows form their own all-zero type blocks and never
  influence the data type chosen for real rows.
- **Narrow reduction dimension**: if `K` is smaller than the type block's `K` (e.g. a 64-wide head
  dimension with a `32x128` type block), the type block shrinks to the full row instead of raising,
  so that a sweep keeps running. `K` must still be a multiple of 16.
- Finer type blocks can never have a higher total squared error than coarser ones, because `1x16`
  divides every other configuration and the selection is error-minimizing. Use this as a sanity
  check when changing the selection rule.

### Testing (CPU only)

```
python tests/test_mixfp4.py        # correctness: tiling, per-type-block uniformity, padding, nvif4 equivalence
python run_mixfp4_sim.py           # NMSE / SQNR sweep over type-block shapes on synthetic tensors
python run_mixfp4_sim.py --model_name llama-2-7b --max_layers 4   # ... on real weights
```

`tests/test_mixfp4.py` includes a negative control: the per-type-block uniformity check must *fail*
on `nvif4` output. Keep that control if you touch the check, otherwise it can silently pass on
anything.

### Measured accuracy (Llama-2-7B, Llama-3.1-8B)

`results/mixfp4_sweep/REPORT.md` holds the full perplexity sweep. The headline, which should shape
any further work on this format:

- **MixFP4 only helps at `1x16`**, where it is bit-identical to `nvif4` and beats NVFP4 by
  0.009 (Llama-2-7B) / 0.064 (Llama-3.1-8B) wikitext ppl.
- **Every type block coarser than one scale block is worse than plain NVFP4**, including all
  hardware-realizable shapes (`8x64` and up): +0.03 wikitext ppl on Llama-2-7B W4A16, +0.014 on
  Llama-3.1-8B. Almost the entire loss happens in the first coarsening step (`1x16` -> `16x16`);
  `16x16` through `32x128` are within ~0.005 ppl of each other.
- RaZeR remains the strongest format in every setting measured.

The cause is visible in the E0M3 selection rates: at `1x16` the choice is genuinely mixed within a
tensor (41% E0M3 in `q_proj`, 60% in `v_proj`), but a large tile has to elect a single winner, so
`q_proj` collapses to ~all E2M1 (5.5% E0M3 at `32x128`, i.e. back to NVFP4) while `v_proj` collapses
to ~all E0M3 (99.6%). The mixing that produced the gain is exactly what the coarse granularity
averages away.

Note also that the selection minimizes MSE, and lower MSE does not always mean lower perplexity --
`nvif4`/`mixfp4_1x16` has clearly lower NMSE than `nvfp4_4over6` yet slightly higher wikitext ppl on
Llama-2-7B. Treat the sim NMSE as a fast proxy only, and confirm with `run_ppl_sweep.py`.

### Selection objective and election rules (why MSE is the wrong criterion)

For `Y = X W^T`, a weight perturbation `dW` moves the output by `dY = X dW^T`, so the expected
squared output error is `tr(dW S dW^T) = ||dW||^2_S` with `S = E[x x^T]`. Plain MSE is `||dW||^2_I`.
They are linked only through the spectrum of `S`:

    lambda_min(S) ||dW||_F^2  <=  ||dW||_S^2  <=  lambda_max(S) ||dW||_F^2

so cutting `||dW||_F^2` by a factor `g` certifies an output-error reduction only when
`g > 1 - 1/kappa(S)`. Measured on Llama-2-7B (4 wikitext batches, `quantize/importance.py`), the
median per-layer `max/mean` of `diag(S)` is ~97 and the median `max/min` is ~4.1e3. The criterion
therefore needs `g` above ~99%, while mix_4_6 achieves 0.002-2% over 4over6 at realizable type
blocks. **An aggregate MSE win at that magnitude certifies nothing about the output error**, which is
why it did not translate into perplexity.

Three knobs now exist on `quant_mix_4_6`:

- `importance=` -- per-input-channel `E[x_j^2]` from `collect_importance()`. Switches the selection
  loss to `sum_j S_jj dW_ij^2`, the diagonal-Hessian estimate of the layer output error (the
  GPTQ/OBQ objective). Uniform importance reproduces MSE exactly.
- `elect="dominance"` -- elect E0M3 only when it is no worse on EVERY scale block of the tile.
  Gives back the pointwise guarantee that 1x16 has for free (verified: 0 blocks harmed at 16x16,
  8x64, 32x128). Safe but fires almost never on large tiles, so it degenerates to 4over6.
- `elect="margin", margin=z` -- elect only when `mean(gain) > z * std(gain) / sqrt(B)`, a one-sided
  test of "the tile's advantage is real" against the block-to-block spread. `z=0` is the old
  behaviour; the rules are nested (`dominance` ⊆ `margin(large)` ⊆ ... ⊆ `argmin`).

On Llama-2-7B v_proj at 8x64, `margin=1` keeps 60% of the MSE gain while cutting harmed blocks from
17.3% to 4.9%; `margin=2` cuts them to 0.73%. `dominance` reaches 0% but elects nothing.

### How to decide E2M1 vs E0M3 (the answer)

**The answer is in two parts, and only the first one generalizes.**

**Part 1 -- widen the block-scale search. Do this unconditionally.** The block scale is
`alpha * block_max / grid_max`. FourOverSix is the two-point search `alpha in {1, 1.5}` on E2M1.
Extending it to `{1, 1.25, 1.5, 2, 3}` (preset `headx`) is **neutral-to-positive and never harmful**
across three models -- about -0.005 mean on Llama-3.1-8B and Llama-2-7B, and a wash (+0.0003 mean)
on Llama-3.2-3B. Take it because it is free and cannot hurt. It costs no metadata -- `alpha` only changes the value
written into the ue4m3 scale field that already exists -- and needs neither a type block nor the
E0M3 hardware path. It is plain NVFP4 with a wider FourOverSix, deployable on the existing kernel.

Why `alpha > 1` and not `alpha < 1`: writing the usable code values in units of the block maximum,

    alpha=1    block max -> code 6   {0,.083,.167,.25,.333,.5,.667,1}   log-spaced
    alpha=1.5  block max -> code 4   {0,.125,.25,.375,.5,.75,1}         FourOverSix
    alpha=2    block max -> code 3   {0,.167,.333,.5,.667,1}            uniform, 6 levels
    alpha=3    block max -> code 2   {0,.25,.5,.75,1}                   uniform, 4 levels

so headroom walks E2M1 from log-spaced-at-full-range to uniform-with-few-levels, discarding exactly
the sparse top of the grid. `alpha < 1` is clipping, which saturates the block maximum, and round 1
rejected it at +0.006 to +0.033 wikitext. Five candidates is where the search saturates -- eight
(`headxx`) is worse.

**Part 2 -- the E0M3 type block is a model-dependent extra.** On Llama-3.1-8B, adding E0M3 headroom
(`heade0`, E0M3 alphas `{1, 7/6, 7/5}`) and electing with `h1.5` is worth a further **-0.018
wikitext**, reaching -0.0265 / -0.0081 at 8x64. On Llama-2-7B the identical configuration is
**+0.0165 / +0.0061**, a loss. E0M3 with `alpha = 7/n` is exactly a uniform *n*-level grid, which is
the one thing E2M1 cannot supply above n=4, so when it helps it helps for a clear reason -- but
whether it helps is a property of the activations, and round 6 shows no calibration-free weight
statistic distinguishes the two models (E0M3 gain fraction 0.205 vs 0.199, identical).

If Part 2 is used anyway, the election rule is `h<lambda>`: elect E0M3 only when the gain the
winning scale blocks collect outweighs the damage the losing ones take by a factor `lambda`,

    sum_{gain>0} gain_b  >  lambda * sum_{gain<0} |gain_b|

which is the exact decision that survives any per-block importance `w_b` in `[1/kappa, kappa]` with
`lambda = kappa^2` (see `_elect_e0m3`). `lambda = 1` is plain argmin, `lambda -> inf` is dominance.

`lambda` is model-dependent and cannot be split the difference on. **`lambda` in [1.5, 2] is optimal
on Llama-3.1-8B and a real loss on BOTH Llama-2-7B (+0.0128) and Llama-3.2-3B (+0.0208);
`lambda = 3` is the only value measured that is non-harmful on all three**, and there it is worth
almost nothing (about -0.002 wikitext). Round 6
tried and failed to predict which regime a model is in from its weights alone.

Everything in `results/decide_r*/REPORT.md` reduces to one principle:

> **A rule of the form "do X when it lowers the quantization error" always loses. The same rule with
> "...by a decisive margin" wins.**

This showed up independently three times, on three unrelated mechanisms, and in the third case it
turned a rejected idea into one of the best configurations measured:

| mechanism | "when it helps" | "when it decisively helps" |
|---|---|---|
| elect E0M3 for a type block | `argmin`: +0.0021 / +0.0185 | `h1.5`: **-0.0117 / -0.0044** |
| rotate a column chunk | `rotcol`: +0.0946 / +0.1431 | `rotmin0.1`: **-0.0149 / -0.0025** |
| clip the block scale | any `alpha < 1`: +0.006 to +0.033 | `clipmin0.3`: **-0.0179 / -0.0159** |

(Llama-3.1-8B W4A16 at 8x64, against `nvfp4_4over6`.) The reason is the same each time: weight MSE
and the true layer output error agree on the *sign* of a large change and disagree freely on small
ones, so a criterion that fires on small gains is fitting noise with respect to the objective that
matters. Round 3 measures this directly -- rotation cuts weight MSE on every layer of Llama-2-7B,
but cuts the true output error only where the MSE gain is big (>15%: q/k/o_proj, -62%/-55%/-13%) and
*raises* it where the MSE gain is small (<6%: v_proj +73%, MLP +3..+8%).

Ranked, at 8x64 on Llama-3.1-8B W4A16, against `nvfp4_4over6`:

| | dwikitext | dc4 |
|---|---|---|
| `mix_4_6_rotmin0.1_h1.5` | -0.0149 | -0.0025 |
| `mix_4_6_h1.5` | -0.0117 | -0.0044 |
| `mix_4_6_m1` (margin z=1) | -0.0108 | -0.0037 |
| `mix_4_6_h2` | -0.0063 | -0.0066 |
| E2M1 only (`_e2m1`), i.e. 4over6 | +0.0001 | -0.0019 |
| `mix_4_6` (argmin) | +0.0021 | +0.0185 |
| `mix_4_6_e0m3` (always E0M3) | +0.0393 | +0.0546 |

For scale, `nvif4` (per-scale-block choice, not realizable) is -0.0387 and `razer_e3m3` is -0.0982.
So the best realizable rule keeps about **a quarter** of what per-block choice would give, and RaZeR
remains far ahead of the whole family.

### Ideas that were tried and do NOT work

- **Clipping the block scale** (`alpha < 1`, i.e. `block_max * 0.9 / grid_max`). Lowers the loss it
  optimizes and costs +0.006 to +0.033 wikitext; clipping E0M3 is worse than clipping E2M1, and
  `clipe0_m2` at 32x128 is the worst row measured. Consistent across 12 configs on Llama-2-7B.
- **MAE / Lp selection losses.** `mae` (= `l1`) is within 0.0005 of `mse`; `l0.5` and `l1.5` are
  worse. The squared-error criterion is not the thing to fix.
- **Row permutation** (`_perm`), sorting output channels by their E0M3 preference so tiles stop
  straddling the boundary. Every variant is worse than the same election rule without it. An 8x64
  tile is 8 rows x 4 scale blocks and the disagreement is mostly *within* a row across its k-blocks,
  so straddling stays near 100% however the rows are ordered. The axis that matters is K, and K
  cannot be permuted per tile without changing the GEMM.
- **Unconditional Hadamard rotation** (`_rot`, `_rotcol`). See round 3: +0.09 to +0.15 wikitext.
- **`corr<r>` and calibration-free `diag(S)` proxies.** See below.

### Two calibration-free objectives that were tried and do NOT work

Both were attempts to close the "MSE is the wrong criterion" gap without calibration data. Both are
cheap to re-derive and both fail for a measurable reason, so do not spend GPU time on them again.

**1. Coherent (correlated-input) error — `corr<r>`.** MSE cannot distinguish 16 errors of `+d` from
16 errors of alternating sign, yet the first accumulates in `sum_j x_j dW_ij` and the second cancels.
Modelling the inputs as equicorrelated, `S = sigma^2 [(1-r) I + r 11^T]`, gives the exact
calibration-free loss `sum_j dW_j^2 + r ((sum_j dW_j)^2 - sum_j dW_j^2)`. It is implemented and
tested (`metric="corr<r>"`).

It is a near no-op, because **the second term is empirically equal to the first**.
`analyze_coherent_error.py` on Llama-2-7B measures `(sum dW)^2 / sum dW^2` per scale block at
**0.998–1.005** for every grid and every clip preset. Round-to-nearest error is white at a
16-element block even under clipping, because only one or two elements per block are actually
clipped and the rest just re-round with random signs. So `coh - incoh` is ~0.3% of the loss and `r`
below 1 perturbs nothing. The hypothesis that "MSE-optimal clipping is overrated because it buys MSE
with coherent error" is **false**: clipping does not raise the coherent share at all.

Note this also disposes of the row-level version. If per-block error sums are independent then
`E[(sum_row dW)^2] = sum_blocks E[(sum_block dW)^2]`, which is exactly the quantity measured.

**2. Calibration-free proxies for `diag(S)`.** The `hess` variant works but needs calibration data.
Two proxies that need none were checked against the measured `E[x_j^2]` in
`results/mix_4_6_sweep/importance_llama-2-7b.pt` (Llama-2-7B, all 224 layers):

| proxy | q/k/v_proj | gate/up_proj | o_proj, down_proj |
|---|---|---|---|
| preceding RMSNorm `gamma^2` | Pearson +0.63, Spearman +0.37 | Pearson **-0.50** | not defined |
| weight column energy `\|\|W_:,j\|\|^2` | -0.54 to +0.65 | -0.72 to +0.01 | -0.19 |

`gamma^2` is the more interesting of the two -- it is a model weight, so it costs no data -- but it
points the **wrong way** on the MLP projections and is only defined for the five layers fed directly
by a norm. Weight column energy has no consistent sign at all. Neither is safe to weight a selection
loss by.

Certificate for the exact objective: with `D = diag(S)`, the diagonal surrogate errs by at most
`||S - D||_2 (||dW_A||_F^2 + ||dW_B||_F^2)`. Electing E0M3 only when the D-weighted gain exceeds
that bound guarantees an improvement in the true `||dW||^2_S`. That is the conservative version of
the margin rule, with `z` standing in for the unmeasured off-diagonal mass.
