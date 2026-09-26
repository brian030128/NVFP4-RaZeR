# MixFP4

<!-- CURATED MIXFP4 SUMMARY: keep experiment history in supporting reports. -->

## 1. What MixFP4 is

MixFP4 is NVFP4 with one extra choice per weight tile: the 4-bit element type.

- **Unchanged from NVFP4:** 4-bit storage, one FP8 E4M3 block scale per 16
  elements along K, and one FP32 per-tensor global scale (`amax / (6·448)`).
- **The per-tile choice:** every weight **type tile** of `rows × 64` elements
  (rows = output channels, 64 = one MMA K-block) is decoded as either
  - **E2M1**, the standard FP4 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, with
    FourOverSix block scaling (per-16-element choice of block max → 6 or → 4); or
  - **E0M3**, the uniform grid `{0, ±1, …, ±7}` (equivalent to signed INT4), with
    block scale `block_max / 7`.
- **What a tile shares:** every 16-element scale block inside a tile uses the
  tile's type and keeps its own E4M3 scale. Both types encode 15 values in 4 bits
  and use the same scale format, so no extra per-element metadata is stored.
- **Activations** stay E2M1 (FourOverSix).

**Hardware path.** The Blackwell block-scaled FP4 MMA reads an operand as E2M1 or
E0M3 from its instruction format field, so the type is a per-operand, per-MMA
choice. E0M3 is an undocumented encoding.

| Path | Weight type tile | How the type is selected |
|---|---|---|
| SM100 (B200/GB200, `tcgen05.mma`) | **256×64**: the kernel's MMA tile N (256) × one 64-K block | Operand-format field of the MMA descriptor, rewritten per K-block |
| SM120 (`mma.sync …m16n8k64`) | **8×64** minimum: the weight (B) operand tile n8 × k64 | Compiled E0M3 instruction variants, dispatched per MMA |

The two geometries reported here are platform-specific: **256×64 is the SM100
path and 8×64 is the SM120 path.**

## 2. GEMM overhead: 8×64 and 256×64

All overheads are relative to the same GEMM with every weight tile in E2M1
(plain NVFP4 arithmetic).

| Type tile | Platform | Setting | Overhead |
|---|---|---|---|
| 256×64 | GB200 (SM100) | 8192³, uniform E0M3 vs uniform E2M1 weights, same executable | **−0.012%** (launch) / **−0.093%** (CUDA graph): within run-to-run noise, i.e. ≈ 0 |
| 256×64 | GB200 (SM100) | Heterogeneous per-tile maps from §3 | *not yet measured* |
| 8×64 | SM120 | Heterogeneous per-tile maps | *to be provided* |

The 256×64 uniform result comes from job 400605 (median of three samples,
256×256×256 GEMM tile, BF16 output, FP32 accumulation, PDL on):
[graph samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_graph.json),
[launch samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_launch.json).
It shows that switching the type costs nothing on SM100. It does not time a
realistic mixed map.

## 3. How tile selection works: multi-round KL-only election

> **Newer alternative (Llama-3.1-8B and three Llama Instruct models so far):** training the map directly with Adam
> on the same KL loss, with no filter and no backtracking, beats this method on
> both datasets at both tile sizes. See §7.

Each tile is either FourOverSix E2M1 (the default) or E0M3. Selection is a greedy
descent on a distillation loss. First-order scores propose which tiles to flip, and
measured loss on held-out documents decides how many flips to take. It works like
gradient descent with a backtracking line search, over a discrete set of moves.

**Data.**
- 128 calibration sequences of 512 tokens: 64 OpenWebMath and 64 CodeParrot.
- 192 separate held-out math/code **development** documents of 512 tokens.
- No WikiText or C4 is used anywhere in selection.

**Weights as a function of the map.** For each weight matrix $W$, two quantized
candidates are computed once from the BF16 weights and never change:
- $B$, the FourOverSix E2M1 quantization;
- $A$, the E0M3 quantization with $\alpha = 1$.

Both share the tensor's FP32 global scale, and each 16-element scale block keeps
its own E4M3 scale. A map $m \in \{0,1\}^{\text{tiles}}$ gives the weights

$$\hat W(m) = B + \sum_{u:\,m_u = 1} P_u \odot (A - B),$$

where $P_u$ is the 0/1 mask of tile $u$ (256×64 or 8×64 elements). Flipping tile
$u$ changes only the entries of that tile, by

$$\Delta W_u = \sigma_u\, P_u \odot (A - B), \qquad
\sigma_u = \begin{cases} +1 & u \text{ is currently E2M1} \\ -1 & u \text{ is currently E0M3 (an undo)} \end{cases}$$

**Objective.** Let $p_t$ be the BF16 teacher's next-token distribution at position
$t$, and $q_{m,t}$ that of the W4A4 model with map $m$. For a sequence $s$ of $T$
tokens,

$$\mathrm{KL}_s(m) = \frac{1}{T-1} \sum_{t=1}^{T-1} \sum_{v \in \text{vocab}}
p_t(v)\, \big[\log p_t(v) - \log q_{m,t}(v)\big].$$

The development objective is the mean over the 192 development documents:
$L_{\text{dev}}(m) = \frac{1}{192} \sum_d \mathrm{KL}_d(m)$. It is measured with
the evaluation protocol: a forward pass only, with one tensor-wide FourOverSix
activation scale per document.

**Step 1: score every flip with one backward pass.** The score of flipping tile $u$
is its first-order effect on each calibration sequence's KL. A first-order Taylor
expansion at the current weights $\hat W$ gives

$$\mathrm{KL}_s\big(\hat W + \Delta W_u\big) \approx \mathrm{KL}_s(\hat W) + g_{u,s},
\qquad g_{u,s} = \langle G_s, \Delta W_u \rangle = \sum_{(i,j) \in u} (G_s)_{ij}\, (\Delta W_u)_{ij},$$

where $G_s = \partial\, \mathrm{KL}_s / \partial \hat W$ is sequence $s$'s gradient
with respect to that layer's weights. The scoring pass computes every $g_{u,s}$
without storing a single weight gradient.

*Forward pass, for each batch of calibration sequences at the current map $m$:*

1. **Teacher.** The BF16 teacher's log-probabilities $\log p_t$ were computed once,
   before any quantization, and are reused in every round.
2. **Weights.** Every text linear layer holds $\hat W(m)$, decoded from the packed
   candidates. The weights are frozen (`requires_grad=False`), so autograd never
   builds or stores a weight gradient.
3. **Graph.** The token embeddings are detached and marked `requires_grad`. The
   backward pass then flows through the activations of every layer, even though no
   parameter requires a gradient.
4. **Activation quantization with a straight-through estimator.** A pre-hook on each
   linear layer replaces its input $x$ with

   $$\tilde x = Q(x) + \big(x - \operatorname{sg}(x)\big),$$

   where $\operatorname{sg}$ is stop-gradient (`detach`). The value is $Q(x)$, but
   the Jacobian is the identity: $\partial \tilde x / \partial x = I$. Here $Q$ is
   FourOverSix with one FP32 factor per token (`quantize_rows`). Each token gets
   its own global scale $\max|x_t| / (6 \cdot 448)$. Each 16-element block then
   picks the E4M3 block scale that maps its maximum to 6 or to 4, whichever has the
   smaller block squared error. Every token is quantized from its own values only.
   This keeps the scoring forward causal.
5. **Save the layer input.** A forward hook on each linear layer keeps its quantized
   input $\tilde x_s \in \mathbb R^{T \times K}$ for each sequence. It also
   registers a hook on the layer's output $y$, which fires during the backward pass.
6. **Loss.** For each sequence, $\mathrm{KL}_s$ is its token-mean KL against the
   teacher, as defined above. The batch loss is $\sum_s \mathrm{KL}_s$.

*Backward pass: one `backward()` per batch.* When it reaches a linear layer, the
output hook receives $\delta = \partial \big(\sum_s \mathrm{KL}_s\big) / \partial y$.
Sequences in a batch never interact: attention is within a sequence and activation
scales are per token. Sequence $s$'s loss therefore reaches only its own slice
$\delta_s \in \mathbb R^{T \times N}$. The hook then does four things, entirely in
FP32:

1. **Per-sequence weight gradient.** Because $y_{s,t} = \hat W \tilde x_{s,t}$, the
   gradient sums outer products over the sequence's tokens, i.e. one matmul:

   $$G_s = \sum_{t} \delta_{s,t}\, \tilde x_{s,t}^{\top} = \delta_s^{\top} \tilde x_s \in \mathbb R^{N \times K}.$$

   $\delta_s$ already contains the effect of this layer's output on every later
   layer and on the logits. The score therefore measures a weight change's effect
   on the model's output distribution, not the local reconstruction error of the
   layer.
2. **Flip direction for the whole matrix.** For every element,
   $D = (A - B) \odot \Sigma$, where $\Sigma$ is $-1$ on currently-E0M3 tiles and
   $+1$ elsewhere. $D$ restricted to tile $u$ is exactly $\Delta W_u$, so a single
   elementwise product serves every tile.
3. **Tile sums.** $E_s = G_s \odot D$ is summed within each tile. For $r \times c$
   tiles, pad the rows up to a multiple of $r$, reshape to
   $(\lceil N/r \rceil, r, K/c, c)$, and sum over the two within-tile axes
   (`reduce`). The result is a $\lceil N/r \rceil \times K/c$ array holding
   $g_{u,s} = \sum_{(i,j) \in u} (E_s)_{ij}$ for every tile of the layer.
4. **Accumulate.** Add $g_{u,s}$ and $g_{u,s}^2$ into FP64 running sums, one pair
   per tile. $G_s$ and $E_s$ are discarded immediately.

Every linear layer's hook fires within the same backward pass, so one forward and
one backward per batch score every tile of every layer. In total that's 128
sequences (16 batches of 8 on Llama, 128 of 1 on Qwen). What persists is two FP64
numbers per tile. Testing each flip directly would instead need one forward pass
per tile, millions of them at 8×64.

In pseudocode, for one linear layer:

```
forward:   x̃ = Q_per_token(x).detach() + (x - x.detach())      # value Q(x), gradient identity
           y = Ŵ x̃ ;  save x̃ ;  y.register_hook(on_backward)
on_backward(δ):                                                  # δ = ∂ Σ_s KL_s / ∂y
           D = (A - B) * where(tile_is_E0M3, -1, +1)             # flip direction, all tiles
           for s in batch:
               G = δ[s].T @ x̃[s]                                  # N×K gradient of KL_s
               g = tile_sum(G * D)                                # ⌈N/r⌉ × K/c scores g_{u,s}
               sum_g += g ; sum_g2 += g**2                        # FP64
after 128 sequences:
           μ  = sum_g / S
           SE = sqrt((sum_g2 - S μ²) / (S - 1)) / sqrt(S)
```

*What the score is not.* It is exact for the model it differentiates, but that
model differs from the one that is deployed and evaluated in two ways:
- the straight-through estimator ignores how a weight change moves the activation
  quantization grid;
- scoring uses per-token activation scales, while the development evaluation, like
  deployment, uses one tensor-wide scale per document.

Both gaps, and the finite size of each flip, are why a score only *proposes* a flip.
Step 3 decides with the measured development KL under the evaluation protocol.

**Step 2: keep flips predicted to help, with confidence.** Over the $S = 128$
calibration sequences, take each tile's mean score and its standard error:

$$\mu_u = \frac{1}{S} \sum_s g_{u,s}, \qquad
\mathrm{SE}_u = \frac{1}{\sqrt S} \sqrt{\frac{1}{S-1} \sum_s \big(g_{u,s} - \mu_u\big)^2}.$$

Tile $u$ is a candidate only if

$$b_u = \mu_u + 2\, \mathrm{SE}_u < 0.$$

$b_u$ is an approximate one-sided 97.7% upper confidence bound on the expected
first-order change in KL per sequence. The filter therefore keeps flips that lower
KL across the calibration sequences, not ones driven by a few sequences. This is
the "decisive margin" principle from earlier rounds of this work. Candidates are
ranked by $b_u$, most negative first: $c_1, c_2, \dots, c_M$.

**Step 3: choose the step size by backtracking on measured loss.** Try
$n = M, \lfloor M/2 \rfloor, \lfloor M/4 \rfloor, \dots, 1$, with at most 22 tries:
1. Flip the top $n$ candidates together, giving map $m'$.
2. Measure $L_{\text{dev}}(m')$.
3. If $L_{\text{dev}}(m') < L_{\text{dev}}(m)$, accept $m'$ and end the round.
   Otherwise undo the flips and halve $n$.

Each try logs the predicted change $\sum_{i \le n} \mu_{c_i}$ next to the measured
change.

**Step 4: re-score at the new point and stop when nothing helps.** The next round
repeats steps 1–3 at the accepted map, with fresh gradients. The loop stops when no
tile passes the filter, or when even $n = 1$ fails to lower $L_{\text{dev}}$. An
accepted step always lowers $L_{\text{dev}}$, so development KL decreases strictly
from round to round.

**Why multiple rounds: the linear prediction overstates a combined step.** For a
step $\Delta = \sum_{u \in C} \Delta W_u$, the second-order expansion is

$$\mathrm{KL}(\hat W + \Delta) \approx \mathrm{KL}(\hat W)
+ \underbrace{\sum_{u \in C} \langle G, \Delta W_u \rangle}_{\text{what the scores add up}}
+ \tfrac12 \sum_{u \in C} \sum_{v \in C} \mathrm{vec}(\Delta W_u)^{\top} H\, \mathrm{vec}(\Delta W_v),$$

where $H$ is the Hessian of KL with respect to all weights. The scores capture only
the linear term, which grows like $n$. The quadratic term has $n^2$ pairwise terms:
- tiles in the same layer interact through shared inputs;
- tiles in different layers interact because each layer's error changes what later
  layers see.

Each flip is also a finite jump (the full E2M1-to-E0M3 difference on its tile), not
an infinitesimal step. The sum of scores therefore overstates the combined effect.
First-round tries from the reported runs:

| Run, first round | Flips | Predicted Δ dev KL | Measured Δ dev KL | Result |
|---|---:|---:|---:|---|
| Llama 8×64, all candidates | 412,228 | −0.405 | **+0.389** | rejected |
| Llama 8×64, 6 halvings | 6,441 | −0.043 | **+0.011** | rejected |
| Llama 8×64, 7 halvings | 3,220 | −0.032 | −0.0014 | accepted (23× smaller than predicted) |
| Qwen 8×64, 6 halvings | 17,215 | −0.024 | −0.0011 | accepted (22× smaller) |
| Qwen 256×64, all candidates | 39,093 | −0.064 | −0.0024 | accepted (27× smaller) |

Even accepted steps realize only about 1/20 to 1/30 of the predicted gain. Large
steps cross over to a net loss, because the quadratic term grows faster than the
linear one.

Electing every candidate in one step (one-shot election) overshoots. Once a step is
taken, the gradients $G_s$ change, and so do the right flips. Re-scoring at every
accepted map, with the step size set by measured loss, avoids both problems. Each
round's candidate count and accepted flips are in the run reports; for example,
Llama 8×64 accepted 3,220, 1,208, 826, 423, 199, 91, 5 and 5 flips before stopping.

**Output.** A per-tile E2M1/E0M3 map, the only runtime artifact. Selection is
deterministic for a fixed setup: an independent re-run reproduced the Llama map
and every evaluation loss bitwise. Implementation: `run_multiround.py
--objective kl`.

## 4. Calibration time and cost

These are **one-time, offline tile-selection costs** of the optimized
implementation that produced every MixFP4 map in §5 and §6. Inference memory is
not reported: this is fake quantization, and peak inference memory will be
measured on the target device.

| Run | Hardware | Scoring passes | Dev evaluations | Setup | Selection time | Peak GPU memory | Peak CPU memory |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B, 256×64 | 1× H200 | 4 | 33 | 1.5 min | **15.3 min** | 64.5 GiB | 42.9 GiB |
| Llama-3.1-8B, 8×64 | 1× H200 | 9 | 118 | 1.8 min | **45.5 min** | 63.7 GiB | 42.9 GiB |
| Qwen3.8-27B, 256×64 | 1× H200 | 3 | 28 | 6.0 min | **1 h 16 min** | 106.6 GiB | 78.1 GiB |
| Qwen3.8-27B, 8×64 | 1× H200 | 10 | 143 | 6.0 min | **5 h 34 min** | 110.7 GiB | 78.2 GiB |

- **Selection time** covers all scoring passes and development evaluations. It
  excludes setup (loading model, teachers and packed candidates) and the final PPL
  evaluation.
- **Scoring pass:** 128 × 512 tokens, forward plus the KL backward.
- **Development evaluation:** 192 × 512 tokens, forward only. These dominate late
  rounds, where backtracking evaluates several step sizes per accepted step.

**What makes it fast**, none of it changing a weight value:
- Both candidates are stored packed as 4-bit codes plus FP8 scales, decoded per
  module and verified bitwise. This is what lets Qwen run on one GPU.
- Activation fake-quantization is vectorized and bitwise-verified at the start of
  every run.
- The CE backward is skipped, because KL-only selection never uses it.
- Llama batches 16 documents per evaluation and 8 sequences per scoring pass.
  Qwen uses one document per pass, because its batched forward is not
  numerically identical to one-at-a-time.

**Speed-up over the earlier reference implementation**, which used unpacked BF16
candidates, looped quantization and a CE backward:

| Run | Reference | Optimized |
|---|---:|---:|
| Llama 256×64 | 43.9 min | **15.3 min (2.9×)** |
| Llama 8×64 | 2 h 15 min | **45.5 min (3.0×)** |
| Qwen 8×64 | 8 h 20 min on 2× H200 | **5 h 34 min on 1× H200** (≈3× fewer GPU-hours) |
| Qwen 256×64 round 0 | 671 s on 2× H200 | **586 s on 1× H200** |

## 5. Perplexity

W4A4 fake quantization: weights as listed, activations FourOverSix (NVFP4
activations for the NVFP4 row). Evaluation uses the released protocol: WikiText-2
test in 2,048-token windows and 256 seed-0 C4 validation crops. Lower is better.
MixFP4 rows are the converged maps from the optimized calibration of §4. Paired
ΔNLL is per window versus FourOverSix, ± 2 SE.

### Llama-3.1-8B

| Policy | E0M3 tiles | WikiText-2 | C4 | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---|
| BF16 (reference) | — | 6.240087 | 8.958212 | — |
| NVFP4 | 0 | 6.940252 | 9.925099 | — |
| NVFP4 FourOverSix | 0 | 6.875525 | 9.823733 | — |
| **MixFP4 8×64** (SM120) | 3,645 | **6.817620** | **9.759931** | −0.00846±0.00171 / −0.00652±0.00203 |
| **MixFP4 256×64** (SM100) | 8,393 | **6.835411** | **9.771621** | −0.00585±0.00187 / −0.00532±0.00220 |

Versus FourOverSix:
- **MixFP4 8×64:** −0.0579 WikiText / −0.0638 C4.
- **MixFP4 256×64:** −0.0401 / −0.0521.
- Trained maps (§7) reach −0.0908 / −0.1483 (8×64) and −0.0700 / −0.1002 (256×64).

### Qwen3.8-27B

| Policy | E0M3 tiles | WikiText-2 | C4 | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---|
| BF16 (reference) | — | 7.050375 | 9.893323 | — |
| NVFP4 | 0 | 7.579994 | 10.230958 | — |
| NVFP4 FourOverSix | 0 | 7.287076 | 10.188365 | — |
| **MixFP4 8×64** (SM120) | 17,571 | **7.205417** | **10.148049** | −0.01127±0.00442 / −0.00396±0.00089 |
| **MixFP4 256×64** (SM100) | 39,095 | **7.223045** | **10.152189** | −0.00883±0.00361 / −0.00356±0.00089 |

Versus FourOverSix:
- **MixFP4 8×64:** −0.0817 WikiText / −0.0403 C4.
- **MixFP4 256×64:** −0.0640 / −0.0362.

**Run-to-run variation.** Earlier reference-implementation runs of the same
algorithm give the spread caused by floating-point summation order, which moves
borderline tiles and compounds over rounds.

| Model | Tile | Optimized run (above) | Reference runs, ΔPPL vs FourOverSix (wiki / c4) |
|---|---|---|---|
| Llama | 8×64 | −0.0579 / −0.0638 | −0.0558 / −0.0732 |
| Llama | 256×64 | −0.0401 / −0.0521 | −0.0335 / −0.0496 |
| Qwen | 8×64 | −0.0817 / −0.0403 | −0.1207 / −0.0326 |
| Qwen | 256×64 | −0.0640 / −0.0362 | −0.0402 / −0.0311 and −0.0856 / −0.0391 |

- **Llama** is stable to within about ±0.01.
- **Qwen varies by up to ±0.04 on WikiText.** Its reference 8×64 map also scored
  better after round 4 than when converged, so late rounds can overfit the 192
  development documents.
- The paired ±2 SE in the tables above does not include this selection variance.

## 6. Zero-shot accuracy

lm-eval 0.4.5, 0-shot: `arc_easy`, `arc_challenge`, `hellaswag`, `openbookqa`,
`boolq`, `winogrande`. The metric is `acc_norm` where defined, else `acc`, and the
mean is unweighted over the six tasks. Activation fake-quantization uses one
tensor-wide scale **per document**, not per lm-eval batch; padding positions are
included in a document's scale. Batch size is 64 for every row, and the MixFP4
rows use the §5 maps (jobs 428890, 428891, 430876). Paired differences are
computed per document within each task and averaged over the six tasks, ± 2 SE.

### Llama-3.1-8B

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference) | 0.8123 | 0.5367 | 0.7884 | 0.4460 | 0.8196 | 0.7356 | 0.6898 |
| NVFP4 | 0.7500 | 0.5111 | 0.7754 | 0.4600 | 0.7920 | 0.7135 | 0.6670 |
| NVFP4 FourOverSix | 0.7567 | 0.5111 | 0.7776 | 0.4420 | 0.8049 | 0.7072 | 0.6666 |
| **MixFP4 8×64** (SM120) | 0.7757 | 0.5111 | 0.7787 | 0.4380 | 0.8141 | 0.7135 | **0.6718** |
| **MixFP4 256×64** (SM100) | 0.7740 | 0.5077 | 0.7751 | 0.4440 | 0.8083 | 0.7080 | **0.6695** |

| Comparison | Δ mean accuracy ± 2 SE |
|---|---|
| MixFP4 8×64 − FourOverSix | +0.0053 ± 0.0065 |
| MixFP4 8×64 − NVFP4 | +0.0048 ± 0.0070 |
| MixFP4 256×64 − FourOverSix | +0.0029 ± 0.0065 |
| MixFP4 256×64 − NVFP4 | +0.0025 ± 0.0070 |

### Qwen3.8-27B

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference) | 0.7306 | 0.5887 | 0.8288 | 0.4640 | 0.8657 | 0.7561 | 0.7057 |
| NVFP4 | 0.7496 | 0.5700 | 0.8227 | 0.4320 | 0.7700 | 0.7380 | 0.6804 |
| NVFP4 FourOverSix | 0.7273 | 0.5725 | 0.8242 | 0.4540 | 0.8061 | 0.7514 | 0.6893 |
| **MixFP4 8×64** (SM120) | 0.7306 | 0.5708 | 0.8231 | 0.4520 | 0.8076 | 0.7466 | **0.6885** |
| **MixFP4 256×64** (SM100) | 0.7462 | 0.5922 | 0.8227 | 0.4540 | 0.8119 | 0.7514 | **0.6964** |

| Comparison | Δ mean accuracy ± 2 SE |
|---|---|
| MixFP4 8×64 − FourOverSix | −0.0008 ± 0.0062 |
| MixFP4 8×64 − NVFP4 | **+0.0081 ± 0.0066** |
| MixFP4 256×64 − FourOverSix | **+0.0071 ± 0.0062** |
| MixFP4 256×64 − NVFP4 | **+0.0160 ± 0.0066** |

**Reading.**
- MixFP4 never loses accuracy to FourOverSix or NVFP4 beyond noise.
- **Significant gains:**
  - Qwen 256×64 beats both baselines, mostly on arc_challenge (+0.020) and
    arc_easy (+0.019).
  - Qwen 8×64 beats NVFP4.
- **Within noise:** the Llama gains (+0.003 to +0.005) and Qwen 8×64 versus
  FourOverSix. These comparisons are underpowered: 2 SE is ±0.006–0.007 on the
  six-task mean.
- PPL gains (§5) and accuracy gains do not rank the same way. On Qwen, 8×64 has
  the larger PPL gain but the smaller accuracy gain.

## 7. Training the map with an optimizer (no backtracking)

This treats the map as trainable parameters and optimizes the §3 KL objective
directly, like training, instead of electing flips in rounds. Everything else
is the same as §3: candidates $A$ and $B$, the 128 calibration sequences, the
BF16 teacher, the scoring forward pass (per-token FourOverSix activations with
a straight-through estimator), and the evaluation windows. So far it has been
run on Llama-3.1-8B (jobs 441206–441208), Llama-3.1-8B-Instruct
(442024, 442074, 442075), Llama-3.2-3B-Instruct (442077–442079) and
Llama-3.2-1B-Instruct (442046–442048).

**Parametrization.** Each tile $u$ gets a real latent logit $\theta_u$, and the
weights are $\hat W(\theta) = B + \sum_u m_u(\theta_u)\, P_u \odot (A - B)$.
- **STE (the method):** $m_u = \mathbb 1[\theta_u > 0]$ in the forward pass, so
  training always runs the deployable hard map. The backward pass uses
  $\partial m_u / \partial \theta_u = 1$ (BinaryConnect-style latent weights).
  Init $\theta = -1$, lr 0.02.
- **Sigmoid (ablation, Llama-3.1-8B 8×64 only):**
  $m_u = \sigma(\theta_u / \tau)$, with $\tau$ annealed geometrically from 1 to
  0.1, and the map rounded at $\theta > 0$ for evaluation. Init $\theta = -3$,
  lr 0.05. It tied STE within noise and adds a temperature schedule, so it was
  dropped.

**Gradient.** A backward hook on each linear layer forms $G = \delta^\top \tilde x$
for the minibatch. It hands the optimizer
$\partial \mathrm{KL} / \partial m_u = \sum_{(i,j) \in u} G_{ij} (A - B)_{ij}$,
which is the §3 tile score without the flip sign, computed on a minibatch.
No weight gradient is stored.

**Optimizer.**
- Adam with $\beta = (0.9, 0.999)$ and $\epsilon = 10^{-12}$. Tile gradients
  are about $10^{-6}$, so the default $10^{-8}$ would dominate the update.
- Constant learning rate, no weight decay.
- Batch 8, 16 steps per epoch, 20 epochs (320 steps).
- **No candidate filter, no backtracking, no acceptance test.** The development
  documents are evaluated every 2 epochs to monitor training only; the map
  reported is the last epoch's.

Adam moves each logit by about lr per step, so the starting margin works like a
soft significance threshold. No tile flips during the first 3 epochs. A tile
whose gradient keeps one sign crosses zero, while one dominated by noise barely
moves.

**Results, Llama-3.1-8B.** ΔPPL is versus FourOverSix. ΔNLL is paired per
window, ± 2 SE, versus the §5 multi-round maps; the multi-round rows reproduce
§5's paired numbers exactly.

| Policy | E0M3 tiles | final dev KL | WikiText-2 | C4 | ΔPPL vs FourOverSix (wiki / c4) | ΔNLL vs multi-round (wiki / c4) |
|---|---:|---:|---:|---:|---|---|
| Multi-round 256×64 (§5) | 8,393 | 0.09572 | 6.835411 | 9.771621 | −0.0401 / −0.0521 | — |
| **Trained STE 256×64** | 38,176 | 0.08868 | **6.805528** | **9.723484** | **−0.0700 / −0.1002** | −0.00438±0.00148 / −0.00494±0.00193 |
| Multi-round 8×64 (§5) | 3,645 | 0.09374 | 6.817620 | 9.759931 | −0.0579 / −0.0638 | — |
| **Trained STE 8×64** | 329,837 | 0.08349 | **6.784682** | **9.675402** | **−0.0908 / −0.1483** | −0.00484±0.00142 / −0.00870±0.00240 |
| **Trained sigmoid 8×64** | 212,778 | 0.08418 | **6.781821** | **9.681478** | **−0.0937 / −0.1423** | −0.00526±0.00152 / −0.00807±0.00226 |

Every trained map beats multi-round significantly, on both datasets and at both
tile sizes. At 8×64 the C4 gain over FourOverSix more than doubles.

**Results, Llama Instruct models.** STE only, with the same settings as
Llama-3.1-8B. All three models share the Llama-3 tokenizer, so the same
calibration windows, development documents and released evaluation windows
apply; the calibration loader re-checks every window's token hash. Each
FourOverSix row is the all-E2M1 map from the same pipeline (a zero-epoch run),
which gives matched per-window NLLs. ΔNLL is paired per window versus
FourOverSix, ± 2 SE. Multi-round maps exist only for 8B-Instruct (jobs 431425
and 431426); their rows are paired against the same FourOverSix run.

| Model | Policy | E0M3 tiles | final dev KL | WikiText-2 | C4 | ΔPPL vs FourOverSix (wiki / c4) | ΔNLL vs FourOverSix (wiki / c4) |
|---|---|---:|---:|---:|---:|---|---|
| Llama-3.1-8B-Instruct | NVFP4 FourOverSix | 0 | 0.10870 | 7.814643 | 11.264191 | — | — |
| | Multi-round 256×64 | 20,993 (4.9%) | 0.08763 | 7.794221 | 11.266143 | −0.0204 / +0.0020 | −0.00262±0.00147 / +0.00017±0.00164 |
| | **Trained STE 256×64** | 44,763 (10.5%) | 0.08036 | **7.710097** | **11.174994** | **−0.1045 / −0.0892** | −0.01347±0.00173 / −0.00795±0.00164 |
| | Multi-round 8×64 | 32,288 (0.24%) | 0.08303 | 7.706456 | 11.162905 | −0.1082 / −0.1013 | −0.01394±0.00174 / −0.00903±0.00239 |
| | **Trained STE 8×64** | 365,992 (2.7%) | 0.07805 | **7.699191** | **11.107474** | **−0.1155 / −0.1567** | −0.01488±0.00178 / −0.01401±0.00274 |
| Llama-3.2-3B-Instruct | NVFP4 FourOverSix | 0 | 0.09620 | 11.945266 | 15.507828 | — | — |
| | **Trained STE 256×64** | 29,408 (17.1%) | 0.08081 | **11.543218** | **15.158546** | **−0.4020 / −0.3493** | −0.03424±0.00267 / −0.02278±0.00255 |
| | **Trained STE 8×64** | 245,646 (4.5%) | 0.07677 | **11.401570** | **15.112609** | **−0.5437 / −0.3952** | −0.04658±0.00264 / −0.02582±0.00209 |
| Llama-3.2-1B-Instruct | NVFP4 FourOverSix | 0 | 0.16656 | 15.356671 | 21.532633 | — | — |
| | **Trained STE 256×64** | 17,064 (28.7%) | 0.12436 | **14.717933** | **20.049759** | **−0.6387 / −1.4829** | −0.04248±0.00298 / −0.07135±0.00392 |
| | **Trained STE 8×64** | 138,907 (7.3%) | 0.10905 | **14.505991** | **19.653187** | **−0.8507 / −1.8794** | −0.05699±0.00275 / −0.09133±0.00416 |

Tile totals: 8B 425,984 (256×64) / 13,631,488 (8×64); 3B 172,032 / 5,505,024;
1B 59,392 / 1,900,544.

- Every trained map beats FourOverSix significantly on both datasets. The gain
  grows as the model shrinks: 8×64 WikiText ΔNLL is −0.015 (8B), −0.047 (3B)
  and −0.057 (1B).
- **Versus multi-round on 8B-Instruct**, paired directly:
  - 256×64: trained is better by −0.01085±0.00155 on WikiText and
    −0.00812±0.00121 on C4. Multi-round's 256×64 map gained almost nothing over
    FourOverSix on this model.
  - 8×64: a tie on WikiText (−0.00094±0.00143) and better on C4
    (−0.00498±0.00137).
- Training takes 18.4 / 21.8 min (8B-Instruct), 10.3 / 10.3 min (3B) and
  5.4 / 5.6 min (1B) at 256×64 / 8×64 on one H200, including the monitor-only
  dev evaluations.
- Dev KL is roughly flat over the last 6 epochs for 3B and 1B, and on
  8B-Instruct 8×64. It was still falling for 8B-Instruct 256×64.

**Calibration time, Llama-3.1-8B.** One H200 per run; setup and final PPL
evaluation are excluded, as in §4.

| Run | Selection time | Work |
|---|---:|---|
| Multi-round 256×64 | 15.3 min | 4 scoring passes + 33 dev evaluations |
| Trained STE 256×64 | 19.2 min | 20 epochs + 12 dev evaluations |
| Multi-round 8×64 | 45.5 min | 9 scoring passes + 118 dev evaluations |
| Trained STE / sigmoid 8×64 | 19.1 / 19.9 min | 20 epochs + 12 dev evaluations |

Training cost does not depend on tile size: it is a fixed 20 epochs of about
46 s each. The 12 monitor-only dev evaluations take about 4 minutes; without
them, training takes about 15.5 min.

**Caveats.**
- One seed and one hyperparameter setting per arm. The ± 2 SE does not include
  selection variance.
- On Llama-3.1-8B, 256×64 had not converged: dev KL was still falling at
  epoch 20.
- On Llama-3.1-8B, 8×64 STE dev KL bottomed at epoch 16 (0.08318) and ended at
  0.08349, while train KL fell to 0.039. This is a mild sign of fitting the
  calibration set.
- Trained maps elect many more E0M3 tiles: on Llama-3.1-8B, 4.5× (256×64) to
  90× (8×64) more than multi-round. The GEMM cost of heterogeneous maps is
  still unmeasured (§2).
- Qwen3.8-27B runs were started and then stopped before finishing, so they have
  no results. Zero-shot accuracy has not been run on trained maps.

### Control: NVFP4 block-scale search trained with the same KL loss (FourOverSix init)

Does the gain come from E0M3, or from training *any* binary weight choice on the KL
loss? The control keeps every weight E2M1 and replaces candidate $A$ with the
FourOverSix candidate that uses the **other** block scale (block max → 6 ↔ block
max → 4) in every 16-element scale block (`--alt scale`,
`quant_nvfp4_4over6_pair`). Everything else is the same as above: the same B
(bitwise FourOverSix; initial dev KL is identical), STE, Adam, lr 0.02, init −1,
20 epochs, and the same data and evaluation windows.
- **1×16** is the natural NVFP4 version: one logit per scale block. It is plain
  NVFP4 on existing kernels, because only the value written into the existing
  ue4m3 scale changes, so it needs no type metadata.
- **8×64 and 256×64** use the MixFP4 geometries: every block in a tile flips its
  scale together.

Llama-3.2-1B-Instruct, jobs 442285–442287, 4.6 min of training each on one H200.
ΔNLL is paired per window, ± 2 SE.

| Policy | switched units | final dev KL | WikiText-2 | C4 | ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---:|---|
| NVFP4 FourOverSix | 0 | 0.16656 | 15.356671 | 21.532633 | — |
| MixFP4 trained STE 256×64 | 17,064 | 0.12436 | 14.717933 | 20.049759 | −0.04248±0.00298 / −0.07135±0.00392 |
| NVFP4 KL scale search 256×64 | 15,125 | 0.12643 | 14.778975 | 20.190540 | −0.03834±0.00285 / −0.06436±0.00346 |
| MixFP4 trained STE 8×64 | 138,907 | 0.10905 | 14.505991 | 19.653187 | −0.05699±0.00275 / −0.09133±0.00416 |
| NVFP4 KL scale search 8×64 | 134,717 | 0.11029 | 14.461707 | 19.689047 | −0.06005±0.00290 / −0.08951±0.00410 |
| **NVFP4 KL scale search 1×16** | 1,170,936 (1.9%) | 0.11092 | **14.443117** | **19.451303** | −0.06133±0.00307 / −0.10166±0.00469 |

Direct paired comparisons (NVFP4 scale search − MixFP4 trained, ΔNLL ± 2 SE, wiki / c4):

| Scale search | vs MixFP4 256×64 | vs MixFP4 8×64 |
|---|---|---|
| 256×64 | +0.00414±0.00239 / +0.00700±0.00189 (MixFP4 better) | +0.01864±0.00236 / +0.02697±0.00195 |
| 8×64 | −0.01756±0.00263 / −0.01815±0.00182 | −0.00306±0.00209 / +0.00182±0.00144 (tie-ish) |
| 1×16 | −0.01885±0.00243 / −0.03030±0.00201 | **−0.00434±0.00214 / −0.01033±0.00166** (NVFP4 better) |

**Reading (1B only, one seed):**
- Most of the §7 gain over FourOverSix comes from KL-training the choice, not from
  E0M3. The E2M1-only control recovers at least 90% of the MixFP4 gain at the same tile (and more
  than 100% on 8×64 WikiText).
- At a matched tile, E0M3 is worth a small but significant amount at 256×64
  (−0.004 / −0.007). At 8×64 it is split: MixFP4 is better on C4 by 0.0018 (just
  outside 2 SE), and the scale search is better on WikiText by 0.0031 (not
  significant).
- **The deployable NVFP4 baseline, per-block KL scale search at 1×16, beats every
  MixFP4 trained map on both datasets.** A reviewer can therefore argue that
  E0M3 is unnecessary on this model. The per-block choice is free in NVFP4 but
  costs a tile-wide type field in MixFP4, and that granularity advantage
  outweighs what E0M3 adds.
- The 1×16 dev KL bottomed at epoch 14 (0.10892) and ended at 0.11092, while
  train KL fell to 0.055. With 60.8 million logits it fits the calibration set
  more than the tile runs do. Tuning it (early stop, lower lr) would only
  strengthen the baseline.
- The fair test of E0M3 is therefore additive: MixFP4 whose E2M1 branch *also*
  uses the KL-trained per-block scale, versus that scale search alone. See the
  next subsection.

### Joint: 1×16 NVFP4 scale search + E0M3 tiles, one KL loss

This is the target method. The KL loss trains two sets of logits at once, with the
same STE/Adam recipe as above:
- one logit per 16-element scale block, choosing block max → 6 or block max → 4;
- one logit per type tile (8×64 or 256×64), choosing E2M1 or E0M3 α=1.

The weight is $W = E + t\,(A - E)$ with $E = S_{lo} + s\,(S_{hi} - S_{lo})$, where
$t$ is the tile type and $s$ the per-block scale choice. The straight-through
gradients are $\partial L/\partial t_u = \langle G, P_u (A - E)\rangle$ and
$\partial L/\partial s_b = \langle G, P_b (1 - t)(S_{hi} - S_{lo})\rangle$, so a
block's scale receives no gradient while its tile is E0M3 (`--alt joint`).

FourOverSix is itself an MSE-driven scale search, so the method does **not** start
from it:
- **`--scale-init nvfp4`** (the target): the scale logits start at plain NVFP4
  (block max → 6 everywhere, $S_{lo}$) with $S_{hi}$ = block max → 4, and KL alone
  decides.
- **`four_over_six`** (an ablation): the scale logits start at FourOverSix's MSE
  choice.
- In both, the matched E2M1-only arm is the 1×16 scale search with the same init.
- Activations stay FourOverSix in every row.
- The max → 6 branch equals `quant_nvfp4` except at exact rounding ties: the
  FourOverSix code path rounds a midpoint down, and `quant_nvfp4` rounds it away
  from zero.

Llama-3.2-1B-Instruct, jobs 442320–442331, 5–6 min of training each.
ΔNLL is paired per window, ± 2 SE.

| Init | Policy | E0M3 tiles | flipped scale blocks in E2M1 tiles | final dev KL | WikiText-2 | C4 |
|---|---|---:|---:|---:|---:|---:|
| — | NVFP4 (0 epochs) | 0 | — | 0.17109 | 15.418092 | 21.608984 |
| — | FourOverSix | 0 | — | 0.16656 | 15.356671 | 21.532633 |
| nvfp4 | scale search 1×16 | 0 | 1,239,338 | 0.11687 | 14.554335 | 19.616772 |
| nvfp4 | **joint 8×64** | 44,563 (2.3%) | 1,021,032 | 0.11819 | **14.513743** | **19.486712** |
| nvfp4 | joint 256×64 | 5,075 (8.5%) | 1,099,886 | 0.12026 | 14.602059 | 19.581057 |
| four_over_six | scale search 1×16 | 0 | 1,170,936 | 0.11092 | 14.443117 | 19.451303 |
| four_over_six | **joint 8×64** | 42,211 (2.2%) | 979,886 | 0.11218 | **14.421883** | **19.339970** |
| four_over_six | joint 256×64 | 4,050 (6.8%) | 1,065,865 | 0.11302 | 14.448973 | 19.462934 |

What E0M3 adds on top of the KL-trained scale (joint − scale search 1×16, same
init, ΔNLL ± 2 SE, wiki / c4):

| Init | joint 8×64 | joint 256×64 |
|---|---|---|
| nvfp4 | **−0.00279±0.00170 / −0.00665±0.00145** | +0.00327±0.00190 / −0.00182±0.00137 |
| four_over_six | −0.00147±0.00199 / **−0.00574±0.00144** | +0.00041±0.00210 / +0.00060±0.00133 |

**Reading (1B, one seed):**
- **At 8×64, E0M3 adds a significant gain on top of the best NVFP4 scale
  search.** From NVFP4 init it is significant on both datasets; from FourOverSix
  init it is significant on C4 and in the same direction on WikiText. The size is
  small, about 0.003–0.007 ΔNLL, roughly a tenth of what the scale search itself
  gains over FourOverSix.
- **At 256×64, E0M3 adds nothing reliable:**
  - NVFP4 init: WikiText is worse and C4 better, both beyond 2 SE.
  - FourOverSix init: a tie.
- **Initialization matters as much as E0M3.** The FourOverSix start beats the
  NVFP4 start by about 0.008 ΔNLL in both arms (scale 1×16:
  −0.00767±0.00264 / −0.00847±0.00156).
  - With init −1 and lr 0.02, a block has to see a consistent gradient sign for
    tens of steps before it flips. So 320 steps from the NVFP4 start rediscover
    only part of the FourOverSix choice: 1.24 million blocks move to max → 4,
    about 2% of the 60.8 million.
  - FourOverSix is data-free, so using it only as the *starting point* of the KL
    search would be defensible. Longer training or a smaller starting margin are
    the other ways to close the gap.
- **Best configuration measured on 1B:** FourOverSix-init joint 8×64, 14.4219 /
  19.3400. That is −0.0628 / −0.1074 ΔNLL versus FourOverSix, and it beats the
  earlier E0M3-only MixFP4 8×64 map by −0.00581±0.00194 / −0.01607±0.00198.
- **Overfitting:** dev KL for the joint runs is lowest around epochs 12–16 and
  ends 0.001–0.003 higher, while train KL keeps falling to 0.055–0.06.

#### More calibration data and longer training

These runs add 192 math and 192 code windows (`--extra-fit 192`) to the pinned
64 + 64, for 512 sequences in total. The new windows come from the same parquet
shards and exclude the pinned documents and every development document.
- **40 epochs (jobs 442398–442403) overfits badly.** Every arm's dev KL is lowest
  at epoch 8 (0.101–0.113, below every short run) and then climbs to 0.120–0.128,
  while train KL falls to 0.043–0.056. The final maps are worse than the short
  runs: for example, NVFP4-init joint 8×64 scores 14.8932 / 19.7139. Two of the six
  jobs died near the end when `/work` filled up; their dev curves are kept.
- **8 epochs, chosen from those dev curves (jobs 442443–442448):**

| Init | Policy | E0M3 tiles | final dev KL | WikiText-2 | C4 |
|---|---|---:|---:|---:|---:|
| four_over_six | scale search 1×16 | 0 | 0.10092 | 14.412138 | 19.530315 |
| four_over_six | joint 8×64 | 66,263 | 0.10211 | **14.396612** | 19.452019 |
| four_over_six | joint 256×64 | 8,886 | 0.10645 | 14.451501 | 19.601044 |
| nvfp4 | scale search 1×16 | 0 | 0.10753 | 14.492434 | 19.616920 |
| nvfp4 | joint 8×64 | 71,851 | 0.10849 | 14.544625 | 19.599020 |
| nvfp4 | joint 256×64 | 11,229 | 0.11152 | 14.579458 | 19.647200 |

Joint − scale search 1×16, same init and budget (ΔNLL ± 2 SE, wiki / c4):
four_over_six 8×64 −0.00108±0.00201 / −0.00402±0.00127; 256×64
+0.00273±0.00217 / +0.00362±0.00161. nvfp4 8×64 +0.00359±0.00206 /
−0.00091±0.00130; 256×64 +0.00599±0.00230 / +0.00154±0.00154.

- **More data improves dev KL (math/code) but hardly helps WikiText/C4.** C4 is
  sometimes worse than the 128-sequence run: four_over_six scale 1×16 19.4513 →
  19.5303. A larger math/code calibration set specializes the map to that domain.
- **Joint 256×64 still loses to scale search alone, so this is an optimization
  failure.** Leaving every tile E2M1 reproduces the scale-only solution, yet joint
  256×64 has the higher dev KL at every checkpoint, in both inits and both
  budgets.
  - The likely cause is the first-order STE step. A scale flip changes 16
    weights, where $\langle G, \Delta W\rangle$ predicts the loss change well; a
    256×64 tile flip changes 16,384 weights, where it does not.
  - A flipped tile also freezes the scale logits of its 1,024 blocks (their
    gradient becomes zero).
  - Proposed fix, not yet run: make tile flips more conservative than scale flips
    (a larger initial tile margin or a smaller tile lr), or train the scale first
    and flip tiles on top.
- With this budget, E0M3's advantage over scale-only is significant only on C4
  with the FourOverSix init at 8×64 (−0.004). Across all runs so far, the most
  robust single configuration is four_over_six joint 8×64.

#### Staged: train the scale first, then flip tiles on top

The same 512 sequences are used. Epochs 1–8 train only the per-block scale logits
(`--scale-epochs 8`), with every tile E2M1. Epochs 9–16 then start the tile logits
from −1, and the scale logits are either frozen (`--stage2 tiles`) or keep
training (`--stage2 joint`). Frozen logits get no gradient, so Adam does not step
them. Jobs 442467 and 442470–442476.

Stage 1 reproduces the 8-epoch scale-only runs up to GPU nondeterminism: epoch-8
dev KL is 0.1006–0.1019 versus 0.1009 (four_over_six) and 0.1072–0.1078 versus
0.1075 (nvfp4). The reference below is therefore the 8-epoch scale-only run.

| Init | Tile | Stage 2 | E0M3 tiles | dev KL ep 8 → 16 | WikiText-2 | C4 | ΔNLL vs scale-only ± 2 SE (wiki / c4) |
|---|---|---|---:|---|---:|---:|---|
| four_over_six | 8×64 | frozen scale | 109,599 | 0.1013 → 0.1001 | 14.423974 | 19.600073 | +0.00082±0.00193 / +0.00357±0.00137 |
| four_over_six | 8×64 | joint | 31,647 | 0.1019 → 0.1051 | 14.528759 | 19.552301 | +0.00806±0.00186 / +0.00113±0.00149 |
| four_over_six | 256×64 | frozen scale | 9,064 | 0.1019 → 0.1038 | 14.448849 | 19.633364 | +0.00254±0.00228 / +0.00526±0.00149 |
| four_over_six | 256×64 | joint | 6,394 | 0.1006 → 0.1066 | 14.475388 | 19.514267 | +0.00438±0.00198 / −0.00082±0.00153 |
| nvfp4 | 8×64 | frozen scale | 116,346 | 0.1072 → 0.1056 | 14.545326 | 19.669998 | +0.00364±0.00188 / +0.00270±0.00140 |
| nvfp4 | 8×64 | joint | 35,289 | 0.1074 → 0.1102 | 14.624975 | 19.680918 | +0.00910±0.00198 / +0.00326±0.00147 |
| nvfp4 | 256×64 | frozen scale | 10,587 | 0.1073 → 0.1101 | 14.561533 | 19.676781 | +0.00476±0.00210 / +0.00305±0.00135 |
| nvfp4 | 256×64 | joint | 7,856 | 0.1078 → 0.1133 | 14.637138 | 19.673571 | +0.00994±0.00206 / +0.00288±0.00145 |

- **No staged run beats scale search alone on WikiText or C4.** Every difference
  is positive or within noise, and most are significant. Adding E0M3 tiles on top
  of a trained 1×16 scale makes the model worse on the evaluation domains.
- **With the scale frozen, 8×64 tile flips still lower the math/code dev KL**
  (0.1013 → 0.1001 and 0.1072 → 0.1056), but the WikiText/C4 NLL goes up.
  - The tiles fit the calibration domain, and the same-domain dev set does not
    detect that.
  - At 256×64, dev KL rises as soon as tiles start flipping, so the large-tile
    moves hurt even in-domain.
- **Continuing to train the scale in stage 2 (`joint`)** overfits it past its
  epoch-8 optimum; dev KL rises in every such run.
- **Conclusion on 1B:** once the NVFP4 per-block scale is trained on the same KL
  loss, E0M3 tiles add no robust out-of-domain gain.
  - The only significant E0M3 gains over a matched scale search are C4 results of
    the 128-sequence 20-epoch joint runs at 8×64 (−0.0067 nvfp4, −0.0057
    four_over_six), and they did not survive more data or staging.
  - A reviewer's "compare against KL-trained NVFP4 scale search" objection is
    therefore not answered on this model.

Implementation: `run_train_map.py`, `slurm/train_map.sbatch`,
`summarize_train_map.py`. Details and per-epoch curves are in
[results/mixfp4_potential/train_map/REPORT.md](results/mixfp4_potential/train_map/REPORT.md).

---

Supporting material: [multi-round study](results/mixfp4_potential/MULTIROUND.md),
[potential ladder and threshold study](results/mixfp4_potential/REPORT.md),
[archived detailed report](MIXFP4_REPORT_DETAILS.md) (earlier one-shot election
and other experiment history).
