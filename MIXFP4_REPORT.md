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

---

Supporting material: [multi-round study](results/mixfp4_potential/MULTIROUND.md),
[potential ladder and threshold study](results/mixfp4_potential/REPORT.md),
[archived detailed report](MIXFP4_REPORT_DETAILS.md) (earlier one-shot election
and other experiment history).
