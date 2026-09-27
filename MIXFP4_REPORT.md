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
| 256×64 | GB200 (SM100) | Heterogeneous trained per-tile maps (§5) | *not yet measured* |
| 8×64 | SM120 | Heterogeneous per-tile maps | *to be provided* |

The 256×64 uniform result comes from job 400605 (median of three samples,
256×256×256 GEMM tile, BF16 output, FP32 accumulation, PDL on):
[graph samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_graph.json),
[launch samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_launch.json).
It shows that switching the type costs nothing on SM100. It does not time a
realistic mixed map.

## 3. How tile selection works: training the map with an optimizer

Each weight tile is either FourOverSix E2M1 (the default) or E0M3. The map of
choices is found by **training it like a set of network parameters**: every tile
gets a real-valued logit, the model is run with the hard E2M1/E0M3 map those
logits imply, and Adam minimizes the KL divergence to the BF16 teacher on
calibration text. The model weights themselves never change. Only the choice
between two fixed quantizations of each tile is learned. There is no candidate
filter, no line search and no acceptance test.

An earlier method elected flips greedily in rounds with a backtracking line
search on held-out loss. It is superseded by this one and documented in
[MIXFP4_MULTIROUND_REPORT.md](MIXFP4_MULTIROUND_REPORT.md) (deprecated).

### 3.1 What is being optimized

**Data.**
- 128 calibration sequences of 512 tokens: 64 OpenWebMath and 64 CodeParrot.
  These are the only data the optimizer sees.
- 192 separate held-out math/code **development** documents of 512 tokens. They
  are evaluated every 2 epochs to *monitor* training. They never change the map.
- No WikiText or C4 is used anywhere in calibration.

**Two fixed candidates per weight matrix.** For each text linear layer's weight
$`W \in \mathbb R^{N \times K}`$, two quantized versions are computed once from the
BF16 weights:
- $`B`$, the FourOverSix E2M1 quantization (plain NVFP4 with the 4-or-6 block-scale
  choice);
- $`A`$, the E0M3 quantization with block scale `block_max / 7`.

Both share the tensor's FP32 global scale, and each 16-element scale block keeps
its own E4M3 scale. A map $`m \in \{0,1\}^{\text{tiles}}`$ selects between them per
tile:

```math
\hat W(m) = B + \sum_{u} m_u \, P_u \odot (A - B),
```

where $`P_u`$ is the 0/1 mask of tile $`u`$ (256×64 or 8×64 elements). With
$`m_u \in \{0, 1\}`$, every element of $`\hat W`$ is exactly the corresponding element
of $`B`$ or of $`A`$, so the trained model is bitwise the deployable one.

**Objective.** Let $`p_t`$ be the BF16 teacher's next-token distribution at position
$`t`$, and $`q_{m,t}`$ that of the W4A4 model with map $`m`$. For a sequence $`s`$ of
$`T`$ tokens,

```math
\mathrm{KL}_s(m) = \frac{1}{T-1} \sum_{t=1}^{T-1} \sum_{v \in \text{vocab}}
p_t(v)\, \big[\log p_t(v) - \log q_{m,t}(v)\big].
```

The training loss for a minibatch $`\mathcal B`$ of 8 sequences is
$`L(m) = \frac{1}{|\mathcal B|} \sum_{s \in \mathcal B} \mathrm{KL}_s(m)`$. The teacher
log-probabilities are computed once, before quantization, and reused throughout.

### 3.2 Parametrization: latent logits with a straight-through estimator

The map is binary, so it has no gradient. Each tile $`u`$ is given a real latent
logit $`\theta_u`$, and

```math
m_u = \mathbb 1[\theta_u > 0] \quad \text{(forward)}, \qquad
\frac{\partial m_u}{\partial \theta_u} := 1 \quad \text{(backward)}.
```

This is the BinaryConnect-style straight-through estimator (STE). The forward
pass always runs the hard, deployable map. The backward pass passes the gradient
with respect to $`m_u`$ straight through to $`\theta_u`$.

That gradient is well defined. $`\hat W`$ is linear in $`m`$, so relaxing $`m_u`$ to a
real number gives a smooth loss. Its derivative at the current hard map is

```math
\frac{\partial L}{\partial m_u}
= \Big\langle \frac{\partial L}{\partial \hat W},\, P_u \odot (A - B) \Big\rangle
= \sum_{(i,j) \in u} G_{ij}\, (A - B)_{ij},
\qquad G = \frac{\partial L}{\partial \hat W}.
```

This is the first-order change in loss per unit of movement from E2M1 toward
E0M3 on tile $`u`$. A negative value means switching the tile to E0M3 is predicted
to lower the KL. A positive value on an E0M3 tile means switching it back is
predicted to help. The same logit handles both directions, so a flip can be
undone later in training.

All tiles start at $`\theta_u = -1`$, i.e. the all-E2M1 FourOverSix map.

### 3.3 One training step

*Forward pass*, for a minibatch of 8 calibration sequences at the current map:

1. **Weights.** Every text linear layer holds $`\hat W(m)`$, decoded from the packed
   candidates. The weights are frozen (`requires_grad=False`), so autograd never
   builds or stores a weight gradient.
2. **Graph.** The token embeddings are detached and marked `requires_grad`, so the
   backward pass flows through the activations of every layer.
3. **Activation quantization with a straight-through estimator.** A pre-hook on
   each linear layer replaces its input $`x`$ with

   ```math
   \tilde x = Q(x) + \big(x - \mathrm{sg}(x)\big),
   ```

   where $`\mathrm{sg}`$ is stop-gradient (`detach`). The value is $`Q(x)`$, and the
   Jacobian is the identity. $`Q`$ is FourOverSix with one FP32 global scale per
   token, $`\max|x_t| / (6 \cdot 448)`$ (`quantize_rows`). Each 16-element block then
   takes whichever E4M3 block scale, mapping its maximum to 6 or to 4, gives the
   smaller squared error. Per-token scales keep the forward causal.
4. **Save the layer input.** A forward hook keeps each linear layer's quantized
   input $`\tilde x`$ and registers a hook on its output $`y`$.
5. **Loss.** $`L`$ is the mean over the minibatch of the per-sequence KL above.

*Backward pass: one `backward()` per step.* When it reaches a linear layer, the
output hook receives $`\delta = \partial L / \partial y`$. Because
$`y = \hat W \tilde x`$, the weight gradient is one matmul over all tokens of the
minibatch, and the hook reduces it to tiles immediately, in FP32:

```math
G = \delta^{\top} \tilde x \in \mathbb R^{N \times K}, \qquad
\frac{\partial L}{\partial m_u} = \sum_{(i,j) \in u} \big(G \odot (A - B)\big)_{ij}.
```

For $`r \times c`$ tiles, the elementwise product is padded to a multiple of $`r`$
rows, reshaped to $`(\lceil N/r \rceil, r, K/c, c)`$ and summed over the two
within-tile axes. $`G`$ is discarded right away, so only one number per tile
persists. $`\delta`$ already carries the effect of this layer's output on every
later layer and on the logits. The gradient therefore measures a tile's effect on
the model's output distribution, not on the layer's local reconstruction error.
Every linear layer's hook fires in the same backward pass, so one forward and one
backward give the gradient for every tile of the model.

*Update.* The tile gradients become $`\theta`$'s gradient through the STE, and Adam
takes a step:
- $`\beta = (0.9, 0.999)`$, constant learning rate 0.02, no weight decay.
- $`\epsilon = 10^{-12}`$. Tile gradients are about $`10^{-6}`$, so PyTorch's default
  $`10^{-8}`$ would dominate $`\sqrt{\hat v}`$ and shrink every step by orders of magnitude.

After the step, every layer whose hard map changed ($`\theta_u`$ crossed zero in
either direction) has $`\hat W`$ re-decoded, and the next step runs on the new map.

*Schedule.* Batch 8, 16 steps per epoch (all 128 sequences, reshuffled each epoch
with seed 0), 20 epochs: 320 steps. The map reported is the hard map after the
last step. It is *not* chosen by development loss.

In pseudocode:

```
θ = full(tiles, -1.0)                                         # all tiles start as FourOverSix E2M1
for epoch in 1..20:
    for batch of 8 calibration sequences:                     # 16 steps per epoch
        Ŵ = where(expand(θ > 0), A, B)                        # hard map, per layer
        forward with x̃ = Q_per_token(x).detach() + (x - x.detach())
        L = mean_s KL(teacher_s ‖ model_s);  L.backward()
        # hook on each linear output, δ = ∂L/∂y:
        θ.grad = tile_sum((δᵀ x̃) * (A - B))                  # ∂L/∂m_u, passed straight through
        adam.step(θ)                                          # lr 0.02, eps 1e-12
    every 2 epochs: dev KL of the hard map                     # monitor only
return θ > 0                                                  # the per-tile E0M3 map
```

### 3.4 Why it works, and what the learning-rate and init do

**The initial margin is a soft significance threshold.** Adam's step on each
coordinate is $`\text{lr} \cdot \hat m / (\sqrt{\hat v} + \epsilon)`$. When a tile's
gradient keeps the same sign from minibatch to minibatch, $`|\hat m| \approx \sqrt{\hat v}`$
and the logit moves by about lr = 0.02 per step. When the sign is dominated by
minibatch noise, $`\hat m`$ averages toward zero and the logit barely moves. Starting
at $`\theta = -1`$, a tile needs about 50 consistent steps (about 3 epochs) before it
can flip, and indeed no tile flips in the first 3 epochs. The tiles that do cross
are the ones whose predicted gain is consistent across the calibration data. That
is the "decisive margin" principle of earlier work, obtained from the optimizer
instead of from an explicit test.

**Gradients are re-evaluated at every map.** The first-order score of a flip
changes once other tiles flip, because tiles interact through shared inputs and
through later layers. Here every step recomputes the gradient at the current hard
map, and each step moves each logit by at most about lr. The map therefore changes
gradually, and interactions are corrected as they appear rather than predicted.

**Nothing on held-out data enters the updates.** The development documents only
monitor training. Neither the map nor the stopping point is selected on them, so
their KL is an unbiased check of generalization within the math/code domain.
WikiText-2 and C4 are then fully out of domain.

**What the gradient is not.** It is exact for the model it differentiates, but that
model differs from the deployed and evaluated one in two ways:
- the activation straight-through estimator ignores how a weight change moves the
  activation quantization grid;
- training uses per-token activation scales, while development evaluation, PPL
  evaluation and deployment use one tensor-wide activation scale per document.

The development KL, measured under the evaluation protocol, is how these gaps are
checked (§6).

**Sigmoid relaxation (ablation).** Instead of the STE, $`m_u = \sigma(\theta_u/\tau)`$
interpolates between the candidates, with $`\tau`$ annealed geometrically from 1 to
0.1, and the map is rounded at $`\theta > 0`$ for every evaluation (init $`-3`$, lr
0.05). On Llama-3.1-8B 8×64 it ties the STE within noise (§5) but adds a
temperature schedule and trains on weights that are not deployable. The STE is
the method.

**Output.** A per-tile E2M1/E0M3 map, the only runtime artifact.
Implementation: `run_train_map.py`, `slurm/train_map.sbatch`,
`summarize_train_map.py`.

## 4. Calibration time and cost

These are **one-time, offline map-training costs** on one H200 per run.
*Training time* covers the 20 epochs plus the 12 monitor-only development
evaluations (together about 4 minutes on Llama-3.1-8B). It excludes setup
(loading the model, computing the teacher log-probabilities and packing the
candidates) and the final PPL evaluation.

| Model | Tile | Setup | Training time | Time per epoch | Peak GPU memory | Peak CPU memory |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 256×64 | 1.8 min | **19.2 min** | 46 s | 60.4 GiB | 42.9 GiB |
| Llama-3.1-8B | 8×64 | 1.7 min | **19.1 min** | 46 s | 60.6 GiB | 43.0 GiB |
| Llama-3.1-8B-Instruct | 256×64 | 1.5 min | **18.4 min** | 45 s | 60.4 GiB | 42.9 GiB |
| Llama-3.1-8B-Instruct | 8×64 | 1.4 min | **21.8 min** | 54 s | 60.6 GiB | 43.1 GiB |
| Llama-3.2-3B-Instruct | 256×64 | 1.2 min | **10.3 min** | 24 s | 35.2 GiB | 43.2 GiB |
| Llama-3.2-3B-Instruct | 8×64 | 1.2 min | **10.3 min** | 24 s | 35.3 GiB | 43.0 GiB |
| Llama-3.2-1B-Instruct | 256×64 | 1.0 min | **5.4 min** | 12 s | 22.9 GiB | 42.9 GiB |
| Llama-3.2-1B-Instruct | 8×64 | 1.0 min | **5.6 min** | 12 s | 23.0 GiB | 42.9 GiB |
| Qwen3.8-27B | 256×64 | 7.0 min | **3 h 5 min** | 490 s | 105.8 GiB | 78.2 GiB |
| Qwen3.8-27B | 8×64 | 7.0 min | **3 h 6 min** | 495 s | 106.7 GiB | 78.4 GiB |
| Qwen3.8-27B from nvidia NVFP4 (§5) | 256×64 | 7.8 min | **2 h 6 min** | 335 s | 96.6 GiB | 98.1 GiB |
| Qwen3.8-27B from nvidia NVFP4 (§5) | 8×64 | 7.9 min | **2 h 3 min** | 327 s | 97.2 GiB | 98.2 GiB |

Qwen runs one sequence per forward pass and accumulates 8 passes per optimizer
step (its batched forward is not identical to batch 1), so a step costs 8
single-sequence passes. The nvidia-start runs are cheaper because the gradient
hook fires only on the 192 MLP matrices that carry tiles.

- **Cost does not depend on tile size.** Each step is one forward and one backward
  over 8 × 512 tokens, and the tile reduction is a reshape-and-sum. 8×64 has 32×
  more tiles than 256×64 and costs about the same.
- **Cost scales with the model**, roughly 12 s / 24 s / 46 s per epoch for 1B /
  3B / 8B.
- **What keeps it cheap**, none of it changing a weight value: both candidates
  are stored packed as 4-bit codes plus FP8 scales and decoded per module
  (verified bitwise); no weight gradient is ever stored, only one FP32 gradient
  and two Adam moments per tile; and a layer's weights are re-decoded only when
  its hard map changes.

## 5. Perplexity

W4A4 fake quantization: weights as listed, activations FourOverSix (NVFP4
activations for the NVFP4 row). Evaluation uses the released protocol:
WikiText-2 test in 2,048-token windows and 256 seed-0 C4 validation crops, with
one tensor-wide activation scale per document. Lower is better. ΔNLL is paired
per window versus FourOverSix, ± 2 SE. All MixFP4 rows are trained STE maps
unless marked.

### Llama-3.1-8B

Jobs 441206 (256×64), 441207 (8×64 STE), 441208 (8×64 sigmoid). FourOverSix
per-window NLLs are from job 336566.

| Policy | E0M3 tiles | WikiText-2 | C4 | ΔPPL vs FourOverSix (wiki / c4) | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---|---|
| BF16 (reference) | — | 6.240087 | 8.958212 | — | — |
| NVFP4 | 0 | 6.940252 | 9.925099 | — | — |
| NVFP4 FourOverSix | 0 | 6.875525 | 9.823733 | — | — |
| **MixFP4 256×64** (SM100) | 38,176 (9.0%) | **6.805528** | **9.723484** | **−0.0700 / −0.1002** | −0.01023±0.00174 / −0.01026±0.00241 |
| **MixFP4 8×64** (SM120) | 329,837 (2.4%) | **6.784682** | **9.675402** | **−0.0908 / −0.1483** | −0.01330±0.00184 / −0.01521±0.00329 |
| MixFP4 8×64, sigmoid ablation | 212,778 (1.6%) | 6.781821 | 9.681478 | −0.0937 / −0.1423 | −0.01372±0.00184 / −0.01459±0.00325 |

MixFP4 closes 11% (256×64) and 14% (8×64) of the WikiText gap between
FourOverSix and BF16, and 12% / 17% of the C4 gap.

### Llama Instruct models

Same settings as Llama-3.1-8B. All three models share the Llama-3 tokenizer, so
the same calibration sequences, development documents and released evaluation
windows apply; the calibration loader re-checks every window's token hash. Each
FourOverSix row is the all-E2M1 map from the same pipeline (a zero-epoch run),
which gives matched per-window NLLs. Jobs: 8B-Instruct 442024, 442074, 442075;
3B-Instruct 442077-442079; 1B-Instruct 442046-442048.

| Model | Policy | E0M3 tiles | WikiText-2 | C4 | ΔPPL vs FourOverSix (wiki / c4) | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---|---:|---:|---:|---|---|
| Llama-3.1-8B-Instruct | NVFP4 FourOverSix | 0 | 7.814643 | 11.264191 | — | — |
| | **MixFP4 256×64** | 44,763 (10.5%) | **7.710097** | **11.174994** | **−0.1045 / −0.0892** | −0.01347±0.00173 / −0.00795±0.00164 |
| | **MixFP4 8×64** | 365,992 (2.7%) | **7.699191** | **11.107474** | **−0.1155 / −0.1567** | −0.01488±0.00178 / −0.01401±0.00274 |
| Llama-3.2-3B-Instruct | NVFP4 FourOverSix | 0 | 11.945266 | 15.507828 | — | — |
| | **MixFP4 256×64** | 29,408 (17.1%) | **11.543218** | **15.158546** | **−0.4020 / −0.3493** | −0.03424±0.00267 / −0.02278±0.00255 |
| | **MixFP4 8×64** | 245,646 (4.5%) | **11.401570** | **15.112609** | **−0.5437 / −0.3952** | −0.04658±0.00264 / −0.02582±0.00209 |
| Llama-3.2-1B-Instruct | NVFP4 FourOverSix | 0 | 15.356671 | 21.532633 | — | — |
| | **MixFP4 256×64** | 17,064 (28.7%) | **14.717933** | **20.049759** | **−0.6387 / −1.4829** | −0.04248±0.00298 / −0.07135±0.00392 |
| | **MixFP4 8×64** | 138,907 (7.3%) | **14.505991** | **19.653187** | **−0.8507 / −1.8794** | −0.05699±0.00275 / −0.09133±0.00416 |

Tile totals: 8B 425,984 (256×64) / 13,631,488 (8×64); 3B 172,032 / 5,505,024;
1B 59,392 / 1,900,544.

**Reading.**
- Every trained map beats FourOverSix significantly on both datasets, on all four
  models and at both tile sizes, although calibration used only math and code.
- 8×64 beats 256×64 on both datasets on every model: the finer tile lets the
  map follow the data more closely, at the cost of the SM120 path.
- The gain grows as the model shrinks: 8×64 WikiText ΔNLL is −0.013 (8B),
  −0.015 (8B-Instruct), −0.047 (3B) and −0.057 (1B). Smaller models also elect a
  larger share of E0M3 tiles.

### Qwen3.8-27B

Same method and hyperparameters: 20 epochs, lr 0.02, init −1, 16 steps per epoch.
Each step accumulates 8 batch-1 passes. The 496 text linear layers carry tiles.
Recurrent, conv, norm, vision and the head are not quantized, as in all Qwen work
here. Jobs 445055 (256×64) and 445056 (8×64). The BF16, NVFP4 and FourOverSix
rows and the FourOverSix per-window NLLs are from job 336969. The multi-round
reference maps are the ones in
[MIXFP4_MULTIROUND_REPORT.md](MIXFP4_MULTIROUND_REPORT.md) §3.

| Policy | E0M3 tiles | WikiText-2 | C4 | ΔPPL vs FourOverSix (wiki / c4) | paired ΔNLL vs FourOverSix (wiki / c4) | paired ΔNLL vs multi-round (wiki / c4) |
|---|---:|---:|---:|---|---|---|
| BF16 (reference) | — | 7.050375 | 9.893323 | — | — | — |
| NVFP4 | 0 | 7.579994 | 10.230958 | — | — | — |
| NVFP4 FourOverSix | 0 | 7.287076 | 10.188365 | — | — | — |
| multi-round 256×64 (deprecated) | 39,095 | 7.223045 | 10.152189 | −0.0640 / −0.0362 | −0.00883±0.00361 / −0.00356±0.00089 | — |
| multi-round 8×64 (deprecated) | 17,571 | 7.205417 | 10.148049 | −0.0817 / −0.0403 | −0.01127±0.00442 / −0.00396±0.00089 | — |
| **MixFP4 256×64** (SM100) | 70,267 (4.7%) | **7.131810** | **10.145408** | **−0.1553 / −0.0430** | −0.02154±0.00389 / −0.00423±0.00101 | −0.01271±0.00331 / −0.00067±0.00084 |
| **MixFP4 8×64** (SM120) | 610,431 (1.3%) | **7.096120** | **10.125233** | **−0.1910 / −0.0631** | −0.02655±0.00438 / −0.00622±0.00101 | −0.01528±0.00449 / −0.00225±0.00082 |

Tile totals: 1,492,480 (256×64) / 47,559,680 (8×64).

- Both trained maps beat FourOverSix significantly on both datasets.
- They close **66% (256×64) and 81% (8×64) of the WikiText gap** between
  FourOverSix and BF16, but only 15% / 21% of the C4 gap. The WikiText gain is
  much larger than on any Llama model, and the C4 gain is smaller.
- Against the deprecated multi-round maps, the trained maps are significantly
  better on WikiText at both tile sizes. On C4, 8×64 is better and 256×64 ties.
- The multi-round Qwen maps varied by up to ±0.04 WikiText PPL between runs. This
  trained-map result is also one seed.

### Starting from nvidia/Qwen3.8-27B-NVFP4

Can the trained map improve an already-optimized production checkpoint?
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
(revision `482ca0f`, Model Optimizer 0.47/0.48) is a **mixed** checkpoint:
- **NVFP4 on the MLP projections (192 matrices) and `lm_head`.** Its block
  scales come from Model Optimizer's Local-Hessian calibration (a
  Hessian-weighted per-block scale search on 2,048 samples; the model card does
  not mention GPTQ). Activations use a static calibrated global scale and dynamic
  E4M3 scales per 16 elements.
- **FP8 W8A8 on every attention projection**, full and linear attention (208
  matrices), with static per-tensor scales.

**Setup** (`run_train_map.py --modelopt`, `quantize/modelopt_ckpt.py`,
`slurm/train_map_modelopt.sbatch`):
- **The checkpoint is emulated exactly.** Its NVFP4 weights are dequantized from
  the packed codes, E4M3 block scales and FP32 global scale. The decode is
  checked bitwise against an independent table lookup, and against the BF16
  originals: mean NMSE 0.0075. FP8 weights are loaded with NMSE ≤ 7e-4.
- **All 783 unquantized tensors** (norms, embeddings, recurrent/conv, vision) are
  bit-identical to our pinned BF16 Qwen3.8-27B.
- **Activations follow nvidia's recipe** on all 401 quantized layers, including
  `lm_head`. The quantizer is per-token independent, so training and evaluation
  use the same one.
- **Tiles exist only on the 192 NVFP4 MLP matrices.** The FP8 attention layers
  and `lm_head` stay exactly as shipped.
  - The E2M1 candidate is **nvidia's own codes and scales, bitwise**.
  - The E0M3 candidate is `block_max/7` on the BF16 weight, under nvidia's global
    scale, so a tile never changes its tensor's scale. nvidia's global scales are
    1.0–2.6× the `amax/(6·448)` rule.
- **Training is unchanged:** 128 math/code sequences, KL to the BF16 teacher,
  STE, Adam, 20 epochs.
- **The baseline** is the zero-epoch run, i.e. nvidia's model as shipped, with
  matched per-window NLLs.

Jobs: 445063 (baseline), 445064 (256×64) and 445065 (8×64).

| Policy | E0M3 tiles | WikiText-2 | C4 | ΔPPL vs nvidia (wiki / c4) | paired ΔNLL vs nvidia (wiki / c4) |
|---|---:|---:|---:|---|---|
| BF16 (reference) | — | 7.050375 | 9.893323 | — | — |
| nvidia NVFP4 (MLP) + FP8 (attention), as shipped | 0 | 7.185603 | 10.179512 | — | — |
| **+ MixFP4 256×64** | 47,724 (4.6%) | **7.159577** | **10.173418** | **−0.0260 / −0.0061** | −0.00363±0.00263 / −0.00060±0.00079 |
| **+ MixFP4 8×64** | 438,750 (1.3%) | **7.147312** | **10.161573** | **−0.0383 / −0.0179** | −0.00534±0.00262 / −0.00176±0.00075 |

Tile totals (MLP only): 1,044,480 (256×64) / 33,423,360 (8×64). Development KL
0.03540 → 0.03448 (256×64) / 0.03388 (8×64).

- **8×64 improves nvidia's checkpoint significantly on both datasets.** It closes
  28% of its WikiText gap to BF16 and 6% of its C4 gap.
- **256×64 is significant on WikiText (−0.026 PPL) but within noise on C4.**
- The gains are much smaller than when starting from our own FourOverSix. Three
  reasons:
  1. Only the MLP carries tiles.
  2. The Local-Hessian E2M1 scales are already a stronger baseline.
  3. The FP8 attention leaves less total error to remove: nvidia's start is dev
     KL 0.0354 against 0.0462 for our W4A4 FourOverSix.
- **For comparison,** nvidia's shipped model scores 7.1856 / 10.1795, against
  7.2871 / 10.1884 for our all-W4A4 FourOverSix. Our all-4-bit trained MixFP4
  maps, with attention also at 4-bit, beat nvidia's FP8-attention model
  significantly. Paired ΔNLL (wiki / c4):
  - 256×64 vs nvidia as shipped: −0.00751±0.00355 / −0.00336±0.00131;
  - 8×64 vs nvidia as shipped: −0.01253±0.00376 / −0.00535±0.00133;
  - 8×64 vs nvidia + MixFP4 8×64: −0.00719±0.00430 / −0.00358±0.00144.
- **Fidelity caveat:** these are fake-quant numbers of an emulation. They were
  not cross-checked against a real vLLM run of the checkpoint.

## 6. Training dynamics

Development KL of the hard map under the evaluation protocol (monitor only), with
the E0M3 tile count in parentheses. Llama-3.1-8B; the all-E2M1 start is 0.10845.

| Epoch | STE 256×64 | STE 8×64 | sigmoid 8×64 |
|---:|---|---|---|
| 4 | 0.10109 (182) | 0.09951 (2,072) | 0.10283 (1,045) |
| 8 | 0.09489 (7,518) | 0.08907 (83,110) | 0.08939 (61,569) |
| 12 | 0.09050 (19,410) | 0.08598 (183,041) | 0.08563 (170,392) |
| 16 | 0.08953 (29,325) | **0.08318** (260,629) | 0.08360 (204,449) |
| 20 | **0.08868** (38,176) | 0.08349 (329,837) | 0.08418 (212,778) |

Final development KL on the Instruct models (FourOverSix → 256×64 / 8×64):
8B-Instruct 0.10870 → 0.08036 / 0.07805; 3B-Instruct 0.09620 → 0.08081 / 0.07677;
1B-Instruct 0.16656 → 0.12436 / 0.10905.

Qwen3.8-27B (all-E2M1 start 0.04625): 256×64 bottomed at 0.04234 (epoch 14) and
ended at 0.04262; 8×64 fell to 0.04164 at epoch 20, roughly flat after epoch 8.
Starting from nvidia's checkpoint (0.03540), both runs fluctuate. 256×64 ends at
0.03448 (minimum 0.03401 at epoch 18) and 8×64 at 0.03388 (minimum 0.03377 at
epoch 12).

- **Held-out KL falls well below the start** on every run, even though the
  development set never enters an update. The training-time approximations of §3.4 do not stop the gradient from
  lowering the KL of the deployed model.
- **256×64 on Llama-3.1-8B had not converged**: dev KL was still falling at epoch 20
  (about 2,000 net flips per epoch). On 8B-Instruct 256×64 it was also still falling.
- **8×64 on Llama-3.1-8B shows mild fitting of the calibration set**: dev KL bottomed
  at epoch 16 (0.08318) and ended at 0.08349, while training KL fell to 0.039. Dev
  KL is roughly flat over the last 6 epochs on 3B, 1B and 8B-Instruct 8×64.
- **Sigmoid settles**: flips fall from 42k per epoch to 3.5k as $`\tau`$ anneals,
  and its rounded map's dev KL tracks the STE's.

Per-epoch curves: [results/mixfp4_potential/train_map/REPORT.md](results/mixfp4_potential/train_map/REPORT.md).

## 7. Caveats and open items

- **One seed and one hyperparameter setting per arm.** The ± 2 SE covers
  evaluation noise, not selection variance. The epoch count (20) is a budget,
  not a convergence criterion.
- **GEMM cost of trained maps is unmeasured.** The maps are heterogeneous and
  elect up to 29% E0M3 tiles (1B, 256×64). §2 shows only that a uniform type
  switch is free on SM100.
- **Not yet run with trained maps:** zero-shot accuracy on any model. It exists
  only for the deprecated multi-round maps, in
  [MIXFP4_MULTIROUND_REPORT.md](MIXFP4_MULTIROUND_REPORT.md) §4.
- **Qwen3.8-27B is one seed per arm.** Its multi-round selections were
  path-sensitive (±0.04 WikiText), so repeat seeds before quoting its trained-map
  gain as a stable number.
- **The nvidia-start numbers are an emulation** of the modelopt checkpoint in fake
  quantization and are not validated against vLLM. The E0M3 candidate there
  uses the plain `block_max/7` scale, not a Local-Hessian search.

---

Supporting material: [trained-map runs](results/mixfp4_potential/train_map/REPORT.md),
[deprecated multi-round election](MIXFP4_MULTIROUND_REPORT.md),
[potential ladder and threshold study](results/mixfp4_potential/REPORT.md),
[archived detailed report](MIXFP4_REPORT_DETAILS.md) (earlier one-shot election
and other experiment history).
