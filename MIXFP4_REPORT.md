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

Each tile is either FourOverSix E2M1 (the default) or E0M3. Selection minimizes
KL(BF16 teacher ‖ quantized model) and works like training with a line search.

**Data.**
- 128 calibration sequences of 512 tokens: 64 OpenWebMath and 64 CodeParrot.
- 192 separate held-out math/code **development** documents of 512 tokens.
- No WikiText or C4 is used anywhere in selection.

**Loop.** Start from all-E2M1 (FourOverSix), then repeat:

1. **Score every flip at the current model.**
   - For each calibration sequence, compute the weight gradient `G` of the KL
     between the BF16 teacher's and the current quantized model's next-token
     distributions. Activations are quantized with a straight-through estimator.
   - A tile's flip score is `⟨G, ΔW_tile⟩`, where `ΔW_tile` is the weight change of
     switching that tile to the other type. Undoing an earlier flip is also a flip.
2. **Filter and rank.** Keep flips whose per-sequence mean + 2 SE < 0, i.e.
   predicted to lower KL with confidence. Rank them by that bound.
3. **Backtrack on measured loss.**
   - Apply the top n candidates, starting with n = all, and measure mean KL on the
     development documents.
   - Accept the first step that lowers it; otherwise halve n and retry.
4. **Stop** when no step lowers development KL.

**Why multiple rounds.** First-order scores are only valid near the current point.
Summed over thousands of tiles, they overstate the combined effect of a step by
5–50×. Electing everything that passes the filter in one step (one-shot election)
overshoots, and in later rounds most flips that pass the filter are rejected. On
Llama 256×64, for example, round 1 had 29,820 candidates but accepted 465; applying
all of them *raised* development KL by 0.11. Re-scoring after every accepted step,
with the step size set by measured loss, avoids this.

**Output.** A per-tile E2M1/E0M3 map, the only runtime artifact. Selection is
deterministic for a fixed setup: an independent re-run reproduced the Llama map
and every evaluation loss bitwise. Implementation: `run_multiround.py
--objective kl`.

## 4. Calibration time and cost

These are **one-time, offline tile-selection costs**. Inference memory is not
reported: this is fake quantization, and peak inference memory will be measured
on the target device.

| Run | Hardware | Implementation | Scoring passes | Dev evaluations | Selection time | Peak GPU memory | Peak CPU memory |
|---|---|---|---:|---:|---:|---|---:|
| Llama-3.1-8B, 256×64 | 1× H200 | reference | 5 | 42 | 43.9 min | 46.1 GiB | 41.2 GiB |
| Llama-3.1-8B, 256×64 | 1× H200 | **optimized** | 4 | 33 | **15.3 min** | 64.5 GiB | 42.9 GiB |
| Llama-3.1-8B, 8×64 | 1× H200 | reference | 10 | 133 | 2 h 15 min | 47.3 GiB | 41.3 GiB |
| Qwen3.8-27B, 256×64 (run 1, to round 5) | 2× H200 | reference | 6 | 63 | 3 h 58 min | ~82 + 94 GiB¹ | 78.1 GiB |
| Qwen3.8-27B, 256×64 (run 2, 2 rounds) | **1× H200** | optimized | 2 | 11 | 47 min | **107.1 GiB** | 78.2 GiB |
| Qwen3.8-27B, 8×64 | 2× H200 | reference | 9 | 137 | 8 h 20 min | 82.2 + 94.2 GiB | 78.2 GiB |

¹ Not logged for this run; the Qwen 8×64 run has the same model, candidates and
teachers on the same two GPUs.

Selection time excludes setup (loading model and teachers, 2–8 min) and the final
evaluation.

**Where the time goes.**

| Model | Scoring pass (reference) | Development evaluation (reference) | Development evaluation (optimized) |
|---|---:|---:|---:|
| Llama-3.1-8B | ~2 min | 0.9 min | 0.3 min |
| Qwen3.8-27B | ~8 min | 3 min | 1.8 min |

Scoring is 128 × 512 tokens, forward plus KL backward. A development evaluation is
192 × 512 tokens, forward only. Late rounds that accept few flips spend most of
their time on backtracking evaluations, while round 0 alone gives most of the
gain.

**The optimized implementation** changes no weight value.
- Both candidates are stored packed as 4-bit codes plus FP8 scales and decoded per
  module, verified bitwise. This lets Qwen run on one GPU.
- Activation fake-quantization is vectorized and verified bitwise at the start of
  every run.
- Llama batches 16 documents per evaluation and 8 sequences per scoring pass.
  Qwen evaluates one document per pass, because its batched forward is not
  numerically identical to one-at-a-time.
- Floating-point summation order changes which borderline tiles are selected.
  The Llama 256×64 optimized run selected 8,393 tiles vs 8,405.

## 5. Perplexity

W4A4 fake quantization: weights as listed, activations FourOverSix (NVFP4
activations for the NVFP4 row). Evaluation uses the released protocol: WikiText-2
test in 2,048-token windows and 256 seed-0 C4 validation crops. Lower is better.
Paired ΔNLL is per window versus FourOverSix, ± 2 SE.

### Llama-3.1-8B

| Policy | E0M3 tiles | WikiText-2 | C4 | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---|
| BF16 (reference) | — | 6.240087 | 8.958212 | — |
| NVFP4 | 0 | 6.940252 | 9.925099 | — |
| NVFP4 FourOverSix | 0 | 6.875525 | 9.823733 | — |
| **MixFP4 8×64** | 3,654 | **6.819751** | **9.750492** | −0.00815±0.00173 / −0.00748±0.00240 |
| **MixFP4 256×64** (run 1) | 8,405 | 6.841998 | 9.774137 | −0.00489±0.00185 / −0.00506±0.00220 |
| MixFP4 256×64 (run 2, optimized) | 8,393 | 6.835411 | 9.771621 | −0.00585±0.00187 / −0.00532±0.00220 |

Versus FourOverSix:
- **MixFP4 8×64:** −0.0558 WikiText / −0.0732 C4.
- **MixFP4 256×64:** −0.0335 / −0.0496 (run 1) and −0.0401 / −0.0521 (run 2).

### Qwen3.8-27B

| Policy | E0M3 tiles | WikiText-2 | C4 | paired ΔNLL vs FourOverSix (wiki / c4) |
|---|---:|---:|---:|---|
| BF16 (reference) | — | 7.050375 | 9.893323 | — |
| NVFP4 | 0 | 7.579994 | 10.230958 | — |
| NVFP4 FourOverSix | 0 | 7.287076 | 10.188365 | — |
| **MixFP4 8×64** | 17,441 | **7.166357** | **10.155802** | −0.01671±0.00381 / −0.00320±0.00084 |
| **MixFP4 256×64** (run 1) | 39,099 | 7.246839 | 10.157245 | −0.00554±0.00343 / −0.00306±0.00092 |
| MixFP4 256×64 (run 2) | 39,092 | 7.201498 | 10.149290 | −0.01181±0.00353 / −0.00384±0.00089 |

Versus FourOverSix:
- **MixFP4 8×64:** −0.1207 WikiText / −0.0326 C4.
- **MixFP4 256×64:** −0.0402 / −0.0311 (run 1) and −0.0856 / −0.0391 (run 2).

**Qwen results depend on the selection path.**
- The two 256×64 runs differ in only 999 of ~39,100 tiles. The difference comes
  from floating-point summation order in scoring (two GPUs vs one), compounded
  over rounds. Yet their WikiText gains differ by 2×.
- In the 8×64 run, the map after round 4 scored better on both corpora
  (7.153788 / 10.145843) than the converged map. Rounds 5–8 lowered development KL
  but not test PPL.
- The paired ±2 SE above does not include this selection variance.
- A reliable Qwen number needs repeated selections, e.g. on different
  calibration halves. Llama shows neither effect.

## 6. Zero-shot accuracy

lm-eval 0.4.5, 0-shot, batch size 8: `arc_easy`, `arc_challenge`, `hellaswag`,
`openbookqa`, `boolq`, `winogrande`. The metric is `acc_norm` where defined, else
`acc`, and the mean is unweighted over the six tasks. BF16 and NVFP4 rows are from
the earlier zero-shot study with the same protocol
([details](MIXFP4_REPORT_DETAILS.md)). FourOverSix and MixFP4 rows are evaluated
together in jobs 427938–427940.

### Llama-3.1-8B

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference) | 0.8106 | 0.5350 | 0.7885 | 0.4480 | 0.8196 | 0.7380 | 0.6899 |
| NVFP4 | 0.7496 | 0.5085 | 0.7743 | 0.4280 | 0.7969 | 0.7182 | 0.6626 |
| NVFP4 FourOverSix | *running* | | | | | | |
| MixFP4 8×64 | *running* | | | | | | |
| MixFP4 256×64 (run 1) | *running* | | | | | | |

### Qwen3.8-27B

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference) | 0.7298 | 0.5896 | 0.8291 | 0.4620 | 0.8670 | 0.7561 | 0.7056 |
| NVFP4 | 0.7542 | 0.5828 | 0.8237 | 0.4460 | 0.7783 | 0.7451 | 0.6883 |
| NVFP4 FourOverSix | *running* | | | | | | |
| MixFP4 8×64 | *running* | | | | | | |
| MixFP4 256×64 (run 1) | *running* | | | | | | |
| MixFP4 256×64 (run 2) | *running* | | | | | | |

---

Supporting material: [multi-round study](results/mixfp4_potential/MULTIROUND.md),
[potential ladder and threshold study](results/mixfp4_potential/REPORT.md),
[archived detailed report](MIXFP4_REPORT_DETAILS.md) (earlier one-shot election
and other experiment history).
