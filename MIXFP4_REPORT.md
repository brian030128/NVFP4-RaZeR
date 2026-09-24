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
