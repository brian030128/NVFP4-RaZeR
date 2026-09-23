# N16K64 PPL table: local reproduction with fake quant and the native mixfp4 kernel

Target: the primary-campaign PPL table on branch `research/mixfp4-n16k64`
(`research/n16k64/campaigns/primary/N16_DECISION.md`, README "Perplexity: a common evaluation
protocol"), for Llama-3.1-8B, Qwen3.8-27B, Qwen3-4B, Mistral-7B-v0.3 and Phi-4 (OLMo-2 excluded
by request).

Machine: one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120, 96 GB, driver 595.71.05),
no Slurm. Python environment: the branch's locked `main.lock.txt` (torch 2.9.0+cu128,
transformers 5.16.1) in conda env `n16k64`. The original ran on A6000 GPUs.

## Two execution paths

| Path | What runs | Code |
|---|---|---|
| **Fake quant** | The campaign's own code, unchanged: `campaign.calibrate` (seed0, CE+KL, k=3) and `campaign.evaluate_ppl` (causal per-token FourOverSix activations, SDPA, released 2,048-token windows). Dequantized BF16 weights and activations feed a BF16 GEMM. | `research/n16k64/software/primary/` |
| **Native kernel** | Every scoped Linear runs as a W4A4 FP4 tensor-core GEMM on the mixfp4 SM120 mixed E2M1/E0M3 kernel (`/home/dev/mixfp4` @ `7b3ab34`), SASS-patched with the repo's own patcher. | `repro_local/realquant/` |

Native kernel builds (docs/mixed_nvfp4_report.md §6 configurations, CUDA 13.1):

| Build | Weights on | Format granule | Used for |
|---|---|---|---|
| `wt_as_A` (`libwt_as_A.so`, sha256 `ae122907…`) | operand A | 16 rows × 64 K | NVFP4, FourOverSix, **N16K64** |
| `b8x64` (`libb8x64.so`, sha256 `0e237ada…`) | operand B | 8 cols × 64 K | FourOverSix (cross-check), **N8K64** |

Both pin the activation operand to E2M1. The native path's verification chain:

1. The kernel's own self-test passes for both builds (random, all-E2M1 and all-E0M3 tagging; relative
   error 0.0017). The unpatched negative control fails at 0.45. The granule maps are exactly 16 contiguous
   rows / 8 contiguous columns, i.e. the N16/N8 tiles.
2. GPU unit gate, 28 cases across both builds: 12 distinct shapes covering every Llama-3.1-8B and
   Qwen3.8-27B linear shape (N=48 included), plus all-E2M1 and all-E0M3 maps:
   - native GEMM vs an FP64 exact-decode reference: 0.0023 relative error, lower than fake quant's 0.0027;
   - unpatched library: 0.20–0.37 (E0M3 tiles present), and bit-identical to the patched library when none are.
3. In every model evaluation:
   - each packed weight decodes **bit for bit** to the campaign's fake-quant weight;
   - installed E0M3 tile counts equal the map headers;
   - the first two activation quantizations of every module are checked bit for bit against `quantize_rows`;
   - a reinstall determinism check is identical.
4. The two builds compute FourOverSix to **bit-identical per-window NLL** on all five models (both
   corpora, every window). They differ in operand placement, warp arrangement and granule.

The only difference from fake quant is therefore the GEMM itself: FP4×FP4 products accumulated in FP32
on the tensor cores, versus BF16-rounded dequantized operands in a BF16 GEMM. The kernel's epilogue
carries only a scalar alpha, so the per-token activation global scale is applied to its bf16 output
afterwards.

> **Correction (2026-09-23).** The first native runs returned the weights-on-A output as a strided view
> of Dᵀ. Downstream attention therefore received non-contiguous q/k/v, and SDPA fell back to its FP32
> math backend instead of the bf16 flash kernel the fake-quant runs use.
>
> The affected results were the native NVFP4, FourOverSix and N16 numbers for all five models.
> `rq.RealLinear` now returns a contiguous tensor; its values are unchanged, verified bitwise. The
> affected policies were re-run as `realfix_<model>`, and all native numbers below come from those runs.
> The weights-on-B runs (N8, FourOverSix-b8x64) were never affected.
>
> The superseded runs remain under `results/real_*` for the record. They too agreed with fake quant
> within noise, so no conclusion changed.

## Reproduction criteria

The original campaign's own pre-registered cross-GPU tolerances (`PROTOCOL_FREEZE.json`,
`numerical_tolerances`):

- W4A4 PPL within 0.5% relative, BF16 within 0.1%.
- Paired effect ΔlogPPL (tierC): same sign, and |local − original| ≤ max(0.25·|original|, 0.002).
- k=3 tile count within 5% (historical tierB).

## Results: original / fake quant / native kernel

| Model | Corpus | NVFP4 | 4Over6 | N8K64(k=3) | N16K64(k=3) |
|---|---|---:|---:|---:|---:|
| Llama3.1-8B | Wiki | 6.9317 / 6.9296 / 6.9331 | 6.8754 / 6.8807 / 6.8835 | 6.8414 / 6.8441 / 6.8479 | 6.8432 / 6.8416 / 6.8427 |
| Llama3.1-8B | C4 | 9.9379 / 9.9302 / 9.9250 | 9.8254 / 9.8141 / 9.8243 | 9.7763 / 9.7774 / 9.7802 | 9.7762 / 9.7771 / 9.7844 |
| Qwen3.8-27B | Wiki | 7.5769 / 7.5446 / 7.5815 | 7.2839 / 7.3016 / 7.2902 | 7.2254 / 7.2569 / 7.2515 | 7.2456 / 7.2587 / 7.2525 |
| Qwen3.8-27B | C4 | 10.2237 / 10.2222 / 10.2272 | 10.1894 / 10.1882 / 10.1868 | 10.1491 / 10.1500 / 10.1473 | 10.1564 / 10.1559 / 10.1568 |
| Qwen3-4B | Wiki | 13.9557 / 13.9584 / 13.9401 | 14.2062 / 14.2183 / 14.1758 | 11.8841 / **11.7066** / **11.6925** | 12.1968 / **12.1095** / **12.0933** |
| Qwen3-4B | C4 | 17.2621 / 17.2493 / 17.2582 | 17.3024 / 17.3175 / 17.2943 | 15.8483 / **15.7092** / **15.6958** | 16.0485 / 15.9714 / **15.9515** |
| Mistral-7B-v0.3 | Wiki | 5.5487 / 5.5508 / 5.5493 | 5.5233 / 5.5210 / 5.5257 | 5.4987 / 5.5029 / 5.5049 | 5.5019 / 5.5029 / 5.5050 |
| Mistral-7B-v0.3 | C4 | 8.0938 / 8.0949 / 8.0985 | 8.0674 / 8.0681 / 8.0669 | 8.0456 / 8.0449 / 8.0508 | 8.0475 / 8.0521 / 8.0471 |
| Phi-4 | Wiki | 6.7013 / 6.6998 / 6.6955 | 6.6641 / 6.6627 / 6.6656 | 6.6152 / 6.6324 / 6.6347 | 6.6288 / 6.6366 / 6.6434 |
| Phi-4 | C4 | 10.5838 / 10.5834 / 10.5824 | 10.5485 / 10.5473 / 10.5463 | 10.5000 / 10.5132 / 10.5161 | 10.5073 / 10.5205 / 10.5241 |

Bold marks cells outside the 0.5% tolerance. Per-cell relative differences, paired effects with 2SE and
native−fake differences are in [REPORT_TABLES.md](REPORT_TABLES.md).

### Scorecard

| Item | Result |
|---|---|
| BF16 PPL (fake path, 10 cells) | all within 0.1% (max 0.021%) |
| NVFP4 / 4Over6 PPL (40 cells, fake + native) | **all within 0.5%** (fake max 0.43%, native max 0.21%) |
| N8 / N16 PPL (40 cells) | 33 within 0.5%; **7 outside, all Qwen3-4B** (−0.60% … −1.61%, lower than original) |
| Effects vs 4Over6 (60 = 3 contrasts × 2 corpora × 5 models × 2 paths) | **57 pass tierC**; 3 fail: Phi-4 N8 Wiki (fake −0.0046, native −0.0046 vs −0.0074), Qwen3.8-27B native N8 Wiki (−0.0053 ± 0.0033 vs −0.0081) |
| Effect direction | every E0M3 map improves on 4Over6 in all 40 cells; NVFP4−4Over6 sign reproduced in all 20 |
| k=3 tile counts (10) | 1 within 5% (Llama N16 +4.8%); others −35% … +11% |
| Native vs fake (40 paired comparisons) | 21 positive / 19 negative, 2 beyond 2SE (about 2 expected at 5%), mean −0.00003 NLL |

## Why the deviations happen

**Tile counts.** The inputs are verified identical:
- calibration token hashes match the archived or original manifests (Llama, Qwen3-4B, Qwen3.8-27B:
  archived; Mistral: every document and token hash; Phi-4: manifest SHA `ad9b2521…` equals
  `PROTOCOL_FREEZE.json`);
- weights match the archived hashes.

The count is not stable, though, because it is steep in the threshold. Near k=3, the number of tiles
elected changes by 106–244% per unit of k, so every observed count difference equals a uniform shift of
the standardized per-tile score by only 0.02–0.12 SE:

| Model | N8 count, local vs original | Count slope at k=3 | k reproducing the original count |
|---|---|---:|---:|
| Llama-3.1-8B | 3,365 vs 3,130 | −224%/unit | 3.03 |
| Qwen3-4B | 8,149 vs 7,349 | −147%/unit | 3.07 |
| Qwen3.8-27B | 3,450 vs 3,946 | −219%/unit | 2.96 |
| Mistral-7B | 6,703 vs 7,501 | −108%/unit | 2.92 |
| Phi-4 | 2,742 vs 4,201 | −244%/unit | 2.88 |

The original campaign documents the same fragility (`RISK_AUDIT.json`):
- a same-GPU calibration repeat gave map Jaccard 0.98;
- changing only the attention backend on one GPU gave k=3 Jaccard 0.29;
- its own archive-to-A6000 move missed the 5% count tolerance on two of three models (+6.9%, −5.5%);
- independent calibration draws spread N16 counts over 2,512–4,278 (Qwen3-4B) and 3,355–4,508 (Mistral).

The shift direction varies by model (+, +, −, −, −), as numerical noise would.

**Qwen3-4B map PPL.** On this model E0M3 election is worth ΔlogPPL −0.18, about −16% PPL and 35× the
Llama effect, so 10.9% / 7.9% more elected tiles translate directly into lower PPL. The effect sizes
pass tierC and lie inside the original method's own seed0+draw range:

| Qwen3-4B N16 effect | Original seed0 + 4 draws | Local |
|---|---|---:|
| Wiki | −0.125 … −0.168 | −0.161 (fake), −0.159 (native) |
| C4 | −0.063 … −0.084 | −0.081 (fake), −0.081 (native) |

**Phi-4 N8 effect.** With 35% fewer tiles, the E0M3 improvement shrinks to about 60–70% of the
original, although it is still significant (−0.0046 ± 0.0014). Direction and existence reproduce;
magnitude scales with the elected tile count.

**Numerical noise scale.** On this machine, changing only the attention implementation (eager vs
SDPA) moves the same FourOverSix evaluation by per-window NLL SD 0.010–0.012 and aggregate
ΔlogPPL −0.0004 / +0.0006. W4A4 per-token activation quantization amplifies kernel-level rounding
differences. All baseline PPL differences from the original lie within this band.

## Native kernel vs fake quant

The native path reproduces the fake-quant numbers within noise on every model: 40 paired comparisons,
21 positive / 19 negative, 2 beyond 2SE, mean −0.00003 NLL. The two kernel builds agree with each other
bit for bit on FourOverSix (verification step 4).

## Latency: Llama-3.1-8B on the native kernels

`realquant/bench_latency_llama.py` → [`results/latency_llama8b.json`](results/latency_llama8b.json).

The study uses the real weights and the evaluated maps on an idle GPU, with two extra, latency-only
E2M1 builds:
- `stock`: the stock CUTLASS SM120 NVFP4 GEMM, `src/nvfp4_gemm.cu`;
- `*_nodisp`: the mixed builds with the format dispatch compiled out (`-DMIXFP4_NO_DISPATCH=1`;
  weights-on-A also needs `-DMIXFP4_PIPE_FLAGS=0`, since its pipelined loop bypasses the switch).

**GEMM kernel time per forward** (224 projection GEMMs; CUPTI device time, so launch gaps are excluded):

| Tokens | BF16 cuBLAS | NVFP4 = 4Over6 (stock) | N16K64 (`wt_as_A`) | N8K64 (`b8x64`) |
|---:|---:|---:|---:|---:|
| 512 | 25.26 ms | 7.01 ms | 7.31 (+4.3%) | 7.65 (+9.1%) |
| 2048 | 86.26 ms | 21.98 ms | 22.95 (+4.4%) | 24.06 (+9.4%) |
| 8192 | 375.68 ms | 93.89 ms | 99.80 (+6.3%) | 106.88 (+13.8%) |

- **4Over6 costs nothing in the GEMM.** It only chooses scale values offline, and the in-model FP4
  GEMM time is 21.53 vs 21.59 ms.
- **MixFP4's GEMM overhead has two parts.** At T=2048 it splits into warp/tile arrangement (+1.1% for
  N16, +4.4% for N8) and format dispatch (+3.4%, +5.2%).
- **Map content does not matter.** A map's E0M3 tiles themselves cost nothing measurable (±0.3%
  against all-flags-clear).
- In-model, with other kernels interleaved, the overhead is +6.1% (N16) and +12.8% (N8).

**Activation quantization** (the harness's PyTorch implementation) is the only runtime cost 4Over6 adds:
two scale candidates plus an error comparison per block. It rises from about 221 to about 270 ms per
forward (+22%); MixFP4 inherits it.

**End-to-end prefill** (T=2048, batch 1, median wall time):

| Policy | Latency | vs NVFP4 |
|---|---:|---:|
| BF16 | 122.2 ms | |
| NVFP4 | 453.5 ms | |
| 4Over6 | 519.0 ms | +14.5% |
| N8K64 | 522.0 ms | +15.1% |
| N16K64 | 531.8 ms | +17.3% |
| Fake-quant 4Over6 | 413.0 ms | |

N16K64 includes about 15–20 ms of transposition, because weights-on-A computes Dᵀ. Its flags-clear
4Over6 control on the same build is 17.6 ms slower than stock 4Over6, of which 1.2 ms is GEMM. These end-to-end
numbers are **not deployment latencies**: unfused PyTorch activation quantization and packing take about
85% of GPU time, against about 5% for the FP4 GEMM.

### Kernel-design fixes

**1. Column-major D for weights-on-A** (`build.sh wt_as_A_colD`).
- The patch is a two-line layout guard on a snapshot copy of the kernel; the mainloop, dispatch and
  SASS sites are untouched, with an identical OMMA census.
- D = W·Xᵀ stored column-major *is* the row-major [tokens, out] tensor, so the transpose disappears.
- Checks: the kernel's own self-test passes, and the unpatched control fails (0.447). Linear outputs
  and full-model logits are bit-identical to the transpose path. Throughput is unchanged: 1397.7 vs
  1396.7 TFLOP/s at 4096³.
- In an interleaved prefill test (`realquant/bench_colD_e2e.py`), N16K64 drops from 501.4 to 489.0 ms,
  2.0 ms faster than N8K64 and 0.8 ms slower than stock 4Over6.

**2. Fused Triton activation quantizer** (`realquant/fused_quant.py`). One launch per Linear quantizes,
packs, and writes UE4M3 scale bytes straight into the GEMM's swizzled layout.

It is **bit-identical** to the reference on real Llama activations (`realquant/test_fused_quant.py`:
448 inputs, 5.1 G elements, 319 M blocks, both quantizers). Matching required three torch/compiler
details:
- torch computes division by a Python scalar as a multiply by the FP32 reciprocal, which differs from
  IEEE division in about 22% of cases;
- torch's 16-term `.sum(-1)` is an adjacent pairwise tree;
- on sm_120, ptxas contracts Triton's packed `mul.rn.f32x2` + `add.rn.f32x2` into FFMA, so the squares
  and tree additions are forced to scalar explicitly-rounded ops.

Logits match the PyTorch-quantizer path bit for bit for every policy.

**End-to-end prefill with both fixes** (`realquant/bench_fused_e2e.py`, single-pass epilogue,
interleaved):

| Policy | PyTorch quant | Fused quant | vs NVFP4 | FP4 GEMM | Act. quant kernel |
|---|---:|---:|---:|---:|---:|
| BF16 | 120.1 ms | | | | |
| NVFP4 (stock) | 436.8 ms | **59.2 ms** | | 22.1 | 6.4 |
| 4Over6 (stock) | 488.6 ms | **59.9 ms** | +1.1% | 21.5 | 7.6 |
| N16K64 (`wt_as_A_colD`) | 489.6 ms | **60.9 ms** | +2.8% | 22.3 | 7.6 |
| N8K64 (`b8x64`) | 491.3 ms | **62.2 ms** | +5.1% | 23.7 | 7.6 |

- With both fixes, every FP4 policy runs at about 2× the BF16 speed.
- 4Over6's whole runtime cost is its second scale candidate in the quantizer (+1.2 ms kernel time).
- MixFP4 adds its GEMM dispatch on top: about +0.8 ms for N16 and +2 ms for N8.
- Not yet done: folding the per-token scale into the GEMM epilogue (the single-pass output stage is
  about 7 ms) and CUDA Graphs (about 2 ms idle). Both would apply to every FP4 policy alike.

## Time per model

GPU stage wall times, from launch records. Llama and Qwen3-4B shared the GPU with other jobs, so their
numbers are upper bounds.

| Model | Download | Calibration | Fake PPL | Native PPL | GPU total |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 21 min | 9 min | 32 min | 62 min | 103 min |
| Qwen3-4B | 10 min | 11 min | 19 min | 35 min | 64 min |
| Qwen3.8-27B | 68 min | 27 min | 62 min | 83 min | 171 min |
| Mistral-7B-v0.3 | 17 min | 6 min | 15 min | 25 min | 46 min |
| Phi-4 | 37 min | 11 min | 23 min | 33 min | 66 min |

The corrected re-runs of the three weights-on-A policies took a further 12, 9, 49, 13 and 16 min
respectively.

## Limitations

- **Single calibration seed on one machine.** Original maps are not published, so tile-level Jaccard
  against the original cannot be computed. Only counts and effects are compared.
- **The native path is an accuracy harness, not a deployment path:**
  - quantization and packing run in PyTorch;
  - the per-token scale is applied after the GEMM, with a second bf16 rounding;
  - E0M3 relies on the undocumented, patched SASS encoding validated in the mixfp4 repo.
- **Protocol deviations:**
  - no Slurm and a single GPU (the original sharded Qwen3.8-27B across two GPUs);
  - `--teacher none` in PPL runs (teacher only feeds KL diagnostics, not PPL);
  - the public `PROTOCOL_FREEZE.json` hash (`f917f4b2…`) differs from the original sealed hash
    (`df78f1fb…`) because the public copy is redacted.

## Reproduce

```bash
source repro_local/env.sh
repro_local/realquant/build.sh wt_as_A lib && repro_local/realquant/build.sh b8x64 lib   # kernel
$PY repro_local/realquant/test_rq.py                                                     # unit gate
repro_local/pipeline.sh <model> [--moments-device cpu]     # calib -> fake PPL -> native PPL
repro_local/rerun_wt_as_A.sh                               # corrected weights-on-A native runs
$PY repro_local/analysis/final_report.py                   # REPORT_TABLES.md
# latency study
for c in stock wt_as_A_nodisp b8x64_nodisp; do repro_local/realquant/build.sh $c lib; done
$PY repro_local/realquant/bench_latency_llama.py
```

Run records live in `/home/dev/n16k64_campaign/runs/<stage>_<model>_attempt*/`: launch record with
source-manifest hash, maps, moments, per-window NLL.

[`results/`](results/) holds a committed copy of the small, load-bearing part of every run:
- launch records and job results;
- per-window NLL reports and window token hashes;
- calibration reports and manifests;
- the ten evaluated N8/N16 k=3 maps, with provenance sidecars;
- the latency study.

Calibration is not bit-reproducible, so these maps are what allows the PPL numbers to be re-evaluated
exactly. Moments, raw scores and per-token arrays are excluded for size.
