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

The only difference from fake quant is therefore the GEMM itself. The kernel's epilogue carries only a
scalar alpha, so the per-token activation global scale is applied to the bf16 output afterwards.

## Reproduction criteria

The original campaign's own pre-registered cross-GPU tolerances (`PROTOCOL_FREEZE.json`,
`numerical_tolerances`):

- W4A4 PPL within 0.5% relative, BF16 within 0.1%.
- Paired effect ΔlogPPL (tierC): same sign, and |local − original| ≤ max(0.25·|original|, 0.002).
- k=3 tile count within 5% (historical tierB).

## Results: original / fake quant / native kernel

| Model | Corpus | NVFP4 | 4Over6 | N8K64(k=3) | N16K64(k=3) |
|---|---|---:|---:|---:|---:|
| Llama3.1-8B | Wiki | 6.9317 / 6.9296 / 6.9382 | 6.8754 / 6.8807 / 6.8826 | 6.8414 / 6.8441 / 6.8479 | 6.8432 / 6.8416 / 6.8468 |
| Llama3.1-8B | C4 | 9.9379 / 9.9302 / 9.9386 | 9.8254 / 9.8141 / 9.8209 | 9.7763 / 9.7774 / 9.7802 | 9.7762 / 9.7771 / 9.7831 |
| Qwen3.8-27B | Wiki | 7.5769 / 7.5446 / 7.5420 | 7.2839 / 7.3016 / 7.2911 | 7.2254 / 7.2569 / 7.2515 | 7.2456 / 7.2587 / 7.2581 |
| Qwen3.8-27B | C4 | 10.2237 / 10.2222 / 10.2225 | 10.1894 / 10.1882 / 10.1899 | 10.1491 / 10.1500 / 10.1473 | 10.1564 / 10.1559 / 10.1572 |
| Qwen3-4B | Wiki | 13.9557 / 13.9584 / 13.9313 | 14.2062 / 14.2183 / 14.2106 | 11.8841 / **11.7066** / **11.6925** | 12.1968 / **12.1095** / **12.0888** |
| Qwen3-4B | C4 | 17.2621 / 17.2493 / 17.2596 | 17.3024 / 17.3175 / 17.3062 | 15.8483 / **15.7092** / **15.6958** | 16.0485 / 15.9714 / **15.9623** |
| Mistral-7B-v0.3 | Wiki | 5.5487 / 5.5508 / 5.5511 | 5.5233 / 5.5210 / 5.5213 | 5.4987 / 5.5029 / 5.5049 | 5.5019 / 5.5029 / 5.5035 |
| Mistral-7B-v0.3 | C4 | 8.0938 / 8.0949 / 8.0934 | 8.0674 / 8.0681 / 8.0669 | 8.0456 / 8.0449 / 8.0508 | 8.0475 / 8.0521 / 8.0505 |
| Phi-4 | Wiki | 6.7013 / 6.6998 / 6.7013 | 6.6641 / 6.6627 / 6.6625 | 6.6152 / 6.6324 / 6.6347 | 6.6288 / 6.6366 / 6.6396 |
| Phi-4 | C4 | 10.5838 / 10.5834 / 10.5807 | 10.5485 / 10.5473 / 10.5404 | 10.5000 / 10.5132 / 10.5161 | 10.5073 / 10.5205 / 10.5176 |

Bold marks cells outside the 0.5% tolerance. Per-cell relative differences, paired effects with 2SE and
native−fake differences are in [REPORT_TABLES.md](REPORT_TABLES.md).

### Scorecard

| Item | Result |
|---|---|
| BF16 PPL (fake path, 10 cells) | all within 0.1% (max 0.021%) |
| NVFP4 / 4Over6 PPL (40 cells, fake + native) | **all within 0.5%** (max 0.46%) |
| N8 / N16 PPL (40 cells) | 33 within 0.5%; **7 outside, all Qwen3-4B** (−0.48% … −1.61%, lower than original) |
| Effects vs 4Over6 (60 = 3 contrasts × 2 corpora × 5 models × 2 paths) | **57 pass tierC**; 3 fail: Phi-4 N8 Wiki (fake −0.0046, native −0.0046 vs −0.0074), Qwen3.8-27B native N8 Wiki (−0.0053 ± 0.0033 vs −0.0081) |
| Effect direction | every E0M3 map improves on 4Over6 in all 40 cells; NVFP4−4Over6 sign reproduced in all 20 |
| k=3 tile counts (10) | 1 within 5% (Llama N16 +4.8%); others −35% … +11% |
| Native vs fake (40 paired comparisons) | 21 positive / 19 negative, **0 beyond 2SE**, mean −0.00008 NLL |

## Why the deviations happen

**Tile counts.** The inputs are verified identical:
- calibration token hashes match the archived manifests (Llama, Qwen3-4B, Qwen3.8-27B) or the original
  seed0 manifests (Mistral: every document and token hash; Phi-4: manifest SHA `ad9b2521…` equals
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
Llama effect, so
10.9% / 7.9% more elected tiles translate directly into lower PPL. The effect sizes pass tierC and lie
inside the original method's own seed0+draw range:

| Qwen3-4B N16 effect | Original seed0 + 4 draws | Local |
|---|---|---:|
| Wiki | −0.125 … −0.168 | −0.161 (fake), −0.162 (native) |
| C4 | −0.063 … −0.084 | −0.081 (fake), −0.081 (native) |

**Phi-4 N8 effect.** With 35% fewer tiles, the E0M3 improvement shrinks to about 60–70% of the
original, although it is still significant (−0.0046 ± 0.0014). Direction and existence reproduce;
magnitude scales with the elected tile count.

**Numerical noise scale.** On this machine, changing only the attention implementation (eager vs
SDPA) moves the same FourOverSix evaluation by per-window NLL SD 0.010–0.012 and aggregate
ΔlogPPL −0.0004 / +0.0006. W4A4 per-token activation quantization amplifies kernel-level rounding
differences. All baseline PPL differences from the original lie within this band.

## Native kernel vs fake quant

The native path reproduces the fake-quant numbers within noise on every model: 40 paired comparisons
with no systematic bias. Llama happened to show 10/10 positive differences and Qwen3-4B 9/10 negative;
none is individually significant.

Per-layer latency: on Llama shapes the FP4 GEMM beats cuBLAS BF16 by 2.7–3× on the large projections
(737–1033 vs 272–360 TFLOP/s) and by 1.5× on the small k_proj. The harness is still about 1.3× slower
end to end (Qwen3.8-27B, sole GPU user: 16.5 vs
12.4 min per policy) because activation quantization, nibble encoding and the per-token epilogue run as
unfused PyTorch ops with verification syncs. See `realquant/bench_linear.py`.

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
$PY repro_local/analysis/final_report.py                   # REPORT_TABLES.md
```

Run records live in `/home/dev/n16k64_campaign/runs/<stage>_<model>_attempt*/`: launch record with
source-manifest hash, maps, moments, per-window NLL.

[`results/`](results/) holds a committed copy of the small, load-bearing part of every run:
- launch records and job results;
- per-window NLL reports and window token hashes;
- calibration reports and manifests;
- the ten evaluated N8/N16 k=3 maps, with provenance sidecars.

Calibration is not bit-reproducible, so these maps are what allows the PPL numbers to be re-evaluated
exactly. Moments, raw scores and per-token arrays are excluded for size.
