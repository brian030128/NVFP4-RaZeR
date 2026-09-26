# TM-OPT+TC ("method B"): the tile-gradient GEMM on tensor cores — report

**Status (2026-09-26 14:30 UTC): stopped at the registered unit test, which failed.** The queue
stopped before the nine runs, as the protocol requires. **The user decides how to continue.**

- **Protocol:** [PROTOCOL_TC.md](PROTOCOL_TC.md), registered 2026-09-26 14:05:12 UTC (sha256
  c84c2908…) before any test or run.
- **Implementation:** TM-OPT+TC computes the tile-gradient GEMM with `torch.mm(dy.T, x,
  out_dtype=float32)`: BF16 tensor cores, FP32 accumulation and FP32 output. This is the task's
  alternative, chosen because it is faster than TF32 (and, in the pre-registration check, also more
  accurate). The choice is disclosed in the protocol.

## 0. Refactor check: PASS

The group-1 TM-OPT command (STE 8x64, 3 epochs, fake evaluation) was rerun with the new code
(profiling regions, a split hook, TC off). It reproduces `runs/g1_ste_tmopt_nob1` **bitwise**:
- θ after all 48 steps;
- every epoch value and the monitor's per-document CE and KL;
- the maps and every WikiText-2 and C4 window NLL.

## 1. Profile: where TM-OPT's 35 s per epoch go (Llama-3.1-8B, 8x64, one epoch, 16 steps)

GPU seconds per epoch, by phase and category (each kernel counted once, under its innermost named
region):

| phase | category | TM-OPT | TM-OPT+TC |
|---|---|---:|---:|
| forward | model GEMMs (bf16) | 3.03 | 2.97 |
| | activation quantization (fused, per token) | 1.49 | 1.48 |
| | other kernels (norms, rotary, elementwise) | 0.79 | 0.79 |
| | lean weight decode | 0.26 | 0.26 |
| | attention | 0.05 | 0.05 |
| | **total GPU / wall** | **5.63 / 4.73** | **5.56 / 4.62** |
| loss | teacher host-to-device copies | 1.30 | 1.29 |
| | KL gradient (chunked) | 0.68 | 0.68 |
| | **total GPU / wall** | **1.99 / 5.29** | **1.98 / 5.15** |
| backward | **tile-gradient GEMM G = dyᵀx** | **14.70** | **2.97** |
| | model GEMMs (bf16) | 3.33 | 3.06 |
| | hook: candidate decode + D = A − B | 2.70 | 2.69 |
| | hook: G⊙D, tile reduction, accumulation | 1.81 | 1.81 |
| | other kernels | 1.26 | 1.25 |
| | lean weight decode (recomputed for the backward) | 0.22 | 0.21 |
| | attention | 0.21 | 0.21 |
| | **total GPU / wall** | **24.31 / 21.35** | **12.28 / 10.69** |
| optimizer | Adam, flips | 0.06 / wall 0.71 | 0.06 / wall 0.68 |
| **epoch** | **wall under the profiler (GPU total)** | **35.3 s (32.0 s)** | **23.0 s (19.9 s)** |

**Where the 35.2 s go:**
- **The tile-gradient GEMM dominates.** At 14.7 s it is 46 % of the epoch's GPU time and 60 % of
  the backward. It runs as an FP32 SIMT GEMM over 4,096 tokens per module and step.
- **The hook costs another 4.5 s:** 2.7 s to decode the candidates and form D = A − B, and 1.8 s
  for G⊙D, the tile reduction and the accumulation.
- **The model's own GEMMs** take only 6.4 s (forward and backward), and the activation quantizer
  1.5 s.
- **The loss phase is host-bound:** 5.3 s of wall time for 2.0 s of GPU work. The teacher
  log-probabilities (1 GB per step, bf16, pageable CPU memory) are concatenated on the CPU and
  copied synchronously. This is a further speed-up opportunity (pinned memory and an asynchronous
  prefetch, or a GPU-resident teacher where memory allows). It is **not** part of this task.

**With TC:**
- **The tile-gradient GEMM is 4.9× faster** (14.70 → 2.97 s).
- **The epoch drops from 35.3 to 23.0 s wall** (−35 %, profiled). That is below QAT C1's
  non-deterministic 26.9 s per epoch; the TC epoch here is deterministic.
- **Nothing else moves.**

## 2. Unit test (registered): **FAIL**

`repro_local/realquant/test_tile_grad_tc.py`, `runs/unit_tests_tc.json`:
- **Cases:** one real and one random matrix of every distinct layer-0 shape of the three models,
  at 8x64, 16x64 and 256x64 (72 cases); 4 batches of 4,096 tokens each.
- **Error:** normwise relative error of the tile sums vs FP64, the maximum over the batches.

| | FP32 path (TF32 off) | TM-OPT+TC |
|---|---:|---:|
| error vs FP64, range over the cases | 2.4e-7 to 1.45e-6 | 3.9e-6 to 6.5e-6 |
| TC error / FP32 error | — | 4.4× to 20.7× |
| GEMM time, sum over the shapes | 123.3 ms | 28.8 ms (4.3× faster) |

- **Registered tolerance:** error(TC) ≤ max(2 × error(FP32), 2e-6). It is **not met in any of the 72
  cases.**
- **The pre-registration check had predicted this.** BF16 tensor cores accumulate FP32 sums less
  accurately than the SIMT FMA path, even though every product is exact. The TF32 variant was
  worse still (about 1e-5).

## Decision for the user (nothing runs until then)

- **(A) Run the registered end-to-end test anyway.** The nine TM-OPT+TC runs and their evaluations,
  about 2.5 h. The per-cell acceptance (not significantly worse than TM-OPT on either corpus) and
  the comparison with #2's seed spread then decide. An error of ~5e-6 in the tile gradients may well
  be smaller than the path changes a new seed causes, but that is what the runs would test.
- **(B) Keep the FP32 tile-gradient GEMM.** TM-OPT stays at ~35 s per epoch.
- **(C) A compensated tensor-core GEMM, a new variant to verify first.** Split the token dimension
  into short chunks (for example the 8 sequences of 512 tokens), run each on tensor cores, and add
  the partial G's in FP32 (or FP64). This bounds the tensor-core accumulation length and should
  bring the error close to the FP32 path, at some cost in memory traffic. It needs its own unit
  test.
- **(D) Relax the tolerance**, for example to 1e-5, and continue with (A).

## Reproduction

```
/home/dev/n16k64_campaign/tm_opt/queue_tc.sh     # copy in runs/queue_tc.sh, log in runs/commands_tc.log
python results/tm_opt/analyze_tc.py profile      # tc_profile.{json,md}
```
