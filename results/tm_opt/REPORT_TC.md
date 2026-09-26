# TM-OPT+TC ("method B"): the tile-gradient GEMM on tensor cores — report

**Status (2026-09-26 18:00 UTC): complete. Stopped, as instructed.**
- **The registered unit test failed** (deviation 1; unchanged).
- **The user chose option A** (deviation 2). The registered end-to-end test was run anyway.

**Verdict: all 9 cells pass the registered acceptance.** In each cell (model × unit), TM-OPT+TC is
not significantly worse than TM-OPT on either corpus (native, mean − 2 SE ≤ 0). One cell is
significantly better.

- **Speed:** per-epoch time falls 35–39 %, and selection time 30–35 %.
- **Memory:** unchanged.
- **Recommendation, by the user's rule: TM-OPT+TC**, with this disclosure:
  - **The registered unit test failed.** TC's tile-gradient sums are 3.9e-6 to 6.5e-6 from FP64 in
    normwise terms, 4.4× to 20.7× the FP32 path's error; the tolerance was max(2 × FP32, 2e-6).
  - **The precision is that of QAT's weight gradients:** BF16 tensor-core products with FP32
    accumulation.
  - **End to end, the maps differ from TM-OPT's the way a new seed's do,** and the PPL differences
    are within #2's seed spread on WikiText-2 (C4 details below).

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

## 3. End to end: nine TM-OPT+TC runs vs the committed TM-OPT runs

- **Settings, as the committed TM-OPT runs:** seed 0, deterministic, STE, main's hyperparameters.
- **Evaluation, per model:**
  - native: FourOverSix, the committed TM-OPT maps and the TC maps, in one process;
  - fake: FourOverSix and the TC maps. The fake TM-OPT side comes from the committed item-#1 fake
    NLLs.
- **Checks passed:** the FourOverSix and TM-OPT window NLLs repeated the committed evaluations
  bitwise, native and fake, on all three models.

**TM-OPT+TC minus TM-OPT** (paired ΔNLL, mean ± 2 SE; n.s. = not significant):

| model | unit | E0M3 tiles TC / TM-OPT | Jaccard with TM-OPT | native ΔWiki | native ΔC4 | acceptable | fake ΔWiki | fake ΔC4 |
|---|---|---|---:|---|---|---|---|---|
| Llama-3.1-8B | 8x64 | 309,517 / 306,968 | 0.593 | +0.00007 ± 0.00143 (n.s.) | +0.00043 ± 0.00103 (n.s.) | **yes** | +0.00068 ± 0.00139 (n.s.) | +0.00084 ± 0.00099 (n.s.) |
| | 16x64 | 201,648 / 200,450 | 0.594 | −0.00062 ± 0.00144 (n.s.) | +0.00051 ± 0.00103 (n.s.) | **yes** | +0.00123 ± 0.00130 (n.s.) | +0.00079 ± 0.00109 (n.s.) |
| | 256x64 | 35,365 / 35,346 | 0.652 | +0.00081 ± 0.00152 (n.s.) | −0.00027 ± 0.00124 (n.s.) | **yes** | +0.00028 ± 0.00157 (n.s.) | +0.00229 ± 0.00178 (worse) |
| Mistral-7B-v0.3 | 8x64 | 339,444 / 340,045 | 0.557 | +0.00003 ± 0.00085 (n.s.) | −0.00015 ± 0.00066 (n.s.) | **yes** | +0.00115 ± 0.00084 (worse) | −0.00103 ± 0.00139 (n.s.) |
| | 16x64 | 224,408 / 223,845 | 0.576 | +0.00066 ± 0.00229 (n.s.) | −0.00026 ± 0.00142 (n.s.) | **yes** | +0.00031 ± 0.00094 (n.s.) | +0.00025 ± 0.00074 (n.s.) |
| | 256x64 | 42,522 / 42,615 | 0.659 | −0.00104 ± 0.00096 (better) | +0.00010 ± 0.00071 (n.s.) | **yes** | −0.00106 ± 0.00093 (better) | −0.00011 ± 0.00085 (n.s.) |
| Phi-4 | 8x64 | 418,023 / 423,066 | 0.500 | +0.00059 ± 0.00120 (n.s.) | −0.00043 ± 0.00071 (n.s.) | **yes** | +0.00034 ± 0.00107 (n.s.) | −0.00031 ± 0.00067 (n.s.) |
| | 16x64 | 287,062 / 281,702 | 0.499 | +0.00080 ± 0.00116 (n.s.) | +0.00009 ± 0.00073 (n.s.) | **yes** | −0.00004 ± 0.00106 (n.s.) | −0.00010 ± 0.00073 (n.s.) |
| | 256x64 | 52,811 / 52,772 | 0.597 | +0.00039 ± 0.00124 (n.s.) | +0.00006 ± 0.00076 (n.s.) | **yes** | +0.00057 ± 0.00131 (n.s.) | +0.00050 ± 0.00071 (n.s.) |

- **Native (primary, registered):** 9 of 9 cells acceptable. There are 17 n.s. comparisons, 1
  significantly better (Mistral 256x64 WikiText-2) and none worse.
- **Fake (secondary, not a criterion):**
  - 2 of 18 comparisons are significantly worse: Llama 256x64 C4 (+0.00229 ± 0.00178) and Mistral
    8x64 WikiText-2 (+0.00115 ± 0.00084);
  - 1 is significantly better;
  - with 18 two-SE comparisons, about one false "worse" is expected by chance. In both cells the
    native result is n.s.
- **Against #2's seed spread** (Llama 8x64: 0.0011 on WikiText-2, 0.0004 on C4):
  - **WikiText-2:** every |native ΔWiki| (at most 0.00104) is within it.
  - **C4:** the native ΔC4 is within 0.0005 everywhere. Three cells (Llama 8x64, Llama 16x64, Phi-4
    8x64: 0.00043–0.00051) are marginally above #2's C4 range of 0.00038, and all are n.s.
  - **Tile overlap:** TC's maps overlap TM-OPT's with Jaccard 0.50–0.66, against 0.44 between
    seeds. TC changes the path the way a new seed does.
- **vs MR-OPT, TM-OPT+TC keeps TM-OPT's advantage** (native):
  - **Llama:** better at all three units on both corpora.
  - **Phi-4:** better on 5 of 6 (16x64 C4 n.s.).
  - **Mistral:** better on 8x64 C4 and 256x64 WikiText-2; the rest n.s. None is worse.

**Time and memory:**

| model | unit | epoch TM-OPT → TC | selection TM-OPT → TC | peak GPU allocated | host RSS |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | 8x64 | 35.2 → 22.8 s | 13.9 → 9.8 min | 40.8 → 40.8 GiB | 42.3 GiB |
| | 16x64 | 35.2 → 22.9 s | 14.0 → 9.8 min | 40.6 → 40.6 GiB | 42.3 GiB |
| | 256x64 | 35.2 → 22.8 s | 14.0 → 9.8 min | 40.5 → 40.5 GiB | 42.2 GiB |
| Mistral-7B-v0.3 | 8x64 | 32.0 → 19.6 s | 12.1 → 7.9 min | 36.2 → 36.2 GiB | 14.7 GiB |
| | 16x64 | 31.9 → 19.7 s | 12.0 → 8.0 min | 36.0 → 36.0 GiB | 14.7 GiB |
| | 256x64 | 31.9 → 19.6 s | 12.0 → 7.9 min | 36.0 → 36.0 GiB | 14.7 GiB |
| Phi-4 | 8x64 | 63.2 → 39.8 s | 23.6 → 15.8 min | 59.7 → 59.7 GiB | 33.7 GiB |
| | 16x64 | 63.1 → 39.8 s | 23.6 → 15.8 min | 59.4 → 59.4 GiB | 33.7 GiB |
| | 256x64 | 63.2 → 39.8 s | 23.6 → 15.9 min | 59.2 → 59.2 GiB | 33.7 GiB |

- **Setup** is unchanged: 1.2–2.1 min.
- **Non-deterministic probe** (TM-OPT+TC, Llama 8x64, `--no-deterministic`, 3 epochs): **21.0,
  20.9, 20.9 s per epoch**, against QAT C1's non-deterministic 26.9 s for one epoch over the same 128
  sequences at batch 8, 16 steps. The deterministic TC epoch is 22.8 s.

## A separate, bitwise-neutral speed-up (not implemented)

**The loss phase is host-bound:** 5.3 s of wall time per Llama epoch against 2.0 s of GPU work. The
teacher log-probabilities (about 1 GB per step, bf16) are concatenated in pageable CPU memory and
copied synchronously.
- **The fix:** pinned host memory with an asynchronous prefetch of the next step's teacher (or a
  GPU-resident teacher where memory allows).
- **Numerics:** it changes nothing (the same bytes arrive).
- **Expected saving:** a few seconds per epoch.
- **Scope:** QAT's KL training could use the same fix. It was not added, as instructed.

## The decision that was needed after the unit test (resolved: option A)


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
/home/dev/n16k64_campaign/tm_opt/queue_tc.sh        # refactor check, profiles, unit test (copy in runs/)
/home/dev/n16k64_campaign/tm_opt/queue_tc_runs.sh   # after the user's decision A: runs, evaluations, probe
python results/tm_opt/analyze_tc.py profile         # tc_profile.{json,md}
python results/tm_opt/analyze_tc.py runs            # tc_runs.{json,md}
```
