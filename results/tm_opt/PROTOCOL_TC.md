# TM-OPT+TC ("method B"): the tile-gradient GEMM on tensor cores — protocol

Written 2026-09-26 on branch `tm-opt`, before any test or run of this change. The hash and
registration time are in `registration_tc.json`. Deviations are appended in the last section,
never edited in place. Nothing is selected on WikiText-2, C4 or zero-shot.

**Task (user-approved, relayed by nvfp4-razer-c9).** Add method B to TM-OPT and measure its effect
on speed and PPL on Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4. The Qwen3.8-27B gate stays closed.

## What changes (`run_train_map.py`)

- **`--tile-grad-tc`, off by default:** the hook's tile-gradient GEMM `G = dy.float().T @ x.float()`
  (FP32 SIMT) is replaced by `tc_matmul(dy.T, x)` = `torch.mm(dy.T, x, out_dtype=torch.float32)`.
  That runs on the **BF16 tensor cores with FP32 accumulation and FP32 output**; nothing is rounded to
  bf16. No global flag is touched (`allow_tf32` stays False).
- **Why this is nearly exact:** dy and x are bf16, so every product is exact in FP32; only the
  accumulation differs. The model's own GEMMs are unchanged.
- **Implementation choice, disclosed.** The task allowed TF32 around the matmul (preferred) or BF16
  inputs with FP32 accumulation and output ("if that is faster"). Before this registration, a
  development check on random bf16 operands measured all three:

  | GEMM | normwise error vs FP64, T = 4096 | time, 4096 × 4096 × 4096 |
  |---|---:|---:|
  | FP32 SIMT (current) | 2.3e-7 to 5.1e-7 | 1.59 ms |
  | TF32 | 9.4e-6 to 1.0e-5 | 0.65 ms |
  | BF16 inputs, FP32 accumulation and output | 4.7e-6 to 5.3e-6 | 0.33 ms |

  - **BF16 is chosen:** it is faster (the task's rule) and also more accurate than TF32.
  - **Expected outcome:** that check suggests the registered unit-test tolerance, max(2 × FP32 error,
    2e-6), will probably not be met. The tolerance is the user's and is registered unchanged; a
    failure stops the queue for the user.
- **Naming:** the configuration is **TM-OPT+TC** (`--tm-opt --tile-grad-tc`). It is not part of the
  TM-OPT preset, and it excludes B1.
- **Profiling:** named phases (forward, loss, backward, optimizer) and regions:
  - lean weight decode;
  - activation quantization;
  - hook candidate decode + D;
  - the tile-gradient GEMM;
  - G⊙(A−B) with tile reduction and accumulation.

  `--profile JSON` profiles one training epoch after the setup, then stops. The hook's one-line
  expression is split into the same operations in the same order.

## Checks and measurements, in order

0. **Refactor check** (before anything else). The exact group-1 TM-OPT command (STE 8x64, 3 epochs,
   init −0.2, fake evaluation, θ hashes) is rerun with the new code, TC off. It must reproduce
   `runs/g1_ste_tmopt_nob1` bitwise: θ after every step, maps, monitor values, window NLLs.
   Otherwise the queue stops.
1. **Profile** (Llama-3.1-8B 8x64, TM-OPT, one epoch, then the same with TC):
   - GPU time per phase and per category: forward, backward, the tile-gradient GEMM, G⊙(A−B) with
     tile reduction, candidate decode, activation quantization, loss, optimizer, and the model's
     GEMMs and attention;
   - the epoch's wall time with the profiler, next to the unprofiled 35.2 s.
2. **Unit test** (`repro_local/realquant/test_tile_grad_tc.py`):
   - **Cases:** one real matrix of every distinct layer-0 shape of the three models, plus a random
     matrix of each shape; 8x64, 16x64 and 256x64; 4 batches of 4,096 tokens.
   - **Comparison:** the TC path and the FP32 path (TF32 off), each against an FP64 reference. A
     path's error is its normwise relative error (max|Δ| / max|ref|), maximized over the batches.
   - **Registered tolerance, in every case:** error(TC) ≤ max(2 × error(FP32), 2e-6).
   - **On failure:** it is recorded as such, the queue stops, and the user decides.
3. **Nine TM-OPT+TC runs.**
   - **Models and units:** Llama → Mistral → Phi-4, each at 8x64, 16x64 and 256x64.
   - **Settings, as the committed TM-OPT runs:** seed 0, deterministic, STE, main's hyperparameters
     (lr 0.02, init −1, Adam β (0.9, 0.999), ε 1e-12, batch 8, 20 epochs, monitor every 2), native
     monitor and evaluation, the same data roots.
4. **Evaluations** (convention (a)):
   - **Native, one process per model:** FourOverSix, the committed TM-OPT maps and the TC maps.
   - **Fake:** FourOverSix and the TC maps.
   - **Repeat check:** the FourOverSix and TM-OPT window NLLs must repeat the committed item-#1
     evaluations bitwise; this is reported.
   - **Pairing with committed NLLs:** MR-OPT (native) and TM-OPT (fake) are paired through the
     committed NLLs.
5. **Non-deterministic timing probe:** TM-OPT+TC on Llama 8x64 with `--no-deterministic`, 3 epochs.
   It is compared with QAT C1's non-deterministic 26.9 s per epoch (batch 8, 16 steps).

## Reported and decided

- **Per cell (model × unit):**
  - native (primary) and fake PPL;
  - paired ΔNLL ± 2 SE vs the committed TM-OPT map, vs FourOverSix and vs MR-OPT;
  - tile overlap with TM-OPT (shared, only each, Jaccard);
  - E0M3 tiles and the development-KL curve.
- **Acceptance per cell:** TM-OPT+TC is not significantly worse than TM-OPT on either corpus:
  mean − 2 SE ≤ 0 on WikiText-2 and on C4.
- **Next to #2's seed spread:**
  - **Why:** TC changes the path the way a new seed would.
  - **The differences:** TM-OPT+TC − TM-OPT is set against #2's seed spread on Llama 8x64
    (ranges 0.0011 / 0.0004) and its seed-vs-seed differences (|ΔNLL| up to 0.0011 / 0.0004).
- **Timing and memory:**
  - per-epoch and selection time, TM-OPT vs TM-OPT+TC, per model and unit;
  - the non-deterministic probe's per-epoch time;
  - peak GPU and host memory.

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after
  the unit test (with the profile), and after the runs. Push each time (never main, no force). If a
  push is blocked, report it and do not retry.
- **Reporting:** report to the user after the profile and unit test, and after the runs. Then stop.
- **Scope:** the Qwen3.8-27B gate, Part C, QAT and zero-shot stay closed or paused.

## Deviations (append-only)

(none yet)
1. **2026-09-26 14:16 UTC: the registered unit test failed; the queue stopped before the nine
   runs, as registered.**
   - **Result:** TC's error vs FP64 is 3.9e-6 to 6.5e-6, against 0.24e-6 to 1.45e-6 for the FP32
     path. The threshold, max(2 × FP32 error, 2e-6), is exceeded in all 72 cases.
   - **Before the test:** the refactor check passed (bitwise) and both profiles completed.
   - **Next:** the user decides (REPORT_TC.md, options A–D).
