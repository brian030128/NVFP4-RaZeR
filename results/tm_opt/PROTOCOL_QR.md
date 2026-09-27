# Part Q (Qwen3.8-27B TM-OPT and TM-OPT+TC) and Part R (final four-model analysis) — protocol

Written 2026-09-26 on branch `tm-opt`, after the TM-OPT+TC report was committed and pushed, and
before any Part Q or R run. The hash and registration time are in `registration_qr.json`.
Deviations are appended in the last section, never edited in place. Nothing is selected or tuned on
WikiText-2, C4 or zero-shot.

**Task (user-approved, relayed by nvfp4-razer-c9).** The user opens the Qwen3.8-27B gate for this
task only.

## Part Q: Qwen3.8-27B, TM-OPT and TM-OPT+TC at 8x64, 16x64 and 256x64 (6 runs)

- **Settings:** those of the other models:
  - seed 0, deterministic, STE;
  - main's hyperparameters: lr 0.02, init θ −1, Adam β (0.9, 0.999), ε 1e-12, constant lr,
    20 epochs, monitor every 2 epochs, monitor batch 16;
  - lean memory, chunked loss, fused activation quantization, native monitor and evaluation,
    single-pass epilogue, `expandable_segments`.
- **Model and data:**
  - the Qwen data of Part C's multimodel data root (`/home/dev/n16k64_campaign/multimodel/data/qwen27b`);
  - the pinned revision 1d4bf0f2…;
  - the native Transformers 5.16.1 Qwen3.5-family code (`Qwen3_5ForConditionalGeneration`,
    `--gpus 1`).
  - **Quantized:** the 496 text Linear matrices. The recurrent, convolution, norm, vision and head
    parameters are not quantized, as in Part C.
- **Code changes, needed for Qwen and changing nothing for the other models:**
  - **The Qwen batch-1 assertion of main's `run_train_map.py` is removed.** The user accepted batched
    numerics for all models: MR-OPT runs every model at batch 16/8, and Qwen3.8-27B's batched forward
    is not bitwise equal to batch 1, as for the other models.
  - **`--probe-steps N`:** stop after N optimizer steps and record the per-phase peak memory (batch
    probe).
  - **An out-of-memory hook:** a CUDA out-of-memory exit records the per-phase peaks in the report.
- **The batch.** The target is batch 8 as for the other models.
  1. **Probe micro-batch 8** (`--batch 8 --accum 1`, one optimizer step, TM-OPT at 8x64) and record
     FIT or OOM with the peak memory.
  2. **If it runs out of memory:** keep the **optimizer batch at 8 sequences** with gradient
     accumulation (`--batch b --accum 8/b`). Use the largest b in {4, 2, 1} whose probe fits.
  3. **What stays the same:** 16 optimizer steps per epoch, the sequences of every step (the
     shuffled order is cut into groups of 8 consecutive sequences either way), the loss (the mean
     over the 8 sequences) and every hyperparameter. Only the forward/backward micro-batching
     changes.
  4. **Recording:** the probes and the chosen b are recorded as a deviation.
  5. **If even micro-batch 1 runs out of memory:** stop and report. No offloading is improvised.
- **Runs** (after the probe): TM-OPT at 8x64, 16x64, 256x64, then TM-OPT+TC at 8x64, 16x64,
  256x64, with the chosen micro-batch. The monitor evaluates at batch 16 (monitor only).
- **Reported per run:**
  - the probes (FIT/OOM), the micro-batch and accumulation;
  - per-epoch time, selection time, and setup time (load, teacher, packing);
  - peak GPU allocated/reserved and host RSS;
  - E0M3 tiles and the development-KL curve.
- **Evaluation** (convention (a); one native and one fake process):
  - the maps: FourOverSix, NVFP4, the three TM-OPT maps and the three TM-OPT+TC maps;
  - BF16 in its own fake process;
  - the checks: window token hashes, and the map expansion checks.

## Part R: final analysis over Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B

- **Methods and units:** TM-OPT and TM-OPT+TC; units 8x64, 16x64 and 256x64.
- **Baselines:** pure NVFP4, FourOverSix and BF16.
- **MR-OPT** is included as a reference where it exists (not for Qwen3.8-27B).

### R1. Calibration cost

- **Per model × unit × method:** per-epoch time, selection time, setup time, and peak GPU/host
  memory.
- **Source:** the existing run records; only what is missing is measured.
- **Context rows:** MR-OPT, and QAT C1 on Llama (`results/cost_comparison`: one epoch over the 128
  calibration sequences, 26.9 s of training, non-deterministic).

### R2. Evaluation-time latency on the SM120 deployment kernels (speed only)

- **Kernels:** the builds in `/home/dev/n16k64_campaign/sm120_bench` (exported from
  `origin/SM120-kernel` f91c109; the working tree is not touched).
- **Policies:**
  - **BF16:** the model as loaded.
  - **Pure NVFP4:** NVFP4 weights and NVFP4 activation quantization on the stock kernel.
  - **FourOverSix:** FourOverSix weights and FourOverSix activation quantization on the stock
    kernel. The weight choice is offline.
  - **MixFP4, the TM-OPT and TM-OPT+TC maps:**
    - 8x64 on `n8k64_wB` (weights on B);
    - 16x64 on `n16k64_wA` (weights on A);
    - **256x64 on `n16k64_wA`, as unions of 16x64 granules.** Each 256-row tile becomes its 16
      16x64 tiles; the conversion is exact and checked.
  - **Stock baselines:** `stock_wA` (the deployment placement) and `stock_wB` (placement-matched
    to 8x64).
- **Profiler decomposition, per policy** (kernel classes by name):
  - FP4 GEMM kernel time (the per-token scale and epilogue are fused into it);
  - activation quantization (NVFP4 single candidate vs FourOverSix two candidates);
  - the mixed-kernel overhead (mixed vs stock GEMM time on the same shapes);
  - everything else: attention, norms and elementwise, the bf16 lm_head;
  - end-to-end prefill.
- **Shapes and timing:**
  - prefill 1×2048 and 4×2048, plus 1×512 with the bimodality caveat;
  - decode (batch 1, tokens/s) only if the narrow-tile builds (`*_n16/_n32/_n64`) build and pass
    their checks without trouble; otherwise not measured;
  - one process per policy, the policy order randomized over at least 5 rounds, medians and
    min–max reported (the 500 W power cap);
  - idle GPU (`require_idle`); all four models if they fit (Qwen BF16 needs about 54 GB).
- **Reported:** overheads in % vs pure NVFP4 and vs FourOverSix.
- **Numerics caveat, stated:** the deployment kernel uses per-token activation scales and a
  one-rounding epilogue; our PPL convention differs. R2 is speed only.

### R3. PPL

- **Per model × unit × method:** WikiText-2 and C4, native (primary, convention (a)) and fake.
- **Paired ΔNLL ± 2 SE vs FourOverSix and vs pure NVFP4;** BF16 alongside.
- **Also:** MR-OPT as a reference, and TM-OPT vs TM-OPT+TC.
- **Source:** the committed evaluations, paired within a process or across processes where the
  FourOverSix NLLs repeat bitwise. Only Qwen3.8-27B is newly evaluated.

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after Part
  Q and after Part R. Push each time (never main, no force). If a push is blocked, report it and do
  not retry.
- **Reporting:** report to the user after the Qwen batch probe, after Part Q, and with the final
  Part R report. Then stop.
- **Scope:** Part C, QAT and zero-shot stay paused. The Qwen gate is open for this task only.

## Deviations (append-only)

1. **2026-09-26 18:12 UTC: the Qwen3.8-27B batch probe; micro-batch 2 × accumulation 4.**
   - **Probes** (TM-OPT at 8x64, one optimizer step of 8 sequences, `expandable_segments`):

     | micro-batch × accum | result | training-phase peak GPU allocated / reserved |
     |---|---|---:|
     | 8 × 1 | OOM | 93.8 / 94.3 GiB |
     | 4 × 2 | OOM | 93.9 / 94.3 GiB |
     | 2 × 4 | **FIT** | 90.0 / 91.5 GiB |

   - **Where the OOMs happened:** both in the training forward, inside the Transformers torch
     implementation of the gated-delta-rule linear attention (the per-row loop of the chunked
     recurrence). The setup phases fit in all three probes: at most 56.0 GiB allocated (candidate
     packing), with a host RSS of 79.3 GiB.
   - **Chosen:** micro-batch 2 with accumulation 4, so the optimizer batch stays at 8 sequences, as
     registered. The steps, the sequences of each step, the loss and every hyperparameter are
     unchanged. The first probe step took 16.2 s.
   - **Micro-batch 1 was not probed:** the rule takes the largest b that fits.
   - **Code:** `run_train_map.py` is as registered (sha256 5d5964473745…). The queue is
     `/home/dev/n16k64_campaign/tm_opt/queue_q.sh` (9d6ea58280e1…). The six runs started at 18:12
     UTC.
   - **R2 preparation:** the latency queue (`sm120_bench/queue_latency.sh`, scripts in
     `results/tm_opt/latency/`) was started at 18:04 UTC. It waits, and runs no GPU work before
     "PART Q DONE", so Part Q's timings are not perturbed. The map conversion (CPU) runs after the
     Qwen runs.
