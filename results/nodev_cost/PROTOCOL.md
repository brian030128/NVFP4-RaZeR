# Task 1: calibration cost without the development set — protocol

Written 2026-09-27 on branch `tm-opt`, before any Task 1 run. The hash and registration time are in
`registration.json`. Deviations are appended in the last section, never edited in place.

**Task** (user-approved, relayed by nvfp4-razer-c9). A real calibration has no development set. So no development
teacher, development evaluation or monitoring may be counted in the calibration cost. The main method is TM-OPT+TC,
deployed and evaluated with NativeLinear (convention (c)). Task 2 follows under its own protocol.

## Definition of calibration cost (registered)

- **Included:**
  - model load;
  - the fit-set data load (tokenizing the 128 calibration windows) and the fit-set teacher precompute (128
    sequences);
  - the method-specific preparation: candidate packing (TM), or the student's construction with its pristine weight
    copies (QAT and scale-only; phase `preparation`);
  - training;
  - writing the output: `map.pt` for TM; the trained state (`state.pt`: the trained weights for QAT, the scale
    factors for scale-only) for the baselines.
- **Excluded:** every evaluation (development, WikiText-2, C4), every verification check and every development step.
  With the new flags none of them runs.
- **Reported per run:**
  - the wall time of each included phase and their total;
  - the process's total wall time;
  - peak GPU allocated and reserved, the maximum over the included phases (`torch.cuda.max_memory_*`, reset at
    each phase start);
  - peak host RSS (psutil sampled every 0.1 s, per phase; also `ru_maxrss`).

## Code (new flags; the default paths are unchanged)

- **`run_train_map.py --no-dev`:** no development data, development teacher or development evaluation (initial,
  monitor, final), and no per-epoch map checkpoints. The map is written in a `write_output` phase, without
  `theta.pt`, which is not part of the output.
- **`run_train_map.py --no-eval`:** no final WikiText-2 / C4 evaluation, which is done separately with
  `run_ppl_deploy.py`.
- **`run_cost_distill.py --no-dev`:** the same, for QAT and scale-only. The fixed learning rate is unchanged; there
  is no quantizer-verification phase; the trained state is written (forced `--save-state`).
- **`run_cost_distill.py --deterministic`:** `torch.use_deterministic_algorithms(True)` with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and a seeded global RNG (the QAT optimizer's stochastic rounding draws from it).
  Default off, as in the recorded runs.

## Runs (one process per run, sequential, idle GPU)

**(a) TM-OPT+TC with `--no-dev --no-eval`, deterministic (the `--tm-opt` preset), the final settings.**
- Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 at 8x64, 16x64 and 256x64: 9 runs.
- Qwen3.8-27B at 8x64, with micro-batch 2 × accumulation 4.
- **Qwen at 16x64 and 256x64 is not run.** Its per-epoch time and memory were unit-independent in Part Q (216.5–217.0 s,
  89.6–90.4 GiB), so its 8x64 cost stands for all units. These runs are added only if the user asks.

**(b) Bitwise check.** Every deterministic `--no-dev` `map.pt` must equal the committed TM-OPT+TC map of the same
model and unit (sha256; `results/tm_opt/runs/tc/*`, `runs/qwen/q_tc_*`). If any map differs, the queue stops before
the next run and the result is reported.

**(c) Llama-3.1-8B baselines, and the matched comparison.**
- **Settings as in `results/cost_comparison`:**
  - QAT C1: `--arm qat --optimizer adamw_fp32 --micro-batch 8 --checkpointing --lr 1e-6 --budget c1`.
  - Scale-only D1, at its chosen learning rate: `--arm scale --micro-batch 8 --lr 1e-3 --budget c1`.
  - Each with `--no-dev` and 1 epoch (16 optimizer steps of 8 sequences).
- **Determinism:** each is run deterministic off (as recorded) and deterministic on.
- **TM-OPT+TC with deterministic off** (`--no-deterministic`), at 8x64, 16x64 and 256x64. Its maps are compared with
  the deterministic ones descriptively (tile count, Jaccard); they are not expected to be bitwise equal.
- **Labels:** every row is labelled deterministic or not. If a deterministic baseline cannot run (an operation without
  a deterministic implementation), that is reported, not worked around.

**Order.** (a) with the (b) check after each run, then (c). Everything uses the data roots and flags of the committed
runs (`--transformers-deviation` for Llama; `PYTHONPATH` with the recorded `torchao` for QAT).

## Report

- **One cost table:** model × unit × method, with the per-phase times, total, peak GPU and host memory.
- **The QAT / scale-only / TM-OPT+TC comparison on Llama,** deterministic on and off.
- **The difference from the earlier with-dev costs** (`results/tm_opt/final_cost.md`).

Then: commit, push, report to the user, and continue with Task 2.

## Deviations (append-only)

(none yet)
