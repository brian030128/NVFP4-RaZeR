# TM-OPT vs MR-OPT on three models, seed variance, and the GEMM cost of E0M3-dense maps — protocol

Written 2026-09-26 on branch `tm-opt`, before any run of these items. The hash and registration
time are in `registration_items.json`. Deviations are appended in the last section, never edited in
place. The study is **descriptive**: nothing is selected on WikiText-2, C4 or zero-shot.

**Task (user-approved, relayed by nvfp4-razer-c9):** items #1, #2 and #3, run in the order
#2 → #1 → #3.
- **The Qwen3.8-27B gate stays closed.**
- **No legacy full run:** the user declined it.
- **MR-OPT is not rerun.** Its committed maps are the comparison (`results/mr_variants`).

## Common settings

- **TM-OPT:** `run_train_map.py --tm-opt` (no B1). STE, deterministic, with main's hyperparameters
  unchanged:
  - lr 0.02, init θ −1, Adam β (0.9, 0.999), ε 1e-12, constant lr;
  - batch 8, 20 epochs, monitor every 2 epochs, monitor batch 16;
  - native monitor and native final evaluation, single-pass epilogue;
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Data roots, as for each model's MR-OPT runs:**
  - Llama: `/home/dev/n16k64_campaign/cost_comparison/data` with `--transformers-deviation`;
  - Mistral and Phi-4: `/home/dev/n16k64_campaign/multimodel/data`.
- **Evaluation:** `run_multiround.py --evaluate-map`, convention (a): 2,048-token windows, one per
  forward, per-window activation scales. The native process is primary and the fake process
  secondary; 16x64 and 256x64 maps are expanded exactly to 8x64 tiles.
- **Per run:** E0M3 tiles; the development-KL curve (monitor); per-epoch time, selection time and
  setup time; peak GPU (allocated/reserved) and host memory.
- **Paired statistics:** per-window ΔNLL, mean ± 2 SE (ddof = 1), classified per corpus:
  - **significantly better** if mean + 2 SE < 0;
  - **significantly worse** if mean − 2 SE > 0;
  - **not significantly different** otherwise.
- **Tile overlap with MR-OPT:** shared tiles, tiles only in TM-OPT, tiles only in MR-OPT, and the
  Jaccard index |A∩B| / |A∪B|.

## #2 Seed variance (Llama-3.1-8B, 8x64), run first

- **Runs:** two new TM-OPT runs with `--seed 1` and `--seed 2`, otherwise identical to the verified
  group-2 run (seed 0, `results/tm_opt/runs/g2_tmopt`). The seed changes only the calibration batch
  order.
- **Evaluation:** one native process (FourOverSix, seeds 0/1/2, MR-OPT 8x64) and one fake process
  (FourOverSix, seeds 0/1/2).
- **Reported:**
  - per seed: PPL (native and fake), E0M3 tiles, the development-KL curve;
  - for each pair of seeds: paired ΔNLL and Jaccard overlap;
  - each seed vs MR-OPT 8x64: paired ΔNLL with its classification, and tile overlap.
- **Is the seed spread small relative to the TM-OPT − MR-OPT gap?** This is fixed now. Per corpus:
  - the **gap** is the mean over the three seeds of their mean ΔNLL vs MR-OPT;
  - the **spread** is the range (max − min) of those three means;
  - the spread is called **small** if it is below |gap| / 2 on both corpora. Whether every seed
    is significantly better than MR-OPT is also stated.
- **Is the local-vs-H200 tile-count difference within the seed spread?**
  - **At each monitored epoch (2, 4, …, 20):** the H200 run's E0M3 count is compared with the
    range of the three local seeds. It counts as within the seed spread at an epoch where it lies
    inside [min, max].
  - **Also reported:** the relative spread (max − min) / mean at each epoch, next to the relative
    local-vs-H200 difference.

## #1 TM-OPT vs MR-OPT on three models (8 new runs)

- **Runs:** Llama-3.1-8B 16x64 and 256x64 (Llama 8x64 is the group-2 seed-0 run, reused);
  Mistral-7B-v0.3 8x64, 16x64 and 256x64; Phi-4 8x64, 16x64 and 256x64.
- **Order:** model by model, each model followed by its evaluations.
- **Evaluation per model:**
  - one native process: FourOverSix, the three TM-OPT maps, the model's three committed MR-OPT
    maps;
  - one fake process: FourOverSix and the three TM-OPT maps.
  - **Repeat check:** the FourOverSix and MR-OPT window NLLs must repeat the committed evaluations
    bitwise; this is reported.
  - **Fake ΔNLL vs MR-OPT** uses the committed fake NLLs of the MR-OPT maps, where FourOverSix's
    fake NLLs repeat.
- **Reported per model × unit:**
  - TM-OPT vs MR-OPT: PPL, paired ΔNLL with its classification on each corpus, E0M3 tiles and tile
    overlap;
  - selection time, setup time, peak memory, and the development-KL curve;
  - both vs FourOverSix.

## #3 GEMM and latency cost of E0M3-dense maps (speed only)

- **Kernel:** the SM120 deployment kernel of `origin/SM120-kernel` (commit f91c109, `sm120/`).
  - **Export, without touching the working tree:** `git archive` into
    `/home/dev/n16k64_campaign/sm120_bench/`. CUTLASS is cloned there from the local copy at the
    pinned commit e64a913.
  - **Build:** `build.py` with CUDA 13.1 (the `mixfp4-cuda131` env) for `n16k64_wA` (16x64 maps,
    weights on A), `n8k64_wB` (8x64 maps, weights on B), and the stock NVFP4 baselines `stock_wA`
    and `stock_wB`. The build's own patch and census verification must pass.
  - **The build runs on the CPU** while #2 and #1 use the GPU.
  - **Before any benchmark, on the idle GPU after #1:** the kernel self-tests (`--selftest`) and the
    `sm120/tests` GEMM tests.
  - **Fallback:** if the build or those tests fail, the `repro_local/realquant` harness
    (b8x64 / wt_as_A) is used instead, and the report says so.
- **Maps, per unit** (8x64 on `n8k64_wB`, 16x64 on `n16k64_wA`):
  - all E2M1 (FourOverSix);
  - the committed MR-OPT map;
  - the TM-OPT map (seed 0 at 8x64; #1's run at 16x64);
  - all E0M3, as the extreme.

  The baseline is the stock NVFP4 kernel of the same operand placement. Our maps are converted to
  `MIXFP4MAP/1` files: the same tiles, with the model id and revision the SM120 exporter expects.
- **Models:** Llama-3.1-8B, then Phi-4 if time allows.
- **GEMM only** (packed operands, `sm120/bench` timing: CUPTI kernel time, median):
  - every distinct Linear shape, at T = 512, 2048 and 8192, optionally 1, 16 and 128;
  - per projection, the MR-OPT and TM-OPT patterns are those of the module with the most E0M3 tiles
    in that map (each map's worst case);
  - all E2M1 and all E0M3 on the same shapes.
- **Full-model prefill latency** (`sm120/bench/model.py` prefill: use_cache=True, median of 5 after
  2 warm-ups), at 1×512, 1×2048 and 4×2048:
  - stock NVFP4 kernel;
  - mixed kernel with FourOverSix (all E2M1);
  - MR-OPT;
  - TM-OPT;

  each per unit, one process per policy.
- **Reported:** GEMM time per shape and T; full-model prefill latency; overhead in % vs stock NVFP4;
  TM-OPT vs MR-OPT in %.
- **All benchmarks run on an idle GPU** (`require_idle`), after #1 and #2.
- **Numerics caveat, stated in the report:** the deployment kernel uses per-token activation scales
  and a one-rounding epilogue. That differs from this study's development and evaluation convention
  (per-document or per-window scales, two roundings). #3 measures speed only.

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> after #2,
  after #1 and after #3, and push each time (never main, no force). If a push is blocked, report it
  and do not retry.
- **Reporting:** report to the user after each item. After #3, stop and wait.
- **Scope:** no multi-model run beyond these items. The Qwen3.8-27B gate, Part C, QAT and zero-shot
  stay closed or paused.

## Deviations (append-only)

(none yet)
