# K-tile dispatch ablation (kernel-opt amendment 20; the paper's Figure 2(a))

Registered before any timing (registration time and hashes: `registration.json`). The request (the user, through the
coordinator, 2026-10-09; the slim scope the user chose): show readers on this GPU (RTX PRO 6000 Blackwell Workstation
Edition) what one format dispatch per K-tile buys over choosing the format per MMA.

## Kernels (M = N = K = 4096 only)

- **stock_ko** (the reference): `build_V`'s `stock_ko` set at this shape, i.e. the adopted table's width (`stock_wA_e64`)
  and scheduler row; NVFP4 weights on A.
- **FP8 reference:** cuBLAS through `torch._scaled_mm`: e4m3 activations [T, K] and weights (W^T), per-tensor FP32
  scales, bf16 output -- the standard PyTorch path on this GPU. On this machine it dispatches cuBLAS's
  `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize256x64x64_stage4_warpsize4x1x1_tensor16x8x32_execute_kernel__5x_cublas`
  (an sm89 XMMA kernel; recorded, not tuned).
- **Per-MMA branch (new, `build_KT`), one per unit**, each with its deployed counterpart's CTA tile, warp arrangement,
  placement, epilogue tile and k_tile body, run with the counterpart's scheduler setting:
  - 16x64: `n16k64_wA_e64_t0_permma` (weights on A, like `mixed_ko`'s width-128 build);
  - 8x64: `n8k64_wB_t0_permma` (weights on B, like `mixed_wB_ko`'s width-128 build).
  - The format is chosen by an ordinary C++ if/else before every MMA (the historical kernel of
    `sm120/kernel/docs/mixed_nvfp4_report.md` section 1), with the compiler's default codegen: the blob generator's
    `BRANCH=a|b` emits `mixfp4::mma_kblock_branch` (the blob's MMA order and per-MMA PTX, each MMA behind its own
    if/else on its granule's flag), and `-DMIXFP4_PER_MMA_BRANCH=1` removes the per-K-tile dispatch from the mainloop.
    ptxas if-converts each if/else into a predicated OMMA pair (`@P` E0M3 / `@!P` E2M1); that is what is measured.
  - Built with their counterparts' `--define` sets (`MIXFP4_DISPATCH_FREQ=1` + `UNIFORM_DISPATCH=1` for 16x64,
    `DISPATCH_FREQ=1` + `PIPE_FLAGS=1` for 8x64), which are inert here (`build_KT_nodef`: identical SASS without them).
  - `build.py` accepts their predicated OMMAs (`allow_predicated`, only these two configurations), and the patcher reads
    their if-converted pairs with `--predicate-aware` (only these two).
- **Per-K-tile:** `build_V`'s `mixed_ko` (16x64, `n16k64_wA_e64_t0`) and `mixed_wB_ko` (8x64, `n8k64_wB_t0`) at their
  deployed 4096³ configurations (table width and scheduler row).
- `build_V`, the deployed tables and every adopted kernel are untouched.

## Gates and checks (CPU and short GPU checks done before this registration; no timing)

- **Inactive unless asked for:** `build_KT_ref` -- `n16k64_wA_e64_t0` (FREQ + UNIFORM), `n8k64_wB_t0` (FREQ + PIPE),
  `stock_wA_e64`, rebuilt from the modified sources -- has `build_V`'s patched and unpatched SASS; the generator's
  default output equals `build_V`'s headers byte for byte; `LOCAL_CHANGES.patch` regenerated from the upstream files
  (their sha256 equal `VENDORED.json`).
- **Correctness (`check.json`):** each per-MMA build's output equals its `build_V` per-K-tile kernel's bitwise on the same
  operands at 4096³, with the real, all-E2M1, all-E0M3 and random 30 % tags. Before timing, the timed operands are
  checked again.
- **Static SASS census (`census.json`, C2 style; static counts, not measured counters):** OMMAs, predicated OMMAs, BRX,
  WARPSYNC, OMMAs per path through the steady k-iteration (the innermost OMMA loop), and the tensor-pipe estimate at
  4096³ = OMMAs per iteration x 32 k-tiles x 8 MMA warps x 1024 output tiles (an issued OMMA takes a tensor-pipe slot
  whether or not its predicate is true).

## Timing (after GSM8K M1 is paused: no other compute process on the GPU)

- **Operands:** C2U's -- weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) quantized FourOverSix per token.
- **Tags:** `real` = `model.layers.0.self_attn.o_proj` of the unit's Llama-3.1-8B TC map (C2U's; 4.44 % / 3.34 %
  E0M3 tiles), and `e2m1` (every tile E2M1) as a second row, on the same binaries.
- **Configurations (10):** stock_ko, fp8, per-MMA and per-K-tile x {16x64, 8x64} x {real, e2m1}.
- **Modes (the C2U / M1 method):**
  - isolated (M1): per call a 512 MiB read-flush and the next weight copy of a rotation over >= 4x the L2; 3 warm-up +
    30 timed calls per configuration per round;
  - b2b: 20 back-to-back calls per block over the same rotating weight copies.
  - CUPTI kernel times (torch.profiler); 3 rounds, round r starting the configuration list at position r x 10 / 3; the
    value of a round is the median of its calls, the reported value the median of the 3 rounds (the pooled median and
    every call are recorded too).
  - No clock locking and no ncu (neither is available), as always; NVML telemetry per block and the board's power
    limit are recorded.
- **Reported (`REPORT.md`, `ktile_ablation.json`, `numbers.tex`, `tflops.png`):** per kernel µs, TFLOP/s
  (2 x 4096³ / time), the overhead against stock_ko, and the per-round values, both modes; the census; a TFLOP/s bar
  chart with one group per unit (stock, FP8, per-MMA branch, per-K-tile; real tags, isolated); LaTeX-ready numbers.
- **Stop and report on:** a failing bitwise check, another compute process during timing, a profiled block with the
  wrong number of kernels.

## Afterwards

GSM8K M1 (paper-eval part M) resumes automatically once the timing is done, under its pause amendment's rule.
