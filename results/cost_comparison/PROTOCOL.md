# Calibration cost vs. QAT (BF16-distillation) cost — protocol

Registered 2026-09-23 before any arm was run. Nothing below may change after results are
seen; unavoidable deviations are appended in the last section with their reason.

Question: how does the cost of our KL-only multi-round MixFP4 election
(`run_multiround.py --objective kl`) compare with QAT that distills from the BF16 model,
and with scale-only distillation, on Llama-3.1-8B, W4A4 with FourOverSix activations,
all on the same single GPU?

## 1. Machine, software, data

- 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), driver 595.71.05,
  241 GB host RAM, 128 CPUs, no swap. No Slurm: `job_id` is null and each report records
  the hostname, GPU and versions instead.
- The environment of the earlier runs on this machine: conda env `n16k64` (Python 3.11.11,
  torch 2.9.0+cu128, transformers 5.16.1, datasets 4.8.5, triton 3.5.0, psutil 7.2.2).
  torchao 0.14.1 (optimizers for arm C only) is installed with `pip --no-deps --target`
  into `/home/dev/n16k64_campaign/cost_comparison/pydeps`, outside the env, and is on
  `PYTHONPATH` only for `run_cost_distill.py`.
- Model: meta-llama/Llama-3.1-8B @ d04e592bb4f6aa9cfee91e2e20afa771667e1d4b, BF16, SDPA;
  every one of the 224 text Linear matrices is checked against its recorded source sha256.
- Data root `/home/dev/n16k64_campaign/cost_comparison/data`, built by
  `prepare_multiround_data.py` (Step 1). Its hashes all match the cluster inputs:
  - the three development `fresh.pt` files (192 documents) are bitwise identical to the
    cluster files: sha256 2d2c4940…, 5c5a1bb2…, eb913e15… (checked by `run_multiround.py`);
  - the fit set: all 128 token sha256 match (`math_code_data`);
  - WikiText-2 / C4 evaluation windows: all 141 + 256 token sha256 match the published
    report (`validate_evaluation_data`);
  - weights: all 224 matrix sha256 match.
  The calibration report itself (sha256 47e0d54c…) is not available. Its fields that the
  scripts read (`revision`, `transformers_version`, `fit`, `matrices`) are taken from the
  historical calibration report `results/math_code_adaptive/calibration_333779_llama8b`.
  That this is the same fit set is shown by the 64 fit/election records of
  `results/task_reorder/llama_diagnosis_20260920/data.json`, which cite 47e0d54c…. Only
  `source` (the local snapshot path) differs.
- Hugging Face Hub access is on (`HF_HUB_OFFLINE=0`). As on the cluster, the fit files are
  read by streaming.

## 2. Arms

All arms use W4A4: weights as listed, and per-document tensor-wide FourOverSix
activations on every quantized Linear input.

**A — FourOverSix RTN (reference).** Every weight gets `quant_nvfp4_4over6`. It is run with
`run_multiround.py --max-rounds 0`, which applies no election and gives the final
evaluation of the FourOverSix weights plus the development KL/CE. Target:
6.875525 / 9.823733 (published).
Calibration cost is zero. The run's teacher and development phases exist only to measure
the development metrics.

**B — ours: KL-only multi-round election** (`run_multiround.py --objective kl`,
selection logic unchanged).

| run | unit | eval batch | score batch | compared with (main) |
|---|---|---:|---:|---|
| B-256-opt | 256x64 | 16 | 8 | optimized run 2: 8,393 tiles, 6.835411 / 9.771621 (report not in repo; numbers from MIXFP4_REPORT.md) |
| B-256-ref | 256x64 | 1 | 1 | run 1: 8,405 tiles, 6.841998 / 9.774137, map sha256 e3655a04… (report in repo) |
| B-8x64-opt | 8x64 | 16 | 8 | 3,654 tiles, 6.819751 / 9.750492, map sha256 f968efb9… (a reference-settings run) |

- All B runs use `--budget-hours 12` instead of the default 3. On H200 the 3-hour cap
  never bound (every published run stopped on "no step lowers"). Here it would only
  truncate a slower search.
- Maps are compared with main by `map_sha256`, which is bitwise identity of `map.pt`.
- The published `map.pt` files are not in the repository, so tile overlap with main
  cannot be computed. Instead we report tile counts, per-round trajectories, and paired
  per-window PPL differences against the published per-window NLLs. We also report tile
  overlap between the local maps.

**B' — B-256-opt with `--skip-ce-backward`.** Scoring skips the CE backward. The KL scores
do not depend on it, so the map must be bitwise identical to B-256-opt's. Reported
separately. `predicted_ce` is then 0 in the log.

**C — full-weight QAT with KL distillation** (`run_cost_distill.py --arm qat`).
- Trainable parameters: all 224 text Linear weights (6,979,321,856 parameters), BF16.
  Embeddings, norms and `lm_head` are frozen.
- Forward pass: every Linear computes `F.linear(Q_act(x), Q_w(W))`, with straight-through
  (identity) gradients through both quantizers.
  - `Q_w` = FourOverSix of the current weight, i.e. `quant_nvfp4_4over6`.
  - `Q_act` = per-document tensor-wide FourOverSix, i.e. `quant_per_document`.
  - Both run as a fused Triton kernel, `quantize/fused_fourover6.py`. It is verified bitwise
    against the reference quantizers on all 224 source weights at the start of every run,
    on the first 64 activation calls, and on all 224 final weights.
  - Deployed and evaluated weights always come from `quant_nvfp4_4over6` itself.
- Loss: per sequence, KL(teacher ‖ student) = `(t.exp() * (t - lp)).sum(-1).mean(-1)`
  (`run_multiround.py`'s direction and token reduction), averaged over the 8 sequences of
  an optimizer step.
- Teacher: BF16 log-probs of the unquantized model, precomputed with batch 1 exactly as
  `run_multiround.py` does and held on the CPU in BF16.
- Optimizer: AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0, constant learning rate,
  no warmup, no clipping. torchao update code: FP32 math, BF16 stochastic rounding of
  the updated BF16 weights, so that small updates are not lost.
- Batch: 8 sequences × 512 tokens per optimizer step. Micro-batch and checkpointing are
  set in §3; gradients are accumulated.
- Order: per-epoch permutations from `torch.Generator().manual_seed(0)`. C1 is the prefix
  of C2's schedule.
- Budgets:
  - **C1, token-matched:** the 128 fit sequences, 1 epoch (16 steps).
  - **C2, time-matched:** repeated epochs of the 128 fit sequences. No new optimizer step
    starts once the time since process start reaches T_B = B-256-opt's local
    `setup_seconds + optimization_seconds`, i.e. its wall clock before the final
    evaluation. B's setup covers model load, teachers and its initial development
    evaluation; C's covers model load, teachers, quantizer verification and its initial
    development evaluation.
  - **C3, larger budgets: 10× = 1,280 and 100× = 12,800 new sequences, 1 epoch each.**
    - Source: `prepare_distill_pool.py --per-source 6400` streams the fit set's own two
      files (OpenWebMath `data/train-00000-of-00114-5a023365406cb9c4.parquet` @ fde8ef8d,
      CodeParrot `file-000000000001.json.gz` @ 35a59fb0) in file order.
    - It excludes the 128 fit and 192 development documents (document sha256) and all
      known windows (token sha256), and skips documents shorter than 512 tokens.
    - It cuts one 512-token window per document at an offset drawn from
      `torch.Generator().manual_seed(20260923)`, the development-set convention.
    - Each source is split equally. 10× uses the first 640 per source, 100× all 6,400.
    - The manifest records every document/token hash and the files' sha256.
    - If a file has fewer than 6,400 eligible documents, 100× becomes the largest equal
      split available, and this is stated.
    - The teacher is precomputed chunk by chunk (256 sequences) just before those steps and
      held on the CPU. The model's latent weights are parked on the CPU while the source
      weights produce the teacher.

**D — scale-only KL distillation** (`run_cost_distill.py --arm scale`).
- Weights frozen. Every 16-element block has a learned FP32 factor f, initialised to 1.
- Block scale = e4m3(clamp(f · s0, 2⁻⁹, 448)), where s0 is the FourOverSix pre-rounding
  scale (block max / 6 or / 4, the choice `quant_nvfp4_4over6` makes). At f = 1 the
  output is bitwise `quant_nvfp4_4over6` (tested).
- Elements are rounded to E2M1 with the current scale.
- Gradients: straight-through across the E4M3 rounding, zero where the clamp is active,
  and LSQ across the element rounding: d(s·R(x/s))/ds = R(u) − u inside the grid, R(u)
  when saturated.
- Parameters: 436,207,616 factors. Optimizer: torch AdamW (fused, FP32), with the same
  β, ε, weight decay 0 and constant LR as C.
- Same activation quantizer, loss, teacher, batch and order as C. Budgets: **D1** = C1,
  **D2** = C2 (the same T_B).
- Deployment is plain NVFP4 (E2M1 codes, E4M3 scales, FP32 tensor scale). No E0M3.

## 3. C memory configuration search (probes)

A probe is `run_cost_distill.py --budget probe --probe-steps 3`: the first 3 optimizer steps
of the C1 schedule, with the arm's full training code path. It fits if it completes 3 steps
without a CUDA OOM. Its measured per-phase memory and steady step time (mean of steps 2–3)
are reported.

1. **Plain configuration (the expected OOM):** FP32 AdamW states (`adamw_fp32`: BF16
   weights, BF16 gradients, FP32 m and v), micro-batch 8, no checkpointing. It is reported
   first, with its peak memory, whether or not it fits.
2. **Value-preserving variants of the same optimizer:**
   - micro-batch 4, 2, 1 without checkpointing;
   - micro-batch 8, 4, 2, 1 with per-decoder-layer activation checkpointing (HF,
     `use_reentrant=False`).

   These change neither the objective nor the update (gradient accumulation only
   re-associates the sum), so they come before any fallback that changes the optimizer
   or the parameterization. The C configuration is the fitting setting from 1–2 with the
   lowest steady step time.
3. **Fallbacks only if no FP32-state setting fits, in the user-specified order**, each with
   the same micro-batch/checkpointing search:
   - 8-bit AdamW: `adamw_8bit`, torchao AdamW8bit, block 256, BF16 stochastic rounding;
   - optimizer offload: `cpu_offload`, torchao CPUOffloadOptimizer; FP32 m, v and the
     update on the CPU, gradients copied to pinned host memory, updated BF16 weights
     copied back;
   - LoRA-QAT: `--arm lora`, rank 16, α = 16, adapters on all 224 matrices (q, k, v, o,
     gate, up, down in all 32 layers), A Kaiming-uniform, B zero; the deployed weight is
     Q(W + B·A); FP32 adapters with torch AdamW.

   The first optimizer with a fitting setting is used, at its fastest fitting setting.
4. **Reference probes (always run, not trained):** `adamw_8bit`, `cpu_offload` and `lora`,
   each at micro-batch 8 without checkpointing and at the chosen C setting. This reports
   every fallback's memory and time even when none is needed.

D uses the same search (settings 1–2 with its own optimizer), picking the fastest fitting
setting.

**Live teacher.** The chosen C and D settings are probed once more with `--live-teacher`: a
second BF16 model on the GPU computes the teacher per micro-batch, with no CPU teacher store.
If that probe OOMs, the live-teacher memory is given as computed: measured peak + 14.96 GiB
of BF16 weights. For B it is computed.

## 4. Hyperparameters and selection

- Learning-rate grids, at most 3 values, fixed now:
  - C (any optimizer): {1e-6, 1e-5, 1e-4};
  - LoRA, if reached: {1e-5, 1e-4, 1e-3};
  - D: {1e-4, 1e-3, 1e-2}.
- Every (arm, budget) trains all grid points. Each run ends with one development
  evaluation: the 192 documents, eval batch 16, W4A4 with the deployed weights, exactly
  `run_multiround.py`'s `dev_eval`.
- Selection: the lowest mean development KL, which is ours' acceptance objective. Ties go
  to the smaller LR. A non-finite loss disqualifies a run.
- WikiText-2 / C4 are evaluated only for the selected run (`--evaluate <run dir>`, which also
  re-checks the development numbers bitwise), and never for the others.
- The cost table counts the whole grid and, separately, the chosen run.

## 5. Evaluation (identical for all arms)

- `run_multiround.py`'s final evaluation: `data(tok, prior, 2048)`, `validate_evaluation_data`
  against `results/kse_paper/job_336566/llama8b/report.json`, per-document FourOverSix
  activation hooks, one 2,048-token window per forward, and WikiText with `use_cache=True`.
- `run_cost_distill.py` contains a verbatim copy of that code. **Equivalence check, before any
  C/D run:** `run_cost_distill.py --arm qat --budget zero` must reproduce arm A's per-document
  development CE/KL and all 397 per-window NLLs bitwise. If it does not, C/D do not start
  until the difference is fixed; the fix is recorded as a deviation.
- Reported for every arm:
  - PPL;
  - paired ΔNLL per window versus arm A (local), mean ± 2 SE with SE = sd(ddof = 1)/√n
    (this reproduces MIXFP4_REPORT's −0.00815 ± 0.00173 / −0.00748 ± 0.00240 for
    8x64);
  - development KL and CE before and after.
- "Beats" means mean + 2 SE < 0 for the paired per-window difference; "worse" means
  mean − 2 SE > 0; otherwise the result is inconclusive.
- The conclusion compares each C/D arm with B-256-opt and B-8x64-opt on both datasets, and
  on cost.

**Caveat, stated in the report:** the 192 development documents are used by every arm for
acceptance or selection, so they are not held out. Development numbers are not
generalization.

## 6. Cost and memory measurements

- **Phases, all arms:** `cost_monitor.PhaseMonitor`.
  - Phases are contiguous: model load, data load, teacher precompute, candidate packing or
    quantizer verification, initial development evaluation, scoring and development
    evaluation (B) or training and teacher chunks (C/D), final development evaluation,
    and final evaluation.
  - At each phase start: `torch.cuda.reset_peak_memory_stats`. At its end:
    `max_memory_allocated` / `max_memory_reserved`.
  - Host RSS: psutil, sampled every 0.1 s from a background thread and at every boundary.
  - `ru_maxrss` is reported as in commit 2b10c66.
  - The run-wide GPU peaks in `run_multiround.py`'s resource block are the maximum over
    phases, which is the same quantity as without resets.
- **GPU-hours:** wall clock on the single GPU.
  - B: `setup_seconds + optimization_seconds`. Its "selection time" (`optimization_seconds`)
    is also given, as in §4 of MIXFP4_REPORT.
  - C/D: process start to the end of training plus the final development evaluation.
  - The WikiText/C4 evaluation is identical for all arms and is listed separately.
- **Tokens and passes:** sequence counts (× 512 tokens) of
  - student or scoring forward and backward passes (B-256-opt runs a CE and a KL backward
    per scoring pass; B' only KL);
  - checkpoint recompute forwards;
  - BF16 teacher forwards;
  - development forwards.
- **Memory breakdown:**
  - Computed exactly from tensor sizes: weights, gradients, optimizer state (per device),
    teacher log-probs on the CPU, parked weight copies.
  - Activations and temporaries: measured peak minus the computed persistent tensors,
    labelled "measured residual".
  - Every number is labelled measured or computed.

## 7. Order of work

1. Build the pool.
2. Run A.
3. Run the equivalence check.
4. Run B-256-opt, which gives T_B.
5. Run B'.
6. Run the C and D probes, including live teacher and reference probes.
7. Run the grids: C1, D1, C2, D2, C3-10×, C3-100×.
8. Run the selected-run evaluations.
9. Run B-256-ref and B-8x64-opt.
10. Write REPORT.md.
11. Commit on `repro/n16k64-rtx-pro-6000` (no push). `inference/` is not touched.

## 8. Commands

```
PY=/home/dev/.conda/envs/n16k64/bin/python
DATA=/home/dev/n16k64_campaign/cost_comparison/data
RUNS=/home/dev/n16k64_campaign/cost_comparison/runs
export HF_HUB_OFFLINE=0
# run_multiround.py: PYTHONPATH=$PWD ; run_cost_distill.py: PYTHONPATH=$PWD:/home/dev/n16k64_campaign/cost_comparison/pydeps
$PY prepare_multiround_data.py --out $DATA
$PY prepare_distill_pool.py --data-root $DATA --per-source 6400 --out $DATA/pool/pool.pt
$PY run_multiround.py --unit 256x64 --objective kl --max-rounds 0 --eval-batch 16 --score-batch 8 --budget-hours 12 \
    --data-root $DATA --transformers-deviation --out $RUNS/A_fourover6
$PY run_multiround.py --unit 256x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 \
    --data-root $DATA --transformers-deviation --out $RUNS/B_256x64_opt            # B'
                                                                                   # adds --skip-ce-backward
$PY run_multiround.py --unit 256x64 --objective kl --budget-hours 12 --data-root $DATA --transformers-deviation --out $RUNS/B_256x64_ref
$PY run_multiround.py --unit 8x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 \
    --data-root $DATA --transformers-deviation --out $RUNS/B_8x64_opt
$PY run_cost_distill.py --arm qat --budget zero --data-root $DATA --transformers-deviation --out $RUNS/eval_equivalence
$PY run_cost_distill.py --arm qat --optimizer adamw_fp32 --lr 1e-5 --budget probe --probe-steps 3 --micro-batch M [--checkpointing] ...
$PY run_cost_distill.py --arm qat --optimizer <chosen> --lr <lr> --budget c1|c2 [--time-budget T_B] --micro-batch M [--checkpointing] --save-state ...
$PY run_cost_distill.py ... --budget pool --pool $DATA/pool/pool.pt --pool-per-source 640|6400 ...
$PY run_cost_distill.py --arm scale --lr <lr> --budget c1|c2 ...
$PY run_cost_distill.py --arm <arm> --budget zero --evaluate $RUNS/<selected run> --data-root $DATA --transformers-deviation --out $RUNS/<selected run>_eval
```

## 9. Deviations known at registration

1. transformers 5.16.1 instead of the calibration's 4.57.3. This is the environment of all
   earlier local runs. `--transformers-deviation` records both versions in each report.
2. The calibration report is reconstructed (§1). The fields the scripts use are verified
   through the fit token hashes and weight hashes; the report's own sha256 cannot match.
3. Software smoke tests were run before registration: a 2-step scale probe and a 3-step
   8-bit QAT probe at micro-batch 1, without development or test evaluation. They produced
   no accuracy numbers and are not part of the results.

## 10. Deviations recorded after registration (append-only)

4. **2026-09-23 22:20 UTC: B' is not bitwise identical to B.** §2 expected B' to reproduce
   B-256-opt's map bitwise. It does not:
   - B-256-opt: 7,625 tiles. B': 4,694 tiles.
   - Round 0 already differs: 15,137 vs 15,133 candidates. That flips the acceptance of the
     second backtracking step, and the searches diverge from there.
   - Cause, from the added diagnostic `results/cost_comparison/determinism_check.py`
     (`runs/determinism_check.json`):
     - on this GPU, SDPA's backward is PyTorch's FlashAttention-2 kernel
       (`flash_bwd_dq_dk_dv_loop_seqk_parallel`, dQ accumulated with atomics);
     - the KL weight gradients of one scoring batch are bitwise identical for only 6 of
       224 modules between two identical runs, and likewise with and without the CE
       backward.
   - The scoring pass is therefore not run-to-run deterministic here. On H200 the published
     re-run was bitwise identical. B and B' are two samples of the same procedure; both are
     reported, and no arm or rule changes.
   - A second diagnostic with `torch.use_deterministic_algorithms(True)` is added to test
     whether CE-first and KL-only KL gradients agree bitwise once the kernels are
     deterministic.
5. **2026-09-23 22:47 UTC: result of the deterministic-kernel check.** With
   `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   FlashAttention-2 runs its deterministic backward, and all 224 KL weight gradients are
   bitwise identical: across repeats, and between CE-then-KL and KL-only
   (`runs/determinism_check_deterministic.json`). Skipping the CE backward therefore
   leaves the KL scores unchanged. The B/B' map difference comes only from the default
   non-deterministic attention backward. The registered B and B' runs keep the default
   kernels, as on H200.
6. **2026-09-23 22:45 UTC: batch sensitivity observed (not a deviation).**
   - The step-1 KL of the probes depends on the micro-batch size.
   - `results/cost_comparison/batch_check.py` (`runs/batch_check.json`) shows the cause:
     BF16 log-probs at batch 1 and batch 8 differ (per-sequence KL(b1 ‖ b8) 3–8e-4;
     layer-0 attention outputs already differ) because different kernels run for
     different shapes.
   - W4A4 amplifies this through rounding flips: per-sequence KL vs the teacher changes by
     up to about 11% between batch sizes.
   - Evaluation is unaffected: the final WikiText/C4 evaluation is unbatched and identical
     for all arms. Development evaluations use eval batch 16 in every arm except
     B-256-ref.
7. **2026-09-24 08:41 UTC: study paused by user decision.**
   - The QAT-vs-ours comparison is paused so that the native-kernel shadow verification of the
     development evaluation can run.
   - `queue3.sh` was stopped first, then the running B-8x64-opt process (after 2 of its rounds;
     3,025 E0M3 tiles). B-8x64-opt is incomplete and is not a result. Its partial output is kept
     at `/home/dev/n16k64_campaign/cost_comparison/runs/B_8x64_opt/`.
   - No further cost-comparison run starts until the study is resumed.
   - Completed arms are listed in `STATUS.md`. REPORT.md is not final.
