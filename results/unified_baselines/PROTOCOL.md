# Unified comparison: TM-OPT+TC, QAT and scale-only — protocol

Written 2026-09-28 on branch `tm-opt`, before any run of this study. The hash and registration time are in
`registration.json`. Deviations are appended in the last section, never edited in place. The study is descriptive.
Nothing is selected or tuned on WikiText-2, C4 or zero-shot.

**Task** (user-approved, relayed by nvfp4-razer-c9). Compare ours (TM-OPT+TC, "OURS") with full-weight QAT (arm C) and
scale-only KL training (arm D, "SCALE") under the same conditions, on two models. Measure memory only on two larger
ones.

**Why.** The only QAT numbers so far (`results/cost_comparison`, arm C) come from the MR-OPT-era conditions: convention
(a) activations, a development set inside the calibration, and fake (a) evaluation. Task 2 (`results/scale_additivity`)
already put OURS and SCALE on Llama-3.1-8B under unified conditions; QAT and a second model are missing.

## Unified conditions (every arm)

- **Data and loss:** the model's 128 × 512-token math/code fit set (its calibration record), with KL(BF16 teacher ‖
  student) per sequence. The teacher is precomputed and held on the host.
- **Schedule:** optimizer batch 8 sequences (16 steps per epoch), 20 epochs (320 steps), deterministic
  (`torch.use_deterministic_algorithms`, seeded RNG).
- **Training activations:** per-token FourOverSix, convention (c). OURS uses `fourover6_rows` in its straight-through
  expression; the baselines use `run_cost_distill.py --act-rows`.
- **No development set in any deployed calibration** (Task 1's definition, `results/nodev_cost/PROTOCOL.md`).
  - **Cost:** the included phases (model load, fit data, fit teacher, preparation, training, writing the output).
  - **Memory:** peak GPU allocated and reserved; peak host RSS sampled every 0.1 s, and `ru_maxrss`.
- **Learning rate.**
  - OURS keeps its fixed TM-OPT+TC setting, lr 0.02.
  - **The baselines' rate is chosen on the development set** (192 math/code documents: Llama's three sets of 64,
    Mistral's `fresh_dev1-3`) by `choose_lr.py`:
    - the lowest final development KL, evaluated with per-token activations and the deployed weight;
    - ties go to the smaller rate; a non-finite KL disqualifies a run.
  - **SCALE:** grid {1e-4, 3e-4, 1e-3}; one extension (3e-5 below, 3e-3 above). This is Task 2's rule.
  - **QAT:** the cost study's registered arm-C grid {1e-6, 1e-5, 1e-4} (`results/cost_comparison/PROTOCOL.md` §4),
    and up to two extensions by the grid's own step (1e-7, then 1e-8 below; 1e-3, then 1e-2 above).
    - **Why the rule matters:** C1 (1 epoch) chose 1e-6, the lowest point, and C2 overfit at 1e-6 when time-matched
      on 128 sequences.
  - **When an extension runs:** only while the lowest KL of the rates run so far sits at an edge, in that edge's
    direction. If it still sits at an edge once that direction's extensions are used up, there is no choice, and
    the study stops and reports.
  - **Selection runs** use the development set: an initial and a final development evaluation, and the check that
    the fused training quantizer equals the deployed weight bitwise. They are not part of any calibration cost.
  - **The deployed run** is the `--no-dev` rerun at the chosen rate. It must repeat the chosen development run's
    320 per-step training KLs exactly; otherwise the study stops and reports.

## Arms

- **OURS:** TM-OPT+TC, the committed maps at 8x64 and 16x64 (`results/tm_opt`, lr 0.02, 20 epochs).
  - **Cost:** Task 1's no-dev reruns, whose maps are bitwise equal to the committed ones.
- **SCALE (arm D):** `run_cost_distill.py --arm scale --micro-batch 8 --budget c1 --epochs 20 --act-rows
  --deterministic`.
  - **Method:** a learned factor on every 16-element block's UE4M3 scale, with the weights frozen.
  - **Optimizer:** torch AdamW (fused, FP32): β = (0.9, 0.95), ε = 1e-8, weight decay 0, constant rate.
  - **Deployment:** plain NVFP4 E2M1.
- **QAT (arm C):** `run_cost_distill.py --arm qat --optimizer adamw_fp32 --micro-batch 8 --checkpointing --budget c1
  --epochs 20 --act-rows --deterministic`.
  - **What is trained:** the scoped Linear weights, in BF16. The update is torchao's AdamW with FP32 moments and BF16
    stochastic rounding, drawn from the seeded global RNG. The same β, ε, weight decay and constant rate as SCALE.
  - **Everything else is frozen:** `requires_grad` is False on every other parameter.
  - **Recorded per run:**
    - which parameters are trained (`trained_parameters`, which asserts that no other model parameter requires a
      gradient);
    - a sha256 over every parameter outside the scoped Linears, before and after training (phase `frozen_check`,
      outside the cost). It must be unchanged. So the deployment needs only the Linears, and the report says
      explicitly that nothing outside them is trained.

**Reused runs (Llama-3.1-8B),** checked by `verify_reuse.py` (`reuse.json`) before registration; all checks pass.
- **OURS:**
  - TM-OPT+TC with per-token training activations, 20 epochs, deterministic, batch 8 without accumulation, lr 0.02,
    16 steps per epoch;
  - Task 1's `--no-dev` reruns wrote bitwise-equal maps;
  - the same holds for Mistral's maps, so no OURS run is repeated.
- **SCALE:** Task 2's deployed run (`results/scale_additivity/runs/scale_nodev`):
  - arm scale, `--act-rows`, 20 epochs of c1 (320 steps), deterministic, batch and micro-batch 8, `--no-dev`;
  - lr 3e-4, chosen by Task 2's rule, which is this study's SCALE rule;
  - its per-step KL equals the chosen development run's;
  - its artifact passed Task 2's exporter and ownership checks.

## Deployment and its checks

- **OURS:** the Parts 2–3 artifacts.
- **SCALE:** `export_map_artifact.py --kind four_over_six --scales state.pt --ownership`, as in Task 2.
- **QAT, new:** `export_map_artifact.py --kind four_over_six --weights state.pt --ownership`. It runs as `auto_stock`.
  1. The model's own weights must equal the calibration record; then the trained weights replace them.
  2. The packed weights must equal the FourOverSix fake quant (`quant_nvfp4_4over6`, and sm120's candidate) of the
     trained BF16 weights bitwise. This is the exporter's own check.
  3. The ownership check must be exact, with 0 format mismatches.
  4. **NativeLinear (c) against fake (c) of the trained weights,** on every released window (`run_ppl_deploy.py`,
     policy `fake:weights:<state.pt>`).
     - Criterion: B2's (i), not significant per corpus, |mean ΔNLL| ≤ 2 SE (`results/deploy_eval/PROTOCOL.md`).
     - The maximum |ΔNLL| and the ΔPPL are reported too.
  - Only the scoped Linears are exported. The evaluation loads the model's own other parameters, which QAT leaves
    unchanged (checked above).

## Evaluation

- **Harness:** `run_ppl_deploy.py`, NativeLinear (c), on the released WikiText-2 and C4 windows, one process per model.
- **Policies:** BF16, NVFP4, FourOverSix, OURS 8x64 and 16x64, SCALE and QAT, plus QAT fake (c) for check 4.
- **Regression check:** BF16, NVFP4, FourOverSix and both OURS units must equal the Parts 2–3 evaluation window for
  window. For Llama, SCALE must equal Task 2's.
- **Comparisons:** paired ΔNLL ± 2 SE per window; "better / n.s. / worse" by the sign of mean ± 2 SE:
  - every arm − FourOverSix (NVFP4 too, for scale);
  - OURS (each unit) − SCALE;
  - OURS − QAT;
  - SCALE − QAT.

## Phases

1. **Llama-3.1-8B** (`phase1_llama.sh`): QAT learning-rate selection, deployed QAT with the repeat check, QAT artifact,
   then the evaluation. A short progress message to nvfp4-razer-c9 follows, then the study continues.
2. **Mistral-7B-v0.3** (`phase2_mistral.sh`): SCALE and then QAT, each with its selection, deployed run, repeat check
   and artifact; then the evaluation.
3. **Phi-4 and Qwen3.8-27B: memory probes only** (`phase3_probes.sh`).
   - **The probe:** `run_cost_distill.py --budget probe --probe-steps 3`, three optimizer steps of 8 sequences at the
     unified settings. The development data is loaded but not evaluated; nothing is saved or deployed.
   - **SCALE:** micro-batch 8 without checkpointing, as arm D. If that does not fit: 4, 2, 1 (with accumulation;
     the optimizer batch stays 8), then 1 with checkpointing.
   - **QAT:** FP32-state AdamW, micro-batch 8, checkpointing. It is expected not to fit.
   - **QAT fallbacks, measured only:** micro-batch 1; `adamw_8bit`; `cpu_offload` (FP32 states on the host); LoRA
     (rank 16), each at micro-batch 8, else 1.
   - **Recorded:** status (complete, or out of memory with the peaks at the failure), peak GPU allocated and reserved,
     and the step time.
   - **No fallback is trained.** Changing the optimizer or the method is the user's decision.
   - **Failures:** an out-of-memory stop is a result. Any other failure is logged and reported.

## Stop rules

- **A registered check fails:** stop and report; nothing is improvised. The checks are:
  - the repeat of the per-step KL;
  - the frozen check;
  - the exporter and ownership checks;
  - check 4;
  - the regression check.
- **A baseline's rate is still at an edge after its extensions:** stop and report.
- **A run fails:** the queue stops (phases 1–2).
- **Automatic checks:** the queue runs `check_eval.py` (the regression check and check 4) right after each model's
  evaluation, and stops on a failure.

## Report

- **`REPORT.md`, with Task 2's structure:**
  - summary;
  - PPL table;
  - paired comparisons;
  - cost table (calibration and memory);
  - learning rates;
  - checks;
  - the phase-3 probes;
  - deviations.
- **Tables:** `unified.{md,json}` from `analyze.py`.
- **Code, the default paths unchanged:**
  - `run_cost_distill.py --model` and the frozen check;
  - `export_map_artifact.py --weights`;
  - `run_ppl_deploy.py`'s `fake:weights:` policy;
  - `choose_lr.py`, `verify_reuse.py`, `analyze.py`.
- **Out of scope:** downstream (lm-eval) accuracy, OURS on QAT weights, and more-data QAT (C3).

## Rules

- Commit and push on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> at the end (never main, no force). If a
  push is blocked, report it and do not retry.
- Report back with a compact table per model, then stop.

## Deviations (append-only)

1. **2026-09-28, after phase 1's results: `analyze.py` label fix, reporting code only.** No run, check, rule or
   comparison changed. Registered sha256 c81a6521ed43…; after this fix 265e3a5f2cf7….
   - **The problem:** Task 2's SCALE run predates the `trained_parameters` record. The cost table's fallback text
     for a missing record said "tile logits" for every arm.
   - **The fix:** it now names what each arm trains: tile logits for OURS, block-scale factors for SCALE.
   - All other code and the queues have their registered hashes.

2. **2026-09-28, after all results: a second `analyze.py` label fix, reporting code only.**
   - **The problem:** the artifact table called every new artifact's packing check "bitwise". For the SCALE
     artifact, the exporter's learned-scale check is value-equal: −0 packs as +0, as in Task 2.
   - **The fix:** the table now says so. The final `analyze.py` sha256 is da87c2a41859…; nothing else changed.

Note, not a deviation: a first registration at 06:06:48Z was replaced at 06:07:14Z, before any run. The replacement
added the automatic check step (`check_eval.py`).
