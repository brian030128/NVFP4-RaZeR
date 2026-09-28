# Unified comparison: TM-OPT+TC, QAT and scale-only — report

Protocol: `PROTOCOL.md`, registered 2026-09-28T06:07:14Z (sha256 b1f2d427…, `registration.json`), before any run of
this study.
- **Registration:** a first registration at 06:06:48Z was replaced before any run, to add the automatic check step.
- **Smoke test:** run before registration; not a result.
- **Two deviations:** label fixes in `analyze.py`, reporting code only (`PROTOCOL.md`, Deviations). Every run's code
  and the queues have their registered hashes.
- **The study is descriptive.** The baselines' learning rates were chosen on the development set. Nothing was
  selected on WikiText-2, C4 or zero-shot.

## Summary

**Unified conditions, every arm:**
- the model's 128 × 512 fit set, KL to the BF16 teacher;
- optimizer batch 8 (16 steps per epoch), 20 epochs, deterministic;
- per-token (c) training activations, and no development set in the deployed calibration;
- evaluation through NativeLinear (c) on the released WikiText-2 / C4 windows.

**The arms:**
- **OURS:** TM-OPT+TC at its fixed lr 0.02.
- **SCALE:** learned UE4M3 block scales; lr from {1e-4, 3e-4, 1e-3}.
- **QAT:** full-weight, BF16 weights with FP32-state AdamW; lr from the registered arm-C grid {1e-6, 1e-5, 1e-4}, with
  up to two ×10 extensions.

**1. QAT needs a learning rate far below the registered grid.**
- On both models the dev-KL choice is 1e-7, reached after both downward extensions.
- It is interior (1e-8 is worse), so the rule chose it.
- 1e-6 is worse, and 1e-5 and 1e-4 diverge (dev KL 0.33 / 1.66 on Llama, 0.50 / 4.54 on Mistral).
- At 1e-7 a BF16 weight moves by about one stochastic-rounding ulp over the run, so QAT here mostly re-decides the
  rounding of weights near a boundary. This is an interpretation, not a measured quantity.

**2. Llama-3.1-8B: OURS beats QAT clearly; SCALE is close to OURS.**
- **Gains over FourOverSix (ΔNLL, wiki / c4):**

  | arm | wiki | c4 |
  |---|---:|---:|
  | OURS | 0.0121–0.0129 | 0.0147–0.0152 |
  | SCALE | 0.0117 | 0.0134 |
  | QAT | 0.0076 | 0.0092 |

  QAT keeps about 60 % of OURS's gain, SCALE about 90 %.
- **OURS − QAT:** −0.0046 to −0.0060, significant on both corpora at both units. SCALE − QAT: −0.0041 / −0.0042,
  significant.
- **OURS − SCALE:** n.s. on WikiText-2; OURS significantly better on C4 (−0.0013 / −0.0018), as in Task 2.

**3. Mistral-7B-v0.3: all three arms are close.**
- **Gains over FourOverSix:** OURS 0.0059–0.0065 / 0.0037–0.0051, SCALE 0.0059 / 0.0043, QAT 0.0050 / 0.0044. All are
  significant.
- **OURS − QAT:** better on WikiText-2 at both units (−0.0015 ± 0.0009 at 8x64, −0.0009 ± 0.0008 at 16x64); n.s. on
  C4.
- **OURS 8x64 − SCALE:** better on C4 (−0.0008 ± 0.0007), n.s. on WikiText-2. OURS 16x64 − SCALE and SCALE − QAT:
  n.s. on both corpora.
- **No arm is significantly better than OURS in any cell on either model.**

**4. Cost without a development set.** Llama / Mistral; the host figure is ru_maxrss.

| arm | calibration | peak GPU | host |
|---|---:|---:|---:|
| OURS | 8.5 / 7.5 min | 40.6 / 36.0 GiB | 18.5 / 14.7 GiB |
| SCALE | 7.2 / 6.1 min | 74.0 / 65.3 GiB | 34.3 / 22.6 GiB |
| QAT | 10.5 / 9.3 min | 85.3 / 81.0 GiB | 45.9 / 34.3 GiB |

The lr selections are extra for the baselines: 3 SCALE runs, and 5 QAT runs of about 11–12 min each.

**5. Larger models: memory probes only.** Three optimizer steps at the unified settings; nothing was trained with a
fallback.
- **Phi-4, 13.6B trained QAT parameters.**
  - FP32-state AdamW QAT does not fit, even at micro-batch 1: its states alone are 109 GB.
  - Fallbacks that fit:
    - 8-bit AdamW: 83.4 GiB, 2.8 s per step;
    - CPU offload of the states: 57.6 GiB, 6.9 s per step;
    - LoRA (rank 16): 37.3 GiB, 3.5 s per step.
  - SCALE fits at micro-batch 2 (86.5 GiB, 2.1 s per step), not at 4 or 8.
- **Qwen3.8-27B, 24.4B trained QAT parameters.**
  - No full-weight QAT variant fits: FP32 states, 8-bit states and CPU offload all run out of memory at micro-batch 8
    and 1, before the first step.
  - LoRA fits: 73.8 GiB, 9.2 s per step.
  - SCALE fits only with checkpointing at micro-batch 1: 82.2 GiB, 33.6 s per step, about 3 h for 20 epochs.
  - For scale: OURS on Qwen took 76 min at 90 GiB (Task 1).

**6. Checks all pass:**
- the deployed runs repeat their chosen development runs' per-step KL (4 of 4);
- nothing outside the scoped Linears changed in any QAT or SCALE run of this study (hash before and after). QAT
  trains exactly the 224 scoped weights. Task 2's Llama SCALE run predates the check; it trains no model parameter by
  construction.
- the new artifacts:
  - QAT: packed = the fake quant of the trained weights, bitwise;
  - SCALE: packed = the learned-scale fake-quant weight, value-equal;
  - ownership exact, with 0 mismatches, for both;
- QAT NativeLinear (c) − fake (c) is negligible on both corpora of both models (B2 criterion);
- all reference policies reproduce the earlier evaluations window for window.

**For the paper:**
- **Under the same data, loss, budget and deployment path, full-weight QAT does not beat TM-OPT+TC.**
  - On Llama it is clearly worse.
  - On Mistral it is worse on WikiText-2 and tied on C4.
  - It needs a learning rate below the registered grid, about twice OURS's GPU memory, and more time.
  - At 14B its standard form no longer fits one 96 GB GPU; at 27B no full-weight form does.
- **SCALE is the stronger baseline:** close to OURS, but never better in any cell, and it uses more GPU memory.
## Llama-3.1-8B (tables)

NativeLinear (c) PPL, WikiText-2 / C4 (released windows):

| policy | WikiText-2 | C4 |
|---|---:|---:|
| BF16 | 6.2403 | 8.9579 |
| NVFP4 | 6.9338 | 9.9284 |
| FourOverSix | 6.8706 | 9.8257 |
| ours-8x64 | 6.7877 | 9.6774 |
| ours-16x64 | 6.7827 | 9.6827 |
| scale | 6.7905 | 9.6949 |
| qat | 6.8187 | 9.7356 |
| qat-fake (fake (c), check only) | 6.8210 | 9.7337 |

Reference policies equal to the earlier evaluations, every window's NLL: BF16 yes, NVFP4 yes, FourOverSix yes, ours-8x64 yes, ours-16x64 yes, scale (Task 2) yes.

Paired ΔNLL ± 2 SE per window (negative: the first is better):

| comparison | WikiText-2 | C4 |
|---|---|---|
| NVFP4 − FourOverSix | +0.00916 ± 0.00210 (worse) | +0.01040 ± 0.00228 (worse) |
| ours-8x64 − FourOverSix | -0.01213 ± 0.00170 (better) | -0.01521 ± 0.00301 (better) |
| ours-16x64 − FourOverSix | -0.01288 ± 0.00177 (better) | -0.01466 ± 0.00307 (better) |
| scale − FourOverSix | -0.01172 ± 0.00173 (better) | -0.01340 ± 0.00268 (better) |
| qat − FourOverSix | -0.00758 ± 0.00151 (better) | -0.00921 ± 0.00208 (better) |
| ours-8x64 − SCALE | -0.00041 ± 0.00142 (n.s.) | -0.00181 ± 0.00119 (better) |
| ours-16x64 − SCALE | -0.00115 ± 0.00157 (n.s.) | -0.00126 ± 0.00121 (better) |
| ours-8x64 − QAT | -0.00455 ± 0.00159 (better) | -0.00600 ± 0.00160 (better) |
| ours-16x64 − QAT | -0.00530 ± 0.00151 (better) | -0.00545 ± 0.00176 (better) |
| SCALE − QAT | -0.00414 ± 0.00162 (better) | -0.00419 ± 0.00139 (better) |

QAT deployment check, NativeLinear (c) − fake (c) of the QAT weights (B2 criterion |mean| ≤ 2 SE):

| corpus | mean ± 2 SE | max abs ΔNLL | ΔPPL | negligible |
|---|---|---:|---:|---|
| wiki | -0.000341 ± 0.001368 | 0.02789 | -0.00232 | yes |
| c4 | +0.000201 ± 0.001164 | 0.08896 | +0.00196 | yes |

Learning rates (development set; not part of any calibration cost):

| arm | rates run (final dev KL) | extensions | chosen |
|---|---|---|---|
| scale | 1e-4: 0.09080, 3e-4: 0.08588, 1e-3: 0.09534 | none | 3e-4 |
| qat | 1e-8: 0.10342, 1e-7: 0.09225, 1e-6: 0.09704, 1e-5: 0.32525, 1e-4: 1.65674 | 1e-7, 1e-8 | 1e-7 |

Calibration cost (Task 1 definition: model load, fit data and teacher, preparation, training, writing the output; no development set) and run checks:

| arm | calibration | GPU allocated / reserved | host sampled / ru_maxrss | deployed run = chosen dev run (per-step KL) | outside the scoped Linears unchanged | trained |
|---|---:|---:|---:|---|---|---|
| ours-8x64 | 8.5 min | 40.6 / 41.6 GiB | 18.6 / 18.5 GiB | — (fixed lr; Task 1: map = committed) | — (no model parameter trained) | tile logits (not model parameters) |
| ours-16x64 | 8.6 min | 40.5 / 41.5 GiB | 18.6 / 18.5 GiB | — (fixed lr; Task 1: map = committed) | — (no model parameter trained) | tile logits (not model parameters) |
| scale | 7.2 min | 74.0 / 75.3 GiB | 34.4 / 34.3 GiB | yes | — (not recorded) | block-scale factors (not model parameters; Task 2 run) |
| qat | 10.5 min | 85.3 / 85.5 GiB | 46.0 / 45.9 GiB | yes | yes | the scoped Linear weights |

Artifacts (new in this study):

| artifact | exporter checks | ownership (exact, format mismatches) | note |
|---|---|---|---|
| llama8b_qat | weights = calibration record, candidates equal (224 modules), packed = fake quant of the trained weights, bitwise | True, 0 | trained weights: 224 of 224 matrices changed |

## Mistral-7B-v0.3 (tables)

NativeLinear (c) PPL, WikiText-2 / C4 (released windows):

| policy | WikiText-2 | C4 |
|---|---:|---:|
| BF16 | 5.3182 | 7.8306 |
| NVFP4 | 5.5506 | 8.0895 |
| FourOverSix | 5.5224 | 8.0623 |
| ours-8x64 | 5.4868 | 8.0215 |
| ours-16x64 | 5.4899 | 8.0323 |
| scale | 5.4902 | 8.0276 |
| qat | 5.4947 | 8.0265 |
| qat-fake (fake (c), check only) | 5.4969 | 8.0302 |

Reference policies equal to the earlier evaluations, every window's NLL: BF16 yes, NVFP4 yes, FourOverSix yes, ours-8x64 yes, ours-16x64 yes.

Paired ΔNLL ± 2 SE per window (negative: the first is better):

| comparison | WikiText-2 | C4 |
|---|---|---|
| NVFP4 − FourOverSix | +0.00510 ± 0.00111 (worse) | +0.00338 ± 0.00093 (worse) |
| ours-8x64 − FourOverSix | -0.00647 ± 0.00094 (better) | -0.00507 ± 0.00097 (better) |
| ours-16x64 − FourOverSix | -0.00590 ± 0.00088 (better) | -0.00372 ± 0.00166 (better) |
| scale − FourOverSix | -0.00585 ± 0.00093 (better) | -0.00431 ± 0.00098 (better) |
| qat − FourOverSix | -0.00502 ± 0.00088 (better) | -0.00444 ± 0.00127 (better) |
| ours-8x64 − SCALE | -0.00062 ± 0.00091 (n.s.) | -0.00076 ± 0.00068 (better) |
| ours-16x64 − SCALE | -0.00006 ± 0.00091 (n.s.) | +0.00059 ± 0.00108 (n.s.) |
| ours-8x64 − QAT | -0.00145 ± 0.00090 (better) | -0.00063 ± 0.00084 (n.s.) |
| ours-16x64 − QAT | -0.00088 ± 0.00082 (better) | +0.00072 ± 0.00079 (n.s.) |
| SCALE − QAT | -0.00083 ± 0.00088 (n.s.) | +0.00013 ± 0.00075 (n.s.) |

QAT deployment check, NativeLinear (c) − fake (c) of the QAT weights (B2 criterion |mean| ≤ 2 SE):

| corpus | mean ± 2 SE | max abs ΔNLL | ΔPPL | negligible |
|---|---|---:|---:|---|
| wiki | -0.000395 ± 0.000764 | 0.01413 | -0.00217 | yes |
| c4 | -0.000463 ± 0.000879 | 0.08556 | -0.00371 | yes |

Learning rates (development set; not part of any calibration cost):

| arm | rates run (final dev KL) | extensions | chosen |
|---|---|---|---|
| scale | 1e-4: 0.03030, 3e-4: 0.02728, 1e-3: 0.03028 | none | 3e-4 |
| qat | 1e-8: 0.03637, 1e-7: 0.02798, 1e-6: 0.06937, 1e-5: 0.49972, 1e-4: 4.53502 | 1e-7, 1e-8 | 1e-7 |

Calibration cost (Task 1 definition: model load, fit data and teacher, preparation, training, writing the output; no development set) and run checks:

| arm | calibration | GPU allocated / reserved | host sampled / ru_maxrss | deployed run = chosen dev run (per-step KL) | outside the scoped Linears unchanged | trained |
|---|---:|---:|---:|---|---|---|
| ours-8x64 | 7.5 min | 36.0 / 36.8 GiB | 14.1 / 14.7 GiB | — (fixed lr; Task 1: map = committed) | — (no model parameter trained) | tile logits (not model parameters) |
| ours-16x64 | 7.5 min | 35.9 / 36.6 GiB | 14.0 / 14.7 GiB | — (fixed lr; Task 1: map = committed) | — (no model parameter trained) | tile logits (not model parameters) |
| scale | 6.1 min | 65.3 / 65.9 GiB | 22.7 / 22.6 GiB | yes | yes | block-scale factors (not model parameters) |
| qat | 9.3 min | 81.0 / 81.1 GiB | 34.3 / 34.3 GiB | yes | yes | the scoped Linear weights |

Artifacts (new in this study):

| artifact | exporter checks | ownership (exact, format mismatches) | note |
|---|---|---|---|
| mistral7b_scale | weights = calibration record, candidates equal (224 modules), packed = learned-scale fake-quant weight, value-equal (-0 packs as +0) | True, 0 | scales moved on 18,953,261 of 436,207,616 blocks |
| mistral7b_qat | weights = calibration record, candidates equal (224 modules), packed = fake quant of the trained weights, bitwise | True, 0 | trained weights: 224 of 224 matrices changed |

## Memory probes (phase 3; 3 optimizer steps of 8 sequences, no deployed output)

| probe | model | arm | optimizer | micro-batch | checkpointing | status | peak GPU allocated / reserved | s per step |
|---|---|---|---|---:|---|---|---:|---:|
| phi4_lora_mb8 | phi4 | lora | torch_adamw | 8 | yes | complete | 37.3 / 38.7 GiB | 3.51 |
| phi4_qat_adamw_8bit_mb8 | phi4 | qat | adamw_8bit | 8 | yes | complete | 83.4 / 86.3 GiB | 2.80 |
| phi4_qat_cpu_offload_mb8 | phi4 | qat | cpu_offload | 8 | yes | complete | 57.6 / 59.1 GiB | 6.93 |
| phi4_qat_fp32_mb1 | phi4 | qat | adamw_fp32 | 1 | yes | oom | 93.7 / 94.1 GiB | — |
| phi4_qat_fp32_mb8 | phi4 | qat | adamw_fp32 | 8 | yes | oom | 93.0 / 94.0 GiB | — |
| phi4_scale_mb2 | phi4 | scale | torch_adamw | 2 | no | complete | 86.5 / 87.1 GiB | 2.12 |
| phi4_scale_mb4 | phi4 | scale | torch_adamw | 4 | no | oom | 93.4 / 94.2 GiB | — |
| phi4_scale_mb8 | phi4 | scale | torch_adamw | 8 | no | oom | 93.8 / 94.1 GiB | — |
| qwen27b_lora_mb8 | qwen27b | lora | torch_adamw | 8 | yes | complete | 73.8 / 77.7 GiB | 9.24 |
| qwen27b_qat_adamw_8bit_mb1 | qwen27b | qat | adamw_8bit | 1 | yes | oom | 93.4 / 94.0 GiB | — |
| qwen27b_qat_adamw_8bit_mb8 | qwen27b | qat | adamw_8bit | 8 | yes | oom | 93.7 / 94.0 GiB | — |
| qwen27b_qat_cpu_offload_mb1 | qwen27b | qat | cpu_offload | 1 | yes | oom | 93.2 / 93.7 GiB | — |
| qwen27b_qat_cpu_offload_mb8 | qwen27b | qat | cpu_offload | 8 | yes | oom | 93.4 / 93.7 GiB | — |
| qwen27b_qat_fp32_mb1 | qwen27b | qat | adamw_fp32 | 1 | yes | oom | 93.4 / 94.0 GiB | — |
| qwen27b_qat_fp32_mb8 | qwen27b | qat | adamw_fp32 | 8 | yes | oom | 93.7 / 94.0 GiB | — |
| qwen27b_scale_mb1 | qwen27b | scale | torch_adamw | 1 | no | oom | 94.0 / 94.1 GiB | — |
| qwen27b_scale_mb1_ckpt | qwen27b | scale | torch_adamw | 1 | yes | complete | 82.2 / 83.5 GiB | 33.59 |
| qwen27b_scale_mb2 | qwen27b | scale | torch_adamw | 2 | no | oom | 94.0 / 94.1 GiB | — |
| qwen27b_scale_mb4 | qwen27b | scale | torch_adamw | 4 | no | oom | 94.0 / 94.1 GiB | — |
| qwen27b_scale_mb8 | qwen27b | scale | torch_adamw | 8 | no | oom | 93.8 / 94.0 GiB | — |

## Files

- **Tables:** `unified.{md,json}`, generated by `analyze.py`.
- **Rules and checks:**
  - `choose_lr.py`, the learning-rate rule;
  - `check_eval.py`, the automatic regression check and check 4;
  - `verify_reuse.py` → `reuse.json`, the reused-run checks.
- **Records:**
  - `runs/<model>/<run>/report.json`: the development runs (`<arm>_dev_lr<rate>`), the deployed runs
    (`<arm>_nodev`) and the evaluation (`eval`);
  - `runs/<model>/lr_choice_<arm>.json`;
  - `runs/probes/<probe>/report.json`;
  - the queues and their log in `runs/queue/`.
- **Artifacts:** `artifacts/<model>_<arm>/{artifact.json, ownership.json}`.
- **Not committed:** the states, packed weights and logs, in `/home/dev/n16k64_campaign/unified_baselines`.
- **Code, the default paths unchanged:**
  - `run_cost_distill.py --model` and its frozen check;
  - `export_map_artifact.py --weights`;
  - `run_ppl_deploy.py`'s `fake:weights:<state.pt>` policy.
