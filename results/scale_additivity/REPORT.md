# Task 2: does scale training add to TM-OPT+TC, or cancel it? — report

Protocol: `PROTOCOL.md`, registered 2026-09-27T22:21:47Z (sha256 3678a7b1…, `registration.json`), before any Task 2
run.
- **One deviation:** `analyze.py` was corrected, reporting code only (`PROTOCOL.md`, Deviations). Every run's code and
  the queue have their registered hashes.
- **The study is descriptive.** Nothing was selected or tuned on WikiText-2, C4 or zero-shot. SCALE's learning rate
  was chosen on the development set.

## Summary

Llama-3.1-8B. Every arm is calibrated on the same 128 fit sequences, with KL to the BF16 teacher, deterministic and
without a development set. Every arm is evaluated on the deployment path, NativeLinear (c), on the released
WikiText-2 (141) and C4 (256) windows.

**1. Scale-only training is a strong baseline, but it does not beat ours.**
- **SCALE** (arm D: a learned factor on every block's UE4M3 scale, 20 epochs, lr 3e-4) improves on FourOverSix by
  ΔNLL −0.0117 (WikiText-2) and −0.0134 (C4). That is 88–97 % of OURS's gain.
- **OURS − SCALE:**
  - WikiText-2: not significant (−0.0012 ± 0.0016 at 16x64, −0.0004 ± 0.0014 at 8x64);
  - C4: OURS significantly better (−0.0013 ± 0.0012 at 16x64, −0.0018 ± 0.0012 at 8x64).
- **Development KL** agrees: SCALE 0.0859. OURS 16x64 scored 0.0837 on the same set in the pre-registration smoke
  test, where the learned-scale path at f = 1 reproduces the map bitwise.
- **Other differences:**
  - SCALE is plain NVFP4 E2M1 and runs on the stock kernel.
  - It costs 7.2 min against OURS's 8.5–8.6 min.
  - It needs 74.0 GiB of GPU memory against 40.5 GiB.

**2. Combining the two does not add their gains.**
- **The combined gain is about one gain, not two.** Each combination gains 0.0118–0.0163 over FourOverSix, against
  0.012–0.015 for either single method and a sum of 0.024–0.029.
- **The interaction is significantly positive in all 8 cells** (2 combinations × 2 units × 2 corpora):
  I = +0.011 to +0.014 ± 0.002–0.003. The combined gain falls short of the sum by about one single method's gain.
- **Registered reading:** "cancelling" in 7 of 8 cells. There, the combination is not significantly better than the
  better single method (OURS in every cell). It is also not significantly worse than it.
- **The one exception:** OURS→SCALE at 16x64 on C4 is "partially overlapping". It is −0.0016 ± 0.0014 better than
  OURS; its WikiText-2 difference is −0.0001 ± 0.0015.
- **Consistent with overlap:** when TM-OPT+TC starts from the learned-scale E2M1 base (SCALE→OURS), it selects about
  a third fewer E0M3 tiles: 129,826 against 201,648 at 16x64, and 203,895 against 309,517 at 8x64. The combinations
  also fit the calibration set better (last-epoch training KL 0.028–0.029, against 0.034 for SCALE and 0.040–0.042
  for OURS), without a matching held-out gain.

**For the paper:**
- **Against the reviewer argument:** at an equal budget, same data, same loss and the deployment evaluation,
  scale-only KL training does not beat TM-OPT+TC. It ties on WikiText-2 and is slightly worse on C4.
- **What must be conceded:**
  - SCALE comes within 3–12 % of TM-OPT+TC's gain, as plain NVFP4 on the stock kernel.
  - Stacking the two methods buys essentially nothing.
- **Agreement with the collaborator branch:** it found that KL scale search recovers most of a trained E0M3 map's
  gain, and that E0M3 on top of trained scales adds no robust out-of-domain gain.

## Perplexity

NativeLinear (c), WikiText-2 / C4; paired ΔNLL ± 2 SE per window.

Labels: `scale` = SCALE, `ours` = OURS, `scale_ours` = SCALE→OURS, `ours_scale` = OURS→SCALE.

Reference policies equal to the Parts 2-3 evaluation, every window's NLL: BF16 yes, FourOverSix yes, NVFP4 yes, ours-16x64 yes, ours-8x64 yes.

| arm | WikiText-2 | C4 |
|---|---:|---:|
| BF16 | 6.2403 | 8.9579 |
| NVFP4 | 6.9338 | 9.9284 |
| FourOverSix | 6.8706 | 9.8257 |
| scale | 6.7905 | 9.6949 |
| ours-16x64 | 6.7827 | 9.6827 |
| scale_ours-16x64 | 6.7899 | 9.6781 |
| ours_scale-16x64 | 6.7819 | 9.6670 |
| ours-8x64 | 6.7877 | 9.6774 |
| scale_ours-8x64 | 6.7849 | 9.6777 |
| ours_scale-8x64 | 6.7851 | 9.6798 |

## Paired comparisons, 16x64

| comparison | WikiText-2 | C4 |
|---|---|---|
| SCALE − FourOverSix | -0.01172 ± 0.00173 (better) | -0.01340 ± 0.00268 (better) |
| SCALE − NVFP4 | -0.02088 ± 0.00234 (better) | -0.02380 ± 0.00413 (better) |
| OURS − FourOverSix | -0.01288 ± 0.00177 (better) | -0.01466 ± 0.00307 (better) |
| OURS − SCALE | -0.00115 ± 0.00157 (n.s.) | -0.00126 ± 0.00121 (better) |
| scale_ours-16x64 − FourOverSix | -0.01182 ± 0.00178 (better) | -0.01513 ± 0.00342 (better) |
| scale_ours-16x64 − SCALE | -0.00009 ± 0.00147 (n.s.) | -0.00173 ± 0.00128 (better) |
| scale_ours-16x64 − OURS | +0.00106 ± 0.00155 (n.s.) | -0.00047 ± 0.00135 (n.s.) |
| ours_scale-16x64 − FourOverSix | -0.01299 ± 0.00177 (better) | -0.01628 ± 0.00344 (better) |
| ours_scale-16x64 − SCALE | -0.00126 ± 0.00145 (n.s.) | -0.00288 ± 0.00128 (better) |
| ours_scale-16x64 − OURS | -0.00011 ± 0.00147 (n.s.) | -0.00162 ± 0.00143 (better) |

| combination | corpus | gain SCALE | gain OURS | sum | combined gain | interaction I (± 2 SE) | reading |
|---|---|---:|---:|---:|---:|---|---|
| scale_ours-16x64 | wiki | +0.01172 | +0.01288 | +0.02460 | +0.01182 | +0.01278 ± 0.00225 (worse) | cancelling |
| scale_ours-16x64 | c4 | +0.01340 | +0.01466 | +0.02806 | +0.01513 | +0.01293 ± 0.00262 (worse) | cancelling |
| ours_scale-16x64 | wiki | +0.01172 | +0.01288 | +0.02460 | +0.01299 | +0.01161 ± 0.00218 (worse) | cancelling |
| ours_scale-16x64 | c4 | +0.01340 | +0.01466 | +0.02806 | +0.01628 | +0.01178 ± 0.00264 (worse) | partially overlapping |

## Paired comparisons, 8x64

| comparison | WikiText-2 | C4 |
|---|---|---|
| SCALE − FourOverSix | -0.01172 ± 0.00173 (better) | -0.01340 ± 0.00268 (better) |
| SCALE − NVFP4 | -0.02088 ± 0.00234 (better) | -0.02380 ± 0.00413 (better) |
| OURS − FourOverSix | -0.01213 ± 0.00170 (better) | -0.01521 ± 0.00301 (better) |
| OURS − SCALE | -0.00041 ± 0.00142 (n.s.) | -0.00181 ± 0.00119 (better) |
| scale_ours-8x64 − FourOverSix | -0.01255 ± 0.00179 (better) | -0.01518 ± 0.00312 (better) |
| scale_ours-8x64 − SCALE | -0.00082 ± 0.00152 (n.s.) | -0.00178 ± 0.00100 (better) |
| scale_ours-8x64 − OURS | -0.00041 ± 0.00151 (n.s.) | +0.00003 ± 0.00106 (n.s.) |
| ours_scale-8x64 − FourOverSix | -0.01252 ± 0.00182 (better) | -0.01496 ± 0.00297 (better) |
| ours_scale-8x64 − SCALE | -0.00080 ± 0.00150 (n.s.) | -0.00156 ± 0.00143 (better) |
| ours_scale-8x64 − OURS | -0.00039 ± 0.00144 (n.s.) | +0.00026 ± 0.00155 (n.s.) |

| combination | corpus | gain SCALE | gain OURS | sum | combined gain | interaction I (± 2 SE) | reading |
|---|---|---:|---:|---:|---:|---|---|
| scale_ours-8x64 | wiki | +0.01172 | +0.01213 | +0.02386 | +0.01255 | +0.01131 ± 0.00229 (worse) | cancelling |
| scale_ours-8x64 | c4 | +0.01340 | +0.01521 | +0.02861 | +0.01518 | +0.01343 ± 0.00267 (worse) | cancelling |
| ours_scale-8x64 | wiki | +0.01172 | +0.01213 | +0.02386 | +0.01252 | +0.01134 ± 0.00221 (worse) | cancelling |
| ours_scale-8x64 | c4 | +0.01340 | +0.01521 | +0.02861 | +0.01496 | +0.01365 ± 0.00323 (worse) | cancelling |

## SCALE learning rate (development set, 20 epochs; not part of any calibration cost)

| learning rate | initial dev KL | final dev KL | run time |
|---|---:|---:|---:|
| 1e-4 | 0.10837 | 0.09080 | 8.6 min |
| 3e-4 (chosen) | 0.10837 | 0.08588 | 8.6 min |
| 1e-3 | 0.10837 | 0.09534 | 8.6 min |

## Calibration cost (Task 1 definition) and tiles

Memory: the maximum over the arm's calibration runs of peak GPU allocated / peak host RSS (sampled).

| arm | calibration | peak GPU / host | E0M3 tiles | blocks whose scale moved |
|---|---:|---:|---:|---|
| scale | 7.2 min (scale_nodev) | 74.0 / 34.4 GiB | 0 | 22,478,968 of 436,207,616 (5.2 %) |
| ours-16x64 | 8.6 min (ours-16x64) | 40.5 / 18.6 GiB | 201,648 | — |
| ours-8x64 | 8.5 min (ours-8x64) | 40.6 / 18.6 GiB | 309,517 | — |
| scale_ours-16x64 | 15.7 min (scale_nodev + scale_ours_16x64) | 74.0 / 34.4 GiB | 129,826 | 22,036,796 of 436,207,616 (5.1 %) |
| scale_ours-8x64 | 15.7 min (scale_nodev + scale_ours_8x64) | 74.0 / 34.4 GiB | 203,895 | 22,133,064 of 436,207,616 (5.1 %) |
| ours_scale-16x64 | 20.2 min (ours-16x64 + ours_scale_16x64) | 74.4 / 34.4 GiB | 201,648 | 18,750,705 of 436,207,616 (4.3 %) |
| ours_scale-8x64 | 20.2 min (ours-8x64 + ours_scale_8x64) | 74.4 / 34.4 GiB | 309,517 | 18,334,060 of 436,207,616 (4.2 %) |

"Gain" is the ΔNLL improvement over FourOverSix (positive = better). The interaction I = X − SCALE − OURS +
FourOverSix per window. I > 0 means the combined gain is smaller than the sum of the single gains.

## Checks

- **Initialization, bitwise:**
  - SCALE at f = 1 is FourOverSix. The initial development KL is 0.10837 at every learning rate.
  - OURS→SCALE at f = 1 is the committed map's weight, checked on every module (`init_equals_map_weight`).
  - In SCALE→OURS, candidate B, packed from the learned scales, decodes bitwise to the learned-scale E2M1 weight
    (candidate store).
- **Training quantizer:** the fused training quantizer equals the deployed weight bitwise after training, in all
  three development runs.
- **Determinism:** the deployed SCALE run (`--no-dev`) repeats the 3e-4 development run's 320 per-step training KLs
  exactly. It is the model that scored development KL 0.0859.
- **Artifacts:** all five learned-scale artifacts pass the exporter's checks and the hardware ownership check. The
  checks are:
  - f = 1 equals the standard candidates;
  - packed = learned fake-quant weight;
  - ownership exact, with 0 format mismatches;
  - E0M3 tags equal the map (the four map artifacts).
- **Evaluation:** the five reference policies (BF16, FourOverSix, NVFP4, OURS 16x64 and 8x64) reproduce the Parts 2–3
  evaluation window for window. This is a regression check of the refactored `run_ppl_deploy.py`. The windows equal
  the SM120 reference.

## Files

- **Tables:** `additivity.{json,md}`, generated by `analyze.py`. `choose_lr.py` is the learning-rate rule.
- **Records:** `runs/<run>/report.json` for the development runs, the deployed runs and the evaluation
  (`runs/llama8b`). The queue and its log are in `runs/queue.sh` and `runs/commands.txt`; the learning-rate choice is
  in `runs/lr_choice.json`.
- **Artifacts:** `artifacts/<arm>/{artifact.json, ownership.json}` and the map provenance.
- **Not committed:** the maps, the scale states, the packed weights and the logs. They are in
  `/home/dev/n16k64_campaign/scale_additivity`.
- **Code, the default paths unchanged:**
  - `run_cost_distill.py --epochs --act-rows --map/--unit`;
  - `run_train_map.py --base-scales`;
  - `export_map_artifact.py --scales/--scales-apply`;
  - `quantize/learned_scale.py`;
  - the `base_given` option of `repro_local/realquant/{native_dev,candidate_store}.py`.
