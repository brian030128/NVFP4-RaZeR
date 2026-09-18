# Reordering handoff — 2026-09-19

Continue with **Llama-3.1-8B and Qwen3.8-27B only**; skip Qwen3-4B.
Target `m256n256k64`, hence weight type tiles **N256 x K64**.
Only weights mix E2M1/E0M3; activations retain FourOverSix E2M1.
All compute, including CPU search and tests, goes through Slurm under
`gov113008`, using `taide` (H100) or `taide_h200` (H200).

## Implemented and checked

- [Plan](MIXFP4_GB200_PLAN.md), linked from the root MixFP4 report and its generator.
- [Method and artifact contract](TASK_REORDER.md): joint row/column search,
  capacity-constrained assignments, exact improving swaps, multistart search,
  fit/election split, and legal 16-column scale-group permutations.
- `run_math_code_calibration.py --reorder-modules REGEX` streams per-sequence
  CE/KL scores at 1x16 granularity. Historical 8x64 scores cannot replace these.
- `run_task_reorder.py` searches one matrix and exports its permutations and
  independently elected 256x64 map.
- Nine focused tests passed in worker step `348924.0`, then again in job
  `349179` before calibration (9/9, 0.629 seconds). Worker syntax checks passed
  in step `348924.1`.

No real-model reordering quality or GB200 throughput result exists yet.
Rotation and the native fused kernel remain planned work.

## Submitted calibration jobs

Both jobs were submitted from `/home/u4320956/NVFP4-RaZeR` on this cluster.
They continue independently of this chat or client machine.

| Model | Job | Resources | Output |
|---|---|---|---|
| Llama-3.1-8B | 349178 | taide, 1 H100, 4 CPUs, 96G | `results/task_reorder/pilot_20260919/llama8b/calibration` |
| Qwen3.8-27B | 349179 | taide_h200, 2 H200, 24 CPUs, 400G | `results/task_reorder/pilot_20260919/qwen27b/calibration` |

At the last check, Llama was pending and Qwen was running, downloading model
files after passing the tests. Check the scheduler for current state:

```bash
squeue -a -u u4320956
sacct -j 349178,349179 --format=JobID,State,Elapsed,ExitCode
tail -c 8000 slurm/logs/reorder_cal_349179.out
tail -c 4000 slurm/logs/reorder_cal_349179.err
```

Use `-a` with `squeue`: the taide partitions are hidden from the default view.
Read bounded byte tails; model download progress uses carriage returns and
can make `tail -n` unexpectedly huge. Poll no more frequently than 30 seconds.

The pilot covers the final layer's **gate_proj, up_proj, down_proj**:
`model.layers.31.mlp.*` for Llama and
`model.language_model.layers.63.mlp.*` for Qwen.
The collector still computes the historical full-model scores as well.
Calibration uses 128 pinned math/code sequences. Search uses 64; independent
election uses the other 64, stratified by source.

The batch script uses `--allow-source-drift` because commit `384b803` appended
the unused paired FourOverSix helper to `quantize/quantizer.py`; the existing
quantizer functions were unchanged. The calibration report records this drift.
Check regenerated historical maps against the shipped calibration maps before
claiming reproduction. Transformer versions are 4.57.3 (Llama) and 5.16.1
(Qwen); job-local dependencies and HF downloads live under worker `/tmp`.

## Work still to do

**Only calibration has been submitted. Search/evaluation dependencies have
not been wired, and no new-job background monitor was started before handoff.**
`scripts/watch_mixfp4_jobs.py` currently knows the earlier coarsening job log
prefixes; generalize those paths before using it for these new jobs.

1. Monitor the two calibration jobs. Require a complete `report.json` and
   all three complete `reorder_scores/NNN/manifest.json` files. Preserve scores
   on shared storage: `.pt` files and Slurm logs are intentionally excluded
   from Git and do not arrive with a clone on another machine.
2. Submit one search per module using the supplied worker script, e.g. after
   successful Llama calibration:

   ```bash
   sbatch --dependency=afterok:349178 --kill-on-invalid-dep=yes \
     slurm/task_reorder.sbatch \
     --scores results/task_reorder/pilot_20260919/llama8b/calibration/reorder_scores/000 \
     --out results/task_reorder/pilot_20260919/llama8b/search/000
   ```

   Repeat for `001` and `002`; Qwen uses dependency `349179` and model directory
   `qwen27b`. Consult each manifest for the module-to-index mapping. Defaults
   are k=3, seed=0, four starts, six rounds, both axes, and 256x64 tiles.
3. Implement the matched model-quality evaluator. Compare FourOverSix,
   identity-layout 256x64, and reordered 256x64 on **the same pilot modules**;
   keep all other weights at FourOverSix. Freeze layouts before evaluating.
   `run_task_reorder.py` currently reports identity election counts/objective
   but does not save the identity mask: export that mask or regenerate it
   from the identical election split for the matched control.
4. Reuse the root report's protocol (`run_kse_paper.py` and
   `run_baseline_protocol_audit.data`): 2048 tokens, 141 WikiText windows,
   256 seed-0 C4 crops, tensor-wide activation factors, SDPA, WikiText cached,
   C4 uncached, float32 PPL aggregation, and paired per-window NLL differences.
   Calibration uses causal per-token activation factors, as recorded in the
   manifests; evaluation must retain the published tensor-wide convention.
5. The exported reference weights are in permuted deployment order. A fake
   quantization quality evaluator may undo both permutations on those mixed
   weights to run the original model graph, preserving each 16-element scale
   group. Validate that algebra separately from native runtime performance.
   No speed claim follows from a quality simulation.

The existing coarsening experiments in `.claude/worktrees/tile256` are separate
work. Do not modify or cancel their jobs while continuing this pilot.
