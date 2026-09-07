# Execution notes

- Job 329471 passed pooled-moment/consensus tests, then stopped before fitting
  or evaluation because the historical score file's task-script hash differs
  from the current repository version. No loss results were produced.
- The restart recomputes all 64 WikiText scores when that hash differs and
  records their maximum absolute difference from the historical scores. It
  does not use the mismatched cached moments for selection or diagnostics.
  Native-model/quantizer source hashes must still match. Token hashes and
  complete baseline fitting/validation losses must reproduce exactly before
  the frozen historical WikiText map is used as a control. Its validation
  losses and acceptance decision must reproduce exactly too.
- This changes computation cost, not fitting windows, selection rules, or
  test data. The original failed job's dependent jobs are canceled and
  resubmitted after the corrected run.
- Before new scores or policy results were produced, job 329481 was restarted
  with an exact tensor-placement optimization: resident frozen parameters are
  not copied to CPU by saved-tensor offload; alternate BF16 weights may stay
  on GPU, while subtraction still occurs in FP32. Every module's score mean
  and SE on two real windows must match the original backend bit-for-bit
  before using this path. This is a compute-cost change, not a new selector.
  The initial baseline-only report is preserved in `seed20260912_preoptimization`.
- Job 329489 rejected the placement optimization: two forward NLLs matched,
  but the per-tile scores were not bit-identical in
  `model.language_model.layers.63.self_attn.q_proj`. No new policy or held-out
  result was produced. The optimized path is disabled by default; all
  scientific runs use the original `save_on_cpu` scoring backend. Its failed
  check is retained as an opt-in diagnostic. The partial report is preserved
  in `seed20260912_backend_check`. No claimed result uses optimized scores.
- Fresh original-backend WikiText scores reproduce all 64 forward losses,
  but not all historical gradient moments: maximum absolute mean/SE
  differences are 8.3432e-6 / 8.7548e-6 across 496 matrices. Both old and fresh
  scores select exactly the same 226 tiles at budget 0.1. Marginal two-SE
  eligibility changes for 54,063 tiles, so exact moment reuse still is not
  assumed. See cache_reproduction.json. This comparison ran as a CPU-only
  step inside the existing H100 Slurm allocation; it did not change the maps.
# Scheduling overlap

The panel remained pending under `QOSMaxJobsPerUserLimit`. Pending 329680,
329681 and 329682 were replaced by 329805 (paired panel), 329806 (paired
teacher) and 329807 (summary after teacher and stress). Each paired job requests
the same aggregate resources as the original two workers: two H100, 24 CPUs,
400 GB RAM. Each subprocess sees one allocated GPU and has a separate
job/worker-local HF cache. No other account jobs were modified. The worker
launcher passed `bash -n` in an existing Slurm CPU step.

After first-seed calibration and isolated probes completed, the independent
smaller-model panel was released to overlap the second target seed and held-out
evaluation. `scontrol` dependency updates returned unspecified errors; pending
329521, 329619 and 329624 were canceled and replaced by 329680 (panel), 329681
(teacher, after panel) and 329682 (summary, after teacher AND target stress).
No scientific settings or data changed. Up to four H100s can now run concurrently.
