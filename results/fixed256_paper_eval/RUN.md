# Fixed-256-only execution record

Job `335407` was submitted with account `gov113008` on `taide_h200`,
requesting eight H200s, eight four-CPU tasks, and 800 GB host memory.
It runs on hgpn46. Each independent evaluation step requests one H200,
four CPUs, and 90 GB host memory. Slurm schedules up to eight steps at once.

The job evaluates 33 cases: one matched FourOverSix baseline and ten frozen
fixed-256 maps for each of Qwen3-4B, Llama-3.1-8B, and Qwen3.8-27B. No
adaptive or weight-MSE evaluation is submitted. All 66 dataset cells are
written to `job_335407`; aggregation validates all cases before updating
the separate 2048-token section in MIXFP4_REPORT.md.

The initial runner disabled caching on both datasets. Its exact Llama
baseline check caught a WikiText difference: 6.881154 versus the released
6.875525, while C4 remained exactly 9.823733. The one-window diagnostic in
`cache_diagnostic.log` isolates the cache setting: identical weights and
tokens give NLL 1.673648715 with caching disabled and 1.672904015 with
caching enabled, the latter exactly matching the release. Toggling TF32
does not change either value in this diagnostic.

The runner now enables fresh per-window WikiText caching and leaves C4
disabled. Already-started cases retain their old code; later steps use the
corrected code and record both cache flags. Dependent job `335428` copies
the results, reruns only affected WikiText cases, and reuses C4 only after
exact input, quantized-weight, and module-scope checks. It preserves each
pre-fix report as `report_cache_disabled.json`. The failed exact-baseline
check is retained in job 335407; it is not an accepted final result.

Free GPU slots in job 335407 were then used to repair both small-model
baselines and all ten affected small-model maps before the allocation ended.
The corrected Llama baseline exactly matches the archived run at
6.875524520874023 / 9.82373332977295. Both small-model baseline C4 first-window
replay checks have zero NLL error. The dependent job therefore only needs
six 27B WikiText repairs (cases 0, 2, 4, 6, 8, and 10); all other completed
cases are copied with their full provenance and verification records.

Final aggregation in job 335428 passed: 33 cases, 66 PPL cells, all cache
settings correct, all maps unchanged, all fixed maps exactly 256 E0M3
blocks, and 54/60 point improvements over the matched baselines. All
small-model inputs match the released reproduction exactly. Results are
in `job_335428/REPORT.md`, `summary.json`, and `fixed256_cells.csv`.
Job 335407 retains its nonzero Slurm exit from the original baseline
assertion even though the affected baseline was successfully repaired
inside that allocation. Final acceptance uses the verified per-case
reports and the successful consolidated validation, not that failed attempt.

The smaller models use Transformers 4.57.3. The 27B native model requires
Transformers 5.16.1, installed into worker-local scratch. Both use the same
Torch 2.9 environment, BF16 weights, SDPA attention, and full targeted-linear
W4A4 with tensor-wide activation scales. The launcher stores downloads and
temporary dependencies only in its own worker-local /tmp directory.

```bash
sbatch slurm/fixed256_paper_eval.sbatch
```
