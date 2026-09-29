# Paper experiments: results

- **Protocol:** `PROTOCOL.md`, registered in `registration.json`.
- **Flow and scripts:** `docs/PAPER_EXPERIMENTS.md`, `experiments/paper/`.
- **This directory:** filled by `experiments/paper/collect_results.py` from the run's output root
  (`/home/dev/n16k64_campaign/paper`). It holds records and tables only; the artifacts, maps and logs stay there.

**Status: complete for all four models.**
- `run_all.sh --models llama8b,mistral7b,phi4` ran 2026-09-28 17:46 → 2026-09-29 02:22 UTC.
- `run_all.sh --models qwen27b` ran 2026-09-29 02:24 → 17:22 UTC.
- Every registered check passed.
- **One deviation (PROTOCOL.md, deviation 1):** step 06 re-run for all four models in 3 rotated rounds, 17:24–17:48.
  The fixed-order GEMM records are kept in `gemm_superseded/`, with their tables in `tables/gemm_superseded.md`.
- Qwen BF16 did not run out of memory on MMLU, so its batch-4 rule was not needed.
- Our method is labelled **FlipQuant (ours)** in the tables (the paper name; code path TM-OPT+TC).

| path | what |
|---|---|
| `tables/main.md` | 8x64 and 16x64: perplexity with paired ΔNLL; downstream accuracy with paired differences and the MMLU tie rates; CUDA-graph prefill latency (primary); GEMM per forward |
| `tables/appendix.md` | 256x64; eager prefill with the host-bound mark; GEMM per shape |
| `tables/tables.json` | everything in the tables, with the check outcomes (`samples_checked`, `graph_equal_eager`) |
| `00_check.json`, `commands.log` | the environment check; every command with its start, end and exit code |
| `maps/`, `artifacts/` | map reuse records (sha256 = committed); export and ownership records (stock_wA and stock_wB for the baselines) |
| `ppl/<model>/<policy>.json` | run_ppl_deploy.py reports: per-window NLL, coverage, the kernel-set description |
| `lmeval/<model>/<policy>.json.gz` | run_lmeval_deploy.py reports: per-example correctness, per-choice log-likelihoods, sample digests, native GEMM counts |
| `latency/<model>/<policy>/round<r>.json` | bench_prefill.py: eager and CUDA-graph prefill per shape, the host-bound flag, the graph = eager check, quantizer reuse |
| `gemm/<model>.json` | bench_gemm.py: GEMM and quantizer kernel time per projection and token count, 3 rotated rounds (deviation 1) |
| `gemm_superseded/<model>.json` | the fixed-order GEMM records superseded by deviation 1 |
| `tables/gemm_superseded.md` | their tables, labelled superseded |
