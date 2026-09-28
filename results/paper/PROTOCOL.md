# Paper experiments, full runs — protocol

Written 2026-09-28 on branch `tm-opt`, before any full run; the hashes and time are in `registration.json`.
Deviations are appended at the end.

The user approved the flow (relayed by nvfp4-razer-c9), with the decisions below. The flow is
`docs/PAPER_EXPERIMENTS.md` with the scripts in `experiments/paper/`, as committed with this protocol:
`registration.json` records the sha256 of every script. The flow was smoke-tested on Llama-3.1-8B (8eda569).

## What is run

- **Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B.
- **Tile units:** 8x64 and 16x64 in the main tables, 256x64 in the appendix.
- **Ours:** TM-OPT+TC — E2M1 tiles FourOverSix, E0M3 alpha = 1, FourOverSix per-token activations, NativeLinear
  convention (c).
- **Accuracy policies:** BF16, NVFP4, FourOverSix, and Ours at 8x64, 16x64 and 256x64.
- **Latency policies:** these plus two kinds of row:
  - NVFP4 and FourOverSix on stock_wB, the 8x64 same-placement references;
  - "Ours with NVFP4 activations", latency only (an install override of the same artifact).
- **Experiment 1:** WikiText-2 and C4 perplexity (step 03).
- **Experiment 2:** MMLU 5-shot and ARC-Challenge, ARC-Easy, HellaSwag, PIQA 0-shot, with lm-eval 0.4.11 and its
  defaults; batch 16 (Qwen 8); NativeLinear only (step 04).
- **Experiment 3.1:** prefill latency, 5 rounds, 1 × {128, …, 8192} and 4 × 2048, eager and CUDA graph (step 05).
- **Experiment 3.2:** GEMM and activation-quantizer kernel time per text-Linear shape and per forward (step 06).

## The user's decisions (2026-09-28)

1. **Prefill latency:** the CUDA-graph numbers are primary (main tables). Eager is supplementary (appendix), with the
   host-bound flag.
2. **Latency protocol:** 5 rounds, all 8 shapes, graph and eager, as designed.
3. **MMLU:** lm-eval's default (per-choice log-likelihoods in BF16). The tie rate per policy is reported in the
   tables.
4. **If Qwen BF16 runs out of memory on MMLU:** re-run only that (policy, task) with `--batch-size 4`, recorded as a
   deviation. Nothing else changes.

## Order

1. `run_all.sh --models llama8b,mistral7b,phi4`, with the committed maps reused (sha256 match).
2. A progress message: a headline perplexity and accuracy table, the latency summary, and any failed check or
   deviation. Then `collect_results.py` copies the records and tables into `results/paper/`, and they are committed
   and pushed.
3. `run_all.sh --models qwen27b`, started right away.
4. Step 07 over all four models, `collect_results.py`, commit and push, the final report, and stop.

## Registered checks

A failure stops the run. It is reported, not worked around.

| check | where |
|---|---|
| each reused map's sha256 equals the committed manifest (`experiments/paper/maps.sha256.json`) | step 01 |
| ownership: every weight element's decoded value and executed format are exact, maps on their own kernel, baselines on stock_wA and stock_wB | step 02 |
| windows: WikiText-2 and C4 token hashes equal the published record (where one exists) and sm120's reference, and are equal across policies | run_ppl_deploy.py; step 07 |
| coverage: every quantized Linear runs natively in every forward; lm-eval also requires that the unscoped Linears (Qwen's vision tower) never run | steps 03, 04, 05 |
| samples: every policy of a (model, task) has the same `sample_digest` (lm-eval's doc / prompt / target hashes) | step 07 |
| accuracy recomputation: each accuracy recomputed from the per-example correctness equals lm-eval's value | step 07 |
| graph = eager: every prefill shape is captured, and the graph's logits equal eager's bitwise | step 05 |
| an idle GPU: nothing else runs on it; steps 00, 05 and 06 refuse a busy GPU | all |

## Outputs

- **Everything:** under `PAPER_OUT` = `/home/dev/n16k64_campaign/paper`.
- **In the repository:** `results/paper/`, filled by `experiments/paper/collect_results.py`:
  - the environment check and `commands.log` (every command, with its exit code);
  - the calibration and export records;
  - per-window perplexity records;
  - the lm-eval records, gzipped: per-example correctness, per-choice log-likelihoods and sample hashes;
  - the latency and GEMM records, and the tables.
- **Not committed:** the artifacts, the maps and the per-command logs.

## Deviations (append-only)

(none yet)
