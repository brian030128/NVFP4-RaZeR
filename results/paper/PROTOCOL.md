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

1. **2026-09-29, before the change is applied: step 06 measures its configurations in a rotated order over 3 rounds.**
   - **Approved by the user** (relayed by nvfp4-razer-c9), after the three-model results. It is applied only after
     the Qwen run ends: that run uses the registered bench_gemm.py unchanged.
   - **Reason: a measurement-order effect in the registered step 06.** bench_gemm.py timed the configurations
     back-to-back, in the same order at every (projection, T): stock_wA, stock_wA with NVFP4 weights, stock_wB, mixed
     16x64, mixed 256x64, n8k64_wB. The same stock_wA kernel measured second (NVFP4 weights) was slower than
     measured first (FourOverSix weights). Per forward:
     - Llama and Mistral: about 0.0 % at T = 128 and 512, +0.7 % and +0.8 % at 2048, +1.4 % at 8192;
     - Phi-4: +7.1 % at 8192, and gate_up_proj (35840×5120) at T = 8192 took 2,446 against 2,186 µs (+11.9 %), with
       the two within 0.7 % at T = 4096.
     The effect grows with the GEMM's load, like a clock or power state. So the GEMM tables' "Ours vs stock" ratios,
     measured later in the order, were inflated at large T. Phi-4's T = 8192 row, +11.1 % per forward for 16x64, is the
     clearest case: 16x64 shows +1.6 % end to end at 1×8192. The end-to-end prefill (separate processes, shuffled
     rounds) is not affected.
   - **Change:**
     - bench_gemm.py measures every configuration in each of 3 rounds;
     - round r starts the configuration list at position 2r (rotated), so every configuration is measured early,
       mid and late;
     - the reported time is the median of the 3 per-round medians (each the CUPTI median of 20 calls);
     - step 06 is re-run for all four models.
   - **The old GEMM records** are kept as `gemm_superseded/`, with their table as `tables/gemm_superseded.md`,
     labelled superseded.
   - **Added to the report:** a consistency check between the per-forward GEMM time difference (Ours minus the
     reference with the same activation quantizer) and the end-to-end CUDA-graph prefill difference, per model,
     comparison and prompt shape. A row is flagged when the two differ by more than 1 % of the reference prefill time.
   - **Found before the rerun, with the old records:** for 16x64 and 256x64 at 1×2048 and 4×2048 the two agree within
     0.5 pp, except Phi-4 at T = 8192. Other rows disagree by 1–5 pp: 1×128 and 1×256, where the isolated GEMM
     difference is larger, and 8x64 at mid lengths, where the end-to-end difference is larger. The old vs new numbers
     are appended below after the rerun.
