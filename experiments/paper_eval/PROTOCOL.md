# Paper evaluation on the RTX PRO 6000 (SM120): protocol for the GPU parts P, S, A, B, C

Registered on branch `paper-eval` before any GPU run of these parts. The scope is the user's decision of 2026-10-07,
relayed by the coordinator: BF16, NVFP4, FourOverSix and FlipQuant only (no FOCUS, GPTQ, IF4, MixFP4, ablation maps or
PTQ variants); FlipQuant = the 5-epoch release maps (256 windows, top-1000, `/home/dev/flipquant_release`, whose tiles
equal the Hugging Face copies); models Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2, Phi-4,
Qwen3.8-27B (flipquant registry keys `qwen3-1.7b`, `qwen3-8b`, `mistral-7b`, `nemotron-nano-9b-v2`, `phi4-14b`,
`qwen3.8-27b`); units 8x64, 16x64, 256x64. D (generative accuracy) and E (lm-eval) get their own registration.

## Common

- **Code:** flipquant main = 120173a (`/home/dev/n16k64_campaign/fqopt/wt`, clean, read-only; no flipquant change);
  this branch's `experiments/paper_eval/` (`run_gpu.py` drives every job, `mem_probe.py` is part B's probe).
- **Kernels:** `build_V` (`/home/dev/n16k64_campaign/kernel_opt/build_V`, the kernel-opt deployment directory since
  4ae044a), passed to flipquant as `--kernel-build-dir`; flipquant's tile table
  (`kernels/sm120/configs/<gpu>.ko.json`) is byte-identical to the tracked one. Every native record must name build_V
  as its build directory, the policy's kernel family, and only libraries whose sha256 are in build_V's manifests.
- **Policies:** `bf16` (`--mode bf16`); `nvfp4` (`--mode native --weight nvfp4 --act nvfp4`, stock_ko); `fo6`
  (`--weight fourover6 --act fourover6`, stock_ko); `fo6-wB` (the same with `--kernel-set auto_stock_wB`,
  stock_wB_ko, the same-placement reference of 8x64); `fq-8x64` / `fq-16x64` / `fq-256x64` (`--weight mixfp4 --act
  fourover6 --map <release map>`, routed by `auto` to mixed_wB_ko / mixed_ko / mixed256_ko).
- **Environment:** one process per job, one job at a time, an idle GPU (no compute process) before each, the default
  caching allocator, no clock locking, HF caches as cached (no download expected; a gated download that fails stops the
  run, tokens are not switched). `queue.log` holds every START/END and the round orders.
- **Outputs:** `/home/dev/n16k64_campaign/paper_eval/{parity,smoke,latency,memory,ownership,fakeppl}/`, logs in
  `logs/`; the reports and tables go to `results/paper_eval/` on this branch.

## P: harness parity (before A)

flipquant's `evaluation.latency prefill` has never been compared with the paper's harness (NVFP4-RaZeR
`experiments/paper/bench_prefill.py`, step 05). On Phi-4, the model both support:
- **RaZeR harness:** this branch's `bench_prefill.py` with `SM120_BUILD_DIR=build_V`; the paper artifacts
  `phi4_fo6` (`--kernel auto_stock`) and `phi4_tc_16x64` (`--kernel auto`), and BF16 (no artifact).
- **flipquant harness:** `--mode bf16`, `fo6`, and FlipQuant 16x64 with the SAME map as the RaZeR artifact (the paper
  run map `paper/maps/phi4_16x64/map.pt`, `--unit 16x64`), build_V.
- All 8 shapes, 7 repetitions, CUDA graphs; 3 rounds, the harness order alternating per round and the policy order
  rotated.
- **Criterion (graph ms, the median over the 3 rounds per shape):** for each policy, flipquant / RaZeR within ±1 % at
  the median over the 8 shapes and within ±3 % at every shape; and the FlipQuant 16x64 / FourOverSix ratio of the two
  harnesses within 0.5 points at the median over shapes. If it fails, the run stops and the coordinator decides.

## S: smoke (not a result)

Every (model, policy), 1 round, shape 1x128, 3 repetitions, with A's checks, into `smoke/`. It proves the routes, the
CUDA-graph capture and the coverage, and times the processes. **Contingency:** a model whose capture fails in S runs
A with `--no-graph` (eager only); its rows are flagged as eager in the tables, and this is reported.

## A: prefill latency (Table latency; prefill only)

- 6 models × 7 policies × 5 rounds; shapes 1x128, 1x256, 1x512, 1x1024, 1x2048, 1x4096, 1x8192, 4x2048; 7 repetitions;
  eager and CUDA graph per process (flipquant prefill).
- Round r runs the 7 policies in the order `random.Random(20260928 + r).shuffle` (step 05's seeds), per model.
- **Registered checks (a failure stops the run):** exit code 0 and a complete record; every shape captured and the
  graph's logits equal eager's bit for bit (except a model under S's contingency); coverage (every quantized Linear
  ran natively); the kernel family and build_V provenance above.
- **Reported:** per (model, shape, policy) the median over rounds of each process's graph median (ms); ratios paired
  within rounds (median [min, max] over rounds): FlipQuant vs FourOverSix (16x64, 8x64, 256x64), FlipQuant 8x64 vs
  FourOverSix weights-on-B, NVFP4 vs FourOverSix, BF16 vs FourOverSix; the eager numbers and the host-bound flags as
  supplementary.

## B: memory

6 models × {bf16, nvfp4, fo6, fq-8x64, fq-16x64, fq-256x64}, one process each, `mem_probe.py` (flipquant's own
loader, as A):
- the installed bytes of the quantized linears (packed FP4, placed scales, bias) and of everything else, the
  quantized linears' BF16 size and the ratio (the draft says 0.28), the whole-model bytes and its ratio to BF16;
- `memory_allocated` after the load, and the peak during one 1x2048 prefill (an eager forward writing the KV cache,
  A's tokens).

## C1: ownership check

The 18 release artifacts (6 models × 3 units) installed with flipquant's `--ownership-check` (every weight element
decoded by the kernel itself must equal its stored value under the map's format; the install raises on any
mismatch), through `evaluation.ppl ... --paper-convention --ownership-check --limit 1`. Reported per artifact: the
modules and weight elements checked, the informative (nonzero) elements, the E0M3 tiles and elements observed, and the
format mismatches (expected 0).

## C2: native vs simulated

5 models (all but Qwen3.8-27B, whose simulated path does not fit one 96 GB GPU) × 3 units: `evaluation.ppl --mode fake
--weight mixfp4 --act fourover6 --map <release map> --paper-convention`, all windows. Against the release evaluation's
native per-window NLLs on the same windows: per (model, unit, corpus) the paired ΔNLL native − simulated ± 2 SE; a
cell agrees when |Δ| ≤ 2 SE. Reported: the count of the 30 cells that agree, and every cell.

## Order and deviations

P, S, A, B, C1, C2, one job at a time. Any deviation (a changed setting, a skipped job, a contingency) is written here
as an amendment before the affected jobs run, or reported as a deviation if found after.

## Amendment 1 (2026-10-07 19:23 UTC): the record check's field names

The parity run stopped at its first native flipquant record (`parity_flipquant_fo6_r1`) on a driver bug, not a
measurement: `check_fq_record` read the install report from `policy.kernel` / `policy.build_dir` and the loaded
libraries from keys named `*sha*`, but flipquant records them under `policy.native.{kernel, build_dir}` and
`kernel_set.kernels` (name → library sha256). Fixed to read those fields (and to fail when no loaded kernel is
recorded); the criteria are unchanged. The stopped record passes the corrected check (stock_ko, build_V, all four
libraries in build_V's manifests). The run resumes; completed records are kept and re-checked by the report.

## Amendment 2 (2026-10-07 19:55 UTC): the parity outcome, accepted by the user

P ran 19:16–19:39 UTC (3 rounds, 18 processes, every record passing amendment 1's checks). Graph ms, the median over
rounds, flipquant vs the step-05 harness: BF16 −0.23 % (worst shape 0.95 %), FourOverSix +0.74 % (2.02 %), FlipQuant
16x64 **+1.02 %** (1.56 %) against the ±1 % bound; the FlipQuant 16x64 / FourOverSix ratio agrees within **0.06 pt**
at the median (bound 0.5). Both harnesses load the same libraries, table, widths and activation quantizer; the run order
and the GPU temperature move absolute times by ±3–5 % (round 1 vs round 2), and a ~+0.6–1.0 % absolute offset remains on
the FP4 policies. The user accepted flipquant's `evaluation.latency prefill` for A: every paper latency number comes
from this one harness, whose absolute times are ~1 % above the step-05 harness's on the FP4 policies
(`results/paper_eval/parity/PARITY.md`).

## Amendment 3 (2026-10-07 19:55 UTC): the hybrid models' env, a model subset, and the order

The user's decisions (relayed 19:5x UTC):
- **n16k64-fast.** Nemotron-Nano-9B-v2 and Qwen3.8-27B run EVERYTHING of this round in a new env n16k64-fast: a clone of
  n16k64 plus, inside that env only, flash-linear-attention 0.5.2 (+ fla-core 0.5.2, einops 0.8.2, ninja), the CUDA 12.8
  compiler and dev headers from conda-forge (no system, driver or /usr/local change), and causal-conv1d 1.7.0 and
  mamba-ssm 2.3.2.post1 built from source for sm_120; torch 2.9.0+cu128, triton 3.5.0 and transformers 5.16.1 unchanged
  (the freeze delta is recorded in `env/`). Before any measurement in it, `verify_fast_env.py` must show that
  transformers resolves the fast implementations (mamba_ssm / causal_conv1d / fla) for both models and that their kernels
  run in a 1x2048 prefill; its record `env/fast_env_verified.json` gates the driver (`run_gpu.py` refuses the hybrid
  models without it). The maps stay as calibrated, under the fallback kernels.
- **F (new): the hybrid models' main-table PPL in n16k64-fast**: BF16, NVFP4, FourOverSix and FlipQuant 8x64 / 16x64 /
  256x64, both corpora, flipquant `evaluation.ppl --paper-convention`, reported against the fallback numbers.
- **The other four models** (Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Phi-4) run in n16k64, the env of every
  earlier record of this study.
- **Driver:** `run_gpu.py <part> --models=a,b` runs a subset, in the registered model order; each part logs its models.
- **Order:** the four models first: S, then B, C1 and C2 (no timing; they may run while n16k64-fast builds on the CPU),
  then A once no build is running. Then the hybrid models after the env verification: S, F, the D pilot (its own
  registration), A, B, C1, and C2 for Nemotron. A never runs for the hybrid models on the fallback.

## Amendment 4 (2026-10-07 20:20 UTC): the D pilot (timing and lengths; not a result)

The user's design, relayed 2026-10-07: measure the generation lengths before the budgets are chosen; no full D run until
the user chooses. `run_pilot_d.py`, flipquant `evaluation.accuracy` natively, Nemotron-Nano-9B-v2 and Qwen3.8-27B in
n16k64-fast, `--recommended-decoding` with thinking on, seed 0:
- **Lengths** (BF16 and FlipQuant 16x64): `--max-new-tokens 32768` for every task; gsm8k, math500 and ifeval
  `--limit 16` (one process), aime `--limit 2 --samples 8` (aime24 + aime25: 16 samples each), batch 16.
- **Batch scaling** (all 6 policies): gsm8k `--max-new-tokens 512` at batch 16 and 32 (`--limit 64`) and 64 (`--limit
  128`, so that a second batch exists: a batch's duration is the gap between two logged batch ends), giving the
  per-step decode time per batch size (with thinking on, nearly every sample reaches 512).
- **Reported:** per task and model the length distribution and the truncation rate at 4k / 8k / 16k / 32k; the time per
  (model, policy, task) at batch 16 / 32 / 64, extrapolated to the full task sizes (GSM8K 1319, MATH-500 500, AIME 60 × 8,
  IFEval 541) from the measured lengths and step times; OOMs as they occur (a failed process is recorded and the pilot
  continues).
- Every log line carries its wall-clock time (each batch's duration); per-sample tokens and truncation are the harness's
  `.jsonl`. Order: after the hybrid models' smoke, before F and A.
- Driver: `run_gpu.py` gained `pplfast` (part F of amendment 3).

## Amendment 5 (2026-10-07 20:20 UTC): n16k64-fast built and verified (records in `results/paper_eval/fast_env/`)

- **Build (env-local only):** the clone of n16k64; pip `--no-deps`: einops 0.8.2, fla-core 0.5.2,
  flash-linear-attention 0.5.2, ninja 1.13.2; conda-forge into the env only (`cuda-version=12.8`, cuda-nvcc 12.8.93,
  cuda-cudart-dev, cuda-cccl, libcublas-dev, libcusparse-dev, libcusolver-dev, openssl pinned; 46 conda packages added,
  none removed or changed); then causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1 built from source with that nvcc
  (gencodes include compute_120 / sm_120; the system g++ 11.4 as host compiler). The pip freeze delta is exactly those
  six packages; torch 2.9.0+cu128, triton 3.5.0 and transformers 5.16.1 are unchanged. `pip check` notes mamba-ssm's
  undeclared-optional requirements apache-tvm-ffi, quack-kernels and tilelang (not installed; Mamba3 / newer kernels,
  unused by Nemotron-H's Mamba2); `import mamba_ssm` succeeds.
- **Verification (`verify_fast_env.py`, BF16, one 1x2048 prefill under the profiler):**
  - Nemotron-Nano-9B-v2: causal_conv1d_fn / _update resolve to the causal_conv1d package (fallback env: transformers'
    torch functions); mamba_ssm's Triton kernels (`_chunk_scan_fwd`, `_chunk_state_fwd`, `_state_passing_fwd`,
    `_bmm_chunk_fwd`, `_chunk_cumsum_fwd`) and causal_conv1d's CUDA kernel run; 1x2048 prefill 141 ms vs 2,335 ms on
    the fallback (16.5×).
  - Qwen3.8-27B: causal_conv1d resolves to the package; fla's gated-delta-rule Triton kernels
    (`chunk_gated_delta_rule_fwd_kernel_h`, `chunk_fwd_kernel_o`, `..._kkt_solve`) and causal_conv1d's kernel run;
    1x2048 prefill 428 ms vs 961 ms (2.2×).
  - The mamba / fla function names are not module attributes of the modeling files; the kernels in the profile are the
    evidence for those paths.
- `fast_env_verified.json` (in the run directory's `env/`, copied to `results/paper_eval/fast_env/`) now gates the hybrid models in the drivers.

## Amendment 6 (2026-10-07 20:35 UTC): IFEval's scorer dependencies in n16k64-fast

The first D-pilot process (Nemotron BF16, gsm8k + math500 + ifeval) failed when scoring IFEval: lm-eval's IFEval
checker imports `langdetect` and `immutabledict`, which neither env had (gsm8k and math500 had finished; the first IFEval
batch was generated, then lost). Installed into n16k64-fast only, pip `--no-deps`: langdetect 1.0.9 and immutabledict
4.3.1 (nltk was present; its `punkt_tab` data downloaded itself into `~/nltk_data` on the first import). The pilot
continued with its next jobs; after its first pass, `run_pilot_d.py` runs again and redoes only that job's IFEval
part (the harness resumes from its `.jsonl`; the partial summary is kept as `main.json.partial_ifeval_failed`). The
freeze delta in `results/paper_eval/fast_env/` now lists eight packages.

## Amendment 7 (2026-10-08 00:41 UTC): the Qwen3.8-27B BF16 AIME length job at batch 8

The pilot's Qwen3.8-27B BF16 AIME length job (batch 16) ran out of memory after 70 min, in SDPA's `repeat_kv` at about
22.7k tokens of context (`Tried to allocate 4.16 GiB`; 79.70 GiB allocated by PyTorch, 11.55 GiB reserved but
unallocated, of 94.97 GiB); no sample was recorded. A repeat at batch 16 would fail the same way: weights (51 GiB) plus
the KV cache and `repeat_kv`'s copies (88 KiB per token and sequence) need 95 GiB at 16 x 32k. The rerun pass runs this
one job at **batch 8** (73 GiB predicted at 32k) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the
fragmentation the failure reported), and its long-context step time is reported as batch 8. The failed run's log is
kept as `logs/dpilot_qwen3.8-27b_bf16_aime.pass1.log`. The driver now keeps an earlier pass's log as
`<job>.pass<k>.log` instead of overwriting it (the rerun of Nemotron BF16's gsm8k / math500 / ifeval job resumes from
its `.jsonl`, and its first pass's log holds the gsm8k and math500 batch times). The pilot's other jobs are unchanged.

## Amendment 8 (2026-10-08 04:58 UTC): the Qwen3.8-27B BF16 AIME batch-8 rerun cancelled

The user's decision (relayed 04:5x UTC): amendment 7's rerun is cancelled before it ran (FlipQuant 16x64's AIME length job
took 14,905 s, two 16-sample batches of ~2.07 h each, and a BF16 batch-8 rerun would delay F, A, B and C by hours).
`run_pilot_d.py` skips that job (`CANCELLED`); the rerun pass keeps only the Nemotron BF16 IFEval redo (amendment 6).
The pilot report prices Qwen3.8-27B BF16 AIME from FlipQuant 16x64's AIME lengths and BF16's measured step model, flagged
as an estimate; at the 32k cap BF16 fits batch ≤ 8 only (or needs a 16k cap for batch 16).

## Amendment 9 (2026-10-08 05:31 UTC): C2's native reference for Nemotron-Nano-9B-v2

Under amendment 3, Nemotron-Nano-9B-v2's simulated run (C2) is in n16k64-fast, while the release evaluation's native
per-window NLLs were measured in the fallback env. A native − simulated difference across envs would include the
env change (the fast Mamba2 / conv kernels vs transformers' torch fallback). For Nemotron, C2's agreement count uses the
native NLLs of part F (`pplfast`, FlipQuant 8x64 / 16x64 / 256x64, the same env and windows); the registered comparison
against the release records is reported alongside as a secondary table. The four non-hybrid models are unchanged
(n16k64 on both sides). Report-only: no job changes (C2 for Nemotron runs after F in the chain, so its reference exists).
`report_bc.py` (C2) and the new `report_f.py` (F vs the fallback numbers) implement this.
