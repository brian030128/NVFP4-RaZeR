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
