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

## Amendment 10 (2026-10-08 06:00 UTC): the driver's completion check

flipquant's `evaluation.ppl` and `evaluation.accuracy` rewrite their report after every corpus / task and add
`resources` only when the process finishes, while `run_gpu.py`'s `done()` took any record without a `status` field as
complete; an interrupted PPL job would have been skipped on a restart (none was: every record so far is finished, and
the one D-pilot case was handled by hand under amendment 6). `done()` now requires `status == "complete"` where a
record has a status (latency, memory) and `resources` otherwise; `report_bc.py` and `report_f.py` skip unfinished
records. Applies to the chain's parts that start after this change (fakeppl, then A, F, B, C1 and C2 for the hybrids);
no measurement changes.

## Amendment 11 (2026-10-08 08:01 UTC): F's NVFP4 row

F stopped at 08:00:54 on its NVFP4 job (exit 1 before loading: flipquant's `--paper-convention` is defined for FourOverSix
activations only). The main table's NVFP4 rows (main-ppl 44f8cea and the paper's Qwen3.8-27B record) use convention
(c), per-token NVFP4 activation scales (`nvfp4_rows`), as A's NVFP4 does (`evaluation.latency`'s default
`--act-scope row`). F's NVFP4 job therefore runs `--weight nvfp4 --act nvfp4 --act-scope row` without
`--paper-convention` (`evaluation.ppl` defaults to `--act-scope document`); every other F job is unchanged. Nemotron's
finished BF16 record is kept; the failed log is kept as `logs/pplfast_nemotron-nano-9b-v2_nvfp4.failed_convention.log`.
The chain resumes from F (F, A, B, C1 for the hybrid models, C2 for Nemotron).

## Parts G-L (registered 2026-10-08 19:42 UTC): the remaining SM120 tables on flipquant paper-sm120-runs

The user's request (relayed 19:0x UTC): the remaining SM120 paper tables on this machine with the co-author's flipquant
branches (map-ablation, ptq-combo, IF4, MIXFP4, FOCUS), merged onto main 120173a as branch `paper-sm120-runs` and
adapted to the paper's final settings. Common settings (all user-approved):
- **Maps (A1):** FlipQuant = the 5-epoch release maps (`/home/dev/flipquant_release`). Everything that trains or takes
  gradients uses the release settings: `--fit-windows 256 --teacher-topk 1000`, 5 epochs, lr 0.02, init -1, seed 0,
  deterministic.
- **Calibration data (A2):** GPTQ, FOCUS, the activation-weighted statistics and the one-shot gradient use the release
  maps' 256 x 512 fit set: the reference calibration record's 128 windows, then the 128 of the release trainer's fit
  extension, rebuilt by `flipquant.data.calibration_release` from the release run's own records (every window's token
  sha256 checked against them); the trainer-based one-shot uses the release runs' data root
  (`/home/dev/n16k64_campaign/fqrel/tmopt_data`) and so the same windows.
- **Environment (A5):** Nemotron-Nano-9B-v2 and Qwen3.8-27B in n16k64-fast, the others in n16k64; native = build_V with
  `--kernel-set auto`, the paper convention (per-token FourOverSix activations; NVFP4 with per-token NVFP4 scales,
  amendment 11). One GPU job at a time. Driver: `run_sm120.py`; records under the run directory's `<part>/`.
- **Code:** flipquant `paper-sm120-runs` @ 7b1cfe3 (pushed; worktree `/home/dev/n16k64_campaign/sm120runs/wt`): main
  120173a + map-ablation dc7ae28 + ptq-combo 05267c6 + IF4 8513264 + MIXFP4 8ec3fbd + FOCUS a06e835 (moved onto main's
  names; its native deployment on main's sm120 path), with the settings above (`calibration_release`,
  `baseline_maps --calib-extension`, `--gptq-calib release`, `train_focus --data release`, `tmopt_ext` with the release
  settings, `mapcheck.is_release`). Tests: n16k64, the whole suite: 1 failure, `test_teacher_topk.py::
  test_topk_kl_bounds_full_kl`, which fails identically on main 120173a (the legacy trainer, not used here), and the
  IFEval scorer test deselected (langdetect is only in n16k64-fast); n16k64-fast, the hybrid / GPTQ / rotation / FOCUS /
  IF4 / Zou / mapcheck / baseline-map / benchmark tests: 129 passed, 17 skipped.
- Stop and report on: a blocked push, installs outside n16k64(-fast), an OOM that would need a changed setting, any
  failing check, an implausible result.

### G: tab:ablation (map selection)

Nemotron-Nano-9B-v2 and Qwen3.8-27B x 8x64 / 16x64 / 256x64. Every baseline map has, in every layer, exactly `k_l`
E0M3 tiles, `k_l` from the release map of the same model and unit (`--match`):
- **Random:** `calibration.baseline_maps --method random --seed 0 / 1 / 2`; the three seeds' per-window NLLs are
  averaged per window before the comparison.
- **Activation-weighted:** `baseline_maps --method act --calib-record <record> --calib-extension <release trainer
  report>`: X = the BF16 inputs on the 256 windows; per tile the drop in the layer-output error from E2M1 to E0M3
  (`H_cc = X_c^T X_c` blocks); top `k_l`. Batch 8 (Qwen3.8-27B: 1, batch1_only).
- **One-shot gradient:** `calibration.tmopt_ext oneshot`: the vendored TM-OPT+TC trainer with the release settings,
  one epoch over the 256 windows with the top-1000 teacher, every optimizer step intercepted before its update (the
  tile gradients summed, no logit moves, all-E2M1 start); per module the `k_l` most negative tiles.
- **FlipQuant:** the release map. **FourOverSix:** part F's records (n16k64-fast, the same windows and convention),
  reused after check G0.
- **Measured:** `evaluation.ppl --mode native --weight mixfp4 --act fourover6 --map <map> --paper-convention`,
  WikiText-2 and C4. **Reported:** paired ΔNLL vs FourOverSix on the same windows, x 1e-3 nats/token, ± 2 SE (WikiText-2
  for the main table, C4 for Appendix A), with every cell's PPL.
- **Check G0 (before the maps):** the FlipQuant 16x64 release map of Nemotron-Nano-9B-v2 evaluated through
  paper-sm120-runs gives part F's per-window NLL bit for bit on both corpora (otherwise F's records are not reused
  and FourOverSix / FlipQuant are re-measured on paper-sm120-runs).
- **Checks:** every baseline map has exactly `k_l` E0M3 tiles per layer (the scripts assert it); every PPL record has
  F's window count; the one-shot run used 256 fit windows and top-1000 (its trainer report).

### H: tab:ptq (PPL only; the GSM8K column is not run)

Nemotron-Nano-9B-v2 and Qwen3.8-27B at 16x64, n16k64-fast, native (build_V, `--kernel-set auto`), the release
revisions. Rows {RTN, GPTQ, Hadamard} x {NVFP4 (NVFP4 activations, per-token scales), FourOverSix, FlipQuant 16x64}:
- **RTN:** part F's records (the same env, windows and conventions; check G0).
- **Hadamard (block 16):** `--rotate hadamard --rotate-block 16` on every quantized linear's input (W H offline, x H
  before the activation quantizer; RotatedNativeLinear: the rotation runs in PyTorch before the native kernel, it is not
  fused -- for the caption). NVFP4: `--weight nvfp4 --act nvfp4 --act-scope row`; FourOverSix and FlipQuant:
  `--paper-convention`. FlipQuant's map is retrained in the rotated basis with the release settings
  (`calibration.tmopt_ext train --unit 16x64 --rotate hadamard --rotate-block 16`, 5 epochs, `--fit-windows 256`,
  `--teacher-topk 1000`, the release runs' data root); the map's meta records rotate / rotate_block, and `--map` accepts
  it only with the same rotation.
- **GPTQ** (the user's decision, relayed before 20:28 UTC: option (b)): flipquant `gptq.py` on the fixed RTN grid, block 128, damp
  0.01 x mean diag(H), no act-order, sequential layerwise, the release 256 x 512 fit set (`--gptq-calib release`), and
  **BF16 propagation** (`--gptq-propagate bf16 --gptq-hessian-input bf16`): during calibration every activation
  quantizer is off -- the layers already done run with quantized weights and BF16 activations, as in weight-only GPTQ
  -- so a weight format's codes do not depend on the activation format it is evaluated with. This is a deviation from
  the co-author's / NVIDIA ModelOpt's default (`quantized`: the layers propagate as they will be evaluated), recorded
  as such. FlipQuant 16x64: the release map fixed, each tile GPTQ-rounded on its format's grid. Native evaluation of
  the codes as they are (`sm120.pack_codes`). Batch 8 (Qwen3.8-27B: 1, batch1_only).
- **Check H1 (codes):** for each hybrid model the NVFP4 codes are computed twice, inside the NVFP4-activation run and
  inside the FourOverSix-activation run (separate caches); their sha256 (`gptq.codes_sha256` in the reports, and the
  code files' tensors) must be equal. The FourOverSix-activation run is the hybrid model's GPTQ‡ cell (J).
- Reported: WikiText-2 and C4 PPL per row, and paired ΔNLL vs the RTN FourOverSix row (± 2 SE).

### I: main table, IF4 and MixFP4 (Zou), simulated

All six models (Nemotron-Nano-9B-v2 and Qwen3.8-27B in n16k64-fast), `evaluation.ppl --mode fake --weight if4` /
`--weight zou_mixfp4` with `--act-method own --unit 1x16` and the default `--e2m1 own` (each method's own weight and
activation rules: per-16-block selection, its own E2M1, per-token activation scales; not the + FourOverSix variant),
the release revisions. Spot check (the four non-hybrid models): per-window NLL equal to the main-ppl (44f8cea) IF4 / Zou
records bit for bit on both corpora, and the installed weight sha256 equal where both record it; any difference is
reported. A Qwen3.8-27B OOM leaves that cell empty and is reported (no setting is changed).


### J: main table, GPTQ‡

NVFP4 weights from `gptq.py` with the settings of H's GPTQ (BF16 propagation, the release fit set, block 128, damp
0.01, no act-order), evaluated natively with FourOverSix per-token activations (`--weight nvfp4 --act fourover6
--any-act --paper-convention`), all six models (the hybrid models' cells are H's FourOverSix-activation NVFP4 runs,
check H1). Batch 8 (Qwen3.8-27B: 1).

### K: main table, FOCUS

`calibration.train_focus` with FOCUS's own settings (FOCUS_PAPER_RUNBOOK.md: `--epochs 1 --batch 32 --lr-scale 5e-3
--lr-sub 1e-3 --topk 1000 --num-sub 2 --init-q 6 --act fourover6 --act-scope row --seed 42`) on the release 256 x 512
fit set (`--data release`): 8 optimizer steps. Micro-batch 8, halved on OOM (the global batch stays 32; any change is
recorded). Deployed as NVFP4 codes (`--focus-deploy`) on the sm120 path with FourOverSix per-token activations
(`--paper-convention`). Five models; Qwen3.8-27B is TBD (FOCUS does not fit one 96 GB GPU; accepted by the user).

### L: the final table main-ppl

`tables_final.py` from per-window NLL: all rows (BF16, NVFP4, FourOverSix, IF4, MixFP4 (Zou), GPTQ‡, FOCUS,
FlipQuant x 3; the hybrid models' BF16 / NVFP4 / FourOverSix / FlipQuant from part F), the 12-pair loss recovered,
daggers (FlipQuant not significantly better than FourOverSix) and FlipQuant vs NVFP4 significance, and the tile
granularity statistics (|8x64 - 16x64| paired ΔNLL: max, mean, significance; 256x64's share of the 16x64 gain overall and
per model).

### Order and code for H-L (registered 2026-10-08 20:31 UTC)

H (Hadamard, then GPTQ for the two hybrid models), I, J (the four non-hybrid models), K, then L; one job at a time,
after G. Code: flipquant paper-sm120-runs @ 1a2094e (7b1cfe3 + `--gptq-propagate` and `gptq.codes_sha256`, with a test:
the codes are equal for NVFP4 and FourOverSix activations with BF16 propagation and differ with the default). G's jobs
from 20:28 UTC ran on the worktree at 1a2094e; its diff from 7b1cfe3 touches only the GPTQ path, which G does not use.

### Amendment 12 (2026-10-08 23:13 UTC): GPTQ's memory for Qwen3.8-27B

H's first Qwen3.8-27B GPTQ job (NVFP4, NVFP4 activations) stopped at 23:10:50 UTC with a CUDA OOM after its last layer
(the codes were complete and saved to the job's code file): the sm120 install packs the codes while GPTQ's collected
codes (unpacked, ~1.25 bytes per weight: ~30 GB for 24.4 G weights) were still on the GPU next to the calibration
layers. Fix, memory only (flipquant paper-sm120-runs c67a5bd): the collected codes are kept on the host (the sm120
install moves each module's codes back itself; the digest reads the host copies), and the cache-load path releases
each original BF16 linear as its layer is installed, as the compute path does. No setting changes. Check: Nemotron's
GPTQ NVFP4 row re-evaluated from its code file with the fix gives the same codes_sha256 and bit-identical per-window NLL
on both corpora (`ptq/nemotron-nano-9b-v2/verify_memfix_gptq_nvfp4.json`). The chain resumes at Qwen3.8-27B's GPTQ (its
first job reloads the saved codes), then I, J, K as registered.

### Amendment 13 (2026-10-09 03:00 UTC): K's training job and its output directory

K's first job (Qwen3-1.7B FOCUS training) stopped at 02:59:29 UTC before training: `calibration.train_focus` requires a
new `--out` directory, and the driver's job runner had created it (as the parent of the job's record). The driver now
runs the FOCUS training without creating that directory (`run_fresh_dir`); nothing else changes. The failed log is
kept as `logs/focus_train_qwen3-1.7b_micro8.failed_existing_dir.log`. The chain resumes with K.
K's restarted training (03:00 UTC) was stopped after 3 steps at 03:01 UTC: it trained on the release windows (512
tokens), but `train_focus` records `--seqlen` (default 2048) in its report and log. flipquant paper-sm120-runs 46dc5a7
makes `--data release` require `--seqlen` equal to the windows' length, and the driver passes `--seqlen 512`; the stopped
log is kept as `logs/focus_train_qwen3-1.7b_micro8.stopped_seqlen_record.log`. K restarts from its first model.

## Part M (registered 2026-10-09 07:29 UTC, commit 352ff6a): tab:ptq's GSM8K column

The user's request (relayed 2026-10-09 07:1x UTC): fill tab:ptq's GSM8K column; independent of the FOCUS upload.
- **Scope:** the 18 tab:ptq configurations of part H, nothing else (no BF16 or other rows): Nemotron-Nano-9B-v2 and
  Qwen3.8-27B x {RTN, GPTQ, Hadamard} x {NVFP4, FourOverSix, FlipQuant 16x64}, native 16x64 in n16k64-fast, build_V
  with `--kernel-set auto`, the release revisions, the same artifacts and activations as H: RTN = part F's
  configurations (the release 16x64 map); GPTQ = H's code files (BF16 propagation; read from the cache with H's
  settings); Hadamard = block 16 in PyTorch with FlipQuant's rotated-basis map. NVFP4 rows: NVFP4 activations with
  per-token scales; the others: per-token FourOverSix. Flags per configuration: `run_gsm8k.py quant()`, H's without
  the PPL-only ones.
- **Harness:** flipquant `evaluation.accuracy` on paper-sm120-runs @ 46dc5a7, unchanged (no option needed: greedy is
  its default and `--no-think` switches thinking off). GSM8K test (openai/gsm8k main @ 740312a, 1,319 problems), its
  prompt (the problem, then "Please reason step by step, and put your final answer within \boxed{}.", in the chat
  template) and its scorer (the last `\boxed{}` after any think block, else the last number; numeric comparison).
- **Decoding (the user's settings):** greedy (`do_sample=False`; the harness passes no temperature / top-p / top-k when
  not sampling; neither model's generation_config sets a repetition penalty or another logits processor), seed 0.
  Stop at the first EOS of the model's generation_config (Nemotron-Nano-9B-v2 [2, 11, 12]; Qwen3.8-27B [248046,
  248044]) or at `--max-new-tokens 2048`.
- **Thinking off:** Qwen3.8-27B: the chat template's `enable_thinking=False` (the template then drops its reasoning-
  effort system message, default xhigh, and pre-fills an empty `<think>\n\n</think>\n\n`). Nemotron-Nano-9B-v2: the
  documented `/no_think` system prompt (the template strips it and pre-fills `<think></think>`). Both through the
  harness's `--no-think`; rendered prompts in `ptq_gsm8k/pilot/<model>/prompt.json`. **Check M1:** no completion
  contains `<think>` or `</think>` (checked on the pilot and on every record; any hit is reported).
- **Batch and padding:** the harness's left padding and its prompt-length order, one batch size per model for all 9
  configurations, decided by M0.
- **M0 pilot (not a result):** RTN FourOverSix, the first 64 problems at batch 16 and the first 64 (Nemotron) / 32
  (Qwen3.8-27B) at batch 1 (`run_gsm8k.py pilot`; every stdout line time-stamped): generated lengths, load time, decode
  time per step, and batch identity (each problem's completion text at batch 16 equal to batch 1's). Rule: a model runs
  at batch 16 only if all its pilot completions at batch 16 equal batch 1's; otherwise at batch 1 (identical to batch 1
  by construction). The batch dependence is reported either way. The ETA comes from the pilot; if the total exceeds
  ~24 h, stop and report the options before M1.
- **M1:** the 18 runs (`run_gsm8k.py run`), resumable (the harness appends each problem to `<out>.jsonl` and resumes from
  it), one job at a time, Nemotron-Nano-9B-v2 first.
- **Report** (`results/paper_eval/ptq_gsm8k/`): accuracy (%) ± 2 SE (binomial, sqrt(p(1-p)/n)) per configuration;
  paired per-problem comparisons -- each format vs FourOverSix within the same PTQ method, and GPTQ / Hadamard
  FlipQuant vs RTN FlipQuant -- as the accuracy difference ± 2 SE of the per-problem difference, with the discordant
  counts and McNemar's exact two-sided p; truncation at 2048 (the harness's flag: generated tokens >= 2048, the pad
  token not counted) and unparseable answers (no `\boxed{}` and no number) per configuration; mean and max generated
  tokens; the per-problem records (the harness's JSONL, gzipped); the GSM8K column of `table_ptq.tex`.
- **Checks:** M1 (thinking off, above); M2: every GPTQ record's `codes_sha256` equals H's record of the same
  configuration; M3: every FlipQuant record installs the map H used (path, modules, E0M3 tiles); M4: 1,319 problems in
  every record. Stop and report on: a failing check, an OOM that would need a changed setting, an implausible result,
  or a total ETA above ~24 h after the pilot.

### Amendment 14 (2026-10-09 08:13 UTC): part M's batch, 64 on both models (the user's choice)

The M0 pilot (paper-eval abc0a00, `results/paper_eval/ptq_gsm8k/pilot/PILOT.md`) found that greedy outputs depend on the
batch size on both models. Batch 16 against batch 1, RTN FourOverSix: Nemotron-Nano-9B-v2 10 of 64 completions
identical (7 extracted answers and 6 correctness outcomes differ), unpadded rows included (1 of 4 identical);
Qwen3.8-27B 1 of 32 identical (2 answers differ). The cause is the batch shape (the kernels' shapes change the rounding,
and the FP4 activation rounding amplifies it), not a padding leak. Batch 1 for the 18 configurations would take ~162 h,
above the ~24 h limit, so M1 stopped and the options went to the user, who chose (2c) (relayed 2026-10-09 08:12 UTC):
- The registered batch rule (batch 16 only if identical to batch 1, else batch 1) is replaced by: **one batch size per
  model, 64**, with identical batches and padding (the harness's prompt-length order and left padding) for all 9
  configurations of a model. The batch dependence is stated here and goes into the caption notes: the GSM8K accuracies
  are those of greedy decoding at batch 64; at another batch size individual completions differ.
- Before M1, a timing check at batch 64 per model (`run_gsm8k.py check --check-batch 64`: RTN FourOverSix, the pilot's
  first 64 problems): memory and the projected total. If it fits in memory and the projected total is <= ~24 h, M1
  starts for all 18 configurations without waiting. On an OOM, that model drops to batch 32 for all 9 of its
  configurations, recorded in an amendment.
- Batch 64 against batch 16 on the shared pilot problems is reported as a side note, not a gate.
- Everything else as registered: greedy, EOS or 2048 new tokens, thinking off, the 1,319 problems, the report (paired
  per-problem comparisons, truncation, unparseable answers) and checks M1-M4.

### Amendment 15 (2026-10-09 13:21 UTC): part M paused for kernel-opt's K-tile ablation (the user's decision)

The user's decision (relayed 2026-10-09 12:2x UTC): pause M1, give the GPU to the K-tile dispatch ablation (kernel-opt
amendment 20), then resume M1 automatically. Keep the protocol's batch identity: every configuration sees the same
64-problem batches and padding as an uninterrupted run.
- **The stop.** This session's attempt to stop the M1 processes was refused by its permission settings; the user stopped
  them (`kill -TERM` of the driver 1949674 and of the running job 1973147) between 13:17:48 and 13:20:53 UTC. The
  queue log has no END line for that job.
- **Finished (complete records):** Nemotron-Nano-9B-v2 9 / 9; Qwen3.8-27B RTN NVFP4 / FourOverSix / FlipQuant 16x64 and
  GPTQ NVFP4 (check M2 passed: codes loaded from part H's cache, codes_sha256 equal).
- **Partial:** Qwen3.8-27B GPTQ FourOverSix: 576 records, 9 whole batches of 64 (`gsm8k_resume_check.py`,
  `results/paper_eval/ptq_gsm8k/pause/`). The harness writes a batch's records only after the whole batch is generated,
  and on a rerun it skips the recorded problems and sorts the rest by prompt length as a fresh run sorts all of them (a
  stable sort); the 576 records are exactly the first 576 problems of that order, written in it, every line parsed, no
  duplicate. So the resumed run's batches are the uninterrupted run's batches 10 ... 21 (the same problems, the same
  left padding; greedy decoding does not read the per-batch seed). **Rule:** a partial configuration resumes only if its
  records pass this check; otherwise they are moved aside (kept) and it reruns from the start. This one passes and
  resumes.
- **Not started:** Qwen3.8-27B GPTQ FlipQuant 16x64 and the three Hadamard configurations.
- **Resume:** after the ablation's timing, `run_gsm8k.py run --batch nemotron-nano-9b-v2:64,qwen3.8-27b:64` (finished
  configurations are skipped by their complete records). Post-hoc check: GPTQ FourOverSix's records, in file order,
  equal the harness's full order (the batches continued the partition).

### Amendment 16 (2026-10-09 16:52 UTC): part M closed as superseded (the user's decision); part N replaces it

The user's decision (relayed 2026-10-09 16:4x UTC): re-measure tab:ptq's GSM8K column with lm-eval's own method, task
`gsm8k_llama` (part N). Part M (flipquant `evaluation.accuracy`, 0-shot `\boxed{}` prompt) is superseded, not failed; its
records are kept as they are, finished and partial, and M1 is not completed. It can go in an appendix.
- **The stop:** the user stopped the M1 driver (1975430) and its job (1984890) at ~16:50 UTC (`kill -TERM`, typed by the
  user); the coordinator saw no process and an idle GPU at 16:50:53 UTC, this session at 16:51:09 UTC. The queue log has
  no END line for that job.
- **State:** 16 of 18 configurations complete -- Nemotron-Nano-9B-v2 9 / 9; Qwen3.8-27B RTN x 3, GPTQ x 3 (GPTQ
  FourOverSix resumed after amendment 15's pause), Hadamard NVFP4. Qwen3.8-27B Hadamard FourOverSix: partial, 768
  records = 12 whole batches in the harness's order, no report file. Qwen3.8-27B Hadamard FlipQuant: not run.
- **Amendment 15's post-hoc check passed:** every complete configuration's records, in file order, are the harness's full
  order (`results/paper_eval/ptq_gsm8k/order_check.json`); the paused one continued the batch partition.
- **Archived:** `results/paper_eval/ptq_gsm8k/GSM8K.md` (marked superseded), `gsm8k.json`, the per-problem records of the
  16 complete configurations (`records/`). They do not feed tab:ptq; part N does.

## Part N (registered 2026-10-09 16:55 UTC): tab:ptq's GSM8K column with lm-eval's gsm8k_llama, thinking off

The user's decision (relayed 2026-10-09 16:4x UTC; replaces part M, amendment 16). Settings are the user's (the MR-GPTQ /
RaZeR settings); nothing in the task is changed.
- **Task:** lm-eval 0.4.11's `gsm8k_llama` exactly as shipped (`lm_eval/tasks/llama3/instruct/gsm8k/gsm8k.yaml`): 8-shot CoT
  with the task's own shots (`first_n`), `apply_chat_template=True` and `fewshot_as_multiturn=True`, greedy
  (`do_sample: false`, `temperature: 0`), `max_gen_toks: 1024`, `until: []` (HFLM adds the tokenizer's EOS), filters
  `strict_match` and `flexible_extract`, metric exact_match. The full test set (1,319; openai/gsm8k main @ 740312a, the
  harness's pin), 1 repeat. lm-eval's seeds as `evaluation.downstream` (0 / 1234 / 1234 / 1234).
- **Thinking off, through lm-eval's own switches:** Qwen3.8-27B: HFLM `enable_thinking=False` (its `chat_template_args`;
  the template then drops its reasoning-effort system message and pre-fills an empty think block). Nemotron-Nano-9B-v2:
  lm-eval's `system_instruction="/no_think"` (the template strips it and pre-fills `<think></think>`).
- **Scope:** the 18 tab:ptq configurations, the artifacts and activations of parts H / M (`run_gsm8k.quant`): the release
  16x64 maps; GPTQ = part H's code files (BF16 propagation), whose codes_sha256 must equal part H's records; Hadamard
  block 16 with the rotated-basis map. NVFP4 rows: NVFP4 activations (per-token scales); the others: per-token
  FourOverSix. Native, build_V, `--kernel-set auto`, n16k64-fast.
- **Harness:** flipquant `evaluation.downstream` on paper-sm120-runs f2a56f1: lm-eval's HFLM on the natively quantized
  model object (`HFLM(pretrained=model)`), so generate_until runs `model.generate` through the native kernels; the native
  coverage check (every quantized Linear in every forward, nothing outside the scope) as in its log-likelihood use. The
  new options are off by default (`--apply-chat-template`, `--system-instruction`, `--enable-thinking`, `--gen-lengths`,
  `--samples-out`, `--doc-ids`). HFLM's max length is the model's (131072 / 262144), so no prompt is truncated (checked
  in N0).
- **Batch:** one fixed batch size per model for all 9 configurations (never "auto"): 64 if it fits, else 32 for that
  model (recorded). lm-eval sorts the requests by prompt length (descending, deterministic), so every configuration of a
  model sees the same batches.
- **N0 (not a result; `run_gsm8k_lmeval.py smoke / pilot`):** smoke -- RTN FourOverSix, 8 problems per model: the rendered
  prompt of one request per model recorded, the off switch present (Qwen: `enable_thinking=False` in the template args
  and the empty `<think>\n\n</think>\n\n`; Nemotron: the `/no_think` system message and `<think></think>`), no think
  content in the outputs beyond the template's empty block. Pilot -- RTN FourOverSix at batch 64 on the 64 questions with
  the most tokens (the full run's longest batch, its memory worst case) plus 64 others (seed 0): memory and time per
  batch, the ETA. Over ~24 h: stop and report the options. A development smoke of the harness (Nemotron, 2 problems)
  ran before this registration and is disclosed (`ptq_gsm8k_lmeval/dev/`).
- **N1:** the 18 runs (`run_gsm8k_lmeval.py run`), one job at a time, Nemotron-Nano-9B-v2 first.
- **Report** (`results/paper_eval/ptq_gsm8k_lmeval/`): strict-match and flexible-extract accuracy ± 2 SE (binomial) per
  configuration (the paper column uses strict-match unless the coordinator says otherwise); paired per-problem
  comparisons with McNemar's exact p (each format vs FourOverSix within a method; GPTQ / Hadamard FlipQuant vs RTN
  FlipQuant), per filter; the generations that used the whole 1024-token budget, the mean / max generated tokens, and the
  answers each filter cannot extract (`[invalid]`); lm-eval's samples files; table_ptq.tex's GSM8K column (strict-match).
- **Checks:** N1 -- no think content (`<think>` / `</think>`) in any output; N2 -- every GPTQ record's codes_sha256 equal
  to part H's; N3 -- every FlipQuant record installs the map part H used (path, modules, E0M3 tiles); N4 -- 1,319
  documents, each with both filters, in every record; N5 -- the native coverage check passed. Stop and report on a
  failing check, an OOM at batch 32, or an implausible result.

### Part N: the run

N1 ran on 2026-10-09 from 17:06:21 to 23:32:50 UTC (registration 8f3fbe8; flipquant paper-sm120-runs f2a56f1), batch 64
on both models, all 18 configurations rc 0, one job at a time; no deviation from the registered method. The completion
watcher of this session waited on its own command line (`pgrep -f` matched itself), so the report ran on 10/10 at
05:16 UTC instead of right after the run; nothing else was affected.
- **Checks:** N2 (GPTQ codes_sha256 = part H's) 6 / 6; N3 (FlipQuant maps = part H's) 6 / 6; N4 (1,319 documents, both
  filters) 18 / 18; N5 (native coverage) 18 / 18; per-problem accuracies equal lm-eval's exact_match exactly.
- **N1 (no think content) 17 / 18 -- reported, a decision for the coordinator:** one output of 23,742 (Qwen3.8-27B
  Hadamard FourOverSix, doc 262) contains a stray `</think>`: the model wrote its answer, the tag, and the same answer
  again ("... The final answer is 270.\n</think>\n\n<same text>"). No reasoning precedes it (the prompt pre-fills the
  empty think block), it ends with EOS at 152 tokens, and both filters extract 270, correct either way.
  **Accepted by the coordinator (2026-10-10); thinking was off, the tag carries no reasoning, the accuracy is
  unaffected. The cell stays as measured; no rerun.**
- **Results:** `results/paper_eval/ptq_gsm8k_lmeval/GSM8K_LMEVAL.md`; tab:ptq's GSM8K column (strict-match) in
  `results/paper_eval/ptq/table_ptq.tex`.

## Part O (registered 2026-10-10 07:34 UTC): round 3 of tab:ptq

The user's request, relayed by the coordinator on 2026-10-10. It follows how IF4 (Fig. 5b), MixFP4 (Table 4, App. C)
and FourOverSix (Table 5) combine their formats with PTQ methods. Scope: Nemotron-Nano-9B-v2 and Qwen3.8-27B at
16x64; the release revisions (`run_sm120.REV`); native evaluation (build_V, `--kernel-set auto`, n16k64-fast); flipquant
paper-sm120-runs; one GPU job at a time; the steps run in this order. Driver: `experiments/paper_eval/run_ptq_round3.py`.
Records are in `RUN/ptq_round3/`; results (one report covering O0-O4) are in `results/paper_eval/ptq_round3/`. Parts H and
N stay unchanged; their GPTQ rows remain the BF16-propagation variant.

- **O0, the BF16 reference row (STEP 0).**
  - **GSM8K.** Part N's protocol exactly (lm-eval 0.4.11 `gsm8k_llama` as shipped, chat template +
    fewshot_as_multiturn, greedy, max_gen_toks 1024, thinking off with lm-eval's own switches, all 1,319 problems,
    the same command), with `--mode bf16` instead of the quantization flags. Nothing is quantized, so there is no
    native coverage check.
  - **Batch.** A pilot on part N's pilot documents per model (the 64 longest questions + 64 others) at batch 64,
    sampling nvidia-smi's device memory every second. A batch b fits if the pilot completes and its peak plus the KV
    growth bound b x 1024 tokens x (64 KiB per token for Qwen3.8-27B: 16 attention layers x 2 x 4 KV heads x 256 x
    2 B; 16 KiB for Nemotron-Nano-9B-v2: 4 layers x 2 x 8 x 128 x 2 B) is at most 97,280 MiB (95 GiB). If it does
    not fit, try 48, then 32, then 16 the same way; the full run uses the largest batch that fits, recorded and
    documented. A full run that still runs out of memory is rerun at the next lower batch, also recorded.
  - **Report.** Strict-match and flexible-extract accuracy ± 2 SE, budget hits, the samples file. Paired per problem
    and filter (McNemar's exact p): each of part N's 18 configurations vs BF16.
  - **PPL** reuses part F's BF16 records (`RUN/pplfast/<model>/bf16.json`). Checked at registration: run by
    n16k64-fast's interpreter, at the release revisions (6533e8d / 1d4bf0f), transformers 5.16.1, torch 2.9.0+cu128
    (the env's current versions; nothing has been installed since), WikiText-2 147 / 145 windows and C4 256.
- **O1, CPU only (STEP 1).** The E0M3 share of the Hadamard-basis FlipQuant 16x64 maps (part H:
  `RUN/ptq/<model>/fq-16x64_hadamard.pt`) and of the unrotated release 16x64 maps, per model and per projection
  type (the module name's last component: q/k/v/o_proj, gate/up/down_proj, the Mamba in/out_proj, the
  Gated-DeltaNet projections), with the tiles that are E0M3 in both. Not all of the difference is the basis: part H
  trained the Hadamard maps in n16k64-fast, and the release maps were trained in n16k64.
- **O2, GPTQ with activation quantization on during calibration (STEP 2; the MixFP4 / FP-Quant setting).**
  - **Settings.** flipquant gptq.py's defaults, passed explicitly: `--gptq-hessian-input quantized`
    (`H = X^T X` from the input after the row's own activation quantizer, per token) and `--gptq-propagate
    quantized` (earlier layers run quantized, as they are evaluated: GPTQ codes and per-token activations, fake
    quant during calibration). Everything else as part H: fixed RTN grid, block 128, damp 0.01, no act-order,
    sequential layerwise, the release 256 x 512 fit set (`--gptq-calib release`), `--gptq-batch` 1 (Qwen) / 8.
  - **Rows.**
    - NVFP4: `--weight nvfp4 --act nvfp4`, per-token NVFP4 activations in calibration and evaluation.
    - FourOverSix: `--weight fourover6 --act fourover6`, per token.
    - FlipQuant 16x64: `--weight mixfp4 --map <release 16x64 map>`, FourOverSix per token. The map is fixed, and each
      tile is GPTQ-rounded on its format's grid (MixFP4's static strategy).
  - **PPL.** Native WikiText-2 / C4 PPL (part H's PPL flags: `--paper-convention` / NVFP4 `--act-scope row`).
    codes_sha256 is recorded; code files are stored under `--codes-root`.
  - **Comparisons.** Paired per-window ΔNLL (± 2 SE) vs RTN FourOverSix (part F's record), and vs part H's
    BF16-propagation GPTQ row of the same format.
- **O3, GPTQ candidates plus a retrained map (STEP 3; FlipQuant's own combination: transform, then select).**
  - **E2M1 candidate.** O2's GPTQ FourOverSix codes.
  - **E0M3 candidate.** A new GPTQ run on the all-E0M3 grid, with O2's settings (activation quantization on,
    per-token FourOverSix): `calibration.gptq_codes --weight mixfp4` with a map that has every tile E0M3. The map
    is written from the release 16x64 map's modules and tile shapes. Its grid is exactly the trainer's candidate A:
    block max / 7 in E4M3 and the global scale shared with FourOverSix.
  - **Map training.** `calibration.tmopt_ext train` with the release settings: TM-OPT+TC, 5 epochs, 256 fit windows,
    top-1000 teacher, lr 0.02, init -1, seed 0, deterministic, batch 2 x accumulation 4. It uses the release runs'
    data root (`fqrel/tmopt_data`). Only the candidates differ from the release map's training.
  - **The new hook (opt-in on paper-sm120-runs; defaults unchanged; no hook = the trainer as before).**
    - **Interface.** `tmopt_launch.py --candidates-e2m1 SRC --candidates-e0m3 SRC`, passed through by `tmopt_ext
      train`. SRC is a flipquant-gptq/1 code file, or `rtn` (flipquant's own RTN codes of the module's BF16 weight).
    - **Mechanism.** Like `--rotate`, it wraps `run_train_map.sha` (called once per module, right before that
      module's candidates are built) to know the current module. It replaces the trainer's candidate quantizers
      (`quant_nvfp4_4over6`, `quant_mix_4_6` in run_train_map) for that module's weight with the given candidates'
      decoded BF16 weights. It also replaces `rq.weight_four_over_six` / `rq.weight_e0m3` (the codes the lean store
      packs) with the given codes. The trainer's own packing check (each stored candidate decodes bit for bit to the
      candidate it holds) runs unchanged.
    - **Guards.** Each module's given scales and global scale must equal the RTN grid of its BF16 weight (GPTQ keeps
      the fixed grid), so codes on any other grid fail. With `rtn`, each candidate must equal the trainer's own
      quantizer output bit for bit. Every quantized module must receive both candidates exactly once.
    - **Metadata.** The map's meta records the sources (path, key, codes_sha256).
  - **Composed evaluation (opt-in in models/cli.py / flipquant/gptq.py).** `--gptq-candidates E2M1 E0M3` with `--ptq
    gptq --weight mixfp4 --map M` runs no GPTQ. Each tile takes its format's GPTQ codes and E4M3 scales (E0M3 tiles
    from the E0M3 file, E2M1 tiles from the E2M1 file), and the global scales must be equal. Both files' GPTQ settings
    (act, damp, block, windows, calibration set and data hash, Hessian input, propagation, rotation) must equal the
    run's; the E2M1 file must be FourOverSix without a map, and the E0M3 file must have every block E0M3. The record
    holds both files' codes_sha256 and that of the composition.
  - **Tests before use.**
    - (a) CPU: the composition (all-E2M1 / all-E0M3 maps give each file's codes; a mixed map gives rtn_grid's mixfp4
      codes for RTN files; mismatched settings or grids are refused).
    - (b) GPU, Qwen3-1.7B, 2 epochs, as tests/test_acceptance_ptq.py (a): the hook with `rtn`, and with RTN code
      files, gives the unhooked trainer's map.pt bit for bit.
    - (c) The two models, release env: `--candidates-* rtn` reproduces the release 16x64 map. The trainer's map.pt
      sha256 must equal the release record's `run_map_sha256`, and the tiles must equal the release map's.
  - **Env for the trainer (both runs per model, the RTN reproduction and the GPTQ-candidate run).** n16k64, the
    release maps' env. The release records show `/home/dev/.conda/envs/n16k64/bin/python`. n16k64-fast's mamba_ssm /
    causal_conv1d / fla kernels change the trainer's forward numerics, so no fast-env run can reproduce the release
    map. This is an exception to "hybrid models run in n16k64-fast"; every evaluation here stays in n16k64-fast.
    **Pending the coordinator's confirmation.** If declined, both runs use n16k64-fast, and (c) becomes: the hook
    with `rtn` equals the unhooked trainer, both in n16k64-fast, bit for bit (one more training run per model), with
    the overlap with the release map reported.
  - **Evaluation.** Native PPL of the retrained map on the GPTQ candidates. Paired ΔNLL vs O2's GPTQ FourOverSix and
    GPTQ FlipQuant (fixed map), and vs RTN FlipQuant (part F). The map's E0M3 share, and its overlap with the release
    map (both, Jaccard, per projection type).
- **O4, GSM8K for the new rows (STEP 4).** O2's three rows and O3's row, both models, part N's protocol at batch 64.
  The ETA goes to the coordinator before it starts. Paired as part N (within a method, each format vs FourOverSix;
  FlipQuant vs RTN FlipQuant), and vs BF16.
- **Checks.**
  - O-1: every GPTQ record has propagate / Hessian input quantized in its key, and every GSM8K record's
    codes_sha256 equals the PPL record's (part N's N2).
  - O-2: the FlipQuant records install the intended map (path, E0M3 tiles).
  - O-3: tests (a)-(c) pass.
  - O-4: part N's N1, N4 and N5 for every GSM8K record (N5 native rows only).
  - O-5: the composed run's file codes_sha256 equal those recorded when the files were written.
- **Stop and report** on a failing check, an OOM beyond the batch rule, an implausible result, or a disk shortfall.
- **Disk (a decision for the user).** A GPTQ code file is 4.3-4.8 GB (Nemotron) or 13.7-15.2 GB (Qwen); O2 + O3
  write eight, about 76 GB. At registration the disk holding `/home/dev` has 27 GB free (99 % used). The driver
  starts a GPTQ job only if its file plus 10 GB fits. Until the user decides where the files go (space freed on
  /home, or another location), only O0, O1 and Nemotron's O2 (about 13.5 GB) can run; the decision is recorded as an
  amendment.
- **ETA (GPU, sequential; from parts H / N).**
  - O0: about 1-1.5 h (Qwen BF16 GSM8K 35-60 min).
  - O2: about 2.5 h (Nemotron about 11 min per row; Qwen about 35-40 min per row).
  - O3: about 4 h in n16k64 (E0M3 GPTQ 11 / 40 min; four trainer runs of about 44 min); about 2.3 h if trained in
    n16k64-fast.
  - O4: about 3 h.
  - In all, about 10-11 h of GPU time, plus waiting for the two decisions above.

### Part O, amendment 1 (2026-10-10 07:39 UTC): the user's two decisions

- **Disk: the GPTQ code files go to the user's vault** (relayed by the coordinator; the user chose it explicitly):
  `/vault/flipquant_paper_eval/ptq_round3/<model>/codes_<row>.pt` (the driver's `--codes-root` default). The report
  records the path and each file's sha256. Nothing on /home is deleted. The first write there (07:38 UTC, a
  `mkdir` plus a one-line write test) was denied by this session's permission classifier. As the coordinator
  instructed, no workaround was tried, and no round-3 code file goes to /home instead. The GPTQ jobs (O2, O3's E0M3
  codes) wait until the user grants writes under `/vault/flipquant_paper_eval/` in this session. The disk guard
  applies to the vault's free space.
- **The trainer env for O3 is n16k64, confirmed by the user.** This is a registered exception to "hybrid models
  run in n16k64-fast", because the release maps were trained in n16k64. It covers the RTN-candidate reproduction
  and the GPTQ-candidate map, two runs per model. Every evaluation stays in n16k64-fast.
- **O1 note** (the coordinator's wording): part H's Hadamard maps were trained in n16k64-fast, so the rotated vs
  unrotated E0M3 shares mix the basis change with the env.
