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

### Part O: the run (written as it goes)

- **The user's /vault grant (07:44 UTC, in this session).** Covers writes under `/vault/flipquant_paper_eval/` for
  part O: create directories, have the GPTQ jobs write their code files there, compute their sha256, and read them back
  for evaluation and map training. Files there may be moved or removed only if they are a partial or failed output of
  a job this session started. Nothing else on /vault is modified, and nothing on /home is deleted.
  `/vault/flipquant_paper_eval/ptq_round3` was created at 07:44 UTC. Order (the user's words): O2 for Nemotron, then
  Qwen, then O3, then O4, after O0. The O3 tool tests the coordinator allowed during the wait were not needed early;
  they run in O3, before its experiment runs.
- **O0 pilot** (`results/paper_eval/ptq_round3/o0_pilot/PILOT.md`, not a result):
  - Nemotron-Nano-9B-v2 fits at batch 64: nvidia-smi peak 55,408 MiB + 1,024 MiB KV bound.
  - Qwen3.8-27B at batch 64 completed (rc 0), but 94,320 + 4,096 MiB > 97,280 MiB, so it does not fit by the rule.
  - Qwen3.8-27B at batch 48 fits: 83,974 + 3,072 MiB.
  - The full BF16 runs therefore use Nemotron 64 and Qwen 48 (started 07:49:51 UTC). Qwen's BF16 reference row uses
    a different batch than part N's quantized rows (64), as the request allows for the reference row; documented
    here and in the report.
- **Code after registration.**
  - Driver:
    - `bf16-pilot` takes `--models`.
    - `--codes-root` defaults to the vault path (amendment 1).
    - New `test-b`: tests/test_acceptance_candidates.py as a GPU job (n16k64). Its outputs, including two RTN code
      files for Qwen3-1.7B of about 0.8 GB each, are test artifacts, not round-3 GPTQ code files; they go under
      `RUN/ptq_round3/test_b`.
  - Report: `report_ptq_round3.py report` builds the round-3 report (O0-O4), with every GPTQ code file's sha256.
  - flipquant paper-sm120-runs **276ce86** (pushed) adds the opt-in options: `--candidates-e2m1 / --candidates-e0m3`
    (tmopt_launch / tmopt_ext train) and `--gptq-candidates` (models/cli, flipquant.gptq.compose_candidates).
    Defaults are unchanged. Test (a), tests/test_gptq_candidates.py, 17 CPU tests, passes: composition; the hook
    driven through run_train_map's lean-store call sequence (`rtn` = the unhooked store byte for byte; code files
    decode to the files' codes); refusal of another grid or swapped files, checked to fail at the intended guard.
    tests/test_gptq.py and tests/test_stage3.py still pass.
- **O0, O2 done.** Every job ran with rc 0 (`results/paper_eval/ptq_round3/REPORT.md`). O2's six code files are on
  /vault, with their sha256 in the report.
- **O3, the E0M3 candidates.** Both GPTQ runs on the all-E0M3 map completed with O2's settings (11:11-12:03 UTC).
- **O3, test (b), first attempt (12:02-12:05 UTC, rc 1).**
  - (g) passed: the hook with `rtn` gave calibration.train_map's map.pt bit for bit (sha256 4f08678a..., 196 modules,
    lean store).
  - (g2) did not run: the test's own helper, which writes the RTN code files, passed a str where `gptq._save` takes a
    Path. That is a bug in the test script, not in the hook.
  - Fixed in flipquant eb10c28 (test file only), and (b) was rerun. Its finished parts (the direct run, (g)) are
    reused by the test's own caching.
- **O3, test (b), passed (12:06-12:07 UTC).** Qwen3-1.7B, 2 epochs:
  - (g): the hook with `rtn` gives calibration.train_map's map.pt bit for bit.
  - (g2): RTN code files give it bit for bit too.
  - Both have sha256 4f08678a...; the files are under `RUN/ptq_round3/test_b/`.
- **O3, test (c), passed (n16k64, 12:07-13:33 UTC).** With `--candidates-* rtn`, both release 16x64 maps are
  reproduced bit for bit. The tiles are equal (80,098 / 105,278 E0M3), and the trainer's map.pt sha256 equals the
  release record's run_map_sha256 (05222ba8... / abb2c598...). Every epoch's train KL, E0M3 count and flips equal the
  release report's, and every module got both candidates exactly once.
- **O3, the experiment, started 13:33 UTC:** the GPTQ-candidate maps (n16k64), then their native PPL.
- **O3, reading /vault.** Nemotron's GPTQ-candidate training (13:33 UTC) loads the two code files memory-mapped
  (flipquant 276ce86's `load_codes_file`). On the FUSE vault, page faults read about 2 MB/s, against about 85 MB/s for
  a full read; 40 trainer threads were waiting on FUSE, and 45 % of the files were resident after about 36 min.
  - The run continues as started: the bytes are the same, it is only slower, and this session kills nothing.
  - flipquant **b5cbc22** (pushed) reads the files in full. Qwen's training and every later evaluation use it; the
    CPU tests (17) still pass.
- **O3, Qwen's GPTQ-candidate training, attempt 1: out of GPU memory** (15:34-15:48 UTC, rc 1, in candidate packing).
  - Cause: with GPTQ codes, run_train_map's RTN pre-packer `pack()` returns None. The trainer's dense fallback then
    keeps both BF16 candidates of every module on the GPU until the candidate loop ends: about 4 bytes per weight, so
    about 97 GB for Qwen.
  - Nemotron's run took the same fallback and fit (about 31 GB). Its map stands. The training reads only the lean
    store, which both paths build with native_dev.pack_candidates from the same codes, checked against the same
    candidate tensors. The fallback holds extra copies only until the loop ends; the trainer then clears them.
  - Fix, flipquant **a53e122** (pushed): with code files, the hook also supplies `pack`. It returns the given codes in
    `packed_candidates.encode`'s layout, checked with the trainer's own decoders (E2M1 bit for bit, E0M3 by value),
    so every module takes the trainer's packed path, as RTN candidates do. With `rtn`, the trainer's own `pack` runs
    unchanged, so test (c) is unaffected. The CPU tests now also check the packed path in all three modes.
  - Test (b) is rerun with this hook in a fresh directory (`RUN/ptq_round3/test_b_b2`): (g) RTN, and (g2) RTN code
    files, which now go through the new `pack`.
  - Qwen's attempt 2 writes `fq-16x64_gptqcand_a2.pt`. tmopt_ext deletes an existing run directory, and nothing on
    /home is deleted, so attempt 1's directory is kept as the record of the failure. The driver maps
    (qwen3.8-27b, gptq) to attempt 2.
- **O3, the store check (the coordinator's request).** Nemotron's GPTQ-candidate map (dense fallback, 276ce86's
  hook) is used only if its lean store equals the one the new `pack` path builds from the same code files.
  - Otherwise Nemotron is retrained as attempt 2 before its PPL.
  - `store_check_round3.py` rebuilds both paths per module and compares the five store tensors byte for byte, with
    per-tensor sha256 and an overall digest:
    - A: 276ce86's Candidates (git show), the trainer's `pack` returning None, then the dense fallback;
    - B: a53e122's Candidates with its `pack`.
  - First attempt, CPU (15:54-15:56 UTC): stopped at the first module. On the CPU, flipquant's RTN FourOverSix grid
    differs from the GPU's in near-tie blocks: the 4-or-6 choice compares summed squared errors, summed in another
    order. The hook's grid guard therefore refused the GPU-made codes.
  - The check now runs as a GPU job: the trainer's own load call, `device_map="cuda"`. The driver's `ppl3` runs it
    first, between Qwen's attempt 2 and the PPL, and stops if the stores differ.
  - Test (b) rerun with a53e122 (`test_b_b2`, 15:51-15:55 UTC) passed:
    - (g) and (g2) both give map.pt 4f08678a...;
    - (g2) now takes the packed path for all 196 modules, with no dense fallback.
- **O3, the store check: passed** (GPU job, 16:42-16:45 UTC). All 120 modules' lean-store tensors are identical byte
  for byte between the dense-fallback path (276ce86) and the hook's `pack` path (a53e122). Both overall digests are
  17f04378...; both total 8,675,574,240 bytes = 8.0798 GiB, the store size the 15:34 run reported. Nemotron's
  GPTQ-candidate map stands; Qwen's is attempt 2 (packed path, no dense fallback).
- **O4 done** (16:54-19:45 UTC, 8 runs, all rc 0, batch 64).
  - Checks: N1 (no think content) 10 / 10 (with O0's BF16); N4 10 / 10; N5 (native coverage) 8 / 8; O-1 (each GSM8K
    record's codes_sha256 = its PPL record's) 8 / 8.
  - Part O is complete: `results/paper_eval/ptq_round3/REPORT.md` and the tab:ptq rows `table_ptq_round3.tex`.
  - Code each run used:
    - The batch-64 pilots ran on f2a56f1.
    - The batch-48 pilot ran on f2a56f1 plus the not yet committed opt-in edits; BF16 mode touches none of them.
    - The full BF16 runs (07:49:51 UTC) and everything after run on 276ce86.

## Part P (registered 2026-10-10 16:13 UTC): Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct in tab:main-ppl

The user's request, relayed by the coordinator on 2026-10-10. Add two models to the calibrated set and to tab:main-ppl,
with every method of that table. Part O is not interrupted: the CPU preparation runs now, and the GPU work starts
after part O, O4 included, unless the coordinator says otherwise. Records go in `RUN/llama_instruct/`, results in
`results/paper_eval/llama_instruct/` plus the updated `results/paper_eval/final/`. Code is on flipquant
paper-sm120-runs (the registry and calibration changes are first on the local branch llama-instruct-dev, 7b472b0, then
merged; nothing vendored is edited) and on paper-eval.

- **Models and access** (the configured token; active account edgeai-lab; no token printed or changed):
  - `llama3.1-8b-instruct`, meta-llama/Llama-3.1-8B-Instruct @ **0e9e39f249a16976918f6564b8830bc894c89659**. Access
    OK. This is the revision of the reference implementation's Instruct calibration record (NVFP4-RaZeR
    `make_instruct_prior.py`), and also the Hub's main today. The vendored trainer knows it as `llama8b_ins`.
  - `llama3.2-1b-instruct`, meta-llama/Llama-3.2-1B-Instruct (registry pin 9213176726f574b556790deb65791e0c5aa438b6).
    **GATED for edgeai-lab: STOPPED** until the user accepts its license on HF. Its steps, its trainer key (the vendored
    trainer does not know it) and its pin check are an amendment before its first run.
- **Calibration data, the release rule: 256 x 512 windows.**
  - The first 128 windows are built by the paper rule: `campaign.data.builder_seed0` with the model's own tokenizer,
    the same math/code streams (open-web-math @ fde8ef8, codeparrot-clean @ 35a59fb), 64 + 64. They are then
    re-tokenized and every token hash is checked. They are **not taken from an archived record**.
  - For Llama-3.1-8B-Instruct the paper rule reproduces Llama-3.1-8B's archived record exactly (documents, offsets,
    token sha256; checked on CPU at registration). That is what `make_instruct_prior.py` assumed ("the Instruct model
    shares the base tokenizer").
  - The next 128 windows come from the extension rule (`fit_extension.py`, seed 20260930) over the same streams. It
    skips the fit documents, the development documents (Llama-3.1-8B's three records for `llama8b_ins`, as the
    vendored trainer defines) and the published C4 evaluation documents (Llama-3.1-8B's), and never touches the
    WikiText or C4 evaluation windows.
  - Every window's token sha256 is recorded (the record and the trainer report's fit extension). For
    Llama-3.1-8B-Instruct the 256 windows are expected to equal base Llama-3.1-8B's release windows; this is checked
    and reported.
  - Records: `calibration.tmopt_common.INSTRUCT_PREPARE`, under the release runs' data root
    (`fqrel/tmopt_data/llama8b_ins`).
- **GPU steps, per model, after part O; the existing main-table rows' exact settings; n16k64; one job at a time.**
  1. **FlipQuant maps, 8x64 / 16x64 / 256x64.** `calibration.train_map` with the release settings: TM-OPT+TC, 256
     windows, 5 epochs, top-1000 teacher, lr 0.02, init -1, eps 1e-12, batch 8, seed 0, deterministic, no
     development set. Time and memory are measured as for the release runs (`fqrel/release.py`: wall time, the
     trainer's phases, device peak), and the fit-extension check is run as the release did.
  2. **Native PPL** (WikiText-2 / C4, all windows; build_V, `--kernel-set auto`): BF16, NVFP4 (per-token NVFP4
     activations, `--act-scope row`), FourOverSix and FlipQuant at each unit (`--paper-convention`).
  3. **IF4 and MixFP4 (Zou), simulated**, as part I: each method's own rules, 1x16, per-token activation scales
     (`--act-method own`).
  4. **GPTQ‡** as part J: gptq.py, BF16 propagation, the release fit set (record + the model's own 16x64 trainer
     report as the extension), block 128, damp 0.01, no act-order; NVFP4 weights with per-token FourOverSix
     activations, native. The codes go to no cache file (disk); codes_sha256 is recorded. If the user later switches
     GPTQ‡ to activation-quantized calibration, these runs are redone with the others.
  5. **FOCUS** as part K: the FOCUS hyperparameters, the release fit set, 8 optimizer steps, deployed as NVFP4 codes,
     native.
  6. **Paired ΔNLL** vs FourOverSix (and FlipQuant vs NVFP4) per cell, with daggers as in part L.
  7. **Mean capped NLL recovery**, the coordinator's definition. Per pair: 100 * min(1, (NLL_NVFP4 - NLL_m) /
     (NLL_NVFP4 - NLL_BF16)) from nll_mean, with no floor, and equal weight over the 14 pairs of the 7 models with
     FOCUS (all but Qwen3.8-27B). A pair with NLL_NVFP4 - NLL_BF16 <= 0 is flagged. Outputs: the per-pair values and an
     updated `table_main_ppl_final.tex` with the two new column groups.
- **Disk (a decision for the user, pending).**
  - /home has 23 GB free. Llama-3.1-8B-Instruct's weights are 16.06 GB (HF cache), Llama-3.2-1B-Instruct's about
    2.5 GB, and an 8B FOCUS state about 5 GB.
  - Nothing is downloaded until the user decides: /vault for the large outputs, space on /home, or both.
  - GPTQ‡ keeps no code file.
- **ETA (GPU, after part O).** About 1 h for Llama-3.1-8B-Instruct and about 25 min for Llama-3.2-1B-Instruct, plus the
  downloads and records, about 20 min. Reports take about 15 min on CPU.
- **Stop and report** on a failing check: the builder rule, the token hashes, the extension check, coverage, an OOM, or
  an implausible result.

### Part P: the run (written as it goes)

- **CPU preparation.**
  - Llama-3.1-8B-Instruct's weights were downloaded at the pin. Its record was built by INSTRUCT_PREPARE (16:16-16:36
    UTC, from the dev worktree at 7b472b0). The paper rule reproduces the base archived record; the record has 224
    matrices.
  - The user freed /home first by deleting three HF-cache models (Qwen3-4B, Mistral-7B-v0.3, Llama-3.1-8B), after this
    session confirmed nothing pending needed their weights.
  - paper-sm120-runs was fast-forwarded to 7b472b0 between O4 jobs (17:03 UTC) and pushed.
- **GPU, Llama-3.1-8B-Instruct** (19:45-20:30 UTC, after O4; one job at a time; all rc 0):
  - maps 3 x about 285 s, device peak about 44.5 GB, E0M3 0.615 % / 0.773 % / 2.218 %, every fit-extension check
    passes;
  - native PPL 6 rows;
  - IF4 and Zou (simulated);
  - GPTQ-double-dagger (BF16 propagation, own 16x64 trainer report as the extension, no code file);
  - FOCUS (micro-batch 8, 247 s).
  - The 256 windows equal base Llama-3.1-8B's release windows, token for token: the 128 of the rule and the 128 of
    the extension.
- **Code.** Driver change: `maps` runs train_map through `run_fresh_dir`, because train_map refuses an existing
  `--out`.
- **A note, not a rerun.** GPTQ-double-dagger's C4 cell for Llama-3.1-8B-Instruct is driven by one window (C4 window
  219: +2.32 nats vs FourOverSix). Without it the C4 ΔNLL vs FourOverSix is -0.23 x 1e-3. This model's C4 windows
  are volatile in general: FourOverSix differs from BF16 by up to 1.1 nats on single windows.
- **Llama-3.2-1B-Instruct:** not run; the license is not accepted for edgeai-lab.
- **Results:** `results/paper_eval/llama_instruct/REPORT.md`; the final table is regenerated
  (`results/paper_eval/final/`, the Llama-3.2-1B-Instruct column TBD). The mean capped NLL recovery is over 12
  common pairs until the 1B is in.

## Part Q (registered 2026-10-10 22:10 UTC): FOCUS on Qwen3.8-27B from the trained state on HF

The user's request, relayed by the coordinator on 2026-10-10 (GPU priority). The goal is to fill the main table's FOCUS
Qwen3.8-27B cells (TBD in part K: FOCUS did not fit one 96 GB GPU for training). The state was trained on the cluster
and is on HF: `edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4` @ b84ff372a7, `focus_256/focus.pt` (18.26 GB) plus
`focus_256/report.json` and `focus_256/fit_extension.json`.

- **Provenance checks before use** (read-only; stop on any difference):
  - **Windows.** The state's 256 windows' token sha256 (report.json data records, fit_extension.json) must equal
    the release fit set for qwen3.8-27b, i.e. the windows of the release map and of part K's FOCUS runs. The windows are
    rebuilt here with flipquant's `calibration_release` (the record plus the release 16x64 trainer report's extension)
    and hashed the same way.
  - **Hyperparameters.** Every one must equal part K's (`run_sm120.FOCUS_COMMON`): data release, 256 x 512, 1 epoch,
    batch 32 (8 steps), lr 5e-3 / 1e-3, num_sub 2, init_q 6, top-1000, act fourover6 / row, seed 42, revision 1d4bf0f2.
  - **Recorded deviation:** micro-batch 1 on 2 x H200 with device_map balanced. The global batch is the same; only the
    summation order differs.
  - The HF repository sha and the file's sha256 are recorded.
- **Download** into the HF cache on /home. The coordinator relaxed the floor to 10 GB now that O4 is done; stop if the
  download would cross it.
- **Evaluation:** exactly part K's, `evaluation.ppl --weight focus --focus <path> --focus-deploy --act fourover6
  --act-scope row --paper-convention`, native (build_V, `--kernel-set auto`), n16k64-fast, the release revision. The
  record goes to `RUN/focus/qwen3.8-27b/ppl.json` (where tables_final.py reads FOCUS). If it does not fit one GPU in
  the deploy path, stop and report; no setting is changed.
- **Report:**
  - paired ΔNLL vs FourOverSix and vs FlipQuant at 3 units;
  - `results/paper_eval/final/` regenerated with the FOCUS Qwen3.8-27B cells and the mean capped NLL recovery over
    the new common pairs (every model with all methods: 7 models = 14 pairs until the 1B is in), flagging any
    non-positive gap, with the per-pair values.
  - Records: `results/paper_eval/focus_qwen27b/`.

### Part Q: the run (2026-10-10)

- **Provenance (CPU, before the download):** every check passes. The state's 256 window token sha256 are identical, in
  order, to the release fit set (record 128 + the release 16x64 trainer report's 128 extension windows); the
  fit_extension records and settings are equal; every part-K hyperparameter is equal. The deviation is recorded: micro 1,
  device_map balanced, 2 x H200, job 519366. KL-top 0.0521 -> 0.0480.
- **Download:** `focus_256/focus.pt` (18,263,340,091 bytes) into the HF cache on /home in 1238 s; sha256 `baf02d9c...`
  equal to the Hub's LFS sha256. /home free afterwards: 17 GB (floor 10 GB).
- **Evaluation:** part K's command, n16k64-fast, build_V `--kernel-set auto` (stock_ko for the NVFP4 codes), 496 / 496
  modules. It fits one GPU: 116 s, torch peak 83.5 GiB. WikiText-2 7.2746 (145 windows), C4 10.1572 (256 windows),
  between BF16 and NVFP4 on both.
- **Paired ΔNLL x 1e-3 (± 2 SE), FOCUS minus:**
  - FourOverSix: -3.80 ± 4.78 / -3.25 ± 1.21*;
  - FlipQuant 8x64: +18.69 ± 4.62* / +1.29 ± 1.08*;
  - FlipQuant 16x64: +12.27 ± 4.36* / +1.60 ± 1.09*;
  - FlipQuant 256x64: +7.61 ± 4.16* / -0.40 ± 1.12.
- **Final table:** regenerated with the FOCUS Qwen3.8-27B cells. The mean capped NLL recovery is now over 14 common
  pairs, with no non-positive gap: FlipQuant 8x64 47.2 %, 16x64 46.1 %, FOCUS 41.6 %, 256x64 40.8 %, IF4 20.9 %,
  Zou 19.3 %, FourOverSix 9.5 %, GPTQ‡ 4.1 %.
- **Records:** `results/paper_eval/focus_qwen27b/` (REPORT.md, focus_qwen27b.json, records/).

## Part P, amendment 1 (registered 2026-10-10 22:42 UTC, commit f0fb41c): Llama-3.2-1B-Instruct

The coordinator relayed this on 2026-10-10, after part Q. The edgeai-lab account now has access to
meta-llama/Llama-3.2-1B-Instruct (checked read-only; no identity switch). Run it in part P with the same methods and
settings as Llama-3.1-8B-Instruct, then regenerate `results/paper_eval/final/` with all 8 models. Keep the /home floor
at 10 GB. Do not upload its artifacts to HF until a request is relayed. One GPU job at a time, n16k64.

- **Pin check (read-only, done).** The registry pin `9213176726f574b556790deb65791e0c5aa438b6` (2024-10-24, "Update
  model card (#25)") is the Hub's main today, and the model is accessible (gated: manual, accepted). The pin is kept.
- **Trainer key `llama1b_ins`, registered at run time.** flipquant paper-sm120-runs aa12408 (first on
  llama-instruct-dev, then fast-forwarded and pushed). No vendored file is edited. Neither the vendored trainer nor
  prepare_model_data knows the model, so it is registered at run time as the release's models without an archived
  record (Phi-4, Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2):
  - The preparation registers it in prepare_model_data.MODELS, in its own process, with no record.
  - The builder rule's three checks run first. They must reproduce the Llama-3.1-8B, Qwen3-4B and Mistral-7B-v0.3
    records exactly. They need those three tokenizers, which are re-downloaded (configs and tokenizer files only, no
    weights).
  - Then `campaign.data.builder_seed0` with this model's tokenizer draws the 128 fit windows. They are re-tokenized and
    every token hash is checked. The matrices are this model's weights.
  - `prepare_development` draws its three development records (fresh_dev1..3, the Llama development rule; this model's
    C4 evaluation documents and earlier draws are excluded), as for those models.
  - calibration.train_map runs the trainer through `tmopt_launch.py --register-model llama1b_ins`. That adds the key to
    run_multiround.LOCAL_DEVELOPMENT (the three draws) and MODELS. PUBLISHED is left alone: there is no published C4
    record for this model, so the trainer records the evaluation windows by token hash.
  - Tests: tests/test_stage3.py (13 passed, 1 skipped). Nothing changes for any other model.
- **Calibration windows: 256 x 512.**
  - The 128 of the paper rule, as above.
  - The 128 of the extension rule (seed 20260930). It skips the fit documents, the 192 drawn development documents,
    and no published C4 documents. The release extension check applies, with `skip_c4_published` = 0 documents.
  - Every window's token sha256 is recorded. Comparing them with base Llama-3.1-8B's release windows is reported only,
    not required: the 1B's development documents differ, so its extension may differ.
- **Steps, exactly as for Llama-3.1-8B-Instruct (part P steps 1–7; `run_llama_instruct.py`):**
  - the record (CPU: weights 2.48 GB at the pin; `original/` is not downloaded);
  - maps 8x64 / 16x64 / 256x64 with the release settings, measured, with the extension check;
  - native PPL: BF16, NVFP4, FourOverSix, FlipQuant x 3;
  - IF4 and MixFP4 (Zou), simulated;
  - GPTQ‡ (BF16 propagation, its own 16x64 trainer report as the extension, no code file);
  - FOCUS (part K's settings, micro 8 halved on OOM);
  - the windows record;
  - then tables_final.py (all 8 models) and report_llama_instruct.py.
- **Final table.** The mean capped NLL recovery is over the 16 common pairs (8 models x 2 corpora; every model has
  FOCUS). Any non-positive gap is flagged, and the per-pair values are reported.
- **Disk.** /home has 17 GB free. Expected use: about 2.5 GB of weights, about 25 MB of tokenizers, a FOCUS state of
  about 0.7 GB, and small records. Stop if a step would cross 10 GB.
- **Stop and report** on a failing builder-rule, token-hash, extension or coverage check, an OOM, or an implausible
  result.
- **ETA.** CPU preparation about 10 min; GPU about 25 min; the report about 10 min.

### Part P, amendment 1: the run (2026-10-10 22:42–23:01 UTC)

All steps rc 0, one job at a time, from flipquant aa12408 (clean).
- **CPU preparation (304 s).**
  - The builder rule reproduces the Llama-3.1-8B, Qwen3-4B and Mistral-7B-v0.3 records (checked in both preparation
    passes).
  - 128 fit windows, every token hash checked; 112 matrices.
  - Three development draws: 192 distinct documents; this model's 231 C4 evaluation documents excluded.
  - The fit set equals Llama-3.1-8B's archived record (documents, offsets, token sha256): the Llama 3.2 tokenizer
    gives the same windows.
- **Maps.** About 82 s each, device peak about 15.8 GB, E0M3 2.030 % / 2.805 % / 11.377 %. Every fit-extension check
  passes: 128 windows, 64 + 64, skips of fit 128 / development 192 / published C4 0, distinct, disjoint, the same
  extension in all three units. The trainer log shows the launcher's REGISTER line.
- **The extension shares no window with Llama-3.1-8B's** (reported only, not required). All 128 of the base model's
  extension documents are among this model's drawn development documents: the development rule takes the next
  documents of the same streams, and the extension rule skips them.
- **PPL (WikiText-2 / C4):**

  | row | WikiText-2 | C4 |
  |---|---:|---:|
  | BF16 | 13.1629 | 18.5189 |
  | NVFP4 | 15.5352 | 21.7669 |
  | FourOverSix | 15.3152 | 21.4488 |
  | IF4 | 15.3448 | 21.0737 |
  | MixFP4 (Zou) | 15.2483 | 21.0489 |
  | GPTQ‡ | 15.5327 | 21.0190 |
  | FOCUS | 14.6611 | 19.8847 |
  | FlipQuant 8x64 | 14.4527 | 19.6791 |
  | FlipQuant 16x64 | 14.4536 | 19.6842 |
  | FlipQuant 256x64 | 14.6573 | 20.0588 |

  - FlipQuant is significantly better than NVFP4 and FourOverSix in every cell (no dagger).
  - FOCUS (micro 8, 44 s training; state 697 MB) minus FlipQuant, paired ΔNLL x 1e-3:
    - 8x64: +14.32 ± 2.56* / +10.40 ± 1.90*;
    - 16x64: +14.26 ± 2.35* / +10.14 ± 1.77*;
    - 256x64: +0.26 ± 2.70 / −8.72 ± 2.09*.
- **A note, not a rerun.** GPTQ‡'s WikiText-2 cell is driven by one window, as for the 8B's C4 cell. Window 88 is
  +2.73 nats vs FourOverSix (NLL 6.20 vs 3.47; BF16 3.34). Without it, ΔNLL vs FourOverSix is −5.30 x 1e-3 instead of
  +14.10 ± 39.30.
- **Final table.** All 8 models, nothing TBD. Mean capped NLL recovery over 16 pairs, with no non-positive gap:

  | row | recovery |
  |---|---:|
  | FlipQuant 8x64 | 47.9 % |
  | FlipQuant 16x64 | 47.0 % |
  | FOCUS | 42.1 % |
  | FlipQuant 256x64 | 41.1 % |
  | IF4 | 20.0 % |
  | Zou | 18.9 % |
  | FourOverSix | 9.5 % |
  | GPTQ‡ | 5.0 % |

  FlipQuant vs NVFP4 is significant in 48 / 48 cells; 0 daggers.
- **Disk.** /home has 14 GB free after the run.
- **Results.** `results/paper_eval/llama_instruct/REPORT.md` (both models; the largest single-window differences for
  both corpora) and `results/paper_eval/final/`. No HF upload.

## Part R (registered 2026-10-11 04:37 UTC, commit 78b0805): FourOverSix's tensor-scale constant, 6 x 448 vs the original 6 x 256

The user's request, relayed by the coordinator on 2026-10-11 (go given).

**Why.** The original FourOverSix paper (arXiv:2512.02010, Sec. 3) sets the tensor scale to amax / (6 x 256), not
amax / (6 x 448). 256 is the largest E4M3 value whose 6/4 multiple (384) is still an E4M3 value, so every block can map
its max onto 4. Our FourOverSix (flipquant `formats.GLOBAL_QMAX` = 6 x 448; `quant_act.cuh`'s 1/2688) and RaZeR's
reimplementation use 6 x 448. With 448, a block whose max exceeds 2/3 of the tensor amax cannot take a true 4: its
4-candidate scale is clamped at 448. So our "FourOverSix" rows, and FlipQuant's E2M1 candidate, are not the original
definition. This part measures how much that matters. **No FlipQuant map is recalibrated here**; that decision waits
for these results.

- **Code** (flipquant paper-sm120-runs; first on a local dev branch, then fast-forwarded and pushed). No vendored file
  is edited, and build_V's binaries and tables are untouched: no rebuild.
  - `flipquant.formats.fourover6(x, scope, gscale, tensor_qmax=448)`: the global scale is amax / (6 x tensor_qmax).
    - 448 divides by the same 2688.0 as today, so the default is bit-identical by construction. 256 is the original
      definition.
    - The E4M3 block-scale clamp stays [2^-9, 448]. With 256, a block scaled to 4 reaches at most 384.
  - `--fo6-tensor-qmax Q` (models/cli.py; default 448) is accepted only with `--weight fourover6 --act fourover6`.
    It is refused with maps (mixfp4), GPTQ, rotation or ModelOpt checkpoints. A non-default Q is recorded in the
    report.
    - Fake mode: `QuantSpec.fo6_tensor_qmax` applies to the weights and to the per-token activations.
    - Native sm120: the weights are flipquant's FourOverSix codes at Q, packed by `models.sm120.pack_codes`. Its check:
      the packed weight decodes to the codes' fake-quant weight bit for bit.
    - Native sm120 activations: a NativeLinear subclass in models/sm120.py runs the vendored Triton per-token
      quantizer (`mixfp4_sm120.quant_act`, the vendored fallback of the fused quantizer, bit-identical to it) with
      the reciprocal fp32(1/(6 x Q)), i.e. fp32(1/1536), in place of fp32(1/2688). It then makes the same build_V GEMM
      call as NativeLinear's unfused branch.
    - Why not the fused quantizer: the CUDA quantizer compiled into build_V's libraries has 1/2688 fixed in its host
      launcher, so it serves only Q = 448.
    - With Q = 448 nothing changes: pack_module and the fused NativeLinear run as now.
- **Tests, before any run** (sm_120 GPU; build_V):
  - (t1) The default is bit-identical.
    - The new `fourover6` (default, and with tensor_qmax=448 explicit) equals aa12408's formats.py (loaded from git)
      bit for bit for every scope (tensor, row, document, static), on random and real-weight tensors.
    - The existing tests pass unchanged: test_formats, which compares bitwise with the vendored reference
      quantizers, and the sm120 native tests.
  - (t2) The unfused route at 448 (the subclass with fp32(1/2688)) gives outputs bit-identical to the vendored fused
    NativeLinear over several shapes and token counts. So the 256 path differs from the default only by the constant.
  - (t3) The 256 path matches a reference implementation of the original FourOverSix, written in the test from the
    paper's definition, independent of formats.py:
    - fake values bit for bit (tensor and row scopes);
    - the native quantizer's activations, decoded in fake order, equal the fake ones bit for bit;
    - the packed weights decode to the fake weights bit for bit;
    - the native layer's output equals fake mode's within the existing native-vs-fake tolerance. This one is by value,
      because the GEMM accumulation order differs.
  - (t4) With 256, no block scale exceeds 384, and no block's 4-candidate is clamped.
- **Runs.** One GPU job at a time; native; build_V; `--kernel-set auto`; `--paper-convention`; the pinned
  revisions; all windows; the hybrids in n16k64-fast, the others in n16k64.
  - FO6-256 on all 8 main-table models.
  - FO6-448 with the same code at its default, on all 8.
    - Nemotron-Nano-9B-v2, Qwen3.8-27B and the two Llama-Instruct models: the rerun must reproduce the existing FO6
      records (parts F and P) per window, bit for bit. This is the end-to-end default check; stop if it fails.
    - Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3 and Phi-4: the table's FO6 records come from the earlier sm120
      evaluation harness (main-ppl / paper; "NativeLinear (c)", auto_stock). The rerun is the same-code reference for
      the paired ΔNLL, and its per-window agreement with the table's records is reported.
  - The WikiText-2 / C4 windows are those of the existing records: counts asserted, token hashes compared where
    recorded.
- **Statistics.**
  - Paired ΔNLL (FO6-256 − FO6-448) ± 2 SE per model and corpus, against the same-code FO6-448. For the four
    non-hybrid models, also against the table's FO6 record.
  - FO6-256 vs NVFP4 (table records).
  - FlipQuant 16x64 (table records; maps calibrated with 448) vs FO6-256: is FlipQuant significantly better in every
    cell?
  - The mean capped NLL recovery of FO6-256 over the 16 pairs, with the table's BF16 and NVFP4, next to FO6-448's.
- **CPU statistics** (no GPU), per model, over its quantized linears' BF16 weights (models.loader.quantizable_linears):
  - the share of 16-element blocks whose max exceeds 2/3 of the tensor amax (448 cannot select a true 4 there);
  - the shares of blocks choosing 4 under 448 and under 256, and the share whose 4-vs-6 choice differs. They come from
    a replica of `fourover6`'s decision, checked against `fourover6`'s codes on every tensor.
  - Activations: at least one block per token by construction (the block holding the row max), i.e. ≥ 16/K, listed
    per K.
  - These are CPU values. formats' FourOverSix can differ between CPU and GPU in rare near-tie blocks (part O).
- **Disk.** No downloads (all 8 models are cached); records only; floor 10 GB (14 GB free).
- **Stop and report** on a failing test, a default-path mismatch, an OOM, or an implausible result.
- **Records.** `results/paper_eval/fo6_tensor_scale/`.
- **ETA.** About 3 h:
  - code and tests about 1.5 h;
  - GPU: 16 PPL runs, about 35 min;
  - CPU statistics, about 45 min, in parallel;
  - the report, about 20 min.

### Part R: the run (2026-10-11 04:37–06:13 UTC)

- **Code:** flipquant paper-sm120-runs d92c032 (dev branch fo6-tensor-qmax-dev, fast-forwarded and pushed). No vendored
  file or build changed.
- **Tests:** tests/test_fo6_tensor_qmax.py, 18 passed on sm_120 with build_V. Everything registered as t2–t4 holds;
  the unfused route at fp32(1/2688) equals the fused NativeLinear on 4 shapes x 5 token counts.
- **Existing suites, as before:**
  - test_formats against the reference quantizers ($REFERENCE_ROOT = the NVFP4-RaZeR checkout): 7 passed.
  - test_linear, test_stage3, test_gptq, test_focus, test_if4, test_zou_mixfp4, test_rotate, test_fake_cache_model and
    test_mapcheck pass.
  - test_sm120_native: no build directory here holds both the paper builds and the optimized sets. On build_T, 10 pass
    and 3 fail for missing optimized sets, the same 3 as on aa12408.
- **t1 (check-default):** the new fourover6 at its default, and with tensor_qmax=448, equals aa12408's formats.py bit
  for bit. 128 cases: every scope, CPU and GPU, 4 random and 4 real weight tensors.
- **Runs:** 16 native PPL runs, all rc 0 (04:48–05:02). The FO6-448 reruns reproduce the table's FourOverSix records
  per window, bit for bit, for all eight models. That includes the four whose records came from the earlier sm120
  harness, so the harnesses agree.
- **FO6-256 vs FO6-448** (paired ΔNLL x 1e-3, wiki / C4):

  | model | wiki | C4 |
  |---|---:|---:|
  | Qwen3-1.7B | −34.28* | +13.63* |
  | Qwen3-8B | −3.62* | −4.25* |
  | Mistral-7B-Instruct-v0.3 | −0.36 | −1.16 |
  | Nemotron-Nano-9B-v2 | +1.37 | +2.88* |
  | Phi-4 | +0.06 | +0.37 |
  | Qwen3.8-27B | +1.83 | −0.68 |
  | Llama-3.2-1B-Instruct | −0.42 | −3.57* |
  | Llama-3.1-8B-Instruct | −6.42* | +1.21 |

  - 256 is better in 5 / 16 cells and worse in 2 / 16.
  - Mean capped NLL recovery: 9.5 % → 11.7 %.
  - FlipQuant (448 maps, not recalibrated) is significantly better than FO6-256 in 16 / 16 cells at each unit.
  - FO6-256 vs NVFP4: better in 13 cells, not significant in 2 (Qwen3-1.7B wiki, Nemotron C4), worse in 1
    (Qwen3-1.7B C4, +7.54*).
- **CPU statistics** (8 models, 04:48–06:13; Qwen3.8-27B in a second process):
  - weight blocks above 2/3 of the amax: 0.0003–0.011 %;
  - blocks changed: 92–95 %;
  - 4-vs-6 choice flips: 18.3–19.2 %;
  - blocks choosing 4: about 44–45 % under both constants;
  - subnormal block scales: 0, except Nemotron (0.024 % / 0.025 %).
- **Results:** `results/paper_eval/fo6_tensor_scale/` (REPORT.md, fo6_tensor_scale.json, records/).

## Part W (registered 2026-10-13 10:16 UTC, at its commit): A's prefill latency for Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct

The user's request, relayed by the coordinator on 2026-10-13. tab:latency replaces its two Qwen3 columns with the two
Llama Instruct models of tab:main-ppl.
- **Protocol: A's, unchanged except the model list** (this file, "A: prefill latency", and amendments 1-3).
  - Kernels build_V; flipquant `evaluation.latency prefill`; CUDA graphs and eager in every process.
  - 8 shapes: 1x128, 1x256, 1x512, 1x1024, 1x2048, 1x4096, 1x8192, 4x2048; 7 repetitions.
  - 7 policies: bf16, nvfp4, fo6, fo6-wB, fq-8x64, fq-16x64, fq-256x64, with A's flags (run_gpu.POLICIES).
  - 5 rounds; one process per (model, policy, round). Round r's order is `random.Random(20260928 + r).shuffle` of the
    run's policy list, A's rule.
  - One job at a time on an idle GPU; no clock locking, no ncu, no reboot.
  - A's checks stop the run: rc 0 and a complete record; graph logits equal to eager's; the policy's kernel family;
    every library from build_V (run_gpu.check_fq_record).
- **Models.**
  - Llama-3.2-1B-Instruct: registry key llama3.2-1b-instruct, revision 9213176726.
  - Llama-3.1-8B-Instruct: llama3.1-8b-instruct, 0e9e39f2.
  - Both run in n16k64 (not hybrid). Their weights are in the HF cache at these revisions (checked: 2.4 GB / 15 GB).
- **Maps.** Part P's, under `RUN/llama_instruct/<model>/maps/flipquant_<unit>.pt`; each file's sha256 equals the one part
  P recorded:
  - 1B: 8x64 4f693ed8..., 16x64 d9dfa5a7..., 256x64 54effb55...;
  - 8B: 8x64 3bf00ea6..., 16x64 502ec6cf..., 256x64 7582c4da....
- **Code: flipquant paper-sm120-runs 6f5ad27** (worktree sm120runs/wt, clean), not A's 120173a. This is forced by the
  model list:
  - 120173a's registry has no llama3.1-8b-instruct, and its llama3.2-1b-instruct is unpinned.
  - 6f5ad27 pins both (part P).
  - The timed path is unchanged: `evaluation/latency.py` is byte-identical to 120173a's, and so is the kernel tile
    table (`kernels/sm120/configs/`).
  - `models/sm120.install`'s default branch (no precomputed codes, no rotation, no FourOverSix tensor-scale override)
    calls the same `pack_module` and `NativeLinear`; 6f5ad27 only adds opt-in branches.
- **Session-drift check.**
  - Run: Mistral-7B-Instruct-v0.3 (mistral-7b, A's cached snapshot c170c708, n16k64, the release maps), policies fo6
    and fq-16x64, all 8 shapes, 5 rounds, the same rule (the shuffle of the two-policy list).
  - Report: each (policy, shape) cell's median over rounds against A's value (58fe644), and the fq-16x64 / fo6 paired
    ratio against A's.
  - It is a check only: A's published Mistral numbers are not replaced. It covers the session and the code change
    above together.
- **Order.** The Mistral check first, then Llama-3.2-1B-Instruct, then Llama-3.1-8B-Instruct.
- **Outputs.** Records `RUN/latency_llama/<model>/<policy>/round<r>.json` (new names; A's records are untouched). Report
  `results/paper_eval/latency_llama/` (LATENCY.md + latency.json, A's format).
  1. Per new model: ms tables and paired ratios vs fo6 per shape, median [min, max] over rounds, plus the 8-shape median
     summary.
  2. A combined 1x2048 table for the six paper models: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Mistral, Nemotron,
     Phi-4, Qwen3.8-27B.
     - The old models' values come from 58fe644, the new ones from W.
     - Rows: bf16, nvfp4, fo6, fq-16x64, fq-8x64, and fq-256x64 as an extra.
     - Raw ms, plus a LaTeX snippet rounded to one decimal from the raw values.
  3. Over the six models:
     - fq-16x64 vs fo6 and fq-8x64 vs fo6: the range over all shapes and models, and per-model medians;
     - at 1x2048: the range of the paired ratios and of the ratio of medians;
     - BF16 / fq-16x64 at 1x2048.
- **Rules.** /home stays above 10 GB free; nothing is deleted; retries get new output names. The registration and the
  results are committed on paper-eval and pushed (that branch only); no HF upload.
- **ETA.** About 50 min of GPU (A's per-process times: about 20 s for a 1.7B, about 45 s for an 8B), plus the report.

### Part W: the run (2026-10-13 10:17–10:57 UTC)

- **Code.** flipquant paper-sm120-runs 6f5ad27 (checked clean at the start), build_V, n16k64.
- **Runs.** 80 processes, every one rc 0 and passing A's checks: complete record, graph logits equal to eager's, the
  policy's kernel family, build_V libraries.
  - Mistral check: 10 processes, 37–48 s each.
  - Llama-3.2-1B-Instruct: 35 processes, 14–17 s each.
  - Llama-3.1-8B-Instruct: 35 processes, 36–54 s each.
- **Drift check.** The first process (Mistral fo6 round 1, after a long-idle GPU) was 4–6 % faster than A at the
  larger shapes; later rounds rose to A's level (warm-up).
  - Its medians over rounds differ from A's by −0.04 to −1.14 % (median −0.36 %, 16 cells).
  - The fq-16x64 / fo6 ratio's median over shapes is +0.88 % (A: +0.82 %).
  - A's Mistral numbers stand.
- **Push.** This session's push of the registration (038aaf4) was denied by its permission classifier; the user
  pushed paper-eval (59114fb..4043bc7).
- **Report.** results/paper_eval/latency_llama/ (LATENCY.md, latency.json; report_latency_llama.py).


## Part X (registered 2026-10-14 10:33 UTC, at its commit e5b5cdc): six zero-shot tasks, six models, ten rows, through flipquant main

The user's request, relayed by the coordinator on 2026-10-14. Every cell is a new run (BF16 included); nothing earlier
is reused as a result.
- **Tasks.** piqa, hellaswag, winogrande, arc_easy, arc_challenge and lambada_openai, all 0-shot, lm-eval 0.4.11, the
  paper's settings. The primary metric is acc_norm for PIQA, HellaSwag and the two ARC tasks, and acc for WinoGrande
  and LAMBADA. Batch 16 (Qwen3.8-27B: 8); per-token activations.
- **Models**, each at its main-table revision, passed as `--revision` to every job:
  - Llama-3.2-1B-Instruct: llama3.2-1b-instruct @ 9213176726f5;
  - Mistral-7B-Instruct-v0.3: mistral-7b @ c170c708c41d;
  - Llama-3.1-8B-Instruct: llama3.1-8b-instruct @ 0e9e39f249a1;
  - Nemotron-Nano-9B-v2: nemotron-nano-9b-v2 @ 6533e8de2c68;
  - Phi-4: phi4-14b @ 2db69c1c3e91;
  - Qwen3.8-27B: qwen3.8-27b @ 1d4bf0f2ff60.
  - Run order: small first, Qwen3.8-27B last, i.e. the order above.
  - Nemotron and Qwen3.8-27B run in n16k64-fast; the others run in n16k64, as every main-table record did
    (amendment 3). The two envs differ only in the hybrid kernels (mamba-ssm, causal-conv1d, fla), which the other
    models never call; lm-eval, transformers, torch and datasets are the same.
- **Rows** (one process per model and row; `evaluation.downstream --tasks <the six>`; native = `--mode native
  --kernel-build-dir build_V --kernel-set auto`, the RTX PRO 6000):
  1. **bf16:** `--mode bf16`.
  2. **nvfp4:** native, `--weight nvfp4 --act nvfp4` (NVFP4 per-token activations).
  3. **fo6:** native, `--weight fourover6 --act fourover6`.
  4. **focus:** native, `--weight focus --focus <state> --focus-deploy --act fourover6 --act-scope row` (the deployed
     NVFP4 codes with FourOverSix activations).
     - States: part K's (`RUN/focus/<model>/state/focus.pt`) for Mistral, Nemotron and Phi-4; part P's
       (`RUN/llama_instruct/focus_states/<model>/state/focus.pt`) for the two Llama models.
     - Qwen3.8-27B: part Q's file, downloaded again (edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4 @ b84ff372a7cc,
       focus_256/focus.pt). It must match part Q's record: 18,263,340,091 bytes, sha256 baf02d9c..., the Hub's LFS
       sha256. Script: zs6_download.py; the cached token is never printed or logged.
  5. **gptq (GPTQ‡):** native, `--weight nvfp4 --act fourover6 --any-act` plus part J's GPTQ flags
     (run_sm120.gptq_flags):
     - `--ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release` with the model's record and
       release trainer report; block 128, damp 0.01, `--gptq-batch 8` (Qwen3.8-27B: 1);
     - `--gptq-cache` = the main-table codes file. Mistral and Phi-4: `RUN/gptqdd/<model>/codes_nvfp4_act-fo6.pt`.
       Nemotron and Qwen3.8-27B: `RUN/ptq/<model>/codes_nvfp4_act-fo6.pt` (their J cells are H's runs).
     - The two Llama models have no codes file (part P computed theirs in-process). Step 0 regenerates them with main's
       GPTQ into `RUN/zeroshot6/gptq/<model>/codes_nvfp4_act-fo6.pt`; their codes_sha256 must equal part P's.
     - Every run's `ptq.codes_sha256` must equal the main-table record's.
  6. **if4:** `--mode fake --weight if4 --act-method own --unit 1x16`.
  7. **zou:** `--mode fake --weight zou_mixfp4 --act-method own --unit 1x16`. Rows 6 and 7 are simulated with each
     method's own weight and activation rules, as in tab:main-ppl (part I).
  8. **fq-8x64 / fq-16x64 / fq-256x64:** native, `--weight mixfp4 --act fourover6 --map <map>`.
     - Maps: the release maps (`/home/dev/flipquant_release/<model>/flipquant_<unit>.pt`) for Mistral, Nemotron, Phi-4
       and Qwen3.8-27B. Their sha256 equals the release record (`summary.json` map_sha256) for all 12 files (checked
       at registration).
     - The two Llama models: part P's maps (`RUN/llama_instruct/<model>/maps`), with the sha256 of part P's record.
     - The driver re-hashes every map, state and codes file before its run.
- **Code: flipquant origin/main 3db0072** (fetched 2026-10-14), in a fresh detached worktree
  `/home/dev/n16k64_campaign/zeroshot6/wt`, used unmodified; flipquant's `main` branch is not touched. Kernels:
  build_V. Env for every job: run_gpu.env() plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the paper's step 04 environment (paper_common.env and
  ACCURACY_ENV).
- **Deviation 1, registry.** main's registry has no llama3.1-8b-instruct (paper-sm120-runs 7b472b0 added it; main
  never merged it).
  - Its jobs run through `zs6_fq.py <module> ...`, a launcher outside the flipquant checkout. It adds 7b472b0's entry
    to `models.registry.MODELS` at run time and then runs the module as `python -m` would:
    `ModelSpec("llama3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f2...")`, every other field
    at its default, as in 7b472b0.
  - Every other model runs `python -m evaluation.downstream` directly.
- **Deviation 2, Mistral and Phi-4 revisions.** main leaves them unpinned. The jobs pass the main table's revisions
  (above), which are also the cached snapshots.
- **Step 0: code identity, before any panel run.** Any failed check stops part X and is reported.
  - **(a) Settings diff.** A written comparison, in results/paper_eval/zeroshot6/STEP0.md, of flipquant
    `evaluation/downstream.py` against NVFP4-RaZeR `run_lmeval_deploy.py` + `experiments/paper/04_downstream.py`
    (kernel-opt 9580a2a, the clean main checkout). It covers:
    - the lm-eval version and the HFLM arguments (batch, BOS, backend);
    - the dataset pins (`lm_eval_datasets.py` sha256, and flipquant's extra lambada_openai pin);
    - few-shot counts, primary metrics and the activation scope;
    - model loading and revisions.
    - Every difference is listed. LAMBADA has no primary metric in `run_lmeval_deploy.py`; only flipquant's pin of
      its dataset is checked (EXTRA_PINS, and the pinned revision in every LAMBADA load).
  - **(b) Same-session harness parity**, `--limit 24`, tasks arc_easy, arc_challenge, hellaswag, piqa and winogrande,
    batch 16; policies BF16, FO6 native and FlipQuant 16x64 native.
    - The model is **base Mistral-7B-v0.3 @ caa1feb0** (deviation 3). NVFP4-RaZeR's harness (kernel-opt
      sm120/eval/common.MODELS) knows Mistral only as the base model, and neither Instruct model; its native artifacts
      are the paper's. flipquant main has the same model pinned as `mistral-7b-base`.
    - NVFP4-RaZeR side: one process,
      `run_lmeval_deploy.py --model mistral7b --data-root /home/dev/n16k64_campaign/multimodel/data` with
      `bf16=bf16`, `fo6=native:paper/artifacts/mistral7b_fo6:auto_stock` and
      `fq-16x64=native:paper/artifacts/mistral7b_tc_16x64:auto`; `PYTHONPATH=sm120:.` and `SM120_BUILD_DIR=build_V`.
    - flipquant side: three processes, `--model mistral-7b-base` with `--mode bf16`, native `--weight fourover6`,
      and native `--weight mixfp4 --map paper/maps/mistral7b_16x64/map.pt --unit 16x64` (the map the artifact was
      exported from; the port validation and part P did the same).
    - Same session, same env. Required for every (policy, task): equal sample_digest, equal per-example correctness,
      bit-identical per-choice log-likelihoods (and targets), and equal metrics. Also both sides' native coverage, and
      build_V libraries.
  - **(c) IF4, Zou, FOCUS deploy and GPTQ‡ against the paths behind tab:main-ppl.**
    - **IF4 / Zou** (Experiment A was not ported; flipquant's `flipquant/if4.py` and `zou_mixfp4.py` came from the
      co-author's IF4 / MIXFP4 branches). Script zs6_identity.py on Mistral-7B-Instruct-v0.3, GPU. Reference:
      NVFP4-RaZeR main-ppl 44f8cea `quantize/adaptive_formats.py`, the code behind main-ppl's IF4 / Zou records,
      which part I matched bit for bit.
      - Weights: every quantized Linear's weight from `quantize(w, rule)` vs flipquant `quantize_weight(w, "1x16",
        "own")`, bitwise. The installed-weight sha256, in module order, against main-ppl's record.
      - Activations: every quantized Linear's input on one 2048-token WikiText-2 window of the BF16 model, from
        `quantize_rows(x, rule)` vs flipquant `quantize_act(x)`, bitwise.
      - End to end: flipquant main's `evaluation.ppl` for if4 and zou_mixfp4 (`--act-method own --unit 1x16
        --limit 4`). Its per-window NLL and installed-weight sha256 must equal main-ppl's records (via part I's copies).
    - **FOCUS deploy:** flipquant main's `evaluation.ppl` (part K's command, `--limit 2`) for all six models. The
      per-window NLL must be bit-identical to the main-table records (part K: Mistral, Nemotron, Phi-4; part P: both
      Llama models; part Q: Qwen3.8-27B).
      - Each state's sha256 must equal its record: the FOCUS HF-upload record for part K's states, HF_UPLOAD.md for
        the 8B, part Q for Qwen3.8-27B. The 1B's state has no recorded sha256; it is recorded here.
    - **GPTQ‡:**
      - Each main-table codes file's code digest (the codes_sha256 formula, from the stored entries) and its key
        (settings and calibration-data sha256) must equal the record's (CPU).
      - The two Llama models' codes are regenerated as above.
      - Then flipquant main's `evaluation.ppl` (part J's command, `--limit 2`) for all six models: `ptq.codes_sha256`
        and the per-window NLL must equal the records.
- **Checks inside the panel** (each stops the run):
  - rc 0; a complete report with the six tasks;
  - every example evaluated (1838 / 10042 / 1267 / 2376 / 1172 / 5153), 0-shot, batch 16 (Qwen3.8-27B 8);
  - lm-eval 0.4.11 and the model's pinned revision;
  - every dataset load at its pinned revision (lambada_openai at 900124bf);
  - each task's sample_digest equal across the model's rows.
  - Native rows: downstream's coverage asserts, and the kernel family (nvfp4 / fo6 / focus / gptq: stock_ko; fq-8x64:
    mixed_wB_ko; fq-16x64: mixed_ko; fq-256x64: mixed256_ko), from build_V (every library sha256 among build_V's
    manifests).
  - GPTQ‡: codes_sha256 as recorded.
  - IF4 / Zou: the installed-weight sha256 equal to the main-table records' (part I; part P for the Llama models).
- **Native / fake cross-check:** Mistral-7B-Instruct-v0.3, fo6 and fq-16x64 also with `--mode fake` (the same flags
  otherwise), all six tasks. Reported per task: accuracy native vs fake, the paired difference with its exact McNemar
  p, the share of examples with the same correctness, and the per-choice log-likelihood differences.
- **Old vs new.** The co-author's numbers in flipquant main `docs/results/acc_panel/acc_panel.md` (fake quant on H200,
  no per-example records here): Mistral, Nemotron, Phi-4 and Qwen3.8-27B for bf16, fo6, focus and fq x 3. Its
  Llama-3.1-8B row is the base model and is not compared. Per task: new, old, the difference, and whether |difference|
  exceeds 2 SE (binomial, unpaired: sqrt(p1 q1 / n + p2 q2 / n)).
- **Report** (results/paper_eval/zeroshot6/: REPORT.md, zeroshot6.json, STEP0.md, step0.json):
  - the 6 x 10 x 6 table with Avg. (the unweighted mean of the six tasks);
  - per model, the pooled paired exact McNemar of every row vs FourOverSix and vs BF16 over all questions of the six
    tasks (acc_panel_tables.py's formula: difference in points, row-only / reference-only correct, two-sided exact p);
  - the old-vs-new comparison and the native / fake cross-check;
  - per-run timings (load, per task, total) and the queue START / END;
  - a LaTeX snippet.
- **Rules.** One GPU job at a time on an idle GPU (run_gpu.run). No ncu, no clock locking, no reboot. /home stays
  above 10 GB free; nothing is deleted; retries get new output names. Wait loops wait on the PID or an END log line.
  Commits and the push go to paper-eval only (each push tried once; a denial is reported, not worked around). No HF
  upload.
- **Records.** `RUN/zeroshot6/`: `step0/` (parity, identity, ppl checks, code digests, download), `gptq/`,
  `panel/<model>/<row>.json`, `crosscheck/mistral-7b/<row>_fake.json`. Logs in `RUN/logs/zs6_*.log`; START / END lines
  in `RUN/queue.log`.
- **ETA.** Step 0 about 1 h (plus the 18 GB download, in parallel). The panel is estimated at 17–20 h, mostly the
  simulated IF4 / Zou rows (about 7–9x a native row) and Qwen3.8-27B; the coordinator's estimate is about 14 h. The
  ETA is re-estimated from the first model's runs and sent when the panel starts.

### Part X, amendment 1 (2026-10-14 10:44 UTC, before any parity result): step 0 (b) on Phi-4

- **What happened.** The first parity job (NVFP4-RaZeR side, base Mistral-7B-v0.3) stopped at model load (rc 1,
  6 s). Base Mistral-7B-v0.3's weight files are no longer in the HF cache; only its config and index remain. The
  same holds for base Llama-3.1-8B. Neither model that NVFP4-RaZeR's harness shares with the request can run without
  downloading 14–16 GB of a model the user removed.
- **The change.** The parity runs on **Phi-4**:
  - registry `phi4-14b` on the flipquant side, `phi4` on the NVFP4-RaZeR side, both at 2db69c1c3e91 (the panel's
    revision; the RaZeR side reads its multimodel/data calibration record);
  - the paper's artifacts `phi4_fo6` (kernel auto_stock) and `phi4_tc_16x64` (auto), the latter exported from
    `paper/maps/phi4_16x64/map.pt`, which flipquant reads with `--unit 16x64`.
  - This is part P's harness-parity arrangement, and Phi-4 is a panel model.
  - Everything else is as registered: tasks, `--limit 24`, batch 16, both sides' env, the criteria.
- **Records.** The failed job's log is kept (`logs/zs6_parity_razer.log`). The new records go to
  `RUN/zeroshot6/step0/parity_phi4/`, under new log names (`zs6_parity_phi4_*`).
- **Already passed.** Step 0's codes-file check passed before this change (4/4: code digests and keys equal the
  records).
- **Header correction.** The registration header's time is corrected to the commit's (10:33 UTC; it read 10:50).

### Part X, step 0: the run (2026-10-14 10:41–11:15 UTC); all checks passed

- **Correction.** Amendment 1's header time is corrected to its commit's: 10:44 UTC (it read 10:55).
- **Records.** results/paper_eval/zeroshot6/STEP0.md and step0.json; RUN/zeroshot6/step0/; the gate file
  `step0_ok.json`, which the panel requires.
- **(a) Settings.**
  - The record functions the six tasks use (`primary`, `mc_record`, `sample_hashes`, `summarize`) are AST-identical
    to run_lmeval_deploy.py's.
  - `summarize_group` differs only by one inlined local variable; it serves MMLU only.
  - `lm_eval_datasets.py` is byte-identical (5290f5f6...).
  - flipquant adds the LAMBADA pin (900124bf) and the LAMBADA primary metric (acc). run_lmeval_deploy.py has neither,
    so it cannot evaluate LAMBADA.
  - Other differences, all of route only: model loading, artifacts vs in-memory packing, one policy per process, and
    TaskManager's include path, which adds only mmlu_pro_loglik.
- **(b) Parity** (Phi-4, amendment 1). 15 of 15 (policy, task) cells are bit-identical: sample digest, per-example
  correctness, every per-choice log-likelihood, metrics.
  - Kernels stock_ko / mixed_ko on both sides, every library from build_V. The tokenization records are equal.
  - NVFP4-RaZeR: 338 s, one process; flipquant: 40 / 52 / 52 s.
- **(c) IF4 / Zou** vs NVFP4-RaZeR 44f8cea `adaptive_formats.py`, on Mistral-7B-Instruct-v0.3.
  - Weights: 224 / 224 Linears bitwise for both rules. The installed-weight sha256 is equal across the reference,
    flipquant and main-ppl's record: 3c4d9b3d... (IF4), 7d098175... (Zou).
  - Activations: 224 / 224 Linear inputs bitwise (458,752 rows) for both rules.
  - `evaluation.ppl` (4 + 4 windows): per-window NLL and installed weights equal to the records.
- **(c) FOCUS deploy.** All six models: per-window NLL (2 + 2 windows) bit-identical to parts K / P / Q.
  - The five recorded state sha256 values match.
  - The 1B's state sha256 is recorded here for the first time: 418006a0....
- **(c) GPTQ‡.**
  - The four main-table codes files: code digest and key equal to the records (CPU).
  - Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct codes regenerated by main's GPTQ: codes_sha256 952850d7... and
    d3fe9816..., equal to part P's. Files under RUN/zeroshot6/gptq/; file sha256 5f65382f... and 06ff8052....
  - All six `evaluation.ppl` checks: codes_sha256, data_sha256 and per-window NLL equal to the records.
  - The only argument difference: part P's records list paper-sm120-runs' opt-in `--gptq-candidates` (unused,
    None), which main lacks.
- **Download.** Qwen3.8-27B's FOCUS state: 18,263,340,091 bytes, sha256 baf02d9c..., equal to part Q's record and to
  the Hub's LFS sha256 (15 min).

### Part X, the panel: the run (2026-10-14 11:17 UTC – 2026-10-15 02:21 UTC); every check passed

- **Code.** flipquant 3db0072, checked clean at the start; build_V. Env n16k64, with n16k64-fast for Nemotron and
  Qwen3.8-27B.
- **Runs.** 62 processes: 6 models x 10 rows, plus the Mistral fake cross-check (fo6, fq-16x64). Every process exited
  rc 0 and passed the registered checks:
  - complete reports, with every example of the six tasks (21,848 per row), 0-shot, batch 16 (Qwen3.8-27B: 8);
  - lm-eval 0.4.11 and the pinned revisions; every dataset load pinned, LAMBADA at 900124bf;
  - equal sample digests within each model (they are also equal across models);
  - native coverage and the kernel families, from build_V;
  - GPTQ‡ codes_sha256 and the IF4 / Zou installed-weight sha256 as recorded; the inputs' sha256 re-checked before
    each run.
- **Time.** 15 h 03 min wall; 14.9 h summed over the processes.
  - A native row took 3 min on the 1B, 4.3–6.2 min on the 7–14B models and about 19 min on Qwen3.8-27B.
  - The simulated IF4 / Zou rows took 4.1–7.6x a native row: 1.8–1.9 h each on Qwen3.8-27B.
  - The fake cross-check rows took 22–23 min each.
- **Old vs new** (the co-author's acc_panel): 0 of 144 cells beyond 2 SE.
- **Report.** results/paper_eval/zeroshot6/ (REPORT.md, zeroshot6.json; STEP0.md, step0.json; report_zeroshot6.py).


## Part Y (registered at its commit, 2026-10-15): GSM8K for the six paper models x ten rows, part N's protocol

The user's request, relayed by the coordinator on 2026-10-15 (option A of the coordinator's choice). Every cell is a new
run; part X's verified artifacts are reused, no earlier output.
- **Protocol: part N's, exactly.**
  - lm-eval 0.4.11 `gsm8k_llama` as shipped: 8-shot CoT (first_n), `apply_chat_template` with
    `fewshot_as_multiturn`, greedy, `max_gen_toks` 1024 (`until: []`, HFLM adds EOS). All 1,319 test problems.
  - Filters: strict_match (primary) and flexible_extract, both recorded.
  - Thinking off through lm-eval's own switches. Qwen3.8-27B: `--enable-thinking false` (chat_template_args).
    Nemotron-Nano-9B-v2: `--system-instruction /no_think`. Llama, Mistral and Phi-4 have no thinking mode and get no
    system instruction.
  - Part N's command (run_gsm8k_lmeval.cmd), with `--gen-lengths` and `--samples-out`; part N's env (run_gpu.env()
    only: the default allocator, no CUBLAS_WORKSPACE_CONFIG; unlike part X).
- **Batch.** 64 for every run, except Qwen3.8-27B BF16 at 48 (O0's memory rule); any other change is a recorded
  deviation.
  - GSM8K generations depend on the batch composition. Part M's pad check: at batch 16 vs batch 1, 10 / 64 Nemotron
    and 1 / 32 Qwen completions were identical.
  - Every row therefore runs the full set at the same batch size, so all rows of a model see the same length-sorted
    batches.
  - Qwen3.8-27B BF16 at 48 is the exception. Its pairs with the batch-64 rows compare different batch compositions, and
    that is flagged wherever it affects a paired comparison: vs BF16, fidelity, and old vs new.
- **Models, revisions, rows, envs: part X's.**
  - The models at the main-table revisions. Nemotron and Qwen3.8-27B run in n16k64-fast, the others in n16k64.
  - Ten rows: BF16; NVFP4 (NVFP4 per-token activations); FourOverSix; FOCUS (deployed codes, FourOverSix
    activations); GPTQ‡ (part J's codes, `--any-act`, FourOverSix activations); IF4† and MixFP4 (Zou)† simulated with
    their own rules at 1x16; FlipQuant 8x64 / 16x64 / 256x64.
  - Native = build_V, `--kernel-set auto`, `--act-scope row` (part N's quantization flags).
- **Artifacts: part X's verified files**, re-hashed before each run against part X's gate (`RUN/zeroshot6/step0/
  step0_ok.json`) and records:
  - the release maps and part P's maps;
  - the FOCUS states, including Qwen3.8-27B's re-downloaded state;
  - the GPTQ‡ codes, including the two Llama code files regenerated in part X.
- **Harness: flipquant paper-sm120-runs 4bbe0b1**, the branch that produced part N (f2a56f1), in the worktree
  sm120runs/wt. Before each job the driver checks that HEAD is 4bbe0b1 and clean. llama3.1-8b-instruct comes from the
  branch's own registry entry (7b472b0), so no runtime launcher is used.
- **Step 0** (any failure stops part Y before the panel):
  - **(a)** `evaluation/downstream.py` and `evaluation/lm_eval_datasets.py` are byte-identical between f2a56f1 (part
    N) and 4bbe0b1, so the generation path is unchanged. Between them only calibration, the opt-in `--fo6-tensor-qmax`
    / `--gptq-candidates` code and the registry changed.
  - **(b) 4bbe0b1 vs main 3db0072** (part X's step 0 tied main to NVFP4-RaZeR). On the default paths the code differs
    in:
    - formats / linear / cli / sm120: the FourOverSix tensor-scale constant passed through, 448 by default;
    - gptq: the opt-in candidates and `load_codes_file`; the cache path is unchanged;
    - data: the cache's temporary file name;
    - the registry: the llama3.1-8b-instruct entry;
    - downstream: f2a56f1's options.
    - Numeric parity: Llama-3.2-1B-Instruct, all ten rows, `evaluation.ppl --limit 2` (2 WikiText-2 + 2 C4 windows)
      on both commits, same session and env. Required: bit-identical per-window NLL, equal GPTQ codes_sha256 and equal
      IF4 / Zou installed-weight sha256.
  - **(c')** (the coordinator's approved replacement for a `--limit 64` check, which cannot match a full run's batches):
    Nemotron FourOverSix and FlipQuant 16x64, native, all 1,319 problems with part N's exact command, env and batch 64.
    - Required: every generation (`resps`) and extracted answer (`filtered_resps`, both filters) byte-identical to
      part N's samples.
    - On any difference: stop and report the agreement rate and the first difference.
    - These two runs use the panel's commit, command and env, so they are the panel's Nemotron FO6 / FQ16 cells (new
      runs; recorded in the report).
- **Order.**
  - Native rows (BF16 and the seven FP4 rows) for all models, small first: Llama-3.2-1B-Instruct, Mistral-7B-Instruct-
    v0.3, Llama-3.1-8B-Instruct, Nemotron-Nano-9B-v2, Phi-4, Qwen3.8-27B.
  - Then IF4 and Zou for all models in the same order.
- **Checks per run** (each stops the run):
  - rc 0; a complete report; 1,319 problems with both filters and the generated length of each; n-shot 8;
  - the batch as planned; the chat options and thinking switches as planned (Qwen's template args
    `enable_thinking=False`, Nemotron's `/no_think`, none otherwise);
  - no `<think>` / `</think>` in any generated output (part N's N1);
  - the pinned dataset (openai/gsm8k 740312ad), lm-eval 0.4.11, the model's revision;
  - equal sample digests across a model's rows;
  - native coverage and the kernel family from build_V;
  - GPTQ‡ codes_sha256 and the IF4 / Zou installed-weight sha256 as recorded.
- **Report** (results/paper_eval/gsm8k6/: REPORT.md + gsm8k6.json):
  - one table per model: strict and flexible ± 2 SE; paired exact McNemar (strict) vs FourOverSix and vs BF16;
  - the summary: strict per model plus the mean;
  - fidelity to BF16, per model and pooled:
    - extracted-answer agreement (strict, flexible);
    - correctness flips (row-only + BF16-only correct);
    - byte-identical generations;
    - exact McNemar of FlipQuant 16x64 vs FOCUS, IF4, Zou and FourOverSix on per-problem agreement with BF16;
  - old vs new against part N / O0 (Nemotron and Qwen3.8-27B: BF16, RTN NVFP4, FO6, FQ16): accuracy, paired
    McNemar, byte-identical share;
  - per-run timings; mean / max generated tokens; budget hits;
  - LaTeX.
- **Rules.**
  - One GPU job at a time on an idle GPU; no ncu, clock locking or reboot.
  - /home stays above 10 GB free; nothing is deleted; retries get new names.
  - Wait loops wait on a PID or an END line.
  - Commits and the push go to paper-eval only (each push tried once; a denial is reported).
- **ETA.** Native rows about 9 h (Qwen3.8-27B about 4.7 h of it, from part N / O0's 33 / 50 min per run). IF4 / Zou
  are decode-bound and not yet measured, estimated at 4–12 h. The ETA is sent when the panel's timing starts.

### Part Y, step 0: the run (2026-10-15 08:04–08:20 UTC); all checks passed

- Registration e7acaa0 (08:02 UTC), drivers 551ef6d (08:04 UTC). Records under RUN/gsm8k6/step0/; report
  results/paper_eval/gsm8k6/STEP0.md and step0.json; the gate RUN/gsm8k6/step0/step0_ok.json.
- **(a)** `evaluation/downstream.py` and `evaluation/lm_eval_datasets.py` are byte-identical between f2a56f1 and
  4bbe0b1.
- **(b)** All ten rows give bit-identical per-window NLL on 4bbe0b1 and main 3db0072 (Llama-3.2-1B-Instruct, 2 + 2
  windows, 20 runs of 10–40 s). GPTQ‡ codes_sha256 (952850d7...) and the IF4 / Zou installed-weight sha256 are equal.
- **(c')** Nemotron-Nano-9B-v2 FourOverSix (373 s) and FlipQuant 16x64 (382 s), all 1,319 problems: 2,638 / 2,638
  samples (1,319 problems x 2 filters) have byte-identical generations and extracted answers to part N's. These two
  runs are the panel's Nemotron FO6 / FQ16 cells.
- The panel started right after.


### Part X, amendment 2 "X2" (registered 2026-10-15, at its commit): Qwen3-1.7B and Qwen3-8B

- **The user's decision** (relayed by the coordinator, 2026-10-15 about 12:00 UTC): run the Qwen3 zero-shot panel
  first, then continue GSM8K (part Y) as planned. The coordinator's reading: between part Y's native rows and its IF4 /
  MixFP4 rows, without stopping or killing anything. This is part Y, amendment 1, below.
- **Models**, each at its main-table revision, passed as `--revision` to every job:
  - qwen3-1.7b @ 70d244cc86cc;
  - qwen3-8b @ b968826d9c46.
  - Order: Qwen3-1.7B, then Qwen3-8B. Both run in n16k64.
- **Unchanged from part X:**
  - **Code:** flipquant main 3db0072, the same detached worktree, checked clean before the run; build_V,
    `--kernel-set auto`.
  - **Tasks:** the six, 0-shot, lm-eval 0.4.11, datasets pinned (LAMBADA at 900124bf). Batch 16, per-token
    activations.
  - **Rows:** the ten, in part X's order. BF16, NVFP4, FourOverSix, FOCUS, GPTQ‡ and FlipQuant 8x64 / 16x64 /
    256x64 run native. IF4† and MixFP4 (Zou)† run simulated, with their own rules.
  - **Running:** part X's env (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`),
    run_gpu.run (START / END in queue.log), one job at a time.
  - **Checks:** part X's check_record, which requires:
    - a complete record with every example, n_shot 0, the batch and the revision;
    - the dataset pins and loads;
    - equal sample digests within the model;
    - native coverage and the kernel families from build_V;
    - the GPTQ‡ codes_sha256 equal to the main-table record, and the IF4 / Zou installed-weight sha256 equal to the
      main-table rows';
    - the FOCUS state and map paths.
  - Every cell is a new run; no earlier output is reused.
- **Inputs.** Each is re-hashed before its run, against these records (part X's verify_inputs, with the Qwen3
  entries):
  - **FOCUS states** RUN/focus/qwen3-*/state/focus.pt, against the sha256 the FOCUS upload recorded
    (results/paper_eval/focus_hf/stage.json): 5fa877acad86e7b9... (1.7B) and e4689e64c66fe50a... (8B).
  - **GPTQ‡ codes** RUN/gptqdd/qwen3-*/codes_nvfp4_act-fo6.pt, against the file sha256 recorded at today's HF staging
    (hf_gptq_focus.py): bb50de2383f7690a... (1.7B) and 335a69b142c726a9... (8B).
    - That staging checked each code digest equal to the main-table record: 415a788a25da... (1.7B), 4cba489acee8... (8B).
    - Each run's own codes_sha256 is also checked against the main-table record (gptqdd/<model>/ppl.json).
  - **Release maps** flipquant_release/<model>/flipquant_<unit>.pt, against the release summary's map_sha256 (part X's
    rule).
  - **IF4 / Zou:** the installed-weight sha256 of the main-table rows (mainrows/<model>/{if4,zou}.json), checked after
    each run.
- **Old vs new:** the co-author's acc_panel (flipquant main docs/results/acc_panel/acc_panel.md) for Qwen3-1.7B and
  Qwen3-8B, as in part X. That is BF16, FourOverSix, FOCUS and FlipQuant x3 over the six tasks: 72 cells, Δ against
  2 SE.
- **Report:** results/paper_eval/zeroshot6_qwen3/ (REPORT.md, zeroshot6_qwen3.json), with part X's tables for the two
  models.
  - Then flipquant paper-sm120-runs results/zeroshot6/README.md is regenerated in the same format, with eight models
    plus the mean: Llama-3.2-1B, Llama-3.1-8B, Qwen3-1.7B, Qwen3-8B, Mistral, Nemotron, Phi-4, Qwen3.8-27B.
  - Each push is tried once.
- **Driver:** run_zs6_qwen3.py: `preflight` (CPU: the inputs against these records) and `panel`. Both use part X's
  run_zeroshot6 functions, and `panel` also holds the GPU as in part Y, amendment 1.
- **ETA:** about 2.5–3.5 h. In part X a native row took 3 min on the 1B and 4–6 min on the 7–14B models, and IF4 /
  Zou rows took 4.1–7.6x a native row.

### Part Y, amendment 1 (registered 2026-10-15, at its commit): X2 between the native and the IF4 / MixFP4 rows

- **The decision:** part X amendment 2, above. Nothing that is running is stopped or killed. Part Y's rows, settings
  and checks do not change. Its IF4 / MixFP4 rows run after X2, in the registered order.
- **The hold uses part Y's own runner.** Before every job, run_ts (run_gsm8k.run_ts, via run_gpu.idle) waits until no
  compute process is on the GPU.
  - X2's driver waits until part Y's last native row has started: the line `START gsm8k6_qwen3.8-27b_fq-256x64` in
    queue.log, or part Y's driver has exited.
  - It then holds a CUDA context until its own last job has ended. The context is one small tensor, about 0.5 GB; that
    row peaks at about 51 GiB of the 96 GiB (part N's FQ16 record).
  - Part Y's next job (Llama-3.2-1B-Instruct IF4) therefore waits in run_ts, and starts by itself once X2's driver
    exits.
  - X2's own jobs wait only for other processes: its idle check ignores itself. The held context runs no kernels, so
    the GPU still runs one job at a time.
  - If X2's driver stops, the hold ends with it and part Y continues.
- **Recorded:** queue.log lines `zeroshot6 X2 ...`, with the times. Part Y's report gives X2's interval between its
  native and IF4 / MixFP4 rows.

### Part Y, amendment 2 (2026-10-15 12:30 UTC): the hold began at 12:14 UTC, so X2 runs between Llama-3.1-8B-Instruct's NVFP4 and FO6 rows

- **What happened.** X2's driver waited for part Y's last native row by searching queue.log for the text
  `START gsm8k6_qwen3.8-27b_fq-256x64`. Its own waiting line (12:14:16 UTC) quoted that text, so the search matched at
  once. The hold began at 12:14:17 UTC, while part Y's Llama-3.1-8B-Instruct NVFP4 row was running.
- **The effect:**
  - Part Y's NVFP4 row finished normally (12:26:46 UTC, rc 0).
  - Part Y's driver then reached its next job, Llama-3.1-8B-Instruct FO6. It waits in run_ts for an idle GPU, as
    amendment 1 intended, but at this earlier boundary.
  - X2 started at 12:26:51 UTC.
  - Nothing was stopped or killed. Every job still runs alone on the GPU (the held context runs no kernels), with
    unchanged inputs, env and checks.
  - The cells are independent processes, so neither part's results depend on the position, and the total time is
    unchanged.
- **The coordinator was notified at 12:16 UTC,** with the option to stop X2's driver before part Y's job ended and so
  keep the registered position. Nothing was stopped, so X2 runs now. This is also the user's literal "run the Qwen3
  zero-shot panel first".
- **The fix** (run_zs6_qwen3.py): the trigger is matched as a whole queue.log line, and the waiting line no longer
  quotes it. The running driver keeps its loaded code; the fix applies to any relaunch.
- **The order now:**
  1. part Y's native rows up to Llama-3.1-8B-Instruct NVFP4;
  2. X2's 20 rows;
  3. part Y's remaining native rows: Llama-3.1-8B-Instruct FO6 onward, then Nemotron, Phi-4 and Qwen3.8-27B;
  4. part Y's IF4 / MixFP4 rows.

### Part X amendment 2 (X2): the run (2026-10-15 12:26–15:37 UTC); every check passed

- **Code.** flipquant 3db0072, clean in the preflight and before the panel; build_V; env n16k64, with part X's env
  variables.
- **Position (part Y amendment 2).** X2 ran between part Y's Llama-3.1-8B-Instruct NVFP4 row (ended 12:26:46 UTC) and
  its FO6 row. That row started at 15:36:33 UTC, one second after X2's driver exited.
- **Runs.** 20 processes (2 models x 10 rows). Every one exited rc 0 and passed part X's check_record. The inputs were
  re-hashed before each run.
- **Time.** 3 h 09 min 41 s wall (12:26:51–15:36:32 UTC); 3.13 h summed over the processes.
  - Native rows: 4.2–4.4 min on Qwen3-1.7B, 5.1–7.4 min on Qwen3-8B (BF16 the longest).
  - IF4 / Zou rows: 21.6–22.0 min on the 1.7B, 31.8–33.9 min on the 8B. That is 5.1–5.2x (1.7B) and 6.1–6.6x (8B) a
    native FourOverSix row.
- **Old vs new** (the co-author's acc_panel): 0 of 72 cells beyond 2 SE. The sample digests are equal across all eight
  models of parts X and X2.
- **Report.** results/paper_eval/zeroshot6_qwen3/ (REPORT.md and zeroshot6_qwen3.json, from report_zs6_qwen3.py).
  zs6_flipquant_readme.py generates the flipquant README with the eight models from the two JSONs.

### Part Y, amendment 3 (registered 2026-10-16, at its commit; the coordinator's OK, relayed): Llama-3.2-1B-Instruct's fake rows render the date of its native rows

- **What happened.**
  - At 02:50:07 UTC the panel stopped on a registered check. The first fake row, Llama-3.2-1B-Instruct IF4 (rc 0,
    3,034 s), had a sample_digest that differs from the model's BF16 row.
  - All 2,638 samples' prompts differ in one line, the system header's "Today Date". Llama-3.2-1B-Instruct's chat
    template writes `strftime_now("%d %b %Y")` there.
  - The model's native rows ran on 2026-10-15 ("Today Date: 15 Oct 2026"); the IF4 row ran after midnight UTC ("16 Oct
    2026"). Every other recorded setting is identical.
- **Which rows are affected.** Only Llama-3.2-1B-Instruct's IF4 and MixFP4 (Zou) rows.
  - Llama-3.1-8B-Instruct's template writes a fixed "Today Date: 26 Jul 2024".
  - Mistral, Nemotron, Phi-4 and Qwen3.8-27B have no date.
  - Parts X / X2 use no chat template.
- **The change.** Those two rows run through gsm8k6_date.py (`python gsm8k6_date.py 2026-10-15 evaluation.downstream
  ...`, from the flipquant worktree, as part X's zs6_fq.py did).
  - It replaces transformers' chat_template_utils module global `datetime` with one whose `now()` returns 2026-10-15
    12:00, so `strftime_now` renders "15 Oct 2026". Nothing else changes: not the flipquant checkout (4bbe0b1), and
    not any other clock.
  - Every other row runs exactly as registered (run_gsm8k6.DATE_PIN lists the two rows).
- **Acceptance.** Unchanged: the registered check. Each rerun row's sample_digest must equal Llama-3.2-1B-Instruct
  BF16's, i.e. all 2,638 prompts are byte-identical. If it does not, the run stops and is reported; the check is not
  loosened.
  - Checked beforehand, on the CPU: with the pin, the tokenizer at 9213176726 renders a system header identical to the
    BF16 record's.
- **The Oct-16 IF4 record** is kept, renamed to if4_date-oct16.{json,log,gsm8k_llama.jsonl.gz}; its sha256 is
  unchanged. It is not a panel cell.
- **Restart.** run_gsm8k6.py panel again. It skips every complete record, checking each again, and continues in the
  registered order: Llama-3.2-1B-Instruct IF4 / Zou (pinned), then Mistral, Llama-3.1-8B-Instruct, Nemotron, Phi-4 and
  Qwen3.8-27B. The scope is the full 12 rows.
- **Report.** The date pin is flagged. Old vs new states the "Today Date" line of any older Llama-3.2-1B-Instruct
  GSM8K record it compares against, if that line differs from 15 Oct 2026.
- **Measured.** The fake rows cost about 9x a native row: Llama-3.2-1B-Instruct IF4 took 50.6 min against 5.6 min for
  FO6. The 12 fake rows take about 32 h.

## Part Z (registered 2026-10-16, at its commit): memory (B) and ownership (C1) for the two Llama-Instruct models; latency-rounds validation (b)

- **The request** (the coordinator, 2026-10-16). Close two paper claims that rest on the old six-model set:
  - **Memory:** "for every model", FlipQuant's installed weights equal NVFP4 / FourOverSix to the byte, and after load
    and at the prefill peak they differ by at most 0.02 GiB.
  - **Ownership:** every release map is exact with 0 mismatches. With the 6 Llama maps that makes 24 maps, and a new
    total of checked weight elements.
- **Models:** llama3.2-1b-instruct @ 9213176726f5 and llama3.1-8b-instruct @ 0e9e39f249a1 (the registry pins of
  paper-sm120-runs).
- **B (memory):** a30df07's protocol (run_gpu.memory). mem_probe.py per (model, policy), for bf16, nvfp4, fo6, fq-8x64,
  fq-16x64 and fq-256x64; flipquant's loader and install; build_V; run_gpu.env.
- **C1 (ownership):** run_gpu.ownership. `evaluation.ppl --paper-convention --ownership-check --limit 1` per (model,
  unit), for 8x64, 16x64 and 256x64.
- **Deviations** (part W's arrangement):
  - **Code:** flipquant paper-sm120-runs (worktree sm120runs/wt, code = 4bbe0b1, checked clean) instead of main
    120173a, whose registry lacks both Instruct models. mem_probe.py takes its flipquant path from MEM_PROBE_FLIPQUANT;
    its default (120173a) is B's.
  - **Maps:** part P's, RUN/llama_instruct/<model>/maps/flipquant_<unit>.pt. The Llama release repositories hold these
    files with their meta cleaned and the tiles identical.
- **Records:** RUN/memory/<model>/<policy>.json and RUN/ownership/<model>_<unit>.json. The report is
  results/paper_eval/llama_bc/ (LLAMA_BC.md, llama_bc.json; report_llama_bc.py). B's and C1's own reports do not change.
- **Criteria** (reported, not gates; a failed job stops the run):
  - the installed quantized linears and the whole model are byte-equal, FlipQuant 8x64 / 16x64 / 256x64 against
    NVFP4 and FourOverSix;
  - max |Δ| after load and at the 1x2048 peak ≤ 0.02 GiB;
  - every map exact, with 0 mismatches.
- **Latency-rounds validation (b)**, the GPU smoke test of the latency-rounds port:
  - flipquant latency-rounds 80cd3f0's scripts/run_prefill_latency.py on llama3.2-1b-instruct: fo6 and fq-16x64,
    2 rounds, 1x128 and 1x2048, reps 3;
  - build_V via --kernel-build-dir; part P's 16x64 map via --model-maps;
  - it passes if every registered check passes and the graph ms are within a few % of part W's (RUN/latency_llama);
  - records RUN/latency_rounds_smoke/; a short note in results/paper_eval/latency_rounds_smoke/NOTE.md.
- **Order:** B, then C1, then validation (b), then part Y amendment 4.

### Part Z: the run (2026-10-16 13:13–13:18 UTC); every job rc 0

- **Code.** flipquant sm120runs/wt at 4bbe0b1, clean; build_V; run_gpu.env (mem_probe with MEM_PROBE_FLIPQUANT).
- **B: memory** (12 processes, 14–18 s each). For both models, FlipQuant 8x64 / 16x64 / 256x64 install
  byte-identical quantized linears and whole models to NVFP4's and FourOverSix's: 547,356,672 B (1B) and 3,925,868,544 B
  (8B) for the quantized linears. After load and at the 1x2048 peak they differ by at most 0.0010 GiB (1B) and
  0.0079 GiB (8B). Both claims hold.
- **C1: ownership** (6 processes, 10–20 s each). All 6 maps are exact, with 0 format mismatches, over every module
  (112 / 224). That is 23,857,201,152 weight elements checked. With C1's 18 maps, the total is 24 maps and
  206,941,323,264 elements, all exact.
- **Validation (b)** (4 processes, 13:17:45–13:18:26 UTC). Every registered check passed. The graph times are within
  0.2–3.4 % of part W's. The note is results/paper_eval/latency_rounds_smoke/NOTE.md.
- **Report.** results/paper_eval/llama_bc/ (LLAMA_BC.md, llama_bc.json).

### Part Y, amendment 4 (registered 2026-10-16, at its commit; the coordinator's decision, relayed): Qwen3.8-27B's fake rows with expandable segments

- **What happened.** At 13:08:31 UTC Qwen3.8-27B IF4 (simulated, batch 64) stopped with rc 1. The log shows
  torch.OutOfMemoryError: 73.58 GiB allocated plus 20.37 GiB reserved but unallocated (fragmentation), of 94.97 GiB.
  It failed to allocate 2.68 GiB in an MLP, in the 2nd batch.
  - The first batch, which holds the longest prompts, had taken 14.5 min.
  - Part N never ran Qwen3.8-27B in fake mode. Part Y's env is part N's: run_gpu.env, with the default allocator.
  - No record was written. The log is kept as if4_oom-b64-default-alloc.log (renamed; sha256 unchanged).
- **The change.** Qwen3.8-27B IF4 and MixFP4 (Zou) run at batch 64 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (part X's setting). It applies only to these two rows (run_gsm8k6.ALLOC). Every other row and record stays as is.
  - The setting is written to the job's log (an ENV line), since the record does not hold the environment.
  - The batch stays 64, so these rows see the same batch composition as the model's native quantized rows.
- **Pre-check, before the full rows** (run_gsm8k6.py amend4_check): Qwen3.8-27B IF4 with `--limit 64` (one batch of
  64), once under the default allocator and once under expandable segments.
  - Required: all 128 samples (64 problems x 2 filters) have byte-identical generations and extracted answers.
  - Otherwise: stop and report, and do not run the full rows.
  - Records: RUN/gsm8k6/amend4/.
- **If a full row still runs out of memory:** stop and report. No fallback to batch 32 or 48: that would break the
  batch-64 pairing, and is the user's call.
- **Report.** These two rows' allocator is flagged next to the date pin (§) and the BF16 batch-48 mark (◊).
- **Addendum (2026-10-16; the user's approval, relayed by the coordinator; registered before the full rows start).** It
  replaces "no fallback" for these two rows only.
  - Each row runs at batch 64 with expandable segments, after the pre-check.
  - If a full row runs out of CUDA memory at 64, that row reruns at batch 48; at 48, at 32 (expandable segments kept).
    An OOM at 32 stops the run and is reported.
  - A fallback happens only on a CUDA out-of-memory error. Any other failure stops the run. Each OOM run's log is kept
    as `<row>_oom-b<batch>-expandable.log` (run_gsm8k6.alloc_row).
  - The record holds the batch the row ran at. The record check accepts a fallback batch only when the OOM log of the
    batch before exists (run_gsm8k6.expected_batches).
  - The report flags any row not at 64, like the BF16 ◊: its pairings with batch-64 rows compare different batch
    compositions. A row at 48 shares Qwen3.8-27B BF16's batch composition, and the report says so.
  - The pre-check rule is unchanged: generations not byte-identical means stop and report. The fallback does not cover
    that.

### Part Y, the panel: the run (2026-10-15 08:20 UTC – 2026-10-16 21:04 UTC); every check passed

- **Code.** flipquant paper-sm120-runs 4bbe0b1 (sm120runs/wt, checked clean before each job); build_V; env n16k64, with
  n16k64-fast for Nemotron and Qwen3.8-27B; part N's env (run_gpu.env) except amendment 4's two rows.
- **Cells.** 60 (6 models x 10 rows). Every one exited rc 0 and passed the registered checks:
  - complete, with all 1,319 problems, 8-shot, and the pinned dataset;
  - each model's chat options, and no think tags;
  - equal sample digests within the model;
  - native coverage and the kernel families from build_V;
  - the GPTQ‡ codes_sha256, the IF4 / Zou installed weights, and the FOCUS state and map paths.
  - The Nemotron FO6 / FQ16 cells are step 0 (c')'s runs.
- **Batch.** 64, except Qwen3.8-27B BF16 at 48 (its record and logged command). Amendment 4's two rows ran at 64; no
  fallback was needed.
- **Amendments:**
  - 1–2: X2 ran 2026-10-15 12:26:51–15:36:32 UTC, between Llama-3.1-8B-Instruct's NVFP4 and FO6 rows.
  - 3: Llama-3.2-1B-Instruct's IF4 / MixFP4 ran with the date pinned; their digests equal BF16's. The unpinned Oct-16
    IF4 record is kept as if4_date-oct16.*.
  - 4 (+ addendum): Qwen3.8-27B's IF4 / MixFP4 ran with expandable segments at batch 64. The pre-check gave 128/128
    byte-identical; the default-allocator OOM log is kept.
- **Old vs new.** The 8 cells against part N / O0 (Nemotron and Qwen3.8-27B; BF16, NVFP4, FO6, FQ16) have 100 %
  byte-identical generations and equal accuracy.
- **Time.** 08:20 UTC Oct 15 – 21:04 UTC Oct 16, 36.7 h wall. That includes X2's 3.2 h, the two stops and part Z. The
  panel's processes sum to 31.7 h.
  - Native rows took 4–50 min.
  - Simulated rows took 2.5–6.9x a native FO6 row. Qwen3.8-27B IF4 / MixFP4 took 3.9 h / 3.3 h.
- **Analysis definitions** (the coordinator's, 2026-10-16).
  - Answer agreement compares extracted answers after the scorer's own normalization.
  - A pair where both sides have `[invalid]` is not agreement, and is counted separately. Each row's no-answer count is
    reported.
  - An unpublished preview had compared raw strings ("20." vs "20").
- **Report.** results/paper_eval/gsm8k6/: REPORT.md and gsm8k6.json; STEP0.md and step0.json.

## Part AA (registered 2026-10-19, at its commit): calibration size, 1,024 / 2,048 fit windows for FlipQuant 16x64 and FOCUS

The user's request, relayed by the coordinator on 2026-10-19. **This registration covers step 0, the windows, the 2k
memory probes and the pilot. The full eight-model study and every 2,048-window calibration need the user's OK (a later
amendment); after the pilot the run stops and reports.**
- **Question.** Do 1,024 or 2,048 windows of 512 tokens, instead of the release's 256 (still 5 epochs), improve
  FlipQuant's downstream accuracy? FOCUS trained on the same windows is the comparison.
- **Motivating paired losses.**
  - Zero-shot: FlipQuant is below FOCUS on Phi-4 and Nemotron-Nano-9B-v2, and below IF4 on Qwen3.8-27B, at all three
    tiles (part X / X2).
  - GSM8K: FlipQuant is below MixFP4 (Zou) on Phi-4 (part Y).
- **Methods. Only the data changes.**
  - **FlipQuant 16x64** with the release settings: `calibration.train_map --unit 16x64 --fit-windows N --epochs 5
    --teacher-topk 1000 --data-root fqrel/tmopt_data`. That is Adam lr 0.02, eps 1e-12, 8 sequences per step (Qwen3.8-27B
    and Nemotron 2 x 4), theta_u = -1, seed 0, deterministic, the top-1000 + tail-bucket teacher, no development set,
    the last-step map. 640 steps at N = 1,024; 1,280 at 2,048.
  - **FOCUS** with part K's FOCUS_COMMON unchanged (run_sm120.FOCUS_COMMON): 1 epoch, global batch 32, lr 5e-3 / 1e-3,
    KL-Top k = 1000, num_sub 2, init_q 6, FourOverSix per-token activations, seed 42, seqlen 512; micro-batch 8, halved
    on OOM (part K's rule; the global batch stays 32). 32 steps at 1,024; 64 at 2,048. Deployed as part K: NVFP4 codes,
    FourOverSix per-token activations, native.
- **Code: flipquant branch calib-size**, cut from paper-sm120-runs caf69be (whose code is 4bbe0b1's, parts Y and Z),
  at f912a30, in the worktree `/home/dev/n16k64_campaign/calib_size/wt`. flipquant main is not touched.
  - f912a30 changes only calibration code and tests. Nothing under evaluation/, flipquant/, models/ or kernels/
    changes, and the vendored calibration/tmopt files are byte-identical (tests/test_stage3.py).
    - `calibration/tmopt_launch.py --teacher-host` and `calibration.train_map --teacher-host`: the top-K teacher's rows
      in host memory instead of on the GPU. The rows are about 3 MB per window, so 1,024 / 2,048 windows add
      2.2 / 5.1 GiB to a 256-window run, and Qwen3.8-27B's 256-window run already peaks at 91.25 GiB of the card's
      95.6 GiB. The losses read the rows only through `topk_teacher.batch_rows`, which moves each micro-batch's rows to
      the GPU, so the map is the GPU-teacher map (checked bitwise in step 0 (b)).
    - `--probe-steps N` for train_map (the vendored trainer's own probe) and train_focus: memory probes that stop
      after N optimizer steps and write no map or state.
    - `calibration.fit_windows`: the trainer's N-window fit set, drawn on the CPU without training. It is written as a
      `--calib-extension` file.
  - Tests: tests/test_calib_size.py, 7 CPU tests, pass. The whole suite on the CPU: 224 passed, 108 skipped, 3 failed.
    The same 3 tests also fail on the unmodified caf69be, so the failures are not this branch's: test_ifeval_scoring
    (a scorer module missing from the env), test_every_block_can_take_four_with_256, and test_topk_kl_bounds_full_kl
    (a BF16 tolerance).
  - Every job checks that the worktree is clean at the registered commit. Kernels: build_V. Envs: n16k64, with
    n16k64-fast for Nemotron and Qwen3.8-27B.
- **Data, per model, with the model's tokenizer.** The release set's rule, extended:
  - **The rule.** The calibration record's 128 windows (`fqrel/tmopt_data/<key>/calibration/report.json`), then N - 128
    by the vendored extension rule (`calibration/tmopt/fit_extension.py`).
    - Seed 20260930, on OpenWebMath train-00000 @ fde8ef8 and CodeParrot-clean file-000000000001 @ 35a59fb.
    - (N - 128) / 2 windows per source, math first, one window per document.
    - It skips the record's fit documents, the model's three development records, and its published C4 evaluation
      documents (where the trainer has them).
  - **Drawn before training** with `calibration.fit_windows` (CPU) for N = 1,024 and 2,048. Records:
    `RUN/calib_size/windows/<model>_<N>.json`.
  - **Nested (256 ⊂ 1,024 ⊂ 2,048).** Each source's windows come from one seeded stream in file order. Checked from
    the records (document sha256, offset, token sha256), before any calibration:
    - each source's first 64 extension windows at 1,024 equal the release extension's (the release 16x64 trainer
      report);
    - each source's first 448 at 2,048 equal the 1,024 set's;
    - the skip counts equal the release run's.
  - **Byte-identical for FlipQuant and FOCUS.** FlipQuant draws the set in its trainer (`--fit-windows N`). FOCUS
    reads `--data release --calib-record <record> --calib-extension <that FlipQuant run's trainer report>`, as part K
    did; that path rebuilds every window and checks each token sha256.
    - Required: the FlipQuant run's extension records equal the pre-drawn file's, and the FOCUS report's token sha256
      list (record + extension, in order) equals the FlipQuant run's.
    - Recorded per (model, N): the sha256 of the ordered token-sha256 list.
  - **Disjoint from the evaluation sets.**
    - (i) By document, through the rule's skip set, as the release.
    - (ii) A 13-gram audit (CPU) of every window of each model's 2,048 set, which contains the 1,024 and 256 sets.
      The windows are compared against the texts actually evaluated:
      - the WikiText-2 test windows and C4 validation windows of the PPL evaluation (`flipquant.data.eval_windows`,
        decoded with the same tokenizer);
      - the six zero-shot tasks' evaluated documents (lm-eval 0.4.11's task objects with flipquant's dataset pins:
        context plus every choice);
      - GSM8K test (question and answer) and gsm8k_llama's eight few-shot exemplars.
    - Normalization: lowercase, every character outside [a-z0-9] a space, a whitespace split; word 13-grams within
      each text (the GPT-3 / lm-eval decontamination convention).
    - A hit is a window that shares at least one 13-gram with an evaluation text. Every hit is listed, with the
      window, its document, the evaluation set and item, and the first shared 13-gram.
    - **Stop rule:** a hit among the 1,024 set's new windows (not in the release 256) stops the pilot before any
      calibration and is reported. Hits in the release 256 are in the existing maps and states and are reported. Hits
      only in the 2,048 set are reported for the later decision.
- **Evaluation: parts X and Y exactly, plus PPL.** Commands, envs, batches and per-run checks are those parts', with
  the new map or state path.
  - **Zero-shot:** `evaluation.downstream --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai`,
    0-shot, batch 16 (Qwen3.8-27B 8). Env: run_gpu.env() + `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Part X's checks: every example, the pins, lm-eval 0.4.11, the
    revision, kernel families and coverage from build_V. Also each task's sample_digest equals the model's part X
    records'.
  - **GSM8K:** part N's / Y's command (`evaluation.downstream --tasks gsm8k_llama --apply-chat-template` + thinking
    switches, batch 64, `--gen-lengths --samples-out`), env run_gpu.env(), part Y's checks.
    - Llama-3.2-1B-Instruct runs through amendment 3's launcher (gsm8k6_date.py 2026-10-15), so its prompts render
      "15 Oct 2026" like every Llama-1B GSM8K record. The sample_digest must equal the model's part Y records'.
  - **PPL:** `evaluation.ppl --paper-convention` (WikiText-2 / C4), native. FlipQuant uses `--weight mixfp4 --act
    fourover6 --map`; FOCUS uses part K's `--weight focus --focus <state> --focus-deploy --act fourover6 --act-scope
    row`.
  - Recorded per map: the E0M3 share (the map's meta).
- **Step 0** (any failure stops part AA before the pilot and is reported):
  - **(a) Code.** `git diff caf69be f912a30` touches only the five files above, and `evaluation/` is byte-identical.
  - **(b) Calibration identity.** Llama-3.2-1B-Instruct FlipQuant 16x64 at the release settings (`--fit-windows 256`,
    part P's command) on calib-size, twice:
    - with the GPU teacher;
    - with `--teacher-host`.
    - Both runs' trainer map (`run_map_sha256`) and tiles must equal part P's.
  - **(c) Evaluation identity**, on calib-size:
    - Llama-1B FlipQuant 16x64 with part P's map: the six zero-shot tasks against part X's record (per task: equal
      sample_digest, per-example correctness, bit-identical per-choice log-likelihoods, equal metrics); GSM8K with
      the date pin against part Y's samples (every generation and extracted answer byte-identical, 1,319 x 2
      filters); PPL against part P's record (bit-identical per-window NLL).
  - **(c2) PPL references** (not a stop condition): FourOverSix, FlipQuant 16x64 at 256 (part P's / the release map)
    and FOCUS at 256 (part P's / part K's state), for both pilot models, evaluated on calib-size. These runs are the
    references of the pilot's PPL statistics (one harness and session for every PPL row). Each is compared with the
    per-window NLL of the record the final table uses (part P; main-ppl 44f8cea and the release records for Phi-4;
    part K), and any difference is reported.
  - **(d) FOCUS reproducibility.** Llama-3.2-1B-Instruct FOCUS at 256 windows is retrained with part P's command on
    calib-size, and its state is compared tensor by tensor with part P's. train_focus was nondeterministic in part U.
    If the state differs, it is evaluated like a pilot row (zero-shot, GSM8K, PPL) as a run-to-run reference for
    FOCUS's 1,024-vs-256 comparison.
- **Memory probes at 2,048 windows** (the two worst cases of the 256-window runs), after step 0:
  - **FlipQuant Qwen3.8-27B 16x64:** `--fit-windows 2048 --epochs 5 --teacher-topk 1000 --teacher-host
    --probe-steps 4` (micro 2 x accumulation 4, as the release). With the teacher on the GPU it cannot fit:
    91.25 + 5.12 = 96.4 GiB allocated, more than the card's 95.6 GiB (from the 256-window record; not run).
  - **FOCUS Phi-4:** FOCUS_COMMON, micro 8, `--probe-steps 2` (8 micro-batches), on the pre-drawn 2,048 set.
  - Recorded:
    - per-phase peak allocated / reserved GiB;
    - the device peak (nvidia-smi every 2 s, part P's sampler);
    - host RSS;
    - the teacher's bytes and device, and the windows' cache files on disk. The teacher has no disk cache: train_map
      keeps it in memory, and FOCUS recomputes the teacher's logits per micro-batch.
- **Pilot:** Llama-3.2-1B-Instruct and Phi-4, N = 1,024, FlipQuant 16x64 and FOCUS: 4 calibrations and 4 evaluation
  rows (zero-shot, GSM8K, PPL).
  - **Order:** Llama-1B FQ16, Llama-1B FOCUS, their six evaluations; then the same for Phi-4.
  - **Records:** `RUN/calib_size/<model>/{fq16_1024/ (train_map's output), focus_1024/state/, eval/<row>/...}`.
  - **Measured per calibration:** wall time (train_map / train_focus, and the job), the trainer's training time, peak
    GPU allocated / reserved, the device peak (sampler), host RSS, and the teacher's bytes.
  - **Disk:** a Phi-4 FOCUS state is 10.2 GB, a Llama-1B state 0.73 GB. /home had 48 GB free at registration.
- **Report** (results/paper_eval/calib_size/: REPORT.md, calib_size.json, WINDOWS.md, STEP0.md, PROBES.md):
  - Per model, the rows FQ16-256, FQ16-1k, FOCUS-256 and FOCUS-1k, with the part X / Y baselines for context (BF16,
    NVFP4, FO6, GPTQ‡, IF4, MixFP4, FQ8 / FQ256): six zero-shot tasks and Avg., GSM8K strict (± 2 SE), PPL.
  - Paired: each new row vs FourOverSix and vs its 256 counterpart, and FQ16 vs FOCUS at the same size (also at 256).
    - Zero-shot: pooled exact McNemar over all questions of the six tasks (part X's formula), and per task.
    - GSM8K: exact McNemar on strict-match.
    - PPL: ΔNLL ± 2 SE per corpus, on the step-0 (c2) references.
  - Also: the E0M3 share at 256 vs 1,024; calibration time and memory; the FOCUS run-to-run reference if (d) applies.
  - The measured times, re-estimated for the full study, and its disk need.
  - What the later parts need, with no runs:
    - Qwen3.8-27B FOCUS on 2 x H200 on the cluster, as focus_qwen27b (account gov113008, partition taide_h200);
    - Qwen3-1.7B / Qwen3-8B have no GSM8K baselines;
    - Qwen3.8-27B and Nemotron FlipQuant at 2 x 4.
- **Rules.**
  - One GPU job at a time on an idle GPU; no ncu, no clock locking, no reboot.
  - /home stays above 10 GB free; nothing is deleted (the coordinator decides about deletions); retries get new
    names.
  - Wait loops wait on a PID or an END line.
  - Commits: paper-eval and flipquant calib-size, each pushed once per stage (the code + registration; the pilot). A
    denied push is reported.
- **ETA.**
  - Windows and the audit: about 1 h of CPU.
  - Step 0: about 40 min of GPU.
  - Probes: about 45 min. Qwen3.8-27B's teacher pass over 2,048 windows is about 22 min, from the release's
    163 s per 256.
  - Pilot: about 2.5 h. Phi-4 FQ16 at 1k is about 30 min and its FOCUS about 28 min; the evaluations take about
    80 min.
  - Total about 5 h.

### Part AA, amendment 1 (registered 2026-10-19, at its commit, before the pilot started; the user's decision, relayed): the full scope

The coordinator relayed on 2026-10-19 (during step 0) that the user approved the full scope. This replaces "stop and
report after the pilot".
- **Scope:** all eight models at 1,024 and then 2,048 windows. FlipQuant 16x64 and FOCUS keep the registered settings,
  and the evaluation stays as registered (zero-shot, GSM8K, PPL).
  - **1,024 panel first:** the pilot (Llama-3.2-1B-Instruct, Phi-4), then Mistral-7B-Instruct-v0.3,
    Llama-3.1-8B-Instruct, Nemotron-Nano-9B-v2, Qwen3.8-27B, Qwen3-1.7B and Qwen3-8B (the coordinator's order).
  - **Then 2,048** for all eight, in the same order, the pilot models first.
- **Gates.** These are the coordinator's rules; each stops the run and is reported:
  - After the pilot: a short report to the coordinator. If every check passed, the run continues without waiting.
  - **It stops** on anything implausible:
    - any failed check;
    - any OOM. FOCUS runs at micro-batch 8 for every local model, as every 256-window FOCUS run did. The registered
      halving rule is withdrawn: an OOM stops the run instead.
    - an E0M3 share above 25 %.
- **Per model, before its first new calibration:**
  - **Windows:** its 1,024 and 2,048 sets drawn (calibration.fit_windows, CPU) and the nesting checks.
  - **Audit:** the 13-gram audit of its 2,048 set, with the positive control. A hit among the new windows of a set
    stops that set's calibrations.
  - **Zero-shot identity.** Part X / X2 ran on flipquant main 3db0072; calib-size's registry and model code are
    paper-sm120-runs'. So the model's FlipQuant 16x64 (256) zero-shot row is re-run on calib-size and must be
    bit-identical to part X / X2's record: per task, digest, per-example correctness, per-choice log-likelihoods and
    metrics. Every model runs `python -m evaluation.downstream`. calib-size's registry has llama3.1-8b-instruct, so
    zs6_fq.py is not used. A difference stops that model's new rows.
  - **GSM8K:** part Y ran on 4bbe0b1, whose evaluation code equals calib-size's. Step 0 (c) stands for every model, and
    each new row's sample_digest must equal the model's part Y records'.
  - **PPL references (as (c2)):** FourOverSix, FlipQuant 16x64 (256) and FOCUS (256) on calib-size, compared with the
    final table's records. Qwen3.8-27B's FOCUS-256 is part Q's state (HF snapshot b84ff372a7, sha256 baf02d9c…).
- **Qwen3-1.7B and Qwen3-8B GSM8K baselines** (native only, as approved). The six rows are BF16, NVFP4, FourOverSix,
  GPTQ‡, FlipQuant 16x64 (256, the release map) and FOCUS (256, part K's state).
  - Inputs: X2's registered inputs (FOCUS-state and codes-file sha256) and the release maps' sha256.
  - Protocol: part Y's (gsm8k_llama, chat template, 8-shot, greedy, 1,024 tokens, batch 64, run_gpu.env()). Thinking
    is off through `--enable-thinking false`, the Qwen3 template's switch, as for Qwen3.8-27B.
  - Checks: part Y's, with `enable_thinking=False`, and equal sample digests across the model's rows.
  - The simulated IF4 / MixFP4 GSM8K rows are not run for Qwen3 (not approved).
- **Qwen3.8-27B:**
  - FlipQuant 16x64 at both sizes runs with `--teacher-host`; the teacher cannot stay on the GPU next to the model at
    either size. It runs micro 2 x accumulation 4, as the release.
  - FOCUS at both sizes: the route is the user's decision. The registered route (2 x H200 on the cluster, as
    focus_qwen27b) cannot be submitted from this machine (no cluster access). Until decided, its FOCUS cells are TBD and
    the panel continues.
- **Disk.**
  - FOCUS states take about 51.7 GB per size for the eight models (Qwen3.8-27B 18.3 GB, Phi-4 10.2 GB). That is more
    than /home holds above its 10 GB floor.
  - Before each FOCUS calibration, the driver checks that /home keeps 10 GB free after a state of the model's
    256-window size plus 1 GB. Otherwise it stops before starting, pending the user's decision.
  - Nothing is deleted. The 1,024 panel's local states fit; the 2,048 states need the decision.
- **Reporting:**
  - short messages after the pilot, after the 1,024 panel and at the end;
  - per model and size: vs FourOverSix, vs the 256 counterpart, and FlipQuant 16x64 vs FOCUS at the same size (exact
    McNemar per model, and pooled over the models with the discordant counts summed); ΔNLL ± 2 SE; E0M3 share;
    calibration time and peak memory.
- **Pushes:** flipquant calib-size and paper-eval, once per completed stage (the pilot, the 1,024 panel, the end).
- **ETA,** from the 256-window records: FlipQuant scales with windows; FOCUS with steps.
  - After the pilot, the 1,024 panel takes about 16 h: FlipQuant 6.4 h (Nemotron 2.9 h, Qwen3.8-27B 2.7 h), FOCUS
    1.2 h, evaluations 5 h, identity and references 1.2 h, Qwen3 GSM8K baselines 2.4 h.
  - The 2,048 panel takes about 23 h: FlipQuant 13.6 h, FOCUS 3.3 h, evaluations 6.4 h.
  - Both exclude Qwen3.8-27B FOCUS.

### Part AA, amendment 2 (registered 2026-10-19, at its commit, during the pilot; the user's two decisions, relayed): Qwen3.8-27B FOCUS locally; FOCUS states on /vault

The coordinator relayed both decisions at 12:55 UTC. Amendment 1 had left them pending. This amendment applies from
the first run beyond the pilot; the pilot runs as registered.
- **Qwen3.8-27B FOCUS (1,024 and 2,048): trained locally on one RTX PRO 6000**, with part U's host-offloaded AdamW.
  - The user approved porting the offload feature only, and pushing it with calib-size. The port is
    `calibration.train_focus --offload` (HostAdamW) and its GPU test, flipquant calib-size 898d00e (from f912a30).
    Part U's `--init fourover6`, `focus.init_m` and the formats refactor are not ported. The part-u branch and part U's
    records stay local and unpushed.
  - HostAdamW keeps the scale parameters on the GPU. Their gradients go into pinned FP32 host accumulators, in the GPU
    path's order, and AdamW runs on the CPU in FP32. The step's gradient sum is bitwise the GPU path's; the update can
    differ by one ulp.
  - **Settings:** FOCUS_COMMON (global batch 32: 32 / 64 steps; the 1,024 / 2,048 windows) with `--offload --micro 1`
    (Qwen3.8-27B is batch1_only), device_map cuda, n16k64-fast.
  - **Recorded deviation from focus_qwen27b:** one PRO 6000 with offloaded AdamW instead of 2 x H200 with device_map
    balanced; micro 1 in both. Only the summation order differs.
  - **Cost** (part U's 256-window offload run: 622 s per step, GPU peak 92.1 GiB, host RSS 84 GiB): about 5.5 h at
    1,024 and 11 h at 2,048.
  - Before the first run, the offload's GPU test (tests/test_focus_offload.py, against the GPU AdamW path) runs as a
    queued job on calib-size.
  - An OOM, or any need to change a setting, stops the run as for every row. Evaluation is native, as registered.
- **FOCUS states on /vault.** Every FOCUS state trained after the pilot goes to
  `/vault/flipquant_paper_eval/calib_size/<model>/focus_<N>/state/`. Only that directory is written on /vault.
  - The state's report.json is copied into `RUN/calib_size/<model>/focus_<N>/` with its job record.
  - The state's sha256 is recorded after the write. It is re-read and verified against that record before each of
    its evaluations; a mismatch stops the run.
  - Other large part AA artifacts may go to the same directory if /home would cross its floor. Nothing is deleted
    anywhere, and /home keeps its 10 GB floor.
  - The pilot's two states stay where the pilot wrote them (RUN on /home).
  - The disk gate of amendment 1 now checks /vault for the state and /home for its floor.
- **Code:** the panel runs on calib-size 898d00e. calib-size is fast-forwarded from f912a30 after the pilot's last job.
  - f912a30..898d00e changes only calibration/train_focus.py (the opt-in `--offload`; a default run constructs the
    same `torch.optim.AdamW`) and adds tests/test_focus_offload.py.
  - evaluation/, flipquant/, models/, kernels/ and calibration/train_map.py are unchanged, so step 0's identity
    stands.
- **ETA:** the full scope grows by about 17 h with Qwen3.8-27B FOCUS (16.6 h of training and evaluation), to about
  55–60 h after the pilot.

### Part AA: step 0, probes and pilot, the run (2026-10-19 12:01–15:10 UTC); every check passed

- **Windows and audit (CPU).** All eight models' 1,024 / 2,048 sets were drawn; 256 ⊂ 1,024 ⊂ 2,048 holds, record by
  record, and the skip counts equal the release's. The 13-gram audit finds 0 hits in every model's 2,048 set. Its
  positive control detects each of five spliced evaluation items in 30 / 30 windows per model. RUN/calib_size/windows/.
- **Step 0.**
  - (a) The diff is as registered.
  - (b) Llama-1B FQ16-256 on calib-size reproduces part P's trainer map (00c5e22b…) with the GPU teacher and with
    `--teacher-host`. The host teacher lowers the peak by exactly the teacher's 0.73 GiB, at +3 % training time.
  - (c) Bit-identical to part X (zero-shot, 6 / 6 tasks) and part Y (GSM8K 2,638 / 2,638, date pinned); PPL equal to
    part P.
  - (c2) The six PPL references are bit-identical to the final table's records.
  - (d) FOCUS is not reproducible run to run: all 224 tensors differ, mean |Δm| 0.0065 against a training drift of
    0.013. The retrained state is evaluated as the run-to-run reference.
- **Probes at 2,048** (both fit):
  - FQ16 Qwen3.8-27B with `--teacher-host`: 90.52 GiB allocated, device peak 95,020 MiB. The teacher is 6.28 GB on the
    host; 13.6 s per step.
  - FOCUS Phi-4: 91.04 GiB allocated (the 256 value), device peak 96,148 MiB; 51.6 s per step.
- **Pilot (1,024 windows).**
  - Calibrations: Llama-1B FQ16 225 s and FOCUS 173 s; Phi-4 FQ16 1,688 s and FOCUS 1,702 s.
  - E0M3 share: Llama-1B 14.8 % (2.8 % at 256); Phi-4 3.7 % (0.6 %).
  - Every gate passed. Evaluations: results/paper_eval/calib_size/REPORT.md.
- **Code for the panel:** calib-size was fast-forwarded to 898d00e at 15:10 UTC (amendment 2), and the panel started
  right after.

### Part AA, amendment 3 (registered 2026-10-20, at its commit, during the 1,024 panel; the user's approval, relayed): a local FOCUS-256 control for Qwen3.8-27B

**Why.** Qwen3.8-27B's FOCUS-1k scored GSM8K 88.40 against 93.56 for the cluster FOCUS-256 (−5.16, exact p = 1e-8),
while its zero-shot (+0.38*) and NLL (−0.0117* / −0.0021*) improved.
- **The failure is early stopping:** 102 of the 106 lost problems are strict [invalid]. The model stops mid-sentence
  ("… The final") or before the final sentence. Strict no-answer 115 vs 35; mean length 148 vs 167 tokens.
- **The cause is confounded.** FOCUS-1k is the first Qwen3.8-27B state trained locally with the offloaded AdamW
  (amendment 2), while FOCUS-256 was trained on the cluster. FOCUS is also not reproducible run to run (step 0 (d)).

**The control.** Qwen3.8-27B FOCUS-256 retrained locally, exactly as the 1,024 run: FOCUS_COMMON, `--offload --micro
1`, device_map cuda, calib-size 898d00e, one RTX PRO 6000, on the release 256 windows (the record plus the release
16x64 trainer report's extension). That gives 8 steps, about 75 min.
- **Checks:** 256 windows, 8 steps, and token sha256 equal in order to the cluster state's (part Q's report.json data
  records).
- **Storage:** `/vault/flipquant_paper_eval/calib_size/qwen3.8-27b/focus_256_local/state/`. The sha256 is recorded at
  the write and verified before each evaluation; the job record and a report copy go to
  `RUN/calib_size/qwen3.8-27b/focus_256_local/`.
- **Evaluation:** zero-shot, GSM8K and PPL, as every row; row name `focus256_local`.
- **Schedule:** right after the 1,024 panel and before the 2,048 panel. The 2,048 panel driver runs it first. Nothing
  else in the 2,048 plan changes.

**Report.**
- **GSM8K diagnostics:** strict [invalid] count, generations ending at "The final [answer [is]]", and mean generated
  length, for the control next to the cluster FOCUS-256 and FOCUS-1k (and FOCUS-2k once it exists).
- **Paired** (zero-shot, GSM8K, ΔNLL):
  - control vs cluster FOCUS-256;
  - FOCUS-1k vs control, which holds the training environment fixed.
- **Qwen3.8-27B FOCUS-2k:** the control is its environment-matched reference. The cluster FOCUS-256 comparisons stay,
  marked as a different environment.
- **Interpretation is not fixed in advance.**
  - If the control also stops early, the cause is the run or the environment.
  - If it does not, the window count is the likelier cause; FOCUS's run-to-run noise still applies.

### Part AA: the 1,024 panel, the run (2026-10-19 15:10 UTC – 2026-10-20 12:24 UTC); every gate passed

- **Per model:** for all eight models, the zero-shot FQ16-256 re-run on calib-size is bit-identical to part X / X2. All
  24 PPL references equal the final table's records. The FOCUS states trained after the pilot are on /vault, and each
  sha256 was verified before every evaluation.
- **E0M3 share at 1,024** (256 in parentheses): Llama-1B 14.8 % (2.8), Phi-4 3.7 (0.6), Mistral 5.3 (1.1), Llama-8B 5.5
  (0.8), Nemotron 5.6 (1.1), Qwen3.8-27B 2.5 (0.4), Qwen3-1.7B 8.2 (1.2), Qwen3-8B 4.4 (0.6). Every value is under the
  25 % gate.
- **Qwen3-1.7B / Qwen3-8B GSM8K baselines** (6 rows each, thinking off): every check passed.
- **Qwen3.8-27B FOCUS-1k** (local, offloaded) stopped early on GSM8K: −5.16* vs the cluster FOCUS-256. The coordinator
  was told, and the user approved amendment 3's local control, which runs first in the 2,048 panel.
- **Logs.** FOCUS calibration job names do not include the window size, and run_sm120.run_fresh_dir writes each log
  fresh. So a later run of the same model overwrote an earlier log:
  - step 0 (d)'s Llama-1B FOCUS-256 rerun log, by the pilot;
  - Qwen3.8-27B's FOCUS-1k log, by the control.
  - Both runs' per-step training logs remain in their states' report.json, and its copies in RUN.
  - The other seven 1,024 FOCUS logs were copied to `logs/cs_focus_<model>_1024_micro8.log` before their 2,048 runs.
    The control's log is copied when it ends (calib_size/keep_control_log.sh).
- **Results:** results/paper_eval/calib_size/REPORT.md (all eight models at 1,024; pooled McNemar per comparison).

### Part AA: FOCUS log names (2026-10-20, the coordinator's request, before the 2,048 FOCUS runs; naming only)

The FOCUS calibration job names, and so their stdout logs, now include the window size, so no 2,048 log overwrites a
1,024 or control log. No run setting changes.
- **Driver code** (any later driver process): `cs_focus_<model>_<dir>_micro<m>`, with `<dir>` = focus_1024 /
  focus_2048 / focus_256_local.
- **The 2,048 driver already running** (started 12:24 UTC with the control) keeps the old names in memory, so it is not
  stopped or changed. Instead, each old name `logs/cs_focus_<model>_state_micro<m>.log` was renamed to
  `…state_micro<m>.before-2048.log` (the 1,024 log) and replaced by a symlink to `logs/cs_focus_<model>_focus_2048_micro<m>.log`.
  The running driver's open(…, "w") writes each 2,048 log into its own file.
  - Done for the seven local models at 12:50 UTC.
  - For Qwen3.8-27B it is done once the control has ended and its log is kept as
    `cs_focus_qwen3.8-27b_256local_micro1.log` (calib_size/qwen27b_log_symlink.sh), well before its 2,048 FOCUS run.
- **The 1,024 logs** stay in three places: the earlier copies `cs_focus_<model>_1024_micro8.log`, the renamed originals
  (identical, cmp-checked), and every state's report.json.

### Part AA, amendment 3: the control, the run (2026-10-20 12:24–14:38 UTC)

- **Training:** 8 steps in 4,453 s (557 s per step), 256 windows equal in order to the cluster state's. GPU peak 91.79 GiB,
  device 95,388 MiB, host RSS 84.7 GiB. The state (18.26 GB) is on /vault; its sha256 was verified before each
  evaluation.
- **GSM8K endings** (strict no-answer / cut at "The final…" / mean tokens; strict, flexible %):
  - control 19 / 6 / 167.5; 95.60, 95.91;
  - cluster FOCUS-256 35 / 11 / 166.5; 93.56, 94.31;
  - FOCUS-1k 115 / 34 / 147.6; 88.40, 90.90.
- **Paired** (zero-shot; GSM8K; ΔNLL Wiki / C4):
  - control vs cluster FOCUS-256: +0.22 (p=0.12); +2.05* (45/18, p=0.0009); −0.0058* / −0.0007;
  - FOCUS-1k vs control (same environment): +0.16 (p=0.29); −7.20* (22/117, p=8e-17); −0.0060* / −0.0014*;
  - FQ16-256 vs control: −0.19 (p=0.22); −0.45 (p=0.48);
  - FQ16-1k vs control: −0.02 (p=0.90); +0.30 (p=0.62).
- **Reading:** the control does not stop early; it does so less often than the cluster state. With the environment held
  fixed, FOCUS-1k stops early, so the window count (32 steps instead of 8) is the likelier cause. Each arm is one FOCUS
  run, and FOCUS's run-to-run noise applies: here the control and the cluster state differ by 2.05 GSM8K points.

### Part AA: the 2,048 panel stopped at its first job (2026-10-20 14:44 UTC, amendment 1's E0M3 gate)

- Llama-3.2-1B-Instruct FQ16 at 2,048 windows (rc 0, 414 s) has an E0M3 share of 27.0 %, above the 25 % gate. The
  driver stopped before any evaluation. The map is kept (RUN/calib_size/llama3.2-1b-instruct/fq16_2048/).
- **The share grows with the steps:** 2.8 % at 256, 14.8 % at 1,024, 27.0 % at 2,048. The growth is even across layer
  types (2,048: 25–31 %), and the count rises in every epoch without a plateau.
- **Within-epoch flips** grow from 84k to 827k (of about 950k tiles), so many tiles cross θ = 0 repeatedly. The training
  KL still decreases (0.0810 at the end of 1,024; 0.0805 at the end of 2,048).
- **Reported to the coordinator** with options: evaluate this map; raise or lift the gate; leave it TBD; stop. Nothing
  runs until the decision.

### Part AA, amendment 4 (registered 2026-10-20, at its commit, before anything ran after the stop; the user's choice, relayed): the E0M3 gate at 40 %; Llama-1B FQ16-2k evaluated; the 2,048 panel resumed

- **Gate:** for the rest of the 2,048 panel, an E0M3 share above **40 %** stops the run, instead of 25 %. Every other gate
  is unchanged (a failed check; any OOM, with no halving; the disk gate).
- **Evaluation and resume.** The 2,048 panel driver is restarted at once, with the new gate, on calib-size 898d00e. Its
  first jobs are Llama-3.2-1B-Instruct FQ16-2k's zero-shot, GSM8K (date pinned) and PPL. The finished calibration is
  not rerun. The result goes to the coordinator as soon as it is in: vs FourOverSix, vs FQ16-256 and FQ16-1k, and vs
  FOCUS-2k once that exists. The panel then continues as registered.
  - The restarted driver uses the size-tagged FOCUS log names in code. They are the same names the symlinks point at.
- **Report additions:**
  - per map and size: the E0M3 share, and the per-epoch E0M3 count and within-epoch flips from the trainer's
    report.json;
  - for the Llama-1B 2k map: the fraction of its E0M3 tiles whose final θ is within one step of the threshold
    (|θ| < 0.02 = lr).
- **θ for that fraction.** The paper's trainer saves θ only when a development set is drawn, and there is none here. So
  after the 2,048 panel, the Llama-1B FQ16-2k calibration is re-run once with a passive θ-saving launcher hook
  (`calibration/tmopt_launch.py --save-theta`, which records the trainer's own Adam parameters after its last step).
  - The hook is a flipquant calib-size commit on a side branch, fast-forwarded after the panel. It changes no setting and
    no evaluation code.
  - The rerun must reproduce the panel's map bit for bit (the trainer's map sha256, deterministic training), so its θ
    is the evaluated map's θ.

### Part AA: the 2,048 panel, the run (2026-10-20 16:14 UTC – 2026-10-22 01:24 UTC); every gate passed after amendment 4

- **E0M3 share at 2,048:** Llama-1B 27.0 % (amendment 4's evaluation), Phi-4 8.2, Mistral 10.7, Llama-8B 11.3, Nemotron
  11.2, Qwen3.8-27B 6.8, Qwen3-1.7B 17.2, Qwen3-8B 8.8. All are under the 40 % gate.
- **Qwen3.8-27B:** FQ16-2k ran with the host teacher (5.2 h, 90.52 GiB, as the probe), FOCUS-2k with the offload (64 steps
  × 560 s).
  - FOCUS-2k does **not** stop early: GSM8K 96.66, strict no-answer 7. That is +1.06* vs the local control and +8.26* vs
    FOCUS-1k.
  - With the environment held fixed, only the 1,024 run stops early. The amendment-3 reading "the window count is the
    likelier cause" is revised: the early stopping is specific to the FOCUS-1k run, which is consistent with FOCUS's
    run-to-run spread. It is not a trend in the window count.
- **Nemotron FQ16-2k:** GSM8K strict −2.27* vs FQ16-256, but flexible-extract −0.30 (p = 0.71). The loss is in the answer
  format only (strict no-answer 31 → 67).
- **Theta (amendment 4).** The Llama-1B FQ16-2k rerun with `--save-theta` (calib-size 9f19863, fast-forwarded after the
  panel) reproduces the panel's map bit for bit (6cd110a6…).
  - 7.9 % of its 256,459 E0M3 tiles have θ < 0.02, i.e. within one step of the threshold.
  - 4.2 % of all 950,272 tiles have |θ| < 0.02.
  - E0M3 θ quantiles (10 / 50 / 90 %): 0.026 / 0.184 / 0.730.
- **Logs:** every 2,048 FOCUS log has its own name, through the symlinks and the driver's new names. No earlier log was
  overwritten after 12:50 UTC on 2026-10-20.
- **GPU time of the part:** 59.6 h (job seconds in queue.log): FlipQuant calibration 20.5 h, FOCUS 21.2 h, GSM8K 11.7 h,
  zero-shot 4.7 h, PPL 0.9 h, probes 0.5 h.
- **Report:** results/paper_eval/calib_size/REPORT.md (findings, the pooled tables with and without Qwen3.8-27B, per model
  and size); STEP0.md, WINDOWS.md, PROBES.md, calib_size.json.

### Part AA, amendment 5 (registered 2026-10-22, at its commit, before any run; the user's request, relayed): GPTQ‡ at 1,024 / 2,048 windows

The user's request (fairness), relayed by the coordinator: recalibrate GPTQ‡ on the same 1,024 and 2,048 windows as
FQ16 and FOCUS, and evaluate it as those rows were.
- **Scope:** the eight part AA models × {1,024, 2,048}: 16 GPTQ‡ calibrations and 16 rows (zero-shot, GSM8K, PPL).
  Nothing else is rerun.
- **GPTQ‡ exactly as the main table.** The settings are part J's (`run_sm120.GPTQ_ROWS["nvfp4-fo6"]` with
  `gptq_flags`); for the two Llama-Instruct models, part P's command.
  - NVFP4 weights on the RTN-fixed grid; FourOverSix per-token activations (`--any-act`).
  - `--gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release`, block 128, damp 0.01.
  - Batch 8; Qwen3.8-27B 1 (batch1_only).
  - The model's calibration record and release revision.
  - The model's env (n16k64-fast for Nemotron and Qwen3.8-27B), no extra environment variables, native build_V.
  - **Only the windows change:** `--gptq-calib-extension RUN/calib_size/windows/<model>_<N>.json`. These are part AA's
    nested sets: the record's 128 windows, then N − 128 by the extension rule.
    - `flipquant.data.calibration_release` rebuilds every window and checks its token sha256 against the file.
    - Before each calibration, the file's token-list sha256 must equal calib_size.json's (WINDOWS.md's prefix), and the
      model's nesting checks must be ok.
    - The cache key records the windows' token-id sha256 (`data_sha256`).
- **Route, as the main table:** the codes are computed inside the row's PPL job. That job is `evaluation.ppl --mode
  native ... --paper-convention` with the GPTQ flags and `--gptq-cache` on /vault.
  - The PPL row is that job's record, as the main table's GPTQ‡ PPL is.
  - GPTQ's own time is the record's `ptq.seconds`. The job's device peak (the release sampler) covers GPTQ and the PPL
    evaluation.
  - Zero-shot (part X's row arguments) and GSM8K (part Y's; Llama-1B's date pinned, Qwen3 with thinking off) then run
    with the same flags and read the cache. flipquant refuses a cache whose key differs.
- **Code:** flipquant calib-size 8dbe2e4. It has 9f19863's code: 9f19863..8dbe2e4 adds results/calib_size and one
  .gitignore line. evaluation/, flipquant/, models/ and kernels/ are unchanged since caf69be (step 0 (a)).
- **Step 0, before anything else:** Llama-3.2-1B-Instruct GPTQ‡ at 256 on calib-size, with part P's command (part P's
  16x64 trainer report as the extension) and a cache in RUN/calib_size/step0/gptq_256/.
  - **Required:** codes_sha256 equal to part P's record, `952850d7…`. That is also the digest of the HF
    GPTQ-BF16Calib upload's codes and of part X step 0's. Otherwise the run stops and is reported.
  - **Also required; a difference stops the run:**
    - the per-window PPL NLL equals part P's record;
    - the zero-shot equals part X's GPTQ‡ row;
    - the GSM8K (date pinned) equals part Y's GPTQ‡ row.

    This is step 0 (c) as done for FQ16. With it, the existing records are this row's 256 references.
  - **Reported:** the cache file's sha256 against the uploaded file's (`5f65382f…`).
  - **Also checked:** the driver's GPTQ arguments for all eight models, with the main-table windows and codes files,
    against parts J / P's logged commands and parts X / Y's row arguments.
- **ETA gate:** next comes Llama-1B GPTQ‡-1k's calibration job, its PPL included. Its GPTQ seconds and peak give the
  ETA, which goes to the coordinator. If it is under about 20 h the run continues; otherwise it waits.
  - **Prior estimate:** GPTQ at 256 took 1.5 h for the eight models (`ptq.seconds`). If it is linear in the window
    count: about 5 h at 1,024 and 10 h at 2,048. The evaluations add about 6.5 h (part AA's FQ16 rows). Total: about
    20–23 h.
- **Order:** 1,024 for the eight models in part AA's ORDER, then 2,048 in the same order. Per model and size:
  calibration (with PPL), zero-shot, GSM8K.
- **Storage:** codes go to `/vault/flipquant_paper_eval/calib_size/<model>/gptq_<N>/codes_nvfp4_act-fo6.pt`. That is
  about 39 GB per size, 78 GB in all.
  - The sha256 is recorded right after the job that writes the file, from a fresh full read. It is verified before each
    later evaluation.
  - The job record goes to RUN/calib_size/<model>/gptq_<N>.job.json.
  - Nothing else is written on /vault. Nothing is deleted anywhere. /home keeps its 10 GB floor; step 0's 0.55 GB
    file stays in RUN.
- **Checks:**
  - calibration: rc 0; `ptq.loaded` false; the key's window count N, release calibration, BF16 Hessian input and
    propagation, damp 0.01, block 128; `gptq_calibration` names the windows file;
  - every evaluation: `ptq.loaded` true, `ptq.codes_sha256` equal to the calibration's, the cache path the /vault file.
    Otherwise parts X / Y's record checks apply; this codes check replaces their 256-codes check.
- **Gates (stop and report):**
  - a failed check;
  - any OOM. No setting is changed to get around one; batch 8 / 1 stays.
  - a predicted OOM: before each 2,048 calibration, the model's peak allocation is extrapolated linearly from its 256
    and 1,024 jobs. If it is above 91 GiB, the run stops before the calibration starts.
  - an implausible result, any of:
    - GPTQ's calibration output error is not below RTN's (`mean_output_error_vs_rtn` ≥ 1);
    - a PPL is non-finite or more than 5 % above GPTQ‡-256's;
    - the zero-shot average is more than 3 points below GPTQ‡-256's;
    - the GSM8K strict score is more than 10 points below GPTQ‡-256's.
- **Report additions:**
  - GPTQ‡ 1,024 / 2,048 rows in every per-model table, and their calibration cost;
  - "GPTQ‡-1k / 2k vs GPTQ‡ (256)", "GPTQ‡-2k vs GPTQ‡-1k", "FQ16-1k vs GPTQ‡-1k" and "FQ16-2k vs GPTQ‡-2k", per model
    and pooled over the eight models;
  - calib_size.json, REPORT.md and the flipquant README, re-rendered and scanned.
  - Push paper-eval and flipquant calib-size only.

### Part AA, amendment 5: step 0 and the ETA gate, the run (2026-10-22 08:37–08:50 UTC); every check passed

- **Arguments.** For all eight models, with the main table's windows and codes files, the driver's PPL command is
  identical to the command the main table ran (parts J / P's job logs). The zero-shot and GSM8K arguments are identical
  to parts X / Y's GPTQ‡ rows'.
- **Llama-3.2-1B-Instruct GPTQ‡-256 on calib-size 8dbe2e4:**
  - codes `952850d7…`, equal to part P's (required);
  - the codes file `5f65382f…`, byte-identical to the HF GPTQ-BF16Calib upload;
  - GPTQ 122 s, job 141 s, device peak 7,152 MiB.
- **Its evaluations, reading that file:**
  - PPL per-window NLL equal to part P's record on WikiText-2 and C4;
  - zero-shot equal to part X's GPTQ‡ row on all six tasks;
  - GSM8K (date pinned) 2,638 / 2,638 samples identical to part Y's.
- **ETA gate, Llama-1B GPTQ‡-1k:**
  - GPTQ 150 s, job 169 s; device peak 10,236 MiB; allocated 6.87 GiB (4.90 at 256).
  - Codes `af0ad75b…`; output error 0.476 × RTN; WikiText-2 15.473, C4 21.101.
  - GPTQ's time is mostly the per-column solve, which does not depend on the window count.
  - The driver's ETA for the rest is 11.8 h. Allowing batch-1 Qwen3.8-27B to scale with its FLOPs, it is about 13 h.
  - That is under 20 h, so the panel continued at 08:51 UTC without waiting; the ETA went to the coordinator.

### Part AA, amendment 5: the panel, the run (2026-10-22 08:51–20:12 UTC); every gate passed

- **Rows:** all 16 (8 models × 1,024 / 2,048) were calibrated and evaluated. Every check passed:
  - the windows' token lists equal calib_size.json's and WINDOWS.md's; the audits pass;
  - each codes file's sha256 was recorded at write and verified before each of its evaluations (34 verifications,
    step 0 included);
  - every evaluation read its row's codes file (`ptq.loaded`, codes digest equal to the calibration's);
  - parts X / Y's record checks pass.
- **Plausibility gates:** none fired.
  - Output error 0.47–0.58 × RTN.
  - Every PPL within +5 % of GPTQ‡-256's. The largest increase is Llama-3.1-8B-Instruct at 2,048 on WikiText-2, +2.8 %.
  - The largest zero-shot drop is Qwen3.8-27B at 1,024, −0.58.
  - The largest GSM8K drop is Nemotron at 2,048, −4.10 strict; its strict no-answer count goes from 60 to 87.
- **Memory.** The extrapolation gate passed for every model, with two caveats.
  - Qwen3.8-27B: its 256 and 1,024 peaks (65.94 GiB) come from a phase that does not depend on the window count, so
    the extrapolation (65.9 GiB) is below its 2,048 peak: 73.17 GiB allocated, device 90,656 MiB of 97,887. The layer
    inputs' copies set that peak, as estimated before the run (about 74 GiB).
  - Nemotron: its 256 job predates part J's amendment 12 memory fix, so its extrapolation (21.1 GiB) means nothing.
    Its 2,048 peak was 35.42 GiB.
- **Time.** GPTQ at 1,024 / 2,048:
  - Llama-1B 150 / 189 s; Phi-4 1,229 / 1,676 s; Mistral 792 / 1,047 s; Llama-8B 790 / 1,045 s;
  - Nemotron 752 / 1,030 s; Qwen3.8-27B 2,580 / 3,594 s; Qwen3-1.7B 221 / 274 s; Qwen3-8B 775 / 1,021 s.
  - Amendment 5's GPU time, step 0 included: 11.4 h (calibration with its PPL 5.1 h, zero-shot 1.8 h, GSM8K 4.5 h).
    The ETA was 11.8 h, about 13 h with the Qwen3.8-27B allowance.
- **Results, pooled over the eight models:**
  - GPTQ‡ vs its 256: GSM8K +0.08 / −0.38 (n.s.) at 1,024 / 2,048; zero-shot +0.15* / +0.32*.
  - FlipQuant 16x64 vs GPTQ‡ at the same size: GSM8K −0.69 (p = 0.059) / +0.83* / +0.99* at 256 / 1,024 / 2,048;
    zero-shot +0.80* / +0.74* / +0.62*.
  - ΔNLL favours FlipQuant beyond 2 SE for every model, size and corpus.
- **Report:** results/paper_eval/calib_size/: REPORT.md (GPTQ‡ rows, comparisons, cost), STEP0.md (amendment 5's
  step 0), calib_size.json (`gptq`: step 0, ETA, memory, GPU time).
- **Codes on /vault:** `/vault/flipquant_paper_eval/calib_size/<model>/gptq_<N>/codes_nvfp4_act-fo6.pt`, 16 files,
  77.6 GB. Each file's sha256 is in its job record.

## Part AB (registered 2026-10-23, at its commit, before anything is staged or uploaded; the user's decision, relayed): the part AA 1,024-window artifacts on Hugging Face

The user decided this in the coordinator's session on 2026-10-23; the coordinator relayed it. **24 new PRIVATE
repositories** under `edgeai-lab`, one of each kind for each of the eight part AA models. No existing repository is
touched, including the co-author's `<Base>-GPTQ-NVFP4` repositories.
- **Names (the user's):** the existing repositories' base names with `-Calib1k`. The base names are
  Llama-3.2-1B-Instruct, phi-4, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, NVIDIA-Nemotron-Nano-9B-v2,
  Qwen3.8-27B, Qwen3-1.7B and Qwen3-8B.
  - `<Base>-FlipQuant-Calib1k`: the FQ16-1k map (16x64) only, for now.
  - `<Base>-FOCUS-NVFP4-Calib1k`: the FOCUS-1k state, `focus.pt`.
  - `<Base>-GPTQ-BF16Calib-NVFP4-Calib1k`: the GPTQ‡-1k codes, `codes_nvfp4_act-fo6.pt`.
- **Sources, each verified against its part AA job record before staging (a fresh full read):**
  - **map:** `RUN/calib_size/<model>/fq16_1024/map.pt`, sha256 = the record's `map_file_sha256`.
    - Its meta is cleaned as for the earlier FlipQuant repositories: the run and calibration-record paths are made
      relative, the GitHub owner is removed from the extension reference, and the reference implementation's branch
      and commit are removed.
    - The tiles and every tensor record must stay bit-identical; both sha256 are recorded.
  - **FOCUS:** the evaluated state as is, sha256 = the record's `state_sha256`. The pilot's two states (Llama-1B, Phi-4)
    are in RUN; the others are on /vault.
  - **GPTQ‡:** the /vault codes file as is, sha256 = the record's `cache_sha256`. Its code digest must equal the
    calibration's `codes_sha256`, and its key must be the main-table variant at 1,024 windows.
- **Staging** goes to `/home/dev/calib1k_hf/<repo>/`: the card, `ppl_summary.json` and the license files are written
  there. The large file is a symbolic link to its verified source, because /home's 10 GB floor does not allow a copy.
  Every scan, hash and upload reads the source through the link. The map is small and is written there, cleaned.
- **Cards:** the layouts of the existing FlipQuant, FOCUS and GPTQ-BF16Calib cards, stating:
  - **Calibration:** 1,024 nested windows (the record's 128 plus 896 by the extension rule), the builder and the
    13-gram audit (0 hits; positive control). FQ16 at 640 steps (the release settings, 5 epochs), FOCUS at 32 steps
    (FOCUS_COMMON), GPTQ‡ on the same windows with the main-table settings.
  - **Results:** the row's part AA results: PPL (WikiText-2, C4), zero-shot (six tasks) and GSM8K, each against its
    256-window version and against FourOverSix. The paired statistics are part AA's (exact McNemar; ΔNLL ± 2 SE).
  - **E0M3 share:** for FlipQuant.
  - **Cost:** of the calibration.
  - **Licenses:** the base model's license files, as before. The Llama models also get "Built with Llama",
    `USE_POLICY.md` and `NOTICE`.
  - **Qwen3.8-27B FOCUS-1k** is the evaluated state as is (the user's choice). Its card states:
    - the early stopping on GSM8K (88.40, strict no-answer 115), and that it is specific to this run: the local 256
      control and the 2,048 run do not stop early;
    - the local training on one GPU with host-offloaded AdamW, as a deviation from the 2 x H200 run of its 256
      version.
- **Gates, as for the earlier uploads; a failure stops the part and is reported:**
  - **Scan:** the earlier patterns (local paths, user names, host names, tokens, e-mail addresses, pointers to a
    reference implementation) over every staged file: text lines, binary metadata strings, archive record names and
    raw bytes.
    - The map and the FOCUS state may have no hit at all.
    - A codes file follows the GPTQ upload's binary rule: a raw-byte hit is accepted only inside q / s tensor storage,
      with no metadata hit, the file sha256 equal to its record, and the exact-token check clean.
    - Text files may have no hit except the kinds accepted before: Meta's reporting address in `USE_POLICY.md`, and
      "commit" / "branch" in the FlipQuant card's own sentences.
  - **Exact-token check:** every staged file's bytes against this machine's HF tokens and their 15-character
    prefixes; 0 hits. The tokens are never printed, logged or written.
  - **Account and names:** the active account is `edgeai-lab`, and none of the 24 names exists.
  - **Privacy before upload:** each repository is created private and checked private (anonymous HTTP 401) before
    any file goes up.
  - **Verification:** every file's sha256 equals the staged file's, by a download into /dev/shm (files up to 1 GiB)
    or by the Hub's LFS sha256 and size (larger files). Also checked: the card metadata, the Hub's YAML validation,
    and the README equal to the staged one.
- **Visibility:** this session makes nothing public.
  - When the 24 uploads are verified, the list and its anonymous status go to the coordinator, and the user changes
    the visibility.
  - The record (`results/paper_eval/hf_calib1k/HF_UPLOAD.md`, with the cards and records) is written and paper-eval
    pushed after the user's change is confirmed, as for the earlier uploads.
- **Order:** Llama-3.2-1B-Instruct first (the smallest, one of each kind), then the others; within a model the map,
  then FOCUS, then GPTQ‡.
- **Code:** `experiments/paper_eval/hf_calib1k.py`. It reuses `focus_hf.py`, `llama_hf.py`, `hf_gptq_focus.py` and
  `hf_token_check.py`, and is CPU and network only (no GPU).

### Part AA, amendment 6 (registered 2026-10-23, at its commit, before any run; the user's request and decisions, relayed): FlipQuant 8x64 and 256x64 at 1,024 windows

- **Request:** FlipQuant 8x64 and 256x64 at 1,024 windows on the eight part AA models, evaluated as the FQ16-1k rows.
- **Scope:**
  - 16 calibrations and 16 rows: 8 models × {8x64, 256x64}, each with zero-shot, GSM8K and PPL.
  - 4 GSM8K rows of the 256-window release maps: Qwen3-1.7B / Qwen3-8B × {8x64, 256x64}, with amendment 1's Qwen3
    GSM8K protocol. With them, "vs its 256" exists for every model on every evaluation. The six other models have
    part Y's rows; zero-shot (parts X / X2) and PPL (the final table) exist for all eight.
- **Settings:** the release hyperparameters at 1,024 windows, as FQ16-1k:
  - `calibration.train_map --unit <8x64 | 256x64> --epochs 5 --teacher-topk 1000 --fit-windows 1024`;
  - 640 steps; Adam lr 0.02, initial logit −1, seed 0; batch 8 (Qwen3.8-27B and Nemotron 2 × 4); deterministic;
  - the same 1,024 windows: each run's fit extension must equal `windows/<model>_1024.json`, record by record;
  - Qwen3.8-27B with `--teacher-host`.
- **Code:** flipquant calib-size 394e683, which is 9f19863's code; only results have changed since.
- **Step 0, before the panel:** Llama-1B at the release 256 windows on calib-size.
  - **Calibration:** each tile's map is recalibrated with part P's command. Its run_map_sha256 and tiles must equal
    part P's (required).
  - **Evaluation:** part P's 8x64 and 256x64 maps are evaluated on calib-size. Their zero-shot, GSM8K (date pinned)
    and PPL must be identical to parts X / Y / P (required), as step 0 (c) checked for 16x64. Then the existing
    records are the 256-window references.
- **Evaluation:** as the FQ16-1k rows, native on build_V:
  - part X's zero-shot;
  - part Y's GSM8K, with Llama-1B's date pin and Qwen3's thinking off;
  - PPL with the paper convention;
  - the kernel families of part X's rows: mixed_wB_ko for 8x64, mixed256_ko for 256x64.
- **Gates:** as before (a failed check; any OOM, with no setting changed; the /home floor). The exception is the E0M3
  share: above 60 % it stops the run (the user's choice for these two tiles).
- **Order:** per model in part AA's ORDER.
  - For each Qwen3 model, its two GSM8K references come first.
  - Then 8x64, then 256x64: each calibration followed by its zero-shot, GSM8K and PPL.
  - After the first model (Llama-1B), the measured ETA goes to the coordinator, and the run continues.
- **Parallel with part AB** (the user's decision, relayed):
  - part AB stays CPU / network only;
  - this amendment reads its windows and models from /home as before;
  - both processes watch /home's 10 GB floor;
  - if part AB's /vault reads slow the GPU jobs noticeably, the coordinator is told, and neither is changed silently.
- **Report**, per model and pooled, as part AA:
  - each tile at 1,024 against its own 256: FQ8-1k vs FQ8-256, FQ256-1k vs FQ256-256;
  - against FourOverSix, FQ16-1k, FOCUS-1k and GPTQ‡-1k;
  - the E0M3 share and the per-epoch churn (E0M3 tiles, flips) per tile and size;
  - the calibration cost.

  calib_size.json, REPORT.md and the flipquant README are updated, and paper-eval and flipquant calib-size are pushed.
- **HF:** the new maps go into no HF repository without the coordinator's OK.

### Part AA, amendment 6: the run (2026-10-23 08:25 UTC – 2026-10-24 05:37 UTC); every gate passed

- **Step 0 (08:25–08:45 UTC): passed.** Llama-1B 8x64 and 256x64 at 256 windows on calib-size 394e683:
  - each map recalibrated with part P's command has the same trainer map sha256 as part P's, and identical tiles;
  - part P's maps of both tiles, evaluated on calib-size, give zero-shot (six tasks), GSM8K (2,638 / 2,638 samples,
    date pinned) and PPL identical to parts X / Y / P.
- **Panel:** all 16 rows (8 models × 8x64 / 256x64 at 1,024) and the 4 Qwen3 GSM8K 256 references. Every check
  passed: each run's fit extension equals the windows file, and parts X / Y's record checks with the kernel families
  mixed_wB_ko / mixed256_ko.
- **E0M3 share at 1,024** (the gate is 60 %):
  - 8x64: 2.3–11.3 %;
  - 256x64: 7.0–39.7 %, the highest Llama-1B's 39.7 %.
  - Every tile's share grows 4–9× from its 256-window map.
- **Memory:** Qwen3.8-27B with the host teacher peaked at 91.0 GiB allocated and 95,500 MiB on the device (8x64); no
  OOM.
- **ETA:** 21.5 h after Llama-1B; the panel took 20.5 h. Part AB's parallel staging and uploads did not slow the GPU
  jobs: Llama-1B's 8x64 calibration took 223 s, the same as its FQ16-1k.
- **GPU time:** 21.2 h (calibration 13.7 h, zero-shot 1.9 h, GSM8K 4.5 h, PPL 0.3 h, the Qwen3 references 0.8 h).
- **Results, pooled over the eight models (8x64 / 256x64):**
  - vs their 256: GSM8K +0.77* / −0.26, zero-shot −0.03 / +0.09;
  - vs FQ16-1k: GSM8K −0.63 (p = 0.063) / −1.39*, zero-shot −0.06 / −0.24*;
  - vs FourOverSix: GSM8K +1.54* / +0.79*, zero-shot +0.77* / +0.58*;
  - vs GPTQ‡-1k: GSM8K +0.20 / −0.56, zero-shot +0.68* / +0.50*;
  - vs FOCUS-1k: GSM8K +1.26* / +0.50 (without Qwen3.8-27B: +0.38 / −0.58), zero-shot −0.22* / −0.41*.

  256x64-1k's NLL is above FQ16-1k's beyond 2 SE on seven of the eight models; 8x64-1k is level with FQ16-1k.
- **Report:** results/paper_eval/calib_size/ (REPORT.md, STEP0.md, calib_size.json `a6`).
- **HF:** the new maps are in no HF repository (the coordinator's condition).

### Part AB: the run (2026-10-23 07:45 UTC – 2026-10-24 08:52 UTC); all 24 repositories uploaded, verified and public

- **Staging:** all 24 sources equal their job records (fresh sha256). Maps: the meta cleaned, tiles and tensor records
  bit-identical. FOCUS states and codes files: symbolic links to the verified sources.
- **Gates:**
  - Scans: maps and FOCUS states have no hit; text files only the kinds accepted before.
  - Codes files: every raw-byte hit inside `q` / `s` storage, no metadata hit. Four coincidental byte runs of a
    non-token pattern (three project-name, one user-name; recorded by class and location) are accepted under the
    earlier rule's extension.
  - Exact-token check: 0 hits on all 111 staged files.
  - Account `edgeai-lab`; none of the 24 names existed.
- **Uploads:** each repository was created private and checked private before upload. Every file's sha256 equals the
  staged file's (a download up to 1 GiB, else the Hub's LFS sha256 and size), and all 24 cards pass the Hub's YAML
  validation with README = staged.
- **Events:**
  - The 24th repository's commit hit the Hub's commit rate limit (HTTP 429), with upload_large_folder retrying about
    twice a second. The coordinator authorized stopping the uploader (12:02 UTC) and one throttled attempt per
    10 minutes (at most 6).
  - The first attempt (13:08 UTC) was refused with HTTP 400, the account's private storage limit (86.7 GB in the 23
    verified repositories); the decision went to the user.
  - After the 23 were made public, the 24th committed at its first attempt (08:49 UTC on 2026-10-24) and was verified
    private.
- **Visibility:** the user asked for the repositories to be public. The coordinator made the change in its own
  session at the user's direct request (the 23 first, the 24th at 08:51 UTC); this session changed no visibility.
  This session's token-free check at 08:52 UTC: all 24 public, each with its file list, README and head equal to the
  verified upload.
- **Record:** results/paper_eval/hf_calib1k/ (HF_UPLOAD.md, the cards, and the records sanitized: no raw-byte match
  is quoted). The record files were checked in-process: 0 exact-token, prefix or token-pattern hits.

### Part AB, amendment 1 (registered 2026-10-24, at its commit, before anything is staged or uploaded; the user's request, relayed): the amendment 6 maps on Hugging Face

- **Request:** add part AA amendment 6's maps (FlipQuant 8x64 and 256x64 at 1,024 windows; 8 models, 16 files) to the
  existing `edgeai-lab/<Base>-FlipQuant-Calib1k` repositories, which hold only `flipquant_16x64.pt` and are public, in
  the layout of the 256-window `<Base>-FlipQuant` repositories.
  - Every commit to these repositories is public immediately (the user's request).
  - No other repository is touched, and no visibility is changed.
  - If the permission system refuses a commit to a public repository, the part stops and is reported; nothing works
    around it.
- **Sources,** verified against their amendment 6 job records (a fresh sha256): `RUN/calib_size/<model>/fq8_1024/map.pt`
  and `fq256_1024/map.pt`. The meta is cleaned as for the 16x64 map (paths relative, owner and reference
  implementation's branch / commit removed), with the tiles and every tensor record bit-identical.
- **Cards:** `README.md` and `ppl_summary.json` regenerated for the three tiles, with the numbers from part AA's record
  at paper-eval 5dcb1ff:
  - the three maps' sha256 and E0M3 shares;
  - the settings of the two new tiles (640 steps, the same 1,024 windows);
  - per tile: PPL, zero-shot and GSM8K against its 256-window version, against FourOverSix, and (8x64, 256x64)
    against the 16x64 map;
  - a note on 256x64: its high E0M3 share and its PPL against the 16x64 map, with significance as measured;
  - the file table with every sha256.

  The 16x64 map and the license files are unchanged.
- **Gates, as for part AB:**
  - each source's sha256 equals its job record; tiles and tensor records bit-identical;
  - the scans of every staged file (the maps' metadata, names and raw bytes: no hit; text files: only the accepted
    kinds);
  - the exact-token check (0 hits).

  Anything outside the rules stops the part.
- **Upload:** one commit per repository (8 in all), adding the two maps and updating README.md and ppl_summary.json.
  The retries on HTTP 429 are gentle: at most one commit attempt per 10 minutes, at most 6 per repository, and any
  other error stops.
- **Verification, without a token:**
  - the file list;
  - every file's sha256 by download (all files are under 1 GiB);
  - README equal to staged;
  - the head;
  - the Hub card check.
- **Record:** an addendum to `results/paper_eval/hf_calib1k/HF_UPLOAD.md` with its records; push paper-eval.

### Part AB, amendment 1: stopped at the first upload (2026-10-24 ~09:55 UTC); the permission system refused committing to a public repository

- **Staging and gates passed** for all eight `<Base>-FlipQuant-Calib1k` repositories:
  - every map's sha256 equals its amendment 6 job record;
  - meta cleaned, tiles and tensor records bit-identical;
  - the uploaded 16x64 map reproduced byte for byte;
  - scans only the accepted kinds;
  - exact-token check 0 hits on 53 files;
  - each repository is this part's, verified and public.
- **The first commit** (Llama-3.2-1B-Instruct-FlipQuant-Calib1k) was refused by this session's permission system ("Create
  Public Surface") before the command ran.
  - As registered and as the coordinator asked, the part stopped there. Nothing was worked around, and no other session
    was asked to run it.
  - A token-free check afterwards: all eight repositories' heads unchanged, and none holds an 8x64 or 256x64 map.
- **The decision is the user's.** The staged files (`/home/dev/calib1k_hf/<repo>/`, records `stage_a1.json`,
  `token_check_a1.json`) are kept for it.

### Part AB, amendment 1: the run after the user's authorization (2026-10-24 14:16–14:18 UTC); all eight repositories updated and verified

- **Authorization:** the user authorized it directly in this session after the stop. The scope was only the eight
  `<Base>-FlipQuant-Calib1k` repositories; per repository the two maps added and README.md / ppl_summary.json updated,
  exactly as staged; no other repository, file or visibility change.
- **Run:** the prepared command re-checked every gate (staged files unchanged, exact-token record clean, each
  repository this part's and public). Then it made one commit per repository (`Add the 8x64 and 256x64 maps (1,024
  windows); update the card`), each committed at its first attempt, with no HTTP 429.
- **Verification, without a token, all eight pass:**
  - the file list equals the staged list;
  - every file downloaded has sha256 equal to staged;
  - the card metadata is correct;
  - the new head's commit is this commit.
- **Card check:** all eight pass the Hub's YAML validation, README equals the staged one, and a local render succeeds.
  All eight repositories are public, as before.
- **Record:** the addendum in `results/paper_eval/hf_calib1k/HF_UPLOAD.md`, with `stage_a1.json` (sanitized),
  `upload_a1.json`, `cards_a1.json`, `token_check_a1.json` and the updated cards. The record files were checked
  in-process: 0 exact-token, prefix or token-pattern hits.
