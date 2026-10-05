# flipquant GEMM parity: flipquant's real path against kernel-opt's adopted kernels (Llama-3.1-8B)

**The request.** The user, through the coordinator, 2026-10-05: does the GEMM in flipquant's code run as fast as the
kernels optimized here? This is a check, not a registered study.

**Summary.**
- flipquant's real path runs exactly the kernels kernel-opt adopted, the same way: the same builds, widths, scheduler
  rows, SASS, operands and outputs.
- Its GEMM time per forward matches within 0.7 % at T ≥ 16 in every run.
- The two timing criteria as asked still fail in every run that includes the flipquant process, on a few µs-scale
  cells at T = 1 and 16:
  - at T = 1 one unit is about 2 % *faster* on flipquant;
  - two or three k/v_proj cells at T = 16 are about 3 % slower.
- These cells are an effect of that process's state, not of its GEMM path. It moves between kernel sets with an
  unrelated `torch.equal`, and it never appears between two RaZeR processes.

## What ran

- **flipquant** (main b1c4123, the clean worktree `fqport/wt`; nothing there was modified):
  - per policy, its own CLI plumbing: `models.cli.build`, which runs `models.loader.load` and then
    `models.razer.install`;
  - options: `--mode native --native-backend razer --razer-build-dir kernels/razer_sm120/build_ko --razer-kernel auto`
    (`auto_stock_wB` for stock_wB_ko), the paper maps (`map.pt`) and the paper checkpoint revision;
  - the timed modules are the NativeLinears that install put into the model.
- **RaZeR** (kernel-opt; the adopted sets of ae389a5, in build_V): the deviation-2 harness's construction (M1).
  NativeLinear is built from the paper artifact, on the set `mixfp4_sm120.model.resolve_kernel` gives (`auto`,
  `auto_stock`, `auto_stock_wB`).
- **Grid:**
  - policies: the 8x64, 16x64 and 256x64 TM-OPT+TC maps (mixed_wB_ko, mixed_ko, mixed256_ko), and FourOverSix weights
    on stock_ko and stock_wB_ko;
  - M1's module per projection: a map's typical module, FourOverSix's first;
  - T ∈ {1, 16, 128, 512, 2048, 8192}: 210 cells.
- **Timing:** deviation-2, the same code on both sides.
  - Cold weights: rotation copies larger than 4× the L2, and a 512 MiB flush.
  - The quantizer runs after the flush and is not timed. The value is the CUPTI time of the GEMM, 3 rounds × 30
    launches.
  - `experiments/kernel_opt/flipquant_parity_gemm.py` drives two persistent worker processes cell by cell. Each
    policy's two sides run back to back, in alternating order.
- **Runs** (2026-10-05, 11:25–12:44 UTC, one at a time on the idle GPU):
  - **1, r2, r3:** the comparison, repeated with fresh processes.
  - **nochk:** timing only, without the per-cell checks.
  - **aa:** two RaZeR harness processes (the A/A control).
  - **rzmrz:** RaZeR's own model path (`eval/common.load_model`, then `mixfp4_sm120.model.install`) against the
    harness.
  - **fqrzm:** flipquant against RaZeR's model path.
  - **fqV:** flipquant loading build_V.
  - **rzK:** both sides on build_ko.

## Identity: what the GEMM runs is the same (every run with the per-cell checks)

- **Per cell:**
  - the same build, width and scheduler row on both sides, in every round, from tables with the same content;
  - the same GEMM and quantizer kernel names (CUPTI), which are also the ones NativeLinear's fused forward launches;
  - the activation quantizer is the picked build's own, in the same mode.
- **Bit for bit:** operands (packed codes, placed scales, global scale) and outputs; the isolated launch equals the fused
  forward.
- **Binaries** (`flipquant_parity_binaries.py`): the same SASS (manifest and recomputed), device code and host code,
  with the same defines and configuration. This holds for the 15 builds the cells ran (`runs/binaries_<run>.json`) and
  for all 22 builds of build_V against build_ko (`binaries_build_ko_vs_build_V.json`).
  - The files are not byte-identical. They differ only in nvcc's per-compilation id inside internal symbol names (the
    cubin's string table, the host's `.rodata` / `.nv_fatbin`), in the static symbol table and in the build-id.

## Timing

Per-forward GEMM time: the sum over the projections of (modules) × (the median time of the timed module). µs values are
run 1's. Bold: outside ±1 %. The A/A column is the second process over the first.

**8x64 map, mixed_wB_ko**

| T | RaZeR (µs) | flipquant (µs) | fq / rz: 1 | r2 | r3 | nochk | A/A control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,323 | 3,322 | 1.000 | 1.000 | 0.999 | 0.998 | 0.999 |
| 16 | 3,313 | 3,319 | 1.002 | 1.002 | 1.002 | 1.001 | 1.001 |
| 128 | 3,568 | 3,568 | 1.000 | 1.000 | 1.001 | 0.999 | 1.000 |
| 512 | 7,693 | 7,694 | 1.000 | 1.003 | 1.001 | 1.000 | 1.001 |
| 2048 | 25,144 | 25,143 | 1.000 | 1.003 | 1.000 | 1.001 | 1.002 |
| 8192 | 94,793 | 94,688 | 0.999 | 1.001 | 1.000 | 1.001 | 1.002 |

**16x64 map, mixed_ko**

| T | RaZeR (µs) | flipquant (µs) | fq / rz: 1 | r2 | r3 | nochk | A/A control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,308 | 3,309 | 1.000 | 0.999 | 1.000 | 0.994 | 1.000 |
| 16 | 3,298 | 3,296 | 0.999 | 0.999 | 1.000 | 1.000 | 0.998 |
| 128 | 3,593 | 3,592 | 1.000 | 0.998 | 0.998 | 0.998 | 1.000 |
| 512 | 7,224 | 7,202 | 0.997 | 1.001 | 0.997 | 1.001 | 1.003 |
| 2048 | 23,806 | 23,778 | 0.999 | 1.005 | 1.000 | 1.000 | 1.001 |
| 8192 | 89,202 | 89,052 | 0.998 | 1.000 | 0.998 | 1.002 | 1.000 |

**256x64 map, mixed256_ko**

| T | RaZeR (µs) | flipquant (µs) | fq / rz: 1 | r2 | r3 | nochk | A/A control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,285 | 3,217 | **0.979** | **0.979** | **0.979** | 0.999 | 1.001 |
| 16 | 3,276 | 3,286 | 1.003 | 1.002 | 1.003 | 1.000 | 1.001 |
| 128 | 3,532 | 3,529 | 0.999 | 0.999 | 0.999 | 1.001 | 1.001 |
| 512 | 7,096 | 7,099 | 1.000 | 1.005 | 1.005 | 1.000 | 1.002 |
| 2048 | 23,855 | 23,837 | 0.999 | 1.007 | 1.000 | 1.000 | 1.002 |
| 8192 | 89,385 | 89,388 | 1.000 | 0.999 | 1.001 | 1.000 | 1.000 |

**FourOverSix, stock_ko**

| T | RaZeR (µs) | flipquant (µs) | fq / rz: 1 | r2 | r3 | nochk | A/A control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,277 | 3,273 | 0.999 | 0.998 | 0.999 | **0.977** | 1.001 |
| 16 | 3,260 | 3,261 | 1.000 | 1.001 | 1.001 | 1.005 | 1.001 |
| 128 | 3,502 | 3,498 | 0.999 | 1.000 | 1.000 | 1.000 | 1.001 |
| 512 | 6,952 | 6,938 | 0.998 | 1.000 | 0.999 | 1.001 | 1.000 |
| 2048 | 23,453 | 23,463 | 1.000 | 1.006 | 1.001 | 1.000 | 1.000 |
| 8192 | 87,822 | 87,728 | 0.999 | 1.001 | 1.000 | 1.000 | 1.001 |

**FourOverSix, stock_wB_ko (weights on B)**

| T | RaZeR (µs) | flipquant (µs) | fq / rz: 1 | r2 | r3 | nochk | A/A control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,614 | 5,620 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 |
| 16 | 5,432 | 5,465 | 1.006 | 1.006 | 1.006 | 1.005 | 1.001 |
| 128 | 4,962 | 4,960 | 1.000 | 1.000 | 0.999 | 1.001 | 1.000 |
| 512 | 7,327 | 7,301 | 0.996 | 0.997 | 0.996 | 0.998 | 0.999 |
| 2048 | 23,413 | 23,385 | 0.999 | 1.003 | 1.000 | 0.999 | 1.003 |
| 8192 | 87,752 | 87,633 | 0.999 | 0.999 | 1.000 | 1.000 | 1.002 |

The control pairs and every cell are in `parity.md` and `cells.md`.

## Criteria (as asked)

| criterion | runs with the flipquant worker (1, r2, r3, nochk, fqrzm, fqV, rzK) | RaZeR-only pairs (aa, rzmrz) |
|---|---|---|
| identical kernel and width selection | **pass** (every run with checks; nochk records the same launches per round) | pass |
| identical binaries | **pass** for SASS, device code and host code; the files differ only as described above | pass |
| per-forward GEMM sum within ±1 % per (unit, T) | **FAIL** at T = 1 only: one unit per run is 1.9–2.3 % *faster* on flipquant (256x64 with the checks, stock_ko without). T = 16 within 0.6 %, T ≥ 128 within 0.7 % | pass (≤ 0.35 %) |
| no cell slower in every round by > 2 % | **FAIL:** 2–3 cells per run (k/v_proj at T = 16, +2.5 to +3.4 %); the same runs have 4 cells *faster* by 3–5 % at T = 1 | pass |

Over all 210 cells the median flipquant / RaZeR ratio is 1.000 in every run.

## What the failing cells are (`diagnostics.md`)

- **The same signature every time.** The k = 4096 projections of one policy at small T: 3–5 % faster at T = 1, about
  3 % slower for k/v_proj at T = 16.
- **Where it appears.** Only in the flipquant worker, and only after every policy is installed: with 256x64 alone it
  is level. It is reproducible run to run.
- **Which policy gets it depends on the allocation sequence.**
  - With the per-cell checks it is 256x64; without them, stock_ko and 16x64.
  - Bisecting the check shows that one `torch.equal` of two outputs moves it. That call changes no kernel input and no
    launch.
- **What it does not follow:**
  - the library files: flipquant on build_V keeps it, and RaZeR on build_ko stays level;
  - where the timed buffers sit:
    - at T = 1 they have the same offsets with and without the trigger; at T = 16 only the input moves;
    - moving every one of them into fresh segments keeps it;
  - model loading: RaZeR's own model path (five model loads and installs, like flipquant's) is level against the
    harness, and flipquant keeps the effect against it.
- **In one RaZeR process,** placing the same buffers or code elsewhere moves these cells by at most about 3 %, never by
  5 %. The tests: the rotation copies, the activation buffers (24 offsets) and six instances of the same builds.
- **Not identified further.** The cause lies below what the process can see, for example where memory or code lands
  physically. ncu was not used.
- **Size.** At T = 1 the whole per-forward GEMM is about 3.3 ms, so the 2 % is about 70 µs per forward. It is in
  flipquant's favour in every run.

## flipquant's own `evaluation.latency gemm` cannot time the adopted sets

The change below was not made; the user decides.
- **The paper sets are hard-coded.** `_run_gemm` and `run_gemm_warm` (`evaluation/latency.py` at b1c4123, lines 544
  and 728) use
  `dict(stock=KernelSet("stock"), mixed=KernelSet("mixed"), stock_wB=Kernel.load("stock_wB"), n8k64_wB=Kernel.load("n8k64_wB"))`.
  - There is no `--razer-kernel` or `--razer-build-dir`; the build directory comes only from `SM120_BUILD_DIR`.
  - The 16x64 and 256x64 rows share the key `mixed`.
- **`isolated_gemm` passes no scheduler row** (lines 466–477). With the adopted sets' tables it would time a different
  launch from NativeLinear's. Its bitwise check would still pass, because the scheduler changes no output bit.
- **The minimal change:**
  1. `--razer-kernel {paper,auto}` (default `paper`, so the paper's numbers stay reproducible) and `--razer-build-dir`
     on `gemm` and `gemm-warm`.
  2. With `auto`, build each configuration's set with `models.razer.resolve_kernel`:
     - `auto` for `fourover6` gives stock_ko;
     - `auto_stock_wB` gives stock_wB_ko;
     - `auto` for `mixfp4` with unit 8x64 / 16x64 / 256x64 gives mixed_wB_ko / mixed_ko / mixed256_ko;
     - the 256x64 rows need their own key.
  3. Pass `schedule=lin.kernel_set.schedule(n, k, t)` (`(0, 1)` without a set) to both `gemm_ptr` calls in
     `isolated_gemm`.
  4. Record the width and the scheduler row per row.
- `gemm-warm` times NativeLinear's forward, which passes the scheduler row already, so it needs only (1) and (2).

## Files

- `parity.md`: every run's per-unit ratios, the criteria per run, the cell-level spread per run.
- `cells.md`: every cell of every run.
- `parity.json`: the summary.
- `runs/`: the raw records, `gemm_<run>.json.gz` and `binaries_<run>.json`.
- `binaries_build_ko_vs_build_V.json`: every build of the two deployment directories.
- `diagnostics.md`, `diagnostics.json`, `diagnostics/`: the bisection and the placement tests.
- Scripts in `experiments/kernel_opt/`:
  - `flipquant_parity_gemm.py` (driver), `run_flipquant_parity.sh`, `flipquant_parity_binaries.py`,
    `flipquant_parity_report.py`;
  - the diagnostics: `flipquant_parity_diagnostics.py`, `_placement.py`, `_codeplace.py`, `_actplace.py`, `_alloc.py`.
- The flipquant worktree stayed clean in every run (checked before and after each run).
- Each run records both trees' commits (NVFP4-RaZeR c9212d2 to 09c182f, all on kernel-opt). Uncommitted at run time
  were only the results directory and scripts the workers do not import. rzK's flipquant worker may have started
  from 869b92a's driver, whose changes are on the driver side only.
- Phi-4 was not run ("only if cheap"): the time went to the diagnostics. It takes about 5 min per run with this driver.
