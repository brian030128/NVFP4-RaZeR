# Plan: the paper's end-to-end latency, re-measured on build_V

**A plan only.** Nothing here has run. It needs the user's go, and then a registered protocol (below).

**Goal.** Re-measure the paper's end-to-end latency on the kernels that now ship (`build_V`, the tracked
`sm120/configs/<gpu>.ko.json`), so the paper's latency tables describe them:
- step 05: CUDA-graph prefill (eager is recorded in the same processes);
- Experiment D: CUDA-graph decode over a StaticCache.

The artifacts, shapes, settings, rounds and per-process scripts stay the paper's; only the kernel builds change.

## 1. What it replaces

- **`results/paper/tables/main.md`:**
  - "Prefill latency, CUDA graph (primary)": the ms tables and the "FlipQuant (ours) / reference − 1" tables of all
    four models, i.e. the 8x64 and 16x64 columns and their FourOverSix / NVFP4 / BF16 references;
  - "GEMM overheads: old method, new method, end to end": the end-to-end column. Its GEMM columns change only if the
    paired GEMM session of `SUMMARY.md` §8 also runs.
- **`results/paper/tables/appendix.md`:**
  - the 256x64 CUDA-graph prefill tables;
  - "Prefill latency, eager (supplementary)";
  - "GEMM vs end-to-end consistency (deviation 2)". That table pairs each GEMM number with its end-to-end number, so
    it is consistent only if the GEMM side is re-measured too (`SUMMARY.md` §8).
- **`results/paper_extra/D/`** (`D.md`, `D.csv`, `D.json`): the decode table of Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4.
- **The text** that quotes these numbers, for example:
  - 16x64 prefill "+1.4 … +3.5 %" against FourOverSix (Llama);
  - 8x64 "+4.2 … +23.5 %";
  - decode "16x64 −0.2 … −0.6 %, 8x64 −4.7 … −28 % against FourOverSix (wA)";
  - any figure drawn from these tables.
- The paper's committed records stay as they are. The re-measurement writes to new directories.

## 2. Harness: NVFP4-RaZeR's (recommended), not flipquant's `evaluation.latency`

Use the paper's per-process scripts unchanged:
- `experiments/paper/bench_prefill.py` (step 05);
- `experiments/paper_extra/bench_decode.py` (Experiment D).

Drive them with an orchestrator modelled on `experiments/kernel_opt/cum_e2e.py` and `cum8_e2e.py` (amendments 9 and
15): `--kernel` per policy and `SM120_BUILD_DIR=build_V`.

Why this harness:
1. **It produced the paper's numbers.** With the scripts, artifacts and protocol unchanged, any difference is the
   kernels.
2. **It has already run on kernel-opt builds twice,** with registered checks:
   - amendment 9: 16x64 on `build_7freq`, and `stock_ko`;
   - amendment 15: 8x64 on `build_P3freq`, `stock_ko` and `stock_wB_ko`.

   Every check passed in both:
   - graph logits = eager, bitwise;
   - decode tokens = eager StaticCache;
   - every loaded library a build of the policy's directory, by sha256;
   - the dispatch defines on exactly the expected builds;
   - the installed set and table are the policy's.
3. **Same artifacts and protocol:** `/home/dev/n16k64_campaign/paper/artifacts`, one process per (model, policy,
   round), 5 rounds in a shuffled or rotated order.
4. **flipquant's harness is a port with untested parity.** `evaluation.latency` ports these scripts (with
   `--kernel-set auto|paper`), but only its GEMM part has been checked against the paper harness (c655939: within
   0.7 % per forward at T ≥ 16; small-T cells off because of process state). Its prefill and decode have not been
   compared, and it has no multi-round driver. Using it would change the harness and the kernels at the same time.
   - Optional, as a cross-check rather than a replacement: flipquant's `evaluation.latency prefill` on Llama-3.1-8B at
     1x128 and 1x2048, on the same build_V. It would show whether the release harness reproduces the paper harness
     (about 5 min).

## 3. Scope

- **Models:**
  - prefill: the paper's four (Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B);
  - decode: Experiment D's three. The decode harness does not support Qwen3.8-27B's hybrid cache.
- **Shapes:**
  - prefill: 1x128, 1x256, 1x512, 1x1024, 1x2048, 1x4096, 1x8192 and 4x2048, 7 repetitions per process;
  - decode: batch {1, 4, 16} × prompt {512, 2048}, then 64 generated tokens.
- **Rounds:** 5. The policy order is shuffled per round as in the paper: seed 20260928 + r for step 05, 20260929 + r for
  D, recorded in `commands.log`.
- **Policies, core (recommended):**

  | policy | artifact | `--kernel` | build | the set the install must report |
  |---|---|---|---|---|
  | `bf16` | the model as loaded | — | — | — |
  | `fo6` | `fo6` | `auto_stock` | build_V | `stock_ko`, tracked ko table |
  | `ours-16x64` | `tc_16x64` | `auto` | build_V | `mixed_ko` (#2's dispatch, t0, REDUX at widths 64/128) |
  | `ours-8x64` | `tc_8x64` | `auto` | build_V | `mixed_wB_ko` (t0, #2's dispatch, pipelined flags) |
  | `ours-256x64` | `tc_256x64` | `auto` | build_V | `mixed256_ko` (g32, REDUX, t0, e64, `table_v` rows) |

  The paper ran `ours-8x64` on `n8k64_wB`. Since amendment 14, `auto` routes 8x64 maps to `mixed_wB_ko`, which gives
  the same outputs bit for bit.
- **Policies, optional** (each keeps one more column of the current tables):
  - `nvfp4` (`auto_stock` → `stock_ko`, with NVFP4 activations);
  - `ours-{8x64,16x64,256x64}-nvfp4act` (latency only, as in the paper);
  - the 8x64 same-placement references `fo6-wB` and `nvfp4-wB` on `stock_wB_ko` (`auto_stock_wB`).

    The deployed comparison for 8x64 is now FourOverSix on `stock_ko` (amendment 10). The wB columns only separate
    placement from format, which the paper's text explains.
  - 256x64 in decode. D did not include it.

## 4. Gates and checks

**Before the run (registered, all must pass):**
- **G0, provenance:**
  - the build_V manifests (22 libraries) and their sha256;
  - the tracked table's sha256 at its commit (4ae044a or later);
  - the artifacts' sha256, equal to the paper run's;
  - the sha256 of `bench_prefill.py`, `bench_decode.py` and the orchestrator.
- **G1, routing:**
  - `install(kernel='auto')` on each map artifact resolves to the expected set from build_V;
  - `auto_stock` resolves to `stock_ko`;
  - the routing tests pass (`test_select.py`, including `test_auto_routes_8x64_maps`).
- **G2, bitwise:**
  - Each build_V policy's whole-model logits equal its paper kernel's (`check_model_logits.py`), on 4 models × 5 shapes.
  - Amendments 17 and 18 passed this (their G5). It is re-run on the registered tree in about 10 min.
- **Smoke:** Llama-3.1-8B, 1 round, prefill 1x128 and 1x2048, decode 1x512 and 4x2048, every policy, into a smoke
  directory that is not used.
- **Environment:**
  - an idle GPU, with no other compute process (the scripts refuse a busy GPU);
  - the default caching allocator;
  - no clock locking; the 500 W power limit and a telemetry sampler are recorded.

**In the run (a failure stops the run):**
- **prefill:** every shape is captured, and the graph's logits equal eager's bit for bit;
- **decode:** the graph's first 33 greedy tokens equal an eager StaticCache decode's, in every setting;
- **per process:**
  - every loaded library is a build_V library (sha256 against the manifests);
  - the uniform-dispatch, pipelined-flag and #2 defines are on exactly the expected builds;
  - the installed set is the policy's family, reading the tracked ko table;
  - coverage: every quantized Linear ran natively.

**Registration:** a protocol amendment (20) in `results/kernel_opt/PROTOCOL.md`, or a new
`results/paper_v/PROTOCOL.md`, with `registration.json` (all sha256s), committed before the GPU run. The output goes
to new directories:
- `/home/dev/n16k64_campaign/paper_v/{latency,decode}`;
- the tables to `results/paper_v/` on a branch.

## 5. Time (one idle GPU, sequential)

The basis is the paper's own per-process times:
- step 05, 8 shapes × 7 repetitions: Llama 38 s, Mistral 36 s, Phi-4 64 s and Qwen3.8-27B 262 s per process (paper
  `commands.log`, 2026-09-29);
- Experiment D: 49 s per process (105 processes in 1 h 25 min).

| part | processes | time |
|---|---:|---:|
| prefill, core 5 policies × 5 rounds × 4 models | 100 | ≈ 2.8 h (Qwen3.8-27B alone 1.8 h) |
| decode, core 5 policies × 5 rounds × 3 models | 75 | ≈ 1.0 h |
| G0–G2, smoke | | ≈ 0.5 h |
| **core total** | | **≈ 4.3 h** |
| each optional prefill policy | 20 | +0.55 h |
| each optional decode policy | 15 | +0.2 h |
| the paper's full sets (11 prefill policies, 7 decode policies) | 325 | ≈ 8 h |
| optional: the paired GEMM session (`SUMMARY.md` §8), the same day and builds | | +1–1.5 h |

The tables (a `07_tables.py` equivalent for the new directory, plus D's report) are CPU work.

## 6. Expected changes (from the records; the run decides)

**Prefill, ours against FourOverSix, the median over the 32 (model, shape) cells:**

| unit | the paper (step 05) | latest end-to-end record on kernel-opt | expected on build_V |
|---|---:|---|---:|
| 16x64 | +1.64 % (models +1.0 … +1.8 %) | +0.93 %: amendment 9, `build_7freq` vs `stock_ko`, before REDUX | ≈ +0.6 … +0.8 % |
| 8x64 | +5.62 % vs FO6 (wA), up to +25 % at 1x128; +4.38 % vs FO6 (wB) | +3.80 %: amendment 15, `build_P3freq` vs `stock_ko`, before the pipelined flags; +0.5 % at 1x128, +7.4 % at 1x512 | ≈ +3.6 … +3.8 % |
| 256x64 | +1.71 % (models +1.0 … +1.9 %) | none: A′ and amendment 18 were never measured end to end | ≈ +0.5 … +0.8 % |

- **How the expected values were derived.** From these records, the end-to-end gap is about 0.45–0.65 of the GEMM
  gap:
  - the paper: 16x64 +1.64 % end to end against +3.5 % GEMM at T ≥ 128;
  - amendment 9: −0.89 % end to end against −1.96 % GEMM.

  Then:
  - 16x64: REDUX's −0.37 % GEMM (−0.6 % at T ≥ 128) takes about 0.2–0.35 points off +0.93 %.
  - 8x64: the flags' −0.17 % GEMM scales +3.80 % by the GEMM ratio at T ≥ 128 (+6.96 → +6.67 %): about +3.6 %.
  - 256x64: the GEMM gap at T ≥ 128 is +1.30 % (V vs `stock_ko`), against +3.65 % in the paper run.
- **The FourOverSix column itself moves.** `stock_ko` was −0.15 % against the paper stock in prefill and +0.03 % in
  decode (amendment 9). BF16 does not change; it is re-measured only so that the table comes from one session.
- **The remaining 8x64 excess sits at 1x512 … 1x1024,** the tile-bound cells (P7).

**Decode, tokens per second against FourOverSix (D's 3 models):**

| unit | the paper (D) | latest record | expected |
|---|---|---|---|
| 16x64 | −0.2 … −0.6 % | −0.21 %: amendment 9, median; REDUX does not act at the decode width 16 | ≈ −0.2 % |
| 8x64 | −4.7 … −28 % vs FO6 (wA), from placement; −0.6 … −3.1 % vs FO6 (wB) | −0.32 %: amendment 15, median (−1.5 … +0.1 %) vs `stock_ko`; the flags are neutral at T ≤ 16 | ≈ −0.3 % |
| 256x64 | not measured | — | ≈ −0.1 … −0.3 % (GEMM +0.2 … +0.8 % at T ≤ 16, C3v), if added |

So the coordinator's notes hold: "16x64 prefill +0.93 % vs FO6 before amendment 17; 8x64 +3.8 %; decode ≈ stock".
build_V should improve the two prefill gaps slightly and leave decode unchanged.

## 7. Decisions for the user

1. **Core or full policy set:** keep the NVFP4, NVFP4-activation and wB columns, or drop them?
2. **The 8x64 reference** in the paper's tables: `stock_ko` (deployed). Keep the same-placement columns?
3. **256x64 in decode?**
4. **Run the paired GEMM session (`SUMMARY.md` §8) first,** the same day and builds, so the GEMM tables and the
   consistency table change together with the end-to-end ones?
5. **Harness:** NVFP4-RaZeR's (recommended) or flipquant's `evaluation.latency`.
