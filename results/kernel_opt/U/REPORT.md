# Kernel-opt U (amendment 17): the uniform-branch dispatch on the wide tiles (part A), the re-tuned 16x64 width cells (part B) — report

2026-10-03, branch `kernel-opt`. The protocol is amendment 17 of `results/kernel_opt/PROTOCOL.md`, registered at
aff7829.
- **The run:** `run_U.sh` ran from 08:11 to 08:49 UTC, with no deviation. It was started as `bash run_U.sh`: the file
  is not executable (mode 100644), and a first start as an executable failed before any step.
- **Files:** tables in `U_tables.md`, data in `U.json`. The disclosed exploration that chose the design is
  `EXPLORATION.md`.

**Adopted 2026-10-03.** The coordinator adopted part A for both families, per the registered rule. `build_U` (the
families `mixed_ko` and `mixed_wB_ko`, plus `stock_wB_e64` for `auto_stock_wB`) is the deployment directory of both
adopted paths; `docs/BUILD_AND_USE.md` lists its builds. Part B is not adopted, since its rule failed, and the table is
unchanged.

**What was measured:**
- **A:** today's adopted paths: 16x64 `mixed_ko` from `build_7freq`, and 8x64 `mixed_wB_ko` from `build_P3freq`.
- **U (part A):** the same families from `build_U`:
  - 16x64 with the uniform dispatch (`MIXFP4_UNIFORM_DISPATCH=1`) at widths 64 and 128;
  - 8x64 with the pipelined flag read (`MIXFP4_PIPE_FLAGS=1`) at widths 64, '128x64' and 128;
  - widths 16 and 32 are today's builds, with identical SASS.
- **UB (part B):** U16 on part B's table.

## The gates

All passed.

| gate | result |
|---|---|
| G0 | 59 files, 88 builds |
| G3 self-tests | 5 × PASS patched / FAIL unpatched; each rebuilt library carries `build_U`'s SASS |
| The patcher check | passed |
| G1 | 55 of 55 configurations keep their patched and unpatched SASS |
| G2 | 10 of 10 builds (census, nothing predicated) |
| G2u | the uniform builds: no BSSY, BSYNC or WARPSYNC between the first and the last OMMA, with a REDUX there; the unchanged builds: today's SASS; the defines as registered |
| pytest on `build_U` | 403 passed, 633 skipped (builds it does not hold) |
| G4 | 33,600 (16x64) + 28,800 (8x64) comparisons, 0 differences |
| G5 | whole-model logits equal for both families |
| G5b | equal on part B's table |

## Part B's table (`table_b/`)

The six cells were re-tuned by 4b's method on part A's builds, with the densest-module tags. Medians in µs:

| cell | today's width | stock's width | width 16 | width 32 | width 64 | width 128 | the tuned width |
|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 @ 128 | 32 | 64 | 17.54 | 10.83 | **10.64** | 15.90 | **64** (changed) |
| 4096x14336 @ 128 | 32 | 64 | 55.98 | **29.57** | 29.95 | 47.02 | 32 |
| 17408x5120 @ 128 | 128 | 64 | 76.14 | 41.41 | 36.00 | **35.39** | 128 |
| 1024x4096 @ 512 | 32 | 64 | 16.94 | **9.92** | 10.53 | 15.94 | 32 |
| 1024x5120 @ 512 | 32 | 64 | 20.29 | **11.87** | 12.02 | 18.64 | 32 |
| 12288x5120 @ 512 | 128 | 64 | 178.13 | 111.12 | 79.74 | **77.38** | 128 |

- **Only 4096x4096 @ 128 changed**, from width 32 to 64, with the scheduler setting (0, 1).
- Its best setting, (1, 1), was −0.7 %, but its rounds overlapped the default's, so it is not decisive.

## M1: per-forward GEMM, 4 models × 12 T, typical and worst tags

### Part A: met for both families

| per forward, vs today | median | below 0 in every round | above 0 in every round | T ≤ 16 | 32–128 | 256–1024 | ≥ 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16x64, typical | **−0.37 %** | 30 of 48 | 4 | −0.04 % | −0.06 % | −0.76 % | −0.51 % |
| 16x64, worst | **−0.49 %** | 36 of 48 | 2 | −0.07 % | −0.08 % | −0.93 % | −0.67 % |
| 8x64, typical | **−0.17 %** | 30 of 48 | 1 | −0.05 % | −0.17 % | −0.46 % | −0.15 % |
| 8x64, worst | **−0.16 %** | 28 of 48 | 6 | +0.16 % | −0.07 % | −0.49 % | −0.17 % |

- **Both rules are met:** negative medians with both tag sets, and no cell above +0.5 % in every round.
- **Range:** 16x64 −1.9 … +0.4 % per cell, largest at T = 256 (−1.6 / −1.4 %); 8x64 −1.1 … +0.4 %.
- **The strict form is not met.** The cells above zero in every round are:
  - 16x64: Phi-4 T ≤ 32 typical (+0.20 … +0.41 %); Llama T=128 worst (+0.10 %); Phi-4 T=32 worst (+0.14 %);
  - 8x64: T ≤ 16 cells of Llama and Mistral worst (+0.22 … +0.38 %); Llama T=4096 worst (+0.10 %); Qwen T=16
    typical (+0.08 %).
- **Most of these are an A/A effect.** At T ≤ 32 those models run widths 16 and 32, whose builds in A and U have
  identical SASS (gate G2u), yet up to +0.4 % is "above zero in every round". This is the noise floor of
  identical computations, as amendment 12's post-hoc A/A found.

### Part B: NOT met

- **Only two (model, T) cells are affected:** 4096x4096 @ 128 is q_proj and o_proj of Llama-3.1-8B and Mistral-7B at
  T=128.

  | cell | typical | worst |
  |---|---|---|
  | Llama T=128 | −0.04 % [−0.20, +0.23] | −0.03 % [−0.06, +0.03] |
  | Mistral T=128 | −0.21 % [−0.28, −0.20] | **+0.44 % [+0.33, +0.50]** |
  | median | −0.13 % | **+0.21 %** |

- **The rule fails** on the worst tags' positive median.
- **The unaffected cells** (identical computations) give −0.02 % (typical) and −0.06 % (worst): the A/A check.
- **Conclusion:** at T=128, width 64 is not robustly better than 32 on E0M3-heavy modules even with the uniform
  dispatch. Part B is not proposed; the adopted table stays as it is.

### The residual gap to `stock_ko` per T (median over the 4 models)

| T | 16x64 today, typical / worst | 16x64 part A, typical / worst | 8x64 today, typical / worst | 8x64 part A, typical / worst | 16x64 ceiling | 8x64 ceiling |
|---:|---|---|---|---|---:|---:|
| 1 | +0.77 / +1.08 % | +0.75 / +0.96 % | +0.86 / +1.35 % | +0.78 / +1.43 % | +0.12 % | −0.28 % |
| 4 | +0.56 / +0.91 % | +0.52 / +0.88 % | +0.81 / +1.18 % | +0.67 / +1.31 % | −0.04 % | −0.39 % |
| 16 | +0.73 / +1.11 % | +0.86 / +1.01 % | +1.03 / +1.54 % | +0.92 / +1.67 % | +0.20 % | −0.11 % |
| 32 | +0.95 / +1.31 % | +0.96 / +1.27 % | +2.12 / +2.89 % | +2.09 / +2.86 % | +0.25 % | +0.70 % |
| 64 | +1.10 / +1.70 % | +0.92 / +1.50 % | +2.37 / +2.75 % | +2.13 / +2.71 % | +0.06 % | +0.85 % |
| 128 | +2.08 / +2.52 % | +1.78 / +2.22 % | +1.66 / +2.37 % | +1.47 / +2.09 % | +0.87 % | −0.05 % |
| 256 | +3.50 / +4.67 % | **+2.11 / +3.17 %** | +5.56 / +6.98 % | +4.90 / +5.86 % | +0.30 % | +2.23 % |
| 512 | +3.47 / +4.67 % | +2.68 / +3.45 % | +9.54 / +10.91 % | +8.90 / +10.30 % | +0.41 % | +5.70 % |
| 1024 | +1.33 / +2.57 % | +0.87 / +1.87 % | +6.13 / +7.41 % | +5.90 / +7.12 % | +0.00 % | +3.20 % |
| 2048 | +1.47 / +2.57 % | +1.04 / +1.88 % | +6.96 / +8.01 % | +6.67 / +7.79 % | −0.09 % | +4.20 % |
| 4096 | +1.95 / +2.85 % | +1.37 / +2.19 % | +7.21 / +8.26 % | +7.35 / +8.01 % | +0.41 % | +4.22 % |
| 8192 | +1.80 / +2.72 % | +1.12 / +1.99 % | +7.31 / +8.69 % | +7.33 / +8.54 % | −0.10 % | +4.25 % |
| **all cells** | **+1.41 / +2.07 %** | **+1.02 / +1.74 %** | **+4.20 / +5.29 %** | **+3.83 / +5.10 %** | +0.12 % | +1.44 % |

- **16x64:** the gap falls by about a third at T ≥ 256.
- **What remains of the dispatch** (part A against its ceiling, typical) is +0.92 % for 16x64 (+1.3 % at T ≥ 2048) and
  +1.90 % for 8x64.
- **8x64 at T ≥ 512** is still dominated by the 1x8 tiles: the ceiling is +3.2 … +5.7 % against `stock_ko`.

## C2U: 4096³, width 128 (`U_tables.md`)

| | b2b | isolated | sustained |
|---|---:|---:|---:|
| 16x64 U vs A, all-E2M1 | −0.49 % | −0.33 % | −1.33 % |
| 16x64 U vs A, real map | −0.66 % | −1.09 % | −1.30 % |
| 16x64 U vs A, all-E0M3 | +0.61 % | +0.40 % | +0.76 % |
| 8x64 U vs A, all-E2M1 | +0.25 % | −0.21 % | +1.22 % |
| 8x64 U vs A, real map | +0.04 % | −0.42 % | +0.04 % |
| 8x64 U vs A, all-E0M3 | −0.41 % | −0.54 % | −0.10 % |

- **The 16x64 uniform dispatch** gains on all-E2M1 and on the real map. It loses 0.4 … 0.8 % on all-E0M3, as in the
  exploration.
- **The 8x64 pipelined flags** are neutral at 4096³.
- **The sustained mode is at the 500 W power cap** in every configuration (SM clock 1,620–2,090 MHz).

## Reading

- **Item 1.** Part A meets its registered rule in both families, so its adoption is proposed: `build_U` as the
  deployment directory of both adopted paths.
  - **The uniform branch does what was asked:** no BSSY, BSYNC or WARPSYNC around the per-k_tile dispatch, the poison
    in place, nothing predicated.
  - **It pays only on the wide 16x64 tiles.** Elsewhere the narrow tiles keep today's dispatch.
  - **The 8x64 gain is the pipelined flag read,** not the uniform index.
- **Item 2.** The cause is established: the width-64 mixed build's dispatch reconvergence code. The uniform index
  removes most of it.
  - **But the re-tune does not move the table robustly.** With the densest tags, width 64 wins in 1 of the 6
    disagreeing cells, and that cell fails part B's rule on the worst tags.
  - **The 16x64 T=128 residual** (+1.8 % typical after part A) is the width choice there plus the remaining dispatch.
- **Maps and tags:** the paper's TM-OPT+TC 16x64 and 8x64 artifacts, as registered. The FlipQuant calibration will
  change later.
