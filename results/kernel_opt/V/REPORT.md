# Kernel-opt V (amendment 18): the 256x64 path brought to the 16x64 state — report

2026-10-03, branch `kernel-opt`. The protocol is amendment 18 of `results/kernel_opt/PROTOCOL.md`, registered at
deb0ba8.
- **The run:** `run_V.sh` ran from 11:16 to 11:58 UTC, with no deviation.
- **Files:** tables in `V_tables.md`, data in `V.json`, the candidate table in `table_v/`. The disclosed exploration that
  chose the design is `EXPLORATION.md`.

**Adopted 2026-10-03.** The coordinator adopted amendment 18 per the registered rule, which was met including its strict
form.
- The tracked `sm120/configs/<gpu>.ko.json` is now `table_v/table_v.json`: the `'mixed256'` width rows and the
  `'mixed256_ko'` scheduler rows added, nothing else changed. The user copied it in, because a permission check
  refused this session's write.
- **The post-adoption check** (`adoption/`): routed as deployed from `build_V` on the tracked table, each unit's
  whole-model logits equal its previous deployment bitwise, on 4 models × 5 shapes. The units are 256x64, 16x64, 8x64,
  `stock_ko` and `stock_wB_ko`.
- `build_V` is the deployment directory of all three units and both stocks. `docs/BUILD_AND_USE.md` lists its builds.
- `'auto'` routes 256x64 artifacts to `mixed256_ko`. `paper_256` still selects A′'s `mixed256` on the paper table.

**What was measured** (the TM-OPT+TC 256x64 artifacts):
- **A:** today's 256x64 path, `mixed256` (A′: g32 builds, default dispatch, site-0 tags), on the paper table, from
  `build_A1`.
- **B:** `mixed256_ko` from `build_V`, at today's widths: the build effect. Its builds are A′'s g32 builds without the
  site-0 tags (t0), with #4's 64×64 epilogue at width 128 and the uniform-branch dispatch (`MIXFP4_UNIFORM_DISPATCH=1`).
- **C:** the candidate, `mixed256_ko` on the re-tuned table: its own widths and scheduler rows.
- **References:** `stock_ko`, the target; the ceiling `nodisp256_ko`, C's tiles with the dispatch compiled out.

## The gates

All passed.

| gate | result |
|---|---|
| G0 | 55 files, 100 builds |
| G3 self-tests | the four new builds: PASS patched / FAIL unpatched, each rebuild with `build_V`'s SASS |
| The patcher check | passed |
| G1 | 55 of 55 existing configurations keep their SASS; the new n16k64_wA_g32_e64_t0 has its census |
| G2 | 22 of 22 builds of `build_V` (census, nothing predicated) |
| G2u | the uniform g32 builds: no BSSY/BSYNC/WARPSYNC between the first and the last OMMA, a REDUX there; amendment 17's ten, A′'s four and stock_ko's four carry `build_U`'s / `build_A1`'s / `build_7`'s SASS |
| pytest on `build_V` | 749 passed, 694 skipped |
| G4 | set:mixed256_ko and its four builds against n16k64_wA: 0 differences |
| G5 | whole-model logits equal: set:mixed256 → set:mixed256_ko, and the routed form (`'auto'` → `mixed256_ko`) |
| G5b | equal on the candidate table |

## The candidate table (`table_v/`)

- **Widths:** 4b's method on the final builds, densest-module tags. 10 of 224 cells differ from today's (the paper
  table's 'mixed' rows):

  | cells | today → candidate |
  |---|---|
  | 4096x4096, 4096x14336 at T=128 | 32 → 64 |
  | 1024x5120 at T=512 | 32 → 64 |
  | 14336x4096 at T=256; 35840x5120 at T=128 | 128 → 64 |
  | 4096x14336 at T=1024 | 64 → 128 |
  | 10240x5120, 17408x5120 and 35840x5120 at T=64 | 64 → 32 |
  | 7680x5120 at T=32 | 32 → 16 |

  The tuned rows equal the adopted 16x64 rows in 220 cells and stock's in 219.
- **Scheduler rows:** amendment 7's decisive rule; 75 of 224 cells moved off (0, 1).
- **The decisive-margin rule on the widths** (a sensitivity, reported only):
  - with the 128-wide default, no cell would differ;
  - with the fallback-width default, 2 cells would differ, by +0.20 % median and +0.36 % at most against the fastest.

## M1: per-forward GEMM, 4 models × 12 T, typical and worst tags

### The rule is met

| per forward | median | below 0 in every round | above 0 in every round | T ≤ 16 | 32–128 | 256–1024 | ≥ 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **candidate vs today, typical** | **−1.29 %** | 48 of 48 | 0 | −0.43 % | −1.12 % | −2.13 % | −1.36 % |
| **candidate vs today, worst** | **−1.26 %** | 46 of 48 | 0 | −0.39 % | −1.12 % | −2.01 % | −1.31 % |
| builds at today's widths vs today, typical | −1.10 % | 48 of 48 | 0 | −0.37 % | −0.72 % | −1.61 % | −1.26 % |
| builds at today's widths vs today, worst | −0.99 % | 41 of 48 | 0 | −0.09 % | −0.49 % | −1.37 % | −1.19 % |
| table: candidate vs builds at today's widths, typical | −0.16 % | 28 of 48 | 10 | −0.06 % | −0.44 % | −0.07 % | −0.16 % |
| table: candidate vs builds at today's widths, worst | −0.22 % | 32 of 48 | 4 | −0.23 % | −0.57 % | −0.23 % | −0.13 % |

- **The rule** (amendment 11's form): negative medians with both tag sets, and no cell above +0.5 % in every round.
  - The candidate's worst cell is +0.02 %.
  - The strict form (no cell above zero in every round) is met as well.
- **The gain per cell is −0.2 … −7.1 % (typical).** It is largest at T = 256 (−4.4 / −4.2 %, median over the models), where the
  re-tuned widths add to the builds: Llama and Mistral at T=256, −7.1 / −6.9 %.
- **Per model** (typical, median over T): Llama −1.43 %, Mistral −1.38 %, Phi-4 −1.20 %, Qwen3.8-27B −1.18 %.
- **The builds carry most of the gain** (−1.10 %). The table adds −0.16 % overall: −1.7 % at T = 128, −2.7 % at
  T = 256 and −0.8 % at T = 1024; elsewhere it is within ±0.5 %.

### The residual gap to `stock_ko` per T (median over the 4 models)

| T | today, typical / worst | candidate, typical / worst | ceiling | candidate vs ceiling (typical) |
|---:|---|---|---:|---:|
| 1 | +0.58 / +0.45 % | +0.17 / +0.07 % | +0.21 % | −0.03 % |
| 4 | +0.34 / +0.27 % | −0.23 / −0.18 % | −0.11 % | −0.11 % |
| 16 | +0.31 / +0.28 % | +0.01 / +0.01 % | +0.16 % | −0.11 % |
| 32 | +1.17 / +1.11 % | +0.16 / +0.27 % | +0.28 % | −0.15 % |
| 64 | +1.53 / +1.58 % | +0.26 / +0.31 % | +0.13 % | +0.13 % |
| 128 | +3.44 / +3.41 % | +0.47 / +0.53 % | +0.11 % | +0.42 % |
| 256 | +5.69 / +5.71 % | +1.06 / +1.23 % | −0.24 % | +1.20 % |
| 512 | +4.36 / +4.35 % | +1.94 / +2.01 % | +0.11 % | +1.83 % |
| 1024 | +3.28 / +3.34 % | +1.22 / +1.23 % | −0.04 % | +1.26 % |
| 2048 | +2.97 / +2.89 % | +1.41 / +1.36 % | −0.01 % | +1.27 % |
| 4096 | +3.11 / +3.01 % | +1.81 / +1.89 % | +0.36 % | +1.47 % |
| 8192 | +3.10 / +3.10 % | +1.57 / +1.67 % | −0.01 % | +1.54 % |
| **all cells** | **+2.59 / +2.61 %** | **+0.68 / +0.78 %** | +0.11 % | +0.79 % |

- **The 256x64 path is now within 0.7 % of `stock_ko`** (typical), against +2.6 % today.
- **At T ≤ 64 it is at stock's latency** (−0.2 … +0.3 %).
- **What remains is the dispatch at T ≥ 256** (candidate vs ceiling +1.2 … +1.8 %). The ceiling itself is at stock
  (+0.1 %).

## C2V: 4096³, width 128

| | b2b | isolated | sustained |
|---|---:|---:|---:|
| C vs A, all-E2M1 | −2.07 % | −1.81 % | −1.43 % |
| C vs A, real map | −1.71 % | −1.88 % | −0.39 % |
| C vs A, all-E0M3 | −1.26 % | −0.52 % | +0.00 % |
| C vs the ceiling, all-E2M1 / real / all-E0M3 (b2b) | +0.90 / +1.26 / +1.24 % | | |
| the ceiling vs `stock_ko` | +0.33 % | −0.25 % | +0.15 % |

- **The new width-128 build is faster on every tag pattern,** all-E0M3 included. This differs from 16x64, where the
  uniform dispatch cost all-E0M3 0.4–0.8 %.
- **The sustained mode is at the 500 W cap.**

## Reading

- **The adoption of amendment 18 was proposed, and adopted on 2026-10-03** (above):
  - `mixed256_ko` with the candidate table's `'mixed256'` and `'mixed256_ko'` rows in the tracked `<gpu>.ko.json`;
  - `build_V` as the deployment directory of all three units;
  - `'auto'` routing 256x64 artifacts to `mixed256_ko` (registered with the sources).
- **The items, as asked:**
  1. **The uniform dispatch:** used at every width. The 4-arm g32 builds had reconvergence code; REDUX removes it.
  2. **#4's tile at width 128:** used, keeping 4 stages. The scheduler rows are tuned (75 cells off the default).
  3. **#2's dispatch:** not used. It does not help with 4 arms (exploration).
  4. **t0:** used together with REDUX (the exploration's −0.3 … −0.5 points at T ≥ 128). The rule is met by the
     combination; t0 was not tested on its own here.
  5. **The act-warm re-tune:** 10 width cells and 75 scheduler cells. The decisive-margin sensitivity changes 0 or 2
     cells.
  6. **M1 and C2V:** above.
  7. **The C3k-style E0M3 sweep** follows on adoption, as a separate descriptive amendment.
- **Maps and tags:** the paper's TM-OPT+TC 256x64 artifacts, as registered. The FlipQuant calibration will change later.
