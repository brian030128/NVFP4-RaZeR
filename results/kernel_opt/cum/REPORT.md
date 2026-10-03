# Kernel-opt amendment 9: the cumulative run — the combined 16x64 build against today's paper builds

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 9 (`registration_9.json`, baec35e). The chain `run_cum.sh` ran
on 2026-10-01 from 05:55 to 09:13 UTC, with no deviation. The scope is the user's decision 7 as narrowed by the
schedule change: the 16x64 family, plus FourOverSix and the paper-build references.

**What was compared:**
- **The combined build:** `mixed_ko` built with #2's dispatch (`build_7freq`). That is #2's pattern-0-first dispatch,
  t0 (no site-0 tags), #4's 64 × 64 epilogue tile at width 128, and the adopted table `<gpu>.ko.json` with the 4b
  widths and scheduler rows. B′ did not supersede #2's dispatch (`results/kernel_opt/Bprime/EXPLORATION.md`).
- **The adopted stock set:** `stock_ko` (#4 + the 4b widths + scheduler rows).
- **The references:** today's paper builds, `mixed` and `stock` in `sm120/build`, on the paper table.

**Files:**
- GEMM (M1): `cum_gemm_tables.md` and `cum_gemm.json`.
- End to end: `cum_e2e_tables.md` and `cum_e2e.json`.
- Gates: `g0_provenance.json`, `g2_sass_freq.json`, `g2_sass_ko.json`, `g4_bitwise_freq.json`, `g5_logits_16x64.json`
  and `g5_logits_fo6.json`.
- The raw records, logs and pytest output are in `/home/dev/n16k64_campaign/kernel_opt/cum`: `gemm/`,
  `e2e/{prefill,decode}/`, `logs/`, `commands.log`.

## In short

- **The outputs are the same, bit for bit.**
  - Every gate passed. G4 made 33,600 GEMM comparisons; G5 compared whole-model logits on 4 models.
  - M1 checked 4,680 timed operands.
  - All 80 prefill and 60 decode processes passed their registered checks: graph equals eager, the right libraries
    loaded, the right set installed.
- **GEMM: the combined build is −1.96 % against the paper build.** That is the median per-forward GEMM time over the 48
  (model, T) cells, typical tags; the range is −0.57 to −5.19 %. It is faster in every round in all 48 cells. With
  worst tags it is −1.54 %.
- **The gap to stock halves at the GEMM level.** With typical tags, the 16x64 path's median gap to its own stock falls
  from +2.90 % (paper) to +1.45 % (combined). Against the paper stock, the combined build is at +0.57 %.
- **Prefill: −0.89 %** against the paper build (median over the 32 (model, shape) cells; −0.23 to −4.57 %).
  - It is faster in every round in 27 of 32 cells and slower in every round in none.
  - The median gap to FourOverSix goes from +1.73 % (paper vs paper) to +0.93 % (combined vs `stock_ko`). Against the
    paper's FourOverSix the combined build is at +0.71 %.
- **Decode: +0.10 % tokens per second** (+0.02 to +0.23 %; better in every round in 14 of 18 cells). At decode the
  gap to FourOverSix was already small: −0.27 % goes to −0.21 % against `stock_ko`.
- **Stock gains too, but less:** −0.64 % at the GEMM level, −0.15 % in prefill and +0.03 % in decode.
  - At the GEMM level it gains about −0.7 to −0.8 % from T = 32 up, where #4 and the scheduler rows act.
  - Its largest gains are where the 4b widths changed: −6.3 % at T = 256 on Llama and Mistral.

## Gates

| gate | result |
|---|---|
| G0 provenance | 41/41 registered files, 20/20 builds (library, SASS, defines), 12/12 with amendment 7's gated SASS |
| G2 census | `build_7freq`'s 4 and `build_7`'s 8 builds: expected census, nothing predicated, stock E2M1 only |
| G3 pytest | 174 passed on `build_7freq` (628 skipped: configurations not built there); 17 passed on `build_7` |
| G4 bitwise | 33,600 comparisons against n16k64_wA (`sm120/build`), 0 differences |
| G5 logits | bitwise equal on 4 models × 5 shapes: paper `mixed` vs combined, paper `stock` vs `stock_ko` |

## M1: GEMM time (deviation-2)

The method: cold weights, activations quantized after the flush, CUPTI, 3 rotated rounds × 30. The value per model and
T is the sum over projections of modules × median time. Below are the medians over the 4 models and the T values in
each band, in %. The per-model tables, with round ranges, are in `cum_gemm_tables.md`.

| T | combined vs paper, typical | combined vs paper, worst | stock_ko vs paper stock | gap to own stock: paper → combined | combined vs paper stock | #2's vs default dispatch |
|---|---:|---:|---:|---|---:|---:|
| ≤ 16 | −0.68 | −0.25 | −0.06 | +1.23 → +0.50 | +0.43 | −0.39 |
| 32–128 | −1.48 | −0.85 | −0.69 | +1.74 → +1.01 | +0.31 | −0.49 |
| 256–1024 | −2.60 | −2.14 | −0.81 | +3.83 → +2.16 | +1.39 | −0.79 |
| ≥ 2048 | −2.33 | −1.91 | −0.75 | +3.45 → +1.87 | +1.11 | −0.59 |
| all 48 cells | −1.96 | −1.54 | −0.64 | +2.90 → +1.45 | +0.57 | −0.56 |

- **Where the gap to own stock grows.**
  - Llama-3.1-8B and Mistral-7B-v0.3 at T = 256: +2.9 → +4.1 % and +2.9 → +4.3 %. There the 4b width of gate/up_proj
    makes `stock_ko` 6.3 % faster, more than it helps the mixed kernel. The combined build is still −5.2 % (Llama)
    against the paper build in that cell, and 2.5 % below the paper stock.
  - Qwen3.8-27B at T = 32: +0.72 → +0.74 %.
- **Worst tags:** one cell is slower in every round: Mistral T = 16, +0.20 % [+0.08, +0.26].
- **#2's dispatch vs the default, on the adopted path:**
  - typical tags: −0.56 % median, faster in every round in 47 of 48 cells;
  - worst tags: −0.22 % median; 12 of 48 cells are slower in every round (up to +0.54 %), all on Llama and Mistral at
    T ≤ 512.
  - The worst modules are the densest in E0M3. C3k found the default dispatch faster from about f = 25 %, and #2's
    faster at f ≤ 5–10 %.
- **Widths.** The adopted table differs from the paper table in only a few (projection, T) cells per model:
  - 16x64, 3–4 cells: Llama and Mistral at gate/up_proj T = 256 and down_proj T = 1024; Phi-4 and Qwen at T = 32–128;
  - stock, 3–13 cells.
- **Telemetry:**
  - busy SM clock median 2,655–2,662 MHz per model, minimum 2,265 MHz (Phi-4);
  - power median 137–144 W, maximum 472 W (500 W limit);
  - the software power cap was active in 1–5 % of the samples.

## End to end

The paper's per-process scripts ran unchanged, as one process per (model, policy, round), in 5 rounds rotated by r − 1.
- Prefill: CUDA graph, the paper's 8 shapes, 7 repetitions.
- Decode: CUDA graph over a StaticCache, D's 6 settings, 64 tokens. It covers Llama, Mistral and Phi-4; the harness
  does not support Qwen3.8-27B's hybrid cache.

Values are the median over rounds of each process's value, and changes are paired within rounds. The per-shape tables
with round ranges are in `cum_e2e_tables.md`.

**Prefill:** per model, the median over the 8 shapes (range), in %. A change in time: negative = faster.

| model | combined vs paper | fo6-ko vs fo6-paper | paper vs fo6-paper | combined vs fo6-ko | combined vs fo6-paper |
|---|---|---|---|---|---|
| Llama-3.1-8B | −0.85 (−3.31 … −0.24) | −0.39 (−1.93 … +0.02) | +1.85 | +1.04 | +0.86 |
| Mistral-7B-v0.3 | −0.98 (−4.57 … −0.74) | −0.34 (−3.40 … −0.03) | +1.83 | +1.11 | +0.70 |
| Phi-4 | −1.05 (−1.81 … −0.23) | −0.16 (−0.70 … −0.10) | +1.89 | +1.09 | +0.84 |
| Qwen3.8-27B | −0.45 (−0.65 … −0.29) | −0.07 (−0.10 … −0.00) | +0.98 | +0.55 | +0.54 |
| all 32 cells | −0.89 (−4.57 … −0.23) | −0.15 (−3.40 … +0.02) | +1.73 | +0.93 | +0.71 |

- **The 5 cells not faster in every round** are Llama 1x512, 1x4096, 1x8192 and 4x2048, and Mistral 1x512. Their
  medians are −0.24 to −1.44 %.
  - In four of them one round of five is above zero:
    - round 1 (the run's first) for Llama 1x4096, 1x8192 and 4x2048, by +0.14 to +0.52 %;
    - round 5 for Llama 1x512, by +0.81 %.
  - Mistral 1x512 has two rounds above zero, at +1.03 and +1.16 %.
  - Llama's and Mistral's 1x512 cells are the noisiest of the run, for every ratio: round ranges of about 2–4 points.
    Phi-4's 1x256 cell is similar.
- **The largest gains are at 1x1024 on Llama and Mistral:** −3.3 % and −4.6 %. That is where the 4b width of down_proj
  changes. FourOverSix gains there too, by −1.9 % and −3.4 %.
- **Qwen3.8-27B gains the least** (−0.45 %), although its M1 GEMM gain (−1.86 %) is like the other models'.
  - Its paper-vs-FourOverSix gap shrinks in the same way: +0.98 % end to end, against +2.56 % in M1.
  - So a smaller share of its prefill time is spent in the quantized GEMMs. This is inferred from the two ratios, not
    measured.

**Decode:** per model, the median over the 6 settings, in %. A change in tokens per second: positive = faster.

| model | combined vs paper | fo6-ko vs fo6-paper | paper vs fo6-paper | combined vs fo6-ko | combined vs fo6-paper |
|---|---|---|---|---|---|
| Llama-3.1-8B | +0.13 (+0.02 … +0.22) | +0.05 | −0.31 | −0.25 | −0.20 |
| Mistral-7B-v0.3 | +0.14 (+0.05 … +0.23) | +0.05 | −0.33 | −0.29 | −0.21 |
| Phi-4 | +0.04 (+0.03 … +0.12) | +0.01 | −0.17 | −0.14 | −0.12 |
| all 18 cells | +0.10 (+0.02 … +0.23) | +0.03 | −0.27 | −0.21 | −0.19 |

- **The 4 cells not better in every round** are Llama and Mistral at 16x2048, and Phi-4 at 4x2048 and 16x2048. Their
  medians are +0.02 to +0.05 %.
- At decode every quantized Linear runs at width 16 in both sets (the records' widths). #4's epilogue tile (width 128
  only) and the 4b widths therefore play no part. The gain is t0, #2's dispatch, and the scheduler rows the adopted
  table sets at T ≤ 16 (14 `mixed_ko` rows).
- There was little to recover: the paper build's decode gap to FourOverSix was already −0.27 % (Experiment D's size).

## Reading

- **The kernel-opt changes carry through to the end-to-end level, diluted by the time outside the quantized GEMMs.**
  - GEMM: −2 % against the paper build.
  - Prefill: about −0.9 % (−0.45 % on Qwen).
  - Decode: about +0.1 % tokens per second.
- **About half of the 16x64 gap to stock remains**: +1.45 % at the GEMM level, and +0.93 % in prefill against
  `stock_ko`.
  - C3k attributes the remainder to the per-k_tile dispatch itself (the ceiling `nodisp_ko` matches `stock_ko`) and,
    on real maps, the E0M3 tiles.
  - B′ did not reduce it.
- **Nothing here is adopted or tuned.** The paper numbers (tm-opt) are untouched. Re-measuring them with the combined
  build is the user's decision.
- **If the paper's latency were re-measured**, the change would be in step 05's 16x64 rows (and the FourOverSix
  reference if `stock_ko` is used). Prefill would be about −0.9 % (−0.5 % on Qwen), and the gap to FourOverSix would
  shrink from about +1.7 % to about +0.9 %.

## Later: amendment 17 (2026-10-03)

The 16x64 path's widths 64 and 128 now take the uniform-branch dispatch (`MIXFP4_UNIFORM_DISPATCH=1`): no BSSY, BSYNC or
WARPSYNC around the per-k_tile dispatch.
- **The effect:** −0.37 % (typical) and −0.49 % (worst) per forward, up to −1.6 % at T = 256. The gap to `stock_ko`
  goes from +1.41 % to +1.02 % (typical, median).
- **The deployment directory** is `build_U` (adopted by the coordinator per the registered rule), superseding
  `build_7freq`.
- **Details:** `results/kernel_opt/U/REPORT.md`.

