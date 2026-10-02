# Kernel-opt 8x64 plan, P3b (option C): the reduced 8x64 table, confirmed — report

2026-10-02, branch `kernel-opt`. The protocol is amendment 12b of `results/kernel_opt/PROTOCOL.md`, registered at
3ef1699. `run_w8p3b.sh` ran 17:27–17:43 UTC with no deviation. Every number is in `w8p3b_tables.md` and `w8p3b.json`;
the selection is in `cells.json`, the tables in `table_p3b/` and `table_a/`.

## In brief

- **Every gate passed.**
  - G0: 39 files and 17 builds.
  - G4: 28,800 comparisons of set:mixed_wB_ko on the reduced table, 0 differences.
  - G5: logits equal on 4 models × 5 shapes.
  - M1's bitwise checks: all equal.
- **The registered rule is met.**
  - After vs before, per forward: median −0.06 % (typical tags) and −0.09 % (worst tags) over the 48 cells.
  - No cell is above zero in every round by more than 0.5 %. The largest every-round minimum is +0.20 % (Mistral-7B,
    T = 32, typical).
- **So the reduced table is adopted.** It is 1b's rows with the 11 selected P3 widths, and no scheduler rows for the
  8x64 path. It now becomes the tracked `sm120/configs/<gpu>.ko.json`, together with stock_wB_ko's rows.
- **Every one of the 23 (model, projection, T) GEMMs whose width changed is faster in this fresh run**, by −0.4 to −12.6 %:
  - Llama / Mistral k/v_proj at T = 128: −10.8 … −11.4 %;
  - Qwen3.8-27B in_proj_a/b at T = 8192: −11.9, −12.6 %;
  - the o_proj '128x64' choices at T = 128: −2.1 … −3.9 %.
- **Per forward the gain is concentrated at T = 128** (−1.1 %); T = 64 is −0.24 %, and every other T is within ±0.1 %.
- **stock_wB tuned alike** (stock_wB_e64 with its rows, adopted after amendment 12) is −0.62 % against stock_wB here.

## The residual gap of the adopted 8x64 path to stock_ko

T and #2's dispatch (amendment 11), on the reduced table; typical tags; per-forward GEMM, median over the 4 models.

| T | before (P2's path) | after (adopted) | after, worst tags | after vs stock_wB tuned alike |
|---:|---:|---:|---:|---:|
| 1 | +0.48 % | +0.30 % | +1.28 % | −38.8 % |
| 4 | +0.41 % | +0.38 % | +1.21 % | −38.9 % |
| 16 | +1.09 % | +1.03 % | +1.79 % | −36.1 % |
| 32 | +1.99 % | +2.00 % | +2.56 % | −33.2 % |
| 64 | +1.42 % | +1.18 % | +1.73 % | −28.0 % |
| 128 | +2.56 % | +1.56 % | +2.30 % | −24.9 % |
| 256 | +5.74 % | +5.70 % | +7.23 % | −15.5 % |
| 512 | +9.08 % | +8.98 % | +10.69 % | +5.1 % |
| 1024 | +6.10 % | +6.04 % | +7.29 % | +3.8 % |
| 2048 | +7.11 % | +7.04 % | +8.07 % | +6.8 % |
| 4096 | +7.02 % | +6.99 % | +8.07 % | +7.2 % |
| 8192 | +7.42 % | +7.41 % | +8.68 % | +7.5 % |

These are per-T medians of each ratio separately.

**From the paper path to now.** Amendment 10 measured 1b's path, before P2, at:
- +1.3 % (T ≤ 16);
- +2.7 % (32–128);
- +8.9 % (256–1024);
- +10.2 % (≥ 2048).

The adopted path is now:
- +0.3 … +1.0 % (T ≤ 16);
- +1.2 … +2.0 % (32–128);
- +5.7 / +9.0 / +6.0 % (256 / 512 / 1024);
- +7.0 … +7.4 % (≥ 2048).

## Maps and tags used

- The paper's TM-OPT+TC 8x64 artifacts, with sha256 in `registration_12b.json`, plus the FourOverSix artifacts for the
  stock references.
- M1: typical is the lower-median module per projection, worst the densest.
- The selection used amendment 12's M1 on the same artifacts.
- The FlipQuant calibration will change later. The table's widths were selected on these maps' tags; the bitwise results
  do not depend on the map.
