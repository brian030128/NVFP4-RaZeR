# Kernel-opt amendment 4: tile-table re-tune — first attempt, not adopted

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 4 (registration `registration_4.json`, 9d45342). The chain
`run_retune.sh` ran on 2026-09-30 from 21:49 to 22:20 UTC with no deviation. Tables: `abR_tables.md`; data: `abR.json`;
new table: `tables/`; diagnostic: `diag/`.

## What ran

- **Tuning:** `tune_tiles.py --families mixed,stock --cold --rounds 3 --iters 30`, on `build_freq`, with the paper
  artifacts' 16x64 tags. It took 8 minutes for 16 shapes × 14 buckets × 4 widths × 2 families.
  - 16 of 224 `mixed` cells changed and 31 of 224 `stock` cells (listed at the end of `abR_tables.md`, with each
    cell's M1 time).
- **Gates, all passed:**
  - GW1: `test_select.py` on `build_freq` (18 passed) and on `build_A1` (24 passed).
  - GW2: whole-model logits under the current vs the new table, bitwise equal on all 4 models × 5 shapes, for 16x64
    (set:mixed, `build_freq`), 256x64 (set:mixed256, `build_A1`) and FourOverSix (set:stock).
  - In M1‴, every new-table call equals its current-table call bitwise on the timed operands (1,260 / 1,260 / 720 /
    2,160 checks).

## M1‴: per-forward GEMM, new vs current table (deviation-2, 4 models × 12 T)

| unit | median over the 48 (model, T) | min | max | cells slower beyond the round range |
|---|---:|---:|---:|---:|
| stock_wA | −0.00 % | −5.77 % | +3.55 % | 16 |
| 16x64 typical | −0.05 % | −3.92 % | +1.53 % | 9 |
| 16x64 worst | −0.04 % | −3.46 % | +1.42 % | 11 |
| 256x64 typical | −0.01 % | −4.85 % | +1.53 % | 14 |
| 256x64 worst | −0.02 % | −4.79 % | +1.59 % | 14 |

**Where it helps and where it hurts:**
- **T = 256:** −3.5 to −5.8 % on every unit for Llama and Mistral. For example 14336x4096 runs at width 64 instead of
  128.
- **T = 1024:** −1.0 to −1.7 % (4096x14336 at width 128 instead of 64).
- **T = 64:** +1.1 to +3.6 % on every unit and model.
- **T = 32:** +0.3 to +0.9 % (Llama, Mistral).
- **Stock at T = 128:** +2.9 to +3.0 % (Llama, Mistral: 4096x4096 and 4096x14336 at width 32 instead of 64).
- **The gap vs stock_wA moves in the wrong way for the wrong reason.** At T = 128 the 16x64 gap shrinks from +3.2 %
  to +0.3 % only because stock gets slower. At T = 256 it widens from +1.7 % to +3.7 % because stock gains more than
  the mixed kernels.

**Adoption criterion** (no unit's per-forward sum slower beyond its round range at any (model, T)): not met. The new
rows are **not adopted**, and the tracked table is unchanged.

## Why the tuner and M1 disagree (diagnostic, disclosed; `diag/diag_tuner.py`, `diag_tuner.out`)

`tune_tiles.py --cold` flushes L2 (512 MiB read) and then launches the GEMM on activations quantized once before the
loop, so the activations are cold too. M1 (and inference) runs the activation quantizer after the flush: the weights
are cold but the fresh activations are L2-warm.

The diagnostic timed the same cells both ways (stock and #2's mixed builds, Llama shapes, T = 32 / 64 / 128, 3 × 30):
- **The M1 order reproduces M1's numbers.** Example: stock 4096x4096 at T = 64, width 32: 8.74 µs, M1's value exactly.
- **The M1 order also reproduces the current table's choices:** width 32 at T = 64, width 64 at T = 128.
- **The tuner's order adds 1.0–2.6 µs at every width** (e.g. 10.80 µs for the same cell) and re-ranks them. At T = 64
  it prefers width 16 for 4096x14336; at T = 128 it prefers 32 over 64.

So the first attempt measured a condition inference never has (cold activations). The re-tune needs the tuner to run
the activation quantizer after the flush, which is the deviation-2 method's actual condition. That is amendment 4b.
