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

---

# Amendment 4b: the re-tune with the activations warm — results

`run_retune_b.sh` ran on 2026-09-30 from 22:23 to 22:55 UTC as registered (`registration_4b.json`, 5f9d342), with no
deviation. Tables: `b/abR_tables.md`; data: `b/abR.json`; new table: `b/tables/` (method string "cold isolated
launches (--cold), 3 rotated rounds x 30 launches per width, the median of the per-round medians, activations
quantized after the flush (--act-warm)").

## What changed

8 of 224 `mixed` cells and 16 of 224 `stock` cells changed (`b/abR_tables.md` lists every affected projection with its
M1 time).
- Both families:
  - 14336x4096 at T = 256: 128 → 64;
  - 4096x14336 at T = 1024: 64 → 128;
  - 35840x5120 at T = 64 / 128: 64 → 32 / 128 → 64;
  - 10240x5120 at T = 32 / 64: 32 → 16 / 64 → 32.
- Stock also changes several Phi-4 and Qwen shapes at T = 32 (32 → 16), plus 12288x5120 at T = 512 and 17408x5120 at
  T = 128.
- The Llama/Mistral T = 64 and T = 128 cells that attempt 1 changed (and slowed) are unchanged now, as the diagnostic
  predicted.

## Gates: all passed

- **GW1:** `test_select.py` on `build_freq` and `build_A1`.
- **GW2:** logits bitwise equal under the current vs the 4b table, on 4 models × 5 shapes, for 16x64, 256x64 and
  FourOverSix.
- **In M1‴:** new = current bitwise on every timed operand.

## M1‴: per-forward GEMM, 4b table vs current table

| unit | median | min | max | T = 256 | T = 1024 |
|---|---:|---:|---:|---|---|
| stock_wA | −0.07 % | −5.61 % | +0.26 % | −5.6 % (Llama, Mistral) | −0.9 to −1.0 % |
| 16x64 typical | −0.04 % | −4.00 % | +0.19 % | −3.8 to −4.0 % | −1.5 % |
| 16x64 worst | −0.05 % | −3.45 % | +0.14 % | −3.4 % | −1.2 to −1.3 % |
| 256x64 typical | −0.04 % | −4.78 % | +0.15 % | −4.7 to −4.8 % | −1.4 to −1.7 % |
| 256x64 worst | −0.08 % | −4.86 % | +0.82 % | −4.8 to −4.9 % | −1.6 % |

The T = 256 and T = 1024 figures are for Llama and Mistral, whose shapes carry the changes. Phi-4 and Qwen move by
−0.2 to +0.8 %, as below.

**The adoption criterion as registered** (no unit slower beyond its round range at any (model, T)) is **not met in
the strict sense.** 20 (model, T, unit) cells are flagged. In 18 of them no width of that unit changed at that
(model, T): current and new are the identical computation, so the flag measures the harness's between-configuration
offset, not the table.
- Most flagged cells are +0.03 to +0.19 %.
- The largest is Phi-4 256x64 worst tags: +0.7 to +0.8 % at T = 1, 4, 16 (unchanged) and at T = 32, 64 (changed).
  The offset is the same whether or not a width changed.
- So every flag is noise or a configuration offset, and no width change measurably slows anything.

**Gap vs stock_wA with both families re-tuned the same way** (per-forward, typical; Llama / Mistral / Phi-4 / Qwen,
current → 4b):

| T | 16x64 | 256x64 |
|---|---|---|
| ≤ 64 | unchanged within ±0.3 pp | unchanged within ±0.5 pp |
| 128 | +3.1 → +3.1 / +3.0 → +3.2 / +1.6 → +1.3 / +1.2 → +1.7 | +3.5 → +3.6 / +3.4 → +3.6 / +1.0 → +0.8 / +0.8 → +1.3 |
| 256 | +2.0 → +3.8 / +2.5 → +4.5 / +3.3 → +3.0 / +3.2 → +3.1 | +1.0 → +1.9 / +1.1 → +2.0 / +2.5 → +2.4 / +2.3 → +2.4 |
| 1024 | +2.8 → +2.2 / +2.8 → +2.2 / +2.9 → +3.1 / +3.0 → +2.9 | +1.8 → +1.3 / +2.0 → +1.3 / +2.0 → +2.1 / +2.2 → +2.3 |
| ≥ 2048 | unchanged within ±0.2 pp | unchanged within ±0.3 pp |

**Reading:**
- A fair re-tune makes everyone faster at T = 256 and T = 1024 in absolute terms, but does not close the gap to stock.
  - At T = 256 stock gains more than the mixed kernels (−5.6 % vs −4.0 % on Llama), so the gap widens.
  - At T = 1024 it narrows by about 0.6 pp.
- The remaining mid-T gaps are kernel-driven, not width-driven. At T = 128 each family keeps its own best width: mixed
  runs Llama q/o/down at 32, stock at 64, because the mixed kernel's 64-wide build is relatively slower there.

**Proposed** (the user's decision): adopt the 4b rows into the tracked table for both families, since it is the
faster table for everyone. Report gap numbers against the 4b-tuned stock from then on.
