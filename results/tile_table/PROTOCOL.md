# SM120 tile table for the RTX PRO 6000 — protocol

Written 2026-09-28 on branch `tm-opt`, before the tuning run; the hash and time are in `registration.json`.
Deviations are appended at the end. Task (user-approved, relayed by nvfp4-razer-c9).

## Background

- **What the table does:** the width-selecting kernel sets (`auto` = mixed n16k64_wA, `auto_stock` = stock_wA) pick
  a CTA-tile width in {16, 32, 64, 128} per (out, in, token-count bucket) from a per-GPU table,
  `sm120/configs/<gpu_slug>.json`.
- **This GPU has none:** so far only the RTX 5090's table existed. On this GPU
  (`nvidia_rtx_pro_6000_blackwell_workstation_edition`), every set used `select.fallback_width`: the narrowest width
  that holds the tokens.
- **Speed only:** all widths are bitwise identical (`tests/test_select.py`), so the table changes speed, never a
  result.
- **8x64 is out of scope:** the `n8k64_wB` build has no width variants, so the table does not apply to 8x64.

## Tuning

- **Tool:** `sm120/bench/tune_tiles.py` (the existing tool), families `mixed` and `stock`, every bucket in
  `select.BUCKETS` (1 to 8192). Each (shape, bucket, family, width) is timed as the CUPTI kernel time, median of 20
  calls, and the fastest width is taken. The raw timings go to `<gpu_slug>.raw.json`.
- **Shapes:** the text Linears of the four paper models, from their calibration records: Llama-3.1-8B,
  Mistral-7B-v0.3, Phi-4 (added to `MODEL_SHAPES`) and Qwen3.8-27B (added; text Linears only). That is 16 distinct
  (out, in) shapes. qwen4b is not tuned.
- **Mixed-family E0M3 tags:** per projection, the tags of the module with the most E0M3 tiles in the final TM-OPT+TC
  16x64 map (`--maps`, the Parts 2–3 `<model>_tc_16x64.mixfp4map`). The tags used per shape are recorded in the
  table's `meta.tags`.
- **Conditions:** idle GPU (the tool refuses otherwise); the repository's `sm120` builds. Their kernel sha256s are
  recorded, and compared with the copy the Part R latency harness used.

## Verification

1. `KernelSet('mixed')` and `KernelSet('stock')` load the new table on this GPU (`table_source` not None).
2. `tests/test_select.py` passes.
3. **Logits:** NativeLinear with the table and with the fallback give bitwise-identical logits.
   - Policies: Llama-3.1-8B's TM-OPT+TC 16x64 artifact (`auto`) and FourOverSix (`auto_stock`).
   - Inputs: a few WikiText-2 windows, at prefix lengths chosen so that some GEMMs run at a width where the table and
     the fallback differ. The per-width call counts show that they did.

## Report (`REPORT.md`, `tile_table.{md,json}` from `analyze.py`)

- **Every departure from the fallback:** each (shape, bucket) where the table's width differs from the fallback's,
  with the kernel-time gain from the raw timings, mixed and stock separately. Also where mixed and stock choose
  different widths.
- **Part R's operating points:** whether the table changes any width at prefill 1x2048 (T = 2048) and 4x2048
  (T = 8192), and at decode batch 1 (T = 1).
  - If it does, the affected Part R latencies are measured again with the table, with the same harness
    (`results/tm_opt/latency/bench_latency.py`) and the same rounds. The old and new numbers are reported.
- **No new batch-size sweep.** Instead, an estimate of the time for decode at batch 1/8/32/64 and prefill
  512/2048/4x2048, for all policies and models, from Part R's recorded run times.

Commit and push on `tm-opt`; then part (2), inference memory, under its own protocol.

## Deviations (append-only)

1. **2026-09-28 14:25 UTC: a same-process A/B was added, after the re-measurement.**
   - **Why:** the cross-session re-measurement of Qwen's 1x2048 prefill (5 rounds, same harness) cannot resolve the
     table's expected effect, about −0.1 %. The prefill is bimodal: repetitions land near 760 or near 880 ms in
     both sessions. The sessions also differ: 4x2048, where no width changed, moved −0.2 % on every policy.
   - **The added measurement** (`ab_same_process.py`), for FourOverSix (`stock`) and TM-OPT+TC 16x64 (`mixed`):
     - one process per policy: Qwen loaded once, the artifact installed once;
     - every NativeLinear's kernel set alternated between the table and the fallback, 5 times each in ABAB order;
     - each block the harness's prefill (2 warm-ups, 7 timed forwards).
   - **Reported alongside** the re-measurement; neither replaces the other.
