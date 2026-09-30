# Deviation 2: per-forward GEMM latency with isolated launches and cold weights — protocol

Written 2026-09-30 on branch `tm-opt`, before any GPU run of this measurement; the hashes and time are in
`registration_gemm_isolated.json`. It is deviation 2 of `PROTOCOL.md`. Deviations of this protocol are appended at the
end. Requested by the user and approved as proposed (relayed by nvfp4-razer-c9, 2026-09-30).

## Why

Step 06 of the paper flow, re-run under deviation 1 (C1), gives these overheads at T ≥ 2048:
- 16x64: +3.6 to +4.8 %;
- 8x64: +10.4 to +12.9 %.

Both are above earlier measurements on this GPU. Item #3's diagnostic (`results/tm_opt/REPORT_ITEMS.md` §#3) found
isolated calls the most stable timing method, and the one that agreed with full-model prefill:
- 16x64 all-E2M1: +1.7 to +4.2 %;
- 8x64: +6.6 to +9.4 %.

Step 06 differs from that in three ways, and each can inflate the gap:
1. **CUPTI over 20 back-to-back calls.** Time spent at a dependent-launch barrier, and a sustained load against the
   500 W power cap.
2. **L2-warm weights.** One weight matrix is reused 20 times.
3. **The densest E0M3 module per projection,** the worst case.

**What is kept.** Step 06's records (`gemm/`) are kept and rendered as the alternative method: "CUPTI, back-to-back
calls, L2-warm, densest module". The fixed-order records that deviation 1 superseded stay in `gemm_superseded/`.

## Method (`experiments/paper/bench_gemm_isolated.py`)

**Scope.**
- **Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B. Every text-Linear shape (projection), at
  T ∈ {128, 256, 512, 1024, 2048, 4096, 8192}.
- **Artifacts:** those of the paper run (`/home/dev/n16k64_campaign/paper/artifacts`).
- **Configurations:**

  | configuration | kernel | weights | activation quantizer |
  |---|---|---|---|
  | stock_wA | `KernelSet('stock')`, the tile table's width | FourOverSix | FourOverSix rows |
  | stock_wA_nvfp4 | the same | NVFP4 | NVFP4 rows |
  | stock_wB | stock_wB | FourOverSix | FourOverSix rows |
  | mixed_16x64_{typical, worst} | `KernelSet('mixed')` (n16k64_wA), the tile table's width | TM-OPT+TC 16x64 | FourOverSix rows |
  | mixed_256x64_{typical, worst} | the same | TM-OPT+TC 256x64 | FourOverSix rows |
  | n8k64_wB_{typical, worst} | n8k64_wB | TM-OPT+TC 8x64 | FourOverSix rows |

**Map tags.** Per projection and FlipQuant (ours) unit:
- **typical:** the module whose E0M3 tile share is the lower median over the projection's modules. With n modules
  sorted by (E0M3 tiles, layer order), it is the one at index ⌊(n − 1) / 2⌋.
- **worst:** the densest module (ties: the first), as step 06.
- **FourOverSix and NVFP4:** the projection's first module, as step 06.
- Each chosen module's name and E0M3 share are recorded.

**One repetition.** Every timed kernel is launched on an idle GPU: synchronized before, nothing queued behind another.
1. **Cold cache.** A reduction reads a 512 MiB buffer, 4× the 128 MiB L2 (134,217,728 bytes, from torch); then
   synchronize. The read evicts the weights and leaves L2 full of clean lines, so the GEMM's misses cause no write-backs.
2. **A fresh activation,** as in a forward: a seeded BF16 input [T, in] is copied from a pool of two into the input
   buffer; then synchronize.
3. **TIMED, the activation quantizer:** `Kernel.quant_rows`, the CUDA kernel that `sm120_linear` launches; then
   synchronize.
4. **TIMED, the GEMM** with its fused epilogue: `Kernel.gemm_ptr`, the kernel that `sm120_linear` launches. It reads
   the weights from DRAM and the just-quantized activations from L2; then synchronize.

**Clock.**
- **Primary:** the CUPTI device time of each timed launch (torch.profiler over one block of repetitions of one
  configuration). It excludes the host launch latency, which an event pair includes. At T = 128 a GEMM takes about
  10–30 µs, so a 5–20 µs gap would pull every ratio toward 0.
- **Also recorded:** an event pair around each GEMM launch, i.e. item #3's clock.

**Repetitions and order.**
- Per (projection, T): 3 rounds. Round r starts the configuration list at position r · len / 3 (deviation 1's rotation).
- Each configuration gets 3 untimed and 30 timed repetitions per round, so 90 timed per configuration.
- Reported: the median over all 90; the interquartile range and the min–max are recorded, and the per-round medians
  too.

**Per-forward sums,** as step 06: each projection's median time times its module count. The quantizer is counted by
its launches, net of the reuse measured in step 05.

**Telemetry.**
- **NVML, before and after every block:** SM and memory clock, power, energy (hence the block's mean power), the
  cumulative power-cap violation time, the clock-event reasons, the temperature.
- **A 100 ms nvidia-smi sampler** runs over each process (`<model>.json.telemetry.csv`).
- **The power limit is 500 W** (default 600 W). Changing it needs root, so it is recorded, not changed. Clocks are not
  locked.

## Registered checks (a failure stops the run; it is reported, not worked around)

**Step 1: the cold-cache verification** (`experiments/paper/check_l2_cold.py`), before any model is timed.
- **Cases:** Llama-3.1-8B's gate_proj (14336×4096) and q_proj (4096×4096), at T = 128 and 2048, for stock_wA
  (FourOverSix) and n16k64_wA (the typical 16x64 tags).
- **Conditions,** each an isolated CUPTI-timed launch; 3 rounds in rotated order × 50 repetitions:
  - warm: the same weights just read by an untimed launch;
  - cold512: the registered method;
  - cold1024: a 1 GiB flush;
  - rotation: no flush, over K distinct weight copies with K × size ≥ 4× L2;
  - event512: cold512 timed by item #3's event pair.
- **Pass rule, per case, on the medians:**
  - cold512 ≥ 0.99 × warm;
  - |cold512 / cold1024 − 1| ≤ 2 %;
  - |cold512 / rotation − 1| ≤ 2 %.
- **Recorded, not rules:** cold512 / warm − 1, the size of the L2 effect; and event512 − cold512, the host enqueue gap.

**Step 2: during the run.**
- Before timing, each configuration's output at each (projection, T) equals NativeLinear's fused forward bitwise
  (int16 views). The isolated calls run the deployment kernels on the deployment arguments.
- Every profiled block holds exactly 30 GEMM and 30 quantizer launches.
- No other compute process is on the GPU at any block (NVML), and the GPU is idle at the start.

## Steps

0. **Smoke run,** after this registration: Llama q_proj, T = 128 and 2048, 5 repetitions, and the check with 10, into
   `/home/dev/n16k64_campaign/paper_smoke`. Code fixes it forces are appended below as deviations, with new hashes.
1. **The cold-cache verification** (above), into `/home/dev/n16k64_campaign/paper/gemm_isolated/l2_check.json`.
2. **The run:** `experiments/paper/06b_gemm_isolated.py`, one process per model, into
   `/home/dev/n16k64_campaign/paper/gemm_isolated/<model>.json`.
3. **Tables:** `experiments/paper/07_tables.py` (with `gemm_isolated_tables.py`), then `collect_results.py` into
   `results/paper/`.

## Report (step 07)

**Main tables.**
- Per model, at all 7 T, per-forward µs for: stock_wA, stock_wB, 16x64 typical / worst, 8x64 typical / worst.
- Overheads against the same-placement stock kernel, plus 8x64 against stock_wA.
- The quantizer, per forward.
- Side by side, per prefill shape:
  - the old step-06 overhead;
  - the new typical / worst overheads;
  - the end-to-end CUDA-graph prefill overhead (FlipQuant (ours) vs FourOverSix, same placement).
- The consistency summary.

**Appendix.**
- 256x64.
- The consistency check. The per-forward GEMM difference (typical tags primary, worst also shown) is set against the
  end-to-end CUDA-graph difference, and flagged beyond 1 % of the reference prefill.
- The per-shape detail: median [interquartile range], with the tile width.
- The tags timed, and the telemetry.
- The alternative method's tables, labelled.

**Also updated after the run** (requested): `results/paper_extra/SUMMARY_ACD_zh.md`.
- Its C1 section gets the new primary numbers; the old ones stay as 「CUPTI、連續呼叫、L2 warm、最密集模組」.
- New rows go into its history table.
- The re-measurement note is removed, and 尚待處理 is updated.

## Changes to files registered by the paper protocol (`registration.json`)

- **`experiments/paper/07_tables.py`:**
  - renders deviation 2's tables through the new `gemm_isolated_tables.py`;
  - labels step 06's tables as the alternative method;
  - without `gemm_isolated/` records it renders byte-identical main.md, appendix.md and gemm_superseded.md, which was
    checked on the committed records. tables.json only gains a `method` key.
- **`experiments/paper/collect_results.py`:** also copies `gemm_isolated/` (the records, `l2_check.json`, the
  telemetry CSVs).
- **Untouched:** `bench_gemm.py`, `06_gemm_latency.py`, and every other registered file.

## Deviations (append-only)

1. **2026-09-30, after step 1 failed: the cold-cache method becomes rotation + flush, and the telemetry is fixed.**
   - **Step 1 as registered failed on 2 of 8 cases.**
     - **The record:** `gemm_isolated/l2_check_v1_failed.json`, 3 rounds × 50; medians [IQR], µs.
     - **The failing cases:** Llama q_proj at T = 2048. There, the rotation over distinct weight copies was slower than
       the 512 MiB flush on one copy:
       - stock_wA: 52.2 [52.0, 52.5] vs 50.0 [49.8, 50.2] (−4.3 %);
       - n16k64_wA: 55.4 [55.0, 55.7] vs 54.2 [53.8, 56.0] (−2.1 %).
     - **The other 6 cases passed.** The 512 MiB and 1 GiB flushes agreed within 1.3 % everywhere. Cold vs warm was
       +42–45 % at gate_proj T = 128 and +7–12 % at q_proj T = 2048.
     - **Item #3's event pair** added 27–38 µs of host enqueue gap per call.
     - **The run stopped there;** no model was timed. A smoke run of the benchmark itself had passed: 18 of 18 outputs
       bitwise equal to NativeLinear's forward, and the launch counts right.
   - **A diagnostic** (approved; a diagnostic, not a result) on the two failing cases, 3 rounds × 50, medians [IQR], µs.
     Script: `experiments/paper/diagnose_cold_cache.py`; record: `gemm_isolated/diagnostic_cold.json`.

     | kernel | flush, one copy | rotation, no flush | rotation + flush | flush, then 2 ms idle |
     |---|---:|---:|---:|---:|
     | stock_wA | 50.2 [49.9, 50.5] | 51.4 [51.1, 51.7] | 50.0 [49.8, 50.3] | 50.9 [50.2, 51.5] |
     | n16k64_wA | 54.2 [53.8, 56.0] | 55.3 [55.0, 55.6] | 53.9 [53.6, 55.7] | 54.8 [54.1, 55.5] |

     - **Reading: H2, the state the flush leaves (clock or power), not H3, address diversity.**
       - Rotation + flush equals the flush alone.
       - An idle gap after the flush moves the time toward the rotation.
   - **Decision** (the user, relayed by nvfp4-razer-c9, before the diagnostic's outcome): the registered method becomes
     rotation + flush.
     - Every call uses the next of K distinct copies of the configuration's weights (packed codes and placed scales;
       K × size ≥ 4× L2), and a 512 MiB read-flush runs before it.
     - The reason: it is the most deployment-like under either hypothesis. A forward reads a different allocation per
       layer, on a busy GPU. Under H2, the reading above, it equals the flush alone.
   - **Step 1's amended pass rule,** per case, on the medians:
     - rotation + flush (512 MiB) ≥ 0.99 × warm;
     - rotation + flush (512 MiB) within 2 % of rotation + a 1 GiB flush.
     - Recorded, not rules: the flush on one copy, the rotation without a flush, and item #3's event clock.
     - Step 1 is re-run with the amended method. If it passes, the full run follows with no further confirmation; if it
       fails, the run stops and is reported.
   - **Telemetry fix** (approved). NVML's energy and power readings do not update within the 10–100 ms
     per-configuration blocks: the smoke gave a mean power of 0.0 W.
     - NVML snapshots are now also taken per (projection, T) block, one to a few seconds long.
     - A mean power is computed only over at least 0.5 s with a changed energy counter.
     - The 100 ms nvidia-smi sampler is summarized into the record: the SM clock, the power, and the samples with the
       software power cap active.
   - **06b's skip rule:** the step-1 record must have *passed* to be skipped. The failed record had status "complete".
   - **New sha256:**
     - `bench_gemm_isolated.py` c41c4133c9b24e5a500536347414dd0158c55d58dc458006013e2ab8e5e55e88;
     - `check_l2_cold.py` a42026d8446c955e0b27a19468fd234b17ad90f3ce6a693ab4b0098f2bd364b5;
     - `gemm_isolated_tables.py` da45e223c5121b87447c5b76c8166599dc9160ff055d88c1bd19421b03506aa3;
     - `06b_gemm_isolated.py` 66d8141ca44d00961edbc4e854cf62587968bee072a067cdbb1f5b11dba1bd24;
     - `diagnose_cold_cache.py` b4c2ccfc5081195390f7e8b0f2b67d8359ba9b6bbbe09cd59ee05f06715d65f7 (new).

   **Follow-up, 2026-09-30 06:36 UTC: step 1 re-run and passed; the full run done** (after 46a35a1).
   - **Step 1 with the amended method:** 8 of 8 cases passed (`gemm_isolated/l2_check.json`).
     - Rotation + 512 MiB flush agrees with rotation + a 1 GiB flush within 1.3 %, and with the flush on one copy
       within 0.7 %.
     - Cold vs warm: +42–44 % (gate_proj, T = 128), +4–6 % (q_proj, T = 128), 0 % (gate_proj, T = 2048), +7–15 %
       (q_proj, T = 2048).
     - Item #3's event pair adds 24–31 µs of host enqueue gap per call.
   - **The run:** 06:23–06:36 UTC, four processes, all exited 0.
   - **Registered checks:** every one passed. 1,890 of 1,890 (projection, T, configuration) outputs are bitwise equal to
     NativeLinear's forward; every profiled block has the right launch counts; no other compute process ran.
   - **The power cap was active** in 3 of 28 (projection, T) blocks on Phi-4 (2.9 s: gate_up_proj at T = 4096 and 8192,
     down_proj at 8192) and in 2 of 84 on Qwen (0.6 s: gate_proj and up_proj at 8192). It hit every configuration of
     those blocks alike; the rotated order spreads it.
   - **Results:** tables regenerated (`tables/main.md`, `appendix.md`, `tables.json`); records in `gemm_isolated/`.
   - **`docs/PAPER_EXPERIMENTS.md`** (registered by the paper protocol) gained a 06b section, and its step-07 GEMM
     bullets were updated. Documentation only.
