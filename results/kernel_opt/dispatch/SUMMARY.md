# Dispatch-skip ceiling (CPU analysis) and D4 (the 8x64 end-to-end excess)

Diagnostics on branch kernel-opt, 2026-09-30; not registered measurements. No kernel or paper result changes.

## Dispatch-skip ceiling (`experiments/kernel_opt/dispatch_ceiling.py`)

**Inputs:**
- the paper run's TM-OPT+TC maps for Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B;
- units 16x64, 256x64 (executed by n16k64_wA as 16-row granules) and 8x64 (the n8k64_wB family);
- each kernel's granules, CTA weight panels, warp spans and per-(warp, k_tile) 4-bit dispatch patterns, as its TiledMma
  lays them out (script docstring).

**Aggregation:**
- Shares are FLOP-weighted over each forward's modules (out × in).
- `perm` is the best output-channel permutation for that metric. It is exactly optimal for panels and warp spans; for
  the patterns it is a pairing heuristic.
- `random` is seeded Bernoulli tags at each module's own density.
- Full table: `summary_table.md`; data: `dispatch_ceiling.json`; the per-variant table with the adjacent-pairing
  sensitivity rows: `dispatch_ceiling_table.md`.

| unit | E0M3 tiles | modules with no E0M3 tile | all-E2M1 CTA panels (map / best perm) | all-E2M1 warp spans (map / best perm) | (warp, k_tile) patterns: all-E2M1 / single-granule / 2+ granules / all-E0M3 | taken branches per (warp, k_tile) |
|---|---:|---:|---:|---:|---:|---:|
| 16x64 | 1.6–3.3 % | 0.0 % | 0.0–0.2 % / 13–26 % | 3.6–10.2 % / 13.5–26.4 % | 87.7–93.9 / 5.9–11.5 / 0.2–0.8 / 0.00 % | 0.06–0.13 |
| 256x64 | 4.1–9.8 % | 0.0–0.5 % | 0.2–6.7 % (already clustered) | 0.2–6.7 % | 81.7–92.1 / 0.0 / 7.7–17.0 / 0.2–1.3 % | 0.16–0.39 |
| 8x64 | 1.3–2.5 % | 0.0 % | 0.0–0.5 % / 20–32 % | 7.2–14.5 % / 20.7–32.6 % | 90.5–95.0 / 4.9–9.0 / 0.1–0.5 / 0.00 % | 0.05–0.10 |

**Reading:**
- **The maps' E0M3 tiles are scattered.** At 16x64 and 8x64 the pattern shares equal those of random tags at the same
  density (e.g. Llama 16x64: 88.9 % vs 88.8 % all-E2M1 k_tiles). No panel over the full K is clean.
- **256x64 maps are clustered by construction.** A 256-row tile covers both granules of a warp, so only patterns 0, 5,
  10 and 15 occur.
- **All-E0M3 per-(warp, k_tile) patterns are practically absent:** 0.00 % at 16x64 and 8x64, 0.2–1.3 % at 256x64.

**Latency ceilings** (from the measured dispatch costs):
- The fixed cost with every k_tile all-E2M1 is +2.0–2.6 % over the no-dispatch build (C2, C3, and e0m3/test_bc +2.5 %).
- A taken branch costs about 1 % per k_tile (4 taken branches = +4.2 %, results/kernel_opt/e0m3).

| skip level | work it could cover | ceiling |
|---|---|---|
| module (run a no-dispatch kernel) | ≤ 0.5 % of FLOPs | ≈ 0.01 % |
| CTA panel over the full K | ≤ 0.5 % (16x64, 8x64), ≤ 6.7 % (256x64) | ≤ 0.2 % |
| the same after an output-channel permutation (needs an output gather) | 13–32 % | ≤ 0.8 %, minus the gather |
| warp span over the full K | 4–15 % | ≤ 0.4 % |
| per (warp, k_tile), i.e. today's granularity: a pattern-0 fast path | 82–95 % of k_tiles | up to about 2 % if the fixed cost is the tree's compares; less if it is the flag reads (to be measured by #2) |
| fewer taken branches for the non-zero patterns | 5–18 % of k_tiles | ≤ 0.15 % (16x64, 8x64), ≤ 0.4 % (256x64) |

**So only the per-k_tile level can pay.** #2's design follows from the histogram:
- test pattern 0 first (1 compare, fall-through);
- then the existing tree unchanged, so every other pattern keeps its taken-branch count plus one jump.
- A second test for all-E0M3 would add a compare to the 5–18 % non-zero patterns, to help 0–1.3 %, so it is not done.

## D4: where the 8x64 end-to-end excess goes (`experiments/kernel_opt/diag_e2e_split.py`, `d4_e2e_split_llama8b.json`)

**Setup:**
- Llama-3.1-8B, one process, CUDA-graph prefill.
- Policies ABCDE-EDCBA: ours-8x64 (n8k64_wB), ours-8x64-opt (#1b set), ours-16x64, fo6, fo6-wB.
- 20 replays each under CUPTI, NVML every 5 ms.
- Figures are the mean of the 2 passes, vs fo6.

| shape | policy | wall | GEMM | quantizer | other kernels | idle | SM clock |
|---|---|---:|---:|---:|---:|---:|---:|
| 1x2048 | ours-8x64 | +2972 µs (+5.0 %) | +3041 µs (+11.4 %) | +52 | +184 | −192 | 1998 vs 2051 MHz |
| 1x2048 | ours-16x64 | +949 | +1045 (+3.9 %) | +14 | +98 | −177 | 2032 |
| 1x4096 | ours-8x64 | +6284 µs (+4.8 %) | +6150 µs (+11.3 %) | +4 | −126 | +231 | 1965 vs 1995 MHz |
| 1x4096 | ours-16x64 | +2294 | +1963 (+3.6 %) | −15 | −49 | +330 | 1991 |

**Reading:**
- **The 8x64 end-to-end excess at T ≥ 2048 is GEMM time.** The quantizer, the other kernels and the idle time explain
  nothing systematic.
- **Inside the power-capped forward the n8k64_wB GEMM is relatively slower than in isolation:** +11.3–11.4 % vs
  stock_wA's, against +9.5–9.9 % isolated (M1). That extra 1.5–2 pp is what the paper's consistency check saw as the
  excess.
- **It comes with a lower SM clock** at the 500 W cap: 1.5–2.6 % below FourOverSix's. That points to more power per
  unit of work for the 1x8-arrangement kernel. The 16x64 kernel shows almost none (+3.9 % in graph vs +3.6 % isolated).
- **The graph's idle time is bimodal per capture,** about 0.15 ms or 2.0–2.4 ms per forward, alternating across
  captures whatever the policy.
  - So it is a capture artifact, not a property of the kernels.
  - It also qualifies the earlier build-mixing reading (the Phi-4 diagnostic D1b): the idle differences seen there are
    within this artifact's range.
