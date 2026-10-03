# Kernel-opt: the 8x64 plan, closing summary

2026-10-02, branch `kernel-opt`; amendments 11–15. The plan is `PLAN.md` (a84ab6f). The goal: the FlipQuant 8x64 GEMM at
stock latency (`stock_ko`, weights on A) at all T, bitwise-safe.
- The order was P2 (+P1) → P4 → P3 → P5 → P6 → P7, with P4 folded into P3.
- The user's go was relayed by the coordinator nvfp4-razer-c9, as were the adoption decisions.
- All commits are local; pushing kernel-opt waits for the user's grant.

## What was done and adopted

| item | amendment | result | status |
|---|---|---|---|
| P2: t0 for the wB family (site-0 tags dropped) | 11 | −0.76 % per forward (−1.3 … −2.2 % at T ≥ 256) | adopted |
| P1: #2's dispatch as the 8x64 default | 11 | −0.53 %, faster in 48 / 48 typical-tag cells | adopted (`build_P2freq`) |
| P4: #4's epilogue tile | 12 (folded) | already 64 × 64 at width 128; '128x64' would lose a stage | only stock_wB_e64 |
| stock_wB tuned alike (stock_wB_e64 + rows) | 12 | −0.52 % against stock_wB | adopted (`stock_wB_ko`) |
| P3: act-warm width and scheduler re-tune | 12 | neutral (−0.01 %); rule not met (one cell +0.84 %) | not adopted |
| P3b: 11 confirmed width cells | 12b | −0.06 %; −1.1 % at T = 128; rule met | adopted (0c8dbcb) |
| P5: no-dispatch ceilings at every width | 13 | the gap splits into +2.1 % dispatch and +1.3 % tiles | reference |
| P6: `'auto'` → the adopted 8x64 set | 14 | routing and logits gates passed | adopted |
| P7: cumulative e2e | 15 | prefill +3.8 %, decode −0.3 % against stock_ko | descriptive |

Every adopted change is bitwise identical to the paper kernel (G4 / G5 at each step).

## Where the 8x64 path stands against stock_ko

**Per-forward GEMM time, isolated (M1, typical tags, median over the 4 models):**

| T | before the plan (amendment 10) | now (amendments 12b / 13) | of which: dispatch / tiles (P5) |
|---|---:|---:|---|
| ≤ 16 | +1.3 % | +0.3 … +1.0 % | +0.9 % / −0.1 % |
| 32–128 | +2.7 % | +1.2 … +2.0 % | +1.4 % / +0.3 % |
| 256 / 512 / 1024 | +8.9 % (band) | +5.7 / +9.0 / +6.0 % | +3.1 % / +3.4 % (band) |
| ≥ 2048 | +10.2 % | +7.0 … +7.4 % | +2.7 % / +4.2 % |

**End to end (P7, CUDA graph, median over the models):**
- prefill: +0.5 % at 1x128, +2.0 % at 1x256, +7.4 % at 1x512, +5.7 % at 1x1024, and +3.7 … +4.8 % at 1x2048 … 1x8192;
- decode: −0.3 % tokens per second, i.e. at stock's speed.

## What is left, and what would move it

- **The same-placement tiles** (+3 … +6 % at T ≥ 256, about 0 at T ≤ 128).
  - The weights-on-B 1x8 / 1x4 arrangement and, at T = 512, a single partial wave of 128 × 128 tiles.
  - Not bitwise-reachable. The only lever is a map-format change (e.g. 16-column granules on B), which needs the
    user's approval.
- **The dispatch** (+1 … +3 %).
  - The fixed cost of the 16-arm tree; the E0M3 tiles add about 1 point on worst tags.
  - No concrete bitwise-safe idea remains (B and B′ failed on 16x64).
- **In the graph** the 8x64 GEMMs run at a 4–5 % lower SM clock under the 500 W cap at large T (P7, D4). This adds
  about 2–3 points to the in-graph GEMM gap. There is no lever without clock control, which is not allowed.
- **Phi-4 1x512** (open): at that shape the adopted path is no faster than the paper kernel end to end, because the
  adjacent non-GEMM kernels are +2.7 % slower next to it. The mechanism is not isolated.
- **Measurement floor** (amendment 12, post hoc): identical computations can differ by up to about 2 % per GEMM across
  configurations. Single-cell differences below that are not evidence.

## Maps and tags

Every measurement used the paper's TM-OPT+TC 8x64 artifacts (FourOverSix for the stocks); M1 used the typical
(lower-median) and worst (densest) modules. The FlipQuant calibration will change later. The bitwise results do not
depend on the map; the dispatch costs and P3b's width choices depend on the E0M3 shares (1.5–3.7 % of tiles here).

## Later: amendment 17 (2026-10-03)

The 8x64 path's widths 64, '128x64' and 128 now take the pipelined flag read (`MIXFP4_PIPE_FLAGS=1`), with #2's dispatch.
- **The effect:** −0.17 % (typical) and −0.16 % (worst) per forward. The gap to `stock_ko` goes from +4.20 % to +3.83 %
  (typical, median).
- **The deployment directory** is `build_U` (adopted by the coordinator per the registered rule), superseding
  `build_P2freq` / `build_P3freq`.
- **Details:** `results/kernel_opt/U/REPORT.md`.

