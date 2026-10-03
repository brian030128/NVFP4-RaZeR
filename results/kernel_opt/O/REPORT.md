# Kernel-opt item 3: the outlier cells — report (exploratory, disclosed, not registered)

2026-10-03, branch `kernel-opt`.

**The request** (item 3, relayed by the coordinator):
- **16x64 at f = 0:** at 4096x4096 T=2048 and 14336x4096 T=512 the dispatch costs ≈5 % (C3k), against ≈2 % elsewhere.
- **8x64:** in the same two cells the no-dispatch ceiling is +14 % / +10 % over `stock_ko` (C3w), while `stock_wB_ko` ≈
  `stock_ko`.
- "Find the cause (wave quantization, scheduler order, L2, power) with bitwise-safe tools (tile/width/schedule choices
  already in the families, CTA order). Report whether anything recovers them."

**Method** (`exploration/`):
- **Scripts:** `quick_O.py` (parts sweep, sched and modes), `quick_O2.py` (K at a fixed tile count) and `quick_O3.py`
  (weight placement). Outputs are in `*.out`; the raw JSON sha256 are in `raw_sha256.txt`.
- **Kernels:** each a single build at its width:
  - `stock_ko` (stock_wA_e64) and `stock_wB_ko` (stock_wB_e64);
  - the ceilings `ceil16` / `ceil8` (no dispatch);
  - A16 / A8: the adopted builds before amendment 17;
  - U16 / U8: amendment 17's builds.
- **Inputs:** f = 0, i.e. all-E2M1 tags for the mixed builds and FourOverSix weights otherwise.
- **Conditions:**
  - **cold** is M1's: weights rotated past the L2 after a 512 MiB flush, activations quantized after the flush, CUPTI;
  - **warm** is back-to-back on L2-resident weights;
  - **sustained** is 2 s of calls with NVML.
- **Checks:** every scheduler setting's output equals the default's, and the carved weight copies compute what the
  originals do. All passed.

## Findings

**1. The outlier is a cold-weight effect.** Against `stock_ko`, the range over the runs of `quick_O` and `quick_O2`
(the allocator's own weight placement):

| cell | condition | 16x64 ceiling | adopted 16x64 (A16) | 8x64 ceiling | adopted 8x64 (A8) |
|---|---|---:|---:|---:|---:|
| 4096x4096 T=2048 | cold | +3.7 … +6.0 % | +7.4 … +8.5 % | +13.5 … +14.4 % | +15.1 … +17.2 % |
| | warm | −0.4 … +0.6 % | +1.3 … +1.9 % | +4.8 … +5.1 % | +6.9 … +7.5 % |
| 14336x4096 T=512 | cold | +0.0 … +3.1 % | +3.2 … +6.5 % | +7.2 … +10.9 % | +8.6 … +13.6 % |
| | warm | −0.1 % | +1.0 … +1.8 % | +4.2 … +5.5 % | +7.0 … +10.0 % |

Warm, both ceilings are at their usual levels (16x64 ≈ stock; 8x64 at its 1x8 tile cost).

**2. It is specific to K = 4096** (`quick_O2.out`). At a fixed tile count, cold / warm against `stock_ko`:

| K | 4096xK T=2048 (512 tiles): ceil16 | A16 | ceil8 | 14336xK T=512 (448 tiles): ceil16 | A16 | ceil8 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | −0.3 / −2.2 % | +5.7 / +1.2 % | +8.3 / +6.3 % | −0.1 / −0.1 % | +3.1 / +3.6 % | +8.3 / +11.2 % |
| 4096 | **+4.8** / +0.6 % | **+7.8** / +1.9 % | **+14.0** / +5.1 % | **+3.1** / −0.1 % | **+6.5** / +1.0 % | **+10.9** / +4.2 % |
| 8192 | −0.1 / +0.2 % | +0.9 / +1.6 % | +4.2 / +4.3 % | +0.1 / −0.8 % | +1.2 / +0.4 % | +4.5 / +10.1 % |
| 14336 | −0.1 / +0.4 % | +0.1 / +1.9 % | +2.5 / +5.5 % | — | — | — |

- **The mixed kernels pay more than stock for cold weights only at K = 4096.** Their cold-vs-warm penalty exceeds
  stock's there by about 4–9 points. At K = 2048 and at K ≥ 8192 it is about the same as stock's, or smaller.
- **Stock's own cold penalty grows with the weight volume:**
  - about 0 at 4–8 MB of weights (4096x2048, 4096x4096);
  - +1.8 % at 14.7 MB (14336x2048);
  - +13 … +19 % from 16 MB up (4096x8192, 4096x14336, 14336x4096, 14336x8192).
- **Why K = 4096 specifically is not isolated.** The mixed kernels have less latency slack per k_tile: the dispatch, and
  the 1x8 arrangement's 1.5× LDSM per OMMA (36 against 24 per main-loop iteration; the 16x64 ceiling has stock's
  instruction counts). At K = 4096 that slack appears to decide how much of the cold stream is hidden.

**3. And to the 2–3-pass regime at K = 4096** (`quick_O_sweep.out`, cold, the default setting). At 4096x4096:

| T | tiles | passes of 188 CTAs | stock_ko | ceil16 | ceil8 |
|---:|---:|---:|---:|---:|---:|
| 1536 | 384 | 2.04 | 49.3 µs | +6.3 % | +15.6 % |
| 1792 | 448 | 2.38 | 49.7 µs | +6.2 % | +14.8 % |
| 2048 | 512 | 2.72 | 51.0 µs | +6.0 % | +14.4 % |
| 2304 | 576 | 3.06 | 74.4 µs | −0.6 % | +4.4 % |
| 2560 | 640 | 3.40 | 77.1 µs | −1.0 % | +4.4 % |
| 4096 | 1024 | 5.45 | 114.3 µs | −0.2 % | +2.4 % |

- **Stock's time is flat across the three-pass band** and jumps by 46 % at 576 tiles. The mixed kernels' extra latency
  has nowhere to hide in that band.
- **At 14336x4096** the same band (448–560 tiles) shows +1.4 … +2.2 % (16x64 ceiling) and +7 … +12 % (8x64 ceiling).
  At 336 tiles (two passes) the values are −0.3 % and +10.3 %; at 896 tiles (4.8 passes), +0.1 % and +4.4 %.

**4. The split between tiles and dispatch at these cells is not stable** (`quick_O3.out`).
- **Moving every weight copy** by 0 … 192 KB moves the 16x64 ceiling's excess between −0.05 and +2.4 % at 4096x4096
  T=2048.
- **The totals stay put:** the adopted 16x64 path at +7.5 … +8.5 % and the 8x64 ceiling at +12.6 … +14.6 %. The K=8192
  control is flat.
- **So C3k's "dispatch ≈5 %" there is placement-dependent.** The stable quantity is the mixed path's total excess. The
  same holds for the run-to-run disagreement of earlier cold runs, e.g. `stock_wB_ko` +0.5 % against +5.3 %, and
  `stock_ko` 51.0 against 54.4 µs.

**5. Nothing in the families recovers the cells.**
- **Scheduler settings** (all 12, the CTA order; `quick_O_sched.out`). Each kernel's best setting, against
  `stock_ko`'s default:

  | cell | stock | ceil16 | A16 | U16 | ceil8 | A8 | U8 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 4096x4096 T=2048 | −1.55 % | +4.34 % | +6.77 % | +5.47 % | +13.29 % | +16.14 % | +15.88 % |
  | 14336x4096 T=512 | −0.44 % | +1.50 % | +6.09 % | +5.04 % | +9.93 % | +12.82 % | +12.52 % |

  No setting closes more than about 1 point, and stock gains as much.
- **Widths in the families:** 64 and '128x64' are +16 … +50 % there, and 8x64 width 64 is +52 … +83 % (2–4× the
  tiles).
- **Amendment 17's builds:** U16 takes about 1 point off the 16x64 excess; U8 is within ±0.5 points.

**6. Not power.** The sustained runs of every kernel sit at the 500 W cap (SM clock 1,590–2,055 MHz), with gaps that
differ from the cold ones. The cold M1 GEMMs last 50–60 µs.

## Reading

- **The cause:** a cold-weight effect, confined to K = 4096 and to its 2–3-pass band.
  - There the mixed kernels pay more for weights streamed from DRAM than stock does. Their smaller per-k_tile latency
    slack (the dispatch; the 1x8 LDSM) is the likely reason; the exact mechanism is not isolated.
  - Wave quantization sets where it bites: stock's time is flat across the three-pass band. L2 residency (warm)
    removes it.
  - Scheduler order and power do not explain it.
- **Recovery:** nothing bitwise-safe within the families recovers it (widths, scheduler settings).
  - Amendment 17's uniform dispatch takes about 1 point off the 16x64 excess.
  - A real fix needs more latency slack in the mixed mainloop: fewer per-k_tile instructions, or deeper prefetch at
    width 128, which the shared memory does not allow beyond 4 stages.
  - For 8x64 it needs the 2-D warp arrangement that only a map-format change allows.
- **Relevance:** cold weights are the inference condition, since each layer's weights stream once per forward. So these
  cells are real for prefill: Llama/Mistral q/o_proj at T ≈ 1.5–2.5k, and 14336-wide gate/up at T ≈ 384–640.
- **Caveat:** no hardware counters (ncu is not allowed here). The mechanism is inferred from the controls above (warm,
  K, placement, settings, power).
