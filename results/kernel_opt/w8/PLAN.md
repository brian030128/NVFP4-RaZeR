# Kernel-opt: the 8x64 plan (for the user's approval)

**Goal:** make the FlipQuant 8x64 GEMM reach stock latency (`stock_ko`, weights on A) at all T, bitwise-safely.

**Today** (amendment 10, `REPORT.md`): the 8x64 path is behind `stock_ko` per forward by
- +1.3 % at T ≤ 16;
- +2.7 % at T = 32–128;
- +8.9 % at T = 256–1024;
- +10.2 % at T ≥ 2048.

At 4096³ that gap splits into the dispatch (+3.8 … +4.4 %), the 1x8 arrangement (+2.3 … +3.3 %), the site-0 tags
(+1.1 … +1.3 %), the real map's E0M3 tiles (+0.5 … +1.1 %) and `stock_ko`'s #4 tuning (about 1 %).

Every item below keeps the outputs bitwise identical, is registered before its measured runs, and has its own build
directory. tm-opt is untouched.

## Ranked by expected gain (per-forward GEMM against stock_ko)

| # | item | expected gain | where | effort | ETA |
|---|---|---|---|---|---|
| P3 | **Re-tune the `mixed_wB` widths with `--act-warm`** (4b's method) and the decisive-margin rule | −1 … −5 % in mid-T cells; about 0 at T ≥ 2048 | T = 128–1024, where the gap is largest (Llama/Mistral T = 512: +15 %) | tuning (GPU about 1–1.5 h), gates, M1 | 3 h |
| P1 | **#2's dispatch in the 8x64 path** (pattern-0-first) | −0.7 % median; −1.0 … −1.1 % at T ≥ 256 | all T; it loses up to +0.75 % on the worst tags at T ≤ 128, and +1.8 % on all-E0M3 maps | already built and gated (#2); the choice of default only | — (with P2) |
| P2 | **t0 for the wB family** (drop the site-0 PRMT tags) | −0.5 … −1.2 % | all T | builds, patcher check on wB, G1–G5, M1, C2w‴ | 3 h |
| P4 | **#4 for the wB family:** 64 × 64 epilogue tile at width 128, per-call raster/swizzle rows; `stock_wB` tuned alike | −0.5 … −1 % | T ≥ 512 | new builds, schedule tuning, gates, M1 | 4–5 h |
| P5 | **The same-placement reference** (decision a): no-dispatch ceilings of the wB family at every width | none (a reference) | T ≤ 1024 | CPU builds, M1 | 1.5 h |
| P6 | **The 8x64 default** (decision b): `install(kernel='auto')` routes 8x64 artifacts to the adopted wB set | none (routing) | — | adoption amendment and gates | 1 h |
| P7 | **A cumulative registered 8x64 e2e**, as amendment 9 | measures P1–P4 together | prefill on 4 models, decode on 3 | registered run | 4.5 h |

**Expected end state if P1–P4 land** (per-forward against `stock_ko`):

| T | today | after P1–P4 |
|---|---:|---:|
| ≥ 2048 | +10.2 % | about +7 … +8 % |
| 256–1024 | +8.9 % | about +5 … +6 % |
| ≤ 128 | +1.3 … +2.7 % | about +0.5 … +1.5 % |

The rest, about 6–7 points at large T, is the 1x8 arrangement, the dispatch's fixed cost and the E0M3 tiles. No known
bitwise-safe idea reaches it; see "Structural" below.

## Order of work, and why

**P2 (with P1) → P4 → P3 → P5 → P6 → P7.** Total about 17 h of work; P2's results come about 3 h after approval.
- **Kernels before tables.** P2 and P4 change the kernels (t0; the epilogue tile and scheduler hooks). P3 then tunes
  the widths and the scheduler rows once, on the final builds, as amendment 7 did for 16x64. Tuning first would mean
  tuning twice.
- **P1 is a build variant of P2.** Each P2 build comes with and without `MIXFP4_DISPATCH_FREQ=1`, so the default
  dispatch is chosen on P2's M1. I recommend #2's, as for 16x64: typical tags win in 47 of 48 cells.
- **P5's builds run CPU-only during P3's tuning.** Its M1 follows P3.

## The items

**P2: t0 for the wB family.**
- The wB blob tags every MMA's SFA register exactly as the wA blob does, but with 8 m-atoms per warp: 32 identity PRMTs
  in the no-dispatch SASS, against 8 for 16x64.
- The steps:
  - `TAG0=0` builds of the five `mixed_wB` configurations, default and freq, patched with `--untagged-site0`.
  - A patcher check on the wB builds: every OMMA's site equals the strict tagged parse (6b's validation, extended).
  - G1–G5, M1 (vs 1b's set and #2's), and C2w‴.
- Bitwise: the same MMAs in the same order.
- Expected −0.5 … −1.2 %. The ceiling gains 1.1–1.3 %. On 16x64 the real kernel kept about a third of its ceiling gain,
  but the wB kernel carries four times the tags.

**P4: #4 for the wB family.**
- The builds: n8k64_wB with `MIXFP4_EPI_TILE_M/N=64` (and its t0 / freq variants), plus `stock_wB_e64`.
- Scheduler rows for `mixed_wB`, and a `stock_wB` set tuned the same way, so the same-placement reference is not left
  behind.
- First, a check (disclosed exploration) that the 1x8 epilogue accepts the 64 × 64 tile without losing a mainloop
  stage. At width 64 it lost one on 16x64.
- Expected −0.5 … −1 % at T ≥ 512. It was −0.6 % on 16x64, family-neutral. Here `stock_ko` already has it, so the gain
  closes the gap.

**P3: the act-warm re-tune.**
- 1b's rows were tuned with cold activations, the flaw that sank re-tune attempt 1. The 4b method (activations quantized
  after the flush) re-ranked small and mid-T widths on 16x64: −3.4 … −5.6 % at T = 256 and −1 … −1.7 % at T = 1024.
- The re-tune covers all five builds, the `'128x64'` key included, and the scheduler rows.
- The decisive-margin rule means a non-default choice must win beyond the round range, as amendment 7's scheduler
  rows did.
- The worst cell (Llama/Mistral T = 512, +15 %) runs the 128 × 128 tile in a single partial wave, a likely candidate
  for the 128 × 64 tile.
- Phi-4 1x512 (1b's open item) is re-checked in P7.

**P5: the same-placement reference (decision a). What I would do:**
- **Not** a narrow stock_wB. It needs a narrow-M SFA path inside CUTLASS's stock collective; that is real kernel work,
  and the result would no longer be stock CUTLASS. The baseline also shows placement itself costs about 0 at large T
  (stock_wB within +0.2 % of stock_ko at T ≥ 2048).
- **Instead,** build the wB family's own no-dispatch ceiling at every width: `n8k64_wB_{m16,m32,m64,n64}_nodisp_t0`
  and `n8k64_wB_nodisp_t0`, with `patch=False` and E2M1 only. Report the 8x64 path against it per T, alongside
  stock_ko (the target) and stock_wB.
- It measures exactly what dispatch work could still recover at each width.

**P6: the 8x64 default (decision b).**
- Today `'auto'` serves 16x64 and 256x64 maps; 8x64 needs an explicit kernel, and the paper used `n8k64_wB`.
- Recommend: `'auto'` → the adopted width-selecting wB set on kernel-opt, with `'paper_wB'` (`n8k64_wB`) for the paper
  path.
- It is bitwise identical (1b's G4/G5, and P2–P4's gates). Adopt it after P2–P4 so the default is the final set.

**P7: the cumulative 8x64 e2e.** The combined build against the paper n8k64_wB, stock_ko and stock_wB: gates, M1,
prefill on 4 models and decode on 3 (Qwen's hybrid cache is unsupported). It includes the Phi-4 1x512 check, and
records D4's in-graph SM clock.

## Structural: no bitwise-safe proposal

- **The 1x8 arrangement** (+2.3 … +3.3 % at 4096³).
  - The arm count is set by the per-warp extent along the weights: 2 n-atoms × 2 k_blocks = 16 arms.
  - A 2x4 arrangement at 128 × 128 gives each warp 4 n-atoms. That means either 256 joint arms (infeasible) or two
    dispatches per k_tile (about +4 %, more than the 3 % it would save).
  - The only lever is the map structure, for example 16-column granules on B. That changes the format, so it needs the
    user's approval and lies outside this branch's bitwise rule.
- **The dispatch's fixed cost** (+2.5 … +3.6 % with #2's) is the largest residual. B (run tables) and B′ (per-warp
  all-E2M1 bits) both failed on 16x64, and I have no further concrete idea. I flag it rather than propose work.
- **D4's power/clock effect.** In the captured forward the 8x64 GEMM runs at a 1.5–2.6 % lower SM clock under the
  500 W cap; the 1x8 arrangement does 1.5× the shared-memory traffic. There is no lever without clock control, which is
  not allowed. P2's fewer instructions may help a little; P7 measures it.
- **The decisive-margin rule** is used in P3's tuning. The Phi-4 1x512 cross-process effect is re-checked in P7.

Not in scope: 256x64 work (none for now), 16x64 (done for now), and the E0M3 amendment (i)–(iii), which is cancelled.
