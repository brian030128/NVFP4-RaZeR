# Kernel-opt A′: 256x64 maps on a 4-arm kernel with 32-row granules — results

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 3. Registration: `results/kernel_opt/registration_3.json`
(148425d). The chain `run_A1.sh` ran on 2026-09-30 from 19:24 to 19:49 UTC on the local RTX PRO 6000 (500 W limit),
with nothing else on the GPU. Tables: `abA1_tables.md`; data: `abA1.json`.

"Today's path" is TM-OPT+TC 256x64 on KernelSet('mixed') from `sm120/build`. Its n16k64_wA granule is one 16-row
m-atom: 2 flag bits per k_block, 16 arms. "A′" is the same maps on KernelSet('mixed256') from `build_A1`: the four
widths rebuilt with `MIXFP4_A_ATOMS_PER_GRANULE=2`, so a warp's two m-atoms share one flag (1 bit per k_block, 4
arms). Both sets use the same tile-table rows, and every call ran at the same width before and after.

## Gates: all passed

| gate | result |
|---|---|
| G3″ self-test | the four g32 builds: PASS patched / FAIL unpatched (same SASS as registered) |
| G1″ SASS | 20/20 other configurations, rebuilt from the A′ sources, keep their patched and unpatched SASS (before builds from `sm120/build`, kernel-opt `build` and `build_e0m3`) |
| G2″ census | g32 128 / 64 / 32 / 16 E2M1 + the same E0M3 OMMAs per width, as expected; 0 predicated |
| G3″ pytest | `test_gemm.py`, `test_select.py` (with `mixed256`) and `test_g32.py`: 623 passed, 0 skipped |
| G-span | measured granules {8p + w, 8p + w + 4} for every g32 width, each inside one 128-row panel and one 256-row tile; n16k64_wA executes each 16-row block's own tag; on a 48-row weight every real block executes its own tag |
| G4″ bitwise | 24,000 comparisons (104 weights: the 256x64 artifacts' real modules re-declared at 256x64 by `retile`, plus synthetic 256x64 maps; 15 T; fused / reuse / unfused / CUDA graph; 4 builds + the set), 0 differences |
| G5″ logits | 4 models × 5 shapes, set:mixed vs set:mixed256: bitwise equal |

`retile` succeeded on every real module, which also confirms that each TC 256x64 artifact is a 256x64 map.

## M1″: per-forward GEMM time of the 256x64 maps (deviation-2 method, cold weights, 4 models × 12 T)

All registered bitwise checks passed (588 / 588 / 336 / 1,008 per model), with no other GPU process.

| | median over the 48 (model, T) | range | cells slower | cells whose round range crosses 0 |
|---|---|---|---|---|
| A′ vs today's path, typical tags | −1.22 % | [−1.94, −0.03] | 0 / 48 | 2 (Llama and Mistral at T = 128: −0.03 / −0.07 %) |
| A′ vs today's path, worst tags | −1.42 % | [−2.64, +0.03] | 1 / 48 (+0.03 %) | 2 |

- The gain is 0.7–1.2 % at T ≤ 64 (Phi-4: 0.1–0.5 %) and 1.1–1.9 % at T ≥ 256.
- The densest modules gain more (worst tags, up to −2.6 %), because with 4 arms an E0M3 k_block no longer costs extra
  taken branches.

**256x64 vs stock_wA, before → after at every T** (typical tags; per-forward GEMM sum, + = slower than stock):

| T | Llama-3.1-8B | Mistral-7B-v0.3 | Phi-4 | Qwen3.8-27B |
|---|---|---|---|---|
| 1 | +1.1 → +0.1 % | +1.2 → +0.3 % | +0.0 → −0.1 % | +1.2 → +0.3 % |
| 4 | +1.2 → +0.4 % | +1.3 → +0.4 % | −0.1 → −0.3 % | +1.2 → +0.4 % |
| 16 | +1.2 → +0.3 % | +1.4 → +0.2 % | +0.2 → −0.0 % | +1.3 → +0.4 % |
| 32 | +1.2 → +0.4 % | +1.4 → +0.4 % | −0.1 → −0.6 % | +1.0 → +0.1 % |
| 64 | +2.0 → +0.8 % | +1.9 → +0.8 % | +0.6 → +0.1 % | +1.6 → +0.8 % |
| 128 | +3.8 → +3.7 % | +3.7 → +3.6 % | +1.6 → +0.5 % | +2.2 → +0.8 % |
| 256 | +2.8 → +1.2 % | +3.1 → +1.2 % | +4.0 → +2.3 % | +4.0 → +2.0 % |
| 512 | +5.6 → +4.2 % | +5.8 → +4.2 % | +3.9 → +2.2 % | +4.4 → +2.7 % |
| 1024 | +3.1 → +1.7 % | +3.0 → +1.7 % | +3.9 → +2.4 % | +3.9 → +2.3 % |
| 2048 | +4.4 → +2.9 % | +4.4 → +2.6 % | +3.4 → +1.9 % | +3.3 → +2.0 % |
| 4096 | +3.5 → +1.8 % | +3.5 → +1.8 % | +3.8 → +2.2 % | +3.5 → +2.1 % |
| 8192 | +4.0 → +2.6 % | +4.1 → +2.8 % | +3.5 → +2.2 % | +3.4 → +2.3 % |

**Relation to #2.** A′ is measured against today's path without #2's define.
- For 256x64 maps, #2 gave −0.56 % median (M1′). A′ gives −1.22 %, and its four-arm tree has no deep path for
  pattern 0 to bypass.
- The two are not stacked here.

## C2″: 4096³ (µs; medians of 3 rotated rounds; the real map is Llama-3.1-8B layer-0 o_proj at 256x64)

| configuration | b2b | isolated | sustained (500 W cap) |
|---|---:|---:|---:|
| stock_wA | 103.0 | 115.1 | 137.9 |
| nodisp | 103.9 | 115.9 | 138.9 |
| n16k64_wA, all-E2M1 | 106.3 | 118.4 | 140.8 |
| g32, all-E2M1 | 105.3 | 116.8 | 139.4 |
| n16k64_wA, real map | 106.9 | 118.8 | 142.4 |
| g32, real map | 105.4 | 117.1 | 140.4 |
| n16k64_wA, all-E0M3 | 110.9 | 124.6 | 147.1 |
| g32, all-E0M3 | 104.8 | 116.5 | 140.4 |

All outputs were bitwise equal.
- **g32 vs n16k64_wA:**
  - all-E2M1 −1.0 / −1.3 / −1.1 %;
  - real map −1.4 / −1.5 / −1.4 %;
  - all-E0M3 −5.6 / −6.5 / −4.5 %.
- **The fixed dispatch cost over nodisp** (all-E2M1) roughly halves: from +2.3 / +2.2 / +1.4 % to +1.3 / +0.8 / +0.4 %.
- **E0M3's penalty is gone.** g32 runs all-E0M3 no slower than all-E2M1; the 16-arm build's all-E0M3 is +6.8 % over
  nodisp, from its branch layout (`results/kernel_opt/e0m3`).
- g32 all-E2M1 is +2.2 / +1.5 / +1.0 % over stock_wA, and nodisp itself is +0.9 / +0.6 / +0.7 %.
- Sustained-mode clocks drift between rounds at the power cap (1.72–2.28 GHz), as in C2′.

**What this says.** Halving the per-k_tile flag reads and the tree depth removes about half of the fixed cost. So the
dispatch's cost is its per-k_tile instructions (the kernels are issue-bound), not memory or latency.

## Adoption: adopted on kernel-opt (2026-09-30, by the user's go)

Every gate passed, and no M1″ cell is slower beyond its per-round range. That meets the amendment's criterion for
routing 256x64 artifacts to `mixed256`.

The user adopted it, relayed by the coordinator.
- `install(kernel='auto')` now runs an artifact whose own map unit (`note.record.unit`, read by
  `artifact.map_unit`) covers whole 128-row panels, i.e. the 256x64 maps, on `KernelSet('mixed256')`. It does so
  when those builds are in the build directory (`SM120_BUILD_DIR=/home/dev/n16k64_campaign/kernel_opt/build_A1`, which
  holds every configuration built from the A′ sources). Otherwise it falls back to `mixed`, which gives the same
  outputs bit for bit.
- The install report's new `routing` field records which set ran and why.
- `'auto_256'` still selects `mixed256` explicitly, and `'auto_mixed'` still selects `mixed`.
- 16x64 and 8x64 artifacts are routed exactly as before.
- tm-opt and the paper's numbers are untouched.

Checks of the routing change:
- `test_g32.py::test_auto_routes_256x64_maps` passes with and without the g32 builds.
- On Llama-3.1-8B, `install(kernel='auto')` with `build_A1` ran the TC 256x64 artifact on the four g32 builds, and its
  logits (1x100) were bitwise equal to set:mixed from `sm120/build`. The TC 16x64 artifact stayed on `mixed`.

M1″ also shows ≥ 1 % per-forward gains at T ≥ 256. By the user's decision (deviation 1 to amendment 2b), the end-to-end
effect is measured later, once, cumulatively with #2 and A.

## Later: amendment 18 (2026-10-03)

The 256x64 path is now `mixed256_ko`. It runs A′'s builds without the site-0 tags, with #4's 64 x 64 epilogue tile at
width 128 and the uniform-branch dispatch (`MIXFP4_UNIFORM_DISPATCH=1`) at every width. It reads its own re-tuned widths
and scheduler rows in the tracked `<gpu>.ko.json`.
- **The effect:** −1.29 % (typical) and −1.26 % (worst) GEMM time per forward against `mixed256` on the paper table, up
  to −7.1 % at T = 256. The gap to `stock_ko` goes from +2.59 % to +0.68 % (typical, median).
- **The routing:** `'auto'` sends 256x64 artifacts to `mixed256_ko`. `paper_256` still selects `mixed256`.
- **The deployment directory** is `build_V` (adopted by the coordinator per the registered rule).
- **Details:** `results/kernel_opt/V/REPORT.md`.
