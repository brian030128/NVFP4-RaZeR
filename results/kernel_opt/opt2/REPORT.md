# Kernel-opt #2: frequency-aware dispatch (all-E2M1 pattern first) — results

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 2, with its deviation 1, and amendment 2b (M2′). Registration:
`results/kernel_opt/registration_2.json`. The gates and measurements ran on 2026-09-30 from 17:53 to 18:43 UTC
(`run_opt2.sh`, then `run_opt2_resume.sh` from G4′). They ran on the local RTX PRO 6000 (500 W limit) with nothing
else on the GPU. Full tables: `ab2_tables.md`; data: `ab2.json`.

"Default" is the paper's builds (`sm120/build`; for 8x64, optimization 1b's `build`). "freq" is the same sources
built with `MIXFP4_DISPATCH_FREQ=1` (`build_freq`), which tests pattern 0 (all-E2M1) first and otherwise enters the
unchanged tree.

## Gates: all passed

| gate | result |
|---|---|
| G1′ hook inert | n16k64_wA and n8k64_wB rebuilt without the define: patched and unpatched SASS equal to `sm120/build` |
| G1′/G2′ SASS | 7/7 builds without a dispatch keep their SASS; 13/13 dispatching builds keep their census, with 0 predicated OMMAs |
| G3′ self-test | n16k64_wA, n16k64_wA_n16, n8k64_wB, n8k64_wB_m16: PASS patched / FAIL unpatched |
| G3′ pytest | `test_gemm.py` + `test_select.py` on `build_freq`: 459 passed |
| G4′ bitwise, weights on A | 40,320 comparisons (144 weights: real 16x64 and 256x64 modules plus synthetic), 0 differences |
| G4′ bitwise, weights on B | 28,800 comparisons (104 weights), 0 differences |
| G5′ logits | 16x64, 256x64 and 8x64 artifacts, all 4 models, 5 shapes: bitwise equal |

Deviation 1 (a script assertion in G4′, fixed; the chain resumed at G4′) is in the protocol.

## M1′: per-forward GEMM time, freq vs default (deviation-2 method, 4 models × 12 T)

All registered bitwise checks passed (1,680 / 1,680 / 960 / 2,880 per model: isolated path = NativeLinear, and
freq = default on the timed operands), with no other GPU process.

| unit | typical tags: median [min, max] over 48 (model, T) | cells slower | worst tags: median [min, max] | cells slower |
|---|---|---|---|---|
| 16x64 | −0.46 % [−1.09, −0.11] | 0 / 48 | −0.14 % [−0.89, +0.47] | 17 / 48 |
| 256x64 | −0.56 % [−1.40, −0.14] | 0 / 48 | −0.50 % [−1.21, +0.08] | 1 / 48 |
| 8x64 | −0.54 % [−1.55, +0.10] | 1 / 48 | −0.10 % [−1.07, +0.94] | 20 / 48 |

The worst tags are the densest module per shape. More non-zero patterns there pay the extra jump in front of the tree,
so the gain shrinks or reverses.

**16x64 vs stock_wA, before → after, at every T** (typical tags; the per-forward GEMM sum, + = slower than stock):

| T | Llama-3.1-8B | Mistral-7B-v0.3 | Phi-4 | Qwen3.8-27B |
|---|---|---|---|---|
| 1 | +1.1 → +0.6 % | +1.2 → +0.7 % | +0.9 → +0.6 % | +1.1 → +0.5 % |
| 4 | +1.1 → +0.7 % | +1.2 → +0.8 % | +0.9 → +0.7 % | +0.9 → +0.6 % |
| 16 | +1.2 → +0.7 % | +1.0 → +0.8 % | +1.0 → +0.7 % | +0.9 → +0.6 % |
| 32 | +1.3 → +0.9 % | +1.3 → +1.1 % | +0.5 → +0.4 % | +0.8 → +0.4 % |
| 64 | +1.9 → +1.5 % | +1.8 → +1.5 % | +1.5 → +1.2 % | +1.5 → +0.9 % |
| 128 | +4.3 → +4.0 % | +4.4 → +4.2 % | +2.5 → +1.9 % | +1.9 → +1.1 % |
| 256 | +3.5 → +2.8 % | +3.4 → +2.8 % | +3.6 → +3.0 % | +4.3 → +3.1 % |
| 512 | +5.6 → +5.1 % | +5.3 → +4.8 % | +4.5 → +3.6 % | +4.4 → +3.6 % |
| 1024 | +3.2 → +2.5 % | +3.0 → +2.5 % | +3.4 → +2.7 % | +3.8 → +3.0 % |
| 2048 | +3.8 → +3.1 % | +4.1 → +3.5 % | +3.3 → +2.7 % | +3.1 → +2.6 % |
| 4096 | +3.5 → +2.9 % | +3.2 → +2.6 % | +3.4 → +3.0 % | +3.2 → +2.6 % |
| 8192 | +3.8 → +3.3 % | +3.9 → +3.4 % | +3.3 → +2.9 % | +3.2 → +2.7 % |

The 256x64 maps (on the same kernels) move the same way: for example, Phi-4 reaches stock at T ≤ 32 (−0.3 to −0.1 %),
and T = 512 goes from +4.2–5.8 % to +3.5–5.2 %. All values are in `ab2_tables.md`.

**The residual 16x64 gap after #2:**
- T ≤ 32: +0.4 to +1.1 %.
- T = 64: +0.9 to +1.5 %.
- T = 128: +1.1 to +4.2 %.
- T ≥ 256: +2.5 to +5.1 %, largest at T = 512.

## C2′: 4096³ (µs; medians of 3 rotated rounds)

| configuration | b2b | isolated | sustained (500 W cap) |
|---|---:|---:|---:|
| stock_wA | 103.2 | 115.0 | 140.4 |
| nodisp (n16k64_wA without dispatch) | 104.0 | 116.4 | 141.0 |
| default, all-E2M1 | 106.4 | 118.5 | 145.0 |
| freq, all-E2M1 | 106.0 | 117.8 | 145.0 |
| default, real map | 106.9 | 119.1 | 146.1 |
| freq, real map | 106.6 | 118.8 | 142.6 |
| default, all-E0M3 | 111.0 | 124.3 | 148.6 |
| freq, all-E0M3 | 112.3 | 126.3 | 150.5 |

**freq vs default:**
- **All-E2M1:** −0.4 % (b2b), −0.6 % (isolated), 0.0 % (sustained).
- **Real map:** −0.3 / −0.3 / −2.4 %.
- **All-E0M3:** +1.2 / +1.6 / +1.3 % (the extra jump).
- **Weights on B:** −1.3 / −1.6 / −1.2 % (all-E2M1) and +2.0 / +2.3 / +1.9 % (all-E0M3).

Sustained mode runs at the 500 W cap. Its SM clocks drift between rounds (1.64–2.32 GHz across cells), so its
single-cell differences (for example −2.4 %) are within that drift.

**What this says about the fixed dispatch cost:**
- Default all-E2M1 is +2.3 % (b2b) / +1.8 % (isolated) over nodisp; freq is +1.9 / +1.1 %.
- Testing pattern 0 first removes only 0.4–0.7 pp of it.
- The rest is not the tree's compares. It comes with having the dispatch at all: the per-k_tile flag reads and the
  branch between the register pipeline's k_blocks, or the larger 16-arm body.
- nodisp itself is +0.8 % / +1.2 % over stock_wA.

## M2′: end-to-end prefill

Triggered by M1′ (amendment 2b) and registered separately. The results are appended here when it finishes.
