# Kernel-opt 8x64 plan, P7: the cumulative registered 8x64 run — report

2026-10-02, branch `kernel-opt`. The protocol is amendment 15 of `results/kernel_opt/PROTOCOL.md`, registered at
417283e. `run_w8p7.sh` ran 18:15–21:24 UTC with no deviation. The e2e numbers are in `cum8_e2e_tables.md` and
`cum8_e2e.json`, the in-graph split in `p7_split_tables.md` and `p7_split.json`. Descriptive: nothing is adopted from it.

**The policies** (each a separate process per model and round; 5 rounds, rotated):
- **ours-8x64-adopted:** the adopted 8x64 path, `'auto'` on `build_P3freq`. That is `mixed_wB_ko`: t0, #2's dispatch,
  and the adopted table's widths.
- **ours-8x64-paper:** the paper's n8k64_wB.
- **fo6-ko:** `stock_ko`, the tuned stock with the weights on A, and the target.
- **fo6-wB-ko:** `stock_wB_ko`, the tuned stock with the weights on B. This was the coordinator's change.

## The gates and checks

- **G0:** 36 files and 11 builds as registered.
- **The routing test** on `build_P3freq`: `'auto'` → `mixed_wB_ko`, and `'auto_stock_wB'` → `stock_wB_ko`.
- **Every registered e2e check passed** for all 32 prefill cells × 4 policies × 5 rounds and all 18 decode cells:
  - graph logits equal eager's, and decode tokens equal eager's;
  - every loaded library is a build of its policy's directory;
  - #2's define is on exactly the 8x64 builds;
  - the installed set and table are the policy's.

## End-to-end prefill (CUDA graph, ms per forward; median over the 4 models; negative = faster)

| shape | adopted vs fo6-ko (models' range) | adopted vs paper 8x64 | paper 8x64 vs fo6-ko | adopted vs fo6-wB-ko | fo6-wB-ko vs fo6-ko |
|---|---:|---:|---:|---:|---:|
| 1x128 | +0.50 % (−0.1 … +0.6) | −14.5 % | +17.9 % | −11.3 % | +13.6 % |
| 1x256 | +1.98 % (+0.8 … +2.7) | −8.4 % | +11.3 % | −5.7 % | +7.6 % |
| 1x512 | +7.41 % (+1.9 … +7.7) | −2.2 % | +9.0 % | +4.4 % | +1.9 % |
| 1x1024 | +5.66 % (+2.4 … +9.2) | −1.2 % | +7.0 % | +4.5 % | +1.1 % |
| 1x2048 | +4.82 % (+2.5 … +5.4) | −1.3 % | +6.2 % | +4.0 % | +0.8 % |
| 1x4096 | +4.14 % (+2.3 … +4.5) | −1.1 % | +5.4 % | +3.3 % | +0.7 % |
| 1x8192 | +3.65 % (+1.9 … +3.8) | −1.0 % | +4.7 % | +3.0 % | +0.6 % |
| 4x2048 | +4.34 % (+2.1 … +4.5) | −1.1 % | +5.6 % | +3.6 % | +0.7 % |

- **Over all 32 cells:** the adopted path is +3.80 % against fo6-ko (median) and −1.27 % against the paper 8x64 kernel.
  - It beats the paper 8x64 kernel in every round in 24 cells and loses in none.
- **Qwen3.8-27B is the lowest at every shape:** −0.1 … +2.5 % against fo6-ko. Its GEMMs are a smaller share of the forward.
- **At 1x128 the adopted path is at stock_ko's latency** (+0.5 %); the paper kernel there is +18 %. The remaining excess
  sits at 1x512 … 1x1024.
- **Same placement costs little at large T:** fo6-wB-ko is within +0.6 … +1.1 % of fo6-ko at T ≥ 1024.

## End-to-end decode (CUDA graph, tokens per second; Llama-3.1-8B, Mistral-7B, Phi-4; positive = faster)

| setting | adopted vs fo6-ko (models' range) | adopted vs paper 8x64 | adopted vs fo6-wB-ko |
|---|---:|---:|---:|
| 1x512 | −0.72 % (−0.7 … −0.0) | +35.6 % | +30.8 % |
| 4x512 | −0.25 % (−0.2 … −0.2) | +26.1 % | +22.8 % |
| 16x512 | −1.41 % (−1.5 … −0.2) | +13.7 % | +11.0 % |
| 1x2048 | −0.25 % (−0.3 … +0.1) | +22.5 % | +19.6 % |
| 4x2048 | −1.16 % (−1.2 … −0.1) | +14.9 % | +12.1 % |
| 16x2048 | −0.37 % (−0.4 … −0.4) | +6.2 % | +5.2 % |

- **Decode:** the adopted 8x64 path runs within −1.5 … +0.1 % of stock_ko's tokens per second (median −0.32 %).
- **Against the paper 8x64 kernel:** +4.6 … +38.1 % (median +18.1 %), from the narrow tiles (optimization 1) and what
  followed.
- **fo6-wB-ko** has no narrow tiles and is 4–25 % slower than fo6-ko in decode.

## The in-graph split (D4's method; `p7_split_tables.md`)

| shape | policy | wall vs fo6-ko | GEMM vs fo6-ko | SM clock (MHz) | power (W) |
|---|---|---:|---:|---:|---:|
| Llama 1x2048 | adopted 8x64 | +4.53 % | +9.10 % | 1980 | 469 |
| | paper 8x64 | +5.65 % | +12.02 % | 2066 | 413 |
| | fo6-ko | — | — | 2081 | 407 |
| | fo6-wB-ko | +0.95 % | +1.84 % | 2298 | 313 |
| Llama 1x4096 | adopted 8x64 | +3.56 % | +11.00 % | 1938 | 500 (cap) |
| | paper 8x64 | +4.79 % | +14.03 % | 1938 | 500 (cap) |
| | fo6-ko | — | — | 2014 | 500 (cap) |
| | fo6-wB-ko | +0.77 % | +1.65 % | 1991 | 500 (cap) |
| Phi-4 1x512 | adopted 8x64 | +6.31 % | +12.52 % | 2572 | 157 |
| | paper 8x64 | +6.54 % | +15.00 % | 2404 | 207 |
| | fo6-ko | — | — | 2524 | 164 |
| | fo6-wB-ko | +0.73 % | +1.39 % | 2602 | 133 |

- **The clock (D4's finding, re-measured on the adopted path).** At Llama 1x2048 and 1x4096 the adopted 8x64 GEMMs run at
  a 3.8–4.9 % lower SM clock than fo6-ko, and at 1x2048 they draw 15 % more power.
  - Their in-graph GEMM gap, +9.1 % and +11.0 %, is larger than M1's isolated +7.2 % and +7.6 % (amendment 12b, Llama).
  - The clock deficit accounts for most of the difference.
- **Idle time.** At Llama 1x2048 it is bimodal per capture, as D4 found: about 0.15 or 2.3 ms for ours-8x64-paper and
  fo6-wB-ko.

## The Phi-4 1x512 check (1b's open item)

- **E2E:** at Phi-4 1x512 the adopted path is +0.77 % against the paper 8x64 kernel, with a round range of −0.59 to
  +1.12 %. That is not distinguishable from equal. Every other Phi-4 prefill shape is −1.2 … −9.9 %. Against fo6-ko it
  is +7.59 %.
- **It is not D4's idle artifact.** In the split, 23 of 24 captures at Phi-4 1x512 have about 2.3 ms of idle. Only one
  capture (paper 8x64) has 0.2 ms. The policies' medians are equal within 80 µs.
- **It is not the clock.** The clock is not lower for the adopted path there (2,572 MHz vs fo6-ko's 2,524 MHz, at
  157 W).
- **It is not the map's tags.** The E0M3 share of M1's typical module equals the model's mean (about 1.5–2 %).
- **Where the time goes in the split:** the adopted path's GEMMs are −2.2 % against the paper kernel's, but its other
  (non-GEMM, non-quantizer) kernels are +2.7 %. The walls differ by −0.2 %.
  - Each policy runs identical non-GEMM work, so the kernels around the 8x64 GEMMs take longer next to the adopted
    builds.
  - At T = 512 those GEMMs include the '128x64' cooperative tile on qkv_proj. The mechanism is not isolated.
- **In-graph GEMM gap:** +12.5 % against fo6-ko, against M1's isolated +7.4 % for Phi-4 at T = 512.
- **Status:** the item remains open, now narrowed to non-GEMM kernel time adjacent to the 8x64 GEMMs at this shape.

## Maps and tags used

- The paper's TM-OPT+TC 8x64 and FourOverSix artifacts, with sha256 in `registration_15.json`. The e2e and the split
  install every module of each model; M1's typical/worst tags do not apply.
- The FlipQuant calibration will change later. The e2e depends on the maps' E0M3 shares through the dispatch, which is
  about 1.5–3.7 % of tiles here.
