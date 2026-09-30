# Why is E0M3 slower? — diagnostic protocol (branch `kernel-opt`)

Written 2026-09-30, before any GPU run of this investigation; the hashes and time are in `registration.json`. Requested
by the user (relayed by nvfp4-razer-c9, 2026-09-30).
- **Diagnostic only:** no paper result, deployment kernel or tile table changes.
- **The diagnostic builds** (`*_xor`) live in a separate build directory, `/home/dev/n16k64_campaign/kernel_opt/build_e0m3`,
  and are never installed.
- **Deviations** are appended at the end.

## Facts to explain

- **C2, n16k64_wA at 4096³:** all-E2M1 +3.0 %, all-E0M3 +7.2 % vs stock_wA. E0M3 costs +4.1 % over E2M1 with the same
  static OMMA count.
- **C3:** the cost tracks the E0M3 share. At T ≥ 2048, random and contiguous tags differ by only about 1 pp, so the cost
  is per unit of E0M3 work, not mixing.
- **RTX 5090 ncu** (`sm120/results/bench/ncu_rtx5090.json`), all-E0M3 vs all-E2M1:
  - no_instruction stalls ×2.6;
  - branch_resolving +21 %;
  - math_pipe_throttle +17 %.

## Hypotheses

- **H1:** the hardware E0M3 MMA has lower throughput than E2M1.
- **H2:** the dispatch path. The arm tree (`dispatch_pattern`) is balanced, 4 compares for every one of the 16 arms, but
  the all-E2M1 arm is reached by fall-through branches while the all-E0M3 arm needs 4 taken branches (instruction-fetch
  redirects).
- **H3:** data-dependent power under the 500 W cap, which lowers clocks.
- **H4:** anything else the tests show.

## Tests A–C (no ncu)

**A — OMMA-only loop** (`experiments/kernel_opt/e0m3/omma_loop.cu`, `test_a.py`).
- **Kernel:** a register-resident loop of 8 independent `prmt` + `mma.sync …m16n8k64…` pairs per iteration. No shared
  memory, no dispatch, no branch in the loop body.
- **Builds:** one library per site, patched by the kernel's own SASS patcher:
  - site 0 E2M1×E2M1;
  - site 1 E0M3×E2M1, the weights-on-A kernels' E0M3;
  - site 2 E2M1×E0M3, n8k64_wB's;
  - site 3 E0M3×E0M3.
- **Checks:** the four unpatched instruction streams must be identical apart from the PRMT selector, and each patch
  census must be complete.
- **Data:** random, constant and zero operands, the same bits for every site.
- **Modes:**
  - isolated launches after 50 ms idle (CUPTI);
  - sustained back-to-back launches for 2 s (CUDA events), with NVML SM clock, power and software-power-cap share;
  - 3 rotated rounds.

**B — arm position** (`MIXFP4_ARM_XOR`).
- **The hook:** in the mixed collective, the dispatch tree is searched on `pattern ^ mask`, and the leaf at position L
  runs pattern L ^ mask's arm.
  - Every arm computes exactly as before; only its code position changes.
  - Unset (0) it is the current dispatch. G1-style check below: the default builds' SASS must not change.
- **Builds:** `n16k64_wA_xor` and `n8k64_wB_xor`, with mask 15, which gives the all-E0M3 arm the all-fall-through path.
- **Timing:** `test_bc.py`, at C2's shape with C2's operands: all-E2M1, all-E0M3 and real tags on default vs xor
  (weights on A), and all-E2M1 / all-E0M3 on n8k64_wB vs n8k64_wB_xor (weights on B).
- **Check:** before timing, the xor builds' outputs must equal the default builds' bitwise.

**C — power and data controls** (`test_bc.py`, the same run).
- **C2's five configurations**, plus:
  - `e0m3_samebits`: e2m1's packed bytes with the E0M3 tag set on every scale byte, i.e. the same bits in another format;
  - `e2m1_const` / `e0m3_const`: constant data (every nibble 0x2, every scale 1.0), tags clear / set.
- **Modes:** C2's back-to-back CUPTI (20 calls), isolated (50 ms idle gaps) and sustained (2 s, NVML). 3 rotated rounds.

**Checks** (a failure stops the investigation and is reported):
- G1-style: `n16k64_wA` and `n8k64_wB` rebuilt from the sources with the hook have the patched and unpatched SASS of
  `sm120/build`;
- the xor builds are bitwise equal to the defaults;
- test A's instruction-stream identity and census.

## Reading rules (fixed before the runs)

Let Δ(X) = t(all-E0M3)/t(all-E2M1) − 1 for kernel X, in a given mode.

- **H1**
  - Supported if test A's isolated site-1 time (random data) is ≥ 2 % above site 0 at equal clocks.
  - Ruled out if within ±1 %.
  - The same for site 2, which covers the weights-on-B case.
- **H2**
  - Supported if Δ(xor) ≤ Δ(default) − 2 pp, i.e. the cost follows the code position.
  - Ruled out if |Δ(xor) − Δ(default)| < 1 pp.
  - The same for the wB pair.
- **H3**
  - Supported if E0M3 draws ≥ 3 % more power than E2M1 on the same data, or runs at a lower median SM clock, in
    test A's or test C's sustained mode, and Δ shrinks by ≥ 2 pp from sustained to isolated.
  - A data-dependent part is shown if Δ with constant data (e0m3_const vs e2m1_const) is ≥ 2 pp smaller than Δ with
    real data.
  - Ruled out if isolated and sustained Δ agree within 1 pp and neither clocks nor power differ.
- **H4:** anything else the controls show. Examples: e0m3_samebits ≠ e0m3 beyond 1 pp (the quantized E0M3 data vs the
  same bits reinterpreted); a residual Δ that no test explains.
- **Output:** a findings table with each hypothesis marked supported, ruled out or inconclusive, and the numbers behind
  it.

## Test D (ncu, later)

- It waits until the user has installed Nsight Compute and enabled counters (relayed by nvfp4-razer-c9).
- It will be appended here as an amendment, with its tool version and metric list, before it runs.
- The metrics: at least `ncu_rtx5090.json`'s, for stock / nodisp / all-E2M1 / all-E0M3 / random50 at 4096³.

## Deviations

1. **Test A was first run with `--iters 64`** instead of the script's default 4096. I underestimated how fast the loop
   runs: each launch took 12.8 µs.
   - That run's "sustained" mode was host-bound (332 W, the power cap never engaged), so it cannot speak to H3.
   - Its isolated result agrees with the re-run (every site within ±0.2 % of site 0).
   - Test A was re-run with the default 4096 iterations (`test_a.json`); the first run is kept as
     `test_a_iters64_hostbound.json`. The conclusions use the re-run.
2. **Test B/C's back-to-back (`b2b`) mode is contaminated by the run order.** Each configuration's b2b block followed
   the previous configuration's 2 s sustained block, which runs at the 500 W cap.
   - So the b2b times are hot-start times, and noisier: e0m3 vs e2m1 +9.2 % b2b, against +4.2 % isolated and +4.0 %
     sustained.
   - The findings use the isolated and sustained modes. The b2b numbers are recorded, not used.
3. **Test D (ncu) is cancelled.** The node cannot enable counters (no reboot, no SYS_ADMIN; relayed by nvfp4-razer-c9),
   so it is not needed for the verdicts (`FINDINGS.md`).
   - Noted per the coordinator: the RTX 5090 ncu record, taken at ncu's base-clock lock, shows all-E0M3 slower than
     all-E2M1 on the same SM design: 4096³ 186.05 vs 173.06 µs (+7.5 %), 14336x4096 T = 2048 281.60 vs 263.55 µs
     (+6.8 %). That survives fixed clocks, consistent with H2 rather than H3.
