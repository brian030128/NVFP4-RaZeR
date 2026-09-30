# Kernel optimization (branch `kernel-opt`) — results

Protocol and registration: `PROTOCOL.md`, `registration.json` (af989b9). tm-opt and its committed paper results are
unchanged. "Before" is `sm120/build`, the builds behind every paper result. "After" is the kernel-opt builds
(`/home/dev/n16k64_campaign/kernel_opt/build`). The user decides whether the paper's latency numbers are re-measured.

## Optimization 1: narrow token tiles for the weights-on-B kernel (8x64 maps)

### What changed

- **Three new builds for the 8x64 maps:** `n8k64_wB_m64`, `n8k64_wB_m32` and `n8k64_wB_m16`.
  - The CTA tile is M × 64 × 128, with M = 64, 32 or 16 tokens.
  - They use CUTLASS's ping-pong schedule with 4 MMA warps (1x4).
  - They keep `n8k64_wB`'s 8x64 granule, its 16-arm dispatch and its per-output MMA sequence.
- **A width-selecting set** `KernelSet('mixed_wB')`, kernel name `'auto_wB'`, over {16, 32, 64, 128 = `n8k64_wB`}.
  - Its tile-table rows were tuned by isolated launches on cold weights: synthetic operands, the TC 8x64 maps' tags.
- **Kernel hooks** (`sm120/kernel/LOCAL_CHANGES.md`), all compile-time and inactive in every existing build:
  - `MIXFP4_TILE_M`;
  - `MIXFP4_PINGPONG`;
  - a narrow-M scale-factor path in the collective (active only when M < 128).

### Correctness: all registered gates passed

| gate | result | record |
|---|---|---|
| G1 inert | All 14 existing configurations rebuilt from kernel-opt sources have identical patched and unpatched SASS. | `g1_g2_sass.json` |
| G2 census | 256 / 128 / 64 OMMAs per site; no predicated OMMA. | `g1_g2_sass.json` |
| G3 self-tests | Upstream driver PASS patched, FAIL unpatched (3 new builds, n8k64_wB, n16k64_wA). `test_gemm.py` + `test_select.py`: 418 passed (all of `sm120/tests`: 523). | `g3_*` |
| G4 kernel bitwise | 24,000 comparisons, 0 differ. See below. | `g4_bitwise.json` |
| G5 model bitwise | Logits before = after on all 4 models, at 1x1, 1x16, 1x100, 1x2048 and 4x512. | `g5_model_logits.json` |

G4's coverage:
- all four models' projection shapes;
- 40 real TC 8x64 modules, and 64 synthetic cases (all-E2M1, all-E0M3, random tags, with and without bias);
- 15 token counts from 1 to 8192;
- four paths through NativeLinear: fused quantizer + GEMM, reuse of the quantized input, unfused, and CUDA graph.

**The paper's accuracy numbers therefore hold unchanged for the after builds.**

### Speed

Full tables: `ab1_tables.md`; data: `ab1_*.csv` and `ab1.json`; raw records: `opt1/`. All three measurements ran on
2026-09-30, 10:41–13:51 UTC, on an idle GPU. All 140 end-to-end processes and 4 GEMM runs exited 0. Every registered
check passed: 2,880 isolated-vs-NativeLinear and before-vs-after bitwise checks, graph = eager logits and tokens, and
library provenance.

**M1: per-forward GEMM time.** Deviation-2 method: isolated launches, cold weights, CUPTI, 3 rotated rounds × 30
repetitions. Typical tags; worst tags agree within 0.8 pp. Every round lies within 0.3 pp of the median at T ≤ 1024,
and within 0.7 pp at T ≥ 2048.

| T | Llama | Mistral | Phi-4 | Qwen | after vs stock_wA (FO6, wA): before → after |
|---|---|---|---|---|---|
| 1–64 | −36.9 to −44.9 % | −36.8 to −45.0 % | −23.0 to −37.3 % | −29.2 to −39.4 % | +31…+85 % → +0.1…+2.8 % |
| 128 | −32.7 % | −32.7 % | −16.6 % | −24.0 % | +26…+55 % → +4.1…+5.6 % |
| 256 | −9.6 % | −9.5 % | +0.0 % | −8.6 % | +18…+40 % → +18…+26 % |
| 512 | −6.0 % | −6.0 % | −0.1 % | −5.3 % | +11…+22 % → +10…+15 % |
| 1024 | −0.5 % | −0.5 % | −0.1 % | −2.6 % | +9…+12 % → +8…+9 % |
| 2048–8192 | −0.6 to +0.2 % | −0.5 to −0.0 % | ±0.0 % | −1.4 to −0.2 % | unchanged, +7…+10 % |

**M2: prefill, CUDA graph** (the paper's primary numbers). Median of 5 rounds, rotated order:

| shape | Llama | Mistral | Phi-4 | Qwen | overhead vs FourOverSix (fo6): before → after |
|---|---|---|---|---|---|
| 1x128 | −18.2 % | −19.2 % | −8.4 % | −5.4 % | L +23.5 → +1.1, M +25.2 → +1.2, P +11.6 → +2.2, Q +6.2 → +0.5 % |
| 1x256 | −5.1 % | −5.4 % | +0.4 % | −1.9 % | L +16.5 → +10.6, M +17.6 → +11.3, P +5.7 → +6.1, Q +4.4 → +2.4 % |
| 1x512 | −2.0 % | −2.2 % | +0.0 % | −0.4 % | L +11.1 → +8.9, M +11.9 → +9.4 % |
| 1x1024 | +1.3 % | +0.4 % | +0.1 % | −0.3 % | L +4.2 → +5.5 % |
| 1x2048 … 4x2048 | −0.1 to +0.2 % | −0.1 to +0.0 % | +0.1 % | −0.2 to −0.0 % | unchanged: L/M +4.4…+6.2, P +5.0…+6.2, Q +2.5…+3.1 % |

**M3: decode, CUDA graph** (Experiment D's harness; median of 5 rounds). Tokens per second:

| model | after vs before | after vs FourOverSix on wA (`auto_stock`): before → after |
|---|---|---|
| Llama-3.1-8B | +35.4 % (1x512) … +6.1 % (16x2048) | −6.1…−26.7 % → −0.2…−1.3 % |
| Mistral-7B-v0.3 | +37.9 % (1x512) … +6.3 % (16x2048) | −6.3…−28.0 % → −0.3…−1.3 % |
| Phi-4 | +25.5 % (1x512) … +4.4 % (16x2048) | −4.7…−20.3 % → +0.1…−0.4 % |

After the change, decode runs `n8k64_wB_m16`, except Phi-4's down_proj at batch 16, where the table picks
`n8k64_wB_m32`. Experiment D found the 8x64 decode deficit to be "almost all placement". This optimization removes it.

### What it does not change, and caveats

1. **T ≥ 1024 is unchanged.** The table keeps `n8k64_wB` there, so the 8x64 large-T overhead (about +7…+10 % GEMM,
   +2.5…+6 % prefill vs FourOverSix) is untouched.
   - Its cause is the 1x8 arrangement and the 16-arm dispatch; see the kernel report's "8-column × 64-K frontier".
2. **Mid T (256–512) still carries +10…+26 % GEMM overhead vs stock_wA.** It is a CTA-count problem.
   - At T = 256, Llama's down_proj runs 64 CTAs of 128×128 over K = 14336: 52.3 µs vs stock_wA's 128×64 tile at 32.9.
   - The ping-pong 64×64 build is slower there (59.0 µs): with 4 MMA warps it computes less per CTA.
   - Proposed next (#1b): a cooperative 128 (tokens) × 64 (weights) build.
3. **Llama 1x1024 is +1.3 % in prefill** (slower in 4 of 5 rounds). This is near the noise floor between processes,
   not a kernel effect.
   - Mistral is +0.4 % and Qwen −0.3 %.
   - Phi-4 at T ≥ 256, which runs the identical 128 kernel before and after, differs by +0.0…+0.4 %.
   - Inside the captured forward (diagnostic D1, same process, ABBA order), the GEMMs at 1x1024 take 0.6 % less time
     after.
4. **Eager prefill (supplementary) at small T is host-bound.** Its after-vs-before differences range from −12 % to
   +19 % with no consistent sign: Llama 1x128 +18.6 %, Mistral −11.3 %, Phi-4 −12.0 %, Qwen 1x256 −10.9 %.
   - Diagnostic D2: enqueueing one GEMM takes 14.4–15.3 µs of host time for every build, narrow or not.
   - Diagnostic D3: within one process, before, after, fo6 and fo6-wB have the same eager host time (37.8–45.5 ms,
     varying with order, not with policy).
   - The variation is between processes. It appears to track how many kernel libraries a process has loaded.
   - Graph numbers are unaffected.
5. **The paper's same-placement reference is no longer like-for-like.** stock_wB has no narrow-M build (CUTLASS's
   stock collective lacks the narrow-M SFA path).
   - After the change, FlipQuant 8x64 is up to 25 % faster than FourOverSix-on-wB in per-token decode time (Mistral
     1x512) and up to 16 % at 1x128 prefill. Most of that is tile availability, not format.
   - The deployment reference (FourOverSix on `auto_stock`) is the fair one here.
   - A narrow stock_wB (patching CUTLASS's stock collective) or a no-dispatch narrow build would restore the
     same-placement comparison. Pending the user's decision.

### Diagnostics (not registered; `opt1/diag/`)

- **D1** `diag_graph_kernels.py`: per-kernel CUPTI times inside the CUDA-graph prefill (Llama, ABBA).
  - The GEMM per-forward sums agree with M1: 1x256 −8.9 %, 1x512 −6.3 %, 1x1024 −0.6 %.
  - k/v_proj on `n8k64_wB_m64` are −45 to −49 % at 1x256 and 1x512.
- **D2** `diag_host_launch.py`: host enqueue time per GEMM launch, per build: 14.4–15.3 µs, the same for all.
- **D3** `diag_gaps.py`: the idle time between kernels in a replay and eager host time.
  - Neither differs by policy within a process.
  - The graph's idle total moves with process history (0.14 to 1.9 ms), so it is not a kernel property.

### If the paper's latency numbers are re-measured with `auto_wB`

Only the 8x64 rows would change: FlipQuant 8x64 prefill (step 05), GEMM (06b), C1 and the decode rows of D.
- **Decode:** from 5–28 % below FourOverSix to within 1.3 %.
- **1x128 prefill:** from +6…+25 % to +0.5…+2.2 %.
- **1x256 prefill:** small gains.
- **T ≥ 1024:** unchanged.
- The 16x64 and 256x64 rows would not change: their kernels are untouched (identical SASS).
