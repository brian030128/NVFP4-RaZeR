# Kernel-opt optimization 1b: a cooperative 128 × 64 weights-on-B tile for mid T — closing report

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 1 (`registration_1b.json`). Gates: a9e502d. The run was paused
and resumed; both notes are in the protocol:
- **Note 1 to amendment 1, the pause.** On 2026-09-30 at 16:23 UTC the user paused 1b, before its decode measurement
  (M3). M1 and M2 were done; the remaining Qwen prefill processes ran with the same settings.
- **Note 2 to amendment 1, the resume.** On 2026-10-01 the user gave the start. The registered condition for M3 holds:
  the 1b re-tune changed decode-bucket widths of three Phi-4 shapes.
  - M3 ran from 12:26 to 13:14 UTC as registered (`resume_m3.sh`): `ab_e2e.py --what decode`, with the same policies,
    settings, rotation, checks and output directory.
  - There was no deviation; all 60 processes passed the registered checks.
  - The libraries, the tile table and the two harness scripts are the registered ones. Code changed since registration
    selects the same builds and table for these policies; this is disclosed in note 2.

**What 1b added.** `n8k64_wB_n64` is a cooperative 128 (tokens) × 64 (weights) CTA tile with 2x4 warps. It has the
per-warp shape and 16-arm dispatch of `n8k64_wB_m64`, so its outputs are bitwise equal. It joins `KernelSet('mixed_wB')`
under the key `'128x64'`, and the `mixed_wB` rows were re-tuned (`opt1b/table_opt1b.json`).

**Files:**
- Tables: `../ab1b_tables.md`. Data: `../ab1b.json`, `../ab1b_{gemm,prefill,decode}.csv`.
- Records: `gemm/`, `e2e/prefill/`, `e2e/decode/`. Logs: `commands_measurements.log`, `commands_e2e.log`.
- Gates: `../opt1b_g*`.

## Gates (all passed)

| gate | result |
|---|---|
| G1 / G2 | 14 existing configurations keep their SASS; the four new builds (n8k64_wB_m64/m32/m16 and n8k64_wB_n64) have their census, with none predicated |
| G3 | n8k64_wB_n64 self-test PASS; pytest 459 passed |
| G4 | 9,600 bitwise comparisons against n8k64_wB, 0 differences |
| G5 | whole-model logits bitwise equal |

## Results

**M1: GEMM time.** Deviation-2: cold weights, CUPTI, 3 rotated rounds × 30. The table shows the per-forward GEMM sum
with typical tags: the 1b set against optimization 1's set, and the gap to stock_wA from optimization 1 to 1b, in %.

| model | T = 128 | T = 256 | T = 512 | T = 1024 | T ≥ 2048 |
|---|---|---|---|---|---|
| Llama-3.1-8B | −0.0 (+3.9 → +3.9) | **−20.6 (+26.2 → +0.2)** | −0.3 (+14.5 → +14.1) | −2.4 (+8.4 → +5.7) | ±0.1 (+9.5…+10.2) |
| Mistral-7B-v0.3 | +0.1 (+3.9 → +3.9) | **−20.6 (+26.2 → +0.2)** | +0.1 (+14.7 → +14.7) | −2.4 (+8.5 → +5.9) | ±0.1 (+9.6…+10.1) |
| Phi-4 | −4.5 (+5.3 → +0.5) | **−8.5 (+17.7 → +7.6)** | −0.8 (+11.2 → +10.3) | +0.1 (+9.0 → +9.0) | ±0.1 (+8.6…+9.6) |
| Qwen3.8-27B | −3.4 (+4.8 → +1.3) | **−8.2 (+18.5 → +8.7)** | −1.2 (+9.5 → +8.2) | −0.6 (+8.6 → +8.0) | ±0.2 (+7.5…+9.3) |

- At T ≤ 64 the 1b set equals optimization 1's within ±0.3 %.
- The 128 × 64 tile fixes the T = 256 CTA-count problem on Llama and Mistral, where 8x64 goes from +26 % to +0.2 % over
  stock_wA. It halves the problem on Phi-4 and Qwen.
- **T = 512 and T ≥ 1024 keep +5.7 … +14.7 %.**
  - The kernel report attributes the large-T part to the 1x8 arrangement and the 16-arm dispatch.
  - Amendment 10's baseline measures that split.

**M2: prefill, CUDA graph.** The 8x64 path against FourOverSix (`auto_stock`), in %, optimization 1's run → 1b's run.
Each run is compared with its own FourOverSix processes:

| model | 1x128 | 1x256 | 1x512 | 1x1024 | 1x2048 | 1x8192 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | +1.1 → +1.1 | **+10.6 → +1.7** | +8.9 → +9.4 | +5.5 → +3.5 | +5.9 → +5.8 | +4.4 → +4.4 |
| Mistral-7B-v0.3 | +1.2 → +1.2 | **+11.3 → +1.7** | +9.4 → +8.4 | +6.9 → +5.3 | +6.2 → +6.3 | +4.7 → +4.7 |
| Phi-4 | +2.2 → +0.6 | +6.1 → +6.3 | +7.2 → **+8.7** | +6.4 → +6.3 | +6.2 → +6.3 | +5.0 → +5.1 |
| Qwen3.8-27B | +0.5 → −0.0 | +2.4 → +1.0 | +2.7 → +2.8 | +3.0 → +3.1 | +3.1 → +3.0 | +2.5 → +2.6 |

- **The T = 256 gain carries over to prefill** on Llama and Mistral: 1x256 goes from +10.6/+11.3 % to +1.7 % over
  FourOverSix.
- **Phi-4 1x512 is the open item, +1.5 points.**
  - At T = 512 the table picks the 128 × 64 build for Phi-4's qkv_proj (7680x5120). In isolation (M1) the 1b set is
    −0.8 % per forward there.
  - The diagnostic D1b (f54d468, `diag/d1b_graph_kernels_phi4.json`) ran both sets in one process. At 1x512 their GEMM
    and wall times agreed within ±2 % (1b's wall +0.3 %). So the difference between the separate-process runs is not
    the 128 × 64 GEMM itself.
  - D4 later found that a captured forward's idle time is bimodal per capture: about 0.15 or 2.0–2.4 ms. That is the
    size of this effect.
  - The cause is not isolated. It is carried into the 8x64 plan.

**M3: decode, CUDA graph** (Experiment D's harness and settings; Llama, Mistral, Phi-4).
- **Against the original n8k64_wB:** tokens per second rise by +35.4 % (Llama 1x512) … +6.1 % (16x2048), +37.9 … +6.4 %
  (Mistral) and +25.5 … +4.4 % (Phi-4). Optimization 1's narrow tiles deliver all of it; 1b does not touch these widths
  on Llama or Mistral.
- **Against FourOverSix on weights-on-A** (`auto_stock`), tokens per second: Llama −0.2 … −1.3 %, Mistral −0.2 … −1.4 %,
  Phi-4 +0.1 … −0.4 %.
- **Against optimization 1's M3** (a separate run; each compared with its own FourOverSix processes):
  - Llama and Mistral are identical within ±0.03 points (the same builds and widths).
  - Phi-4 differs only where the 1b table changed its decode widths, and by little:
    - batch 4, where gate_up_proj moves from width 16 to 32: −0.18 and −0.09 points;
    - batch 16, where down_proj goes 32 → 16 and qkv_proj 16 → 32: +0.11 and +0.08 points.
  - Given the ±0.03-point agreement on Llama and Mistral, these are probably real, but small.

## Status

1b is closed. Its builds and table are what the 8x64 baseline (amendment 10) measures as "the current best 8x64 path".
Whether `auto_wB` becomes the 8x64 default on kernel-opt is pending decision (b). The same-placement reference question
is pending decision (a). Both are addressed in the 8x64 plan.
