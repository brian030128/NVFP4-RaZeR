# Paper evaluation on the RTX PRO 6000 (SM120): BF16, NVFP4, FourOverSix and FlipQuant

The user's request of 2026-10-07 (relayed by the coordinator): the paper's SM120 evaluation for BF16, NVFP4, FourOverSix
and FlipQuant only, on the six paper models (Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2,
Phi-4, Qwen3.8-27B) at 8x64 / 16x64 / 256x64. FlipQuant = the 5-epoch release maps (256 windows, top-1000).
Protocol and amendments 1–11: `experiments/paper_eval/PROTOCOL.md` (hashes in `registration.json`).

Environments: n16k64 for the four non-hybrid models; n16k64-fast (n16k64 plus the hybrid layers' fast kernels,
amendments 3 and 5, `fast_env/`) for Nemotron-Nano-9B-v2 and Qwen3.8-27B. Kernels: build_V. The maps are as calibrated
(under the fallback kernels for the hybrid models).

| part | directory | status (2026-10-08) | result |
|---|---|---|---|
| tables from records | `cpu_tables/` | done | main PPL with loss recovered (FourOverSix 5.8 %, FlipQuant 78.9 / 78.3 / 64.2 % over 12 pairs), tile granularity, calibration cost |
| P: harness parity | `parity/` | done; accepted (amendment 2) | flipquant `evaluation.latency` vs the step-05 harness on Phi-4: FlipQuant 16x64 +1.02 % (bound 1 %), FQ/FO6 gap within 0.06 pt |
| F: hybrid PPL in n16k64-fast | `pplfast/` | done | 21 of 24 fast − fallback differences within 2 SE (all < 0.0011 nats); FlipQuant vs FourOverSix keeps sign and significance |
| A: prefill latency | `latency/` | done | 210 records, graph == eager; FlipQuant 16x64 +0.5–0.8 %, 8x64 +2.3–4.5 %, 256x64 +0.2–1.9 % vs FourOverSix (median over 8 shapes) |
| B: memory | `memory/` | done | quantized linears 0.2812–0.2813x BF16; whole model 0.31–0.41x; FlipQuant = NVFP4 = FourOverSix |
| C1: ownership | `verify/` | done | 18 / 18 artifacts exact, 0 format mismatches |
| C2: native vs simulated | `verify/` | done | 28 of 30 cells within 2 SE (Qwen3.8-27B excluded: its simulated path does not fit one GPU) |
| D: generative accuracy | `d_pilot/` | pilot done; full run ON HOLD (user) | Qwen3.8-27B's hard AIME problems run ≥ 30k tokens; full D 180–480 GPU-hours depending on the option |
| E: lm-eval | `e_estimate/` | estimate only; ON HOLD (user) | ~30 h (23–44 h), ~16 h without MMLU-Pro |

Notes for the text:
- Nemotron-Nano-9B-v2's 256x64 costs +1.9 % in prefill (16x64 +0.6 %); on the other models 256x64 ≈ 16x64.
- Qwen3.8-27B at 1x128: the stock weights-on-B FourOverSix kernel is +11.5 % vs FourOverSix (all 5 rounds), so
  "FlipQuant 8x64 vs FourOverSix (wB)" is −10 % at that shape; FlipQuant 8x64's own kernel is +0.4 % vs FourOverSix.
- F's NVFP4 rows use per-token NVFP4 scales (the main table's convention (c); amendment 11).
