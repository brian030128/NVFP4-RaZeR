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


## The remaining SM120 tables (parts G-L, 2026-10-08/09; flipquant paper-sm120-runs 46dc5a7)

The co-author's branches map-ablation, ptq-combo, IF4, MIXFP4 and FOCUS merged onto main 120173a and adapted to the
final settings (release maps; the release 256 x 512 fit set via `calibration_release`, token sha256 checked;
n16k64 / n16k64-fast; build_V, `--kernel-set auto`). Protocol: `experiments/paper_eval/PROTOCOL.md` parts G-L,
amendments 12-13.

| part | directory | result |
|---|---|---|
| G: tab:ablation | `ablation/` | FlipQuant beats Random and Activation-weighted in all 12 cells and One-shot in 10 of 12 (the two exceptions: 256x64 WikiText-2, not significant); One-shot is far worse than FourOverSix at 8x64 / 16x64 on Nemotron (verified: not a sign error) |
| H: tab:ptq | `ptq/` | GPTQ (BF16 propagation, the user's decision; check H1: codes identical under NVFP4 / FourOverSix activations) helps NVFP4, but GPTQ + FlipQuant ≈ GPTQ + FourOverSix; Hadamard (block 16, not fused) hurts every format, FlipQuant stays best under it |
| I: IF4, MixFP4 (Zou) | `mainrows/` | all 6 models; the 4 non-hybrid models bit-identical to main-ppl (per-window NLL and installed weights) |
| J: GPTQ‡ | `final/` | NVFP4 GPTQ codes with FourOverSix activations, 6 models: 4.6 % loss recovered |
| K: FOCUS | `final/` | 5 models (Qwen3.8-27B TBD), 8 steps on the release set, deployed natively: 83.3 % over its 10 pairs vs FlipQuant 82.1 / 82.4 / 67.4 % on the same pairs |
| L: final main-ppl | `final/` | loss recovered (12 pairs): FourOverSix 6.7 %, IF4 13.4 %, MixFP4 (Zou) 9.4 %, GPTQ‡ 4.6 %, FlipQuant 78.9 / 78.2 / 64.7 %; FlipQuant vs NVFP4 significant in 36 / 36 cells, 0 daggers; \|8x64 − 16x64\| max 0.0064; 256x64 keeps 81.1 % (62.7-84.7 % per model) |

## FOCUS on Hugging Face and tab:ptq's GSM8K column (2026-10-09)

| part | directory | status | result |
|---|---|---|---|
| FOCUS HF upload | `focus_hf/` | done | the 5 FOCUS states (part K) as PRIVATE repos `edgeai-lab/<Base>-FOCUS-NVFP4`: `focus.pt` as is (sha256 source = uploaded; 0 scan hits, no metadata change), model card, `ppl_summary.json`, license files; every remote file's sha256 equal to the staged one; Hub YAML validation passed |
| M0: GSM8K pilot | `ptq_gsm8k/pilot/` | done (not a result) | greedy outputs depend on the batch size on both models (batch 16 vs 1: 10 of 64 / 1 of 32 completions identical; not a padding leak); batch 1 would take ~162 h; the user chose batch 64 (amendment 14) |
| M: tab:ptq GSM8K (flipquant harness) | `ptq_gsm8k/` | superseded (amendment 16) at 16 / 18 | 0-shot `\boxed{}`, greedy, thinking off, batch 64; records archived, not in the table |
| N: tab:ptq GSM8K (lm-eval `gsm8k_llama`) | `ptq_gsm8k_lmeval/` | done (2026-10-09 17:06–23:33 UTC) | 8-shot CoT, chat template, greedy, 1024 tokens, thinking off, batch 64; strict-match fills table_ptq.tex; checks N2–N5 18/18, N1 17/18 (one stray `</think>` in an answer, score unaffected) |
