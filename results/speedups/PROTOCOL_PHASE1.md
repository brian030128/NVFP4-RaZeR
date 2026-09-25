# Calibration speed-ups — Phase 1 protocol: items 1 and 3 (bitwise-neutral)

Written 2026-09-25 after the Phase 0 profile (`PHASE0.md`) and before any Phase 1 test or run. The
hash and time are in `registration_phase1.json`. Deviations are appended in the last section.

**Context.** The final model set is Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B. All models
calibrate with development-evaluation batch 16 and scoring batch 8. Part C and the QAT study stay
paused.

## What changes (flags, both off by default)

- **`--fused-act-quant`** (item 1). Two quantizers are replaced by Triton kernels:
  - **Scoring:** the per-token FourOverSix quantizer (`quantize_rows`) becomes
    `quantize/fused_fourover6.fourover6_rows`. It is used in the unchanged straight-through
    expression `q + (x − x.detach())`, so the forward bits (including the −0 → +0 of the addition)
    and the autograd graph are the legacy ones.
  - **Fake evaluator:** the per-document quantizer (`quant_per_document`) becomes
    `fourover6(x, documents)`.

  At run time, the first 64 calls of each are checked bitwise against the legacy quantizers, and
  the fake evaluator's also against `quant_nvfp4_4over6`. The native evaluator already quantizes
  its activations with a fused kernel.
- **`--single-pass-epilogue`** (item 3). The native evaluator writes `torch.mul(D, gs, out=bf16)`:
  one FP32 multiply, rounded once to bf16, exactly as the two-pass `(D.float() * gs).to(bf16)`.

## PASS criteria (all must hold; "bitwise" means equal bit patterns, signed zeros included)

1. **Unit tests** (`repro_local/realquant/test_phase1_speedups.py`). The shapes are every distinct
   Linear shape (N, K) of the four final models, at 8×512 and 16×512 tokens. They include Phi-4's
   fused qkv/gate_up and Qwen3.8-27B's linear-attention projections. Three pairs must be bitwise
   equal:
   - `fourover6_rows` vs `quantize_rows`;
   - `fourover6` (512 tokens per document) vs `quant_per_document`;
   - the single-pass vs the two-pass epilogue.

   The inputs are synthetic and adversarial: outlier channels, four decades of channel scales,
   exact zeros, an all-zero row, and values that round to zero.
2. **End-to-end, Llama-3.1-8B.** DET-NATIVE-256x64 and DET-NATIVE-8x64 are rerun with
   `--fused-act-quant --single-pass-epilogue`. The runs are otherwise identical to Part B's lean
   runs: `--memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16
   --score-batch 8 --record-dev-values --dump-round0-scores`, with the same data.
   - **Against the committed records** (`results/native_decision{,_8x64}/runs/det_native*`), bitwise:
     - per-round candidate counts, every try's step size and predicted/measured development change;
     - decisions, the initial (fake and native) and final per-document development CE/KL;
     - final maps and final per-window WikiText-2/C4 NLL.
   - **Against Part B's lean runs** (which record them), bitwise: every try's per-document
     development CE/KL, and the round-0 score mean/SE.
   - The comparison is `results/lean_memory/compare_runs.py`.
3. **Architecture checks, Phi-4 and Qwen3.8-27B, at batch 16/8.** Two runs per model, legacy (both
   flags off) and new (both on), each `--memory-mode lean --unit 8x64 --dev-backend native
   --deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values`.
   - **Required bitwise equal:** round-0 score mean/SE of every tile, and the per-document
     development CE/KL of the first fake and first native development evaluation.
   - **Qwen3.8-27B memory rule:** if it does not fit on the GPU at batch 16/8 in lean mode, the
     measured memory is recorded, its check is skipped, and this is reported. No offloading is
     improvised.
   - **Qwen3.8-27B documentation:** one extra run at batch 1/1 records its batched-vs-single
     per-document development differences. This is documentation, not a criterion.
   - **Mistral** is not needed: its Linear shapes and batch settings equal Llama's.

**STOP rule.** Any bitwise mismatch stops Phase 1. It is reported, and the flags stay off.

## Reported (not criteria)

- **Time:** scoring per pass, development evaluation per try, native build, optimization and
  evaluation, against Part B's lean runs (legacy) and within the architecture-check pairs.
- **Memory:** peak GPU allocated/reserved and host RSS, on the same comparisons.

## Rules

- **Commit** after Phase 1 with its report, as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw>.
- **Push:** then push `repro/n16k64-rtx-pro-6000` to origin. Never push to main, never force. Before
  the first push, check `git status`, confirm that no large binary or data file is committed, and
  that the zero-shot WIP stays out.
- **Out of scope:** Part C, QAT and zero-shot stay paused/untouched.

## Deviations (append-only)

(none yet)
