# Qwen3.8-27B at batch 16/8 — Option A protocol: chunked loss and expandable segments

Written 2026-09-25, while Phase 2 runs and before any GPU run of this change. The hash and time are
in `registration_chunked.json`. Deviations are appended in the last section.

**Why.** Every model calibrates at batch 16/8 (user decision). Qwen3.8-27B ran out of GPU memory at
16/8 in lean mode (`REPORT_PHASE1.md`): each FP32 [batch, T, vocab] loss tensor of the batch-16
development evaluation is 7.56 GiB, and 26.9 GiB of the reserved memory was fragmented.

## What changes (a flag, off by default, plus an allocator setting)

- **`--chunked-loss`** (`chunked_loss.py`).
  - **Development evaluation:** CE and KL are computed over chunks of at least 2 documents (an odd
    one joins the last chunk). The per-sequence means are taken on the full [batch, T−1] per-token
    tensor, exactly as before.
  - **Scoring (KL objective, `--skip-ce-backward`):** each chunk's loss graph is differentiated
    against a bf16 leaf of that chunk's logits. The chunk loss is `.mean(-1).sum()`, so every
    token's upstream gradient is the same fl(1/(T−1)). The chunk gradients are copied into one
    logits-gradient buffer, and the model's backward starts from `logits.backward(buffer)`.
  - **What stays unchanged:** the lm_head GEMM, the operations, and every per-token value.
  - **Why chunks of 2:** with fewer than about 750 rows, PyTorch splits each vocabulary row-sum
    across two thread blocks. Two documents (1,022 rows) exceed that threshold, as a whole batch
    does.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** It only changes how the caching allocator
  maps memory. It is confirmed to have no numeric effect through criteria 2 and 3.

## Criteria (all bitwise; any mismatch stops and is reported)

1. **Unit tests** (`repro_local/realquant/test_chunked_loss.py`).
   - **Cases:** the four final models' vocabularies (32,768, 100,352, 128,256 and 248,320), with
     512-token documents at batches 1, 2, 3, 4, 8 and 16.
   - **Required equal:** the chunked per-document CE and KL and the whole-batch expression, and (at
     batches ≤ 8) the chunked scoring gradient into the logits and the autograd gradient of
     `kl.sum()`.
2. **Llama-3.1-8B end-to-end.** DET-NATIVE-256x64 with Phase 1's items on, `--chunked-loss` and
   `expandable_segments`: lean, deterministic, skip-CE, batch 16/8, `--record-dev-values
   --dump-round0-scores`.
   - **Against the committed record** (`compare_runs.py`): tries, the per-document values of every
     try, decisions, map, window NLL and round-0 scores. The per-document and round-0 references
     come from Part B's lean run.
3. **Phi-4.** Round-0 scores and the first development evaluations at batch 16/8, lean, 8x64,
   `--stop-after-scoring`, must be equal between two runs:
   - Phase 1's `phi4_new_16x8` (items on, whole-batch loss, default allocator);
   - items on + `--chunked-loss` + `expandable_segments`.
4. **Qwen3.8-27B** (lean, 8x64, items on, `expandable_segments` in both runs of a pair).
   - **(a)** At 4/4, where the whole-batch loss fits: round-0 scores and the first development
     evaluations must be equal, whole-batch vs chunked.
   - **(b)** At 16/8 with `--chunked-loss`: round-0 scoring and the first development evaluations
     must run to completion. Peak GPU memory (allocated/reserved), host memory, scoring time per
     pass and development-evaluation time are recorded.
     - **Not criteria:** the Phase 2 B1 kernel is not used here. Those times are those of a
       `--stop-after-scoring` run.
   - **(c) Documentation, not a criterion:** Qwen3.8-27B's per-document development differences
     between 16/8 (from (b)) and 1/1 (Phase 1's `qwen27b_new_1x1`, the same items).

**If Qwen3.8-27B still does not fit at 16/8,** the queue stops and the memory breakdown is reported.
Batch sizes are not changed and no other offloading is improvised. The user chooses between the
remaining options: B, a smaller batch for Qwen; or C, batch 1 for Qwen.

## Rules

- **Commit and push:** commit with the usual identity and a short report, separately from Phase
  2's commit. Try to push `repro/n16k64-rtx-pro-6000` once (never main, no force). If the push is
  blocked, report it and do not retry.
- **Out of scope:** Part C, QAT and zero-shot stay paused/untouched.

## Deviations (append-only)

1. **2026-09-25 16:25 UTC: criterion 4(a) cannot be evaluated.**
   - **What failed:** the whole-batch Qwen3.8-27B reference run at 4/4 (`qwen27b_whole_4x4`) ran
     out of GPU memory. Its development evaluations at batch 4 completed. The failure came in the
     scoring forward pass, inside a quantized Linear under autograd: 93.8 GiB allocated, of which
     the lean store is 25.5 GiB.
   - **Why chunking cannot fix it:** the activations kept for the backward pass fill the GPU at
     4 × 512 tokens. Chunking the loss removes only the vocabulary-sized loss tensors (1.9 GiB
     each at batch 4), so it cannot make scoring at batch 8 fit.
   - **Code change:** an out-of-memory hook was added to `run_multiround.py`. On a CUDA
     out-of-memory exit it records the per-phase memory peaks and the allocator summary in the
     report. It changes nothing else.
   - **What runs next:** the registered 16/8 run (criterion 4(b)), to record the memory breakdown
     that the stop rule asks for. The queue then stops there.
