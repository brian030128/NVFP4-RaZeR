# Qwen3.8-27B at batch 16/8 — Option A: chunked loss and expandable segments

**Verdict: Option A is bitwise-verified, but Qwen3.8-27B still does not fit at batch 16/8.** It is
judged under [PROTOCOL_CHUNKED.md](PROTOCOL_CHUNKED.md), registered 2026-09-25 15:49:12 UTC (sha256
405a7078…), before any GPU run. There is one deviation (below).

- **Option A is bitwise-neutral.** The evidence:
  - unit tests on all four vocabularies;
  - Llama-3.1-8B end to end (every recorded value equal to the committed run);
  - Phi-4 round-0 scores and development evaluations.
- **Option A cuts Llama's peak GPU memory by 7.4 GiB allocated and 12.5 GiB reserved.**
- **Qwen3.8-27B:**
  - Development evaluation at **batch 16 now fits** (peak 40.0 GiB fake, 53.1 GiB native).
  - **Scoring at batch 8 does not fit:** it runs out of memory in its first forward pass, at
    93.8 GiB allocated. That memory holds the activations kept for the backward pass, not the loss.
  - The whole-batch run at scoring batch 4 runs out of memory in the same place.
- **Stopped here, as registered.** Batch sizes were not changed and no offloading was improvised.
  The choice between B (a smaller batch for Qwen) and C (batch 1 for Qwen) is the user's.

## Criteria

1. **Unit tests** (`chunked_unit_tests.json`): **PASS.** All 24 cases (vocabularies 32,768,
   100,352, 128,256 and 248,320 × batches 1, 2, 3, 4, 8, 16) are bitwise equal to the whole-batch
   expression: per-document CE and KL, and at batches ≤ 8 the scoring gradient into the logits.
2. **Llama-3.1-8B, DET-NATIVE-256x64** (items 1 and 3, `--chunked-loss`, expandable segments):
   **PASS, bitwise against the committed record.**
   - 8 rounds and 63 tries; the 63 × 192 per-document values; decisions.
   - The map and round-0 scores (identical file hashes); all 141 + 256 window NLLs.

   This also shows that `expandable_segments` has no numeric effect.
3. **Phi-4, batch 16/8:** **PASS.** Chunked vs whole-batch (Phase 1's run) round-0 scores and the
   first fake and native development evaluations are bitwise equal.
4. **Qwen3.8-27B:**
   - **(a) Not evaluable.** The whole-batch reference at 4/4 ran out of memory in scoring (below).
   - **(b) Out of memory at 16/8 in scoring.** The breakdown is below.
   - **(c)** Documented from the development evaluations that did complete (below).

| Llama DET-NATIVE-256x64 (lean, batch 16/8, items 1+3) | whole-batch loss | chunked + expandable segments |
|---|---:|---:|
| optimization | 18.5 min | 19.2 min (+4 %) |
| scoring per pass / development evaluation per try | 48.4 / 11.44 s | 48.8 / 12.05 s |
| peak GPU allocated / reserved | 48.1 / 54.3 GiB | **40.7 / 41.8 GiB** |
| peak host RSS | 43.8 GiB | 42.1 GiB |

## Qwen3.8-27B memory breakdown (lean, 8x64, items 1+3, chunked, expandable segments, eval 16 / score 8)

| phase | peak allocated | peak reserved | host RSS |
|---|---:|---:|---:|
| model load (BF16 weights on the GPU) | 51.0 GiB | 51.0 GiB | 52.1 GiB |
| teacher precompute (320 sequences, 248k vocabulary, kept in host RAM) | 52.2 GiB | 52.4 GiB | 78.2 GiB |
| candidate packing (the BF16 matrices are freed as they are packed; the lean store is 25.5 GiB) | 56.2 GiB | 57.1 GiB | 78.8 GiB |
| fake development evaluation, batch 16 | 40.0 GiB | 54.7 GiB | 79.0 GiB |
| native development evaluation, batch 16 | 53.1 GiB | 54.7 GiB | 79.4 GiB |
| scoring, batch 8: **out of memory** in the first forward (in the activation quantizer's call) | 93.8 GiB | 94.2 GiB | 78.9 GiB |

**What fills the GPU in scoring.**
- **Resident during scoring:** the lean store (25.5 GiB) plus the unquantized parameters
  (embeddings, lm_head, norms, convolutions and vision tower, about 5 GiB).
- **The rest:** about 62 GiB is taken by activations stored for the backward pass before the first
  forward pass even finishes. At batch 4 the whole-batch run also ran out of memory in the forward,
  at the same 93.8 GiB, so one 512-token sequence needs more than about 15 GiB of stored activations.
- **Not measured:** which modules store them. The Qwen3.5 linear-attention layers run transformers'
  pure-PyTorch fallback here, because flash-linear-attention and causal-conv1d are not installed.
  That fallback keeps FP32 intermediates for the backward pass; it is a likely large consumer.
- **Batch 1/1 fits** (Phase 1: round-0 scoring completed in 478 s).
- **Batch 2 was not run.** By the estimate above it would need more than about 62 GiB, and whether
  it fits is not known.

## Documentation: Qwen3.8-27B per-document development values, batch 16 vs batch 1

From this run's development evaluations and Phase 1's batch-1/1 run, with the same items. None of
the 192 per-document values is identical:

| evaluation | max \|Δ\| per document | mean \|Δ\| | mean, batch 16 | mean, batch 1 |
|---|---:|---:|---:|---:|
| fake, KL | 0.0279 | 0.0044 | 0.046137 | 0.046193 |
| fake, CE | 0.0577 | 0.0116 | 1.372786 | 1.373172 |
| native, KL | 0.0286 | 0.0041 | 0.046471 | 0.045689 |
| native, CE | 0.0420 | 0.0094 | 1.375144 | 1.374438 |

## Deviation

1. **The 4/4 whole-batch reference ran out of memory in scoring**, so criterion 4(a) could not be
   evaluated.
   - An out-of-memory hook was added to `run_multiround.py`: on a CUDA out-of-memory exit it records
     the per-phase memory peaks and the allocator summary.
   - The registered 16/8 run then produced the breakdown above.
   - The Llama and Phi-4 runs ran the registered code. Every run records its source hashes.

## Flag status

`--chunked-loss` stays opt-in; it is verified bitwise. It saves 7–12 GiB on Llama at about 4 % more
time. Making it the default is left to the user.
