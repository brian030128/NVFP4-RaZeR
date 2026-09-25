# Calibration speed-ups — Phase 1: fused activation quantization and single-pass epilogue

**Verdict: PASS.** It is judged under [PROTOCOL_PHASE1.md](PROTOCOL_PHASE1.md), registered
2026-09-25 13:07:22 UTC (sha256 bada43e6…), before any Phase 1 test or run.

Items 1 and 3 are bitwise-neutral:
- on every Linear shape of the four final models (unit tests);
- end-to-end on Llama-3.1-8B at both units (identical to the committed runs in every recorded
  value);
- on Phi-4's round-0 scores and first development evaluations.

The Qwen3.8-27B check could not run: batch 16/8 does not fit on the GPU in lean mode (below). With
both items on, Llama calibration runs **14 % faster** (256x64: 21.6 → 18.5 min; 8x64: 33.8 → 29.0 min).

The flags stay opt-in: `--fused-act-quant`, `--single-pass-epilogue`. The profile behind the
estimates is in [PHASE0.md](PHASE0.md).

## 1. Unit tests (`phase1_unit_tests.json`): PASS, bitwise

The shapes are all 16 distinct Linear shapes (N × K) of Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and
Qwen3.8-27B, at 8×512 and 16×512 tokens. They include Phi-4's fused `qkv_proj` (7680×5120) and
`gate_up_proj` (35840×5120), and Qwen3.8-27B's linear-attention `in_proj_qkv/z/a/b` (a and b have
48 rows) and `out_proj`.

The inputs are adversarial: outlier channels, four decades of channel scales, exact zeros, an
all-zero token row, and values that round to signed zero. Every pair is bitwise equal (int16, so
signed zeros count):
- `fourover6_rows` = `quantize_rows`;
- `fourover6(x, documents)` = `quant_per_document`;
- `torch.mul(D, gs, out=bf16)` = `(D.float() * gs).to(bf16)`.

## 2. End-to-end, Llama-3.1-8B: PASS, bitwise

DET-NATIVE-256x64 and DET-NATIVE-8x64 were rerun with both flags and Part B's lean settings (batch
16/8, deterministic, skip-CE). `compare_runs.py` checked them against the committed records and
Part B's lean runs.

| | DET-NATIVE-256x64 | DET-NATIVE-8x64 |
|---|---|---|
| rounds / tries identical (counts, sizes, predicted and measured changes, decisions) | 8 / 63 | 9 / 114 |
| per-document development CE/KL of every try | 63 × 192 equal | 114 × 192 equal |
| round-0 score mean/SE (and file hash) | 425,984 tiles equal | 13,631,488 tiles equal |
| initial (fake and native) and final per-document values | equal | equal |
| final map (and `map.pt` sha256) | equal | equal |
| final WikiText-2 (141) / C4 (256) window NLL | equal | equal |

At run time, the first 64 calls of each fused quantizer were also checked against its reference;
they passed.

| time and memory (lean) | 256x64 legacy → new | 8x64 legacy → new |
|---|---:|---:|
| optimization | 21.6 → **18.5 min (−14 %)** | 33.8 → **29.0 min (−14 %)** |
| scoring per pass | 60.2 → 48.4 s (−20 %) | 60.2 → 48.7 s (−19 %) |
| native development evaluation per try | 12.97 → 11.44 s (−12 %) | 13.02 → 11.43 s (−12 %) |
| initial fake development evaluation | 35.9 → 16.1 s (−55 %) | — |
| final fake WikiText-2/C4 evaluation | 198 → 89 s (−55 %) | 198 → 88 s (−56 %) |
| setup | 121 → 98 s | 119 → 98 s |
| peak GPU allocated / reserved | 48.1 / 54.9 → 48.1 / 54.3 GiB | 49.2 / 55.9 → 49.2 / 56.1 GiB |
| peak host RSS | 43.9 → 43.8 GiB | 44.1 → 44.1 GiB |

"Legacy" is Part B's lean run of the same configuration: same machine, flags off.

## 3. Architecture checks at batch 16/8 (lean, native decisions, `--stop-after-scoring`)

**Phi-4: PASS.** Legacy and new are bitwise equal:
- the fake and native initial per-document development CE and KL (192 documents);
- round-0 score mean/SE in all 160 matrices.

The scoring pass takes 108.6 s in legacy and 95.4 s new (−12 %).

**Qwen3.8-27B: skipped, it does not fit at batch 16/8 in lean mode.** The legacy run ran out of GPU
memory in its first fake development evaluation, in `per_sequence_losses`:
- **The failing allocation:** at batch 16 each FP32 loss tensor over the 248,320-token vocabulary
  is 16 × 511 × 248,320 × 4 B = 7.56 GiB, and the KL expression needs several.
- **At the failure:** 89.5 GiB in use, of which 61.9 GiB allocated and 26.9 GiB reserved but
  fragmented.
- **What passed before the failure:** the 25.5 GiB lean store (496 matrices, no fallbacks), and
  the native kernel's start-up checks on the Qwen3.5 architecture (start, mixed with 23,777,954
  E0M3 tiles, and restored maps: 0 mismatches).

As the protocol requires, the memory is recorded and no offloading was improvised.

A batch-1/1 run with the new flags fits. Its round-0 scoring pass takes 478 s, and its initial
development KL is 0.04619 (fake) / 0.04569 (native). Without a batched run, Qwen3.8-27B's
batched-vs-single per-document differences could not be recorded.

Two options for the user's decision, not taken here:
- compute the per-document losses in chunks of documents, which removes the vocabulary-sized
  batch-16 temporaries without changing any value;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the 27 GiB of fragmentation.

## Deviations

None.

## Reproduction

```
python repro_local/realquant/test_phase1_speedups.py results/speedups/phase1_unit_tests.json
/home/dev/n16k64_campaign/speedups/queue_phase1.sh          # copy in runs/; checks in runs/checks/
```

- **Run records:** in `runs/`, which holds the report.json of every run, the checks and
  `commands.log`.
- **Large files:** the per-run tensors (maps, `dev_values.pt`, `round0_scores.pt`) stay in
  `/home/dev/n16k64_campaign/speedups/phase1`.
