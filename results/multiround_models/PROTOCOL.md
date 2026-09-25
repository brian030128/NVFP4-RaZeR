# KL-only multi-round MixFP4 calibration on the N16K64 models (Part C) — protocol

Written 2026-09-24, before any Part C run and before Part C's data preparation. This part runs
only if Part B (`results/lean_memory/PROTOCOL.md`) passes. Hashes and the registration time are
recorded twice: `registration.json` for this protocol, and `registration_<model>.json` per model,
written before that model's first run. Deviations are appended in the last section.

## Models, in this order

1. Qwen3-4B (`qwen4b`)
2. Mistral-7B-v0.3 (`mistral7b`)
3. Phi-4 (`phi4`)
4. Qwen3.8-27B (`qwen27b`)

This is the N16K64 set; OLMo-2-13B is excluded. The pinned revisions are those of the N16K64
campaign: `1cfa9a72…`, `caa1feb0…`, `2db69c1c…`, `1d4bf0f2…`. Qwen3.8-27B loads through
`Qwen3_5ForConditionalGeneration` from transformers 5.16.1, as in CLAUDE.md.

The quantized matrices are every text `nn.Linear` except the output head: 252 for Qwen3-4B, 224
for Mistral, 160 for Phi-4 and 496 for Qwen3.8-27B. Recurrent, convolution, norm, vision and head
parameters stay BF16.

## Stages per model

Each stage is reported to the user when it completes.

### 1. Data (`prepare_model_data.py`, CPU)

The layout is that of Llama's local data, under `/home/dev/n16k64_campaign/multimodel/data/<model>`.

**Fit set: 128 windows (64 OpenWebMath + 64 CodeParrot) of 512 tokens.**
- An existing calibration record is reproduced with `math_code_data`, which asserts every token
  hash:
  - Qwen3-4B and Qwen3.8-27B: `results/math_code_adaptive/calibration_333779_qwen4b` and
    `calibration_333787_qwen27b`, which also give the per-matrix weight hashes;
  - Mistral: the N16K64 record `research/n16k64/campaigns/primary/calibration_manifests/mistral7b_seed0.json`.
- Phi-4 has no record, so its fit set is drawn with the rule that drew Llama's:
  - The rule is `run_pooled_scale.shared_data`'s math/code part at the pinned dataset files, run
    as `campaign.data.builder_seed0`.
  - The builder must first reproduce the Llama, Qwen3-4B and Mistral records exactly (document,
    offset, token hash). If it does not, Phi-4 is stopped and reported.

**Development set: 192 windows.** For Qwen3-4B, Mistral and Phi-4 they are drawn with
`run_fine_row_research.fresh_data`, the code that drew Llama's three sets:
- seed 20260919, file order;
- 32 math + 32 code windows per draw, skipping fit documents, fit windows, documents under 512
  tokens and excluded documents;
- three draws in sequence, each excluding the model's C4 evaluation documents and the documents
  of the earlier draws.

**Qwen3.8-27B development set.** main's three sets are reproduced and matched to the fresh.pt
sha256 values that `results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json` records:
- `fisher_subset_validate_v2_confirm`;
- `preserved_row_confirm`;
- `ce_target_combinations_confirm`.

The pilot drew six sets in sequence with the same code. The first excluded the published C4
documents, and each later set also excluded all earlier ones. If a hash cannot be reproduced,
Qwen3.8-27B is stopped and reported; no new documents are drawn without the user's decision.

Every document hash, offset and token hash is recorded, and every file is accepted by its hash.

### 2. Registration

`registration_<model>.json` records:
- this protocol's sha256;
- the code hashes;
- the data hashes;
- the kernel library hash;
- the time.

### 3. Batching check (before calibration)

Two runs of `run_multiround.py --memory-mode lean --unit 8x64 --objective kl --dev-backend native
--deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values`.
They differ only in batch size:

| run | batch sizes |
|---|---|
| A | `--eval-batch 16 --score-batch 8` (Llama's) |
| B | `--eval-batch 1 --score-batch 1` |

A and B are compared bitwise on:
- the fake initial development per-document CE and KL (192);
- the native initial development per-document CE and KL;
- the round-0 score mean and SE of every unit.

If all are identical, calibration uses 16/8. Otherwise it uses 1/1.

For Qwen3.8-27B, A uses `--eval-batch 2 --score-batch 2`. main already found its batched forward
is not identical, and a 16-document FP32 log-softmax over its 248k vocabulary costs about 8 GiB
per tensor. The check is recorded either way.

### 4. Pre-check (before calibration)

`--evaluate-map FourOverSix=fourover6` is run in lean mode, one process per backend (native,
fake), on WikiText-2 and C4. Both must hold:
- **NLL gap:** the per-window mean NLL difference, native − fake, is within ±0.01 nats (the bug
  threshold) on each corpus;
- **Native checks:**
  - on the architecture, the native checks pass bitwise: the start map, a random mixed map and
    the restored start map at run start;
  - the evaluated map shows 0 mismatches;
  - the first 64 activation calls match.

Otherwise the model is stopped and reported.

### 5. Calibration (8x64, then 256x64)

**Settings: identical to the Llama DET-NATIVE runs.**
- `run_multiround.py --objective kl --dev-backend native --skip-ce-backward --deterministic
  --memory-mode lean` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Strict-decrease acceptance (the default).
- Per-document tensor-wide FourOverSix activations for development and final evaluation;
  per-token rows for scoring.
- Batch sizes from stage 3, plus `--record-dev-values`.
- `--transformers-deviation` only where the calibration record names another transformers
  version (Qwen3-4B: 4.57.3).

**Budget cap.** `--budget-hours` is 12 for Qwen3-4B, Mistral and Phi-4, and 36 for Qwen3.8-27B,
whose batch-1 search is several times longer. The cap is meant never to bind; if it does, that is
reported as a result.

**Reported per unit:**
- rounds, tries and tiles;
- setup, total optimization, scoring per pass and development evaluation per try;
- peak GPU (allocated/reserved) and host memory.

**Memory stop rule.** If Qwen3.8-27B does not fit on the 96 GB GPU even in lean mode, the run
stops with the measured memory reported. No other offloading is improvised.

### 6. Evaluation (PPL only; no zero-shot)

WikiText-2 and C4 windows as in `run_multiround.py` (`run_baseline_protocol_audit.data`: 2048
tokens, one window per forward). The released windows are validated where a published record
exists (Qwen3.8-27B). Otherwise the windows are recorded by token hash, and all evaluations of a
model must have identical windows.

**Processes.**
- `--evaluate-map` in lean mode, `--unit 8x64`, one process per backend:
  - native (primary);
  - fake (secondary).
- Maps:
  - FourOverSix;
  - NVFP4: `quant_nvfp4` weights and per-window tensor-wide `quant_nvfp4` activations, the
    `nvfp4_tensor` baseline;
  - MixFP4 8x64;
  - MixFP4 256x64, whose map is expanded exactly to 8x64 tiles, with the element masks checked
    equal.
- BF16 runs in its own fake process, on the model as loaded.

**Statistics.**
- PPL.
- Per-window paired ΔNLL, mean ± 2 SE (SD with ddof = 1, over √n windows), against FourOverSix
  in the same process. BF16 is paired across processes on identical windows.

Nothing is selected or tuned on WikiText-2 or C4.

### 7. Report and commit

- **Report:** `results/multiround_models/<model>/` holds the report, the run records, the maps'
  hashes and the tables.
- **Commit:** after each model, on `repro/n16k64-rtx-pro-6000`, with the usual identity. Nothing
  is pushed. The uncommitted zero-shot work stays untouched and out of every commit, and the QAT
  study stays paused.

## Deviations (append-only)

1. **2026-09-25, added check (before any model's run): regression of the shared code on Llama.**
   - **Why:** Part C adds to three files that Part B verified (`run_multiround.py`,
     `native_dev.py`, `candidate_store.py`), which are now committed as 0587b60.
   - **The run:** one Llama-3.1-8B run with the Part C code, `--memory-mode lean --unit 256x64
     --dev-backend native --stop-after-scoring --dump-round0-scores --record-dev-values`, with
     Part B's flags.
   - **The criterion:** its fake and native initial per-document development CE and KL and its
     round-0 scores must be bitwise identical to Part B's lean DET-NATIVE-256x64 run.
   - **If it fails,** Part C stops and the failure is reported.
