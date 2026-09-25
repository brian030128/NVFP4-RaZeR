# Lean memory mode — Part B protocol: bitwise verification against the committed Llama-3.1-8B runs

Written 2026-09-24, before any Part B run. Hashes and the registration time are in
`registration.json`. Deviations are appended in the last section, never edited in place.

## What is tested

`run_multiround.py --memory-mode lean` (Part A). The default stays `legacy`, today's code path,
until this protocol passes.

- **One candidate store.** Both candidates (FourOverSix E2M1 base, E0M3 α=1 alternative) are kept
  only in the native packed format, `repro_local/realquant/candidate_store.py`: codes, scale bytes
  and the FP32 global scale. The fake path decodes base and alternative from it with a Triton
  kernel. `quantize/packed_candidates` storage is no longer kept.
- **Native weights exist only during an evaluation.** `NativeDev(store=...)` builds every module's
  packed weight for the map at the start of each development (or final) evaluation. `remove()`
  frees them.
- **No resident BF16 weight for the 224 quantized matrices.** Each matrix's BF16 weight is freed
  once it is packed. `candidate_store.lean_forward` decodes the current map's weight on every
  forward. Under autograd, saved-tensor hooks replace the saved weight view by the map snapshot
  and decode it again for the backward. Embeddings, norms and `lm_head` are unchanged.

The Part A unit checks passed before this protocol was written (`unit_checks.json`, from
`repro_local/realquant/test_lean_memory.py`):
- For all 224 matrices, the store's decode equals `decode_base` / `decode_alt` bitwise, signed
  zeros included.
- The lean decode equals legacy `apply()`'s weight at 256x64 and at 8x64, for:
  - the start map;
  - a random mixed map;
  - the restored start map;
  - the four committed final maps.
- A whole-model forward and backward on a mixed map gives bitwise equal logits and input
  gradient.

## Runs (8, sequential, on the one RTX PRO 6000; nothing else runs on the GPU meanwhile)

All runs use the data and flags of the committed runs:

- `--objective kl --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `--data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation`.

They add `--record-dev-values --dump-round0-scores`. Both only record values and change no
computation.

| configuration | committed run | reruns |
|---|---|---|
| DET-FAKE-256x64 | `results/native_decision/runs/det_fake` (bb997ea) | `--unit 256x64 --dev-backend fake`, legacy, then lean |
| DET-NATIVE-256x64 | `results/native_decision/runs/det_native` (bb997ea) | `--unit 256x64 --dev-backend native`, legacy, then lean |
| DET-FAKE-8x64 | `results/native_decision_8x64/runs/det_fake_8x64` (9bd01c4) | `--unit 8x64 --dev-backend fake`, legacy, then lean |
| DET-NATIVE-8x64 | `results/native_decision_8x64/runs/det_native_8x64` (9bd01c4) | `--unit 8x64 --dev-backend native`, legacy, then lean |

The committed final maps are the `map.pt` files whose sha256 the committed records name. They
are under `/home/dev/n16k64_campaign/native_decision{,_8x64}/`, and the comparison script checks
their hash first.

**Why the legacy path is rerun.** The committed records hold each try's mean development change,
not its per-document values, and they hold no round-0 scores. The criteria need both. The legacy
reruns (`--memory-mode legacy`, recording on) provide these references. Each legacy rerun must
itself reproduce its committed record on every recorded field, exactly like the lean run.

## PASS criteria (all must hold, for all four configurations)

"Bitwise" means equal IEEE-754 bit patterns, so −0.0 vs +0.0 is a mismatch. The comparison is
`compare_runs.py`, run after every run.

1. **Each rerun against the committed record** (legacy and lean):
   - the per-round candidate count;
   - every try's step size, predicted CE/KL change and measured development CE/KL change;
   - every acceptance decision: the accepted size, and the tile count and development CE/KL after
     each round;
   - the initial per-document development CE and KL, plus the fake initial values for native
     runs;
   - the final per-document development CE and KL, the stop reason and the final tile count;
   - the final map, every module's tensor;
   - the final per-window WikiText-2 (141) and C4 (256) NLL and PPL.
2. **Lean against the legacy rerun:**
   - every try's per-document development CE and KL (192 documents);
   - the round-0 per-unit CE and KL score mean and SE of every module (the CE scores are zero
     under `--skip-ce-backward`).

**STOP rule.** Any mismatch stops the queue at the failing configuration. Nothing further runs,
Part C does not start, and the mismatch is reported as measured.

## Reported (not criteria), legacy vs lean for each configuration

- **Time:** scoring per pass, development evaluation per try, native build time per evaluation
  (lean, native runs), setup, total optimization and evaluation.
- **Memory:** peak GPU allocated and reserved, overall and per phase, and peak host RSS.
- **Accounting:** round 0's score dump (`--dump-round0-scores`) runs inside the scoring phase in
  both modes.

## Rules

- **Lean as default:** it may become the default only after this protocol passes, in a separate
  change.
- **Out of scope:** the zero-shot work (Task 2, on hold) stays untouched and uncommitted, and the
  QAT study stays paused.
- **No tuning:** nothing is tuned on WikiText-2 or C4.
- **Commit:** after Part A+B, on `repro/n16k64-rtx-pro-6000`, with the usual identity. Nothing is
  pushed.

## Deviations (append-only)

1. **2026-09-24 19:19 UTC: a queue-script bug, not a measurement.**
   - **What went wrong:** in `queue_b.sh`, the shell helpers `run` and `check` assigned the shared
     variable `NAME`. The check after the first run was therefore called on a nonexistent directory
     (`legacy_legacy_det_fake_256x64`). It failed with FileNotFoundError, and the queue stopped
     ("STOP mismatch").
   - **What was affected:** nothing had been compared. The run itself (legacy rerun of
     DET-FAKE-256x64) finished normally, rc=0.
   - **Recovery:**
     - The check was run by hand on the correct directory and passed: every recorded field
       equal, including the map file and all 141 + 256 window NLLs.
     - The queue resumed as `queue_b2.sh` from the next run (lean DET-FAKE-256x64), with
       function-local variables.
   - **Unchanged:** the runs, flags, criteria and code.
