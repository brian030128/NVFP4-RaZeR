# TM-OPT: the trained tile map with the MR-OPT optimizations — verification protocol (Llama-3.1-8B)

Written 2026-09-26 on branch `tm-opt`, before any TM-OPT test or run. The hash and registration time are
in `registration.json`. Deviations are appended in the last section, never edited in place.

**Context (user decisions, relayed by nvfp4-razer-c9).**
- **Focus:** the trained-map method of `main` (`run_train_map.py`; `MIXFP4_REPORT.md` §7;
  `results/mixfp4_potential/train_map/REPORT.md`), on the new branch `tm-opt` (`main` c078210 merged).
- **The method, unchanged:**
  - one logit θ per tile;
  - STE (hard map, dm/dθ = 1) or sigmoid (m = σ(θ/τ));
  - Adam on the KL loss (ε 1e-12); batch 8; 20 epochs of 16 steps;
  - no backtracking; the development set is a monitor only.
- **Scope:** this protocol covers only the port and its verification, on Llama-3.1-8B. The
  multi-model TM-OPT vs MR-OPT comparison waits for the user's approval.

## TM-OPT: what changes

Each optimization sits behind a flag of `run_train_map.py` whose default is the legacy behaviour.
`--tm-opt` turns them all on; `--no-<flag>` turns one off.

| flag | what it does | numerics vs legacy |
|---|---|---|
| `--memory-mode lean` | One native-format candidate store. Every quantized Linear decodes its weight on each call, in the forward and again for the backward (saved-tensor hooks). No BF16 copy of the 224 matrices stays resident. | STE: the store's decode of the hard map, bitwise `apply(n, True)`. Sigmoid: `b + m·(a − b)` with the legacy arithmetic and dtypes. Bitwise. |
| `--fused-act-quant` | Training: `fourover6_rows`. Monitor and final evaluation: `fourover6` per document. | Bit-exact to `quantize_rows` / `quant_per_document` (the first 64 calls of each are checked). Bitwise. |
| `--tile-grad-kernel` (B1) | The batch sum of G ⊙ (A − B) per tile, from the packed store (`tile_score.tile_sums`, the Phase 2 kernel with the batch as one sequence and sign +1). No SE, no flip sign. | FP32; only the summation order differs. Not bitwise. |
| `--chunked-loss` | The step's KL gradient into the logits two documents at a time (`chunked_loss.train_kl_gradient`); the monitor's per-document CE/KL the same way. | Bitwise (whole-batch expressions). |
| `--deterministic` | `torch.use_deterministic_algorithms(True)`. | — |
| `--dev-backend native`, `--eval-backend native` | Monitor development evaluations and the final WikiText-2/C4 evaluation on the native b8x64 kernel (`NativeDev`, from the lean store), with `--single-pass-epilogue`. The fake evaluation stays available. | A different evaluator. It does not affect the map: the monitor never changes it. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Set by the caller; recorded. | Verified neutral earlier. |

**Applied to every run, legacy included (local-run adaptations, not optimizations):**
- `--data-root`: the local copy of the Llama data, as for all local Llama runs.
- `--transformers-deviation`: the calibration record is from transformers 4.57.3; 5.16.1 is
  installed. This is recorded.
- No Slurm.
- A phase monitor for memory and timing. It synchronizes the GPU only at phase boundaries.

## Verification (Llama-3.1-8B, 1× RTX PRO 6000, `CUBLAS_WORKSPACE_CONFIG=:4096:8`)

Order: 1 (unit tests) → group 1 (bitwise) → group 2 (B1, end to end). Any failure stops the queue
before the next group; it is then reported.

### 1. Unit tests (`repro_local/realquant/test_tm_opt.py`)

- **(a) Chunked training loss vs the legacy step:** `train_kl_gradient` against the legacy
  `(kl.sum() / batch).backward()`.
  - **Cases:** the vocabularies of Llama, Mistral and Phi-4 (128,256 / 32,768 / 100,352); batches
    1–8; 512-token documents.
  - **Criterion:** the logits gradient and the per-sequence KL are bitwise equal.
- **(b) B1 tile sums vs the legacy hook** `reduce((dyᵀx) ⊙ (A − B))`.
  - **Cases:** one real matrix of every distinct layer-0 shape of Llama-3.1-8B, Mistral-7B-v0.3 and
    Phi-4, plus a random matrix of each shape; at 8x64, 16x64 and 256x64; four batches of
    8 × 512 tokens each.
  - **Criterion:** normwise relative error (max|Δ| / max|ref|) ≤ 1e-6 in every case.
  - **Also reported:** max absolute and elementwise relative error; time per batch.

### Group 1: bitwise, B1 off

- **Pairs:** in each pair, the legacy path (`--deterministic`) against TM-OPT without B1
  (`--tm-opt --no-tile-grad-kernel --dev-backend fake --eval-backend fake`: lean, fused, chunked,
  deterministic). Both use fake monitor and final evaluation.
- **Common settings:** `--eval-every 1 --record-theta-hashes`, batch 8.
- **(a) STE 8x64, 3 epochs:** lr 0.02, `--init-logit -0.2`, so that tiles flip within 3 epochs and
  the hard-map path is exercised. With main's init of −1, nothing flips in 3 epochs.
- **(b) Sigmoid 8x64, 2 epochs:** lr 0.05, init −3 (main's sigmoid settings); τ anneals over the 2
  epochs.
- **Criterion, all bitwise:**
  - θ after every optimizer step (sha256 of every logit);
  - per epoch: train KL, E0M3 count, hard flips, τ;
  - the monitor's per-document development CE and KL: initial, every epoch and final;
  - the per-epoch maps, the final map and the θ file;
  - the final WikiText-2 and C4 window NLLs.

### Group 2: B1

- **(a) The unit tests of 1(b).**
- **(b) End to end.** Two full runs, STE 8x64 with main's settings (20 epochs, lr 0.02, init −1,
  batch 8, monitor every 2 epochs): TM-OPT (B1 on) vs TM-OPT `--no-tile-grad-kernel`.
  - **Evaluation:** both final maps are evaluated with this branch's evaluator
    (`run_multiround.py --evaluate-map`, convention (a), 2,048-token windows). The native process
    is primary and also includes FourOverSix; the fake process is secondary.
  - **Reported:** the E0M3 tiles of each map and their difference; the map overlap; flips per
    epoch; the max |Δθ|; paired per-window ΔNLL (B1 map minus no-B1 map, native) ± 2 SE.
  - **Registered criterion:** mean − 2 SE ≤ 0 on WikiText-2 and on C4, i.e. within ±2 SE or
    better.
- **(c) Time and memory, per epoch:**
  - legacy (group 1 STE run; the θ-hashing time is recorded and subtracted);
  - TM-OPT without B1;
  - TM-OPT.

  Reported for each: training seconds per epoch, selection time, and peak GPU (allocated/reserved,
  and in training) and host memory.
- **(d) Reference only, not a criterion** (a different GPU and code path): main's H200 results,
  fake evaluation.
  - **STE 8x64:** 6.7847 / 9.6754; comparable to this run's fake evaluation of the TM-OPT map.
  - **STE 256x64:** 6.8055 / 9.7235; not run here.

## Rules

- **Commit and push:** commit on `tm-opt` as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw>, then
  push `tm-opt` (never main, no force). If the push is blocked, report it and do not retry.
- **After the report:** stop. Nothing is selected or tuned on WikiText-2, C4 or zero-shot.
- **Before the verification (not a criterion):** a development smoke run may catch crashes; it is
  discarded. Any code change after registration is recorded as a deviation with the new hashes.
- **Scope:** the Qwen3.8-27B gate, Part C, QAT and zero-shot stay closed/paused. The zero-shot
  WIP is stashed on `repro/n16k64-rtx-pro-6000` (and backed up outside the repository).

## Deviations (append-only)

1. **2026-09-26 07:07 UTC: unit test 1(b) failed; the queue stopped before group 1, as registered.**
   - **1(a) passed:** the chunked training loss is bitwise equal to the legacy step in all 24 cases
     (three vocabularies, batches 1–8).
   - **1(b) failed:** B1's tile sums exceed the normwise tolerance of 1e-6 in 2 of 72 cases:
     1.41e-6 (Mistral `gate_proj`, real, 8x64) and 1.39e-6 (Llama `k_proj`, real, 16x64). The
     others reach up to 9.6e-7.
   - **Also:** B1 is slower than the legacy hook here (589 ms vs 463 ms, summed over one batch of
     every case).
   - **Diagnostic afterwards, not a criterion** (`diagnose_b1.py`, `runs/diagnose_b1.json`): both
     paths were compared with an FP64 reference, on the three models' layer-0 shapes at 8x64 and
     16x64, 4 batches each.
     - **Error vs FP64:** the legacy hook is up to 1.09e-6 off and B1 up to 1.32e-6. B1 is the
       closer of the two in only 5 of 24 cases; for most shapes its error is about twice the
       legacy hook's.
     - **So the 1e-6 tolerance is at noise level here.** It was set in Phase 2 for 512-token
       per-sequence sums; these are 4,096-token batch sums, where the legacy FP32 GEMM itself is
       ~1e-6 from the exact value.
     - **Speed:** B1 was timed in 7 launch configurations on the five largest shapes. None beats
       the legacy single FP32 GEMM: at best 76 ms vs 56 ms. B1's MR-OPT gain came from replacing
       per-sequence GEMMs and FP64 statistics; the training step has one GEMM per module anyway.
   - **Nothing else has run.** Groups 1 and 2 wait for the user's decision on B1.
