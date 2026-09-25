# Lean memory mode for multi-round calibration (Parts A and B)

**Verdict: PASS.** It is judged under the pre-registered bitwise criteria ([PROTOCOL.md](PROTOCOL.md),
registered 18:42:49 UTC with sha256 ba6723b0…, before any Part B run).

`run_multiround.py --memory-mode lean` reproduces all four committed Llama-3.1-8B runs bit for
bit:
- every candidate count, step size, per-document development CE/KL, acceptance decision and final
  map;
- every final WikiText-2/C4 window NLL;
- the round-0 tile scores.

It does this with **13–24 GiB less peak GPU memory** (−21 % to −33 %), and it is not slower.

| configuration | peak GPU allocated, legacy → lean | optimization, legacy → lean |
|---|---:|---:|
| DET-FAKE-256x64 | 60.9 → **47.9 GiB** | 31.3 → 30.1 min |
| DET-NATIVE-256x64 | 72.1 → **48.1 GiB** | 23.6 → 21.6 min |
| DET-FAKE-8x64 | 62.0 → **49.1 GiB** | 86.8 → 83.7 min |
| DET-NATIVE-8x64 | 73.3 → **49.2 GiB** | 36.5 → 33.8 min |

The default stays `--memory-mode legacy`. Lean is opt-in, and Part C uses it explicitly.

## Part A: what changed (`--memory-mode lean`; legacy is untouched and still the default)

1. **One candidate store** (`repro_local/realquant/candidate_store.py`).
   - **What is stored:** both candidates (FourOverSix E2M1 and E0M3 α=1), only in the native packed
     format. That is 4-bit codes and UE4M3 scale bytes per candidate, plus one FP32 global
     scale: 7.31 GiB for Llama-3.1-8B.
   - **How the fake path decodes:** a Triton kernel, computing rq.decode's
     `bf16((codebook[nibble] * scale) * gs)`.
   - **What is gone:** the separate `quantize/packed_candidates` store (also 7.31 GiB), which
     legacy keeps next to the native one in native runs.
2. **No resident native weight outside an evaluation.**
   - `NativeDev(store=...)` builds every module's packed weight for the current map at the start
     of each development or final evaluation, and `remove()` frees them.
   - The full rebuild costs 0.15 s per evaluation. Legacy's incremental rebuild costs 0.11 s.
3. **No resident BF16 weight for the 224 quantized matrices.**
   - Each matrix's BF16 weight is freed once it is packed.
   - `candidate_store.lean_forward` decodes the current map's weight on every forward.
   - Under autograd, saved-tensor hooks replace the saved transposed weight view by the map
     snapshot and the view's geometry. They decode the weight again when that layer's backward
     runs, so the backward sees a bitwise identical tensor in the same view.
   - Embeddings, norms and `lm_head` are unchanged.

### Part A unit checks (`unit_checks.json`, `repro_local/realquant/test_lean_memory.py`): PASS

- **Store decode:** for all 224 matrices, the store's Triton decode of both candidates equals
  `decode_base` / `decode_alt` of the legacy fake store bitwise as int16. Signed zeros are
  included: 292,345,147 negative zeros in the E2M1 candidate, none in E0M3 by construction.
- **Decoded weight vs `apply()`:** the lean per-layer decoded weight equals the weight legacy
  `apply()` installs, bitwise, at 256x64 and at 8x64. This holds for the start map, a random
  mixed map (212,624 / 6,815,542 E0M3 tiles), the restored start map and the four committed final
  maps. The PyTorch reference decode (`select` + `decode_packed`) agrees too.
- **Autograd:** a whole-model forward and backward (two development documents, scoring's
  straight-through activation hooks, a mixed 8x64 map) gives bitwise equal logits and input
  gradient with resident weights (`nn.Linear`) and with `lean_forward`.

## Part B: bitwise verification against the committed runs

**Runs.** Eight runs, sequential on the one RTX PRO 6000. They use the committed runs' data and
flags (`--objective kl --skip-ce-backward --eval-batch 16 --score-batch 8 --deterministic`), plus
`--record-dev-values --dump-round0-scores`. For each configuration there is a legacy rerun, then
a lean run.

**Why legacy was rerun.** The committed records hold per-try mean development changes, not
per-document values, and no round-0 scores. The legacy reruns supply these references. Each of
them first had to reproduce its committed record on every recorded field.

| configuration | committed | legacy rerun vs committed | lean vs committed | lean vs legacy rerun |
|---|---|---|---|---|
| DET-FAKE-256x64 | bb997ea | PASS (5 rounds, 44 tries) | PASS | PASS: 44 × 192 values, 425,984 round-0 scores |
| DET-NATIVE-256x64 | bb997ea | PASS (8 rounds, 63 tries) | PASS | PASS: 63 × 192 values, 425,984 round-0 scores |
| DET-FAKE-8x64 | 9bd01c4 | PASS (10 rounds, 129 tries) | PASS | PASS: 129 × 192 values, 13,631,488 round-0 scores |
| DET-NATIVE-8x64 | 9bd01c4 | PASS (9 rounds, 114 tries) | PASS | PASS: 114 × 192 values, 13,631,488 round-0 scores |

**What was compared.** Every comparison is exact on IEEE-754 bit patterns (`compare_runs.py`):
- per-round candidate counts;
- every try's step size and predicted and measured development change;
- every acceptance decision, and the per-round tiles and development loss;
- initial (and fake initial) and final per-document development CE/KL;
- the stop reason;
- the final map (every module; the `map.pt` files also have identical sha256);
- all 141 WikiText-2 and 256 C4 window NLLs and both PPLs.

Round-0 score files also have identical sha256 between legacy and lean.

### Time and memory, legacy vs lean

| configuration | run | checks | rounds / tries | tiles | scoring per pass | dev evaluation per try | native build per evaluation | setup | optimization | peak GPU allocated / reserved | peak host RSS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET-FAKE-256x64 | legacy | PASS | 5 / 44 | 7,617 | 69.7 s | 34.7 s | — | 97 s | 31.3 min | 60.9 / 63.4 GiB | 43.7 GiB |
| DET-FAKE-256x64 | lean | PASS | 5 / 44 | 7,617 | 60.5 s | 34.2 s | — | 103 s | 30.1 min | 47.9 / 55.8 GiB | 43.7 GiB |
| DET-NATIVE-256x64 | legacy | PASS | 8 / 63 | 8,385 | 69.5 s | 13.7 s | 0.11 s | 118 s | 23.6 min | 72.1 / 74.6 GiB | 43.9 GiB |
| DET-NATIVE-256x64 | lean | PASS | 8 / 63 | 8,385 | 60.2 s | 13.0 s | 0.15 s | 121 s | 21.6 min | 48.1 / 54.9 GiB | 43.9 GiB |
| DET-FAKE-8x64 | legacy | PASS | 10 / 129 | 3,454 | 70.2 s | 34.9 s | — | 95 s | 86.8 min | 62.0 / 64.6 GiB | 44.1 GiB |
| DET-FAKE-8x64 | lean | PASS | 10 / 129 | 3,454 | 60.8 s | 34.2 s | — | 104 s | 83.7 min | 49.1 / 57.0 GiB | 44.2 GiB |
| DET-NATIVE-8x64 | legacy | PASS | 9 / 114 | 3,801 | 69.5 s | 13.7 s | 0.11 s | 121 s | 36.5 min | 73.3 / 75.8 GiB | 44.2 GiB |
| DET-NATIVE-8x64 | lean | PASS | 9 / 114 | 3,801 | 60.2 s | 13.0 s | 0.15 s | 119 s | 33.8 min | 49.2 / 55.9 GiB | 44.1 GiB |

| configuration | phase | legacy peak GPU allocated | lean peak GPU allocated |
|---|---|---:|---:|
| DET-FAKE-256x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-256x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-256x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-FAKE-256x64 | candidate_packing | 25.7 GiB | 18.4 GiB |
| DET-FAKE-256x64 | initial_dev_eval | 42.0 GiB | 29.0 GiB |
| DET-FAKE-256x64 | scoring | 60.9 GiB | 47.9 GiB |
| DET-FAKE-256x64 | dev_evaluation | 42.1 GiB | 29.1 GiB |
| DET-FAKE-256x64 | final_evaluation | 24.9 GiB | 11.9 GiB |
| DET-NATIVE-256x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-256x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-256x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-NATIVE-256x64 | candidate_packing | 33.1 GiB | 18.4 GiB |
| DET-NATIVE-256x64 | initial_dev_eval | 49.4 GiB | 29.0 GiB |
| DET-NATIVE-256x64 | native_dev_evaluation | 53.2 GiB | 32.9 GiB |
| DET-NATIVE-256x64 | scoring | 72.1 GiB | 48.1 GiB |
| DET-NATIVE-256x64 | dev_evaluation | 53.3 GiB | 32.9 GiB |
| DET-NATIVE-256x64 | final_evaluation | 36.1 GiB | 12.0 GiB |
| DET-FAKE-8x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-8x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-8x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-FAKE-8x64 | candidate_packing | 25.8 GiB | 18.4 GiB |
| DET-FAKE-8x64 | initial_dev_eval | 42.0 GiB | 29.0 GiB |
| DET-FAKE-8x64 | scoring | 62.0 GiB | 49.1 GiB |
| DET-FAKE-8x64 | dev_evaluation | 42.8 GiB | 29.9 GiB |
| DET-FAKE-8x64 | final_evaluation | 25.6 GiB | 12.6 GiB |
| DET-NATIVE-8x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-8x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-8x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-NATIVE-8x64 | candidate_packing | 33.1 GiB | 18.4 GiB |
| DET-NATIVE-8x64 | initial_dev_eval | 49.4 GiB | 29.1 GiB |
| DET-NATIVE-8x64 | native_dev_evaluation | 53.3 GiB | 32.9 GiB |
| DET-NATIVE-8x64 | scoring | 73.3 GiB | 49.2 GiB |
| DET-NATIVE-8x64 | dev_evaluation | 54.1 GiB | 33.7 GiB |
| DET-NATIVE-8x64 | final_evaluation | 36.9 GiB | 12.8 GiB |

- **Dev evaluation per try** is the mean over all backtracking tries (the `dev_evaluation`
  phase). For native runs this is 13.7 s. The Task 1 and Task 1b reports' "14.4 s per try" is
  the time of the initial native evaluation only; their total optimization times and speed-ups
  are unaffected.
- **Scoring.** Lean scoring is faster (about 60 vs 70 s per pass), not slower. Its forward
  decodes and its backward recomputes cost less than legacy's PyTorch decode of the old fake
  store, which the scoring hooks call for the flip direction.
- **Where the memory goes.** Lean removes three things from GPU memory:
  - the 13.0 GiB of resident BF16 weights;
  - the duplicate 7.31 GiB fake store (in native runs, where legacy holds both stores);
  - the 3.7 GiB resident native per-map weight.

  The scoring peak falls from 60.9–73.3 to 47.9–49.1 GiB. The development-evaluation peak falls
  from 42–53 to 29–33 GiB.
- **Host memory is unchanged (about 44 GiB).** It is dominated by the BF16 teacher
  log-probabilities, which both modes keep in host RAM.

## Deviations

1. **A queue-script variable bug** (PROTOCOL.md, deviation 1): the first check was invoked on a
   nonexistent directory. It was re-run by hand on the correct run and passed, and the queue
   resumed. No run, flag, criterion or code changed.

## Notes

- **Registered code.** Every Part B run imported exactly the registered code.
  - Each run records the sha256 of the five files it loads (`run_multiround.py`, `quantizer.py`,
    `causal_four_over_six.py`, `native_dev.py`, `candidate_store.py`). All eight records equal
    `registration.json`.
  - One slip for the record: while run 1 was in progress (about 18:50–18:53 UTC), Part C
    additions were briefly written into `native_dev.py` and `candidate_store.py` in this checkout.
  - Run 1 never loads either file, and both were restored to their registered bytes (sha256
    checked) before run 2 started.
  - Part C was then developed in a separate git worktree, so no queued run could import
    unregistered code.
- **CPU work alongside the runs.** Part C's CPU-only data preparation ran at nice 19 during run 1
  (18:57–19:10 UTC). It is in neither the criteria nor the lean/legacy pairing.
- **Default mode.** The default is left at `legacy`, so the documented Task 1 and Task 1b
  reproduction commands keep their recorded memory profile.

## Reproduction

```
DATA=/home/dev/n16k64_campaign/cost_comparison/data
FLAGS="--objective kl --data-root $DATA --transformers-deviation --skip-ce-backward --eval-batch 16 \
       --score-batch 8 --budget-hours 12 --deterministic --record-dev-values --dump-round0-scores"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$PWD HF_HUB_OFFLINE=0
python repro_local/realquant/test_lean_memory.py $DATA results/lean_memory/unit_checks.json MAPS...
python run_multiround.py --unit 256x64 $FLAGS --dev-backend fake --memory-mode lean --out $R/lean_det_fake_256x64
# ... the eight runs of runs/queue_b2.sh; each checked with
python results/lean_memory/compare_runs.py COMMITTED_REPORT COMMITTED_MAP RUN_DIR [LEGACY_DIR]
python results/lean_memory/analyze_partb.py $R        # summary.json, tables.md
```

- **Environment:** torch 2.9.0+cu128, transformers 5.16.1 (the recorded deviation from 4.57.3),
  triton 3.5.0, `libb8x64.so` (sha256 0e237ada…).
- **Records:** run records are in `runs/` (report.json of each run, the check outputs,
  `commands.log`, and the queue scripts). The large per-run files (`round0_scores.pt`,
  `dev_values.pt`) stay in `/home/dev/n16k64_campaign/lean_memory`; their sha256 is in each
  report.json.
