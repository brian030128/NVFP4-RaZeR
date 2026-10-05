# The calibration cost of TM-OPT+TC with flipquant's top-K teacher — protocol

Registered before any measured run. The hashes are in `registration.json`.

**The request.** The user, through the coordinator, 2026-10-05: measure the calibration time and memory of TM-OPT+TC
with flipquant's top-K teacher (K = 256) and 5 epochs, against the paper setting (the full-vocabulary teacher, 20
epochs). By the user's scope decision the study covers Llama-3.1-8B only. There is no PPL and no accuracy claim.

**The implementation.** Branch `topk-cal`, cut from `flipquant-maps` c89ee2c, so tm-opt and `~/flipquant` are
untouched.
- **`topk_teacher.py`** holds flipquant's definition verbatim: `tail_logprob`, `kl_topk_per_sequence`, and
  TopKTeacher's `append_logprobs` and `batch`, from `calibration/train_map.py` at main d2dd92e (introduced in
  c2674a9).
  - **The teacher:** per window, the top-K of the FP32 teacher log-probabilities. It keeps the values in BF16 and the
    token ids in int32, plus the exact FP32 log-mass of the rest. All of it stays on the GPU.
  - **The loss:** KL(teacher ‖ student) over the K tokens plus one tail bucket. That is the exact KL of the
    coarse-grained distributions, a lower bound on the full KL.
- **`run_train_map.py --teacher-topk K`** keeps that teacher instead of the full-vocabulary BF16 teacher on the host.
  K = 0, the default, runs the unchanged full-KL expressions.
- **`chunked_loss.train_kl_gradient_topk`** applies the top-K KL in the chunked loss of TM-OPT+TC, two documents at a
  time. The lean store, the fused activations, the TC tile-gradient GEMM, expandable segments and determinism all stay
  on.
- The whole-batch branch, without `--chunked-loss`, evaluates the same expression. The measured runs do not use it.
- The run report gains `teacher_storage`: the bytes, the devices, the layout and, for K > 0, the mean and maximum tail
  mass.

**Checks before measuring** (done before registration):
- **(b) Agreement with flipquant** (`repro_local/realquant/test_topk_teacher.py`, output `test_topk_teacher.json`). The
  test runs flipquant's functions from its source (d2dd92e), under deterministic algorithms.
  - Random logits, for the vocabularies of Mistral, Phi-4, Llama and Qwen3.8-27B, micro-batches 8 and 2, K = 256: the
    teacher rows and the tail mass, the chunked per-sequence KL, the bf16 logit gradient of a training step, and the
    whole-batch branch are all bitwise equal to flipquant's.
  - Real logits (Llama-3.1-8B's BF16 teacher against a FourOverSix-weight student, on the first 8 calibration windows):
    all bitwise equal.
  - Finite differences, float64: autograd against central differences, 1.3e-8 relative; against the closed form,
    2.6e-16.
- **(a) K = 0 is the paper path.** Llama-3.1-8B 8x64, TM-OPT+TC with the paper settings below, 20 epochs,
  `--teacher-topk 0`: the map's sha256 must equal the paper map (8b412c67…). Result: see "Amendments and results of
  the checks" below.
- **Disclosed:** a one-epoch smoke run of C into a scratch directory.

**The runs** (`run_topk_cal.sh`, one at a time, on an idle GPU):
- **Model and settings:** Llama-3.1-8B, unit 16x64. The unit does not change the cost (results/nodev_cost).
- **Paper settings:** `--tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1`, batch 8, seed 0, deterministic,
  `--no-dev --no-eval`.
- **Environment:** the data root and environment of results/nodev_cost (`--transformers-deviation`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

| run | teacher | epochs | gate |
|---|---|---:|---|
| A | full vocabulary | 20 | its map must equal the paper's 16x64 map (54070819…), else STOP |
| B | full vocabulary | 5 | |
| C | top-256 (`--teacher-topk 256`) | 5 | |

- B against C isolates the top-K effect; A against C is the user's question.
- 5 epochs are 80 optimizer steps. With lr 0.02 and init −1 a tile needs about 50 consistent steps to flip, so C and B
  will select far fewer E0M3 tiles than A.

**Recorded** from each run's `report.json`:
- **Time:**
  - the phases (PhaseMonitor): `model_load`, `data_load`, `teacher_precompute` (C: including the top-K selection),
    `candidate_packing` (the lean packing), `training` and `write_output`;
  - **setup** = model_load + data_load + teacher_precompute + candidate_packing;
  - the per-epoch times; **training** = their sum; **total** = the run's total seconds.
- **Memory:**
  - the run-wide peak GPU allocated and reserved (the maximum over the phases);
  - the peak host RSS, sampled and as `ru_maxrss`;
  - the teacher's storage, in bytes and by device.
- **Informational:**
  - C's tail mass;
  - the final E0M3 tile count;
  - the training KL per epoch. C's is the top-K KL, a lower bound on the full KL, so it is not comparable with A's and
    B's.

**Report** (`REPORT.md`):
- the absolute numbers of A, B and C;
- C as ratios to A and to B:
  - time: setup, per epoch, training and total;
  - memory: the GPU peaks, the host RSS peak and the teacher storage;
- today's A next to the results/nodev_cost record, as a consistency check. That record is 8.5–8.6 min, 22.7–22.9 s per
  epoch, 40.4–40.6 GiB peak GPU allocated and 18.5 GiB `ru_maxrss`.

Nothing is tuned or adopted from this study.

## Amendments and results of the checks

**Results of the checks (2026-10-05, before registration):**
- **(b) PASSED, all bitwise** (`test_topk_teacher.json`).
  - Random logits: 8 cases, all four vocabularies at micro-batches 8 and 2.
  - Real logits: Llama-3.1-8B, mean tail mass 0.0198, top-K KL 0.109.
  - Finite differences: 1.28e-8 and 1.20e-8 relative at K = 8 and K = 64; the closed form agrees to 2.6e-16.
- **(a) PASSED** (`check_a_llama8b_8x64.json`). With `--teacher-topk 0` the Llama-3.1-8B 8x64 map's sha256 is
  8b412c67d9ea4129…, equal to the paper map, with 309,517 E0M3 tiles. The run took 8.64 min. Its teacher is 16.78 GB of
  BF16 log-probabilities on the host.
- **The disclosed smoke run of C** (1 epoch, scratch):
  - 19.6 s for the epoch;
  - a teacher of 100.7 MB on the GPU, mean tail mass 0.0184 (at most 0.0557 per window);
  - 41.1 GiB peak GPU allocated and 15.9 GiB host `ru_maxrss`;
  - no E0M3 tile after 16 steps.

### The run

`run_topk_cal.sh` ran on 2026-10-05 from 08:08 to 08:23 UTC (registration 715baa0), with no deviation.
- **The gate:** A's map is the paper's 16x64 map, bitwise (54070819…).
- **Total time:** C is 157.2 s, against 518.3 s for A (0.303) and 174.7 s for B (0.900).
- **Per epoch:** 20.14 s against 23.0 s (0.875).
- **GPU:** peak allocated 41.07 against 40.48 GiB (1.015).
- **Host:** the peak RSS is 15.7–15.9 against 18.1–18.2 GiB (0.86–0.88), and in C it is the model load. After the load,
  host RSS is 2.55 against 18.2 GiB (0.140).
- **Teacher:** 100.7 MB on the GPU, against 16.8 GB on the host (0.006).
- **Consistency:** today's A is within 1 % of results/nodev_cost's run in time, equal in GPU memory, and its map is the
  same.
- **Results:** `REPORT.md`. The tables and data are in `topk_cal.md` and `topk_cal.json`.
