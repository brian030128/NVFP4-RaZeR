# The calibration cost of TM-OPT+TC with flipquant's top-K teacher — report

The protocol is `PROTOCOL.md` (registration 715baa0). `run_topk_cal.sh` ran on 2026-10-05 from 08:08 to 08:23 UTC,
with no deviation. A's map equals the paper's Llama-3.1-8B 16x64 map bitwise (54070819…), as the queue's gate required.
Tables are in `topk_cal.md`, data in `topk_cal.json`, and the runs' reports in `runs/`.

**The setting.** Llama-3.1-8B, 16x64, TM-OPT+TC with the paper settings (lr 0.02, init −1, batch 8, seed 0),
deterministic, `--no-dev --no-eval`, one run at a time on an RTX PRO 6000.

| run | teacher | epochs |
|---|---|---:|
| A | the paper's full-vocabulary teacher | 20 |
| B | the full-vocabulary teacher | 5 |
| C | flipquant's top-256 teacher with one tail bucket (`--teacher-topk 256`) | 5 |

The teacher is checked bitwise against flipquant's definition (check (b)), and `--teacher-topk 0` is the paper path
(check (a); see the protocol).

## Results

| | A | B | C | C / A | C / B |
|---|---:|---:|---:|---:|---:|
| **setup** (s) | 58.2 | 59.6 | 56.5 | 0.971 | 0.948 |
| — of which teacher precompute (s) | 7.9 | 7.3 | 6.4 | | |
| — of which lean packing (s) | 36.7 | 36.4 | 36.1 | | |
| **per epoch** (s) | 23.01 | 23.02 | 20.14 | 0.876 | 0.875 |
| **training** (s) | 460.1 | 115.1 | 100.7 | 0.219 | 0.875 |
| **total** (s) | 518.3 (8.64 min) | 174.7 (2.91 min) | 157.2 (2.62 min) | 0.303 | 0.900 |
| **peak GPU allocated** (GiB) | 40.48 | 40.48 | 41.07 | 1.015 | 1.015 |
| peak GPU reserved (GiB) | 41.45 | 41.51 | 42.09 | 1.016 | 1.014 |
| **peak host RSS**, sampled / `ru_maxrss` (GiB) | 18.19 / 18.13 | 18.24 / 18.18 | 15.72 / 15.88 | 0.864 / 0.876 | 0.862 / 0.873 |
| peak host RSS after the model load (GiB) | 18.19 | 18.24 | 2.55 | 0.140 | 0.140 |
| **teacher storage** | 16,778 MB, host | 16,778 MB, host | 100.7 MB, GPU | 0.0060 | 0.0060 |
| tail mass, mean / max window | | | 0.0184 / 0.0557 | | |
| final E0M3 tiles (information) | 201,648 | 4,055 | 3,699 | | |
| final-epoch training KL (information) | 0.0424 | 0.0860 | 0.0826 (top-K KL) | | |

The setup phases are model load (1.8 s), data load, teacher precompute and lean packing. The model-load and post-load
host RSS rows were added when the tables were made, to show where the host peak sits. They come from the same per-phase
records.

## Reading

- **Against the paper setting (A), C takes 30 % of the time.** That is 2.6 min against 8.6 min, and almost all of it
  comes from the epochs: 5 against 20 is 25 %, and the top-K loss saves another 12.5 % per epoch. The setup is nearly
  the same (−3 %): the lean packing (36 s) dominates it, and the teacher precompute takes only 6–8 s with either teacher.
- **The top-K effect alone (C against B) is −10 % total.** Per epoch it is 20.1 against 23.0 s (−12.5 %); the setup
  differs by −5 %.
  - The full teacher's 16.8 GB on the host is copied to the GPU every epoch, two documents at a time. C's teacher is
    already on the GPU, but its loss adds a scatter and a log-sum-exp over the vocabulary. These are the evident
    differences; how much each contributes was not profiled.
- **GPU memory is not reduced: C needs 0.6 GiB more** (41.07 against 40.48 GiB allocated). The top-K store itself is
  only 0.1 GB. The rest is presumably the loss's FP32 temporaries per chunk, i.e. the scatter copy and the gather's
  gradient, in place of the full teacher's chunk. This was inferred, not measured.
- **Host memory falls by 86 % after the model load**, from 18.2 to 2.55 GiB, because the full-vocabulary teacher
  (16.8 GB) is no longer on the host.
  - The process peak falls only 12–14 % (18.2 to 15.7–15.9 GiB). In C that peak is the model load itself
    (transformers' loading path, 15.7 GiB), which every run has. With the full teacher, the teacher sets the peak
    instead.
- **The teacher is 167 times smaller** (100.7 MB against 16.8 GB) and lives on the GPU. Its tail bucket holds 1.8 % of
  the teacher's probability mass on average (at most 5.6 % in a window).
- **The maps differ by construction, so this is not an accuracy comparison.** In 5 epochs (80 steps) few tiles flip:
  B selects 4,055 E0M3 tiles and C 3,699, against A's 201,648 after 20 epochs. That is about 2 % of A, as expected for
  lr 0.02 and init −1, where a tile needs about 50 consistent steps. C's training KL is the top-K KL, a lower bound on
  the full KL, so it cannot be compared with A's or B's. No PPL was measured.

**Consistency with the recorded no-dev cost** (results/nodev_cost's Llama-3.1-8B 16x64 run of A's configuration,
2026-09-28):

| | today's A | nodev_cost | ratio |
|---|---:|---:|---:|
| total | 8.64 min | 8.58 min | 1.007 |
| per epoch | 23.01 s | 22.85 s | 1.007 |
| setup | 58.2 s | 57.6 s | 1.011 |
| peak GPU allocated | 40.48 GiB | 40.48 GiB | 1.000 |
| peak host RSS, `ru_maxrss` | 18.13 GiB | 18.54 GiB | 0.978 |
| map | 54070819… | 54070819… | equal |

Today's A is within 1 % of the record in time, equal in GPU memory, and its map is the same. The C / A ratios therefore
hold against the recorded numbers to within about 1 % in time and 2 % in host memory.

**Code:** `run_train_map.py --teacher-topk K`, `topk_teacher.py` and `chunked_loss.train_kl_gradient_topk`, on branch
`topk-cal`. Nothing was adopted or tuned. tm-opt and `~/flipquant` are untouched.
