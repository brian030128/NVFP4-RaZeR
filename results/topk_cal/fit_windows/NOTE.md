# Calibration cost with 256 and 512 fit windows and the top-1000 teacher (Llama-3.1-8B)

**The request.** The user, through the coordinator, 2026-10-05: the time and memory of 256 windows × top-1000 × 5 epochs
and 512 windows × top-1000 × 5 epochs, on Llama-3.1-8B only. A window is 512 tokens; the paper uses 128. This is a
note, not a registered study.

**Code** (branch `topk-cal`): `run_train_map.py --fit-windows N`.
- **The first 128 windows** are the paper's fit set (the model's record), as before.
- **The other N − 128** follow flipquant's extension rule (`fit_extension.py`; flipquant/data.py
  `calibration_sets_extended` at b1c4123), applied on top of the paper's 128:
  - its `_stream_windows`, verbatim, on the paper record's two streams (OpenWebMath train-00000 @ fde8ef8 and
    CodeParrot-clean file-000000000001 @ 35a59fb), half math and half code;
  - file order, documents of at least 512 tokens, one window per document at `random.Random(20260930)`, i.e. flipquant's
    `SEED + 2`;
  - it skips the paper's 128 fit documents, its 192 development documents (the three development records) and its 231
    published C4 evaluation documents. WikiText-2 is another corpus.
- **How it differs from flipquant's own rule:** flipquant extends its own 128-window set and skips its own fit and
  development documents. Here the base and the skip set are the paper's.
- **Records:** every extra window is recorded in the run's report (`fit_extension`) by document sha256, offset and
  token sha256. The 256-window set is a per-source prefix of the 512-window set.
- **The default** (`--fit-windows 128`) is the unchanged path. Check: `--teacher-topk 256 --epochs 5` reproduced
  topk-cal run C's map bit for bit (af6c7515…, `runs/check_C.json`).

**Runs** (`run_fit_windows.sh`): Llama-3.1-8B 16x64, TM-OPT+TC with the paper settings otherwise (lr 0.02, init −1,
batch 8, seed 0), deterministic, `--no-dev --no-eval`, one at a time on an idle GPU, 10:25–10:43 UTC.
- **E128**, 128 windows with the top-1000 teacher, was added as a reference. It separates the effect of K = 1000 from
  the window count.
- **A and C** are topk-cal's runs of 08:08–08:23 UTC, same machine and day (`../REPORT.md`).

| | A | C | E128 | E256 | E512 |
|---|---:|---:|---:|---:|---:|
| fit windows / teacher top-K / epochs | 128 / full / 20 | 128 / 256 / 5 | 128 / 1000 / 5 | 256 / 1000 / 5 | 512 / 1000 / 5 |
| optimizer steps | 320 | 80 | 80 | 160 | 320 |
| setup (s): load + data + teacher + packing | 58.2 (1.8 + 11.9 + 7.9 + 36.7) | 56.5 | 57.3 | 71.7 (1.8 + 20.8 + 12.0 + 37.0) | 83.8 (1.8 + 22.6 + 23.2 + 36.4) |
| per epoch (s) | 23.01 | 20.14 | 20.11 | 40.35 | 80.76 |
| training (s) | 460.1 | 100.7 | 100.5 | 201.8 | 403.8 |
| **total** | **518.3 s (8.6 min)** | 157.2 s | 157.9 s (2.6 min) | **273.5 s (4.6 min)** | **487.7 s (8.1 min)** |
| total / A | 1 | 0.303 | 0.305 | **0.528** | **0.941** |
| peak GPU allocated / reserved (GiB) | 40.48 / 41.45 | 41.07 / 42.09 | 41.35 / 42.39 | 41.71 / 42.69 | 42.44 / 43.42 |
| peak host RSS, `ru_maxrss` (GiB) | 18.13 | 15.88 | 15.88 | 15.88 | 15.88 |
| host RSS after the model load (GiB) | 18.19 | 2.55 | 2.61 | 2.74 | 2.67 |
| teacher storage | 16.8 GB, host | 101 MB, GPU | 393 MB, GPU | 785 MB, GPU | 1.57 GB, GPU |
| tail mass, mean / max window | — | 1.84 % / 5.6 % | 0.73 % / 2.1 % | 0.73 % / 3.4 % | 0.76 % / 10.3 % |
| final E0M3 tiles (information) | 201,648 | 3,699 | 3,736 | 32,125 | 106,404 |
| final-epoch training KL (information) | 0.0424 (full KL) | 0.0826 | 0.0850 | 0.0788 | 0.0669 (top-K KL) |

Ratios to A for every row are in `fit_windows.md`, and all values in `fit_windows.json`.

**Reading:**
- **256 windows take 4.6 min (0.53 of A); 512 windows take 8.1 min (0.94 of A).**
  - E512 runs the same 320 optimizer steps as A, at 1.26 s per step against A's 1.44 s. The top-K loss is 12 % faster
    per step. The top-K runs take 1.26 s per step whatever the window count.
  - E512's setup is 26 s longer than A's: 11 s more to stream the extra windows and 15 s more for the teacher's 512
    forwards.
- **K = 1000 costs no time against K = 256** (E128 against C: 20.11 against 20.14 s per epoch). It costs only the
  store: 3.07 MB per window against 0.79 MB.
- **GPU memory grows with the store:** +0.28 / +0.64 / +1.37 GiB over C for 128 / 256 / 512 windows at K = 1000. That
  is exactly how much larger each store is than C's. E512 peaks at 42.4 GiB allocated, 2.0 GiB above A.
- **Host memory stays at 2.6–2.7 GiB after the model load** for every window count. The 15.9 GiB process peak is the
  model load itself.
- **The maps are not comparable to the paper's,** so this is not an accuracy comparison. With 5 epochs, E512's 320
  steps flip 106,404 tiles to E0M3, about half of A's 201,648 after the same number of steps. Each epoch there covers 4
  times as many windows. The training KLs are top-K KLs, lower bounds on the full KL. No PPL was measured.

**Files:** `run_fit_windows.sh`, `analyze_fit.py`, `fit_windows.{md,json}`; `runs/` holds the four reports, with every
extra window's record, and the queue log. The maps are in `/home/dev/n16k64_campaign/topk/fit_windows/`.
