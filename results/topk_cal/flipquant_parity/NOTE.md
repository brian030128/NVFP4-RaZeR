# flipquant parity: calibration time and memory, flipquant vs NVFP4-RaZeR

**The request.** A light parity check, asked by the user through the coordinator on 2026-10-05. It compares flipquant's
`python -m calibration.train_map_razer` (flipquant main b1c4123, run from a clean worktree) with NVFP4-RaZeR's
`run_train_map.py` (topk-cal 400c314) under identical settings. It is not registered: 1–2 models, not a study.

**The setup.** RTX PRO 6000, one run at a time on an idle GPU (checked before each run), on 2026-10-05.
- Deterministic, `--no-dev --no-eval`, the paper's settings unless stated.
- Every run of the queue (`run_parity.sh`) went through `measure.py`. It records the end-to-end wall time and samples
  at 10 Hz the RSS of the top process (flipquant: the wrapper) and of its descendants (the trainer subprocess)
  separately.
- The trainer's own numbers come from its PhaseMonitor, in both repos (`parity.py`, with `../analyze.py`'s metrics).

| model | NVFP4-RaZeR | flipquant |
|---|---|---|
| Llama-3.1-8B 16x64, A (full vocabulary, 20 epochs) | topk-cal's run A (08:08 UTC, this day) | A′ (09:35 UTC) |
| Llama-3.1-8B 16x64, C (top-256, 5 epochs) | topk-cal's run C (08:20 UTC) | C′ (09:44 UTC) |
| Phi-4 16x64 (full vocabulary, 20 epochs) | `run_train_map.py`, the paper's command (10:02 UTC) | back to back with it (09:47 UTC) |

flipquant used its prepared data root and RaZeR the paper's data roots. Their calibration records are equal (stage 3
validation in flipquant).

**Pass criteria** (as asked):
- the same map sha256;
- per-epoch and total time within ±3 %;
- GPU peak within 0.1 GiB;
- host RSS within 0.5 GiB.

Times are the trainer's; the wrapper's end-to-end overhead is reported separately.

## Results (`parity.md`, `parity.json`; flipquant / RaZeR)

| | Llama A | Llama C | Phi-4 |
|---|---:|---:|---:|
| map sha256 | equal (54070819…) | equal (af6c7515…) | equal (110243a6…) |
| per epoch | 22.89 / 23.01 s = 0.995 | 20.13 / 20.14 s = 0.999 | 39.90 / 39.93 s = 0.999 |
| trainer's total | 516.2 / 518.3 s = 0.996 | 157.7 / 157.2 s = 1.003 | 895.6 / 894.8 s = 1.001 |
| setup | 58.4 / 58.2 s | 57.0 / 56.5 s | 97.5 / 96.1 s |
| flipquant end to end (wrapper overhead) | 529.4 s (+13.2 s) | 168.8 s (+11.1 s) | 908.7 s (+13.1 s); RaZeR's process 902.3 s, ratio 1.007 |
| peak GPU allocated / reserved | +0.00 / +0.02 GiB | +0.00 / +0.04 GiB | +0.00 / +0.02 GiB |
| peak host RSS, `ru_maxrss` | 18.10 / 18.13 GiB (−0.03) | 15.88 / 15.88 GiB (0.00) | 28.23 / 28.23 GiB (0.00) |
| peak host RSS, sampled at 10 Hz | 18.16 / 18.19 GiB (−0.03) | 15.01 / 15.72 GiB (**−0.72**) | 28.09 / 27.51 GiB (**+0.57**) |
| host RSS after the model load | 18.16 / 18.19 GiB | 2.60 / 2.55 GiB | 14.91 / 14.90 GiB |
| the wrapper process's own RSS | 0.78 GiB | 0.78 GiB | 0.90 GiB |

**Verdict.**
- **The maps are identical.** Per-epoch and trainer-total times are within 0.5 %. GPU peaks are equal (allocated) or
  within 0.04 GiB (reserved). The exact host peak (`ru_maxrss`) is within 0.03 GiB. All of these pass in every pair.
- **The sampled host RSS misses the 0.5 GiB criterion in two pairs** (Llama C −0.72 GiB, Phi-4 +0.57 GiB). This is
  the trainer's measurement, not the wrapper, and not a real difference:
  - in all four runs concerned the sampled peak falls in the model-load phase;
  - there, the 10 Hz sampling reads 0.14–0.87 GiB below the exact `ru_maxrss` depending on when the samples land on a
    short spike;
  - the exact peaks are identical, and after the load the two agree within 0.05 GiB.
- **The wrapper's real costs:**
  - **Time:** flipquant's end-to-end time exceeds its trainer's total by 11–13 s. This covers the wrapper's own start
    and preparation check, the trainer subprocess, and the map conversion with its checks.
    - RaZeR's bare process, measured the same way on Phi-4, also exceeds its trainer's total by 7.5 s. Against it, the
      wrapper's net extra is about 6 s (908.7 against 902.3 s end to end, 1.007).
    - RaZeR's end-to-end time was not measured for the Llama runs. Against its trainer total, the short top-K run takes
      7 % longer end to end (168.8 against 157.2 s), and the 20-epoch run 2 % (529.4 against 518.3 s).
  - **Memory:** its own process stays resident at 0.8–0.9 GiB during the run, so flipquant's process tree uses that
    much more host memory than RaZeR's single process.

**Files:** `run_parity.sh`, `measure.py`, `parity.py`; `runs/` holds the trainer reports, the wrappers'
`reproduction.json`, the `measure.py` records and the queue log. The maps are in `/home/dev/n16k64_campaign/fqparity/`;
their sha256 values are in the reports. ~/flipquant was not modified (the worktree stayed clean).
