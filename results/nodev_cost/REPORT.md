# Task 1: calibration cost without the development set — report

Protocol: `PROTOCOL.md`, registered 2026-09-27T18:33:28Z (sha256 d00dd4acf66d…, `registration.json`), before any run.
The code that ran is the registered code (`run_train_map.py` db6b57d9…, `run_cost_distill.py` 9fd4217f…, `queue.sh`
8608d20f…; the committed files have the same hashes). No deviations.

## Summary

**The definition.** A real calibration has no development set. The calibration cost therefore counts:
- model load;
- the fit data and fit teacher (128 sequences);
- the method's preparation;
- training;
- writing the output.

No development teacher, development evaluation, monitoring, verification or WikiText/C4 evaluation runs
(`--no-dev`, `--no-eval`). In every run the process's total time equals the sum of these phases (within 1 s, checked by
`analyze.py`).

**TM-OPT+TC, the final method** (deterministic, final settings). Host memory is given sampled (every 0.1 s) / as the
process's lifetime peak `ru_maxrss`:

| model | calibration | peak GPU allocated | peak host RSS |
|---|---:|---:|---:|
| Llama-3.1-8B | 8.5–8.6 min | 40.4–40.6 GiB | 18.6 / 18.5 GiB |
| Mistral-7B-v0.3 | 7.5 min | 35.8–36.0 GiB | 14.0–14.2 / 14.7 GiB |
| Phi-4 | 14.8–14.9 min | 58.9–59.4 GiB | 28.1–28.3 / 28.5 GiB |
| Qwen3.8-27B (8x64, micro-batch 2 × accumulation 4) | 76.0 min | 90.1 GiB | 51.3 / 52.2 GiB |

- **Units:** the cost does not depend on the unit. 8x64, 16x64 and 256x64 are within 0.1 min.
- **Qwen** was run at 8x64 only, as registered. Part Q measured its per-epoch time and memory as unit-independent
  (216.5–217.0 s, 89.6–90.4 GiB), so the 8x64 cost stands for all three units. 16x64 and 256x64 are added only if
  asked.
- **Time against the earlier with-dev accounting** (setup + selection): 17–24 % less on the 7–14B models and 12 %
  less on Qwen.
- **Host memory against the with-dev accounting** (`ru_maxrss` on both sides; the with-dev records hold only that):
  - Llama falls from 42.3 to 18.5 GiB and Qwen from 79.5 to 52.2 GiB; Phi-4 from 33.7 to 28.5 GiB.
  - The with-dev runs held the development teacher's log-probabilities on the host (full vocabulary, bf16).
  - Mistral's peak is unchanged at 14.7 GiB. It is set by the model load in both runs; during training its host
    memory falls from 13.1 to 6.7 GiB.
  - Qwen's peak without dev is its model load (51.3 GiB sampled); its training phase uses 33.7 GiB.

**The maps are unchanged.** All 10 deterministic `--no-dev` maps are bitwise equal (sha256) to the committed
TM-OPT+TC maps. The per-epoch trajectory is identical in every field but the time (E0M3 count, hard flips, training KL,
τ, learning rate, step). The development monitor never touched training.

**Llama-3.1-8B, TM-OPT+TC against the baselines at their recorded budgets.** One epoch is 16 optimizer steps of 8
sequences for all three methods. Host memory is `ru_maxrss`.

| method | deterministic | calibration | training | peak GPU allocated | host RSS |
|---|---|---:|---:|---:|---:|
| TM-OPT+TC (20 epochs) | yes | 8.5–8.6 min | 454–458 s (22.7–22.9 s per epoch) | 40.4–40.6 GiB | 18.5 GiB |
| TM-OPT+TC (20 epochs) | no | 8.2 min | 423–425 s (21.2 s per epoch) | 40.4–40.6 GiB | 18.5–18.6 GiB |
| QAT C1 (1 epoch) | yes | 1.7 min | 32 s | 85.3 GiB | 46.0 GiB |
| QAT C1 (1 epoch) | no | 2.1 min | 30 s | 85.3 GiB | 46.0 GiB |
| scale-only D1 (1 epoch) | yes | 1.3 min | 19 s | 74.0 GiB | 34.3 GiB |
| scale-only D1 (1 epoch) | no | 1.6 min | 19 s | 74.0 GiB | 34.3 GiB |

- **Per epoch**, TM-OPT+TC's training (21–23 s) lies between scale-only's (19 s) and QAT's (30–32 s). Its total is
  larger because its budget is 20 epochs against their one.
- **Memory:** TM-OPT+TC uses 47–48 % of QAT's peak GPU memory and 55 % of scale-only's. Its host memory is 18.5 GiB,
  against 46.0 and 34.3 GiB.
- **Determinism slows training only for TM-OPT+TC**, by 7–8 % (454–458 s against 423–425 s). The QAT and scale-only
  training phases are within 2 s of each other.
- **Determinism speeds up the setup, as a side effect.**
  - **Where it shows:** in all five non-deterministic Llama runs the fit-teacher phase took 14 s, against 7–8 s in the
    six deterministic Llama runs. Run order does not explain it: the deterministic QAT and scale-only runs ran between
    non-deterministic ones.
  - **The cause, verified with `d2h_determinism.py`** (output in `d2h_determinism.txt`):
    - Under `torch.use_deterministic_algorithms(True)`, PyTorch fills newly allocated memory
      (`torch.utils.deterministic.fill_uninitialized_memory`). The fill touches the host buffer that the teacher
      log-probabilities are copied into before the GPU→host copy runs.
    - Copying 8.4 GB runs at 7.7 GB/s with the fill and 1.7–1.9 GB/s without it.
    - Deterministic mode with the fill disabled is as slow as non-deterministic mode.
  - **Other phases, by inference, not measured separately:** QAT's and scale-only's preparation (a host copy of the
    pristine weights) and QAT's state write (16 GB to the host) are 6–7 s longer without determinism, the same kind of
    copy. Their model load is 4–5 s longer; that was not investigated.
- **The non-deterministic TM-OPT+TC maps differ** from the deterministic ones: Jaccard 0.50–0.60, and E0M3 tile counts
  within 2 %. Their accuracy is not evaluated here.

## All runs

Included phases (PROTOCOL.md): model load, fit data, fit teacher, preparation (TM: candidate packing), training, writing the output. Memory: maximum over the included phases; host RSS sampled every 0.1 s, and ru_maxrss (the process's lifetime peak).

| model | unit | method | deterministic | model / data / teacher / prep. / training / write (s) | calibration | peak GPU allocated / reserved | peak host RSS sampled / ru_maxrss | check |
|---|---|---|---|---|---:|---:|---:|---|
| Llama-3.1-8B | 16x64 | TM-OPT+TC | yes | 2 / 12 / 7 / 36 / 457 / 0 | 8.6 min | 40.5 / 41.5 GiB | 18.6 / 18.5 GiB | map = committed |
| Llama-3.1-8B | 256x64 | TM-OPT+TC | yes | 2 / 12 / 8 / 36 / 458 / 0 | 8.6 min | 40.4 / 41.4 GiB | 18.6 / 18.5 GiB | map = committed |
| Llama-3.1-8B | 8x64 | TM-OPT+TC | yes | 2 / 13 / 7 / 36 / 454 / 0 | 8.5 min | 40.6 / 41.6 GiB | 18.6 / 18.5 GiB | map = committed |
| Llama-3.1-8B | 16x64 | TM-OPT+TC | no | 2 / 12 / 14 / 39 / 425 / 0 | 8.2 min | 40.5 / 41.6 GiB | 18.6 / 18.5 GiB | 201,113 tiles (det 201,648), Jaccard 0.526 |
| Llama-3.1-8B | 256x64 | TM-OPT+TC | no | 2 / 12 / 14 / 38 / 424 / 0 | 8.2 min | 40.4 / 41.3 GiB | 18.6 / 18.6 GiB | 35,229 tiles (det 35,365), Jaccard 0.597 |
| Llama-3.1-8B | 8x64 | TM-OPT+TC | no | 2 / 12 / 14 / 39 / 423 / 0 | 8.2 min | 40.6 / 41.7 GiB | 18.6 / 18.5 GiB | 314,422 tiles (det 309,517), Jaccard 0.502 |
| Llama-3.1-8B | — | QAT C1 | yes | 22 / 12 / 8 / 15 / 32 / 15 | 1.7 min | 85.3 / 85.5 GiB | 46.1 / 46.0 GiB | lr 1e-06, 16 steps |
| Llama-3.1-8B | — | QAT C1 | no | 27 / 12 / 14 / 22 / 30 / 22 | 2.1 min | 85.3 / 85.5 GiB | 46.0 / 46.0 GiB | lr 1e-06, 16 steps |
| Llama-3.1-8B | — | scale-only D1 | yes | 23 / 12 / 7 / 16 / 19 / 2 | 1.3 min | 74.0 / 75.3 GiB | 34.3 / 34.3 GiB | lr 0.001, 16 steps |
| Llama-3.1-8B | — | scale-only D1 | no | 27 / 12 / 14 / 22 / 19 / 2 | 1.6 min | 74.0 / 75.3 GiB | 34.3 / 34.3 GiB | lr 0.001, 16 steps |
| Mistral-7B-v0.3 | 16x64 | TM-OPT+TC | yes | 2 / 11 / 6 / 36 / 393 / 0 | 7.5 min | 35.9 / 36.6 GiB | 14.0 / 14.7 GiB | map = committed |
| Mistral-7B-v0.3 | 256x64 | TM-OPT+TC | yes | 2 / 11 / 6 / 36 / 393 / 0 | 7.5 min | 35.8 / 36.5 GiB | 14.2 / 14.7 GiB | map = committed |
| Mistral-7B-v0.3 | 8x64 | TM-OPT+TC | yes | 2 / 12 / 6 / 36 / 394 / 0 | 7.5 min | 36.0 / 36.8 GiB | 14.1 / 14.7 GiB | map = committed |
| Phi-4 | 16x64 | TM-OPT+TC | yes | 3 / 11 / 10 / 72 / 794 / 0 | 14.8 min | 59.1 / 62.1 GiB | 28.1 / 28.5 GiB | map = committed |
| Phi-4 | 256x64 | TM-OPT+TC | yes | 3 / 12 / 10 / 72 / 795 / 0 | 14.9 min | 58.9 / 62.1 GiB | 28.3 / 28.5 GiB | map = committed |
| Phi-4 | 8x64 | TM-OPT+TC | yes | 3 / 12 / 10 / 73 / 794 / 0 | 14.8 min | 59.4 / 62.5 GiB | 28.1 / 28.5 GiB | map = committed |
| Qwen3.8-27B | 8x64 | TM-OPT+TC | yes | 5 / 12 / 80 / 129 / 4335 / 0 | 76.0 min | 90.1 / 91.6 GiB | 51.3 / 52.2 GiB | map = committed |

Compared with the earlier with-dev records (TM: setup + selection; QAT/D: every phase of the recorded run). Host memory: ru_maxrss on both sides (the with-dev records hold only ru_maxrss).

| model | unit | method | with dev | without dev | change | host ru_maxrss with / without dev |
|---|---|---|---:|---:|---:|---|
| Llama-3.1-8B | 16x64 | TM-OPT+TC | 11.2 min | 8.6 min | -23 % | 42.3 / 18.5 GiB |
| Llama-3.1-8B | 256x64 | TM-OPT+TC | 11.2 min | 8.6 min | -23 % | 42.2 / 18.5 GiB |
| Llama-3.1-8B | 8x64 | TM-OPT+TC | 11.2 min | 8.5 min | -24 % | 42.3 / 18.5 GiB |
| Mistral-7B-v0.3 | 16x64 | TM-OPT+TC | 9.1 min | 7.5 min | -19 % | 14.7 / 14.7 GiB |
| Mistral-7B-v0.3 | 256x64 | TM-OPT+TC | 9.1 min | 7.5 min | -18 % | 14.7 / 14.7 GiB |
| Mistral-7B-v0.3 | 8x64 | TM-OPT+TC | 9.1 min | 7.5 min | -18 % | 14.7 / 14.7 GiB |
| Phi-4 | 16x64 | TM-OPT+TC | 17.9 min | 14.8 min | -17 % | 33.7 / 28.5 GiB |
| Phi-4 | 256x64 | TM-OPT+TC | 18.0 min | 14.9 min | -17 % | 33.7 / 28.5 GiB |
| Phi-4 | 8x64 | TM-OPT+TC | 17.9 min | 14.8 min | -17 % | 33.7 / 28.5 GiB |
| Qwen3.8-27B | 8x64 | TM-OPT+TC | 86.2 min | 76.0 min | -12 % | 79.5 / 52.2 GiB |
| Llama-3.1-8B | — | QAT C1 | 3.9 min | 2.1 min | -47 % | 71.5 / 46.0 GiB |
| Llama-3.1-8B | — | scale-only D1 | 3.1 min | 1.6 min | -47 % | 58.4 / 34.3 GiB |

## Files

- **Tables:** `cost.{json,md}`, generated by `analyze.py`. It reads the maps from `/home/dev/n16k64_campaign/nodev_cost/runs`
  for the hash cross-check and the Jaccard overlaps.
- **Records:** `runs/<run>/report.json` holds every phase with its time and memory. TM runs also give the map sha256
  and per-epoch entries. The queue is in `runs/queue.sh` and `runs/commands.txt`.
- **Not committed:** the maps (bitwise equal to the committed runs') and QAT's 16 GB trained state. Both are in
  `/home/dev/n16k64_campaign/nodev_cost/runs`.
- **Determinism benchmark:** `d2h_determinism.{py,txt}`.
- **Code:** `run_train_map.py --no-dev --no-eval`; `run_cost_distill.py --no-dev --deterministic`. The default paths
  are unchanged.
