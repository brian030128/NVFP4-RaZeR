# Part Q (Qwen3.8-27B) and Part R (final four-model analysis) — report

Protocol: `PROTOCOL_QR.md`, registered 2026-09-26T17:52:02Z (sha256 05d84fd17661…, `registration_qr.json`),
before any Part Q or R run. Deviation 1 records the batch probe. Nothing was selected or tuned on WikiText-2, C4 or
zero-shot. The Qwen3.8-27B gate was open for this task only.

This file holds Part Q; Part R is added when it is complete.

## Part Q: Qwen3.8-27B, TM-OPT and TM-OPT+TC at 8x64, 16x64 and 256x64

### Summary

- **Batch:** micro-batch 8 and 4 run out of memory; **micro-batch 2 with accumulation 4 fits** (90.0 GiB
  allocated), so the optimizer batch stays at 8 sequences (deviation 1).
- **All six maps are significantly better than FourOverSix and than pure NVFP4 on both corpora**, in native and in
  fake evaluation.
- **TM-OPT 8x64 is the best Qwen map** (native):
  - WikiText-2: FourOverSix 7.3092 → **7.0739** (ΔNLL −0.0327 ± 0.0051). BF16 is 7.0509, so the map
    closes 91 % of FourOverSix's WikiText gap to BF16.
  - C4: 10.1840 → **10.1257** (−0.0057 ± 0.0010), 20 % of the C4 gap (BF16 9.8935).
- **Cost:**

  | | per epoch | selection |
  |---|---:|---:|
  | TM-OPT | 248 s | 90 min |
  | TM-OPT+TC | 217 s | 80 min |

  Both methods need 6.5 min of setup, 90 GiB of peak GPU memory (of 95) and 79 GiB of host RSS.
- **TM-OPT+TC on Qwen**, compared with the three smaller models:
  - **Speed-up:** only 12.7 % per epoch, against 35–39 % on Llama, Mistral and Phi-4.
  - **C4:** its maps are not significantly different from TM-OPT's at every unit.
  - **WikiText-2:** its maps are significantly worse at 8x64 (+0.0046 ± 0.0024) and 256x64
    (+0.0046 ± 0.0029), and significantly better at 16x64 (−0.0045 ± 0.0026).
  - **Against TC's acceptance rule:** 8x64 and 256x64 would not pass the rule TC met on all 9 cells of the other
    models (mean − 2 SE ≤ 0 vs TM-OPT on both corpora); 16x64 would. Part Q registered no pass/fail rule for Qwen.
  - **What it may mean:** the three WikiText differences have the same size and mixed signs, which looks like
    variation between optimization trajectories rather than a systematic loss. No Qwen seed spread was measured, so
    this cannot be separated. For scale, three Llama TM-OPT seeds differ by at most 0.0011.

### Batch probe (deviation 1)

TM-OPT at 8x64, one optimizer step of 8 sequences per probe:

| probe | micro-batch × accum | result | training-phase peak GPU allocated / reserved | host RSS peak | step |
|---|---|---|---:|---:|---:|
| qwen_probe_b8 | 8 × 1 | OOM | 93.8 / 94.3 GiB | 79.4 GiB | — |
| qwen_probe_b4 | 4 × 2 | OOM | 93.9 / 94.3 GiB | 79.4 GiB | — |
| qwen_probe_b2 | 2 × 4 | **FIT** | 90.0 / 91.5 GiB | 79.3 GiB | 16.2 s |

- **Where the OOMs happened:** both in the training forward, inside the Transformers torch implementation of
  the gated-delta-rule linear attention. Flash-linear-attention is not installed in the pinned environment.
- **Setup fits at every micro-batch:** at most 56.0 GiB (candidate packing).
- **What the micro-batching leaves unchanged:** the steps, the sequences of each step, the loss (the mean over
  8 sequences) and every hyperparameter. Micro-batch 1 was not probed: the rule takes the largest b that fits.

### Runs

Seed 0, deterministic, STE, lr 0.02, θ init −1, 20 epochs of 16 steps, monitor every 2 epochs at batch 16.
Setup is split as model / data / teacher / packing / initial dev; memory is peak GPU allocated / reserved.

| method | unit | E0M3 tiles | per epoch | selection | setup | memory | host RSS | dev KL, initial → final |
|---|---|---:|---:|---:|---|---:|---:|---|
| TM-OPT | 8x64 | 573,586 (1.21 %) | 248.5 s | 90.5 min | 6.5 min (4 / 11 / 199 / 132 / 43 s) | 90.4 / 92.0 GiB | 79.4 GiB | 0.04647 → 0.04198 |
| TM-OPT | 16x64 | 373,010 (1.57 %) | 248.3 s | 90.4 min | 6.5 min (4 / 12 / 202 / 130 / 42 s) | 89.9 / 91.5 GiB | 79.3 GiB | 0.04647 → 0.04175 |
| TM-OPT | 256x64 | 67,379 (4.51 %) | 247.5 s | 90.1 min | 6.5 min (4 / 13 / 198 / 130 / 43 s) | 89.6 / 91.0 GiB | 79.3 GiB | 0.04647 → 0.04272 |
| TM-OPT+TC | 8x64 | 614,657 (1.29 %) | 216.6 s | 79.8 min | 6.5 min (4 / 12 / 197 / 132 / 42 s) | 90.4 / 92.0 GiB | 79.5 GiB | 0.04647 → 0.04222 |
| TM-OPT+TC | 16x64 | 375,835 (1.58 %) | 217.0 s | 79.9 min | 6.6 min (4 / 12 / 203 / 131 / 43 s) | 89.9 / 91.5 GiB | 79.3 GiB | 0.04647 → 0.04211 |
| TM-OPT+TC | 256x64 | 61,295 (4.11 %) | 216.5 s | 79.7 min | 6.6 min (4 / 14 / 202 / 132 / 42 s) | 89.6 / 91.0 GiB | 79.4 GiB | 0.04647 → 0.04253 |

- **Development KL** (monitor only; never used to select): it falls from 0.0465 to 0.0418–0.0427 in all six runs.
  The per-epoch curves are in `final_qwen.md`.
- **Tile counts:** the E0M3 count still grows at epoch 20 in every run, as on the other models.
- **Unexplained, not profiled:** why TC saves less time here (31.4 s per epoch) than on the other models.
  Hypotheses, none measured:
  - the forward and backward through the torch linear-attention fallback take a larger share of the step;
  - micro-batch 2 gives the tile-gradient GEMMs a 4× shorter reduction dimension and 4× more calls.

### PPL (WikiText-2 and C4, convention (a); native is primary)

Paired ΔNLL per window, mean ± 2 SE. FourOverSix, NVFP4 and the six maps share one process per backend; BF16 has its
own fake process.

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |
|---|---|---:|---:|---|---|---|---|
| native | FourOverSix | 7.3092 | 10.1840 | — | — | −0.03596 ± 0.00684 (better) | −0.00359 ± 0.00114 (better) |
| native | NVFP4 | 7.5768 | 10.2206 | +0.03596 ± 0.00684 (worse) | +0.00359 ± 0.00114 (worse) | — | — |
| native | TM-OPT 8x64 | **7.0739** | **10.1257** | −0.03272 ± 0.00505 (better) | −0.00574 ± 0.00097 (better) | −0.06868 ± 0.00899 (better) | −0.00932 ± 0.00143 (better) |
| native | TM-OPT 16x64 | 7.1462 | 10.1342 | −0.02255 ± 0.00477 (better) | −0.00490 ± 0.00088 (better) | −0.05851 ± 0.00850 (better) | −0.00849 ± 0.00128 (better) |
| native | TM-OPT 256x64 | 7.1624 | 10.1488 | −0.02028 ± 0.00376 (better) | −0.00346 ± 0.00093 (better) | −0.05624 ± 0.00831 (better) | −0.00705 ± 0.00136 (better) |
| native | TM-OPT+TC 8x64 | 7.1062 | 10.1261 | −0.02816 ± 0.00476 (better) | −0.00570 ± 0.00095 (better) | −0.06412 ± 0.00859 (better) | −0.00928 ± 0.00131 (better) |
| native | TM-OPT+TC 16x64 | 7.1144 | 10.1335 | −0.02701 ± 0.00498 (better) | −0.00497 ± 0.00091 (better) | −0.06297 ± 0.00934 (better) | −0.00856 ± 0.00130 (better) |
| native | TM-OPT+TC 256x64 | 7.1951 | 10.1482 | −0.01573 ± 0.00366 (better) | −0.00352 ± 0.00076 (better) | −0.05169 ± 0.00772 (better) | −0.00711 ± 0.00111 (better) |
| fake | FourOverSix | 7.3007 | 10.1858 | — | — | −0.03591 ± 0.00743 (better) | −0.00397 ± 0.00111 (better) |
| fake | NVFP4 | 7.5676 | 10.2263 | +0.03591 ± 0.00743 (worse) | +0.00397 ± 0.00111 (worse) | — | — |
| fake | TM-OPT 8x64 | 7.0696 | 10.1283 | −0.03216 ± 0.00463 (better) | −0.00566 ± 0.00099 (better) | −0.06807 ± 0.00930 (better) | −0.00963 ± 0.00142 (better) |
| fake | TM-OPT 16x64 | 7.1335 | 10.1326 | −0.02317 ± 0.00409 (better) | −0.00524 ± 0.00096 (better) | −0.05907 ± 0.00870 (better) | −0.00920 ± 0.00123 (better) |
| fake | TM-OPT 256x64 | 7.1623 | 10.1447 | −0.01914 ± 0.00414 (better) | −0.00405 ± 0.00085 (better) | −0.05505 ± 0.00844 (better) | −0.00801 ± 0.00131 (better) |
| fake | TM-OPT+TC 8x64 | 7.0895 | 10.1241 | −0.02935 ± 0.00465 (better) | −0.00608 ± 0.00107 (better) | −0.06526 ± 0.00911 (better) | −0.01005 ± 0.00151 (better) |
| fake | TM-OPT+TC 16x64 | 7.1007 | 10.1246 | −0.02777 ± 0.00440 (better) | −0.00603 ± 0.00092 (better) | −0.06368 ± 0.00907 (better) | −0.01000 ± 0.00137 (better) |
| fake | TM-OPT+TC 256x64 | 7.1983 | 10.1486 | −0.01412 ± 0.00353 (better) | −0.00366 ± 0.00094 (better) | −0.05003 ± 0.00802 (better) | −0.00763 ± 0.00126 (better) |
| fake | BF16 | 7.0509 | 9.8935 | — | — | — | — |

TM-OPT+TC minus TM-OPT (paired within each process):

| backend | unit | ΔWiki | ΔC4 |
|---|---|---|---|
| native | 8x64 | +0.00456 ± 0.00244 (worse) | +0.00004 ± 0.00081 (n.s.) |
| native | 16x64 | −0.00446 ± 0.00264 (better) | −0.00007 ± 0.00079 (n.s.) |
| native | 256x64 | +0.00455 ± 0.00290 (worse) | −0.00006 ± 0.00088 (n.s.) |
| fake | 8x64 | +0.00281 ± 0.00259 (worse) | −0.00042 ± 0.00069 (n.s.) |
| fake | 16x64 | −0.00460 ± 0.00287 (better) | −0.00079 ± 0.00077 (better) |
| fake | 256x64 | +0.00502 ± 0.00293 (worse) | +0.00039 ± 0.00074 (n.s.) |

- **The unit ordering is the other models':** 8x64 is best, then 16x64, then 256x64. The exception is TC at
  16x64 (WikiText).
- **These numbers are only compared within this table.** `results/task_sensitivity_qwen38` used different C4
  windows and another quantization pipeline.

### Checks

- **Evaluation windows:** 145 WikiText-2 and 256 C4 windows. They are hash-identical to the SM120 evaluation
  reference (`sm120/eval/reference/qwen27b_windows_*.json`) in all three evaluation processes.
- **Maps:** every map in both evaluation processes has its run's sha256 and E0M3 count. The 16x64 and 256x64
  maps run as whole 8x64 units, and their own counts are recorded next to them. No lean or native map
  mismatches.
- **Each run's own final evaluation** (native, in the training process) repeats bitwise in the native evaluation
  process, for all six maps.

Tables: `final_qwen.{json,md}` and the Qwen section of `final_ppl.{json,md}` (`analyze_final.py qwen|ppl`).
Records: `runs/qwen/` (the reports, `queue_q.sh`, `commands_q.log`).

## Part R

In progress. The R2 latency queue (`sm120_bench/queue_latency.sh`, scripts in `latency/`) started at
2026-09-27 07:01 UTC, after Part Q.
