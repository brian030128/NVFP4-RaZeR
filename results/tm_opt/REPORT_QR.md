# Part Q (Qwen3.8-27B) and Part R (final four-model analysis) — report

Protocol: `PROTOCOL_QR.md`, registered 2026-09-26T17:52:02Z (sha256 05d84fd17661…, `registration_qr.json`),
before any Part Q or R run. Nothing was selected or tuned on WikiText-2, C4 or zero-shot. The Qwen3.8-27B gate was
open for this task only.

Deviations:
1. The Qwen batch probe.
2. The user's decision that **TM-OPT+TC is the paper's final method**, with identical settings for all models except the
   batch (Qwen micro-batch 2 × accumulation 4, optimizer batch 8). This decision came after Part Q, so Part Q's
   summary compares the two methods neutrally; Part R presents TM-OPT+TC first.
3. R2's benchmark stage was restarted after a script bug, and Qwen decode was not measured.

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

## Part R: final four-model analysis — TM-OPT+TC is the final method

**Final method:** TM-OPT+TC (`run_train_map.py --tm-opt --tile-grad-tc`, deviation 2).
- **Settings:** seed 0, deterministic, STE, lr 0.02, θ init −1, Adam ε 1e-12, 20 epochs of 16 steps, and an
  optimizer batch of 8 sequences on every model.
- **Only per-model difference:** the batch. Qwen3.8-27B runs micro-batch 2 × accumulation 4.

**Comparisons:** TM-OPT, MR-OPT (Llama, Mistral and Phi-4 only), FourOverSix, pure NVFP4 and BF16.
**Models and units:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B, at 8x64, 16x64 and 256x64.

### Summary

- **Accuracy (R3, native evaluation):**
  - **Vs FourOverSix and vs pure NVFP4:** TM-OPT+TC is significantly better in all 24 model × unit × corpus
    cells.
  - **Vs MR-OPT:** 13 better, 5 not significantly different, 0 worse (18 cells).
  - **Vs TM-OPT:** 20 not significantly different, 2 better, 2 worse. The 2 worse are Qwen WikiText-2 at 8x64
    and 256x64 (disclosed below).
  - **Share of FourOverSix's gap to BF16 closed by the 8x64 map** (WikiText-2 / C4): Llama 14 / 16 %,
    Mistral 17 / 17 %, Phi-4 25 / 21 %, Qwen 78 / 20 %.
- **Calibration cost (R1):**
  - **Selection time:** 7.9–9.8 min on the 7–8B models, 15.8 min on Phi-4 and 80 min on Qwen3.8-27B.
  - **Setup:** 1.2–2.1 min, 6.5 min on Qwen.
  - **Vs TM-OPT:** 30–35 % less selection time (11–12 % on Qwen).
  - **Vs MR-OPT:** 44–80 % less.
  - **Peak GPU memory:** the same as TM-OPT: 36–60 GiB, 90 GiB on Qwen.
- **Latency on the SM120 deployment kernels (R2, speed only):**
  - **16x64 and 256x64 maps** (weights on A, `n16k64_wA`; 256x64 as 16x64 granules):
    - prefill: +1.6–2.1 % at 4×2048 vs FourOverSix and NVFP4 (+0.9–1.0 % on Qwen);
    - decode: −0.3 to −0.6 % tokens/s vs FourOverSix.
  - **8x64 maps** (weights on B, `n8k64_wB`):
    - prefill: +4.5–5.7 % at 4×2048 (+2.5–2.8 % on Qwen);
    - decode: −2.0 to −3.1 % vs FourOverSix on the same placement.
    - Weights-on-B has no narrow-tile decode kernels, so its decode is 20–28 % slower than the weights-on-A path.
  - **Mixed FP4 GEMM kernel vs the stock GEMM** on the same placement: +10–11 % (8x64) and +3.6–4.3 %
    (16x64/256x64).
  - **FourOverSix's activation quantizer** (two candidates) costs 6.5–7.2 % more than NVFP4's, at most 0.2 % of the
    prefill.
  - **BF16:** prefill is 2.0–2.1× slower than NVFP4 on the 7–14B models (1.6× on Qwen); decode is 1.7–2.0× slower.
- **Disclosures:**
  - **The TC unit test failed:** 3.9–6.5e-6 error vs FP64, against 0.24–1.45e-6 for the FP32 path
    (`PROTOCOL_TC.md` deviation 1).
  - **Qwen WikiText-2:** TC's maps differ from TM-OPT's by ±0.0046 with mixed signs (Part Q).
  - **Qwen speed-up:** TC is only 12.7 % faster per epoch on Qwen.
  - **R2 is speed only:** the deployment kernel's per-token activation scales and one-rounding epilogue differ
    from the PPL convention (convention (a)).

### R1. Calibration cost

Sources: the committed run records; nothing was re-measured (`final_cost.{json,md}`, `analyze_final.py cost`).
- **Selection:** the 20 epochs plus the monitor evaluations (TM-OPT and TM-OPT+TC), or every round with its
  development evaluations (MR-OPT).
- **Setup:** model load, data load, teacher logits, candidate packing and the initial development evaluation.

| model | unit | TM-OPT+TC per epoch | TM-OPT+TC selection | vs TM-OPT | vs MR-OPT | setup | peak GPU allocated | host RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 8x64 | 22.8 s | 9.8 min | −30 % | −65 % | 1.4 min | 40.8 GiB | 42.3 GiB |
| Llama-3.1-8B | 16x64 | 22.9 s | 9.8 min | −30 % | −78 % | 1.3 min | 40.6 GiB | 42.3 GiB |
| Llama-3.1-8B | 256x64 | 22.8 s | 9.8 min | −30 % | −44 % | 1.4 min | 40.5 GiB | 42.2 GiB |
| Mistral-7B-v0.3 | 8x64 | 19.6 s | 7.9 min | −34 % | −80 % | 1.2 min | 36.2 GiB | 14.7 GiB |
| Mistral-7B-v0.3 | 16x64 | 19.7 s | 8.0 min | −34 % | −57 % | 1.2 min | 36.0 GiB | 14.7 GiB |
| Mistral-7B-v0.3 | 256x64 | 19.6 s | 7.9 min | −35 % | −65 % | 1.2 min | 36.0 GiB | 14.7 GiB |
| Phi-4 | 8x64 | 39.8 s | 15.8 min | −33 % | −56 % | 2.1 min | 59.7 GiB | 33.7 GiB |
| Phi-4 | 16x64 | 39.8 s | 15.8 min | −33 % | −64 % | 2.1 min | 59.4 GiB | 33.7 GiB |
| Phi-4 | 256x64 | 39.8 s | 15.9 min | −33 % | −55 % | 2.1 min | 59.2 GiB | 33.7 GiB |
| Qwen3.8-27B | 8x64 | 216.6 s | 79.8 min | −12 % | — | 6.5 min | 90.4 GiB | 79.5 GiB |
| Qwen3.8-27B | 16x64 | 217.0 s | 79.9 min | −12 % | — | 6.6 min | 89.9 GiB | 79.3 GiB |
| Qwen3.8-27B | 256x64 | 216.5 s | 79.7 min | −11 % | — | 6.6 min | 89.6 GiB | 79.4 GiB |

- **MR-OPT's time grows with its number of rounds** (8–20 scoring passes of 32–67 s each, plus the
  development evaluations). TM-OPT+TC's time is fixed by its 20 epochs.
- **Memory:** TM-OPT+TC's peak equals TM-OPT's. MR-OPT's is within 1.6 GiB of both.
- **Context: QAT C1 on Llama** (`results/cost_comparison`):
  - one epoch of full-weight QAT over 128 calibration sequences, non-deterministic;
  - 26.9 s of training, about one TM-OPT+TC epoch;
  - 85.3 GiB of GPU memory (2.1× TM-OPT+TC's) and 71.5 GiB of host RSS (1.7×).
- **Non-deterministic TM-OPT+TC** (3-epoch probe, Llama 8x64): 20.9 s per epoch against 22.8 s deterministic.
- **The full table**, every method with its setup split, is in `final_cost.md`.

### R2. Evaluation-time latency on the SM120 deployment kernels (speed only)

**Method**

- **Kernels:** the SM120 kernel of `origin/SM120-kernel` f91c109, exported to
  `/home/dev/n16k64_campaign/sm120_bench` (the repository working tree was not touched).
- **Policies** (one process per model, policy and round):
  - BF16;
  - NVFP4 and FourOverSix on `stock_wA` (the deployment placement) and on `stock_wB` (placement-matched to
    8x64), with NVFP4 and FourOverSix activation quantization respectively;
  - the TM-OPT+TC and TM-OPT maps: 8x64 on `n8k64_wB`, 16x64 on `n16k64_wA`, and 256x64 on `n16k64_wA`.
    Each 256-row tile becomes its 16x64 granules; the E0M3 area is checked unchanged, and the exporter checks
    every packed weight against the fake-quant weight.
  - The weights-on-A policies use the kernel sets that choose the tile width per token count: 128-wide
    tiles for every prefill, 16-wide for decode. The call counts are recorded.
- **Rounds:** 5 rounds; the order of the 11 policies is shuffled in every round (seed 20260926, orders in
  `runs/latency/bench/order_*.json`).
- **Prefill:** one forward over random tokens with the KV cache written; 2 warm-ups, then 7 forwards timed
  with CUDA events. Reported: the median of the per-round medians, and overheads paired within rounds.
- **Profiler decomposition:** 3 more forwards (1 on Qwen) under torch.profiler. Every kernel is attributed to the
  innermost module region open at its launch:
  - the FP4 GEMM, whose epilogue fuses the per-token scale, global scale, bias and bf16 rounding;
  - the activation quantizer;
  - the BF16 lm_head;
  - attention;
  - Qwen's linear attention;
  - everything else.
- **Decode:** batch 1, a 512-token prompt, 64 tokens, eager and CUDA-graph. A CUDA-graph result counts only if its
  tokens equal the eager reference (deviation 3).
- **GPU:** RTX PRO 6000 Blackwell, idle (checked before every process), 500 W power cap.

**TM-OPT+TC**

Prefill overheads are paired within rounds (median over 5 rounds). "Same placement" is `stock_wB` for 8x64 and
`stock_wA` otherwise. The FP4 GEMM column is the mixed kernel's GEMM time vs FourOverSix's stock GEMM at 4×2048.

| model | map | 1×2048 vs NVFP4 / FourOverSix | 4×2048 vs NVFP4 / FourOverSix | 4×2048 vs FourOverSix, same placement | FP4 GEMM, same placement | decode tok/s (vs FourOverSix, same placement) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 8x64 (`n8k64_wB`) | +5.9 % / +5.7 % | +5.1 % / +5.1 % | +4.6 % | +11.1 % | 101.3 (−2.9 %) |
| Llama-3.1-8B | 16x64 (`n16k64_wA`) | +2.3 % / +1.7 % | +1.9 % / +1.6 % | +1.6 % | +4.1 % | 136.9 (−0.5 %) |
| Llama-3.1-8B | 256x64 (`n16k64_wA`) | +2.3 % / +1.8 % | +1.9 % / +1.6 % | +1.6 % | +4.3 % | 137.0 (−0.5 %) |
| Mistral-7B-v0.3 | 8x64 (`n8k64_wB`) | +7.2 % / +6.9 % | +5.6 % / +5.4 % | +5.0 % | +11.3 % | 106.5 (−3.1 %) |
| Mistral-7B-v0.3 | 16x64 (`n16k64_wA`) | +2.6 % / +2.5 % | +1.8 % / +1.8 % | +1.8 % | +4.2 % | 147.1 (−0.5 %) |
| Mistral-7B-v0.3 | 256x64 (`n16k64_wA`) | +2.7 % / +2.7 % | +2.0 % / +2.0 % | +2.0 % | +4.3 % | 147.0 (−0.6 %) |
| Phi-4 | 8x64 (`n8k64_wB`) | +6.5 % / +6.1 % | +5.7 % / +5.4 % | +4.5 % | +10.4 % | 76.0 (−2.0 %) |
| Phi-4 | 16x64 (`n16k64_wA`) | +2.2 % / +1.9 % | +1.8 % / +1.6 % | +1.6 % | +3.7 % | 95.0 (−0.3 %) |
| Phi-4 | 256x64 (`n16k64_wA`) | +2.6 % / +2.1 % | +2.1 % / +1.8 % | +1.8 % | +4.0 % | 95.1 (−0.3 %) |
| Qwen3.8-27B | 8x64 (`n8k64_wB`) | (host-bound) | +2.8 % / +2.6 % | +2.5 % | +11.1 % | not measured |
| Qwen3.8-27B | 16x64 (`n16k64_wA`) | (host-bound) | +1.0 % / +0.9 % | +0.9 % | +3.6 % | not measured |
| Qwen3.8-27B | 256x64 (`n16k64_wA`) | (host-bound) | +0.9 % / +0.9 % | +0.9 % | +3.8 % | not measured |

**Absolute numbers** (prefill in ms at 1×2048 / 4×2048; decode is CUDA-graph tokens/s at batch 1):

| model | BF16 | NVFP4 (`stock_wA`) | FourOverSix (`stock_wA`) | TM-OPT+TC 8x64 | TM-OPT+TC 16x64 | TM-OPT+TC 256x64 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 126.1 / 527.7; 80.5 tok/s | 59.7 / 265.2; 136.0 | 59.8 / 265.9; 137.6 | 63.0 / 278.7; 101.3 | 60.9 / 270.0; 136.9 | 61.0 / 270.6; 137.0 |
| Mistral-7B-v0.3 | 120.3 / 504.4; 83.8 | 53.4 / 242.8; 145.9 | 53.6 / 243.3; 147.9 | 57.3 / 256.3; 106.5 | 54.9 / 247.5; 147.1 | 54.9 / 247.7; 147.0 |
| Phi-4 | 232.1 / 985.6; 46.4 | 105.8 / 464.6; 94.4 | 106.2 / 465.5; 95.3 | 112.8 / 490.9; 76.0 | 108.3 / 473.2; 95.0 | 108.5 / 474.0; 95.1 |
| Qwen3.8-27B | 767.1 / 2616.5 | 763.5 / 1640.7 | 754.6 / 1641.4 | 754.6 / 1684.1 | 767.7 / 1655.5 | 757.2 / 1656.3 |

**Decomposition** (NVFP4, 4×2048, share of GPU kernel time):
- **FP4 GEMM:** 40–45 % on the three smaller models; 23 % on Qwen, where the torch implementation of the
  linear attention takes 54 %.
- **Activation quantizer:** 6.6–7.7 % (3 % on Qwen).
- **lm_head:** 3–11 %, depending on the vocabulary.
- **Attention:** 14–16 %.
- **Other:** 27–30 %, mostly the fp32 RMSNorm and the SiLU/multiply kernels.

What changes between policies:
- **The mixed-kernel overhead:** only the FP4 GEMM class changes between a map and FourOverSix on the same
  placement: +10–11 % (8x64) and +3.6–4.3 % (16x64/256x64). On Llama, Mistral and Phi-4, every other class moves
  by less than 0.4 ms (paired medians). Qwen's single-forward profiles scatter by up to 5.5 ms per class.
- **The activation quantization:** FourOverSix's quantizer takes 6.5–7.2 % longer than NVFP4's (Llama at 4×2048: +1.2
  ms on 18.7 ms). That is +0.05–0.2 % of the prefill at 4×2048.
- **TM-OPT and TM-OPT+TC have the same latency** within 0.6 ms at 4×2048 on every model (Llama: 279.1 vs 278.7 ms
  at 8x64, 269.9 vs 270.0 ms at 16x64). Their E0M3 counts differ by 0.1–9 %, and item #3 found the E0M3 density
  worth at most 0.7 % of the prefill.

**Caveats**
- **1×512 cannot resolve map effects:** each process lands at about 30 or 36 ms (Llama), whatever the policy,
  as in item #3. That prefill is host-bound; the numbers are in `latency.md`.
- **Qwen3.8-27B:**
  - **Host-bound at 1×512 and 1×2048:** BF16 is as fast as NVFP4 there, because the Python loops of the torch
    linear-attention fallback dominate. Only 4×2048 is GPU-bound.
  - **Its per-class times are less precise:** they come from one profiled forward per round, and they depend on
    the power state of that round (kernels speed up by up to 9 % in rounds with host stalls). The paired medians
    in the summary table are used, and the end-to-end 4×2048 timings stay tight (NVFP4 1638–1642 ms).
  - **Decode is not measured:** the SM120 decode functions do not support the Qwen3.5 hybrid cache
    (deviation 3).
- **Decode placement:** 8x64 decode runs on 128-wide token tiles (no narrow weights-on-B builds exist).
  Its −2 to −3 % is relative to `stock_wB`, which has the same limitation.
- **Numerics:** the deployment kernel uses per-token activation scales and one rounding in the epilogue. R2
  measures speed only; the accuracy numbers come from R3's convention.

Checks:
- **Narrow-tile builds:** all six built. The three mixed ones match their expected SASS census and pass their
  self-tests (the stock ones have no format granule to test). `tests/test_select.py` passes 12/12 with none
  skipped.
- **Exports:** 32 exports, every packed weight equal to its fake-quant weight.
- **Coverage:** in every process, all 224 / 224 / 160 / 496 scoped Linears ran natively, with no fallback and
  E0M3 counts equal to the maps'.
- **Tile widths:** 128-wide for every prefill, 16-wide for every decode.
- **Decode tokens:** every Llama, Mistral and Phi-4 CUDA-graph decode matches its eager reference.

### R3. PPL

WikiText-2 and C4 under convention (a). Native evaluation is primary; paired ΔNLL per window, mean ± 2 SE.
- **Sources:** the committed evaluations and Part Q's.
- **Pairing across processes:** only where FourOverSix's window NLLs repeat bitwise (true for every merged
  process).
- **The full tables** (native and fake, every method, BF16) are in `final_ppl.md` (`analyze_final.py ppl`).

TM-OPT+TC minus each comparison, native (WikiText-2; C4):

| model | unit | TM-OPT+TC PPL | vs FourOverSix | vs NVFP4 | vs TM-OPT | vs MR-OPT |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 8x64 | 6.7827 / 9.6795 | −0.01362 ± 0.00177; −0.01496 ± 0.00284 | −0.02237 ± 0.00226; −0.02534 ± 0.00411 | +0.00007 ± 0.00143 (n.s.); +0.00043 ± 0.00103 (n.s.) | −0.00451 ± 0.00158; −0.00873 ± 0.00323 |
| Llama-3.1-8B | 16x64 | 6.7865 / 9.6863 | −0.01305 ± 0.00178; −0.01426 ± 0.00280 | −0.02180 ± 0.00229; −0.02464 ± 0.00407 | −0.00062 ± 0.00144 (n.s.); +0.00051 ± 0.00103 (n.s.) | −0.00578 ± 0.00171; −0.00565 ± 0.00194 |
| Llama-3.1-8B | 256x64 | 6.8012 / 9.7228 | −0.01090 ± 0.00176; −0.01050 ± 0.00263 | −0.01965 ± 0.00221; −0.02088 ± 0.00381 | +0.00081 ± 0.00152 (n.s.); −0.00027 ± 0.00124 (n.s.) | −0.00524 ± 0.00171; −0.00376 ± 0.00210 |
| Mistral-7B-v0.3 | 8x64 | 5.4862 / 8.0261 | −0.00659 ± 0.00108; −0.00495 ± 0.00079 | −0.01250 ± 0.00114; −0.00862 ± 0.00094 | +0.00003 ± 0.00085 (n.s.); −0.00015 ± 0.00066 (n.s.) | +0.00045 ± 0.00091 (n.s.); −0.00091 ± 0.00065 |
| Mistral-7B-v0.3 | 16x64 | 5.4951 / 8.0264 | −0.00497 ± 0.00225; −0.00492 ± 0.00122 | −0.01088 ± 0.00232; −0.00860 ± 0.00109 | +0.00066 ± 0.00229 (n.s.); −0.00026 ± 0.00142 (n.s.) | −0.00074 ± 0.00236 (n.s.); −0.00093 ± 0.00121 (n.s.) |
| Mistral-7B-v0.3 | 256x64 | 5.4899 / 8.0326 | −0.00591 ± 0.00110; −0.00415 ± 0.00082 | −0.01182 ± 0.00121; −0.00782 ± 0.00097 | −0.00104 ± 0.00096 (better); +0.00010 ± 0.00071 (n.s.) | −0.00093 ± 0.00092; −0.00036 ± 0.00078 (n.s.) |
| Phi-4 | 8x64 | 6.6128 / 10.4954 | −0.00784 ± 0.00155; −0.00477 ± 0.00086 | −0.01378 ± 0.00196; −0.00865 ± 0.00111 | +0.00059 ± 0.00120 (n.s.); −0.00043 ± 0.00071 (n.s.) | −0.00369 ± 0.00143; −0.00249 ± 0.00078 |
| Phi-4 | 16x64 | 6.6161 / 10.5034 | −0.00735 ± 0.00152; −0.00402 ± 0.00083 | −0.01329 ± 0.00196; −0.00790 ± 0.00107 | +0.00080 ± 0.00116 (n.s.); +0.00009 ± 0.00073 (n.s.) | −0.00193 ± 0.00145; −0.00076 ± 0.00078 (n.s.) |
| Phi-4 | 256x64 | 6.6308 / 10.5137 | −0.00512 ± 0.00141; −0.00303 ± 0.00083 | −0.01106 ± 0.00182; −0.00691 ± 0.00107 | +0.00039 ± 0.00124 (n.s.); +0.00006 ± 0.00076 (n.s.) | −0.00317 ± 0.00140; −0.00095 ± 0.00078 |
| Qwen3.8-27B | 8x64 | 7.1062 / 10.1261 | −0.02816 ± 0.00476; −0.00570 ± 0.00095 | −0.06412 ± 0.00859; −0.00928 ± 0.00131 | **+0.00456 ± 0.00244 (worse)**; +0.00004 ± 0.00081 (n.s.) | — |
| Qwen3.8-27B | 16x64 | 7.1144 / 10.1335 | −0.02701 ± 0.00498; −0.00497 ± 0.00091 | −0.06297 ± 0.00934; −0.00856 ± 0.00130 | −0.00446 ± 0.00264 (better); −0.00007 ± 0.00079 (n.s.) | — |
| Qwen3.8-27B | 256x64 | 7.1951 / 10.1482 | −0.01573 ± 0.00366; −0.00352 ± 0.00076 | −0.05169 ± 0.00772; −0.00711 ± 0.00111 | **+0.00455 ± 0.00290 (worse)**; −0.00006 ± 0.00088 (n.s.) | — |

Unmarked differences are significantly better (mean + 2 SE < 0).

**Verdict counts** (native, models × units × corpora):

| TM-OPT+TC vs | better | not significant | worse |
|---|---:|---:|---:|
| FourOverSix | 24 | 0 | 0 |
| NVFP4 | 24 | 0 | 0 |
| MR-OPT | 13 | 5 | 0 |
| TM-OPT | 2 | 20 | 2 |

Reference PPLs (WikiText-2 / C4):

| model | BF16 | FourOverSix (native) | NVFP4 (native) |
|---|---|---|---|
| Llama-3.1-8B | 6.2403 / 8.9579 | 6.8757 / 9.8254 | 6.9361 / 9.9279 |
| Mistral-7B-v0.3 | 5.3182 / 7.8306 | 5.5225 / 8.0660 | 5.5552 / 8.0957 |
| Phi-4 | 6.4615 / 10.3098 | 6.6649 / 10.5456 | 6.7046 / 10.5866 |
| Qwen3.8-27B | 7.0509 / 9.8935 | 7.3092 / 10.1840 | 7.5768 / 10.2206 |

- **Unit ordering:** 8x64 has the lowest PPL on every model and corpus. 256x64 has the highest, except Mistral
  WikiText-2 (256x64 5.4899, 16x64 5.4951). Units were not tested against each other.
- **Share of FourOverSix's gap to BF16 closed** by TM-OPT+TC (WikiText-2 / C4):

  | model | 8x64 | 16x64 | 256x64 |
  |---|---|---|---|
  | Llama | 14 / 16 % | 13 / 15 % | 11 / 11 % |
  | Mistral | 17 / 17 % | 13 / 17 % | 16 / 14 % |
  | Phi-4 | 25 / 21 % | 24 / 18 % | 17 / 13 % |
  | Qwen | 78 / 20 % | 75 / 17 % | 44 / 12 % |

  Qwen's WikiText-2 gap is small to begin with (0.036 NLL, against 0.097 on Llama).
- **Fake evaluation** gives the same picture (`final_ppl.md`).

### Records

- **Part R tables:**
  - `final_cost.{json,md}`, `final_ppl.{json,md}` (`analyze_final.py`);
  - `latency/latency.{json,md}` (`latency/analyze_latency.py`).
- **R2 scripts:**
  - `latency/convert_maps.py`, `latency/bench_latency.py`;
  - the queues `runs/latency/queue_latency.sh` and `queue_latency_bench.sh`.
- **R2 records:** `runs/latency/`:
  - every per-process result and the policy orders;
  - map provenance and artifact metadata (no weights);
  - the narrow-build manifests;
  - logs and `commands_r2.log`.
- **Part Q records:** `runs/qwen/`.
