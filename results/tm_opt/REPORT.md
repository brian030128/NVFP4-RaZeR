# TM-OPT verification (Llama-3.1-8B) — report

**Status (2026-09-26 08:15 UTC): complete. Stopped, as the rules require.** No multi-model run starts
without the user's approval.

- **Protocol:** [PROTOCOL.md](PROTOCOL.md), registered 2026-09-26 07:06:31 UTC (sha256 bc3702b3…),
  before any test or run.
- **Deviation 1:** unit test 1(b) failed, with a diagnostic afterwards.
- **Deviation 2:** the user's decision, option A — B1 out of the TM-OPT preset; group 2 is one full
  run.

## Verdict

- **TM-OPT without B1 reproduces the legacy trained-map method bitwise** on this machine (group 1),
  while being faster and lighter:
  - **per epoch:** 35.2 s, against 63.6 s for legacy (1.8× faster);
  - **selection time:** 13.9 min for 20 epochs, against about 27.4 min estimated for legacy;
  - **peak GPU memory:** 40.8 / 42.0 GiB, against 59.0 / 61.6 GiB (allocated / reserved).
- **The full 20-epoch STE 8x64 TM-OPT run** elects 306,968 E0M3 tiles.
  - **Native evaluation:** WikiText-2 6.7822, C4 9.6754. The paired ΔNLL vs FourOverSix is
    −0.01369 ± 0.00192 / −0.01539 ± 0.00290, significantly better on both corpora.
  - **Fake evaluation:** 6.7792 / 9.6734.
  - **Main's H200 run** (fake evaluation, context only) gave 6.7847 / 9.6754 with 329,837 tiles.
- **B1 is not part of TM-OPT** (user decision). `--tile-grad-kernel` stays an opt-in flag, default
  off. The reasons, from the diagnostic after the unit-test failure:
  - it is slower here: the training step needs one batch GEMM per module, so there is nothing to
    fuse away (76 vs 56 ms per batch on the largest shapes);
  - its error is at the FP32 noise level for 4,096-token sums. Against FP64, the legacy hook is up
    to 1.09e-6 off and B1 up to 1.32e-6.

## What TM-OPT is (branch `tm-opt`, `run_train_map.py`)

Main's trained-map method is unchanged: one logit θ per tile, STE or sigmoid, Adam on the KL loss
(ε 1e-12), batch 8, 20 epochs, no backtracking, and a monitor-only development set.

Each optimization sits behind a flag whose default is the legacy behaviour. `--tm-opt` turns on the
TM-OPT preset:
- `--memory-mode lean`: one native-format candidate store; each weight is decoded per call in the
  forward and for the backward.
- `--fused-act-quant`;
- `--chunked-loss`: `chunked_loss.train_kl_gradient`;
- `--deterministic`;
- `--dev-backend native` and `--eval-backend native`, with `--single-pass-epilogue`.

The caller also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Unit tests (`runs/unit_tests.json`)

| test | result |
|---|---|
| 1(a) chunked training loss vs the legacy step (3 vocabularies × batches 1–8) | **PASS**: logits gradient and per-sequence KL bitwise in 24/24 cases |
| 1(b) B1 tile sums vs the legacy hook, normwise ≤ 1e-6 (72 cases) | **FAIL** (reported as such): 1.41e-6 and 1.39e-6 in 2 cases; B1 left out of TM-OPT |

## Group 1: bitwise vs legacy, B1 off — PASS

Legacy (`--deterministic`) against TM-OPT without B1. For comparability, both use the fake monitor
and fake final evaluation.

| run | steps with θ bitwise equal | flips per epoch | final E0M3 tiles | every other recorded value | final fake PPL (both) |
|---|---:|---|---:|---|---|
| STE 8x64, 3 epochs (init −0.2) | 48 / 48 | 141,699 / 672,620 / 518,396 | 980,969 | equal | 6.8259 / 9.7118 |
| sigmoid 8x64, 2 epochs (init −3) | 32 / 32 | 0 / 0 | 0 | equal | 6.8872 / 9.8294 |

- **"Every other recorded value"** means:
  - per epoch: train KL, E0M3 count, flips and τ;
  - the monitor's per-document CE and KL: initial, every epoch and final;
  - the per-epoch and final maps, and the θ file;
  - every WikiText-2 and C4 window NLL.
- **The sigmoid run's hard map never flipped** in 2 epochs. Its soft weights changed at every
  step, and θ was compared after each of those steps.

## Group 2: one full TM-OPT run (STE 8x64, main's settings: lr 0.02, init −1, 20 epochs, monitor every 2)

**Time and memory:**

| run | training s / epoch | selection time | peak GPU allocated / reserved | host RSS |
|---|---:|---:|---:|---:|
| legacy (group 1 STE, 3 epochs, fake monitor) | 63.6 | 5.4 min for 3 epochs | 59.0 / 61.6 GiB | 43.6 GiB |
| legacy, 20-epoch estimate (63.6 s per epoch, 33.9 s per fake monitor evaluation × 11) | 63.6 | ~27.4 min | 59.0 / 61.6 GiB | 43.6 GiB |
| TM-OPT, fake monitor (group 1 STE, 3 epochs) | 34.9 | 2.7 min for 3 epochs | 40.6 / 41.6 GiB | 42.0 GiB |
| **TM-OPT (group 2, 20 epochs, native monitor)** | **35.2** | **13.9 min** | **40.8 / 42.0 GiB** | 42.2 GiB |

- **Where the selection time goes:** 703 s of training, 121 s for 10 native monitor evaluations
  (12.1 s each) and 12 s for the final monitor evaluation.
- **Other costs:** setup took 2.0 min and the final native WikiText-2/C4 evaluation 1.1 min.
- **Peak GPU memory** is set by training; every other phase stays at or below 18.3 GiB.

**Quantized model** (this branch's evaluator, convention (a); FourOverSix and the TM-OPT map in one
native and one fake process):

| map | E0M3 tiles | native WikiText-2 | native C4 | native ΔWiki vs FourOverSix | native ΔC4 vs FourOverSix | fake WikiText-2 | fake C4 | fake ΔWiki | fake ΔC4 |
|---|---:|---:|---:|---|---|---:|---:|---|---|
| FourOverSix | 0 | 6.8757 | 9.8254 | — | — | 6.8872 | 9.8294 | — | — |
| **TM-OPT STE 8x64** | 306,968 | **6.7822** | **9.6754** | −0.01369 ± 0.00192 (better) | −0.01539 ± 0.00290 (better) | 6.7792 | 9.6734 | −0.01581 ± 0.00199 (better) | −0.01599 ± 0.00359 (better) |

**Training curve** (development KL from the native monitor, alongside main's H200 run for context):

| epoch | E0M3 tiles | dev KL | H200 E0M3 tiles | H200 dev KL (fake monitor) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.10811 | 0 | 0.10845 |
| 4 | 1,894 | 0.10024 | 2,072 | 0.09951 |
| 8 | 72,715 | 0.08850 | 83,110 | 0.08907 |
| 12 | 167,760 | 0.08661 | 183,041 | 0.08598 |
| 16 | 241,822 | 0.08434 | 260,629 | 0.08318 |
| 20 | 306,968 | 0.08288 | 329,837 | 0.08349 |

- **Train KL** falls to 0.0403 (main's run: 0.039).
- **The run follows main's H200 trajectory closely.** The E0M3 counts are 7–13 % lower at every
  point from epoch 8, and the final PPL (fake) is 6.7792 / 9.6734 against 6.7847 / 9.6754.
- **This is a reference only.** The GPU and the FP32 summation orders differ, so the path differs
  at noise level.

**Checks:**
- **The run's own native final evaluation repeated bitwise** in the joint process.
- **The FourOverSix native window NLLs repeated the MR-OPT study's evaluation bitwise.**
- **The lean and native map checks at startup:** no mismatches.
- **Source hashes:** every run recorded hashes equal to the registered files, as amended in
  deviation 2.

## Context: same evaluator and windows as the MR-OPT study (not a criterion)

The FourOverSix NLLs repeat across the two processes, so TM-OPT's native window NLLs can be paired
with the MR-OPT study's Llama maps:

| TM-OPT STE 8x64 minus | ΔWiki | ΔC4 |
|---|---|---|
| MR-OPT 8x64 (6.8134 / 9.7644; selection 28.0 min) | −0.00458 ± 0.00158 (better) | −0.00916 ± 0.00323 (better) |
| MR-OPT 256x64 (6.8369 / 9.7594; selection 17.5 min) | −0.00802 ± 0.00169 (better) | −0.00865 ± 0.00239 (better) |

- **This matches main's H200 finding** (§7: −0.00484 ± 0.00142 / −0.00870 ± 0.00240 vs multi-round
  8x64).
- **It is one model and one seed.** The multi-model TM-OPT vs MR-OPT comparison is not started; it
  waits for the user's approval.

## Reproduction

```
python repro_local/realquant/test_tm_opt.py OUT_JSON           # unit tests 1(a), 1(b)
python results/tm_opt/diagnose_b1.py OUT_JSON                  # the B1 diagnostic (deviation 1)
/home/dev/n16k64_campaign/tm_opt/queue2.sh                     # groups 1 and 2 (copy in runs/queue2.sh, log in runs/commands.log)
python results/tm_opt/compare_tm.py bitwise NAME RUN_A RUN_B   # group 1 checks
python results/tm_opt/compare_tm.py e2e RUNS                   # group 2 tables (tables.md, summary.json)
```

Run records (`report.json`) are in `runs/`. The maps, θ files and per-epoch maps stay in
`/home/dev/n16k64_campaign/tm_opt/runs/`.
