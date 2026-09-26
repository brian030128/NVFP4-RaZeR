# TM-OPT: seed variance, three models, and the GEMM cost of E0M3-dense maps — report

**Status (2026-09-26 13:40 UTC): #2, #1 and #3 are done. Stopped, as instructed.**

**Protocol.** [PROTOCOL_ITEMS.md](PROTOCOL_ITEMS.md), registered 2026-09-26 08:57:46 UTC (sha256
97cacaf8…), before any run.
- **The study is descriptive.** Nothing is selected on WikiText-2, C4 or zero-shot.
- **#3's scripts** were recorded before any #3 step (`registration_items_bench.json`).

## #2 Seed variance (Llama-3.1-8B, TM-OPT STE 8x64)

**Setup.** Seeds 1 and 2 are identical to the verified group-2 run (seed 0) except the calibration
batch order. The evaluation is native (primary) and fake, with FourOverSix, the three seeds and
MR-OPT 8x64 in one native process.
- **Checks passed:** the FourOverSix and MR-OPT 8x64 window NLLs repeated the committed evaluations
  bitwise (native and fake), and seed 0 repeated its own group-2 evaluation.

| seed | E0M3 tiles | final dev KL | native WikiText-2 | native C4 | fake WikiText-2 | fake C4 | native ΔWiki vs MR-OPT 8x64 | native ΔC4 vs MR-OPT 8x64 | Jaccard with MR-OPT |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0 | 306,968 | 0.08288 | 6.7822 | 9.6754 | 6.7792 | 9.6734 | −0.00458 ± 0.00158 (better) | −0.00916 ± 0.00323 (better) | 0.004 |
| 1 | 307,600 | 0.08411 | 6.7791 | 9.6719 | 6.7806 | 9.6759 | −0.00505 ± 0.00173 (better) | −0.00952 ± 0.00351 (better) | 0.004 |
| 2 | 308,688 | 0.08388 | 6.7748 | 9.6755 | 6.7760 | 9.6771 | −0.00567 ± 0.00154 (better) | −0.00915 ± 0.00338 (better) | 0.004 |

For reference, MR-OPT 8x64 scores 6.8134 / 9.7644 and FourOverSix 6.8757 / 9.8254 (native).

- **Every seed is significantly better than MR-OPT 8x64 on both corpora**, native and fake. In the
  fake evaluation the ΔNLL is −0.0055 to −0.0062 on WikiText-2 and −0.0086 to −0.0090 on C4.
- **Every seed is significantly better than FourOverSix:** native −0.0137 to −0.0148 on WikiText-2
  and −0.0154 to −0.0158 on C4.
- **The two methods pick mostly different tiles.** Each TM-OPT map contains only about 1,170 of
  MR-OPT's 3,801 tiles (31 %). The Jaccard index is tiny because TM-OPT elects 81× more tiles.
- **Time and memory per seed:** selection 13.9 min; 35.0–35.2 s per epoch; setup 1.3–2.0 min; peak
  GPU 40.8 / 42.0 GiB; host 42.2 GiB.

**Seed vs seed** (native paired ΔNLL, first minus second):

| seeds | ΔWiki | ΔC4 | Jaccard | tiles shared / only first / only second |
|---|---|---|---:|---|
| 0 − 1 | +0.00046 ± 0.00157 (n.s.) | +0.00036 ± 0.00108 (n.s.) | 0.446 | 189,414 / 117,554 / 118,186 |
| 0 − 2 | +0.00109 ± 0.00138 (n.s.) | −0.00001 ± 0.00107 (n.s.) | 0.438 | 187,586 / 119,382 / 121,102 |
| 1 − 2 | +0.00062 ± 0.00155 (n.s.) | −0.00038 ± 0.00097 (n.s.) | 0.441 | 188,705 / 118,895 / 119,983 |

- **The seeds' maps overlap only by ~44 %** (Jaccard), and about 118k–121k tiles differ per pair.
  Yet no pair differs significantly on either corpus.
- **Many tile sets of about 307k tiles do equally well.** The tile identity is not stable across
  seeds; the quality is.

**Is the seed spread small relative to the TM-OPT − MR-OPT gap?** Yes, on both corpora, by the
registered rule (spread < |gap| / 2):

| corpus | gap: mean over seeds of ΔNLL vs MR-OPT | seed spread (range of the three means) | small | every seed significantly better |
|---|---:|---:|---|---|
| WikiText-2 | −0.00510 | 0.00109 | **yes** | yes |
| C4 | −0.00927 | 0.00038 | **yes** | yes |

**Is the local-vs-H200 tile-count difference within the seed spread? Only through epoch 6; from
epoch 8 on it is not.**

| epoch | H200 E0M3 tiles | local seeds 0 / 1 / 2 | H200 inside the local range | relative seed spread | seed 0 vs H200 |
|---:|---:|---|---|---:|---:|
| 4 | 2,072 | 1,894 / 1,805 / 2,123 | yes | 16.4 % | −8.6 % |
| 6 | 24,585 | 22,876 / 21,767 / 26,053 | yes | 18.2 % | −7.0 % |
| 8 | 83,110 | 72,715 / 70,347 / 78,151 | no | 10.6 % | −12.5 % |
| 12 | 183,041 | 167,760 / 164,601 / 169,079 | no | 2.7 % | −8.3 % |
| 16 | 260,629 | 241,822 / 238,717 / 242,855 | no | 1.7 % | −7.2 % |
| 20 | 329,837 | 306,968 / 307,600 / 308,688 | no | 0.6 % | −6.9 % |

- **The difference is systematic, not seed noise.** The three local seeds converge to within 0.6 %
  of each other by epoch 20, while all stay about 7 % below the H200 run.
- **Probable causes, not verified here:**
  - the H200 run used the calibration record's transformers version, while ours runs 5.16.1
    (a recorded deviation);
  - the GPU's floating-point reductions differ.
- **The PPL gap is tiny:** local fake 6.776–6.781 / 9.673–9.677 against H200 6.7847 / 9.6754.

**Development KL (native monitor) per seed:**

| epoch | seed 0 | seed 1 | seed 2 |
|---:|---:|---:|---:|
| 0 | 0.10811 | 0.10811 | 0.10811 |
| 4 | 0.10024 | 0.09828 | 0.09918 |
| 8 | 0.08850 | 0.08897 | 0.08825 |
| 12 | 0.08661 | 0.08573 | 0.08562 |
| 16 | 0.08434 | 0.08537 | 0.08432 |
| 20 | 0.08288 | 0.08411 | 0.08388 |

The table is in `items_seeds.md`; everything is in `items_seeds.json`.

## #1 TM-OPT vs MR-OPT on three models

**Setup.**
- **TM-OPT:** STE, the TM-OPT preset, main's hyperparameters (lr 0.02, init −1, 20 epochs, batch
  8), deterministic.
- **The comparison:** MR-OPT's committed maps (`results/mr_variants`), not rerun.
- **Evaluation, one native and one fake process per model:**
  - native: FourOverSix, the three TM-OPT maps and the three MR-OPT maps;
  - fake: FourOverSix and the TM-OPT maps. Its MR-OPT side comes from the committed fake NLLs.
- **Checks passed, all three models:**
  - the FourOverSix window NLLs repeated bitwise, native and fake;
  - every MR-OPT map's native window NLLs repeated the committed evaluations bitwise.

**TM-OPT minus MR-OPT, per model × unit × corpus** (paired ΔNLL, mean ± 2 SE; "better" / "worse"
= significantly, "n.s." = not significantly different):

| model | unit | native ΔWiki | native ΔC4 | fake ΔWiki | fake ΔC4 |
|---|---|---|---|---|---|
| Llama-3.1-8B | 8x64 | −0.00458 ± 0.00158 (better) | −0.00916 ± 0.00323 (better) | −0.00568 ± 0.00156 (better) | −0.00897 ± 0.00325 (better) |
| | 16x64 | −0.00516 ± 0.00168 (better) | −0.00616 ± 0.00189 (better) | −0.00544 ± 0.00154 (better) | −0.00692 ± 0.00185 (better) |
| | 256x64 | −0.00604 ± 0.00164 (better) | −0.00349 ± 0.00166 (better) | −0.00439 ± 0.00163 (better) | −0.00588 ± 0.00247 (better) |
| Mistral-7B-v0.3 | 8x64 | +0.00041 ± 0.00092 (n.s.) | −0.00075 ± 0.00071 (better) | −0.00034 ± 0.00091 (n.s.) | −0.00024 ± 0.00076 (n.s.) |
| | 16x64 | −0.00140 ± 0.00089 (better) | −0.00066 ± 0.00085 (n.s.) | −0.00217 ± 0.00092 (better) | −0.00113 ± 0.00087 (better) |
| | 256x64 | +0.00012 ± 0.00089 (n.s.) | −0.00046 ± 0.00075 (n.s.) | +0.00001 ± 0.00098 (n.s.) | −0.00044 ± 0.00069 (n.s.) |
| Phi-4 | 8x64 | −0.00428 ± 0.00126 (better) | −0.00206 ± 0.00072 (better) | −0.00371 ± 0.00125 (better) | −0.00120 ± 0.00076 (better) |
| | 16x64 | −0.00273 ± 0.00131 (better) | −0.00084 ± 0.00076 (better) | −0.00316 ± 0.00159 (better) | −0.00110 ± 0.00077 (better) |
| | 256x64 | −0.00356 ± 0.00131 (better) | −0.00101 ± 0.00078 (better) | −0.00288 ± 0.00131 (better) | −0.00175 ± 0.00078 (better) |

- **Native (primary): 14 of 18 comparisons significantly better, 4 not significantly different,
  0 worse.**
- **Llama and Phi-4:** better at every unit on both corpora.
- **Mistral:** better on WikiText-2 at 16x64 and on C4 at 8x64 (a marginal margin). The other four
  are n.s. On Mistral the two methods reach about the same gain over FourOverSix.
- **The fake evaluation gives 14 better, 4 n.s. and 0 worse** (Mistral: better only at 16x64).

**Summary per model × unit:**

| model | unit | method | E0M3 tiles | native WikiText-2 | native C4 | final dev KL | selection | setup | peak GPU allocated / reserved | host RSS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 8x64 | MR-OPT | 3,801 | 6.8134 | 9.7644 | 0.09441 | 28.0 min | 1.6 min | 41.8 / 43.3 GiB | 42.2 GiB |
| | | TM-OPT | 306,968 | 6.7822 | 9.6754 | 0.08288 | 13.9 min | 2.0 min | 40.8 / 42.0 GiB | 42.2 GiB |
| | 16x64 | MR-OPT | 4,995 | 6.8259 | 9.7412 | 0.09264 | 45.4 min | 1.7 min | 41.2 / 42.4 GiB | 42.2 GiB |
| | | TM-OPT | 200,450 | 6.7907 | 9.6813 | 0.08450 | 14.0 min | 1.4 min | 40.6 / 41.8 GiB | 42.3 GiB |
| | 256x64 | MR-OPT | 8,385 | 6.8369 | 9.7594 | 0.09399 | 17.5 min | 1.7 min | 40.7 / 41.6 GiB | 42.3 GiB |
| | | TM-OPT | 35,346 | 6.7957 | 9.7254 | 0.08828 | 14.0 min | 1.4 min | 40.5 / 41.7 GiB | 42.2 GiB |
| Mistral-7B-v0.3 | 8x64 | MR-OPT | 123,180 | 5.4837 | 8.0334 | 0.02764 | 40.5 min | 1.5 min | 37.0 / 38.2 GiB | 14.7 GiB |
| | | TM-OPT | 340,045 | 5.4860 | 8.0274 | 0.02680 | 12.1 min | 1.2 min | 36.2 / 37.0 GiB | 14.7 GiB |
| | 16x64 | MR-OPT | 127,060 | 5.4992 | 8.0338 | 0.02863 | 18.5 min | 1.5 min | 36.4 / 37.5 GiB | 14.7 GiB |
| | | TM-OPT | 223,845 | 5.4915 | 8.0285 | 0.02689 | 12.0 min | 1.2 min | 36.0 / 36.9 GiB | 14.7 GiB |
| | 256x64 | MR-OPT | 22,082 | 5.4950 | 8.0354 | 0.02819 | 22.4 min | 1.4 min | 35.8 / 36.8 GiB | 14.7 GiB |
| | | TM-OPT | 42,615 | 5.4957 | 8.0317 | 0.02810 | 12.0 min | 1.2 min | 36.0 / 36.7 GiB | 14.7 GiB |
| Phi-4 | 8x64 | MR-OPT | 57,098 | 6.6373 | 10.5216 | 0.03481 | 36.0 min | 2.6 min | 61.3 / 64.2 GiB | 33.6 GiB |
| | | TM-OPT | 423,066 | 6.6089 | 10.4999 | 0.03469 | 23.6 min | 2.1 min | 59.7 / 63.4 GiB | 33.7 GiB |
| | 16x64 | MR-OPT | 117,122 | 6.6288 | 10.5113 | 0.03473 | 43.5 min | 2.6 min | 60.1 / 63.0 GiB | 33.7 GiB |
| | | TM-OPT | 281,702 | 6.6108 | 10.5024 | 0.03520 | 23.6 min | 2.1 min | 59.4 / 63.4 GiB | 33.7 GiB |
| | 256x64 | MR-OPT | 59,953 | 6.6519 | 10.5237 | 0.03492 | 35.3 min | 2.6 min | 59.0 / 61.7 GiB | 33.6 GiB |
| | | TM-OPT | 52,772 | 6.6282 | 10.5131 | 0.03499 | 23.6 min | 2.1 min | 59.2 / 63.2 GiB | 33.7 GiB |

FourOverSix (native): Llama 6.8757 / 9.8254, Mistral 5.5225 / 8.0660, Phi-4 6.6649 / 10.5456. Both
methods are significantly better than FourOverSix in every cell.

- **Selection time.** TM-OPT's is fixed per model (20 epochs of 35.2 / 32.0 / 63.2 s plus 11
  monitor evaluations): 14.0 min for Llama, 12.0 min for Mistral and 23.6 min for Phi-4. MR-OPT's
  depends on the unit and the model: 17.5–45.4 min. **TM-OPT is faster in all nine cells**, by
  1.25× (Llama 256x64) to 3.3× (Mistral 8x64).
- **Memory.** Peak GPU and host memory are about equal; both methods use the same lean
  optimizations.
- **Tiles.**
  - **Larger at the fine units:** TM-OPT elects many more E0M3 tiles at 8x64 and 16x64 (Llama 81× /
    40×, Mistral 2.8× / 1.8×, Phi-4 7.4× / 2.4×).
  - **At 256x64:** 4.2× on Llama, 1.9× on Mistral, and 12 % fewer on Phi-4.
  - **Overlap:** the two methods' maps overlap little (Jaccard 0.004–0.24).
- **The final development KL does not predict the ranking on Phi-4.** There TM-OPT's final
  (monitor) dev KL is equal or higher at 16x64 and 256x64, yet TM-OPT is better on both test
  corpora. This fits the earlier finding that MR-OPT's accepted steps can overfit the development
  set on Phi-4.
- **For reference only:** main's H200 STE 256x64 Llama (fake) gave 6.8055 / 9.7235; this run's fake
  evaluation gives 6.8016 / 9.7164.

Tables: `items_models.md`; everything: `items_models.json`.

## #3 GEMM and latency cost of E0M3-dense maps (speed only)

**Verdict: E0M3 density costs almost nothing.** At full-model prefill (1×2048 and 4×2048), TM-OPT's
maps are within +0.7 % of MR-OPT's on Llama and Phi-4, at both units. That holds although TM-OPT
has up to 81× more E0M3 tiles (2.3–2.9 % of Llama's tiles).

The cost that remains belongs to the mixed kernel itself and does not depend on the map:
- +4.2–5.0 % over the stock NVFP4 kernel at 8x64 (weights on B);
- +1.3–3.0 % at 16x64 (weights on A, the deployment configuration).

**Numerics caveat.** The deployment kernel uses per-token activation scales and a one-rounding
epilogue. This study's development and evaluation convention differs (per-document or per-window
scales, two roundings). #3 measures speed only.

**Kernel and checks.**
- **Source:** the SM120 subtree of `origin/SM120-kernel` (f91c109), exported with `git archive` to
  `/home/dev/n16k64_campaign/sm120_bench` (the working tree was not touched). CUTLASS was cloned
  there at the pinned commit e64a913 from the local mixfp4 copy; CUDA 13.1.115 (the `mixfp4-cuda131`
  env), host g++ 11.4.
- **Built configurations:** `n16k64_wA`, `n8k64_wB`, `stock_wA`, `stock_wB`. Every build passed
  `build.py`'s patch verification:
  - OMMA census 512 + 512;
  - no predicated OMMA;
  - 168 registers, no stack.

  The manifests are in `runs/sm120_bench/`.
- **Self-tests** (`--selftest`) of both mixed kernels on this GPU: **PASS** (the patched kernel
  passes, the unpatched one fails as it must).
- **`sm120/tests/test_gemm.py`:** 83 passed, 194 skipped (only the configurations that were not
  built).
- **Maps:** our maps were converted to `MIXFP4MAP/1` files with the tiles unchanged (round-trip
  checked), and the artifacts were exported by `sm120/eval/export_artifact.py`, which checks every
  packed weight against the fake-quant weight.
- **Coverage:** in every prefill policy, all 224 (Llama) or 160 (Phi-4) Linears ran natively.
- **Environment:** RTX PRO 6000 Blackwell, idle (`require_idle`). There is no per-GPU tile table
  for this GPU, so explicit kernel configurations were used (128-wide token tiles). No small-T
  (decode) kernels were built; the smallest T measured is 128.

### Full-model prefill (ms; registered: one process per policy, median of 5 after 2 warm-ups)

| model | unit | prefill | stock NVFP4 | mixed, FourOverSix | MR-OPT | TM-OPT | TM-OPT vs MR-OPT |
|---|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 8x64 | 1×2048 | 59.65 | 62.42 (+4.6 %) | 62.39 (+4.6 %) | 62.39 (+4.6 %) | +0.0 % |
| | | 4×2048 | 265.81 | 277.14 (+4.3 %) | 276.86 (+4.2 %) | 277.20 (+4.3 %) | +0.1 % |
| | 16x64 | 1×2048 | 58.58 | 60.34 (+3.0 %) | 59.65 (+1.8 %) | 59.88 (+2.2 %) | +0.4 % |
| | | 4×2048 | 264.06 | 267.70 (+1.4 %) | 267.81 (+1.4 %) | 267.99 (+1.5 %) | +0.1 % |
| Phi-4 | 8x64 | 1×2048 | 107.27 | 112.65 (+5.0 %) | 112.33 (+4.7 %) | 112.37 (+4.7 %) | +0.0 % |
| | | 4×2048 | 469.97 | 490.68 (+4.4 %) | 489.65 (+4.2 %) | 489.79 (+4.2 %) | +0.0 % |
| | 16x64 | 1×2048 | 105.79 | 107.79 (+1.9 %) | 107.56 (+1.7 %) | 107.71 (+1.8 %) | +0.1 % |
| | | 4×2048 | 463.78 | 471.17 (+1.6 %) | 470.69 (+1.5 %) | 470.90 (+1.5 %) | +0.0 % |

**The registered 1×512 prefill cannot resolve map effects.** Its results jump between processes,
about 30 ms or about 36 ms whatever the policy (for example, Llama stock 16x64: 35.9 ms; mixed
FourOverSix 16x64: 30.1 ms). The tables in `gemm/tables.md` include them for completeness; the
diagnostic below confirms that this is process-to-process, not map-dependent.

### GEMM only (registered: CUPTI kernel time, median; `gemm/tables.md` has every shape and T)

Summed over one module per projection. For MR-OPT and TM-OPT this is each map's densest module per
projection, their worst case. Shares of stock NVFP4 time:

| model | unit | T | mixed, all E2M1 | MR-OPT | TM-OPT | all E0M3 | TM-OPT vs MR-OPT |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 8x64 | 512 | +9.3 % | +9.5 % | +10.6 % | +12.5 % | +1.0 % |
| | | 2048 | +10.6 % | +11.7 % | +20.2 % | +21.7 % | +7.6 % |
| | | 8192 | +42.8 % | +41.9 % | +44.9 % | +45.0 % | +2.1 % |
| | 16x64 | 512 | +2.7 % | +3.2 % | +3.7 % | +10.7 % | +0.5 % |
| | | 2048 | +4.3 % | +4.8 % | +12.5 % | +16.4 % | +7.4 % |
| | | 8192 | +33.9 % | +33.5 % | +35.0 % | +40.1 % | +1.1 % |
| Phi-4 | 8x64 | 512 | +10.5 % | +11.9 % | +14.9 % | +17.6 % | +2.7 % |
| | | 2048 | +26.9 % | +35.2 % | +36.4 % | +39.2 % | +0.9 % |
| | | 8192 | +40.9 % | +43.4 % | +44.4 % | +47.2 % | +0.7 % |
| | 16x64 | 512 | +3.4 % | +3.4 % | +5.4 % | +9.5 % | +1.9 % |
| | | 2048 | +18.4 % | +26.9 % | +28.2 % | +33.7 % | +1.1 % |
| | | 8192 | +31.1 % | +34.6 % | +34.5 % | +40.4 % | −0.1 % |

**These registered GEMM overheads at T ≥ 2048 are inflated by the measurement order** (diagnostic
below). The benchmark always measured the stock kernel first and then the mixed kernels, back to
back, while the GPU was running into its 500 W software power cap.

### Diagnostic (protocol deviation 1; not a replacement of the registered numbers)

**GEMM:** three timing methods on the same operands, in shuffled order over 5 rounds, with GPU
telemetry. Llama shapes; overhead vs stock:

| unit | shape | T | isolated call: all E2M1 / TM-OPT | chained calls: all E2M1 / TM-OPT | CUPTI: all E2M1 / TM-OPT |
|---|---|---:|---|---|---|
| 8x64 | q_proj | 2048 | +9.2 / +10.1 % | +9.3 / +10.3 % | +9.8 / +10.8 % |
| | | 8192 | +9.1 / +10.1 % | +11.4 / +12.1 % | +11.6 / +12.6 % |
| | gate_proj | 2048 | +6.9 / +7.9 % | +9.4 / +10.4 % | +9.6 / +10.7 % |
| | | 8192 | +9.4 / +10.7 % | +18.7 / +20.5 % | +4.3 / +5.5 % |
| | down_proj | 2048 | +6.6 / +7.7 % | +10.0 / +11.8 % | +10.8 / +12.2 % |
| | | 8192 | +9.2 / +11.1 % | +20.4 / +24.3 % | +7.3 / +9.2 % |
| 16x64 | q_proj | 2048 | +4.2 / +4.9 % | +3.5 / +3.9 % | +3.3 / +3.7 % |
| | | 8192 | +2.9 / +3.5 % | +3.7 / +4.0 % | +3.2 / +3.7 % |
| | gate_proj | 2048 | +2.6 / +3.2 % | +3.7 / +4.3 % | +3.7 / +4.4 % |
| | | 8192 | +3.5 / +4.1 % | +6.0 / +6.9 % | +3.7 / +2.1 % |
| | down_proj | 2048 | +1.7 / +2.4 % | +3.1 / +4.2 % | +3.5 / +4.5 % |
| | | 8192 | +2.9 / +3.7 % | +4.8 / +7.1 % | +2.8 / +3.7 % |

- **The TM-OPT pattern on the densest module** costs only 0.6–1.9 percentage points more than all
  E2M1 (isolated calls).
- **The mixed kernel itself (all E2M1) costs:**
  - at 8x64: +6.6–9.4 % (isolated);
  - at 16x64: +1.7–4.2 %.
- **The registered +31–45 % at T = 8192 does not reproduce once the order is shuffled.**
- **The power cap is active often:** the GPU reported it after 84 of 180 measurements. The SM clock
  ranged from 1,890 to 2,790 MHz, and the power reached 406 W.
- **Consequences for the timing methods:**
  - **Chained timing and CUPTI depend on that state at large T.** CUPTI's duration of chained
    launches also includes time spent at the dependent-launch barrier: for example, stock
    gate_proj at T = 8192 measures 985 µs by CUPTI against 790 µs isolated.
  - **Isolated calls are the most stable measure;** they agree with the full-model prefill
    overheads.

**Prefill:** the eight Llama processes, run twice more in alternating orders. Median of the three
processes:

| unit | prefill | stock NVFP4 | mixed, FourOverSix | MR-OPT | TM-OPT | TM-OPT vs MR-OPT |
|---|---|---:|---:|---:|---:|---:|
| 8x64 | 1×2048 | 59.65 | 62.43 (+4.7 %) | 62.39 (+4.6 %) | 62.57 (+4.9 %) | +0.3 % |
| | 4×2048 | 265.81 | 277.14 (+4.3 %) | 276.86 (+4.2 %) | 277.20 (+4.3 %) | +0.1 % |
| 16x64 | 1×2048 | 58.89 | 59.95 (+1.8 %) | 59.66 (+1.3 %) | 60.06 (+2.0 %) | +0.7 % |
| | 4×2048 | 264.06 | 268.01 (+1.5 %) | 267.81 (+1.4 %) | 268.02 (+1.5 %) | +0.1 % |

- **At 1×2048 and 4×2048** the three processes agree within about 1 %.
- **At 1×512** each process lands at about 30 or about 36 ms regardless of the policy. For example,
  TM-OPT 16x64 measured 30.8 / 30.5 / 30.7 ms, while MR-OPT 16x64 measured 30.4 / 36.0 / 36.4 ms.
  The short prefill is host-bound, and its level varies from process to process. **No map effect can
  be read at 1×512 with this method.**

**Not measured:**
- decode and small T (below 128), because the narrow-tile builds were not made;
- Phi-4 in the diagnostic, where only the registered numbers exist;
- the stock-vs-mixed overhead on the SM120 branch's RTX 5090 tile table.

Records: `gemm/summary.json`, `gemm/tables.md`, `runs/sm120_bench/`.

## Reproduction

```
/home/dev/n16k64_campaign/tm_opt/queue_items.sh            # #2, #1 (copy in runs/queue_items.sh; log runs/commands_items.log)
python results/tm_opt/analyze_items.py seeds               # items_seeds.{json,md}
python results/tm_opt/analyze_items.py models              # items_models.{json,md}
/home/dev/n16k64_campaign/sm120_bench/build_all.sh         # #3 kernels (copies in runs/sm120_bench/)
/home/dev/n16k64_campaign/sm120_bench/queue_bench.sh       # #3 registered benchmarks
/home/dev/n16k64_campaign/sm120_bench/queue_diag.sh        # #3 diagnostic (deviation 1)
python results/tm_opt/gemm/analyze_gemm.py llama8b phi4    # gemm/summary.json, gemm/tables.md
```
