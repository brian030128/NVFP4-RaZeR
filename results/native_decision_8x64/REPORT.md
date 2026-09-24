# Native-decision calibration at 8x64 (Task 1b)

**Verdict: SAFE**, under the pre-registered PPL-only criterion ([PROTOCOL.md](PROTOCOL.md),
registered 13:53 UTC with sha256 9c027d31…, before either run; the code is unchanged since
bb997ea).

- **Native evaluation (primary):** DET-NATIVE-8x64 minus DET-FAKE-8x64 is inconclusive on both
  corpora: WikiText-2 −0.00135 ± 0.00143, C4 +0.00120 ± 0.00147 per window.
- **Fake evaluation (secondary):** also inconclusive on both (−0.00007 ± 0.00148, +0.00077 ±
  0.00176).
- **Speed:** native decisions cut 8x64 optimization time by **2.37×, from 86.7 to 36.6 min**. A
  development evaluation takes 14.4 s per try instead of 34.9 s, and the development evaluation
  dominates 8x64 selection. At 256x64 the speed-up was 1.32×.

The unit is set by `--unit 8x64` only. The native kernel `libb8x64.so` puts the weights on
operand B with a format granule of 8 output rows × 64 K, so **one 8x64 map tile is exactly one
kernel granule**.

## Runs

Both runs are deterministic (`--deterministic`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). The flags
are `--unit 8x64 --objective kl --skip-ce-backward --eval-batch 16 --score-batch 8
--budget-hours 12`, with the same data. They differ only in `--dev-backend`.

| | DET-FAKE-8x64 | DET-NATIVE-8x64 |
|---|---:|---:|
| development KL (own backend), start → end | 0.106448 → 0.092620 | 0.108106 → 0.094413 |
| rounds (scoring passes) / development evaluations | 10 / 130 | 9 / 115 |
| final E0M3 tiles | 3,454 | 3,801 |
| setup / optimization | 94 s / **5,205 s (86.7 min)** | 119 s / **2,194 s (36.6 min)** |
| scoring per pass / development evaluation per try | 70.1 s / 34.9 s | 69.5 s / 14.4 s |
| peak GPU allocated / reserved | 62.0 / 64.6 GiB | 73.3 / 75.8 GiB (native weight packs) |
| peak host RSS | 43.7 GiB | 44.0 GiB |

**Where the paths diverge.**
- Round 0 is identical in both runs: 379,759 candidates, the same tries, 2,966 tiles accepted.
- Round 1 has the same 600,542 candidates and the same tries down to 1,172 flips. At that try,
  **fake rejects (ΔKL +0.000029, a margin of 3e-5) and native accepts (ΔKL −0.00055)**. Fake
  goes on to accept 586.
- From round 2 the states differ.

| round | DET-FAKE-8x64: candidates / accepted / tiles | DET-NATIVE-8x64: candidates / accepted / tiles |
|---:|---|---|
| 0 | 379,759 / 2,966 / 2,966 | 379,759 / 2,966 / 2,966 |
| 1 | 600,542 / 586 / 3,028 | 600,542 / **1,172** / 3,204 |
| 2 | 450,973 / 880 / 3,272 | 408,618 / 1,596 / 3,792 |
| 3 | 429,274 / 209 / 3,257 | 460,343 / 449 / 3,795 |
| 4 | 346,217 / 169 / 3,290 | 367,725 / 179 / 3,804 |
| 5 | 396,294 / 96 / 3,270 | 395,694 / 48 / 3,794 |
| 6 | 362,416 / 44 / 3,276 | 365,358 / 44 / 3,806 |
| 7 | 354,259 / 345 / 3,455 | 367,349 / 5 / 3,801 |
| 8 | 355,610 / 1 / 3,454 | 365,242 / 0 (stop) |
| 9 | 339,843 / 0 (stop) | |

## Evaluation (WikiText-2 / C4, released windows, per-window paired ΔNLL ± 2 SE)

| map | E0M3 tiles | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---|---|
| **native kernel (primary)** | | | | | |
| FourOverSix | 0 | 6.875681 | 9.825390 | — | — |
| DET-FAKE-8x64 | 3,454 | 6.822587 | 9.752643 | −0.00775 ± 0.00171 | −0.00743 ± 0.00156 |
| DET-NATIVE-8x64 | 3,801 | 6.813365 | 9.764371 | −0.00910 ± 0.00175 | −0.00623 ± 0.00177 |
| **fake (secondary)** | | | | | |
| FourOverSix | 0 | 6.887180 | 9.829412 | — | — |
| DET-FAKE-8x64 | 3,454 | 6.818290 | 9.753089 | −0.01005 ± 0.00191 | −0.00780 ± 0.00188 |
| DET-NATIVE-8x64 | 3,801 | 6.817782 | 9.760603 | −0.01013 ± 0.00176 | −0.00703 ± 0.00160 |

Both maps are significantly better than FourOverSix on both corpora, under both backends.

**The criterion:**

| DET-NATIVE-8x64 minus DET-FAKE-8x64 | ΔWiki | ΔC4 |
|---|---|---|
| native evaluation (criterion) | −0.00135 ± 0.00143 (inconclusive) | +0.00120 ± 0.00147 (inconclusive) |
| fake evaluation | −0.00007 ± 0.00148 (inconclusive) | +0.00077 ± 0.00176 (inconclusive) |

Neither corpus is significantly worse, so the verdict is **SAFE**.

### Context only (not a criterion): 8x64 vs Task 1's 256x64 DET maps, native evaluation

This task's native FourOverSix evaluation reproduces Task 1's bitwise on every window, so pairing
per-window NLLs across the two evaluation processes is valid.

| | ΔWiki | ΔC4 |
|---|---|---|
| DET-FAKE-8x64 minus DET-FAKE (256x64) | −0.00134 ± 0.00164 (inconclusive) | −0.00279 ± 0.00108 (better) |
| DET-FAKE-8x64 minus DET-NATIVE (256x64) | −0.00209 ± 0.00157 (better) | −0.00070 ± 0.00111 (inconclusive) |
| DET-NATIVE-8x64 minus DET-FAKE (256x64) | −0.00269 ± 0.00171 (better) | −0.00159 ± 0.00174 (inconclusive) |
| DET-NATIVE-8x64 minus DET-NATIVE (256x64) | −0.00344 ± 0.00172 (better) | +0.00051 ± 0.00157 (inconclusive) |

The 8x64 maps are never worse than the 256x64 maps. They are better on one corpus in each
pairing, and inconclusive on the other.

## Verification

- **Bitwise checks at startup of the native run:**
  - all 224 packed candidates;
  - the start map, a random mixed 8x64 map (about 6.8 million E0M3 tiles) and the restored
    start map;
  - the first 64 activation calls.
- **Bitwise checks at evaluation:** every evaluated map (0 mismatches) and its first 64
  activation calls.
- **Fake map evaluation:** it reproduces each DET run's built-in fake final evaluation bitwise, per
  window, for both maps.
- **Cross-process pairing:** the native FourOverSix evaluation is bitwise identical to Task 1's,
  which validates the context comparison.

## Caveats

- **One selection per backend.** The paired ±2 SE does not include selection variance. At
  256x64 (Task 1), fake-decided maps from different runs already differed from each other by
  0.0018–0.0025 on native WikiText.
- **Near-threshold divergence.** The divergence point is a step fake rejected by a margin of 3e-5
  in development KL, far inside evaluator noise.
- **The development documents are not held out.** The 192 documents decide every step.
- **Nothing was tuned on test data.** Nothing was selected or tuned on WikiText or C4. Zero-shot
  (Task 2) stays on hold and is not part of this verdict.

## Reproduction

```
DATA=/home/dev/n16k64_campaign/cost_comparison/data
R=/home/dev/n16k64_campaign/native_decision_8x64
export HF_HUB_OFFLINE=0 PYTHONPATH=$PWD
COMMON="--unit 8x64 --objective kl --data-root $DATA --transformers-deviation"
DET="--skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multiround.py $COMMON $DET --dev-backend fake   --out $R/det_fake_8x64
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multiround.py $COMMON $DET --dev-backend native --out $R/det_native_8x64
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map DET-FAKE-8x64=$R/det_fake_8x64/map.pt --evaluate-map DET-NATIVE-8x64=$R/det_native_8x64/map.pt"
python run_multiround.py $COMMON --eval-backend native $MAPS --out $R/eval_native
python run_multiround.py $COMMON --eval-backend fake $MAPS --out $R/eval_fake
python results/native_decision_8x64/analyze_8x64.py $R        # summary.json, tables.md
```

- Environment: torch 2.9.0+cu128, transformers 5.16.1 (a recorded deviation from 4.57.3), triton
  3.5.0, driver 595.71.05, native kernel `libb8x64.so` (sha256 0e237ada…).
- Map hashes: DET-FAKE-8x64 5b9a88fd…, DET-NATIVE-8x64 471aa56a….
- Run records: `runs/`. Hashes at registration: `registration.json`.
