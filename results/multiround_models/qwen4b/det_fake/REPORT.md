# Qwen3-4B native-decision safety test (addendum to Part C)

Protocol: [PROTOCOL.md](PROTOCOL.md), registered 2026-09-25 05:28:41 UTC (sha256 383fd260…), before
either run. Code is Part C's (commit 6587a5b). DET-FAKE uses exactly the flags of Part C's
DET-NATIVE runs except `--dev-backend fake`: lean, deterministic, KL-only, strict acceptance,
batch 1/1, same data.

## Verdicts (criterion: native evaluation, DET-NATIVE minus DET-FAKE, per-window ΔNLL ± 2 SE)

| unit | ΔWiki | ΔC4 | verdict |
|---|---|---|---|
| 256x64 | −0.00247 ± 0.00328 (inconclusive) | −0.00116 ± 0.00156 (inconclusive) | **SAFE** |
| 8x64 | +0.03255 ± 0.00293 (worse) | +0.02014 ± 0.00183 (worse) | **NOT SAFE** |

- **256x64: the C4 loss is not caused by native decisions.** The fake-decided 256x64 map is also
  significantly worse than FourOverSix on C4:
  - native evaluation: +0.00861 ± 0.00201, against DET-NATIVE's +0.00744 ± 0.00211;
  - fake evaluation: +0.00752 ± 0.00188.

  The regression belongs to the model and the 256x64 unit (a math/code-calibrated 256x64 map
  transfers badly to C4 on Qwen3-4B), whichever evaluator makes the decisions.
- **8x64: the fake-decided map is far better on both corpora.** It has 187,147 E0M3 tiles against
  10,786, and native PPL 12.628 / 16.718 against 13.045 / 17.058. The two searches were identical
  for six rounds; the native search then stopped one round before a single step that changed the
  map (below).

## Where and how the paths diverge

**256x64.**
- Round 0 is identical (6,662 accepted).
- Round 1, 248-tile try: fake accepts it (development KL change −0.00055); native rejects it
  (+0.00134) and accepts 124.
- **Final:** DET-FAKE 7 rounds, 6,720 tiles; DET-NATIVE 5 rounds, 6,673 tiles.

**8x64.**
1. **Rounds 0–5** are identical: 10,505, 972, 684, 408, 190, 183 accepted.
2. **Round 6**, 191-tile try: native accepts it (−0.00020); fake rejects it (+0.00069) and later
   accepts 95.
3. **Rounds 7–10:** both paths take small steps and stay near 10,800 tiles.
   - Native: 94, 1, 10, then nothing in round 10 (every step size raises development KL), so it
     stops at 10,786 tiles.
   - Fake: 86, 89, 42, 19.
4. **Round 11 (fake only):** the first try, all 176,728 candidates at once, lowers the fake
   development KL by 0.00087 and is accepted. Every earlier round's full-size try had raised it by
   0.03–0.33. The map jumps from 10,803 to 187,107 tiles.
5. **Rounds 12–18 (fake only):** the fake path takes 209, 94, 47, 5, 2, 1 and 0 tiles, then stops
   at 187,147 tiles.

**Reading.** The 8x64 failure is path dependence, not a bias of native decisions. The native path
stopped at a point where no step lowered its development KL. The fake path, one small accepted
step later, met a round whose whole candidate set passed the strict-decrease test by a thin
margin, and that step turned out much better on WikiText-2 and C4. Whether the native evaluator
would have accepted a similar step on its own path is not measured.

## Runs

| run | E0M3 tiles | scoring passes / dev evaluations | development KL (own evaluator) | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | peak host RSS |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| DET-FAKE-256x64 | 6,720 | 7 / 53 | 0.11312 → 0.10114 | 1.7 min | 52.3 min | 72.8 s | 50.6 s | 9.6 / 13.1 GiB | 49.4 GiB |
| DET-NATIVE-256x64 | 6,673 | 5 / 39 | 0.11228 → 0.10077 | 2.3 min | 23.6 min | 71.7 s | 27.7 s | 9.6 / 13.1 GiB | 49.4 GiB |
| DET-FAKE-8x64 | 187,147 | 19 / 225 | 0.11312 → 0.09247 | 1.8 min | 209.8 min | 72.5 s | 50.0 s | 10.2 / 13.7 GiB | 49.4 GiB |
| DET-NATIVE-8x64 | 10,786 | 11 / 130 | 0.11228 → 0.09642 | 2.2 min | 73.4 min | 72.1 s | 28.0 s | 10.2 / 14.0 GiB | 49.4 GiB |

At batch 1, a native development evaluation (28 s) is 1.8× faster than a fake one (50 s).

## Evaluation (WikiText-2 146 windows, C4 256; paired per-window ΔNLL ± 2 SE vs FourOverSix)

| unit | backend | map | E0M3 tiles | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---|---:|---:|---:|---|---|
| 256x64 | **native** | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| 256x64 | **native** | DET-NATIVE-256x64 | 6,673 | 13.9906 | 17.4348 | −0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| 256x64 | **native** | DET-FAKE-256x64 | 6,720 | 14.0251 | 17.4551 | −0.01337 ± 0.00442 (better) | +0.00861 ± 0.00201 (worse) |
| 256x64 | fake | DET-NATIVE-256x64 | 6,673 | 14.0873 | 17.4644 | −0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| 256x64 | fake | DET-FAKE-256x64 | 6,720 | 13.9997 | 17.4526 | −0.01731 ± 0.00500 (better) | +0.00752 ± 0.00188 (worse) |
| 8x64 | **native** | DET-NATIVE-8x64 | 10,786 | 13.0453 | 17.0581 | −0.08579 ± 0.00557 (better) | −0.01440 ± 0.00258 (better) |
| 8x64 | **native** | DET-FAKE-8x64 | 187,147 | 12.6275 | 16.7179 | −0.11835 ± 0.00616 (better) | −0.03454 ± 0.00311 (better) |
| 8x64 | fake | DET-NATIVE-8x64 | 10,786 | 13.0396 | 17.0706 | −0.08836 ± 0.00647 (better) | −0.01461 ± 0.00255 (better) |
| 8x64 | fake | DET-FAKE-8x64 | 187,147 | 12.6095 | 16.7192 | −0.12190 ± 0.00713 (better) | −0.03541 ± 0.00279 (better) |

FourOverSix (fake): 14.2442 / 17.3219.

| DET-NATIVE minus DET-FAKE | 256x64 ΔWiki | 256x64 ΔC4 | 8x64 ΔWiki | 8x64 ΔC4 |
|---|---|---|---|---|
| native evaluation (criterion) | −0.00247 ± 0.00328 | −0.00116 ± 0.00156 | +0.03255 ± 0.00293 (worse) | +0.02014 ± 0.00183 (worse) |
| fake evaluation | +0.00623 ± 0.00395 (worse) | +0.00068 ± 0.00147 | +0.03354 ± 0.00275 (worse) | +0.02080 ± 0.00163 (worse) |

## Checks

- **FourOverSix** per-window NLL in every process here equals Part C's evaluation bitwise, native
  and fake. Pairing across processes is valid.
- **DET-NATIVE maps** evaluated here reproduce Part C's evaluation of the same maps bitwise. For
  256x64 this confirms that Part C's in-process conversion to 8x64 tiles is exact.
- **Native evaluations:** 0 packed-weight mismatches, and the first 64 activation calls of every
  map were bitwise.

## Caveats

- **One selection per backend per unit.** The paired ±2 SE excludes selection variance, and at
  8x64 the selection variance is plainly large: one accepted step decides the outcome.
- **The decisive round-11 step passed by a thin margin.** Its development KL change was −0.00087
  over 192 documents, on a strict-decrease rule.
- **Nothing was selected or tuned on WikiText-2 or C4.**

## Reproduction

`/home/dev/n16k64_campaign/multimodel/queue_qwen4b_det_fake.sh` (a copy is in `runs/`), then
`python results/multiround_models/qwen4b/det_fake/analyze_det_fake.py` (writes `summary.json` and
`tables.md`).

- **Map hashes:** DET-FAKE-256x64 and DET-FAKE-8x64 are in `summary.json`.
- **Run records:** in `runs/`.
