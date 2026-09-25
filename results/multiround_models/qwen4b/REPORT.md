# Qwen3-4B: KL-only multi-round MixFP4 calibration (Part C, model 1 of 4)

Protocol: [../PROTOCOL.md](../PROTOCOL.md), registered 2026-09-24 18:56:44 UTC (sha256 468978e2…).
This model was registered 2026-09-25 01:18:53 UTC, before its first run (`registration.json`).
Every stage passed or completed; nothing was tuned on WikiText-2 or C4.

**Result, native kernel (primary):**
- **MixFP4 8x64 beats FourOverSix significantly on both corpora:** WikiText-2 −0.0858 ± 0.0056
  and C4 −0.0144 ± 0.0026 nats per window. PPL goes from 14.214 / 17.306 to **13.045 / 17.058**.
- **MixFP4 256x64 is mixed:** better on WikiText-2 (−0.0158 ± 0.0040) but **significantly worse
  on C4** (+0.0074 ± 0.0021).

The fake backend (secondary) agrees on every sign and significance.

## Evaluation (WikiText-2 146 windows, C4 256 windows, 2048 tokens; paired per-window ΔNLL ± 2 SE vs FourOverSix)

| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| **native** | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| **native** | NVFP4 | — | 13.9418 | 17.2685 | −0.01933 ± 0.00490 (better) | −0.00214 ± 0.00221 |
| **native** | **MixFP4 8x64** | 10,786 | **13.0453** | **17.0581** | **−0.08579 ± 0.00557 (better)** | **−0.01440 ± 0.00258 (better)** |
| **native** | MixFP4 256x64 | 6,673 (256x64) | 13.9906 | 17.4348 | −0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | NVFP4 | — | 13.9488 | 17.2785 | −0.02096 ± 0.00536 (better) | −0.00251 ± 0.00218 (better) |
| fake | MixFP4 8x64 | 10,786 | 13.0396 | 17.0706 | −0.08836 ± 0.00647 (better) | −0.01461 ± 0.00255 (better) |
| fake | MixFP4 256x64 | 6,673 (256x64) | 14.0873 | 17.4644 | −0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| fake | BF16 | — | 13.6588 | 16.6409 | −0.04197 ± 0.00543 (better) | −0.04010 ± 0.00281 (better) |

**How each map was evaluated.**
- **FourOverSix and MixFP4:** weights as the name says, with per-window tensor-wide FourOverSix
  activations.
- **NVFP4:** `quant_nvfp4` weights and per-window tensor-wide `quant_nvfp4` activations (the
  `nvfp4_tensor` baseline).
- **The 256x64 map** is evaluated inside the 8x64 process. Each 256-row tile is expanded to 32
  eight-row tiles, and the element masks were checked equal.
- **Native checks:** every native map had 0 packed-weight mismatches, and the first 64 activation
  calls of every map were bitwise.
- **Windows:** all three evaluation processes used identical windows (token hashes). No published
  window record exists for this model.
- **BF16 cross-check:** the BF16 PPL equals the earlier N16K64 local reproduction exactly
  (13.6588 / 16.6409), so the evaluation windows are that campaign's.

## Stages

1. **Data (hashes in `registration.json` and the data root's `prepare_summary.json`).**
   - **Fit set:** the archived calibration record
     `results/math_code_adaptive/calibration_333779_qwen4b` (the N16K64 seed0 fit, identical). All
     128 token hashes were reproduced, as were the 252 matrix weight hashes.
   - **Development set:** 192 new documents in three `fresh_data` draws (Llama's rule). Each draw
     excludes this model's C4 evaluation documents and the earlier draws.
2. **Batching check: not bitwise identical, so calibration uses batch 1 / 1.**
   - Batch 16/8 against 1/1 changes per-document development CE by up to 0.096 and KL by up to
     0.057 (fake and native).
   - Round-0 scores differ in all 252 matrices.
   - The mean development KL moves little: native 0.11261 vs 0.11228.
3. **Pre-check: PASS.**
   - Native − fake FourOverSix mean NLL is −0.0021 ± 0.0046 (WikiText-2) and −0.0009 ± 0.0015
     (C4), both inside ±0.01.
   - The native checks passed on the Qwen3 architecture: start, random mixed (3,548,164 tiles) and
     restored maps, 0 mismatches, and 64 of 64 activation calls bitwise.
4. **Calibration** (lean, native development decisions, deterministic, KL-only, batch 1/1):

| unit | E0M3 tiles | scoring passes / dev evaluations | development KL | setup | optimization | scoring per pass | dev evaluation per try | native build per evaluation | peak GPU allocated / reserved | peak host RSS |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 10,786 | 11 / 130 | 0.11228 → 0.09642 | 2.2 min | **73.4 min** | 72.1 s | 28.0 s | 0.12 s | **10.2** / 14.0 GiB | 49.4 GiB |
| 256x64 | 6,673 | 5 / 39 | 0.11228 → 0.10077 | 2.3 min | **23.6 min** | 71.7 s | 27.7 s | 0.12 s | **9.6** / 13.1 GiB | 49.4 GiB |

- **Stops:** both runs stopped because no step lowered the development KL. The 12 h cap never
  bound.
- **Accepted steps per round:**
  - 8x64: 10,505, 972, 684, 408, 190, 183, 191, 94, 1, 10, 0;
  - 256x64: 6,662, 124, 55, 28, 0.
- **Host memory** is dominated by the BF16 teacher log-probabilities over the 152k vocabulary.

## Observations (not criteria)

- **MixFP4 8x64 goes below BF16 on WikiText-2** (13.05 vs 13.66), while C4 stays above BF16
  (17.06 vs 16.64).
  - The objective pulls the quantized model toward BF16 on math/code, so this overshoot on
    WikiText is a property of this post-trained model, not of the objective.
  - The N16K64 campaign's maps (one-shot CE+KL election with k = 3, per-token activations; a
    different protocol) went much further below BF16 (n8_k3: 11.71 / 15.71).
- **FourOverSix is worse than plain NVFP4 on WikiText-2** for this model under per-window
  tensor-wide activations (−0.019 in NVFP4's favour).
- **The 256x64 unit transfers badly to C4 here.** Its map lowered math/code KL (0.1123 → 0.1008)
  but raised C4 NLL significantly. The finer 8x64 unit did not.

## Reproduction

```
DATA=/home/dev/n16k64_campaign/multimodel/data
python prepare_model_data.py --out $DATA --model qwen4b     # CPU; checks every hash
python results/multiround_models/register_model.py qwen4b $DATA
/home/dev/n16k64_campaign/multimodel/queue_model.sh qwen4b 12 "--transformers-deviation" 16 8   # runs/queue_model.sh
python results/multiround_models/analyze_model.py qwen4b /home/dev/n16k64_campaign/multimodel/runs/qwen4b
```

- **Environment:** torch 2.9.0+cu128, transformers 5.16.1 (the calibration record names 4.57.3:
  `--transformers-deviation`), triton 3.5.0, `libb8x64.so` (sha256 0e237ada…).
- **Map hashes:** 8x64 15e2de9d…, 256x64 57b213af….
- **Records:** run records are in `runs/` (report.json of every run, `batching.json`,
  `precheck.json`, `commands.log`). The maps, dev values and round-0 scores are in
  `/home/dev/n16k64_campaign/multimodel/runs/qwen4b`.
