# Mistral-7B-v0.3: KL-only multi-round MixFP4 calibration (Part C, model 2 of 4)

Protocol: [../PROTOCOL.md](../PROTOCOL.md), registered 2026-09-24 18:56:44 UTC (sha256 468978e2…).
This model was registered 2026-09-25 03:28:41 UTC, before its first run (`registration.json`).
The code is unchanged from Qwen3-4B's registration (commit 6587a5b). Every stage passed or
completed; nothing was tuned on WikiText-2 or C4.

**Result, native kernel (primary): both MixFP4 maps beat FourOverSix significantly on both
corpora.**
- **8x64:** WikiText-2 −0.0040 ± 0.0010 and C4 −0.0041 ± 0.0008 nats per window. PPL goes from
  5.5225 / 8.0660 to 5.5004 / 8.0334.
- **256x64:** −0.0049 ± 0.0010 and −0.0033 ± 0.0010, for PPL 5.4956 / 8.0397.

Plain NVFP4 is significantly worse than FourOverSix on both corpora. The fake backend agrees on
every sign and significance.

## Evaluation (WikiText-2 163 windows, C4 256 windows, 2048 tokens; paired per-window ΔNLL ± 2 SE vs FourOverSix)

| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| **native** | FourOverSix | 0 | 5.5225 | 8.0660 | — | — |
| **native** | NVFP4 | — | 5.5552 | 8.0957 | +0.00591 ± 0.00113 (worse) | +0.00367 ± 0.00092 (worse) |
| **native** | **MixFP4 8x64** | 120,931 | **5.5004** | **8.0334** | **−0.00400 ± 0.00097 (better)** | **−0.00405 ± 0.00083 (better)** |
| **native** | **MixFP4 256x64** | 22,264 (256x64) | **5.4956** | **8.0397** | **−0.00487 ± 0.00095 (better)** | **−0.00327 ± 0.00097 (better)** |
| fake | FourOverSix | 0 | 5.5260 | 8.0665 | — | — |
| fake | NVFP4 | — | 5.5531 | 8.0958 | +0.00488 ± 0.00118 (worse) | +0.00363 ± 0.00094 (worse) |
| fake | MixFP4 8x64 | 120,931 | 5.4987 | 8.0271 | −0.00495 ± 0.00093 (better) | −0.00490 ± 0.00081 (better) |
| fake | MixFP4 256x64 | 22,264 (256x64) | 5.4996 | 8.0357 | −0.00480 ± 0.00098 (better) | −0.00383 ± 0.00085 (better) |
| fake | BF16 | — | 5.3182 | 7.8306 | −0.03833 ± 0.00128 (better) | −0.02969 ± 0.00177 (better) |

The evaluation conventions are those of the Qwen3-4B report:
- **NVFP4:** `quant_nvfp4` weights with per-window tensor-wide NVFP4 activations.
- **The 256x64 map** is evaluated inside the 8x64 process, with element masks checked equal.
- **Native checks:** every native map had 0 packed-weight mismatches, and the first 64 activation
  calls were bitwise.
- **Windows:** identical in all three processes. BF16 equals the N16K64 local reproduction exactly
  (5.3182 / 7.8306).

## Stages

1. **Data.**
   - **Fit set:** the N16K64 calibration record
     `research/n16k64/campaigns/primary/calibration_manifests/mistral7b_seed0.json`. All 128 token
     hashes were reproduced. The 224 matrix hashes were computed from the pinned snapshot
     `caa1feb0…`.
   - **Development set:** 192 new documents in three `fresh_data` draws (Llama's rule).
2. **Batching check: not bitwise identical, so batch 1 / 1.**
   - Batch 16/8 against 1/1 changes per-document development CE by up to 0.029 and KL by up to
     0.015.
   - Round-0 scores differ in all 224 matrices.
3. **Pre-check: PASS.**
   - Native − fake FourOverSix mean NLL is −0.0006 ± 0.0008 (WikiText-2) and −0.0001 ± 0.0012
     (C4).
   - The native checks passed on the Mistral architecture: start, mixed (6,815,542 tiles) and
     restored maps with 0 mismatches, and 64 of 64 activation calls bitwise.
4. **Calibration** (lean, native development decisions, deterministic, KL-only, batch 1/1):

| unit | E0M3 tiles | scoring passes / dev evaluations | development KL | setup | optimization | scoring per pass | dev evaluation per try | native build per evaluation | peak GPU allocated / reserved | peak host RSS |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8x64 | 120,931 | 14 / 191 | 0.04231 → 0.02889 | 2.1 min | **78.6 min** | 99.9 s | 17.5 s | 0.15 s | **16.9** / 18.4 GiB | 14.7 GiB |
| 256x64 | 22,264 | 13 / 126 | 0.04231 → 0.02873 | 2.0 min | **57.1 min** | 98.9 s | 17.1 s | 0.15 s | **16.9** / 18.3 GiB | 14.7 GiB |

- **Stops:** both runs stopped because no step lowered the development KL. The 12 h cap never
  bound.
- **Round 0 accepted a large step at both units:**
  - at 8x64, 118,276 of 473,106 candidates (the quarter-size try);
  - at 256x64, all 20,042 candidates.
- **Accepted steps per round:**
  - 8x64: 118,276, 29, 6,950, 770, 253, 255, 123, 62, 15, 14, 7, 28, 3, 0;
  - 256x64: 20,042, 3,256, 27, 1,813, 50, 20, 11, 161, 18, 7, 4, 3, 0.
- **Memory:** host memory is small here because of Mistral's 32k vocabulary (teacher
  log-probabilities).

## Observations (not criteria)

- **Much denser maps than Llama or Qwen3-4B.** Mistral's maps elect a far larger share of tiles:
  120,931 of 13.6 million 8x64 tiles, and 22,264 of 426k 256x64 tiles (5.2 %).
- **Unit size barely matters here.** Development KL falls by a third at both units
  (0.0423 → 0.0289 / 0.0287), and the two units end within noise of each other on both corpora.
- **N16K64 comparison.** The N16K64 campaign's one-shot k = 3 map at 8x64 (per-token activations;
  a different protocol) scored 5.5029 / 8.0449 in fake evaluation, against this run's
  5.4987 / 8.0271.

## Reproduction

```
DATA=/home/dev/n16k64_campaign/multimodel/data
python prepare_model_data.py --out $DATA --model mistral7b
python results/multiround_models/register_model.py mistral7b $DATA
/home/dev/n16k64_campaign/multimodel/queue_model.sh mistral7b 12 "" 16 8     # ../queue_model.sh
python results/multiround_models/analyze_model.py mistral7b /home/dev/n16k64_campaign/multimodel/runs/mistral7b
```

- **Map hashes:** 8x64 7a28d7e0…, 256x64 cb868fe9….
- **Records:** run records are in `runs/`. The maps, dev values and round-0 scores are in
  `/home/dev/n16k64_campaign/multimodel/runs/mistral7b`.
