# MixFP4 potential over FourOverSix at the deployed 256×64 tile (Llama-3.1-8B)

MixFP4 here is FourOverSix E2M1 (per-block α∈{1, 1.5}) plus E0M3 α=1 on elected
256×64 weight tiles. All arms use W4A4 fake quantization on the released
2,048-token WikiText-2 / seed-0 C4 windows. Deltas are PPL versus FourOverSix
(6.875525 / 9.823733); paired ΔNLL ± 2SE is per window.

## Headline

**Loosening the in-place election threshold from k=3 to k=1.5–2 roughly triples
MixFP4's 256×64 gain.** That puts the deployable tile at the level of the 8×64 k=3
result. No reordering is involved.

| Policy | E0M3 units | WikiText-2 | ΔWiki | C4 | ΔC4 | paired ΔNLL wiki / c4 |
|---|---:|---:|---:|---:|---:|---|
| FourOverSix | 0 | 6.875525 | — | 9.823733 | — | — |
| 256×64 CE+KL k=3 (current raw256) | 187 tiles | 6.866879 | −0.0086 | 9.801361 | −0.0224 | −0.0013±0.0016 / −0.0023±0.0015 |
| **256×64 CE+KL k=1.5 (dev-selected)** | 15,399 tiles | **6.856118** | **−0.0194** | **9.776559** | **−0.0472** | −0.0028±0.0018 / −0.0048±0.0023 |
| 256×64 CE+KL k=2 (best on test) | 4,093 tiles | 6.846889 | −0.0286 | 9.774796 | −0.0489 | −0.0042±0.0016 / −0.0050±0.0018 |
| 8×64 CE+KL k=3 (report) | 3,345 tiles | 6.849275 | −0.0262 | 9.773040 | −0.0507 | — |
| 8×64 CE-only k=3 (report) | 22,906 tiles | 6.836686 | −0.0388 | 9.768658 | −0.0551 | — |
| 1×16 MSE ceiling (not deployable) | 53% of blocks | 6.833680 | −0.0418 | 9.762418 | −0.0613 | −0.0061±0.0020 / −0.0063±0.0028 |

Against the 1×16 MSE granularity ceiling, the dev-selected k=1.5 map captures
**46% (WikiText) / 77% (C4)** of the gain. The current k=3 map captures 21% / 36%.
The remaining unlocked potential at 256×64 is about **0.022 WikiText / 0.014 C4**.

## Threshold frontier at 256×64 (CE+KL)

| k | 0 | 1 | 1.5 | 2 | 2.5 | 3 |
|---|---:|---:|---:|---:|---:|---:|
| tiles | 166,420 | 43,974 | 15,399 | 4,093 | 855 | 187 |
| ΔWiki | +0.1153 | +0.0135 | −0.0194 | −0.0286 | −0.0170 | −0.0086 |
| ΔC4 | +0.0714 | −0.0315 | −0.0472 | −0.0489 | −0.0321 | −0.0224 |
| dev ΔCE (192 docs) | — | −0.0025 | **−0.0116** | −0.0092 | −0.0077 | −0.0054 |

Single-objective rules at 256×64 are weaker than CE+KL at the same k (CE k=3
−0.0047/−0.0240; KL k=3 −0.0043/−0.0306; CE k=1 +0.0058/−0.0468).

**k is selected without test data.** `run_k_dev.py` scores
k∈{1, 1.5, 2, 2.5, 3, 3.5, 4} by full-model W4A4 CE on the 192 held-out
OpenWebMath/CodeParrot documents of the three recorded confirmation sets. It reads
no WikiText or C4 and picks **k=1.5**. k=2 is better on test, but it was not the
dev choice. At dev k=3.5 (66 tiles) and k=4 (43 tiles) the gain equals k=3, so a
few dozen tiles carry the current map's whole gain.

## 1×16 task-loss election fails catastrophically

| 1×16 rule | blocks | WikiText | C4 |
|---|---:|---:|---:|
| CE+KL k=3 | 62,394 | 6.8955 | 9.8525 |
| CE+KL k=2 | 2.65M | 8.1009 | 11.2836 |
| CE+KL k=1 | 38.2M | 51.4995 | 40.7845 |
| KL k=3 | 712,281 | 24.5200 | 25.8745 |
| KL k=2 | 10.7M | 259.0249 | 142.2206 |

`diagnose_fine_masks.py`: the elected blocks concentrate in late layers (185k of
the KL k=3 blocks are in `layers.31.mlp.down_proj`), and their E0M3 change is about
2.2× the FourOverSix rounding error. Picking the most negative of ~437M noisy
first-order scores selects the largest moves aligned with one gradient, and
together they overshoot. Summing 1,024 atoms per 256×64 tile averages that noise.
This is why tile-level election works and atom-level election does not. The
usable granularity ceiling is therefore the MSE one.

## Provenance

- Re-scoring (job 422983) reproduces the frozen CE/KL k=3 maps **bitwise**
  (compact-mask SHA256 `6e94897e…`, 187 / 3,345 tiles). Every rule arm re-checks
  this before quantizing.
- `run_math_code_calibration.py --summary-scores` persists per-8×64-tile mean/std
  and per-sequence 256×64 sums. `--fine-masks` accumulates per-1×16 moments on the
  GPU and saves bit-packed masks for k∈{0, 1, 2, 3}.
- Evaluation: `run_mapped_gptq.py --policy rtn_rule|rtn_mse1x16|rtn_fine1x16`.
  The FourOverSix and raw256 controls reproduce the report exactly.
- Jobs 422983, 422984, 423007, 423025–423032, 423042, 423044–423049, 423053;
  gov113008, H200 `dev`.
- Scope: one model, one calibration set, one seed. The dev-selected k still needs
  a fresh-gate check and Qwen replication before any report claim.
