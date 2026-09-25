# MR-OPT variants: significant steps and warm start — report

**Status (2026-09-25 22:40 UTC).**
- **Llama-3.1-8B:** done.
- **Mistral-7B-v0.3:** done.
- **Phi-4:** running.
- **Qwen3.8-27B:** not started. The study stops at its gate until the user confirms.

**Protocol.** [PROTOCOL.md](PROTOCOL.md), registered 2026-09-25 17:28:59 UTC (sha256 c6f29d27…),
before any run. Two deviations have been appended since:
1. A clarification of the decision rule, before any run.
2. The user's confirmation of the official rule, at 17:46 UTC, before any variant run.

**The official rule.** A variant is ACCEPTABLE for a model and unit when its paired ΔNLL against
MR-OPT satisfies mean − 2 SE ≤ 0 on both WikiText-2 and C4. The secondary reading
(mean + 2 SE ≤ 0) is also reported.

## Verdict so far: MR-OPT stays

- **No variant is acceptable on Llama-3.1-8B** under the official rule. Each one is significantly
  worse than MR-OPT on at least one corpus at at least one unit. In brackets is the lower end of
  each failing comparison, mean − 2 SE of the paired ΔNLL:

  | variant | significantly worse than MR-OPT (native evaluation) |
  |---|---|
  | MR-OPT+SIG | 256x64: WikiText-2 (+0.0011) and C4 (+0.0013); 8x64: WikiText-2 (+0.0033) |
  | MR-OPT+WS | 256x64: C4 (+0.0006); 8x64: WikiText-2 (+0.0018) |
  | MR-OPT+SIG+WS | 256x64: WikiText-2 (+0.0004); 8x64: WikiText-2 (+0.0031) |

  Two failures are marginal (+0.0004, +0.0006). Every variant also has a clear failure, with a
  lower end of at least +0.0018 at 8x64, so the verdict does not depend on the marginal ones.

- **On Mistral-7B-v0.3, only MR-OPT+WS at 256x64 is acceptable.**
  - MR-OPT+WS at 8x64 is significantly worse on WikiText-2 (lower end +0.0009).
  - MR-OPT+SIG and MR-OPT+SIG+WS fail at both units, on WikiText-2 (lower ends +0.0009 to +0.0016)
    and, at 256x64, also on C4 (+0.0004).
- **Llama alone decides the recommendation.** The recommendation needs a variant that is
  acceptable at both units on *every* tested model. No variant can now meet that, whatever Phi-4
  shows, so **MR-OPT stays**.
  - The remaining variant runs on Phi-4 can only describe how the variants behave on another
    model.
  - This is a report only; the user decides.
- **The secondary reading gives the same answer.** No variant passes it on either model.
- **The variants are much faster.** Total optimization time over both units:

  | configuration | Llama-3.1-8B | Mistral-7B-v0.3 | both |
  |---|---:|---:|---:|
  | MR-OPT | 45.5 min | 62.8 min | 108.3 min |
  | MR-OPT+SIG | 20.1 min | 21.3 min | 41.4 min (−62 %) |
  | MR-OPT+WS | 20.6 min | 22.8 min | 43.4 min (−60 %) |
  | MR-OPT+SIG+WS | 12.7 min | 12.2 min | 24.9 min (−77 %) |

  Every variant map is still significantly better than FourOverSix on both corpora at both units,
  on both models. They keep part of MR-OPT's gain, not all of it (tables below).

## Llama-3.1-8B

**Sanity gate: PASS.** The MR-OPT maps are byte-identical to the committed DET-NATIVE maps
(`map.pt` sha256 6e9704f5… at 256x64, 471aa56a… at 8x64). Every window NLL of the two MR-OPT maps
also equals Phase 2's evaluation of the committed maps, on both backends.

**Checks.**
- All evaluation processes used identical windows (token hashes).
- The 256x64 maps expanded exactly to 8x64 tiles with no element-mask mismatch (for example
  8,385 → 268,320 tiles).
- For every map, the first 64 native activation quantizations were bitwise equal to the reference
  quantizer (the check raises on any mismatch).
- Every run recorded source hashes equal to `registration.json`.

### Calibration

| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 256x64 | MR-OPT | 8 / 64 | 8,385 | 0.10811 → 0.09399 | 1.7 min | **17.5 min** | 35.7 s | 12.1 s | 40.7 / 41.6 GiB | 42.3 GiB |
| 256x64 | MR-OPT+SIG | 4 / 37 | 3,846 | 0.10811 → 0.09671 | 1.6 min | **9.6 min** | 35.9 s | 12.0 s | 40.7 / 41.8 GiB | 42.2 GiB |
| 256x64 | MR-OPT+WS | 6 / 25 | 7,620 | 0.10811 → 0.09489 | 1.6 min | **8.4 min** | 35.9 s | 12.0 s | 40.7 / 41.7 GiB | 42.3 GiB |
| 256x64 | MR-OPT+SIG+WS | 4 / 21 | 3,873 | 0.10811 → 0.09701 | 1.7 min | **6.4 min** | 35.9 s | 12.0 s | 40.7 / 41.6 GiB | 42.2 GiB |
| 8x64 | MR-OPT | 9 / 115 | 3,801 | 0.10811 → 0.09441 | 1.6 min | **28.0 min** | 34.3 s | 12.0 s | 41.8 / 43.3 GiB | 42.2 GiB |
| 8x64 | MR-OPT+SIG | 3 / 45 | 1,475 | 0.10811 → 0.09827 | 1.7 min | **10.5 min** | 34.3 s | 12.0 s | 41.8 / 43.3 GiB | 42.1 GiB |
| 8x64 | MR-OPT+WS | 9 / 36 | 3,353 | 0.10811 → 0.09370 | 1.7 min | **12.2 min** | 34.4 s | 12.0 s | 41.8 / 43.3 GiB | 42.1 GiB |
| 8x64 | MR-OPT+SIG+WS | 3 / 24 | 1,476 | 0.10811 → 0.09830 | 1.6 min | **6.3 min** | 34.4 s | 11.9 s | 41.8 / 43.2 GiB | 42.2 GiB |

- **Stop reason:** every run stopped because no step was accepted. With SIG, "accepted" means the
  step passed the significance test.
- **Where the speed comes from:** fewer development evaluations. Per-pass and per-try costs do not
  change.
  - **SIG** stops after 3–4 rounds, with 39–46 % of MR-OPT's tiles and a clearly higher final
    dev KL.
  - **WS** caps each round's first try at twice the last accepted step. After the first round it
    needs 1–8 tries per round, against MR-OPT's 5–19. Its steps then shrink round by round:
    - 256x64 accepted 7,568, 473, 236, 118, 3 tiles;
    - 8x64 accepted 2,966, 1,483, 370, 370, 185, 370, 185, 92 tiles.
- **Memory** is the same in every configuration: the peak sits in scoring.

### Native evaluation (primary) and acceptability

| unit | configuration | tiles shared with MR-OPT / only MR-OPT / only variant | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | 6.8369 | 9.7594 | −0.00566 ± 0.00171 (better) | −0.00673 ± 0.00169 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 3,659 / 4,726 / 187 | 6.8552 | 9.7838 | −0.00298 ± 0.00166 (better) | −0.00425 ± 0.00154 (better) | +0.00268 ± 0.00155 (worse) | +0.00249 ± 0.00123 (worse) | **NO** / no |
| 256x64 | MR-OPT+WS | 7,423 / 962 / 197 | 6.8409 | 9.7781 | −0.00508 ± 0.00163 (better) | −0.00482 ± 0.00159 (better) | +0.00058 ± 0.00151 | +0.00191 ± 0.00128 (worse) | **NO** / no |
| 256x64 | MR-OPT+SIG+WS | 3,697 / 4,688 / 176 | 6.8506 | 9.7700 | −0.00366 ± 0.00148 (better) | −0.00566 ± 0.00167 (better) | +0.00200 ± 0.00158 (worse) | +0.00108 ± 0.00152 | **NO** / no |
| 8x64 | MR-OPT | — | 6.8134 | 9.7644 | −0.00910 ± 0.00175 (better) | −0.00623 ± 0.00177 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 1,247 / 2,554 / 228 | 6.8469 | 9.7679 | −0.00420 ± 0.00158 (better) | −0.00587 ± 0.00191 (better) | +0.00490 ± 0.00160 (worse) | +0.00036 ± 0.00228 | **NO** / no |
| 8x64 | MR-OPT+WS | 2,843 / 958 / 510 | 6.8359 | 9.7522 | −0.00581 ± 0.00172 (better) | −0.00747 ± 0.00200 (better) | +0.00330 ± 0.00152 (worse) | −0.00124 ± 0.00248 | **NO** / no |
| 8x64 | MR-OPT+SIG+WS | 1,249 / 2,552 / 227 | 6.8453 | 9.7687 | −0.00443 ± 0.00159 (better) | −0.00579 ± 0.00193 (better) | +0.00468 ± 0.00159 (worse) | +0.00044 ± 0.00227 | **NO** / no |

ΔNLL is the paired per-window mean ± 2 SE (ddof = 1): 141 WikiText-2 windows and 256 C4 windows
of 2,048 tokens. Here "better" and "worse" mean the ±2 SE interval excludes 0.

**How much of MR-OPT's gain over FourOverSix each variant keeps** (native ΔNLL ratio), against its
share of MR-OPT's optimization time:

| unit | configuration | WikiText-2 | C4 | dev KL reduction | optimization time |
|---|---|---:|---:|---:|---:|
| 256x64 | MR-OPT+SIG | 53 % | 63 % | 81 % | 55 % |
| 256x64 | MR-OPT+WS | 90 % | 72 % | 94 % | 48 % |
| 256x64 | MR-OPT+SIG+WS | 65 % | 84 % | 79 % | 37 % |
| 8x64 | MR-OPT+SIG | 46 % | 94 % | 72 % | 38 % |
| 8x64 | MR-OPT+WS | 64 % | 120 % | 105 % | 43 % |
| 8x64 | MR-OPT+SIG+WS | 49 % | 93 % | 72 % | 22 % |

- **SIG's losses follow its development loss.** It stops early and its dev KL stays higher.
  Significant steps are therefore a stricter stopping rule, not a cleaner one.
- **WS at 8x64 does not follow its development loss.** It ends with a *lower* dev KL than MR-OPT
  (0.09370 vs 0.09441), yet WikiText-2 is significantly worse (+0.0033 ± 0.0015); its C4 is
  inconclusive.
- **Multiple tests.** The rule makes twelve 2-SE comparisons per model, so a single marginal
  "worse" can occur by chance. As noted above, each variant's verdict also rests on a clear
  failure at 8x64.

### Reference maps

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 6.8757 | 9.8254 | — | — |
| native | NVFP4 | 6.9361 | 9.9279 | +0.00875 ± 0.00209 (worse) | +0.01038 ± 0.00203 (worse) |
| native | MR-OPT 256x64 | 6.8369 | 9.7594 | −0.00566 ± 0.00171 (better) | −0.00673 ± 0.00169 (better) |
| native | MR-OPT 8x64 | 6.8134 | 9.7644 | −0.00910 ± 0.00175 (better) | −0.00623 ± 0.00177 (better) |
| fake | FourOverSix | 6.8872 | 9.8294 | — | — |
| fake | NVFP4 | 6.9437 | 9.9302 | +0.00817 ± 0.00208 (worse) | +0.01020 ± 0.00207 (worse) |
| fake | MR-OPT 256x64 | 6.8315 | 9.7737 | −0.00811 ± 0.00172 (better) | −0.00569 ± 0.00148 (better) |
| fake | MR-OPT 8x64 | 6.8178 | 9.7606 | −0.01013 ± 0.00176 (better) | −0.00703 ± 0.00160 (better) |
| fake | BF16 | 6.2403 | 8.9579 | | |

### Llama wall time

2 h 23 min in all (17:29:46 → 19:52:30 UTC):

| part | wall time |
|---|---:|
| MR-OPT runs | 52 min |
| SIG runs | 27 min |
| WS runs | 27 min |
| SIG+WS runs | 19 min |
| native, fake and BF16 evaluations | 17 min |

## Mistral-7B-v0.3

**Checks.**
- All evaluation processes used identical windows (token hashes).
- The 256x64 maps expanded exactly to 8x64 tiles with no element-mask mismatch.
- For every map, the first 64 native activation quantizations were bitwise equal to the reference
  quantizer.
- Every run recorded source hashes equal to `registration.json`.

### Calibration

| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 256x64 | MR-OPT | 14 / 132 | 22,082 | 0.04218 → 0.02819 | 1.4 min | **22.4 min** | 32.9 s | 6.7 s | 35.8 / 36.8 GiB | 14.7 GiB |
| 256x64 | MR-OPT+SIG | 3 / 21 | 21,696 | 0.04218 → 0.03133 | 1.4 min | **4.2 min** | 33.0 s | 7.6 s | 35.8 / 36.8 GiB | 14.7 GiB |
| 256x64 | MR-OPT+WS | 16 / 45 | 21,681 | 0.04218 → 0.02906 | 1.4 min | **13.7 min** | 32.8 s | 6.7 s | 35.8 / 36.8 GiB | 14.7 GiB |
| 256x64 | MR-OPT+SIG+WS | 3 / 19 | 21,696 | 0.04218 → 0.03133 | 1.4 min | **3.9 min** | 33.1 s | 7.6 s | 35.8 / 36.7 GiB | 14.7 GiB |
| 8x64 | MR-OPT | 20 / 237 | 123,180 | 0.04218 → 0.02764 | 1.5 min | **40.5 min** | 31.6 s | 7.6 s | 37.0 / 38.2 GiB | 14.7 GiB |
| 8x64 | MR-OPT+SIG | 9 / 100 | 121,965 | 0.04218 → 0.02928 | 1.5 min | **17.1 min** | 31.6 s | 7.5 s | 37.0 / 38.2 GiB | 14.7 GiB |
| 8x64 | MR-OPT+WS | 9 / 36 | 120,750 | 0.04218 → 0.02841 | 1.4 min | **9.1 min** | 31.5 s | 7.6 s | 37.0 / 38.3 GiB | 14.7 GiB |
| 8x64 | MR-OPT+SIG+WS | 8 / 34 | 121,759 | 0.04218 → 0.02906 | 1.4 min | **8.3 min** | 31.5 s | 7.4 s | 37.0 / 38.3 GiB | 14.7 GiB |

- **Stop reason:** every run stopped because no step was accepted.
- **Mistral needs more rounds than Llama.** MR-OPT takes 14 and 20 rounds, against Llama's 8 and
  9. Its late rounds accept small steps: at 256x64, rounds 8–12 accept 1–18 tiles each.
- **SIG and SIG+WS reach the same 256x64 map** (sha256 58dc2e4a…). Both accept MR-OPT's first two
  steps (19,994 and 3,104 tiles); after that no step passes the significance test.
- **WS at 256x64** follows MR-OPT for two rounds. It then takes 13 more rounds of 1–96 tiles each,
  in 1–10 tries per round (6 of them at the first try).
- **Host RSS is 14.7 GiB**, against Llama's 42 GiB, mostly because Mistral's teacher logits, kept
  in host RAM, have a 32k vocabulary instead of 128k.

### Native evaluation (primary) and acceptability

| unit | configuration | tiles shared with MR-OPT / only MR-OPT / only variant | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | 5.4950 | 8.0354 | −0.00498 ± 0.00098 (better) | −0.00380 ± 0.00074 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 21,228 / 854 / 468 | 5.5085 | 8.0443 | −0.00252 ± 0.00094 (better) | −0.00269 ± 0.00079 (better) | +0.00245 ± 0.00085 (worse) | +0.00110 ± 0.00070 (worse) | **NO** / no |
| 256x64 | MR-OPT+WS | 21,259 / 823 / 422 | 5.4969 | 8.0359 | −0.00464 ± 0.00102 (better) | −0.00374 ± 0.00080 (better) | +0.00034 ± 0.00087 | +0.00006 ± 0.00072 | **yes** / no |
| 256x64 | MR-OPT+SIG+WS | 21,228 / 854 / 468 | 5.5085 | 8.0443 | −0.00252 ± 0.00094 (better) | −0.00269 ± 0.00079 (better) | +0.00245 ± 0.00085 (worse) | +0.00110 ± 0.00070 (worse) | **NO** / no |
| 8x64 | MR-OPT | — | 5.4837 | 8.0334 | −0.00704 ± 0.00097 (better) | −0.00405 ± 0.00075 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 116,129 / 7,051 / 5,836 | 5.4960 | 8.0374 | −0.00481 ± 0.00100 (better) | −0.00356 ± 0.00077 (better) | +0.00223 ± 0.00086 (worse) | +0.00049 ± 0.00069 | **NO** / no |
| 8x64 | MR-OPT+WS | 118,339 / 4,841 / 2,411 | 5.4936 | 8.0371 | −0.00525 ± 0.00096 (better) | −0.00359 ± 0.00089 (better) | +0.00179 ± 0.00087 (worse) | +0.00046 ± 0.00071 | **NO** / no |
| 8x64 | MR-OPT+SIG+WS | 116,279 / 6,901 / 5,480 | 5.4934 | 8.0324 | −0.00527 ± 0.00091 (better) | −0.00418 ± 0.00079 (better) | +0.00176 ± 0.00083 (worse) | −0.00013 ± 0.00075 | **NO** / no |

ΔNLL is the paired per-window mean ± 2 SE (ddof = 1): 141 WikiText-2 windows and 256 C4 windows.

**Share of MR-OPT's gain over FourOverSix kept, and share of its optimization time:**

| unit | configuration | WikiText-2 | C4 | dev KL reduction | optimization time |
|---|---|---:|---:|---:|---:|
| 256x64 | MR-OPT+SIG | 51 % | 71 % | 78 % | 19 % |
| 256x64 | MR-OPT+WS | 93 % | 98 % | 94 % | 61 % |
| 256x64 | MR-OPT+SIG+WS | 51 % | 71 % | 78 % | 18 % |
| 8x64 | MR-OPT+SIG | 68 % | 88 % | 89 % | 42 % |
| 8x64 | MR-OPT+WS | 75 % | 89 % | 95 % | 23 % |
| 8x64 | MR-OPT+SIG+WS | 75 % | 103 % | 90 % | 20 % |

- **As on Llama, the failures are mostly on WikiText-2.** On C4 the only significant differences
  from MR-OPT are SIG's and SIG+WS's at 256x64.
- **WS at 8x64** keeps 95 % of the dev KL reduction but only 75 % of the WikiText-2 gain.

### Reference maps

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 5.5225 | 8.0660 | — | — |
| native | NVFP4 | 5.5552 | 8.0957 | +0.00591 ± 0.00113 (worse) | +0.00367 ± 0.00092 (worse) |
| native | MR-OPT 256x64 | 5.4950 | 8.0354 | −0.00498 ± 0.00098 (better) | −0.00380 ± 0.00074 (better) |
| native | MR-OPT 8x64 | 5.4837 | 8.0334 | −0.00704 ± 0.00097 (better) | −0.00405 ± 0.00075 (better) |
| fake | FourOverSix | 5.5260 | 8.0665 | — | — |
| fake | NVFP4 | 5.5531 | 8.0958 | +0.00488 ± 0.00118 (worse) | +0.00363 ± 0.00094 (worse) |
| fake | MR-OPT 256x64 | 5.4937 | 8.0365 | −0.00586 ± 0.00095 (better) | −0.00372 ± 0.00125 (better) |
| fake | MR-OPT 8x64 | 5.4865 | 8.0309 | −0.00717 ± 0.00100 (better) | −0.00442 ± 0.00116 (better) |
| fake | BF16 | 5.3182 | 7.8306 | | |

**Part C context (not a criterion of this study).** The MR-OPT maps here are Part C candidates at
batch 16/8. Part C's committed Mistral maps were calibrated at batch 1/1 and evaluated on the same
windows: every FourOverSix window NLL is equal between the two native evaluation processes. Paired
native ΔNLL, the batch-16/8 MR-OPT map minus the batch-1/1 Part C map:

| unit | WikiText-2 | C4 |
|---|---|---|
| 256x64 | −0.00011 ± 0.00088 (5.4950 vs 5.4956) | −0.00052 ± 0.00095 (8.0354 vs 8.0397) |
| 8x64 | **−0.00304 ± 0.00089 (better; 5.4837 vs 5.5004)** | +0.00000 ± 0.00066 (8.0334 vs 8.0334) |

### Mistral wall time

2 h 40 min in all (19:52:30 → 22:32:46 UTC):

| part | wall time |
|---|---:|
| MR-OPT runs | 69 min |
| SIG runs | 27 min |
| WS runs | 29 min |
| SIG+WS runs | 18 min |
| evaluations | 17 min |

## Phi-4

Pending.

## Reproduction

```
/home/dev/n16k64_campaign/mr_variants/queue.sh          # copy in runs/queue.sh, log in runs/commands.log
python results/mr_variants/analyze_variants.py MODEL    # MODEL/summary.json, MODEL/tables.md (llama8b, mistral7b)
```

The run records (`report.json`) are in `runs/<model>/`. The maps and per-document development
values stay in `/home/dev/n16k64_campaign/mr_variants/runs/`.
