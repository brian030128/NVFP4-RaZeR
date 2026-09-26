# MR-OPT variants: significant steps and warm start — report

**Status (2026-09-26 02:00 UTC): Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 are done.** The queue
stopped at the Qwen3.8-27B gate at 01:58:45 UTC. Nothing runs until the user confirms Qwen3.8-27B
and its settings. The study so far took 8 h 29 min of wall time.

**Addendum (2026-09-26 07:00 UTC): MR-OPT at 16x64** on the same three models; see
[the addendum section](#addendum-mr-opt-at-16x64). The 16x64 variants were cut by the user. The gate
is still closed.

**Protocol.** [PROTOCOL.md](PROTOCOL.md), registered 2026-09-25 17:28:59 UTC (sha256 c6f29d27…),
before any run. Two deviations have been appended since:
1. A clarification of the decision rule, before any run.
2. The user's confirmation of the official rule, at 17:46 UTC, before any variant run.

**The official rule.** A variant is ACCEPTABLE for a model and unit when its paired ΔNLL against
MR-OPT satisfies mean − 2 SE ≤ 0 on both WikiText-2 and C4. The secondary reading
(mean + 2 SE ≤ 0) is also reported.

## Verdict: MR-OPT stays

**No variant is acceptable at both units on every model, so the recommendation is MR-OPT.** This
is a report only; the user decides.

| variant | Llama 256x64 | Llama 8x64 | Mistral 256x64 | Mistral 8x64 | Phi-4 256x64 | Phi-4 8x64 |
|---|---|---|---|---|---|---|
| MR-OPT+SIG | NO | NO | NO | NO | yes (better on WikiText-2) | yes (better on both) |
| MR-OPT+WS | NO | NO | yes | NO | NO (C4, lower end +0.00001) | yes |
| MR-OPT+SIG+WS | NO | NO | NO | NO | yes (better on both) | yes |

"Better" means significantly better than MR-OPT: mean + 2 SE < 0.

- **Llama alone rules out every variant.** Each variant has a clear failure at 8x64, on WikiText-2,
  with a lower end (mean − 2 SE of the paired ΔNLL) of at least +0.0018 (details below).
- **Under the secondary reading** (mean + 2 SE ≤ 0) no variant passes at both units on any model.
- **The effect of significant steps is model-dependent.**
  - **On Llama and Mistral, SIG stops early**, with a higher final dev KL than MR-OPT:
    - on Llama after 3–4 rounds, with 39–46 % of MR-OPT's tiles;
    - on Mistral after 3 and 9 rounds, against MR-OPT's 14 and 20.

    SIG and SIG+WS are significantly worse than MR-OPT on WikiText-2 at both units on both models.
    The smallest margin is Llama's SIG+WS at 256x64 (lower end +0.0004).
  - **On Phi-4, SIG and SIG+WS are significantly better than MR-OPT.** Examples: SIG at 8x64 is
    −0.0051 ± 0.0015 on WikiText-2 and −0.0031 ± 0.0009 on C4; SIG+WS at 256x64 is
    −0.0044 ± 0.0014 and −0.0015 ± 0.0008.
  - **Why on Phi-4:** MR-OPT reaches the lowest dev KL there but accepts large steps that barely
    lower it. One such step is 39,279 tiles for a dev KL change of −0.00006. Those steps do not
    carry over to WikiText-2 and C4.
  - **What this does not allow:** picking a variant per model from these results would be
    selecting on WikiText-2 and C4, which this project does not do. A per-model rule would need a
    criterion fixed on development data only.
- **The variants are much faster.** Total optimization time over both units:

  | configuration | Llama-3.1-8B | Mistral-7B-v0.3 | Phi-4 | all three |
  |---|---:|---:|---:|---:|
  | MR-OPT | 45.5 min | 62.8 min | 71.3 min | 179.6 min |
  | MR-OPT+SIG | 20.1 min | 21.3 min | 18.1 min | 59.5 min (−67 %) |
  | MR-OPT+WS | 20.6 min | 22.8 min | 32.4 min | 75.8 min (−58 %) |
  | MR-OPT+SIG+WS | 12.7 min | 12.2 min | 15.1 min | 40.0 min (−78 %) |

- **Every variant map is significantly better than FourOverSix** on both corpora at both units,
  on all three models.

### Llama details

In brackets is the lower end of each failing comparison, mean − 2 SE of the paired ΔNLL:

| variant | significantly worse than MR-OPT on Llama (native evaluation) |
|---|---|
| MR-OPT+SIG | 256x64: WikiText-2 (+0.0011) and C4 (+0.0013); 8x64: WikiText-2 (+0.0033) |
| MR-OPT+WS | 256x64: C4 (+0.0006); 8x64: WikiText-2 (+0.0018) |
| MR-OPT+SIG+WS | 256x64: WikiText-2 (+0.0004); 8x64: WikiText-2 (+0.0031) |

Two failures are marginal (+0.0004, +0.0006). Every variant also has a clear failure, with a lower
end of at least +0.0018 at 8x64, so the verdict does not depend on the marginal ones.

### Mistral details

- **Only MR-OPT+WS at 256x64 is acceptable.**
- **MR-OPT+WS at 8x64** is significantly worse on WikiText-2 (lower end +0.0009).
- **MR-OPT+SIG and MR-OPT+SIG+WS** fail at both units. On WikiText-2 the lower ends are +0.0009 to
  +0.0016; at 256x64 they also fail on C4 (+0.0004).

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

**Checks.**
- All evaluation processes used identical windows (token hashes).
- The 256x64 maps expanded exactly to 8x64 tiles with no element-mask mismatch.
- For every map, the first 64 native activation quantizations were bitwise equal to the reference
  quantizer.
- Every run recorded source hashes equal to `registration.json`.

### Calibration

| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 256x64 | MR-OPT | 14 / 85 | 59,953 | 0.03765 → 0.03492 | 2.6 min | **35.3 min** | 67.0 s | 14.0 s | 59.0 / 61.7 GiB | 33.6 GiB |
| 256x64 | MR-OPT+SIG | 3 / 27 | 13,273 | 0.03765 → 0.03568 | 2.6 min | **9.4 min** | 67.0 s | 14.0 s | 59.0 / 62.0 GiB | 33.6 GiB |
| 256x64 | MR-OPT+WS | 8 / 30 | 28,561 | 0.03765 → 0.03537 | 2.5 min | **15.7 min** | 67.1 s | 14.0 s | 59.0 / 61.9 GiB | 33.6 GiB |
| 256x64 | MR-OPT+SIG+WS | 2 / 18 | 13,237 | 0.03765 → 0.03633 | 2.6 min | **6.2 min** | 67.1 s | 14.0 s | 59.0 / 61.8 GiB | 33.7 GiB |
| 8x64 | MR-OPT | 10 / 108 | 57,098 | 0.03765 → 0.03481 | 2.6 min | **36.0 min** | 65.3 s | 14.1 s | 61.3 / 64.2 GiB | 33.6 GiB |
| 8x64 | MR-OPT+SIG | 2 / 29 | 5,470 | 0.03765 → 0.03675 | 2.6 min | **8.7 min** | 65.4 s | 14.0 s | 61.3 / 64.1 GiB | 33.7 GiB |
| 8x64 | MR-OPT+WS | 8 / 35 | 12,478 | 0.03765 → 0.03517 | 2.6 min | **16.7 min** | 65.5 s | 14.0 s | 61.3 / 64.3 GiB | 33.6 GiB |
| 8x64 | MR-OPT+SIG+WS | 3 / 25 | 5,653 | 0.03765 → 0.03577 | 2.6 min | **8.9 min** | 65.3 s | 14.1 s | 61.3 / 64.2 GiB | 33.7 GiB |

- **Stop reason:** every run stopped because no step was accepted.
- **MR-OPT accepts large steps that barely lower the dev KL:**
  - In round 0 at 256x64 it accepts all 26,474 candidates at the first try, reaching a dev KL of
    0.03691. SIG rejects that step as not significant. Its half step of 13,237 tiles reaches a
    *lower* dev KL, 0.03633.
  - Later, MR-OPT accepts 24,478 tiles for a dev KL change of −0.00009 (256x64, round 4), and
    39,279 tiles for −0.00006 (8x64, round 8).
  - It ends with 59,953 and 57,098 tiles, against SIG's 13,273 and 5,470.
- **The lowest dev KL is not the best test loss.** MR-OPT has the lowest final dev KL at both units.
  Yet SIG, with fewer tiles and a higher dev KL, is significantly better on WikiText-2 at both
  units, and on C4 at 8x64 (below).

### Native evaluation (primary) and acceptability

| unit | configuration | tiles shared with MR-OPT / only MR-OPT / only variant | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |
|---|---|---|---:|---:|---|---|---|---|---|
| 256x64 | MR-OPT | — | 6.6519 | 10.5237 | −0.00195 ± 0.00147 (better) | −0.00208 ± 0.00083 (better) | — | — | — |
| 256x64 | MR-OPT+SIG | 12,539 / 47,414 / 734 | 6.6420 | 10.5218 | −0.00344 ± 0.00131 (better) | −0.00226 ± 0.00080 (better) | −0.00149 ± 0.00130 (better) | −0.00018 ± 0.00071 | **yes** / no |
| 256x64 | MR-OPT+WS | 27,059 / 32,894 / 1,502 | 6.6525 | 10.5321 | −0.00185 ± 0.00137 (better) | −0.00128 ± 0.00083 (better) | +0.00010 ± 0.00149 | +0.00079 ± 0.00078 (worse) | **NO** / no |
| 256x64 | MR-OPT+SIG+WS | 12,482 / 47,471 / 755 | 6.6225 | 10.5081 | −0.00638 ± 0.00152 (better) | −0.00357 ± 0.00082 (better) | −0.00443 ± 0.00138 (better) | −0.00149 ± 0.00082 (better) | **yes** / yes |
| 8x64 | MR-OPT | — | 6.6373 | 10.5216 | −0.00415 ± 0.00131 (better) | −0.00228 ± 0.00080 (better) | — | — | — |
| 8x64 | MR-OPT+SIG | 5,058 / 52,040 / 412 | 6.6036 | 10.4893 | −0.00923 ± 0.00161 (better) | −0.00535 ± 0.00089 (better) | −0.00508 ± 0.00146 (better) | −0.00307 ± 0.00085 (better) | **yes** / yes |
| 8x64 | MR-OPT+WS | 11,424 / 45,674 / 1,054 | 6.6314 | 10.5145 | −0.00504 ± 0.00137 (better) | −0.00296 ± 0.00074 (better) | −0.00089 ± 0.00121 | −0.00068 ± 0.00073 | **yes** / no |
| 8x64 | MR-OPT+SIG+WS | 5,299 / 51,799 / 354 | 6.6376 | 10.5239 | −0.00410 ± 0.00112 (better) | −0.00206 ± 0.00077 (better) | +0.00005 ± 0.00123 | +0.00023 ± 0.00075 | **yes** / no |

ΔNLL is the paired per-window mean ± 2 SE (ddof = 1): 141 WikiText-2 windows and 256 C4 windows.

- **MR-OPT+WS at 256x64 fails by a hair:** on C4 the lower end is +0.000014. Even if it passed,
  MR-OPT+WS would still fail on Llama and Mistral, so the recommendation does not depend on it.
- **Phi-4's MR-OPT gains over FourOverSix are the smallest of the three models.** At 256x64 the
  WikiText-2 gain (−0.00195 ± 0.00147) is barely significant.

**Share of MR-OPT's gain over FourOverSix kept, and share of its optimization time** (above 100 %
means a larger gain than MR-OPT's):

| unit | configuration | WikiText-2 | C4 | dev KL reduction | optimization time |
|---|---|---:|---:|---:|---:|
| 256x64 | MR-OPT+SIG | 177 % | 109 % | 72 % | 27 % |
| 256x64 | MR-OPT+WS | 95 % | 62 % | 84 % | 45 % |
| 256x64 | MR-OPT+SIG+WS | 327 % | 172 % | 48 % | 18 % |
| 8x64 | MR-OPT+SIG | 223 % | 234 % | 32 % | 24 % |
| 8x64 | MR-OPT+WS | 121 % | 130 % | 87 % | 46 % |
| 8x64 | MR-OPT+SIG+WS | 99 % | 90 % | 66 % | 25 % |

### Reference maps

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---|---|
| native | FourOverSix | 6.6649 | 10.5456 | — | — |
| native | NVFP4 | 6.7046 | 10.5866 | +0.00594 ± 0.00182 (worse) | +0.00388 ± 0.00104 (worse) |
| native | MR-OPT 256x64 | 6.6519 | 10.5237 | −0.00195 ± 0.00147 (better) | −0.00208 ± 0.00083 (better) |
| native | MR-OPT 8x64 | 6.6373 | 10.5216 | −0.00415 ± 0.00131 (better) | −0.00228 ± 0.00080 (better) |
| fake | FourOverSix | 6.6667 | 10.5437 | — | — |
| fake | NVFP4 | 6.7029 | 10.5882 | +0.00543 ± 0.00181 (worse) | +0.00421 ± 0.00106 (worse) |
| fake | MR-OPT 256x64 | 6.6458 | 10.5293 | −0.00314 ± 0.00129 (better) | −0.00136 ± 0.00083 (better) |
| fake | MR-OPT 8x64 | 6.6369 | 10.5149 | −0.00448 ± 0.00129 (better) | −0.00273 ± 0.00078 (better) |
| fake | BF16 | 6.4615 | 10.3098 | | |

The MR-OPT maps are Part C's Phi-4 candidates. Part C has no earlier Phi-4 maps to compare with.

### Phi-4 wall time

3 h 26 min in all (22:32:46 → 01:58:45 UTC):

| part | wall time |
|---|---:|
| MR-OPT runs | 82 min |
| SIG runs | 28 min |
| WS runs | 43 min |
| SIG+WS runs | 25 min |
| evaluations | 28 min |

## Gate: Qwen3.8-27B (waiting for the user)

The queue has stopped, and nothing runs until the user decides:

1. **Which configurations.** Given the verdict: MR-OPT alone (the Part C candidate), or all four.
2. **The scoring batch.**
   - **Development evaluation** at batch 16 fits.
   - **Scoring at batch 8 or 4** runs out of memory ([REPORT_CHUNKED.md](../speedups/REPORT_CHUNKED.md)).
   - **Batch 2** fits at a 91.2 GiB peak and takes 209 s per pass; **batch 1** fits at 62.3 GiB and
     takes 360 s per pass ([PROBE_QWEN27B_BATCH.md](../speedups/PROBE_QWEN27B_BATCH.md)).
   - Either departs from the 16/8 rule. Batched forwards are not batch-invariant, so the choice
     changes the search path.
3. **Units:** 256x64 and 8x64, as for the other models, unless the user says otherwise.

**Rough time, at scoring batch 2** (209 s per pass, 43 s per native development evaluation, about
6 min of setup per run):
- **MR-OPT at both units:** about 3.5–7 h. The range depends on whether Qwen needs Llama-like
  (8–9) or Mistral-like (14–20) rounds.
- **All four configurations:** about 7–16 h.
- **Evaluations:** roughly another hour.

## Addendum: MR-OPT at 16x64

**Scope.** Registered in [ADDENDUM_16x64.md](ADDENDUM_16x64.md) (sha256 0944c9e8…,
2026-09-26 04:03:55 UTC), before any 16x64 test or run.
- **Why:** user request; the SM120 deployment kernel's weight granule is 16x64.
- **What ran:** only MR-OPT. The user cut the 16x64 variants (addendum deviation 1).
- **Unaffected:** the recommendation over 8x64 and 256x64 (MR-OPT).

**The development numerics are not the deployment path's.** A 16x64 map is evaluated on the b8x64
build, as two 8x64 granules per tile; its native weight is exact. However:
- The repro study's bit-identity of the `wt_as_A` and `b8x64` builds covered FourOverSix only (no
  E0M3 tiles), in the repro's own evaluator.
- The SM120 deployment kernel (`origin/SM120-kernel`, `sm120/NUMERICS.md`) uses per-token
  activation scales and a one-rounding epilogue. It states that the repro harness's two-rounding
  path is not bit-identical to it.
- This study uses per-document or per-window activation scales (convention (a)) and rounds twice.

The addendum gives the details.

### Pre-run checks: all passed

| check | result |
|---|---|
| B1 unit tests at 16x64 (24 cases, the three models' layer shapes) | max normwise relative error: g 2.2e-7, μ 2.4e-7, SE 6.9e-8 (tolerance 1e-6); 0 of 1,206,272 candidate tiles differ; B1 1.9× faster |
| Round-0 candidates, legacy hook vs B1 | identical sets: Llama 196,733 of 6,815,744 tiles; Mistral 244,371 of 6,815,744; Phi-4 357,367 of 13,312,000 (max normwise μ / SE differences ≤ 8.4e-7) |
| Native build of 16x64 maps | start, random-mixed and restored maps bitwise equal to `apply()` on all three models |
| The 8x64 path is unchanged | a Llama 8x64 run reproduced Phase 2's round-0 score file (sha256) and MR-OPT 8x64's initial development values |

Details: `checks_16x64.json`. B1's round-0 scoring pass takes 34.9 s, 31.9 s and 66.0 s,
against the legacy hook's 48.4 s, 45.4 s and 94.4 s (Llama, Mistral, Phi-4).

### Results (native evaluation, convention (a))

"E0M3 share" is the fraction of weight elements in E0M3 tiles.

| model | unit | E0M3 tiles (own unit) | E0M3 share | rounds / dev evaluations | optimization | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Llama-3.1-8B | 8x64 | 3,801 | 0.03 % | 9 / 115 | 28.0 min | 6.8134 | 9.7644 | −0.00910 ± 0.00175 (better) | −0.00623 ± 0.00177 (better) |
| | **16x64** | 4,995 | 0.07 % | 15 / 184 | **45.4 min** | **6.8259** | **9.7412** | −0.00727 ± 0.00163 (better) | −0.00861 ± 0.00219 (better) |
| | 256x64 | 8,385 | 1.97 % | 8 / 64 | 17.5 min | 6.8369 | 9.7594 | −0.00566 ± 0.00171 (better) | −0.00673 ± 0.00169 (better) |
| Mistral-7B-v0.3 | 8x64 | 123,180 | 0.90 % | 20 / 237 | 40.5 min | 5.4837 | 8.0334 | −0.00704 ± 0.00097 (better) | −0.00405 ± 0.00075 (better) |
| | **16x64** | 127,060 | 1.86 % | 10 / 105 | **18.5 min** | **5.4992** | **8.0338** | −0.00423 ± 0.00098 (better) | −0.00400 ± 0.00081 (better) |
| | 256x64 | 22,082 | 5.18 % | 14 / 132 | 22.4 min | 5.4950 | 8.0354 | −0.00498 ± 0.00098 (better) | −0.00380 ± 0.00074 (better) |
| Phi-4 | 8x64 | 57,098 | 0.21 % | 10 / 108 | 36.0 min | 6.6373 | 10.5216 | −0.00415 ± 0.00131 (better) | −0.00228 ± 0.00080 (better) |
| | **16x64** | 117,122 | 0.88 % | 13 / 126 | **43.5 min** | **6.6288** | **10.5113** | −0.00542 ± 0.00137 (better) | −0.00326 ± 0.00087 (better) |
| | 256x64 | 59,953 | 7.21 % | 14 / 85 | 35.3 min | 6.6519 | 10.5237 | −0.00195 ± 0.00147 (better) | −0.00208 ± 0.00083 (better) |

FourOverSix: 6.8757 / 9.8254 (Llama), 5.5225 / 8.0660 (Mistral), 6.6649 / 10.5456 (Phi-4).

**MR-OPT 16x64 minus the other units** (paired ΔNLL, mean ± 2 SE):

| model | vs 8x64: WikiText-2 | vs 8x64: C4 | vs 256x64: WikiText-2 | vs 256x64: C4 |
|---|---|---|---|---|
| Llama-3.1-8B | +0.00184 ± 0.00166 (worse) | −0.00238 ± 0.00248 | −0.00161 ± 0.00157 (better) | −0.00187 ± 0.00210 |
| Mistral-7B-v0.3 | +0.00281 ± 0.00090 (worse) | +0.00005 ± 0.00071 | +0.00075 ± 0.00087 | −0.00020 ± 0.00076 |
| Phi-4 | −0.00127 ± 0.00129 | −0.00098 ± 0.00076 (better) | −0.00347 ± 0.00137 (better) | −0.00118 ± 0.00085 (better) |

**Fake evaluation of MR-OPT 16x64** (ΔNLL vs FourOverSix):

| model | WikiText-2 | C4 |
|---|---|---|
| Llama-3.1-8B | 6.8166 (−0.01030 ± 0.00189) | 9.7446 (−0.00866 ± 0.00271) |
| Mistral-7B-v0.3 | 5.5002 (−0.00468 ± 0.00098) | 8.0353 (−0.00388 ± 0.00152) |
| Phi-4 | 6.6317 (−0.00526 ± 0.00134) | 10.5158 (−0.00264 ± 0.00089) |

All of these are significant.

**Checks.**
- Every evaluation process of a model used identical windows.
- In the new native processes, the FourOverSix, MR-OPT 8x64 and MR-OPT 256x64 window NLLs repeated
  the earlier evaluation bitwise; so did FourOverSix in the new fake processes.
- No map mismatches. The 16x64 maps expanded exactly to 8x64 tiles.

**Observations.**
- **MR-OPT 16x64 beats FourOverSix significantly** on both corpora on all three models, native and
  fake.
- **There is no consistent ranking of the three units:**
  - **Llama:** 16x64 lies between the others on WikiText-2, worse than 8x64 and better than
    256x64, both by small margins. C4 is inconclusive against both.
  - **Mistral:** 16x64 is worse than 8x64 on WikiText-2 and equal to 256x64 within noise.
  - **Phi-4:** 16x64 is the best of the three. It is better than 256x64 on both corpora and better
    than 8x64 on C4.
- **The search effort at 16x64 depends on the model:**
  - Llama took its longest search: 15 rounds and 45.4 min, against 28.0 min at 8x64.
  - Mistral took its shortest: 10 rounds and 18.5 min.
  - Phi-4 took 13 rounds and 43.5 min.

**Wall time:** 2 h 47 min in all (04:04 → 06:51 UTC):

| part | wall time |
|---|---:|
| checks | 21 min |
| MR-OPT Llama | 49 min |
| MR-OPT Mistral | 22 min |
| MR-OPT Phi-4 | 49 min |
| evaluations | 27 min |

## Reproduction

```
/home/dev/n16k64_campaign/mr_variants/queue.sh          # copy in runs/queue.sh, log in runs/commands.log
python results/mr_variants/analyze_variants.py MODEL    # MODEL/summary.json, MODEL/tables.md (llama8b, mistral7b, phi4)
/home/dev/n16k64_campaign/mr_variants/queue16.sh        # the 16x64 addendum (copy in runs/queue16.sh, log in runs/commands_16x64.log)
python results/mr_variants/analyze_16x64.py mropt       # mropt_16x64.json, mropt_16x64.md
```

The run records (`report.json`) are in `runs/<model>/`. The maps and per-document development
values stay in `/home/dev/n16k64_campaign/mr_variants/runs/`.
