# MR-OPT variants: significant steps and warm start — report

**Status (2026-09-25 20:00 UTC).**
- **Llama-3.1-8B:** done (this section).
- **Mistral-7B-v0.3:** running.
- **Phi-4:** queued.
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

- **Llama alone decides the recommendation.** The recommendation needs a variant that is
  acceptable at both units on *every* tested model. No variant can now meet that, whatever
  Mistral-7B-v0.3 and Phi-4 show, so **MR-OPT stays**.
  - The remaining variant runs on Mistral and Phi-4 can only describe how the variants behave on
    other models.
  - This is a report only; the user decides.
- **The secondary reading gives the same answer.** No variant passes it on Llama either.
- **The variants are much faster.** Total Llama optimization time over both units:
  - MR-OPT: 45.5 min;
  - MR-OPT+SIG: 20.1 min (−56 %);
  - MR-OPT+WS: 20.6 min (−55 %);
  - MR-OPT+SIG+WS: 12.7 min (−72 %).

  Every variant map is still significantly better than FourOverSix on both corpora at both units.
  They keep part of MR-OPT's gain, not all of it (table below).

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

Pending.

## Phi-4

Pending.

## Reproduction

```
/home/dev/n16k64_campaign/mr_variants/queue.sh          # copy in runs/queue.sh, log in runs/commands.log
python results/mr_variants/analyze_variants.py llama8b  # llama8b/summary.json, llama8b/tables.md
```

The run records (`report.json`) are in `runs/<model>/`. The maps and per-document development
values stay in `/home/dev/n16k64_campaign/mr_variants/runs/`.
