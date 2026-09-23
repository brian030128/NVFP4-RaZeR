# Multi-round relinearized election at 256×64 (Llama-3.1-8B)

One-shot election scores every tile once, at FourOverSix, and elects all tiles
whose first-order bound is negative. Multi-round election instead works like
training with a line search:

1. **Score.** At the *current* quantized model, compute per-sequence CE and/or
   KL(BF16 teacher) gradients on the 128 calibration sequences. Take the
   directional score of every legal flip, E2M1→E0M3 or undo.
2. **Rank.** Candidates are flips whose bound (mean + 2 SE) is negative. The
   2 SE filter is fixed a priori (from the earlier relinearized study) and was
   not tuned.
3. **Backtrack.** Apply the top n, n/2, n/4, … candidates. Accept the first step
   that lowers the objective's loss on the 192 held-out development documents
   (math/code, tensor-wide W4A4).
4. **Repeat** until no step lowers it. Evaluate WikiText-2/C4 once, on the final
   map.

Scoring and acceptance use the same objective: CE, KL, or both (both dev losses
must fall). Implementation: `run_multiround.py`; jobs 423826–423828.

## Results versus FourOverSix (6.875525 / 9.823733)

| Selection | E0M3 tiles | WikiText-2 | ΔWiki | C4 | ΔC4 | paired ΔNLL ± 2SE, wiki / c4 |
|---|---:|---:|---:|---:|---:|---|
| One-shot CE+KL k=3 (current report) | 187 | 6.866879 | −0.0086 | 9.801361 | −0.0224 | −0.0013±0.0016 / −0.0023±0.0015 |
| One-shot CE+KL k=1.5 (dev-selected) | 15,399 | 6.856118 | −0.0194 | 9.776559 | −0.0472 | −0.0028±0.0018 / −0.0048±0.0023 |
| **Multi-round CE+KL** | 8,132 | **6.836714** | **−0.0388** | 9.777874 | −0.0459 | **−0.0057±0.0018 / −0.0047±0.0016** |
| Multi-round KL | 8,405 | 6.841998 | −0.0335 | 9.774137 | −0.0496 | −0.0049±0.0019 / −0.0051±0.0022 |
| Multi-round CE | 12,491 | 6.852695 | −0.0228 | 9.773471 | −0.0503 | −0.0033±0.0020 / −0.0051±0.0022 |
| 8×64 CE+KL k=3 (report) | 3,345 | 6.849275 | −0.0262 | 9.773040 | −0.0507 | — |
| 1×16 MSE ceiling (not deployable) | 53% of blocks | 6.833680 | −0.0418 | 9.762418 | −0.0613 | −0.0061±0.0020 / −0.0063±0.0028 |

Multi-round CE+KL on the deployable 256×64 tile recovers **93% (WikiText) / 75%
(C4)** of the 1×16 MSE ceiling. It beats the 8×64 k=3 map on WikiText. No choice
in the procedure looked at WikiText or C4.

## What it changes about earlier conclusions

- **KL-only is not inherently harmful.** One-shot KL-only at 8×64 was catastrophic
  (report: 7.3566 WikiText), and one-shot 1×16 KL reached 24–259 PPL. With
  re-scoring and backtracking, KL-only is competitive (−0.0335 / −0.0496). The
  earlier failure was the one-shot step, not the KL objective.
- **CE+KL versus CE-only still favours the conjunction, for a new reason.**
  CE-only reaches the lowest development CE (1.46901 vs 1.46944) but the worst
  WikiText (−0.0228). The KL term transfers across domains; CE fit on math/code
  does not.
- **The k threshold stops mattering.** A fixed 2 SE noise filter plus measured-loss
  backtracking replaces the threshold sweep.

## Mechanism evidence from the backtracking logs

- **The first-order prediction overshoots.** In round 0 the summed tile scores
  predicted ΔCE −0.110 (CE) and −0.043 (CE+KL); the measured values were
  −0.0115 and −0.0091. KL-only's predicted −0.066 KL gave −0.0013.
- **Later rounds fail at the top of the ranking.** After a step, the
  highest-ranked re-scored flips are often harmful even singly: one CE-only tile
  predicted −0.0006 and measured +0.0019. Many accepted steps are close to net
  zero in tile count (e.g. KL round 2: 151 flips, net +1 tile), i.e. mostly
  undos. The linearization at the new point frequently points back toward the
  old one, so each round's gain shrinks until none survives.
- **Halving keeps the top of the ranking.** Backtracking always retains the most
  confident flips, so a single mis-scored top tile blocks a round. Pruning
  individually harmful flips, instead of halving, is the obvious refinement.

## Related results from this session

- **Fresh gate (dev-selected one-shot k=1.5, criterion frozen before sampling).**
  64 new math/code documents; CE −0.00737 ± 0.00546 vs raw256 and −0.01860 ±
  0.00777 vs FourOverSix. **Passed.** (`k_fresh_llama_*.json`, job 423754)
- **Qwen3.8-27B, one-shot CE+KL 256×64, Llama-chosen thresholds.**
  - FourOverSix and k=3 reproduce the report exactly (7.287076/10.188365 and
    7.266300/10.176030).
  - k = 2.5 / 2 / 1.5 / 1 give ΔWiki −0.047 / −0.067 / −0.113 / −0.167 and
    ΔC4 −0.015 / −0.034 / −0.052 / −0.069.
  - The 8×64 k=3 map is −0.072 / −0.038, so on Qwen the 256×64 tile at k≤1.5
    beats 8×64 on both corpora.
  - This is the threshold sweep that was stopped as parameter search; it is kept
    as evidence that the one-shot k=3 map under-elects on Qwen as well.
    (`qwen27b/`, jobs 423757–423762)

## Not yet done

- A fresh-document gate for the multi-round CE+KL map. Its development documents
  were used for acceptance, so they are no longer held out.
- Multi-round on Qwen3.8-27B, and at 1×16 (the one-shot catastrophe) as a ceiling.
- Backtracking that prunes harmful flips instead of halving.
