# A summary: per-block selection rules coarsened to hardware tiles (Llama-3.1-8B, Mistral-7B-v0.3, Phi-4)

**Run:** registered protocol (`PROTOCOL.md`, c923f32). 60 policies, one process each, all exited 0 (`commands.log`); the
records are in `ppl/<model>/`.
- **Times (UTC):** Llama-3.1-8B 2026-09-29 17:47–19:12; Mistral-7B-v0.3 20:54–22:20; Phi-4 22:20 to 2026-09-30 00:30.
- **Deviation 1** (analysis script only): three fixes to `A_analyze.py`. No GPU run was repeated and no number
  changed.
- **Checks, on every model:**
  - every policy evaluated the same windows (token hashes);
  - the re-run references (BF16, NVFP4, FourOverSix, FlipQuant (ours) at 8x64/16x64/256x64) equal the Parts 2–3 fake
    (c) records, window by window;
  - zero-scale blocks: 0 in every arm.
- **Tables** are in `A.md`; the data in `A.csv`, `A_primary.csv`, `A_retention.csv`, `A_mixing.csv` and `A.json`.
- **Paper tables:** `A_table_<model>_<corpus>.tex`. Llama's WikiText-2 table is the main one; the rest go to the
  appendix.
- **Amendment 2** (registered 9883703, 2026-09-30) adds a fourth arm, IF4 (Cook et al.) + FourOverSix (our variant), in
  12 more policies (06:57–07:59 UTC, all exited 0).
  - Its registered CPU check passed. On 11 modules its candidates equal Zou + FourOverSix's FP candidate and IF4's
    INT4 candidate bitwise, and ties keep FP. The existing rules' outputs are unchanged (30 of 30).
  - Its windows equal every other policy's.
  - The existing arms and references were not re-run.
  - The primary comparisons and the mechanism below cover the three original rules. The fourth arm has its own
    section.

ΔNLL in nats per token, paired over windows, WikiText-2 and C4. "Significant" means |Δ| > 2 SE. Negative means the first
policy is better. Each range below spans rules, tiles (8x64, 16x64, 256x64) and both corpora.

## The primary comparisons

**(a) Coarsening degrades every rule.** R@g − R@1x16 is significant and positive in all 54 comparisons.

| model | range |
|---|---|
| Llama-3.1-8B | +0.0056 to +0.0135 |
| Mistral-7B-v0.3 | +0.0016 to +0.0050 |
| Phi-4 | +0.0018 to +0.0057 |

**(b) The gain over the rule's own base survives only in part, and only on Mistral's WikiText-2.**
- **At 1x16, every rule beats its base,** in all 18 comparisons: Llama −0.0056 to −0.0109, Mistral −0.0025 to −0.0057,
  Phi-4 −0.0019 to −0.0049.
- **At the tiles:**

  | model | significantly better than the base | significantly worse |
  |---|---|---|
  | Llama-3.1-8B | 0 of 18 | 10 of 18 |
  | Mistral-7B-v0.3 | 6 of 18, all on WikiText-2: IF4 at every tile (−0.0015 to −0.0029), MixFP4 (Zou et al.) at 8x64 and 256x64, + FourOverSix at 8x64 (−0.0016) | 0 |
  | Phi-4 | 0 of 18 | 3 of 18 (+ FourOverSix on WikiText-2, +0.0019 to +0.0029) |

**(c) FlipQuant (ours) beats every rule at the same tile.** tc@g − R@g is significant and negative in all 54
comparisons.

| model | against all rules | against MixFP4 (Zou et al.) + FourOverSix |
|---|---|---|
| Llama-3.1-8B | −0.0110 to −0.0234 | −0.0110 to −0.0179 |
| Mistral-7B-v0.3 | −0.0043 to −0.0085 | −0.0043 to −0.0067 |
| Phi-4 | −0.0033 to −0.0114 | −0.0033 to −0.0099 |

MixFP4 (Zou et al.) + FourOverSix shares FlipQuant's FourOverSix E2M1 base and its uniform grid, so only the selection
method differs.

**(d) Retention of the 1x16 gain over FourOverSix.**
- **The 1x16 gain is significant in 17 of 18 (model, rule, corpus) cases.** The exception is MixFP4 (Zou et al.) on
  Phi-4 WikiText-2 (+0.0011 ± 0.0016), so it has no fraction.
- **42 of the 51 retained fractions are negative:** there, the coarse rule is worse than FourOverSix.
  - **IF4 (Cook et al.)** retains −229 % to +9 %.
  - **MixFP4 (Zou et al.)** retains −188 % to −43 %.
  - **MixFP4 (Zou et al.) + FourOverSix** retains −150 % to +49 %.
- **Only one fraction is significantly above zero:** + FourOverSix at 8x64 on Mistral WikiText-2, 49 % [23, 79].
- **The intervals are wide on Mistral and Phi-4,** where the 1x16 gains are small (+0.0011 to +0.0033 nats). For
  example, IF4 at 8x64 on Mistral C4 retains 9 % [−159, 86].

## Mechanism

- **The preferences are mixed inside practically every tile.** Take each rule's 1x16 choices and group them into tiles:
  on every model, at least 99.999 % of the 8x64 tiles contain both formats, and all but at most one of the 16x64 and
  256x64 tiles do.
  - The minority format holds 36.5–38.1 % of a tile's blocks on average for IF4 and MixFP4 (Zou et al.), and 42.1–46.4 %
    for + FourOverSix.
  - That is close to what independent per-block choices would give. So the preferences do not cluster at the tile
    scale.
- **The tile sum then elects the majority.** IF4 and MixFP4 (Zou et al.) prefer the uniform grid in 62–63 % of 16-blocks.
  Their uniform share becomes 91–94.5 % at 8x64, 96–98.4 % at 16x64 and 99.1–99.7 % at 256x64. So the coarse rules are
  close to uniform INT4 everywhere.
  - MixFP4 (Zou et al.) + FourOverSix: 53–54 % at 1x16. It falls with coarsening on Llama and Mistral (to 41 % and 36 %
    at 256x64) and rises on Phi-4 (to 69 %).
  - FlipQuant (ours): 1.6–2.5 % at 8x64, 2.2–3.3 % at 16x64, 6.4–10.0 % at 256x64.
- **Weight error does not predict the loss.**
  - MixFP4 (Zou et al.) + FourOverSix at 8x64 has 3.7–4.4 % less weight squared error than FourOverSix on every model.
    Its ΔNLL against FourOverSix is −0.0016 to +0.0029: one of six significantly better, two significantly worse.
  - FlipQuant (ours) at 8x64 has the same weight error as FourOverSix (−0.0 to +0.1 %). Its ΔNLL is −0.0141 / −0.0141
    (Llama), −0.0059 / −0.0058 (Mistral) and −0.0077 / −0.0049 (Phi-4), all significant.

## Amendment 2: IF4 (Cook et al.) + FourOverSix (our variant)

**The arm.** It mirrors MixFP4 (Zou et al.) + FourOverSix.
- **FP candidate:** FourOverSix, the E2M1 base of our maps.
- **Uniform candidate:** IF4's INT4, with a shared max/6 scale and the 7/6 and 6/7 constants.
- **Selection:** per block or tile sum; ties keep FP.
- **Provenance:** Cook et al. offer IF4 and FourOverSix as separate schemes, so the combination is ours.

**(a) Coarsening degrades it:** 16 of 18 comparisons are significant, all positive.

| model | range | not significant |
|---|---|---|
| Llama-3.1-8B | +0.0070 to +0.0127 | — |
| Mistral-7B-v0.3 | +0.0006 to +0.0054 | C4 at 8x64 and 256x64 |
| Phi-4 | +0.0015 to +0.0050 | — |

**(b) Gain over its base, FourOverSix.**
- **At 1x16 it is significantly better on every model and corpus:** Llama −0.0071 / −0.0055, Mistral −0.0039 / −0.0015,
  Phi-4 −0.0036 / −0.0026.
- **At the tiles:**
  - Llama: 3 of 6 significantly worse (+0.0034 to +0.0055), none better.
  - Mistral: 1 of 6 significantly worse (+0.0015, 8x64 WikiText-2), none better.
  - Phi-4: 1 of 6 significantly better (−0.0011, 8x64 C4), none worse.

**(c) FlipQuant (ours) beats it at the same tile** in all 18 comparisons: Llama −0.0105 to −0.0199, Mistral −0.0038 to
−0.0074, Phi-4 −0.0032 to −0.0091.

**(d) Retention of the 1x16 gain over FourOverSix.**
- The 1x16 gain is significant in all 6 (model, corpus) cases.
- 13 of the 18 retained fractions are negative.
- **Only one is significantly above zero:** Phi-4 C4 at 8x64, 41 % [3, 74].
- Mistral C4 at 8x64 retains 58 % [0, 134], on a 1x16 gain of only +0.0015.

**The contrast: IF4 + FourOverSix − MixFP4 (Zou et al.) + FourOverSix,** i.e. the two uniform candidates on the same
FourOverSix base.
- **At 1x16 the two differ by at most 0.0017:** Llama −0.0001 / +0.0001, Mistral −0.0006 / +0.0009, Phi-4 −0.0017 /
  −0.0001. Two of these are significant: Mistral's C4 and Phi-4's WikiText-2.
- **At the tiles the differences are small and change sign.**
  - WikiText-2: Llama +0.0033 and +0.0028 (8x64, 16x64); Mistral +0.0032, +0.0012 and −0.0010 (256x64); Phi-4 −0.0017
    (8x64). These are significant.
  - C4: every difference is within ±0.0009.
- **So neither uniform candidate is consistently better.** IF4's INT4 and Zou's E1M2 are the same ±7 integer grid.
  They differ only in where the E4M3 rounding of the scale happens (max/6 then × 6/7, or max/7).

**Mechanism.** Unchanged from MixFP4 (Zou et al.) + FourOverSix.
- At 1x16, ≥ 99.999 % of the tiles contain both formats; the minority holds 42.2–46.5 % of a tile.
- **Uniform share:**

  | model | 1x16 | 8x64 | 16x64 | 256x64 |
  |---|---:|---:|---:|---:|
  | Llama-3.1-8B | 53.5 % | 48.9 % | 47.8 % | 41.2 % |
  | Mistral-7B-v0.3 | 53.3 % | 47.7 % | 46.1 % | 35.6 % |
  | Phi-4 | 54.3 % | 54.3 % | 55.5 % | 67.5 % |

- **Weight error vs FourOverSix:** −20.2 to −20.8 % at 1x16, and −3.7 to −4.4 % at 8x64, where the arm gains nothing
  significant except Phi-4's C4.

## Perplexity, WikiText-2 / C4 (the same fake (c) simulator for every row)

| model | FourOverSix | NVFP4 | best rule at 1x16 (by WikiText-2) | the same rule at 8x64 | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 6.8807 / 9.8141 | 6.9296 / 9.9302 | IF4 + FourOverSix 6.8319 / 9.7601 | 6.8929 / 9.8476 | 6.7846 / 9.6767 | 6.7825 / 9.6778 |
| Mistral-7B-v0.3 | 5.5210 / 8.0681 | 5.5508 / 8.0949 | IF4 + FourOverSix 5.4998 / 8.0557 | 5.5295 / 8.0610 | 5.4886 / 8.0216 | 5.4882 / 8.0221 |
| Phi-4 | 6.6627 / 10.5473 | 6.6998 / 10.5834 | IF4 + FourOverSix 6.6386 / 10.5203 | 6.6637 / 10.5362 | 6.6115 / 10.4955 | 6.6118 / 10.5056 |

"IF4 + FourOverSix" is IF4 (Cook et al.) + FourOverSix (our variant), amendment 2. It is the best 1x16 rule by WikiText-2
on all three models.
- Among the three original rules, the best were MixFP4 (Zou et al.) + FourOverSix on Llama (6.8324 / 9.7595 → 6.8705 /
  9.8422 at 8x64) and on Mistral (5.5028 / 8.0482 → 5.5120 / 8.0612).
- On Phi-4 it was IF4 (6.6451 / 10.5195 → 6.6692 / 10.5602).

## Post hoc (not a registered comparison)

FlipQuant (ours) at 8x64 and 16x64 is also significantly better than every rule at its unrealizable 1x16, on every
model and both corpora:
- Llama: −0.0070 to −0.0100;
- Mistral: −0.0026 to −0.0040;
- Phi-4: −0.0012 to −0.0066.

At 256x64 it is too, except on Phi-4's C4 (−0.0002 to −0.0005, not significant).

Against IF4 (Cook et al.) + FourOverSix at 1x16 the same holds at 8x64 and 16x64, on every model and corpus:
- Llama: −0.0069 to −0.0086;
- Mistral: −0.0020 to −0.0042;
- Phi-4: −0.0014 to −0.0041.

At 256x64 it holds except on Phi-4 (−0.0014 / −0.0004, not significant).

FlipQuant (ours) uses calibration data: math and code windows from open-web-math and codeparrot-clean, never WikiText-2
or C4. The rules use none.

**Caveat.** FlipQuant (ours) here runs in the fake (c) simulator, so its numbers differ slightly from the native
main-table numbers.
