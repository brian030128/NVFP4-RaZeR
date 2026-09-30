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

## Perplexity, WikiText-2 / C4 (the same fake (c) simulator for every row)

| model | FourOverSix | NVFP4 | best rule at 1x16 (by WikiText-2) | the same rule at 8x64 | FlipQuant (ours) 8x64 | FlipQuant (ours) 16x64 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 6.8807 / 9.8141 | 6.9296 / 9.9302 | + FourOverSix 6.8324 / 9.7595 | 6.8705 / 9.8422 | 6.7846 / 9.6767 | 6.7825 / 9.6778 |
| Mistral-7B-v0.3 | 5.5210 / 8.0681 | 5.5508 / 8.0949 | + FourOverSix 5.5028 / 8.0482 | 5.5120 / 8.0612 | 5.4886 / 8.0216 | 5.4882 / 8.0221 |
| Phi-4 | 6.6627 / 10.5473 | 6.6998 / 10.5834 | IF4 6.6451 / 10.5195 | 6.6692 / 10.5602 | 6.6115 / 10.4955 | 6.6118 / 10.5056 |

"+ FourOverSix" is MixFP4 (Zou et al.) + FourOverSix.

## Post hoc (not a registered comparison)

FlipQuant (ours) at 8x64 and 16x64 is also significantly better than every rule at its unrealizable 1x16, on every
model and both corpora:
- Llama: −0.0070 to −0.0100;
- Mistral: −0.0026 to −0.0040;
- Phi-4: −0.0012 to −0.0066.

At 256x64 it is too, except on Phi-4's C4 (−0.0002 to −0.0005, not significant).

FlipQuant (ours) uses calibration data: math and code windows from open-web-math and codeparrot-clean, never WikiText-2
or C4. The rules use none.

**Caveat.** FlipQuant (ours) here runs in the fake (c) simulator, so its numbers differ slightly from the native
main-table numbers.
