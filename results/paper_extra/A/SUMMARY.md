# A summary: per-block selection rules coarsened to hardware tiles (Llama-3.1-8B)

**Run:** registered protocol (`PROTOCOL.md`, c923f32), 2026-09-29 17:47–19:12 UTC. 20 policies, one process each, all
exited 0 (`commands.log`); the records are in `ppl/llama8b/`.
- **Deviation 1** (analysis script only): three fixes to `A_analyze.py`, no GPU run repeated and no number changed.
- **Checks:**
  - every policy evaluated the same windows (token hashes);
  - the re-run references (BF16, NVFP4, FourOverSix, FlipQuant (ours) at 8x64/16x64/256x64) equal the Parts 2–3 fake
    (c) records, window by window;
  - zero-scale blocks: 0 in every arm.
- **Tables** in `A.md`; data in `A.csv`, `A_primary.csv`, `A_retention.csv`, `A_mixing.csv`, `A.json`. Paper tables:
  `A_table_llama8b_wiki.tex` (main), `A_table_llama8b_c4.tex` (appendix).

ΔNLL in nats per token, paired over windows, ± 2 SE, WikiText-2 / C4. Negative = the first policy is better.

**The hypothesis holds: the per-block rules lose their 1x16 gain at every hardware tile.**
- **(a) Degradation, R@g − R@1x16:** all 18 comparisons are significant and positive.
  - IF4 (Cook et al.): +0.0112 to +0.0135.
  - MixFP4 (Zou et al.): +0.0112 to +0.0133.
  - MixFP4 (Zou et al.) + FourOverSix: +0.0056 to +0.0098.
- **(b) Gain over the rule's own base:**
  - **At 1x16, every rule beats its base:** IF4 −0.0087 / −0.0109, MixFP4 (Zou et al.) −0.0105 / −0.0109, MixFP4 (Zou
    et al.) + FourOverSix −0.0070 / −0.0056.
  - **At 8x64, 16x64 and 256x64 no rule beats its base.**
    - IF4 is worse than its base: +0.0033 to +0.0037 on WikiText-2 (significant), and +0.0003 to +0.0026 on C4 (2 of 3
      significant).
    - MixFP4 (Zou et al.): +0.0008 to +0.0028 / +0.0002 to +0.0015 (one of six significant).
    - MixFP4 (Zou et al.) + FourOverSix: −0.0015 (not significant) at 8x64 on WikiText-2, otherwise +0.0012 to +0.0039
      (4 of 6 significant).
- **(c) FlipQuant (ours) against the rule at the same tile, tc@g − R@g:** −0.0110 to −0.0234, all 18 significant.
  - Against MixFP4 (Zou et al.) + FourOverSix, which has the same FourOverSix E2M1 base and the same uniform grid, so
    only the selection differs: −0.0126 / −0.0170 at 8x64, −0.0171 / −0.0179 at 16x64, −0.0127 / −0.0110 at 256x64.
- **(d) Retention of the 1x16 gain over FourOverSix:**
  - The 1x16 gain is significant for every rule on both corpora: +0.0058 / +0.0041 (IF4), +0.0068 / +0.0043 (MixFP4
    (Zou et al.)), +0.0070 / +0.0056 (+ FourOverSix).
  - **IF4 and MixFP4 (Zou et al.) retain less than nothing.** Their coarse tiles are worse than FourOverSix: −66 % to
    −113 % retained on WikiText-2, −160 % to −229 % on C4.
  - **MixFP4 (Zou et al.) + FourOverSix:** +21 % [−11, 45] at 8x64 on WikiText-2, and −17 % to −70 % elsewhere. The
    upper 95 % bound never exceeds 45 %.

**Mechanism.**
- **The preferences are mixed inside every tile.** From each rule's own 1x16 choices, 100.0 % of the 8x64, 16x64 and
  256x64 tiles contain both formats, for all three rules and every projection.
  - The minority format holds 37–38 % of a tile's blocks on average for IF4 and MixFP4 (Zou et al.), and 42–46 % for +
    FourOverSix.
- **The tile sum then elects the majority almost everywhere.** IF4 prefers INT4 in 62 % of its 16-blocks, and the
  uniform share becomes 91 % at 8x64, 96 % at 16x64 and 99 % at 256x64 (MixFP4 (Zou et al.): the same to within
  0.1 pp). So coarse IF4 / MixFP4 (Zou et al.) is close to uniform INT4 everywhere, which is no better than NVFP4.
  - MixFP4 (Zou et al.) + FourOverSix: 54 % → 49 / 48 / 41 %. FlipQuant (ours): 2.3 / 3.0 / 8.3 %.
- **Weight error does not predict the loss.**
  - MixFP4 (Zou et al.) + FourOverSix at 8x64 has 3.8 % less weight squared error than FourOverSix, and its ΔNLL is
    −0.0015 (not significant) / +0.0029 (significantly worse).
  - FlipQuant (ours) at 8x64 has the same weight error as FourOverSix (+0.1 %), and its ΔNLL is −0.0141 / −0.0141.
  - At 1x16 the rules cut the weight error by 18–20 %, for −0.0058 to −0.0070 on WikiText-2.

**Perplexity, WikiText-2 / C4** (all in the same fake (c) simulator):
- BF16 6.2403 / 8.9579.
- NVFP4 6.9296 / 9.9302.
- FourOverSix 6.8807 / 9.8141.
- NVFP4 weights + FourOverSix act. 6.9002 / 9.8809.
- FlipQuant (ours): 8x64 6.7846 / 9.6767; 16x64 6.7825 / 9.6778; 256x64 6.8022 / 9.7264.
- The best 1x16 rule is MixFP4 (Zou et al.) + FourOverSix at 6.8324 / 9.7595; at 8x64 it gives 6.8705 / 9.8422.

**Post hoc (not a registered comparison).** FlipQuant (ours) at each realizable tile is also significantly better than
each rule at its unrealizable 1x16:
- 8x64: −0.0070 to −0.0082 / −0.0085 to −0.0100;
- 16x64: −0.0073 to −0.0085 / −0.0084 to −0.0099;
- 256x64: −0.0044 to −0.0057 / −0.0034 to −0.0049.

FlipQuant (ours) uses calibration data, math and code windows from open-web-math and codeparrot-clean, never WikiText-2
or C4. The rules use none.

**Caveat.** FlipQuant (ours) here is the fake (c) simulator, so it differs slightly from the native main-table numbers.
One model so far; Mistral-7B-v0.3 and Phi-4 follow under the same protocol.
