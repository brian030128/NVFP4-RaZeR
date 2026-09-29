# D summary: decode latency with a CUDA graph over a StaticCache

**Run:** registered protocol (`PROTOCOL.md`, 36d5b97), 2026-09-29 19:29–20:54 UTC, idle GPU. **No deviations.**
- 105 processes: 3 models × 7 policies × 5 rounds, the policy order shuffled per round. All exited 0
  (`commands.log`).
- **The registered check passed in all 630 (policy, round, setting) records:** the graph's first 33 greedy tokens
  equal an eager StaticCache decode's. Coverage passed in every native process (asserted in-process).
- **Records** in `decode/<model>/<policy>/round<r>.json`. Tables in `D.md`, data in `D.csv` and `D.json`.
- **Qwen3.8-27B is out of scope:** the decode harness does not support its hybrid cache.
- **The measurement is stable.** FourOverSix's tokens per second varies across the 5 rounds by at most 0.16 %, and by
  0.29–0.61 % at 16x2048.

Tokens per second, the median of 5 rounds. Changes are paired within rounds. Settings are batch × prompt, then 64
generated tokens.

**FlipQuant (ours) costs little at decode.**
- **16x64, the mixed weights-on-A set (`auto`), vs FourOverSix (`auto_stock`):**
  - Llama −0.2 to −0.6 %, Mistral −0.1 to −0.6 %, Phi-4 −0.2 to −0.4 %.
  - Negative in every round, except at 16x2048, where the range crosses zero in 1 round of 5.
  - Both sets decode at width 16 for batch 1, 4 and 16 (the tile table).
- **8x64 (`n8k64_wB`) vs FourOverSix on `stock_wB`, the same placement:**
  - Llama −0.9 to −2.9 %, Mistral −0.8 to −3.1 %, Phi-4 −0.6 to −1.9 %.
  - Negative in every round; largest at 1x512.

**The placement costs far more than the format.**
- `stock_wB` and `n8k64_wB` have one 128-wide build and no small-T widths.
- So FourOverSix on wB decodes 4.1–25.7 % slower than FourOverSix on the weights-on-A set. Example, Llama 1x512:
  138.3 vs 104.5 tokens/s.
- Against FourOverSix on weights-on-A, FlipQuant (ours) 8x64 is −4.7 to −28.0 %, almost all of it placement.

**Other references.**
- **NVFP4 vs FourOverSix:** −0.1 to −1.3 % on the same weights-on-A kernels, and −0.2 to −1.0 % on wB. The cause
  was not isolated: the two differ in both the weights and the activation quantizer.
- **FP4 vs BF16:** FourOverSix decodes 1.18–1.77× faster than BF16 on Llama and Mistral, and 1.24–2.06× on Phi-4.
  The gain is largest at batch 1 and shrinks at 16x2048, where attention over the cache dominates.

**Recorded, not a check: agreement with an eager DynamicCache decode.**
- The lowest agreement per policy is 3–76 % of tokens.
- BF16 shows the same early divergence (lowest 39–76 %), so it is not a quantization effect. Greedy decodes from
  random-token prompts part ways early between two numerically different attention paths.
- The registered comparison, graph vs eager on the same StaticCache, is exact everywhere.
