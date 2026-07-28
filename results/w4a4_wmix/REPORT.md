# W4A4, weights mixed only: mix_4_6 @16x64 against the 4over6 baseline

**Every row on this page quantizes activations with plain `nvfp4_4over6`.** Only the weight
quantizer varies. That is the difference from `results/w4a4_models/`, where the `mix_4_6` rows
quantize *both* operands with `mix_4_6` and the weight-side and activation-side effects are
confounded.

Weight type block is `16x64` throughout (hardware-realizable: a union of two `n8 x k64` B-operand
tiles). Group size 16, seq len 2048, wikitext-2 and C4, `run_ppl_sweep.py --sweep w4a4_wmix`.

### llama-3.1-8b

| format | wikitext | c4 | dwikitext | dc4 |
|---|---|---|---|---|
| `nvfp4_4over6` (W+A) | 6.8838 | 9.8241 | - | - |
| `mix_4_6` @16x64 W | 6.8943 | 9.8573 | +0.0105 | +0.0332 |
| `mix_4_6_m2` @16x64 W | **6.8731** | **9.8207** | **-0.0107** | **-0.0034** |

### qwen3-8b

| format | wikitext | c4 | dwikitext | dc4 |
|---|---|---|---|---|
| `nvfp4_4over6` (W+A) | 10.0098 | 13.7516 | - | - |
| `mix_4_6` @16x64 W | 10.0759 | 13.7773 | +0.0661 | +0.0257 |
| `mix_4_6_m2` @16x64 W | 10.0106 | 13.7911 | +0.0008 | +0.0395 |

## Findings

**1. The W4A4 regression at 16x64 is entirely weight-side.** Holding the activations at plain 4/6
recovers essentially nothing relative to mixing both operands:

| model | both operands mixed | weights only mixed | difference |
|---|---|---|---|
| llama-3.1-8b | 6.8946 | 6.8943 | 0.0003 |
| qwen3-8b | 10.0765 | 10.0759 | 0.0006 |

(the both-operand numbers are from `results/w4a4_models/`). Under 0.001 ppl apart on both models.
Mixed-type activations were never the problem, so there is nothing to win back on the activation
side -- the type-block election on the weights is what costs the perplexity.

**2. Plain MSE + argmin loses on both models**, +0.0105 and +0.0661 wikitext. This is the predicted
outcome, not a surprise: `argmin` compares summed errors over the whole 16x64 tile, so a tile elects
E0M3 whenever the total favours it, and individual scale blocks inside that tile end up worse than
they would have been under plain 4/6. See the "Selection objective and election rules" section of
CLAUDE.md -- an aggregate MSE win of a few percent certifies nothing about layer output error.

**3. `margin=2` rescues llama-3.1-8b but only reaches parity on qwen3-8b.** It is a clear win on
llama (-0.0107 wikitext, -0.0034 c4, a 0.021 wikitext swing away from argmin). On qwen it lands on
top of the baseline on wikitext (+0.0008, i.e. the guard has degenerated to nearly-always-4/6) and
is **worse on c4** (+0.0395). So the guard reliably removes the *regression*, but it does not
deliver a consistent *improvement* over 4over6 at a realizable type block. The honest reading is
that `mix_4_6` at 16x64 is at best on par with plain 4over6, model-dependent, and only when the
election is guarded.

## A caveat on the baselines

The `nvfp4_4over6` row was re-measured in-process rather than taken from `results/w4a4_models/`,
which turned out to matter:

| model | stored | re-measured | difference |
|---|---|---|---|
| llama-3.1-8b | 6.8773 / 9.8177 | 6.8838 / 9.8241 | +0.0065 / +0.0064 |
| qwen3-8b | 10.0098 / 13.7516 | 10.0098 / 13.7516 | 0.0000 / 0.0000 |

qwen reproduces bit-for-bit; llama-3.1-8b does not, despite the same script, the same seeded
evaluation data, and the same commit. The cause is not established -- most likely a different GPU
(and therefore different cuBLAS kernel selection) in the earlier run. **The deltas above are all
against the in-process baseline.** Had the stored llama number been used instead, the plain-MSE
regression would have read +0.0170 rather than +0.0105 and the `margin=2` win would have read
-0.0042 rather than -0.0107. Any future comparison at this effect size (~0.01 ppl) should
re-measure its own baseline in the same process.
