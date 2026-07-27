# How much does the E0M3 branch actually get used?

Share of weight elements that differ from the SAME configuration with `elect="never"`, i.e. with the
E0M3 branch switched off. Llama-3.1-8B, first seven linear layers, type block 8x64.

| config | q_proj | k_proj | v_proj | o_proj | gate | up | down |
|---|---|---|---|---|---|---|---|
| `clipheade0_h1.5` | 1.1% | 0.8% | 0.3% | 21.5% | 20.1% | 21.4% | 22.1% |
| `h1.5` (plain 4over6 candidate set) | 0.6% | 0.4% | 0.2% | 16.3% | 14.8% | 15.9% | 16.7% |
| `clipbothx_clipmin0.3_h1.5` | 0.3% | 0.2% | 0.0% | 11.5% | 10.4% | 11.5% | 11.8% |
| **`clipdense9_h1.5`** (best W4A4) | 0.1% | 0.1% | 0.0% | 5.0% | 4.1% | 4.5% | 4.9% |
| `clipbothx_clipmin0.3_h3` | 0.0% | 0.0% | 0.0% | 0.9% | 0.8% | 0.8% | 0.9% |

Three readings:

1. **E0M3 is essentially unused on q/k/v_proj** (<= 1.1% under every configuration). It appears only
   on o_proj and the MLP.
2. **The richer the E2M1 scale search, the LESS E0M3 is elected — and the better the result.**
   `clipheade0_h1.5` uses E0M3 on 22% of down_proj and scores -0.0174 at W4A4; `clipdense9_h1.5`
   uses it on 4.9% and scores -0.0323. The ordering is inverted: more E0M3 is worse.
3. **`clipbothx_clipmin0.3_h3` is E2M1-only in all but name** (<= 0.9%). Its gain is the gated
   clipping, not the element-type decision.

Caveat: weights only. W4A4 also quantizes activations, where `nvif4` helps more (-0.0779 against
-0.0457 at W4A16), so activation-side usage may be higher and is measured separately.

Reproduce: `results/e0m3_usage/weights_llama-3.1-8b_8x64.txt` holds the raw output.


## Activation side (W4A4), same measurement

Real wikitext activations from `model.layers.0` of Llama-3.1-8B, A-operand type block 16x64,
**no rotation anywhere**. Share of elements differing from the same config with `elect="never"`:

| config | q/k/v in | o_proj in | gate/up in | down in |
|---|---|---|---|---|
| `h1.5` (plain 4over6 candidate set) | 0.0% | 5.8% | 2.2% | 0.2% |
| `clipbothx_clipmin0.3_h1.5` | 0.0% | 4.6% | 1.5% | 0.2% |
| `h3` | 0.0% | 3.3% | 0.0% | 0.1% |
| **`clipdense9_h1.5`** (best W4A4) | 0.0% | 2.5% | 0.3% | 0.2% |
| `clipbothx_clipmin0.3_h3` | 0.0% | 1.3% | 0.0% | 0.1% |

E0M3 is **never** elected for the q/k/v inputs — the post-layernorm hidden state — and reaches at
most ~6% on the attention output. Usage is even lower than on weights, and the same inversion holds:
the best-scoring configuration uses E0M3 the least.

## What this means for "does E0M3 help?"

At the hardware-realizable type block, E0M3 is elected on **under 6% of elements in either operand**,
and the configurations that elect it most score worst. Its value is real but lives entirely at a
granularity the MMA cannot express — on Llama-3.1-8B W4A4, without any rotation:

| | dwikitext | dc4 | mean |
|---|---|---|---|
| `mix_4_6_1x16` — element type per 16-element scale block | -0.0810 | -0.1006 | **-0.0908** |
| `nvif4` — same, without the 4/6 choice | -0.0689 | -0.0868 | -0.0779 |
| `h1.5` @ 8x64 — element type per realizable tile | -0.0079 | -0.0085 | -0.0082 |
| E2M1 only | +0.0024 | +0.0002 | +0.0013 |

**An 11x gap between the per-block choice and the realizable one.** That gap, not the choice rule,
is the thing standing between MixFP4 and a real gain.
