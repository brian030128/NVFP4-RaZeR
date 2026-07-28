# W4A4, weights mixed only: what it takes to make the E0M3 election pay at 16x64

Every row here quantizes **weights** with a `mix_4_6` variant at type block `16x64` and
**activations** with plain `nvfp4_4over6`. Group size 16, seq len 2048, wikitext-2 + C4.
All numbers measured on **RTX A6000 only** -- see "Two GPU architectures" below.

Baselines (`nvfp4_4over6` weights and activations, calibration-free):

| model | wikitext | c4 |
|---|---|---|
| llama-3.1-8b | 6.8773 | 9.8177 |
| qwen3-8b | 10.0208 | 13.7555 |

## Every selection rule tried, as deltas against those baselines

| rule | what it does | llama dwiki / dc4 | qwen dwiki / dc4 |
|---|---|---|---|
| `mix_4_6` | argmin of summed weight MSE | +0.0212 / +0.0296 | +0.0481 / +0.0339 |
| `mix_4_6_b3` | bounded harm, eps=3 | +0.0205 / +0.0299 | +0.0804 / +0.0408 |
| `mix_4_6_b2` | bounded harm, eps=2 | -0.0038 / +0.0003 | +0.0329 / +0.0188 |
| `mix_4_6_p0.7` | HQQ-style robust loss, sum abs(e)^0.7 | +0.0093 / +0.0119 | -0.0145 / +0.0118 |
| `mix_4_6_m2` | margin z=2 on the mean gain | +0.0012 / -0.0016 | -0.0178 / +0.0166 |
| `mix_4_6_dom` | elect only if no block harmed | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| `mix_4_6_hesst` | CALIBRATED election, argmin | +0.0018 / -0.0012 | +0.0414 / +0.0174 |
| **`mix_4_6_hesst_m1`** | **CALIBRATED election + margin 1** | **-0.0019 / -0.0049** | **-0.0084 / +0.0031** |

## Findings

**1. The regression is weight-side, and it is a selection failure, not a format limitation.**
Holding activations at 4/6 reproduces the both-operands-mixed numbers to within 0.001 ppl, so mixed
activations were never the cause. And `quant_mix_4_6(elect="never")` is now *bit-identical* to
`quant_nvfp4_4over6` on every real tensor tested, so the format is a strict superset: anything worse
than 4over6 is the chooser picking wrong.

**2. MSE is not merely a weak proxy, it is actively misleading.** `argmin` *always* lowers total
weight MSE -- it cannot do otherwise, since electing E2M1 everywhere is available -- by +3.4%
(gaussian) to +23.2% (heavy-tail) on synthetic data. It still loses 0.02-0.05 ppl on both models.
The reason is visible per block: argmin harms 24-40% of scale blocks and pays for it with larger
wins elsewhere. The MSE ledger nets positive; perplexity does not.

**3. Calibrating the TYPE election alone recovers almost all of the loss.** `hesst` weights only the
E2M1-vs-E0M3 comparison by `E[x_j^2]` (wikitext TRAIN split, never the eval data). The
per-scale-block 4-vs-6 scale search stays on plain MSE -- verified bit-identical to the
uncalibrated path, with `importance_scope="all"` as a passing negative control -- and the
`nvfp4_4over6` baseline needs no calibration at all. Combined with a mild margin (`hesst_m1`), it is
the only configuration that improves 3 of 4 cells, and its worst cell (+0.0031) is an order of
magnitude better than every other rule's worst cell.

**4. But the surviving gain is small, and it is not universal.** -0.0019/-0.0049 (llama) and
-0.0084/+0.0031 (qwen). E0M3 at a hardware-realizable 16x64 tile is best described as *roughly
free* once elected correctly, not as a source of accuracy. For comparison, the data-free dense
headroom search on the same setup (`results/decide_w4a4_final/`, parallel investigation) gives
-0.0187/-0.0247 on llama **with the E0M3 election switched off entirely**. The value at this
granularity is in the per-scale-block scale choice, not the type-block election.

**5. Calibration alone is not enough; it needs the guard.** `hesst` with plain argmin is neutral on
llama and clearly bad on qwen (+0.0414). Only `hesst_m1` goes negative. This matches
`quantize/importance.py`: the diagonal estimate of `S = E[x x^T]` has a residual bounded by the
off-diagonal mass, and the margin is what absorbs it.

**6. Rules that guarantee safety guarantee nothing else.** `dominance` elects E0M3 on **0.0%** of
16x64 tiles -- verified bit-identical to `elect="never"` -- so it is exactly 4over6. With 64 scale
blocks per tile, requiring unanimity is unachievable. The bounded-harm rule has the same problem
from the other side: its useful window is narrow (b1 fires on 0.1% of tiles, b2 on 13.0%, b3 on
37.2% against argmin's 37.9%), and b3 has already collapsed into argmin.

## Two things that invalidated earlier conclusions

**Two GPU architectures on this machine.** GPUs 0-3 are RTX A6000, GPUs 4-6 are RTX 6000 Ada.
The same configuration gives *deterministic but different* perplexity on each: llama-3.1-8b
`nvfp4_4over6` is 6.8773 on A6000 and 6.8838 on Ada; qwen3-8b is 10.0098 on Ada and 10.0208 on
A6000. Within one architecture the pipeline is bit-reproducible -- three independent A6000 runs of
the same config agreed to the last bit (6.877294063568115) -- so there is no run-to-run noise floor
and a 0.005 difference is real. Across architectures a 0.006-0.011 offset is not interpretable at
all. **Never mix the two in one table.**

**A rounding tie-break bug (now fixed).** `_quant_e2m1` rounded ties away from zero
(`floor(|m|+0.5)`) while `quant_nvfp4_4over6` rounds them toward zero (`ax <= mid_value[i]`). The
rules agree everywhere except exact midpoints -- but the block scale is `float8_e4m3fn` with a 3-bit
mantissa, so real weights land on midpoints for **0.158%** of elements (100% of the divergences were
verified to be exact ties). This gave every `mix_4_6` row a spurious +0.0017 wikitext / +0.0090 c4
handicap, larger than most effects being measured. Fixing it did **not** shift rows by a constant:
correcting the tie-break changes the per-block errors and therefore which tiles get elected, so
llama's `m2` went from -0.0088 (apparent win) to +0.0012 (neutral). **All pre-fix numbers are void.**

## Reproducing

```bash
# uncalibrated rules
python run_ppl_sweep.py --model_name llama-3.1-8b --sweep w4a4_wmix2 \
    --datasets wikitext,c4 --output results/w4a4_wmix_a6000_fixed/llama-3.1-8b.json
# calibrated type election (collects E[x_j^2] from the wikitext train split first)
python run_ppl_sweep.py --model_name llama-3.1-8b --sweep w4a4_wmix3 \
    --datasets wikitext,c4 --output results/w4a4_wmix_calib/llama-3.1-8b.json
```

Run on GPUs 0-3 only. Data type qualifiers added by this work: `p<p>` (HQQ Lp loss), `b<eps>`
(bounded harm election), `hesst` (calibrated type election, scale search left on MSE).
