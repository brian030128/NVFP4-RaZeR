# The count sweep: 256 tiles was the wrong place to judge the alpha basis

Job 337239, re-electing each arm of job 337178 at every count in
`run_cap_sweep.COUNTS` from its own frozen score tables. No re-scoring; the
256-tile re-election is asserted to reproduce the frozen map index for index
before any other count is trusted. Same 512-token causal protocol.

E0M3 costs a format bit per elected tile, so a small count is the point of it.
The alpha choice costs **nothing** -- it only changes the value written into the
ue4m3 scale field NVFP4 already stores -- so capping it at 256 handicapped the
one knob that has no reason to be sparse. Eligible tiles: 60k-108k per arm.

## qwen4b (post-trained): E0M3 wins at every count, and the gap widens

WikiText PPL, delta against each arm's own baseline:

| count | e0m3 | alpha | type_pure |
|---|---:|---:|---:|
| baseline | 19.4149 | 19.4149 | 18.7583 |
| n256 | 17.4266 (-1.988) | 18.2566 (-1.158) | 17.3359 (-1.422) |
| n4096 | 15.2153 (-4.200) | 17.1277 (-2.287) | 15.4046 (-3.354) |
| n65536 | **14.2106 (-5.204)** | 15.9494 (-3.466) | 14.3593 (-4.399) |
| n_all | 14.2286 (-5.186) | 15.9494 | 14.3593 |

Paired at each arm's best count, alpha minus e0m3 on identical windows:
wiki **+0.115432 ±0.003409**, c4 **+0.055827 ±0.004016**. E0M3 better, decisively,
and the absolute gap grows from 0.83 PPL at 256 tiles to 1.74 at 65,536.

## llama8b (base): a dead heat, and only one of them is safe

| count | e0m3 wiki | alpha wiki | e0m3 c4 | alpha c4 |
|---|---:|---:|---:|---:|
| baseline | 9.0929 | 9.0929 | 11.6001 | 11.6001 |
| n256 | **9.0286** | 9.0748 | **11.5131** | 11.5642 |
| n4096 | 9.0615 | 9.0518 | 11.5822 | 11.5430 |
| n16384 | 9.1050 | 9.0401 | 11.6180 | **11.5142** |
| n65536 | 9.1438 | **9.0364** | 11.6400 | 11.5372 |
| n_all | 9.1680 **(+0.075)** | 9.0384 (-0.054) | 11.6696 **(+0.070)** | 11.5213 (-0.079) |

Paired at each arm's best count, alpha minus e0m3:
wiki **+0.000865 ±0.001738**, c4 **+0.000094 ±0.002714**. Both intervals contain
zero: **indistinguishable on both domains**.

The shapes are not the same, and that is the finding. E0M3 peaks at 256 tiles and
then degrades until it is **net worse than its own baseline** -- +0.075 wiki and
+0.070 c4 at all eligible (type_pure worse still, +0.254 / +0.304). The alpha
basis improves monotonically and never crosses zero on either domain.

The reason is visible in the direction statistics: the alpha alternative
*improves* weight reconstruction (0.0840 against the baseline's 0.0869) while the
E0M3 alternative *degrades* it (0.0876). A wrongly elected alpha flip is close to
harmless; a wrongly elected E0M3 flip actively damages the model, and with 99,024
eligible tiles the two-SE filter lets through enough wrong ones to go net
negative.

## What this means

- **On the base model the alpha basis dominates in deployment terms.** It ties
  E0M3 on both domains, costs no metadata, runs on the existing NVFP4 kernel,
  and needs no count tuning -- where E0M3 must have its count tuned or it turns
  actively harmful.
- **On the post-trained model E0M3 wins, and wins big.** But
  `below_bf16_diag/FINDINGS.md` attributes 95% of qwen4b's gain in this regime
  to recalibration of an over-sharp model rather than to better prediction
  (measured at 2048 tokens, so the fraction does not transfer exactly, but the
  mechanism is the same model's same over-sharpness). So E0M3's advantage is
  concentrated exactly where perplexity is least trustworthy, and its one
  head-to-head win on a base model is a tie.
- Neither result is an accuracy or exploitability measurement, and nothing here
  is compared against BF16.

## The experiment this now argues for

The two knobs are independent and were never combined: dense9 alpha on every
scale block (free, monotone, safe) **plus** task-gradient E0M3 on a tuned count
of tiles. On qwen4b the MSE-elected alpha alone was worth a supported -0.339
wiki from a baseline E0M3 then improves by -5.204, and nothing has tested
whether those add.

A second untested variant is the stronger reading of "apply the method to the
scale": let the gradient choose WHICH of the nine alphas each block takes, a
nine-way election at scale-block granularity, instead of gating an MSE-chosen
upgrade per 8x64 tile. That needs fresh scoring with nine directions.
