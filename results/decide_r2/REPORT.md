
### llama-3.1-8b — W4A16   (baseline: mix_4_6_e2m1_8x64, sorted by wikitext)

| config                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------|----|-----------|-----------|-----------|-----------|
| fp16                   |    |    6.2398 |   -0.3587 |    8.9580 |   -0.4641 |
| nvfp4_razer_e3m3       |    |    6.5003 |   -0.0982 |    9.2929 |   -0.1292 |
| mix_4_6_1x16           | -  |    6.5534 |   -0.0451 |    9.3660 |   -0.0561 |
| nvif4                  |    |    6.5598 |   -0.0387 |    9.3711 |   -0.0509 |
| mix_4_6_h1.5_8x64      | y  |    6.5874 |   -0.0111 |    9.4191 |   -0.0030 |
| mix_4_6_m1_8x64        | y  |    6.5877 |   -0.0108 |    9.4184 |   -0.0037 |
| mix_4_6_h2_8x64        | y  |    6.5922 |   -0.0063 |    9.4154 |   -0.0066 |
| mix_4_6_tol1_8x64      | y  |    6.5944 |   -0.0040 |    9.4202 |   -0.0018 |
| mix_4_6_m2_8x64        | y  |    6.5946 |   -0.0039 |    9.4188 |   -0.0033 |
| mix_4_6_rm1_8x64       | y  |    6.5959 |   -0.0026 |    9.4209 |   -0.0012 |
| mix_4_6_v0.5_8x64      | y  |    6.5967 |   -0.0017 |    9.4430 |   +0.0209 |
| mix_4_6_h3_8x64        | y  |    6.5971 |   -0.0014 |    9.4196 |   -0.0025 |
| mix_4_6_perm_m2_8x64   | y  |    6.5972 |   -0.0013 |    9.4244 |   +0.0023 |
| mix_4_6_h5_8x64        | y  |    6.5972 |   -0.0013 |    9.4207 |   -0.0014 |
| mix_4_6_h20_8x64       | y  |    6.5979 |   -0.0006 |    9.4228 |   +0.0007 |
| mix_4_6_rm3_8x64       | y  |    6.5980 |   -0.0005 |    9.4229 |   +0.0008 |
| mix_4_6_h10_8x64       | y  |    6.5981 |   -0.0004 |    9.4223 |   +0.0003 |
| nvfp4_4over6           |    |    6.5984 |   -0.0001 |    9.4239 |   +0.0019 |
| mix_4_6_v0.9_8x64      | y  |    6.5984 |   -0.0001 |    9.4224 |   +0.0003 |
| mix_4_6_m3_8x64        | y  |    6.5985 |   -0.0000 |    9.4217 |   -0.0003 |
| mix_4_6_tol0.5_8x64    | y  |    6.5985 |   -0.0000 |    9.4227 |   +0.0007 |
| mix_4_6_dom_8x64       | y  |    6.5985 |   -0.0000 |    9.4219 |   -0.0001 |
| mix_4_6_tol0.1_8x64    | y  |    6.5985 |   -0.0000 |    9.4219 |   -0.0001 |
| mix_4_6_e2m1_8x64      | y  |    6.5985 |   +0.0000 |    9.4221 |   +0.0000 |
| mix_4_6_v0.6_8x64      | y  |    6.5985 |   +0.0000 |    9.4203 |   -0.0018 |
| mix_4_6_tol0.25_8x64   | y  |    6.5986 |   +0.0001 |    9.4229 |   +0.0009 |
| mix_4_6_perm_rm2_8x64  | y  |    6.5992 |   +0.0007 |    9.4222 |   +0.0002 |
| mix_4_6_perm_h3_8x64   | y  |    6.5993 |   +0.0008 |    9.4242 |   +0.0021 |
| mix_4_6_rm2_8x64       | y  |    6.5994 |   +0.0009 |    9.4213 |   -0.0008 |
| mix_4_6_v0.75_8x64     | y  |    6.5997 |   +0.0012 |    9.4208 |   -0.0013 |
| mix_4_6_8x64           | y  |    6.6006 |   +0.0021 |    9.4405 |   +0.0185 |
| mix_4_6_perm_v0.5_8x64 | y  |    6.6042 |   +0.0057 |    9.4357 |   +0.0136 |
| mix_4_6_perm_8x64      | y  |    6.6055 |   +0.0070 |    9.4360 |   +0.0139 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 2 settles (Llama-3.1-8B, W4A16, 33 configs, all at 8x64)

This model is the right place to ask the question. `nvif4` -- which picks the element type per 16
element scale block, the finest choice there is -- beats 4over6 by -0.039 wikitext here, against
+0.014 on Llama-2-7B. So there is 0.045 of headroom between the `1x16` upper bound and a type block
that has to elect one winner for 32 scale blocks, and the election rule is what decides how much of
it survives.

Everything is measured against `mix_4_6_e2m1_8x64`, this code path with E0M3 switched off.

**Plain argmin is worse than not having E0M3 at all** (+0.0021 / +0.0185). Minimising the tile's
summed error is the MSE-optimal choice and it loses to never electing. That is the whole problem
restated: the criterion is not the objective.

**The useful rules are the MILD ones.**

| rule | dwikitext | dc4 | share of the 1x16 gain kept |
|---|---|---|---|
| `h1.5` (robust, kappa^2=1.5) | -0.0111 | -0.0030 | 25% |
| `m1` (margin z=1) | -0.0108 | -0.0037 | 24% |
| `h2` | -0.0063 | -0.0066 | 14% |
| `m2` | -0.0039 | -0.0033 | 9% |
| `h3` | -0.0014 | -0.0025 | 3% |
| `dom`, `tol0.1`, `tol0.25`, `tol0.5` | ~0.0000 | ~0.000 | 0% (they elect nothing) |

The ordering is the opposite of round 1, where Llama-2-7B preferred `h3`. That is consistent rather
than contradictory: Llama-2-7B has no E0M3 signal to recover (its `nvif4` is +0.014), so the best
thing a rule can do there is elect nothing, and the most conservative rule wins by default. On a
model where E0M3 genuinely helps, the rule has to be permissive enough to actually fire. **kappa^2
between 1.5 and 2 is the range that works on both.**

`tol`, `vote` and `relmargin` are all dominated. `v0.5` is notable for being decent on wikitext
(-0.0017) and bad on c4 (+0.0209) -- ignoring magnitudes entirely throws away real information.

**Row permutation does not work** (`perm` rows, -0.0013 to +0.0070, each one worse than the same
election rule without it). The idea was to sort rows by their E0M3 preference so that tiles stop
straddling the boundary. The diagnostic in `analyze_rotation.py` says why it cannot work at this
shape: an 8x64 tile holds 8 rows x 4 scale blocks = 32 scale blocks, and the disagreement is mostly
*within* a row across its k-blocks, not between rows. Straddling stays at ~100% of tiles no matter
how the rows are ordered, so the sort buys nothing -- and it costs something, presumably by breaking
up whatever similarity adjacent output channels had. Sorting rows is the wrong axis; the axis that
matters is K, and K cannot be permuted per tile without changing the GEMM.
