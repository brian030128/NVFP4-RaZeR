
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                     | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_perm_h1.5_8x64     | y  |    6.5898 |   -0.0086 |    9.4223 |   -0.0016 |
| mix_4_6_rot_1x16           | -  |    6.5956 |   -0.0028 |    9.4255 |   +0.0016 |
| nvfp4_4over6               |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_perm_rot_h1.5_8x64 | y  |    6.6824 |   +0.0840 |    9.5585 |   +0.1346 |
| mix_4_6_rot_m1_8x64        | y  |    6.6924 |   +0.0940 |    9.5714 |   +0.1475 |
| mix_4_6_rot_8x64           | y  |    6.6926 |   +0.0942 |    9.5490 |   +0.1250 |
| mix_4_6_rotcol_h1.5_8x64   | y  |    6.6929 |   +0.0946 |    9.5670 |   +0.1431 |
| mix_4_6_rot_h1.5_8x64      | y  |    6.7015 |   +0.1031 |    9.5741 |   +0.1502 |
| mix_4_6_rot_h2_8x64        | y  |    6.7033 |   +0.1049 |    9.5747 |   +0.1508 |
| mix_4_6_rot_e2m1_8x64      | y  |    6.7051 |   +0.1067 |    9.5925 |   +0.1685 |
| mix_4_6_rot_h3_8x64        | y  |    6.7132 |   +0.1148 |    9.5804 |   +0.1565 |
| mix_4_6_rot64_h1.5_8x64    | y  |    6.7187 |   +0.1203 |    9.5797 |   +0.1558 |
| mix_4_6_rot64_e2m1_8x64    | y  |    6.7445 |   +0.1461 |    9.6008 |   +0.1769 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 3 settles (Llama-3.1-8B, W4A16): rotating everything is a large loss

A normalized Hadamard applied to each 16-column chunk of the reduction dimension cuts weight NMSE
by 25% on q_proj and 7% on average (`analyze_coherent_error.py`, `analyze_rotation.py`), makes the
blocks measurably Gaussian (block max/rms 2.38 -> 1.98, kurtosis 3.29 -> 2.30), and flips the
element-type decision from 4.4% E0M3 tiles to 86.3% on q_proj. Every proxy says it should work.

**It costs +0.09 to +0.15 wikitext perplexity.** This is the largest MSE-to-perplexity divergence
anywhere in this codebase, and it is not explained by the usual suspect: the diagonal-Hessian
(importance-weighted) weight error *also* drops, by 40%.

What does explain it is the TRUE layer output error `||X dW^T||^2`, measured on real activations
(2 wikitext batches, `model.layers.0`):

| layer | weight MSE | importance-weighted | true output error |
|---|---|---|---|
| q_proj | -24.9% | -61.7% | **-62.2%** |
| k_proj | -18.0% | -55.0% | **-54.7%** |
| o_proj | -5.7% | -5.6% | **-13.2%** |
| v_proj | -2.0% | +71.6% | **+72.7%** |
| gate_proj | -0.1% | +4.7% | **+5.5%** |
| up_proj | -0.4% | +7.4% | **+8.0%** |
| down_proj | -0.9% | +3.2% | **+2.5%** |

Rotation is not good or bad, it is *per layer*: a large win on q/k/o_proj and a real loss on v_proj
and the whole MLP. Since the MLP is most of the parameters, the sum is negative. Plain weight MSE
improves on every single layer, so "rotate when it lowers the error" rotates all of them and eats
the damage -- which is exactly what the `rot`, `rotcol` and `rot64` rows measure.

Note `rotcol` (rotate a chunk iff rotation lowers its error, threshold 0) lands at +0.0946, barely
distinguishable from rotating everything. Selectivity with a zero threshold is not selectivity.

**The separating signal is the SIZE of the MSE gain, and it needs no calibration.** The layers
rotation helps are those where it cuts weight MSE by more than ~15%; the layers it hurts are those
where the MSE barely moves, and there rotation only scrambles the direction of the error for
nothing. Requiring a minimum fractional gain (`rotmin<t>`) reproduces the ground-truth split from
the weights alone -- share of chunks rotated at t=0.10:

| q_proj | k_proj | o_proj | v_proj | gate | up | down |
|---|---|---|---|---|---|---|
| 46.9% | 41.4% | 25.0% | **0%** | **0%** | **0%** | **0%** |

which is precisely the set the measured output error wanted. Round 4 tests whether that translates.

Also here: `rot` at a 1x16 type block is -0.0028, against -0.045 for the same thing unrotated, so
rotation costs ~0.042 even at the finest granularity -- the loss is intrinsic to the rotation, not
an artifact of the coarse type block. And `perm_h1.5` (-0.0086) is again slightly worse than plain
`h1.5` (-0.0111), confirming round 2's negative result on row permutation.
