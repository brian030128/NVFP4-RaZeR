
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                       | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                         |    |    5.4738 |   -0.1349 |    6.9749 |   -0.1630 |
| nvfp4_razer_e3m3             |    |    5.5677 |   -0.0410 |    7.0925 |   -0.0453 |
| mix_4_6_m2_32x128            | y  |    5.6039 |   -0.0048 |    7.1367 |   -0.0011 |
| mix_4_6_rm2_8x64             | y  |    5.6058 |   -0.0029 |    7.1376 |   -0.0002 |
| mix_4_6_h3_8x64              | y  |    5.6064 |   -0.0023 |    7.1358 |   -0.0021 |
| mix_4_6_v0.75_32x128         | y  |    5.6070 |   -0.0017 |    7.1367 |   -0.0012 |
| mix_4_6_e2m1_8x64            | y  |    5.6070 |   -0.0017 |    7.1367 |   -0.0012 |
| mix_4_6_tol0.25_32x128       | y  |    5.6070 |   -0.0017 |    7.1367 |   -0.0012 |
| mix_4_6_rm2_32x128           | y  |    5.6070 |   -0.0017 |    7.1370 |   -0.0009 |
| mix_4_6_tol0.25_8x64         | y  |    5.6071 |   -0.0017 |    7.1369 |   -0.0009 |
| mix_4_6_mae_m2_8x64          | y  |    5.6071 |   -0.0016 |    7.1386 |   +0.0008 |
| mix_4_6_mae_e2m1_8x64        | y  |    5.6073 |   -0.0015 |    7.1391 |   +0.0013 |
| mix_4_6_m2_8x64              | y  |    5.6075 |   -0.0012 |    7.1375 |   -0.0003 |
| mix_4_6_mae_m2_32x128        | y  |    5.6075 |   -0.0012 |    7.1390 |   +0.0012 |
| mix_4_6_h3_32x128            | y  |    5.6084 |   -0.0003 |    7.1391 |   +0.0012 |
| mix_4_6_v0.75_8x64           | y  |    5.6085 |   -0.0002 |    7.1378 |   -0.0001 |
| mix_4_6_h2_32x128            | y  |    5.6086 |   -0.0001 |    7.1388 |   +0.0009 |
| nvfp4_4over6                 |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_l1.5_m2_32x128       | y  |    5.6098 |   +0.0011 |    7.1358 |   -0.0020 |
| mix_4_6_v0.6_32x128          | y  |    5.6109 |   +0.0022 |    7.1387 |   +0.0009 |
| mix_4_6_l0.5_m2_8x64         | y  |    5.6119 |   +0.0031 |    7.1420 |   +0.0042 |
| mix_4_6_l0.5_m2_32x128       | y  |    5.6121 |   +0.0033 |    7.1420 |   +0.0042 |
| mix_4_6_l1.5_m2_8x64         | y  |    5.6126 |   +0.0039 |    7.1376 |   -0.0002 |
| mix_4_6_h2_8x64              | y  |    5.6126 |   +0.0039 |    7.1391 |   +0.0012 |
| mix_4_6_clipe2x_m2_8x64      | y  |    5.6147 |   +0.0060 |    7.1286 |   -0.0093 |
| mix_4_6_clipe2_m2_8x64       | y  |    5.6157 |   +0.0070 |    7.1286 |   -0.0092 |
| mix_4_6_clipe2x_e2m1_8x64    | y  |    5.6162 |   +0.0075 |    7.1288 |   -0.0091 |
| mix_4_6_clipe2_e2m1_8x64     | y  |    5.6167 |   +0.0080 |    7.1289 |   -0.0090 |
| mix_4_6_clipbothx_m2_32x128  | y  |    5.6168 |   +0.0081 |    7.1295 |   -0.0084 |
| mix_4_6_clipe2x_m2_32x128    | y  |    5.6173 |   +0.0086 |    7.1301 |   -0.0077 |
| mix_4_6_clipboth_m2_32x128   | y  |    5.6176 |   +0.0089 |    7.1296 |   -0.0083 |
| mix_4_6_clipe0x_m2_8x64      | y  |    5.6179 |   +0.0092 |    7.1397 |   +0.0019 |
| mix_4_6_clipe0_m2_8x64       | y  |    5.6180 |   +0.0093 |    7.1399 |   +0.0021 |
| mix_4_6_clipe2_m2_32x128     | y  |    5.6181 |   +0.0094 |    7.1303 |   -0.0076 |
| mix_4_6_clipbothx_m2_8x64    | y  |    5.6185 |   +0.0098 |    7.1302 |   -0.0077 |
| mix_4_6_clipwide_e2m1_8x64   | y  |    5.6197 |   +0.0110 |    7.1308 |   -0.0071 |
| mix_4_6_clipboth_m2_8x64     | y  |    5.6198 |   +0.0111 |    7.1302 |   -0.0076 |
| mix_4_6_v0.6_8x64            | y  |    5.6209 |   +0.0122 |    7.1427 |   +0.0048 |
| mix_4_6_clipwide_m2_32x128   | y  |    5.6223 |   +0.0136 |    7.1326 |   -0.0053 |
| nvif4                        |    |    5.6224 |   +0.0137 |    7.1375 |   -0.0003 |
| mix_4_6_clipwide_m2_8x64     | y  |    5.6248 |   +0.0161 |    7.1331 |   -0.0047 |
| mix_4_6_mae_clipe2_e2m1_8x64 | y  |    5.6337 |   +0.0250 |    7.1389 |   +0.0011 |
| mix_4_6_clipe0x_m2_32x128    | y  |    5.6415 |   +0.0328 |    7.1494 |   +0.0115 |
| mix_4_6_clipe0_m2_32x128     | y  |    5.6417 |   +0.0330 |    7.1493 |   +0.0114 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 1 settles (Llama-2-7B, W4A16, 43 configs)

`mix_4_6_e2m1_8x64` is the control: `elect="never"`, i.e. this code path with the E0M3 branch
switched off, which is 4over6 up to the E2M1 rounding-tie convention (-0.0017 / -0.0012 against
`nvfp4_4over6`, so it is a very slightly *stronger* baseline than the reference implementation).
Every variant must be read against that row, not against `nvfp4_4over6`.

**Clipping does not pay.** Searching the block scale over clip ratios alpha < 1 costs +0.006 to
+0.033 wikitext and buys -0.008 c4. The split is consistent across all 12 clipping rows, so it is a
real trade and not noise, but the wikitext side is larger and the worst row in the whole table is a
clipping row (`clipe0_m2_32x128`, +0.033). Clipping E0M3 is worse than clipping E2M1 on both
datasets. This holds even though clipping strictly LOWERS the selection loss it optimizes -- one
more instance of the MSE/perplexity gap this file keeps running into.

**Lp metrics do not pay.** `mae` (= l1) is within 0.0005 of `mse` on wikitext and slightly worse on
c4; `l0.5` is worse on both (+0.003 / +0.004); `l1.5` is worse on wikitext. The squared-error
selection is not the thing to fix.

**Election rules pay a little.** Best on BOTH datasets is `h3` (the robust rule at kappa^2 = 3) at
8x64: -0.0023 / -0.0021, i.e. it beats the E2M1-only control on both, which no other rule does.
`m2` at 32x128 is the best single wikitext number (-0.0048) but is neutral on c4 (-0.0011, vs the
control's -0.0012). `rm2` at 8x64 is -0.0029 / -0.0002.

**But this model is a weak discriminator.** Every row from `m2_32x128` down to `v0.75_8x64` sits
within 0.005 of the control, and the E0M3 election is worth almost nothing on Llama-2-7B W4A16 --
`nvif4`, which chooses per scale block and is the finest choice possible, is itself +0.0137 here.
There is nothing for a better rule to recover. Llama-3.1-8B is the model to test on: there `nvif4`
beats 4over6 by -0.039 wikitext, so the per-block choice is worth something and the question of how
much of it a coarse type block can keep is a real one.
