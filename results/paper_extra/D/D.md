# Experiment D: decode latency (CUDA graph, StaticCache, 64 generated tokens)

## Llama-3.1-8B

Tokens per second (median over rounds); change vs FourOverSix in % (paired within rounds; for FlipQuant (ours) 8x64 also vs FourOverSix on wB).

| policy | 1x512 | 4x512 | 16x512 | 1x2048 | 4x2048 | 16x2048 | decode widths |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | 80.5 (-41.8 %) | 283.9 (-39.1 %) | 776.5 (-29.2 %) | 66.2 (-32.5 %) | 206.6 (-28.7 %) | 386.5 (-15.2 %) | — |
| NVFP4 | 136.6 (-1.2 %) | 462.1 (-0.9 %) | 1090.8 (-0.5 %) | 97.3 (-0.8 %) | 287.8 (-0.6 %) | 455.3 (-0.1 %) | b1: [16]; b4: [16]; b16: [16] |
| FourOverSix | 138.3 | 466.5 | 1096.7 | 98.1 | 289.5 | 456.1 | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 16x64 | 137.6 (-0.6 %) | 464.7 (-0.4 %) | 1093.8 (-0.3 %) | 97.8 (-0.3 %) | 288.7 (-0.3 %) | 455.2 (-0.2 %) | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 8x64 | 101.4 (-26.7 %) [wB -2.9 %] | 369.2 (-20.8 %) [wB -2.0 %] | 952.7 (-13.1 %) [wB -2.0 %] | 80.0 (-18.5 %) [wB -1.8 %] | 249.4 (-13.8 %) [wB -2.0 %] | 427.9 (-6.1 %) [wB -0.9 %] | b1: n8k64_wB; b4: n8k64_wB; b16: n8k64_wB |
| NVFP4 (wB) | 103.5 (-25.2 %) | 373.7 (-19.9 %) | 967.3 (-11.8 %) | 80.9 (-17.6 %) | 253.1 (-12.6 %) | 430.8 (-5.5 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |
| FourOverSix (wB) | 104.5 (-24.5 %) | 376.8 (-19.2 %) | 972.5 (-11.3 %) | 81.5 (-16.9 %) | 254.5 (-12.1 %) | 431.7 (-5.4 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |

Graph = eager StaticCache greedy tokens (33 per setting): passed in all 210 (policy, round, setting) records (registered check). Lowest agreement with an eager DynamicCache decode, per policy (recorded, not a check): {'bf16': 0.3939393939393939, 'nvfp4': 0.7424242424242424, 'fo6': 0.18181818181818182, 'ours-16x64': 0.24242424242424243, 'ours-8x64': 0.3712121212121212, 'nvfp4-wB': 0.7424242424242424, 'fo6-wB': 0.18181818181818182}.

## Mistral-7B-v0.3

Tokens per second (median over rounds); change vs FourOverSix in % (paired within rounds; for FlipQuant (ours) 8x64 also vs FourOverSix on wB).

| policy | 1x512 | 4x512 | 16x512 | 1x2048 | 4x2048 | 16x2048 | decode widths |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | 83.8 (-43.4 %) | 293.9 (-40.6 %) | 795.6 (-29.9 %) | 68.4 (-33.8 %) | 212.4 (-29.3 %) | 391.8 (-15.4 %) | — |
| NVFP4 | 146.2 (-1.3 %) | 489.7 (-1.0 %) | 1129.1 (-0.6 %) | 102.4 (-0.9 %) | 298.4 (-0.6 %) | 462.6 (-0.1 %) | b1: [16]; b4: [16]; b16: [16] |
| FourOverSix | 148.1 | 494.8 | 1135.6 | 103.4 | 300.3 | 463.1 | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 16x64 | 147.2 (-0.6 %) | 492.6 (-0.5 %) | 1132.3 (-0.3 %) | 103.0 (-0.4 %) | 299.4 (-0.3 %) | 462.7 (-0.1 %) | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 8x64 | 106.7 (-28.0 %) [wB -3.1 %] | 386.5 (-21.9 %) [wB -2.1 %] | 981.4 (-13.6 %) [wB -2.1 %] | 83.3 (-19.5 %) [wB -1.9 %] | 257.0 (-14.4 %) [wB -2.1 %] | 434.2 (-6.2 %) [wB -0.8 %] | b1: n8k64_wB; b4: n8k64_wB; b16: n8k64_wB |
| NVFP4 (wB) | 109.0 (-26.4 %) | 391.5 (-20.9 %) | 997.2 (-12.2 %) | 84.2 (-18.5 %) | 261.0 (-13.1 %) | 437.0 (-5.7 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |
| FourOverSix (wB) | 110.1 (-25.7 %) | 394.7 (-20.2 %) | 1002.5 (-11.7 %) | 84.9 (-17.9 %) | 262.5 (-12.6 %) | 438.0 (-5.4 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |

Graph = eager StaticCache greedy tokens (33 per setting): passed in all 210 (policy, round, setting) records (registered check). Lowest agreement with an eager DynamicCache decode, per policy (recorded, not a check): {'bf16': 0.3939393939393939, 'nvfp4': 0.030303030303030304, 'fo6': 0.6893939393939394, 'ours-16x64': 0.06060606060606061, 'ours-8x64': 0.696969696969697, 'nvfp4-wB': 0.030303030303030304, 'fo6-wB': 0.6893939393939394}.

## Phi-4

Tokens per second (median over rounds); change vs FourOverSix in % (paired within rounds; for FlipQuant (ours) 8x64 also vs FourOverSix on wB).

| policy | 1x512 | 4x512 | 16x512 | 1x2048 | 4x2048 | 16x2048 | decode widths |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | 46.4 (-51.4 %) | 162.5 (-50.3 %) | 457.2 (-35.9 %) | 40.2 (-43.3 %) | 125.0 (-37.8 %) | 232.5 (-19.4 %) | — |
| NVFP4 | 94.7 (-1.0 %) | 324.2 (-0.9 %) | 711.2 (-0.3 %) | 70.3 (-0.7 %) | 199.8 (-0.6 %) | 288.5 (-0.1 %) | b1: [16]; b4: [16]; b16: [16] |
| FourOverSix | 95.6 | 327.1 | 713.8 | 70.8 | 200.9 | 288.8 | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 16x64 | 95.3 (-0.4 %) | 326.4 (-0.2 %) | 712.4 (-0.2 %) | 70.7 (-0.2 %) | 200.6 (-0.2 %) | 288.4 (-0.2 %) | b1: [16]; b4: [16]; b16: [16] |
| FlipQuant (ours) 8x64 | 76.2 (-20.3 %) [wB -1.9 %] | 265.1 (-18.9 %) [wB -1.1 %] | 650.1 (-9.0 %) [wB -1.3 %] | 59.0 (-16.7 %) [wB -1.1 %] | 178.4 (-11.2 %) [wB -1.2 %] | 275.3 (-4.7 %) [wB -0.6 %] | b1: n8k64_wB; b4: n8k64_wB; b16: n8k64_wB |
| NVFP4 (wB) | 77.0 (-19.4 %) | 265.8 (-18.7 %) | 655.9 (-8.1 %) | 59.3 (-16.3 %) | 179.6 (-10.6 %) | 276.4 (-4.3 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |
| FourOverSix (wB) | 77.7 (-18.8 %) | 268.1 (-18.0 %) | 658.2 (-7.7 %) | 59.7 (-15.8 %) | 180.6 (-10.1 %) | 276.9 (-4.1 %) | b1: stock_wB; b4: stock_wB; b16: stock_wB |

Graph = eager StaticCache greedy tokens (33 per setting): passed in all 210 (policy, round, setting) records (registered check). Lowest agreement with an eager DynamicCache decode, per policy (recorded, not a check): {'bf16': 0.7575757575757576, 'nvfp4': 0.36363636363636365, 'fo6': 0.09090909090909091, 'ours-16x64': 0.44696969696969696, 'ours-8x64': 0.6060606060606061, 'nvfp4-wB': 0.36363636363636365, 'fo6-wB': 0.09090909090909091}.

