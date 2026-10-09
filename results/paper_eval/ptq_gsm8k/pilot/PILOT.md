# Part M0: GSM8K pilot (not a result)

RTN FourOverSix, greedy, thinking off, 2048 new tokens at most; the first 64 problems at batch 16, the first 64 (Nemotron-Nano-9B-v2) / 32 (Qwen3.8-27B) at batch 1. Batch identity: each problem's completion text at batch 16 against batch 1's.

## Nemotron-Nano-9B-v2

Thinking-off prompt (end): `' \\boxed{}.\n<SPECIAL_11>Assistant\n<think></think>'`; registry switch {'system': '/think'}.

| batch | problems | accuracy | mean / median / max tokens | truncated | unparseable | think markers | s per step | s per problem | total s (load included) | GPU peak GiB |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 92.2 % | 277 / 287 / 468 | 0 | 0 | 0 | 0.0287 | 0.83 | 89 | 24.7 |
| 1 | 64 | 95.3 % | 314 / 286 / 2048 | 1 | 0 | 0 | 0.0282 | 8.93 | 606 | 24.7 |

Batch identity (batch 16 vs batch 1, 64 problems): 10 identical, 54 differ (7 with a different extracted answer, 6 with a different correctness).
- gsm8k/0: first difference at character 81 (lengths 336 / 330); answers 18 / 18; correct True / True
- gsm8k/1: first difference at character 39 (lengths 552 / 565); answers 3 / 3; correct True / True
- gsm8k/2: first difference at character 13 (lengths 1022 / 878); answers 70000 / 70000; correct True / True
- gsm8k/3: first difference at character 23 (lengths 679 / 723); answers 540 / 540; correct True / True
- gsm8k/4: first difference at character 238 (lengths 1190 / 8674); answers 20 / 40; correct True / False
- gsm8k/5: first difference at character 756 (lengths 1428 / 1444); answers 64 / 64; correct True / True
- gsm8k/7: first difference at character 135 (lengths 1268 / 1527); answers 160 / 160; correct True / True
- gsm8k/8: first difference at character 77 (lengths 1160 / 1380); answers 315 / 45; correct False / True
- gsm8k/9: first difference at character 79 (lengths 1020 / 1166); answers 460 / 460; correct True / True
- gsm8k/10: first difference at character 176 (lengths 1122 / 1155); answers 366 / 366; correct True / True
- gsm8k/12: first difference at character 159 (lengths 1159 / 1296); answers 12 / 13; correct False / True
- gsm8k/13: first difference at character 17 (lengths 1054 / 1144); answers 18 / 18; correct True / True
- gsm8k/14: first difference at character 231 (lengths 1452 / 1416); answers 60 / 60; correct True / True
- gsm8k/15: first difference at character 76 (lengths 804 / 558); answers 125 / 125; correct True / True
- gsm8k/16: first difference at character 85 (lengths 626 / 676); answers 230 / 230; correct True / True
- gsm8k/17: first difference at character 79 (lengths 1226 / 1235); answers 57500 / 57500; correct True / True
- gsm8k/18: first difference at character 272 (lengths 906 / 1079); answers 7 / 7; correct True / True
- gsm8k/19: first difference at character 199 (lengths 1666 / 1751); answers 6 / 6; correct True / True
- gsm8k/20: first difference at character 187 (lengths 1273 / 1384); answers 15 / 15; correct True / True
- gsm8k/23: first difference at character 310 (lengths 840 / 847); answers 8 / 8; correct True / True

Padding (gsm8k_padcheck.py): rows with no left padding in their batch-16 batch: 1 of 4 identical to batch 1; padded rows: 9 of 60. The dependence is not only padding.

Estimated hours (resampling the pilot's completion lengths; batches 32 / 64 not piloted, their step time assumed 2 % / 10 % above batch 16's; GPTQ +2 min per load and Hadamard +20 % per step assumed):

| batch | per configuration | 9 configurations |
|---:|---:|---:|
| 1 | 3.07 | 29.5 |
| 16 | 0.44 | 4.3 |
| 32 | 0.30 | 3.0 |
| 64 | 0.22 | 2.2 |

## Qwen3.8-27B

Thinking-off prompt (end): `'_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`; registry switch {'chat_template_kwargs': {'enable_thinking': True}}.

| batch | problems | accuracy | mean / median / max tokens | truncated | unparseable | think markers | s per step | s per problem | total s (load included) | GPU peak GiB |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 95.3 % | 441 / 370 / 2048 | 1 | 0 | 0 | 0.0879 | 8.05 | 465 | 67.5 |
| 1 | 32 | 100.0 % | 429 / 383 / 1608 | 0 | 0 | 0 | 0.0861 | 37.44 | 1223 | 67.5 |

Batch identity (batch 16 vs batch 1, 32 problems): 1 identical, 31 differ (2 with a different extracted answer, 2 with a different correctness).
- gsm8k/0: first difference at character 0 (lengths 839 / 949); answers 18 / 18; correct True / True
- gsm8k/2: first difference at character 118 (lengths 1335 / 1473); answers 70000 / 70000; correct True / True
- gsm8k/3: first difference at character 60 (lengths 678 / 932); answers 540 / 540; correct True / True
- gsm8k/4: first difference at character 0 (lengths 875 / 1151); answers 20 / 20; correct True / True
- gsm8k/5: first difference at character 3 (lengths 1456 / 1433); answers 64 / 64; correct True / True
- gsm8k/6: first difference at character 12 (lengths 838 / 608); answers 260 / 260; correct True / True
- gsm8k/7: first difference at character 0 (lengths 1567 / 1765); answers 160 / 160; correct True / True
- gsm8k/8: first difference at character 0 (lengths 1940 / 1920); answers 45 / 45; correct True / True
- gsm8k/9: first difference at character 131 (lengths 1229 / 1220); answers 460 / 460; correct True / True
- gsm8k/10: first difference at character 240 (lengths 1341 / 1428); answers 366 / 366; correct True / True
- gsm8k/11: first difference at character 845 (lengths 1053 / 1159); answers 694 / 694; correct True / True
- gsm8k/12: first difference at character 112 (lengths 5017 / 5039); answers 12 / 13; correct False / True
- gsm8k/13: first difference at character 74 (lengths 1716 / 1347); answers 18 / 18; correct True / True
- gsm8k/14: first difference at character 985 (lengths 1444 / 1327); answers 60\ / 60; correct False / True
- gsm8k/15: first difference at character 213 (lengths 1270 / 1308); answers 125 / 125; correct True / True
- gsm8k/16: first difference at character 48 (lengths 1046 / 1077); answers 230 / 230; correct True / True
- gsm8k/17: first difference at character 3 (lengths 1601 / 1426); answers 57500 / 57500; correct True / True
- gsm8k/18: first difference at character 136 (lengths 719 / 735); answers 7 / 7; correct True / True
- gsm8k/19: first difference at character 121 (lengths 1940 / 1927); answers 6 / 6; correct True / True
- gsm8k/20: first difference at character 38 (lengths 1929 / 2071); answers 15 / 15; correct True / True

Padding (gsm8k_padcheck.py): rows with no left padding in their batch-16 batch: 0 of 0 identical to batch 1; padded rows: 1 of 32. The dependence is not only padding.

Estimated hours (resampling the pilot's completion lengths; batches 32 / 64 not piloted, their step time assumed 2 % / 10 % above batch 16's; GPTQ +2 min per load and Hadamard +20 % per step assumed):

| batch | per configuration | 9 configurations |
|---:|---:|---:|
| 1 | 13.81 | 132.7 |
| 16 | 2.29 | 22.1 |
| 32 | 1.48 | 14.3 |
| 64 | 0.97 | 9.4 |

