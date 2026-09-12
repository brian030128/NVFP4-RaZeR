#### Does the budget-matched result hold on accuracy?

The perplexity table above compares the two objectives at an equal budget of switches, because at a fixed k they elect very different numbers of tiles. The same comparison on the multiple-choice panel, with every delta measured against the FourOverSix base by paired McNemar over documents.

**Llama-3.1-8B**

| objective | k | tiles | panel mean | d vs base (paired) | p |
|---|---:|---:|---:|---:|---:|
| — | — | 0 | 0.6720 | — | — |
| max(CE, KL) | 6 | 145 | 0.6737 | +0.0013 | 0.467 |
| max(CE, KL) | 5 | 267 | 0.6767 | +0.0032 | 0.079 |
| max(CE, KL) | 4 | 541 | 0.6712 | +0.0000 | 1 |
| max(CE, KL) **(shipped)** | 3 | 3,345 | 0.6714 | +0.0010 | 0.617 |
| KL only | 6 | 1,653 | 0.6701 | +0.0003 | 0.906 |
| KL only | 5 | 2,950 | 0.6706 | -0.0002 | 0.93 |
| KL only | 3 | 32,774 | 0.6601 | -0.0120 | 9.74e-08 |

Head to head against the best conjunction setting that elects no more tiles:

- `k6_kl` (1,653 tiles) against `k5` (267): -0.0029, p = 0.112 -- indistinguishable
- `k5_kl` (2,950 tiles) against `k5` (267): -0.0034, p = 0.0721 -- indistinguishable
- `k3_kl` (32,774 tiles) against `k5` (267): -0.0152, p = 8.57e-12 -- significantly different

**Qwen3-4B**

| objective | k | tiles | panel mean | d vs base (paired) | p |
|---|---:|---:|---:|---:|---:|
| — | — | 0 | 0.6231 | — | — |
| max(CE, KL) | 6 | 222 | 0.6252 | +0.0025 | 0.235 |
| max(CE, KL) | 5 | 576 | 0.6266 | +0.0060 | 0.00412 |
| max(CE, KL) | 4 | 1,837 | 0.6222 | +0.0061 | 0.00457 |
| max(CE, KL) **(shipped)** | 3 | 7,912 | 0.6385 | +0.0133 | 6.76e-10 |
| KL only | 5 | 1,226 | 0.6287 | +0.0067 | 0.00148 |
| KL only | 4 | 3,569 | 0.6345 | +0.0105 | 1.07e-06 |
| KL only | 3 | 21,528 | 0.6412 | +0.0161 | 9.03e-14 |

Head to head against the best conjunction setting that elects no more tiles:

- `k5_kl` (1,226 tiles) against `k5` (576): +0.0007, p = 0.754 -- indistinguishable
- `k4_kl` (3,569 tiles) against `k4` (1,837): +0.0044, p = 0.0335 -- significantly different
- `k3_kl` (21,528 tiles) against `k3` (7,912): +0.0028, p = 0.157 -- indistinguishable

Each frontier is pooled across jobs but confined to one GPU model (Llama-3.1-8B on H100 (2 point(s) on other hardware dropped); Qwen3-4B on H100 (2 point(s) on other hardware dropped)), because §1a's control found the evaluation exact within a GPU model and 2.8% of documents flipped across two. The 8 policies measured twice on that hardware agree on every document, so the pooling is exact rather than assumed.

