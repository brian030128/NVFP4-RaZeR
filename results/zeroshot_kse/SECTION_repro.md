#### Is the paired test reading hardware noise?

Every conclusion above rests on an exact McNemar test over per-document outcomes, so it is worth running that test where the answer is known. Below is the identical comparison applied to two jobs of the **same** policy, on the same weights, data and library versions -- the true difference is zero by construction. `b` and `c` count the documents only one of the two runs gets right; a significant p here would mean §1a is reading the hardware as if it were the method.

| model | policy | jobs | GPUs | documents | b | c | delta | McNemar p |
|---|---|---|---|---:|---:|---:|---:|---:|
| llama8b | four_over_six | 337838 vs 339051 | H100 | 18627 | 0 | 0 | +0.0000 | 1 |
| llama8b | bf16 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| llama8b | nvfp4 | 338236 vs 338343 | H100 | 1319 | 0 | 0 | +0.0000 | 1 |
| qwen4b | four_over_six | 337839 vs 339082 | H200 vs H100 | 18627 | 267 | 247 | -0.0011 | 0.402 |

Re-running on the same GPU model reproduces the evaluation exactly: 3 such pairs, and not one of 21,265 scored documents changes outcome. The evaluation itself is deterministic. Across GPU models it is not. Up to **2.8% of documents flip** -- where two continuations score within rounding of each other, and winogrande's differ only by a pronoun, a different GEMM kernel is enough to reverse the comparison. Aggregate accuracy still moves by well under a point, because the flips go both ways.

The point of the table is that none of the 4 shows a significant asymmetry. That symmetry is the case McNemar conditions on: the test is computed from the *difference* between `b` and `c`, not from their size, so noise that inflates both equally cancels. The comparisons in §1a are also made within a single job, where the evaluation is exactly reproducible, so they do not carry even this term. What it does mean is that a single document's outcome is not a portable property of a policy, and that accuracies here should be read to a few tenths of a percent rather than to the digits lm-eval prints.

