# Tile-count sweep on held-out C4 (job 335887)

Re-election of the unchanged CE/KL two-SE rule from each model's saved 192-sequence score
table at nine nested counts. No model is re-scored. Re-election at 256 reproduces the frozen
`pooled192` map bitwise for every model, and each 256 row reproduces the published
`results/c4_frozen` perplexity to six decimals. Evaluation is the unchanged held-out C4 recipe.

[Protocol](PROTOCOL.md).

## Result

**256 is not the best count for three of the four models, and the curve shape is strongly**
**model-dependent.** Two models leave most of the available gain unclaimed at 256; two others
are harmed at large counts. No single constant is simultaneously good for all four.

| Model | eligible / total tiles | best count | best ΔPPL | ΔPPL at 256 | supported harm from |
|---|---:|---:|---:|---:|---|
| OLMo-1B | 30,485 / 2,097,152 | 4,096 | -0.140184 | -0.101424 | none measured |
| Pythia-1.4B | 76,665 / 2,359,296 | 1,024 | -0.621039 | -0.308493 | 16,384 tiles |
| Qwen3-4B | 80,978 / 7,096,320 | 80,978 | -2.863637 | -1.025501 | none measured |
| Llama-3.1-8B | 97,908 / 13,631,488 | 1,024 | -0.090881 | -0.090246 | 97,908 tiles |

## Per-model curves

ΔPPL and ΔNLL are against the matched FourOverSix baseline; negative is better. `sig` marks
a descriptive paired two-SE interval excluding zero (YES a gain, HARM a regression).

### OLMo-1B

Baseline C4 PPL 14.984432; 30,485 of 2,097,152 tiles have a negative two-SE score.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE | sig |
|---|---:|---:|---:|---:|---|
| baseline | 0 | 14.984432 | — | — | — |
| 16 | 16 | 14.946808 | -0.037624 | -0.002514 ±0.002172 | YES |
| 64 | 64 | 14.918822 | -0.065610 | -0.004388 ±0.002285 | YES |
| 256 | 256 | 14.883008 | -0.101424 | -0.006792 ±0.002574 | YES |
| 1,024 | 1,024 | 14.868634 | -0.115798 | -0.007758 ±0.003034 | YES |
| 4,096 | 4,096 | 14.844247 | -0.140184 | -0.009399 ±0.003269 | YES |
| 16,384 | 16,384 | 14.863563 | -0.120868 | -0.008099 ±0.002979 | YES |
| 65,536 | 30,485 | 14.854211 | -0.130221 | -0.008728 ±0.003658 | YES |
| all eligible | 30,485 | 14.854211 | -0.130221 | -0.008728 ±0.003658 | YES |

### Pythia-1.4B

Baseline C4 PPL 20.471159; 76,665 of 2,359,296 tiles have a negative two-SE score.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE | sig |
|---|---:|---:|---:|---:|---|
| baseline | 0 | 20.471159 | — | — | — |
| 16 | 16 | 20.418796 | -0.052363 | -0.002561 ±0.002857 | · |
| 64 | 64 | 20.328652 | -0.142507 | -0.006986 ±0.003037 | YES |
| 256 | 256 | 20.162666 | -0.308493 | -0.015184 ±0.003167 | YES |
| 1,024 | 1,024 | 19.850120 | -0.621039 | -0.030807 ±0.003188 | YES |
| 4,096 | 4,096 | 19.890992 | -0.580167 | -0.028750 ±0.003516 | YES |
| 16,384 | 16,384 | 20.652269 | +0.181110 | +0.008808 ±0.004875 | HARM |
| 65,536 | 65,536 | 22.421023 | +1.949864 | +0.090982 ±0.006580 | HARM |
| all eligible | 76,665 | 22.677991 | +2.206832 | +0.102378 ±0.006664 | HARM |

### Qwen3-4B

Baseline C4 PPL 21.928243; 80,978 of 7,096,320 tiles have a negative two-SE score.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE | sig |
|---|---:|---:|---:|---:|---|
| baseline | 0 | 21.928243 | — | — | — |
| 16 | 16 | 21.723072 | -0.205171 | -0.009401 ±0.003048 | YES |
| 64 | 64 | 21.400644 | -0.527599 | -0.024354 ±0.002932 | YES |
| 256 | 256 | 20.902742 | -1.025501 | -0.047895 ±0.003392 | YES |
| 1,024 | 1,024 | 20.223081 | -1.705162 | -0.080951 ±0.004184 | YES |
| 4,096 | 4,096 | 19.574356 | -2.353887 | -0.113555 ±0.005325 | YES |
| 16,384 | 16,384 | 19.258204 | -2.670039 | -0.129838 ±0.006384 | YES |
| 65,536 | 65,536 | 19.065141 | -2.863102 | -0.139914 ±0.007585 | YES |
| all eligible | 80,978 | 19.064606 | -2.863637 | -0.139942 ±0.007806 | YES |

### Llama-3.1-8B

Baseline C4 PPL 11.600109; 97,908 of 13,631,488 tiles have a negative two-SE score.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE | sig |
|---|---:|---:|---:|---:|---|
| baseline | 0 | 11.600109 | — | — | — |
| 16 | 16 | 11.571491 | -0.028618 | -0.002470 ±0.002631 | · |
| 64 | 64 | 11.566999 | -0.033109 | -0.002858 ±0.002578 | YES |
| 256 | 256 | 11.509863 | -0.090246 | -0.007810 ±0.002113 | YES |
| 1,024 | 1,024 | 11.509227 | -0.090881 | -0.007865 ±0.002663 | YES |
| 4,096 | 4,096 | 11.528225 | -0.071884 | -0.006216 ±0.002806 | YES |
| 16,384 | 16,384 | 11.567794 | -0.032314 | -0.002790 ±0.003410 | · |
| 65,536 | 65,536 | 11.599627 | -0.000482 | -0.000042 ±0.003299 | · |
| all eligible | 97,908 | 11.647632 | +0.047524 | +0.004088 ±0.003532 | HARM |

## Reading

Qwen3-4B improves monotonically to the end of its eligible set, reaching -2.863637 PPL
against -1.025501 at 256: the fixed cap claims about a third of the available gain.
Pythia-1.4B doubles its gain at 1,024 and then reverses into supported harm beyond 16,384,
ending +2.206832 worse than baseline. OLMo-1B peaks near 4,096. Llama-3.1-8B is flat
between 256 and 1,024 and decays to supported harm only when every eligible tile is taken.

The two previously measured adaptive rules therefore failed at the wrong end. The
curvature-penalised selector chose 0-8 blocks and the description-cost threshold
under-selected; both shrank a count that, for three of these four models, should have
grown. The existing backtracking rule is one-sided in the same way -- it halves its budget
on failure and never raises it -- so it cannot reach the optima measured here either.

Nothing in this directory selects a count. Reading a preferred count off these tables would
be selection on the evaluation set, and the counts are nine correlated comparisons per model
against a single calibration draw. The sweep establishes only that the headroom above 256 is
large and that the penalty for overshooting is also large, so a count rule must be
predictive rather than fixed. Whether a rule that never sees this evaluation set can find
these optima is measured separately in [adaptive_count](../adaptive_count/).
