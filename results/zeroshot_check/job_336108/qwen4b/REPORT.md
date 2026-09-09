# Zero-shot accuracy check: qwen4b

Perplexity cannot distinguish a genuinely better model from a less overconfident
one. These are the same frozen policies on zero-shot multiple choice, where a
smoothing artefact should not help. BF16 restores pristine weights and removes
activation quantization; every other row uses the paper-aligned W4A4 path.

| Policy | arc_easy | arc_challenge | hellaswag | openbookqa | boolq | winogrande | mean |
|---|---|---|---|---|---|---|---|
| bf16 | 0.7816 | 0.5358 | 0.6846 | 0.4020 | 0.8495 | 0.6519 | 0.6509 |
| four_over_six | 0.7462 | 0.4906 | 0.6616 | 0.3940 | 0.8358 | 0.6212 | 0.6249 |
| n256 | 0.7563 | 0.4881 | 0.6652 | 0.3880 | 0.8330 | 0.6409 | 0.6286 |
| n65536 | 0.7778 | 0.5043 | 0.6725 | 0.4060 | 0.8453 | 0.6314 | 0.6395 |

Tasks skipped because their dataset could not be loaded in this environment: piqa.

If accuracy tracks the perplexity ordering, the perplexity gain reflects a
better model. If accuracy is flat or lower while perplexity improves sharply, the
perplexity gain on this model is a metric artefact and must be reported as such.
