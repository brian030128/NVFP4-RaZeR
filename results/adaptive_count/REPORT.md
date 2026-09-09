# Calibration-chosen tile count (job 335904)

The count is chosen by the lowest actual next-token loss over calibration-source
documents, across the same nine nested prefixes measured in
[cap_sweep](../cap_sweep/REPORT.md) (job 335887). The held-out C4 set is never
consulted for the choice; its perplexities below are a readout of an already-made
decision. All 14 policies evaluated by both jobs agree to the last digit.

`sel` chooses on 192 documents drawn from the same three calibration sources but disjoint
from the 192 scoring documents. `fit` chooses on the scoring documents themselves and is
reported only to show the overfitting that invites.

## Adaptive versus the fixed 256 cap

| Model | chosen (sel) | tiles | ΔPPL chosen | ΔPPL at 256 | ΔPPL best possible | vs fixed 256 |
|---|---:|---:|---:|---:|---:|---|
| OLMo-1B | n4096 | 4,096 | -0.140184 | -0.101424 | -0.140184 | -0.038761 (supported) |
| Pythia-1.4B | n4096 | 4,096 | -0.580167 | -0.308493 | -0.621039 | -0.271675 (supported) |
| Qwen3-4B | n_all | 80,978 | -2.863637 | -1.025501 | -2.863637 | -1.838136 (supported) |
| Llama-3.1-8B | n1024 | 1,024 | -0.090881 | -0.090246 | -0.090881 | -0.000635 (inconclusive) |

The calibration rule improves on the fixed cap in 4/4 point comparisons, three with
supporting descriptive paired two-SE intervals. It lands on the exact argmin of the
measured curve for OLMo-1B, Qwen3-4B and Llama-3.1-8B; on Pythia-1.4B it takes 4,096
where 1,024 was best, keeping most of the gain and staying clear of the harm region that
begins at 16,384.

## Why the selection set must be disjoint from the scoring set

| Model | chosen on `fit` | ΔPPL | chosen on `sel` | ΔPPL |
|---|---:|---:|---:|---:|
| OLMo-1B | n65536 | -0.130221 | n4096 | -0.140184 |
| Pythia-1.4B | n4096 | -0.580167 | n4096 | -0.580167 |
| Qwen3-4B | n_all | -2.863637 | n_all | -2.863637 |
| Llama-3.1-8B | n_all | +0.047524 | n1024 | -0.090881 |

Choosing on the scoring documents overshoots on two of four models, and on
Llama-3.1-8B it selects every eligible tile, which is a supported regression of
+0.047524 PPL. The scores were fitted on those documents, so their loss keeps falling
past the point where held-out loss turns. A disjoint selection set from the same sources
costs one extra forward pass per candidate count and removes the failure.

## Limits

This is the 512-token held-out C4 protocol of `results/c4_frozen`, with causal per-token
activation factors and eager attention. It is not the 2048-token paper-aligned protocol
used by `results/fixed256_paper_eval`, and these numbers are not comparable to the
published RaZeR table. The maps are protocol-independent, so the shape of the finding is
expected to carry, but its magnitude and the location of the harm onset must be
re-measured under tensor-wide activation factors and 2048-token windows before any
paper claim rests on them.

Four models, one calibration draw each, one selection draw each. Two-SE intervals are
descriptive evaluation-window intervals; they do not cover calibration-draw or
selection-draw variability, and the nine counts per model are correlated comparisons.
C4 is a calibration source for these maps, so this remains held-out within-source
evaluation. Confirmation requires the untouched literature, science, government and
WikiText families with the count rule fixed in advance.
