# Real-weight study: proposal failure isolated

The frozen weight-only panel and both diagnostics completed successfully:
Slurm jobs 331682 (panel/tests), 331688 (summary), and 331709 (exact finite
interventions). No text dataset, tokenizer, activation, label, or model forward
pass entered selection or diagnostics. No map was chosen using task loss.

## Full matrices

The panel contains twelve q_proj/down_proj matrices from first, middle, and last
layers of pinned Llama-3.2-1B-Instruct and Llama-3.1-8B development checkpoints.
There are 565,248 candidate 8x64 tiles. The approximate solver changed five
tiles across two matrices and left ten matrices unchanged.

| Changed matrix | Audited spectral-error change | Frobenius-error change | Selected tiles |
|---|---:|---:|---:|
| Llama 3.2 1B Instruct, layer 0 down_proj | -0.2877% | -0.000197% | 1 |
| Llama 3.1 8B, layer 0 q_proj | -0.3500% | +0.023294% | 4 |

Full-matrix spectral values are estimates, not certificates. The rank-8/64-step
selector and rank-16/128-step audit agree within 6.22e-7 relative on exported
matrices. Maximum audit relative eigen residual is 1.07e-6. These diagnostics
support numerical consistency but cannot certify the global largest eigenvalue.
Baseline estimated stable ranks range from 148.58 to 654.37.

The fixed MSE comparator lowers estimated spectral error on ten of twelve
matrices. Thus the sparse approximate spectral solver is not competitive here
even on its own intended objective. All twelve runs stop after a rejected
proposal, not after exhausting their 16-step cap.

Total measured selection time is 8.36 seconds on one H100, peak allocated GPU
memory is 2.40 GiB, and the panel runner took 14.02 seconds including diagnostics.
This is a sampled-matrix timing, not whole-model quantization cost. The recorded
candidate_seconds field includes the weight hash operation as well as candidate
construction; it should not be read as isolated quantizer kernel time.

## Distinguishing estimator error from proposal error

Thirty-six fixed 32x128 crops preserve the full-matrix quantized candidates and
their tensor normalization. Exact float64 SVD evaluates every final crop map.

| Diagnostic | Result |
|---|---:|
| Exact coordinate solver improves spectral error | 33/36 |
| Approximate solver improves spectral error | 16/36 |
| Approximate solver worse than exact reference | 31/36 |
| Approximate solver regresses from baseline | 0/36 |
| Largest crop relative spectral-estimation error | 8.63e-7 |

The eigenvalue estimate is accurate on these crops, but the proposal rule loses
many available improvements. Exact coordinate descent itself is not a global
optimum; the earlier synthetic enumeration already established that limitation.

## Exact finite interventions explain the failure

A separately declared post-panel diagnostic evaluates every individual toggle
at each crop's all-E2M1 baseline, using exact singular vectors and values.
In 20/36 crops, the best toggle for the OLD worst input direction lowers error
in that direction while INCREASING the actual spectral norm. In 17 of those
20 crops, another individual toggle lowers the true spectral objective.
Old/new worst-direction absolute cosines are at most 0.566 in the harmful cases.
Candidate and source-weight hashes match the original panel.

This is a specific, reproducible mechanism: improving a single current worst
direction does not control the new worst direction after a finite switch.
The score ||(E+Delta)v||^2 is only a lower bound on ||E+Delta||_2^2 for unit v.
Stopping after its best-ranked proposal fails can therefore miss useful moves.
Using more accurate power iterations alone cannot fix the demonstrated crop
failure, because the diagnostic uses exact singular vectors already.

## Research implication

The immediate obstacle is a weak optimizer, not evidence that spectral error
cannot be improved on these weights. A next solver must account for new adverse
directions during candidate selection, with a mathematically justified bound or
joint objective. Merely sweeping proposal counts or numerical ranks until model
loss improves would violate the intended generalization protocol.

The objective also remains unvalidated: the synthetic typical-input tradeoffs
and the 8B matrix's Frobenius increase persist. Neither a correct optimizer nor
a smaller spectral norm establishes improved LLM loss. No whole-model map or
cross-domain evaluation was produced in this feasibility stage. Both models are
development evidence from one family, not new-family generalization evidence.

## Artifacts

- [Frozen panel protocol](WEIGHT_PROTOCOL.md)
- [Full matrix results](weights_331682/REPORT.md)
- [Exact crop and audit diagnostics](weights_331682/DIAGNOSTICS.md)
- [Post-panel diagnostic protocol](PROPOSAL_DIAGNOSTIC_PROTOCOL.md)
- [Exact finite-switch results](weights_331682/PROPOSAL_DIAGNOSTICS.md)

Raw reports preserve all policies, traces, offsets, hashes, timing and masks.
Saved masks cover sampled matrices only; they are not deployment exports.
