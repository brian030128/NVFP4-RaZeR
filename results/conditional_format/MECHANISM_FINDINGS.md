# Conditional format cost: first promising mechanism result

Job 331863 completed all numerical tests and the fixed nine-matrix panel.
Unlike the preceding spectral and activation-bound prototypes, this direction
passed its predeclared screening criterion without any parameter changes.

| Held-out inputs | Geometric mean reduction versus dynamic MSE | Matrices improved |
|---|---:|---:|
| WikiText validation | 2.68% | 9/9 |
| GSM8K reference text | 2.62% | 9/9 |
| MBPP reference text | 2.78% | 9/9 |

These are layer-output reconstruction measurements on identical pristine
inputs, not perplexity or generated-task accuracy. They cover full q/k/v
matrices at layers 0/8/15 of Llama-3.2-1B-Instruct. The twenty-seven cells are
not twenty-seven independent model replications.

One pass over 32 Wiki training windows produced the shared input second moments.
All policies used those exact moments and the same 64-column compensation
algorithm. Only the format objective differed. Weights and maps were frozen
before loading held-out inputs. No dataset-specific configuration was selected.

The conditional score measures the error remaining after an optimal update to
the future weight coordinates. It therefore accounts for which candidate's
error is easier to compensate. This is a direct use of established conditional
quadratic optimization, not a new universal weight-statistic rule. The sum of
recorded conditional costs reproduced the final damped reconstruction objective,
providing a numerical check of the derivation and implementation.

The crucial comparator is dynamic MSE with compensation; mixed-format GPTQ
integration alone already appears in prior work. The strong uniform layer
result motivates full-model evaluation, but does not establish novelty or task
generalization. Full-model continuation is specified before its results in
[MODEL_PROTOCOL.md](MODEL_PROTOCOL.md), including the unchanged algorithm,
an OPT architectural test, A16/A4 diagnostics and explicit screening criteria.

- [Mechanism protocol](PROTOCOL.md)
- [Complete numerical table](job_331863/REPORT.md)
- [Raw traces, hashes, calibration and evaluation fingerprints](job_331863/report.json)
