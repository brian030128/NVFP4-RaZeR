# Frozen Llama-3.1-8B 256×64 task accuracy

Raw 187-tile map versus accepted 147-tile reordered map. Zero-shot, no chat template, same H200 and batch ordering for each paired comparison. No benchmark fitting.

| Task | Questions | Raw | Reordered | Δ percentage points | Wins / losses | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| mmlu_non_stem | 10889 | 62.715% | 62.935% | +0.220 | 135 / 111 | 0.2848 |
| arc_challenge | 1172 | 51.280% | 51.195% | -0.085 | 11 / 12 | 1 |

MMLU non-STEM is document-weighted across the frozen humanities, social-sciences and other subjects; it is not a full MMLU score. ARC uses acc_norm. Exact two-sided McNemar tests are Holm-adjusted across these two endpoints. Per-subject results are descriptive. A nonsignificant positive delta is not established improvement. This measures the fake-quantized model, not the unverified native backend.
