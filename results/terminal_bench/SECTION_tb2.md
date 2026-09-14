## Terminal-Bench (Qwen3.8-27B)

Terminal-Bench runs the model as an agent in a real container: it issues shell commands over many turns and the task's own verifier decides pass or fail. It therefore probes what the §1a multiple-choice panel cannot -- multi-turn instruction following, tool syntax, and recovery from the model's own errors -- on the one model in the panel large enough to attempt the tasks at all.

Served in process from the same weights the perplexity and accuracy runs use, so the activation-quantization hooks are live (BF16 = BF16, NVFP4 FourOverSix W4A4 = W4A4, MixFP4 k=3 W4A4 = W4A4). 13 tasks, 1 trial each, GPU tasks excluded.

| policy | precision | resolved | of | pass rate | errored | median agent min |
|---|---|---:|---:|---:|---:|---:|
| BF16 | BF16 | 0 | 12 | 0.0% | 8 | 8 |
| NVFP4 FourOverSix W4A4 | W4A4 | 0 | 11 | 0.0% | 9 | 8 |
| MixFP4 k=3 W4A4 | W4A4 | 1 | 13 | 7.7% | 7 | 8 |

The last column is the median minutes the agent phase ran. `errored` counts trials that never reached the verifier at all, so the pass rate is over the trials that did.

**These are not capability numbers.** The agent phase is capped, and the cap fires on most trials that get scored -- up to 13 of them here, which the verifier then grades on whatever state the agent had reached. Serving runs in process through Transformers so the W4A4 activation hooks stay live, at roughly 2 tokens per second, and a multi-turn task needs far longer than the cap allows. The cap is identical across policies, so the comparison between them is fair; what it measures is what each policy achieves within a fixed budget, and every pass rate here is a lower bound on the model.

### Paired against NVFP4 FourOverSix W4A4

Only tasks both policies completed are compared, and only the tasks they disagree on carry information -- an exact two-sided sign test on those.

| policy | wins | losses | ties | sign-test p |
|---|---:|---:|---:|---:|
| BF16 | 0 | 0 | 11 | 1 |
| MixFP4 k=3 W4A4 | 1 | 0 | 10 | 1 |

### Per task

| task | BF16 | NVFP4 FourOverSix W4A4 | MixFP4 k=3 W4A4 |
|---|---:|---:|---:|
| `break-filter-js-from-html` | fail | fail | fail |
| `largest-eigenval` | fail | fail | pass |
| `llm-inference-batching-scheduler` | fail | fail | fail |
| `log-summary-date-ranges` | fail | fail | fail |
| `modernize-scientific-stack` | fail | fail | fail |
| `path-tracing` | fail | fail | fail |
| `portfolio-optimization` | fail | fail | fail |
| `pytorch-model-cli` | fail | fail | fail |
| `regex-chess` | fail | fail | fail |
| `reshard-c4-data` | fail | fail | fail |
| `winning-avg-corewars` | fail | fail | fail |

**2 of 13 tasks are not in the per-task table** because at least one policy did not produce a verifier result for them (`feal-linear-cryptanalysis`, `prove-plus-comm`). They are excluded from the paired test rather than counted as failures.

**What 11 tasks can settle.** At one trial per task, a policy would have to win 6 tasks against zero losses before a two-sided sign test cleared 0.05, so this is sized to catch a large regression, not to certify equivalence. Read it as a smoke test of agentic capability under W4A4 rather than as a ranking.

