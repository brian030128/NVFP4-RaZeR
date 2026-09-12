# Terminal-Bench 4.0 on Qwen3.8-27B: cancelled, and why

Three policies were run against `terminal-bench/terminal-bench@4.0.0`, 12 tasks each, one trial
per task, GPU tasks excluded, each served in process from the same weights the perplexity and
accuracy runs use so the activation-quantization hooks stay live.

| policy | precision | job | trials completed | tasks solved | elapsed |
|---|---|---|---:|---:|---|
| BF16 | BF16 | 339084 | 10 of 12 | **0** | 6:23 |
| NVFP4 FourOverSix | W4A4 | 339085 | 8 of 12 | **0** | 5:56 |
| MixFP4 k=3 | W4A4 | 339086 | 7 of 12 | **0** | 5:50 |

**Zero solves in 25 trials.** The benchmark cannot discriminate between quantization policies at
this model scale, because it does not discriminate between the model and a model that does
nothing: every trial scores 0, so every paired comparison is a tie and the sign test has no
discordant pairs to read. Running the remaining 11 trials would have cost about six more GPU
hours to confirm a result already visible at 25.

That is a statement about the benchmark's difficulty against a 27B, not about the method. It is
recorded here rather than dropped, because "we tried Terminal-Bench 4.0 and it returned no
signal" is the honest outcome and is worth knowing before anyone spends the hours again.

The runs were cancelled before harbor's own `timeout` expired, so the per-trial directories under
the compute nodes' `/tmp` were never copied back -- the table above is recovered from the harbor
progress logs in `kse_job_3390{84,85,86}/`, which do persist. Nothing is lost that the zero
rewards had not already settled.

## What replaced it

Terminal-Bench **2.0** (`terminal-bench@2.0`, the default legacy registry), 89 tasks in harbor
format, all of which download and none of which declare a GPU. It sits below 4.0 in difficulty,
so it has a chance of producing the nonzero pass rates a comparison needs.

Terminal-Bench **1.0** was considered and is not reachable. It is `terminal-bench-core@0.1.1` in
the older `laude-institute/terminal-bench` registry, and harbor 0.22 rejects those entries with
`tasks: Field required [type=missing]` -- they predate the field and carry the docker-compose
task layout the old `tb` CLI ran, not harbor's. Running it would mean standing up that harness
(docker, against the singularity path everything here is built on) or porting the task
definitions.
