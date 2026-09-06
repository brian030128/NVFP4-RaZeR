# Development observations

These observations motivated the frozen follow-up in PROTOCOL.md. The final
test data were not used to choose the follow-up budget or confidence margin.
See the per-model report.json files for every sequence-level measurement.

## Task gradients identify useful finite switches

All values below are mean NLL changes per token. Fit has 16 sequences; probe
has 8 separate sequences. The tensor perturbation is a legal E2M1-to-E0M3
switch on each selected 8x64 tile, with both grids fixed at alpha=1.

| Model | Intervention | Predicted fit change | Measured fit change | Measured probe change |
|---|---|---:|---:|---:|
| Qwen3-4B | strongest beneficial 256 | -0.236158 | -0.135531 | -0.151871 |
| Qwen3-4B | strongest harmful 256 | +0.254380 | +0.521988 | +0.561732 |
| Qwen3-4B | strongest beneficial 4096 | -0.859937 | -0.256350 | -0.259686 |
| Qwen3-4B | strongest harmful 4096 | +0.874707 | +2.857491 | +2.844101 |
| Llama-3.1-8B | strongest beneficial 256 | -0.027624 | -0.009646 | -0.005980 |
| Llama-3.1-8B | strongest harmful 256 | +0.024495 | +0.024074 | +0.012639 |
| Llama-3.1-8B | strongest beneficial 4096 | -0.126023 | -0.014797 | -0.014333 |
| Llama-3.1-8B | strongest harmful 4096 | +0.119116 | +0.179598 | +0.063166 |

The sign information transfers to probe text for these interventions; the
linearized magnitude is not a reliable prediction of the finite change.

## Broad selection fails

Validation is 16 additional sequences, disjoint from fit and probe.

| Model | Rule | Fraction of tiles | Validation NLL change |
|---|---|---:|---:|
| Qwen3-4B | mean(score) < 0 | 0.501043 | +0.507741 |
| Qwen3-4B | mean(score) + 2 SE < 0 | 0.127782 | +0.213497 |
| Llama-3.1-8B | mean(score) < 0 | 0.502905 | +1.266382 |
| Llama-3.1-8B | mean(score) + 2 SE < 0 | 0.033313 | +0.192855 |

Per-tile stability is insufficient. Millions of finite changes cannot be
treated as independent infinitesimal perturbations. These observations
motivate the fixed cumulative predicted-loss budget, but do not by themselves
identify how much failure comes from interactions, higher-order terms,
activation-rounding discontinuities, or selection noise.

## Frozen sparse rule on the development pair

The same 0.1-nat budget and 2-SE eligibility rule select 50 tiles on Qwen3-4B
and 6387 on Llama-3.1-8B. Validation NLL changes are -0.085777 and -0.014756.
Matched random masks preserve each projection's selected count and stay near
baseline. Final test measurements and seed replications belong in REPORT.md.

## Which tiles are selected

A CPU analysis step inside allocation 328989 inspected the saved maps. The
standing fixed rule selects very few of the tiles chosen by task sensitivity:

| Model | Task-selected tiles | Also selected by impg16_h10 |
|---|---:|---:|
| Qwen3-4B | 50 | 3 |
| Llama-3.1-8B | 6387 | 81 |
| Qwen3-8B | 291 | 18 |
| Llama-3.2-1B-Instruct | 501 | 18 |
| Qwen3-14B | 462 | 24 |
| Llama-3.1-8B-Instruct | 2543 | 51 |

Their locations differ as well. Early v_proj matrices feature prominently in
the Llama selections; late q_proj and MLP matrices feature prominently in the
Qwen selections. Llama-3.1-8B also selects 815 tiles in its final down_proj.
These are descriptions of the resulting maps, not hand-written module rules.
The matched-random control preserves each projection's selected count, so it
tests tile identity beyond merely choosing which projections receive changes.

## Cluster execution record

Initial submissions encountered socket timeouts and a cluster-side
`get_api_token` error. A timed-out request was later accepted as job 328953;
that duplicate was stopped. Completed initial diagnostics: srun jobs 328955
(Qwen) and 328961 (Llama). Follow-ups use arrays with concurrency %2 and
dependencies, each task requesting exactly one H100:

* 328973: two development models, frozen sparse policy, full final evaluation.
* 328977: four transfer models, after all tasks of 328973 finish.
* 328979: independent-seed repeats of the development pair, after 328977.
* 328981: CPU tensor summarization and plotting on one allocated H100 node
  (GPU idle), after 328979.

No model work, tensor analysis, test execution, or plotting runs on the login
node. Source editing, log reads, and Slurm queries/submissions run there.

Further jobs: 328990 verified the calibration-only export; 329025 runs both
64-sequence Llama seeds; 329030 summarizes them and runs the latest structural
checks. Array 329033 depends on successful completion of 329030 and tests
discrete trust backtracking on both seeds, reusing 64-sequence scores.
Each array is limited to two one-H100 tasks; dependencies prevent overlap
with the preceding compute/summary jobs.

329034 summarizes the backtracking results. 329036 verifies the production
calibration-only validation gate on the repeated Qwen (accepted, 31 tiles)
and Llama (rejected, zero exported E0M3 tiles) maps. 329038 performs fresh
64-sequence scoring and unchanged backtracking on seed 20260908 for both
development models. 329039 generates the final combined report, CSVs and
figures, verifies exported map identities, and runs structural checks.
These jobs follow the same dependency chain, with a maximum of two H100s.
329049 regenerates the final report with the explicit C4-domain caveat and
reruns the checks after that reporting update; no model experiments repeat.
