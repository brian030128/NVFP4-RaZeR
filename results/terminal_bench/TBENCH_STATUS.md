# Terminal-Bench on Qwen3.8-27B: what was run, and why none of it is a result yet

## Correction

An earlier version of this file concluded that Terminal-Bench 4.0 "cannot discriminate between
quantization policies at this model scale" because all 25 completed trials scored zero. **That
conclusion was wrong, and the zeros were very largely the harness configuration rather than the
model.** The correction is recorded here rather than by quietly rewriting, because the wrong
version was stated confidently.

What the 4.0 runs actually establish is only that 25 trials scored zero. The 2.0 runs, which
kept their per-trial directories, show what that number is made of.

## Terminal-Bench 2.0, 20 tasks per policy

| policy | trials | scored | solved | scored after a timeout | never scored (container) |
|---|---:|---:|---:|---:|---:|
| BF16 | 20 | 12 | 0 | 11 | 8 |
| NVFP4 FourOverSix W4A4 | 20 | 11 | 0 | 10 | 9 |
| MixFP4 k=3 W4A4 | 20 | 13 | **1** | 12 | 7 |

**A second correction, to the first correction.** The table above originally read "1 of 20
reached the verifier" for both policies. That was a bug in the analysis, not in the run: a trial
can carry both an exception and a verifier reward, because the agent phase can time out *after*
the task is already solved and harbor grades it anyway. `largest-eigenval` under MixFP4 is
exactly that -- `AgentTimeoutError` and `reward: 1.0` together. Both scripts treated any
exception as "never scored" and discarded the reward, which hid the only solve in the whole
experiment. Reward now decides and the exception is context.

So 11 to 13 of 20 trials were scored, and only the 7 to 9 container failures truly never ran.
The pass rate to quote is over the scored trials.

### The result, such as it is

BF16 0 of 12, FourOverSix 0 of 11, MixFP4 k=3 1 of 13. Paired against FourOverSix on the tasks
both scored, MixFP4 wins 1 and loses 0 (sign test p = 1.0) and BF16 ties 0-0. Nothing separates
the policies, which at this sample size is the only outcome available: six clean wins are needed
to clear 0.05.

What it does support, weakly, is that W4A4 does not visibly destroy agentic capability relative
to BF16 -- all three sit at the same near-zero rate, and the one solve belongs to a quantized
policy. Read it as a smoke test, not a ranking.

### Cause 1: the agent timeout was set far below what this model needs

`AGENT_TMULT=0.25` caps the agent phase at 900 seconds. Serving is in process through
Transformers so that the W4A4 activation hooks stay live, which measures about **2 tokens per
second** on this model (`results/terminal_bench/tput_339031/throughput.json`). A 15 to 20 turn
task at roughly 400 tokens a turn needs 6,000 to 8,000 tokens, or 50 to 65 minutes. Fifteen
minutes cannot finish one, so the timeout fires on nearly every task that does not fail earlier
for another reason -- 10 to 12 times per policy. Those trials are still graded, on whatever
state the agent had reached, which is why every pass rate here is a lower bound rather than a
measurement of the model. The multiplier was chosen to bound the wall clock of the run; what it
actually bounded was the number of trials that could produce a score.

The 4.0 runs used the same 0.25. Their zeros are therefore not evidence about 4.0's difficulty.

### Cause 2: the container prelude cannot supply python3 on these images

`scripts/harbor-container-prelude.sh` points `/usr/bin/python3` at whatever interpreter the
image already has. Terminal-Bench 4.0 images carry one; 2.0 images do not, so harbor's bootstrap
falls back to installing it and dpkg fails inside the user namespace:

    dpkg-deb: error: paste subprocess was killed by signal (Broken pipe)
    dpkg: error processing archive .../libpython3.12-minimal_3.12.3-1ubuntu0.17_amd64.deb
    [harbor] FATAL: cannot install /usr/bin/python3

The prelude's `path-exclude` entries for `/usr/share/{doc,man,info}` are the first thing to
suspect: they were added to avoid a cross-device rename when installing tmux, and dpkg pipes the
archive through a filter when they are set. That has not been confirmed, and the prelude's doc
exclusion is load-bearing for a different failure, so it needs testing rather than deleting.

## What a real Terminal-Bench number would cost

At 2 tokens per second a task needs roughly an hour of agent time, so 20 tasks is about 20 GPU
hours per policy and 60 for the three-policy comparison. That buys a comparison which, at one
trial per task, still needs **six clean wins** before an exact two-sided sign test clears 0.05 --
so it would be sized to catch a large regression, not to rank the policies.

The throughput is the binding constraint, and it is not incidental: the in-process server exists
because neither vLLM nor SGLang can run the simulated W4A4 activation hooks, and serving BF16
through a fast engine while the quantized policies go through a slow one would not be a
comparison. Any cheaper version of this measurement has to start there.

## Files

Per-trial results for the 2.0 runs are under `kse_job_3399{98,99}/jobs_<policy>/`; the failure
breakdown above comes from `scripts/tbench_failure_report.py`. The 4.0 runs were cancelled
before harbor's copy-back, so only their progress logs survive in
`kse_job_3390{84,85,86}/harbor_*.log` -- which is why the 4.0 zeros cannot now be broken down
the same way, and one reason not to cancel a run whose per-trial data has not been written out.
