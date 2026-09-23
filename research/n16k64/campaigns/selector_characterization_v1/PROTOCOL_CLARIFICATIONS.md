# Operational clarifications (no selector or endpoint changes)

Before launching granularity evaluations, their plans include the existing
frozen N8 and N16 maps as reproduction controls alongside the fresh FourOverSix
baseline. `scripts/n8_reproduction_audit.py --N 8` and `--N 16` require matching installed
weights and every window at the existing absolute NLL tolerance of 1e-10;
it does not fit a tolerance to new outcomes. This adds no selector or scientific
endpoint. The resource estimate includes one N8 and one N16 control per model
(13.7083 historical-rate GPU-hours for the initially missing matrix plus
controls, excluding waits/retries/accuracy; not a current remaining-cost measurement).
This is a plan and estimate, not a claim that reproduction has passed.

The frozen protocol's resource section records the user's explicit local GPU
authorization: at most three physical A6000/Ada devices in total, homogeneous
devices per run, foreign/unresolved owners forbidden. This supersedes the
stale phrase “no GPU without scheduler access” in its secondary-accuracy
description. The original frozen bytes are retained; no Slurm/H200 is required.

The first two Ada pilots are operationally complete, but do not reproduce
historical A6000 per-window NLL within the frozen tolerance. Both have matching
installed-weight and input-token hashes, and identical results to each other.
The second pilot used the historical teacher-instance and thread settings.
This rules out those changed settings as a sufficient explanation; it does
not establish the device architecture as the sole cause. Current source-file
hashes also differ from early primary-run manifests and are recorded explicitly.

The final reasoned device-class test completed on an exclusive A6000 as
attempt4: all tested Llama baseline/N16 windows matched exactly. No tolerance,
benchmark, map or selected tile changed. `ANCHOR_GATE.json` now permits
Llama scientific launches. Recovered-cache Mistral attempt2 and Qwen attempt2
subsequently also passed their own baseline/N16 prefix anchors exactly.
Waiting for an occupied device is not counted as a failed experiment.

At 2026-09-21 17:03 UTC, `pilot_llama_attempt3` passed empty-device
preflight but a foreign compute process subsequently appeared. The attempt
was invalidated after 25.27 seconds (0.007020 A6000 GPU-hours), with no
admissible numerical result. `pilot_llama_attempt4` is its first operational
co-tenancy replacement, retaining the same device-class diagnostic and
tolerance; it is not a third numerical repair. All attempt3 evidence remains
untouched. See `results/PILOT_COTENANCY_EVENT.json`. Resource accounting now
includes terminal invalid launches even when their reports are incomplete.

The parent-moment streaming adapter observes existing per-sequence N8 scores
after their CPU copy and accumulates sums/squared sums at N32–N256. It does
not refit candidates or approximate covariance from marginal SEs. It has not
yet passed a model-level calibration reproduction gate and is not evidence
that missing parent maps or quality results exist.

No new multiple-comparison significance claim is made in this characterization.
The 2,000 paired bootstrap intervals are pointwise exploratory summaries;
historical gate conclusions retain their own frozen multiplicity rules.

## Prospective full-window ingestion control

Before any full evaluation was launched, the pending T2/T3 plans were extended
with a full FourOverSix baseline anchor, saved as distinct `_anchored.json`
plans. Initial plan files are retained. This addresses numerical drift observed
in the pilots; it does not change the frozen selector, endpoints or tolerance.
Admission requires matching runtime/input identity, weight hash, and every
baseline window's NLL within 1e-10 of the historical reference. Aggregate
agreement cannot conceal cancelling window errors. Failed admission retains
the run but excludes it from paired analyses. Fresh and reused results have
separate status labels. Three CPU rejection fixtures test this logic, not
model-level correctness. The resource estimate includes five additional
baseline policies per model (four T2 draws and one T3 batch); actual costs
remain unmeasured. No full-window anchor is claimed to have passed yet.
