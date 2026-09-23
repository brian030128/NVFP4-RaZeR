# Reproduce and resume

## Current retry authority — 2026-09-23

The user explicitly authorized **one additional attempt each**, only for Llama
seed0 calibration, Mistral seed0 calibration, and Qwen coarse-granularity
evaluation. See `plans/additional_retry_authorization_20260923.json` and
`results/ADDITIONAL_RETRY_PLAN.json`. Earlier no-attempt4 statements below
record the previous limit and are superseded only for these three named jobs.
No fifth attempt or fourth Llama secondary-accuracy attempt is authorized.
The grant is not numerical acceptance. All original gates remain unchanged.

Mistral uses the CPU-tested `calibration_repair_run_v2.py` with its exact scoped
authorization record. Llama uses `calibration_llama_retry_run.py` and
`stream_calibrate_llama_historical.py`: the V30 source manifest verifies the
same recovered quant hash, but Llama's historical `--raw sample` explicitly
has **no** `--subset-moments`. Four CPU tests check scope, command transformation,
runpy imports and no-authorization refusal. The parent observer is unchanged.
Qwen uses the existing guarded `gpu_run_wait.py` and the pinned v2 map plan.
Read `NEXT_ACTION.md` and actual live PIDs before any launch; these authorized
attempts are already submitted, not commands to duplicate.

Only validated exact score streams and N8/N16 moments may admit regenerated
parents. A successful process exit is insufficient. Additional coarse maps
can be constructed only after `validate_parent_moments.py` accepts them.
Three physical GPUs remains the maximum including the ongoing Mistral accuracy.
Both authorized calibration attempts subsequently completed but failed exact
historical score identity. `results/AUTHORIZED_CALIBRATION_AUDIT.json` records
the forward matches and N8/N16 mask discrepancies. Their parent sets are not
admitted; do not resubmit either attempt or create attempt5. Qwen's authorized
evaluation passed full admission: eight new coarse cells and exact full N8/N16
reproduction on both corpora. All three authorized attempts are terminal; no
GPU launch remains active. Mistral accuracy subsequently passed full admission;
Qwen/Mistral's 32 new task cells plus 48 historical cells give 80/96 coverage.
Do not rerun either accepted accuracy model. Llama's 16 cells remain blocked.

Resource-only waiting reassignment uses `scripts/reassign_waiting.py` and
requires an exact owned PID/start-time, no GPU child or lease, and the shared
registry lock. It supports the pinned PPL wait and accuracy adapters, **not**
the final calibration-repair adapter. No running evaluation may be moved.
The previous helper is preserved at
`scripts/history/reassign_waiting_pre_accuracy.py` (SHA-256
`73cc5559b25eadecf7be0038c2c0104e1e0eb041c9cd44267020c3ae049b2061`),
matching both earlier reassignment records. This history copy is evidence,
not an alternative launch entrypoint. UUID/name reassignment does not change
the frozen scientific plan or grant an additional failed-attempt retry.

Operational clarification: the frozen `evaluation.secondary_accuracy` text
retains an obsolete "no GPU without scheduler access" phrase. The user's
explicit local-GPU authorization and the same file's `resources.gpu_policy`
supersede that scheduler requirement (also recorded in `DATA_AUDIT.md`). They
do not waive ownership checks, the three-device cap, numerical identity gates
or retry limits. The hash-frozen protocol file itself is not rewritten.

The T2 controller `scripts/t2_continue.py` exited normally after all 150 primary
PPL cells passed ingestion on 2026-09-22. Secondary accuracy remains pending;
do not restart completed PPL plans. Use `--inspect-only`
to inspect its next action without launching evaluations. Without that flag,
it observes existing launchers, validates completed results through
`quality.py`, and submits only missing frozen model/draw plans using the
exclusive A6000 launcher. A model has at most one in-flight plan. It never
automatically retries an unaccepted terminal attempt. Inspect
`research_runs/mixfp4_selector_characterization_20260921T133858Z/controller/t2_status.json`
from the repository root. Do not run another concurrent T2 ingestion writer
while this controller is active. Its process must remain attached/monitored;
a log file alone is not a session notification mechanism.

Run from repository root at source HEAD
`db63419cc33b2bbbda2117aad636435a2956532d`. Do not switch an active checkout.
The scripts resolve the repository from their own location. Input locations
are local-only historical artifacts; they are not distributed model weights
or guaranteed GitHub downloads. The verified handoff ZIP SHA-256 is
`54a5eadeceafab590cb670f79c7f8a981f0132531d7e33c8e23206777a4e0e74`.

## Environment and inputs

Python 3.11.11, PyTorch 2.9.0+cu128, NumPy 2.4.4, transformers 5.16.1,
SciPy and Matplotlib from the existing primary campaign environment.
These are the execution-environment versions, recorded in actual per-run
`job_result.json` files. `results/ENVIRONMENT.json` preserves the **initial
audit shell** (Python 3.12.9, PyTorch 2.8.0, NumPy 2.5.1, no Slurm), not the
environment used to generate new GPU outcomes; its initial zero GPU-hours
is not the campaign's current cost. Use `results/GPU_COST.json` for terminal
costs and the run-specific environment/source manifests for provenance.
No installation or model download is needed in the original workspace.
External reproduction needs the exact cached models/tokenizers, frozen data
revisions, calibration manifests, moments, maps and evaluation arrays listed
in `artifact_inventory.csv` and `results/INPUTS.json`.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
SC=research/n16k64/campaigns/selector_characterization_v1
PY=research_runs/mixfp4_n16k64_full_validation_20260911T065444Z/env/venv_main/bin/python
$PY "$SC/scripts/test_math.py"
$PY "$SC/scripts/supporting_audit.py"
$PY "$SC/scripts/raw_artifact_search.py"
$PY "$SC/scripts/stability.py"
$PY "$SC/scripts/quality.py"
$PY "$SC/scripts/objective_tables.py"
$PY "$SC/scripts/granularity.py"
$PY "$SC/scripts/granularity_quality.py"
$PY "$SC/scripts/mechanism_review.py"
$PY "$SC/scripts/reorder_context.py"
$PY "$SC/scripts/composition_context.py"
$PY "$SC/scripts/secondary_accuracy.py"
$PY "$SC/scripts/accuracy_samples.py"
$PY "$SC/scripts/map_count_summary.py"
$PY "$SC/scripts/plot_results.py"
```

These CPU commands regenerate derived tables/plots in this new campaign;
they never edit archived inputs. `audit.py` additionally verifies the frozen
protocol and refuses a changed definition. Map writes are immutable: an
existing map must be byte-identical. Preserve completed outputs/checkpoints
before intentionally changing an analysis implementation.

## GPU resume rules

Registry qualification: the generic launcher's original `RUN_LOG.jsonl`
events label every non-calibration job `T2`, including granularity/reproduction
PPL jobs. This is a provenance-label defect, not their scientific scope. The
`scientific_task` field in `results/GPU_POLICY_AUDIT.json` resolves membership from the
actual command/frozen plan; original events remain unchanged. Use that field
and the T2/T3 job plans when counting task-specific attempts.

Read `TASK_STATUS.json`, `NEXT_ACTION.md`, `results/ANCHOR_GATE.json` and
`RUN_LOG.jsonl` first. Inspect running processes and the local lease directory
before starting anything; do not duplicate the waiting pilot.

`gpu_run.py` checks both nvidia-smi/NVML and PID owners before launch,
at most every 30 seconds during work, at child phase boundaries and afterward.
It waits on unavailable hardware; foreign or unknown owners fail closed.
It permits at most three campaign leases and uses one homogeneous GPU per run.
Only its own child may be terminated on invalidation. All failed attempts stay.

The Llama device-class diagnostic attempt4 completed and passed the prefix
anchor exactly; attempt3 remains invalid due to foreign co-tenancy. The
following records the completed command, not a request to submit a duplicate:

```bash
$PY "$SC/scripts/gpu_run.py" --name pilot_llama_attempt4 \
  --model llama8b --pilot --teacher instance --device-type a6000
$PY "$SC/scripts/anchor_audit.py"
```

Do not rerun that name if its directory already exists. Inspect it and resume
analysis; do not erase it. All three model prefix anchors have subsequently
passed. The following is the historical first T2 submission, now superseded
by the complete 150/150 PPL panel; it is not a pending or queued job:

```bash
$PY "$SC/scripts/gpu_run.py" --name t2_llama_draw1_attempt1 \
  --model llama8b --plan "$SC/plans/llama8b_draw1_anchored.json" --teacher instance --device-type a6000
```

Mistral/Qwen require `--hub-cache` pointing to the verified follow-up cache
in `results/RECOVERED_CACHE_VERIFICATION.json`; the default primary hub lacks
those snapshots. `pilot_mistral_attempt2` and `pilot_qwen_attempt2` already
exist. The explicit hub option only changes file resolution, not revisions,
formats, evaluation windows, or numerical tolerance. Reverify cache content
with `scripts/verify_recovered_cache.py` if it moves or changes.

Use `job_plan.csv` for all missing logical cells; never select plans by observed
quality. Regenerate `quality.py` to verify hashes, runtime/window identity,
token sums and pairing before accepting new outcomes.

Use the `_anchored.json` plans, not the retained initial plans. Each adds a
full FourOverSix baseline validation control. New results are admitted only
when every baseline window reproduces the historical NLL within the original
1e-10 tolerance, with identical runtime/input identity and installed-weight
hash. Pilot agreement alone is insufficient. New admitted results are labeled
`validated_new`; historical reuse remains `validated_reuse`.

For missing Llama/Mistral parent scores, `--calibrate-stream` observes the
unchanged scorer and saves parent sufficient statistics. It remains behind
the model anchor gate. Its newly regenerated N8/N16 scores/maps must also
reproduce the historical anchor before any new coarse map is accepted.
Do not claim the adapter has been model-validated merely because it parses.

After a complete regeneration, run `scripts/validate_parent_moments.py` with
the same Python environment. It requires exact historical per-module score
stream hashes and N8/N16 moments before writing the accepted-parent registry.
Then `granularity.py` consumes the validated parent moments, and
`remaining_plan.py` emits their evaluation plans. An empty registry means
no regenerated calibration has passed; it is not a successful model test.

All three A6000 baseline/N16 prefix anchors have passed. Full-window baseline
validation remains mandatory before admitting fresh outcomes. Granularity
plans additionally include the frozen N8/N16 maps; after those runs complete,
execute `scripts/n8_reproduction_audit.py --N 8` and the same command with
`--N 16` using the same Python environment. `results/N8_REPRODUCTION.json`
and `results/N16_REPRODUCTION.json` distinguish these model-level checks from CPU
map-reader tests. Its current pending status is not a numerical failure.

Use `plans/*_granularity_anchored_v2.json` for subsequent granularity runs.
The original Qwen plan is retained as failed-attempt evidence: its logical
`n8_reproduction`/`n16_reproduction` names were incorrectly used as the
header `map_policy` values. v2 changes only these to `n8_joint`/`n16_joint`;
map bytes, hashes and logical output names remain unchanged. Run
`scripts/test_map_backend.py` before submission; its plan-metadata test uses
the actual evaluator reader and verifies rejection of the old labels.
See `results/GRANULARITY_PLAN_REPAIR.json`. Qwen attempt2 was subsequently
invalidated by foreign co-tenancy, as was Llama calibration attempt2.
Llama calibration attempt3 and Qwen granularity attempt3 were invalidated
again and exhausted their two repair retries. Their coarse-N job-plan commands
are blank and status is `blocked_retry_limit`, not pending automatic execution.
A coordinated exclusive window plus explicit additional-retry authority is
needed before any corresponding attempt4. See
`results/COTENANCY_RETRY_LIMIT.json` and `results/GPU_RESOURCE_BLOCKERS.json`.

Future still-authorized jobs use `scripts/gpu_run_wait.py --only-uuid
GPU-9cec7336-5b30-3f86-35e7-06919156e7da` with the usual launcher arguments.
This adapter hash-pins `gpu_run.py` through `gpu_run_on_uuid.py`, narrows
candidate selection and waits on prelaunch inventory query failures without
creating a GPU attempt. It retains all ownership and lease checks. `--inspect-only` validates its source
without GPU access. The T2 controller used the same `--only-uuid`
restriction. At the 2026-09-22 06:18 UTC checkpoint, Qwen draw3/draw4 were
validated, Llama draw4 was running, and Mistral draw4 attempt3 was waiting;
inspect current live PIDs before any submission. The UUID is this workspace's device
identity, not a portable default or a guaranteed exclusive reservation.
Mistral `calibration_mistral7b_attempt1` was invalidated by GPU inventory
timeouts; no parent moments were admitted. Attempt2 was queued at 06:28 UTC,
with `--calibrate-stream --device-type a6000` and the verified recovered hub,
on separate A6000 UUID `GPU-2e6984d6-fde8-3ed3-2f83-14ec2228a8b9` so it does
not compete with the T2 queue. Its initial preflight rejected a foreign PID
and waited without launching a model. See
`results/MISTRAL_CALIBRATION_REPAIR2_PLAN.json`; check live state before retrying.
That original wait was subsequently safely reassigned, before any lease/model,
to `calibration_mistral7b_attempt2_reassigned` on `GPU-9cec...` after the already
running T2 job. The original directory and `waiting_reassignment.json` remain
intact. `scripts/reassign_waiting.py` holds the shared lease lock, checks exact
PID/start-time/UID/command/cwd, and refuses any launcher with a lease, child or
launch record. Only the resource UUID and run-directory name change; this is
the same second attempt, not another retry. After T2 finished, this calibration
started at 08:13:45 UTC on its newly verified A6000. It completed operationally
but was rejected for nonidentical score streams and N8/N16 maps; see
`results/MISTRAL_CALIBRATION_IDENTITY_DIAGNOSTIC.json`. No coarse parent set
from that attempt is eligible for reuse.
The GPU policy audit
hashes the record and checks scientific arguments are unchanged.
On completion, run `validate_parent_moments.py` before `granularity.py`;
reject nonmatching sequence-score digests rather than relaxing tolerances.

The remaining executable work is T2 secondary accuracy and Mistral stored-map
full-window reproduction. All three models' missing coarse T3 evaluations are
now blocked by exhausted repair limits; see `results/GPU_RESOURCE_BLOCKERS.json`.
Llama/Mistral additionally lack admitted exact parent moments. GPU authority
does not override the repair limit.
Wait for uncontended A6000 devices. Do not weaken tolerance, silently merge
Ada/A6000 baselines, or consume foreign GPUs. Each failure has at most two
reasoned repair retries; independently feasible CPU work continues.

The later bounded Ada resource probes for Mistral and Qwen both completed
operationally but failed the same numerical prefix tolerance, despite exact
token and installed-weight identities. `results/ADA_ACCURACY_RESOURCE_RESULT.json`
records the maximum differences and costs; no Ada accuracy adapter/evaluation
was promoted. Along with the earlier Llama failure, this supplies model-specific
evidence for retaining the A6000 waits. It is not a native-performance test
or authority for another exhausted T3 attempt.

## Secondary accuracy preparation and ongoing execution

`scripts/secondary_accuracy.py` audits historical coverage and freezes three
`plans/*_accuracy_seed0_anchored.json` plans: fresh FourOverSix baseline plus
CE/KL matched-quota maps, fixed seed0, full8 suite. These are **accuracy** plans;
do not pass them to the current PPL-only `gpu_run.py` interface.
`results/SECONDARY_ACCURACY_PLAN.json` records the exact historical recipe
sources, batch size, package version, and approximately 9.45 GPU-hours for
nine policy evaluations at historical rates. This is not measured new cost.

`scripts/prepare_accuracy_cache.py` creates a byte-verified dedicated Arrow
cache in the new runtime's `cache/accuracy_datasets`, excluding historical
lock files. The 258-file/192-split audit is in
`results/ACCURACY_CACHE_AUDIT.json`. Original caches stay unchanged. This proves
copy integrity and row counts only, not identity of future tokenized prompts.
Before new accuracy reuse, verify baseline per-example identities and retain
exact task configs, valid/failed samples and policy map hashes.

The separate `scripts/accuracy_run.py` adapter pins the unchanged launcher's
SHA-256, changes its evaluator payload/cache/audit labels in memory, optionally
restricts the UUID, and uses the hash-pinned `gpu_run_wait.py` helper only for
prelaunch inventory errors. Errors there mean waiting without a GPU lease;
its lease/preflight/monitor/postflight checks are otherwise unchanged. It never edits the
live launcher. Four CPU transformation tests and `--inspect-only` for each
model passed. Qwen's guarded first repair subsequently passed full baseline
equivalence and admitted 16 new task cells; see `results/ACCURACY_NEW_AUDIT.json`.
The example below is static inspection only, not an outstanding launch:

```bash
$PY "$SC/scripts/accuracy_run.py" --name accuracy_llama8b_seed0_attempt1 --model llama8b --only-uuid GPU-9cec7336-5b30-3f86-35e7-06919156e7da --inspect-only
# Do not remove --inspect-only: consult current retry limits and live state.
```

Current model runs use the pinned `cadence_launch.py --entry accuracy_run.py`
wrapper; do not start a duplicate or bypass an exhausted retry budget. The adapter
accepts no alternate task subset, policy or limit. Do not bypass it with a
direct unguarded evaluator command, and do not mark static inspection as
scientific accuracy acceptance.

`accuracy_new_results.py` is the separate new-run admission path. It pins the
frozen plan, checks source against accepted PPL runs, checks every GPU monitor
gap, verifies exact historical baseline prompt IDs/responses/correctness and
runtime, then writes matched-policy paired arrays without overwriting historical
ones. Eight CPU fixtures test rejection of aggregate-only matches, monitoring
gaps, duplicate endpoints and overwriting verified history. Its initial output
had zero accepted new models; this historical state is superseded by the
current audited coverage in `results/ACCURACY_NEW_AUDIT.json`, not evidence
that pending or invalid models passed. `secondary_accuracy_current.csv`
combines the historical 48 task cells with admitted new cells, keeping the
original historical coverage table unchanged.

Qwen `secondary_accuracy_qwen4b_attempt1` failed closed at `load_model` when
a foreign process appeared after its clean preflight; no outputs were admitted.
Its first repair was queued as `secondary_accuracy_qwen4b_attempt2` on GPU3.
After the successful Llama reproduction released GPU1, that **never-launched**
wait was safely reassigned to `secondary_accuracy_qwen4b_attempt2_reassigned`.
The original directory, preflight hash and `waiting_reassignment.json` remain.
GPU1 was then reacquired by a foreign process and the new preflight correctly
rejected launch. This is still the first repair waiting, not an additional
scientific attempt, and not a guarantee of a sustained exclusive interval.
Consult `results/GPU_QUEUE.json` before submitting anything; do not duplicate
the live accuracy wait or the ongoing Llama/Mistral accuracy evaluations.

### Independent Llama reproduction checkpoint

`remaining_plan.py` also freezes `plans/llama8b_reproduction_only_v1.json`
from the verified existing N8/N16 maps when exact coarse moments are unavailable.
It contains only FourOverSix plus full-window N8 and N16 reproduction, not
calibration or a coarse-map retry. The real evaluator map reader verifies its
policy headers in `test_map_backend.py`. Its first repair has now passed full
N8/N16 reproduction on both corpora (`results/N8_REPRODUCTION.json` and
`results/N16_REPRODUCTION.json`); do not resubmit it. The command below is the
historical first-attempt record, not a pending execution instruction:

```bash
$PY "$SC/scripts/gpu_run_wait.py" --only-uuid GPU-d0ab9929-0ef6-d612-0cb5-718f4fdd6c24 --name reproduction_llama8b_attempt1 --model llama8b --plan "$SC/plans/llama8b_reproduction_only_v1.json" --teacher instance --device-type a6000
```

This command was queued at 2026-09-22 06:39 UTC and initially waited because
of a foreign compute PID. It is recorded, not an invitation to resubmit a
duplicate. See `results/LLAMA_REPRODUCTION_LAUNCH_PLAN.json` and live state.
Because that UUID stayed foreign-occupied, the never-launched wait was safely
reassigned to `reproduction_llama8b_attempt1_reassigned` on `GPU-9cec...`, after
the Mistral calibration. The lease-lock/PID/owner/child checks and zero-GPU
reassignment record are the same as above. Neither baseline/maps nor scientific
arguments changed, and the original waiting directory is retained.
That reassigned first reproduction attempt was subsequently invalidated by a
foreign compute PID after clean preflight. Do not reuse its output. A new run
directory and a reasoned resource repair are required for its first retry;
the original two-retry ceiling remains in force.
The first repair, `reproduction_llama8b_attempt2`, subsequently completed and
passed both N8/N16 full-window checks (maximum NLL difference 0 on all four
map/corpus comparisons). No further reproduction retry is needed. This does
not resolve the separate exhausted Llama calibration budget or coarse cells.

The final Mistral calibration repair uses `calibration_repair_run.py`, with
`--name calibration_mistral7b_attempt3 --model mistral7b --calibrate-stream`,
the verified recovered hub and an explicit A6000 UUID. It restores the original
`--raw sample --subset-moments` options and imports the exact historical quant
text only after matching SHA-256
`210d182478d9bab9ce6c9b8e6d8ff831420ca83f31a2ecf170998c13ca641aca`.
The restoration is process-local; no historical source file is edited. CPU
tests verify the transformation and unchanged observer. This is a justified
final reproduction attempt, not proof that the mismatch is repaired. Exact
score stream, N8/N16 moments, ownership and provenance admission remains
mandatory; a further mismatch must be reported, not retried indefinitely.

**Terminal update:** `calibration_mistral7b_attempt3` failed before model load
because `runpy.run_path` did not add the sibling script directory. The old
adapter is retained byte-for-byte; `stream_calibrate_historical_v2.py` adds
only that path bootstrap. Run the CPU-only regression check with:

```bash
$PY "$SC/scripts/test_historical_runpy.py"
```

The test executes isolated runpy imports, reproduces the old failure, confirms
v2 succeeds, and never calls model/calibration main or initializes CUDA.
It is not score-identity acceptance. All calibration retries are exhausted;
**do not run attempt4** without new explicit authorization. The CPU-prepared
`calibration_repair_run_v2.py` pins the old launcher, repaired adapter and cadence
guard. `validate_parent_moments.py` verifies the authorization copy, launcher
lineage and adapter hash in addition to the unchanged scientific identity gates.
Alternatively provide the verified original full seed0 sequence-score shards.
The independent stored-map reproduction completed on guarded attempt2 and
passed both full-window N8/N16 audits; attempt1 was invalidated for monitor
gaps (see below). Do not rerun the completed control.

CPU-only checks (no GPU query):

```bash
$PY "$SC/scripts/test_calibration_repair_v2.py"
$PY "$SC/scripts/calibration_repair_run_v2.py" --inspect-only \
  --only-uuid GPU-9cec7336-5b30-3f86-35e7-06919156e7da
```

Conditional continuation, **not authorized or executed in this checkpoint**:
after a real additional-retry grant, provide a JSON record with exactly
`authorization_type=explicit_user_additional_retry`,
`scope=one_additional_mistral_seed0_calibration_attempt`,
`run_name=calibration_mistral7b_attempt4_authorized`, and `user_evidence`
referencing that actual grant. Never manufacture this record from a CPU fixture.
Then use the following entry (the named UUID is still subject to preflight):

```bash
$PY "$SC/scripts/calibration_repair_run_v2.py" \
  --authorization-record "$ACTUAL_USER_GRANT_JSON" \
  --only-uuid GPU-9cec7336-5b30-3f86-35e7-06919156e7da \
  --model mistral7b --name calibration_mistral7b_attempt4_authorized \
  --calibrate-stream --device-type a6000 --hub-cache "$RECOVERED_HUB"
```

`RECOVERED_HUB` must resolve to the verified recovered cache described above.
No fourth calibration run or authorization record has been created. The path
repair has CPU coverage only; historical score/map identity is still unresolved.

This does not authorize another Llama calibration or Qwen granularity attempt
past their two-repair limits. Full baseline, N8 and N16 ingestion/identity gates
remain mandatory before any reproduction is called successful.

`plans/mistral7b_reproduction_only_v1.json` is a CPU-validated fallback containing
only FourOverSix and the two already verified historical N8/N16 masks. After
the final calibration repair failed at import before model loading, this plan
was launched as `reproduction_mistral7b_attempt1` through the exclusive A6000
launcher, with the recovered hub and `--teacher instance`. Inspect its live
or terminal record before scheduling anything; do not duplicate it. This
independent stored-map reproduction cannot authorize
coarse maps, another calibration retry, or a Qwen attempt4. Baseline, N8 and
N16 admission remains unchanged.

Subsequently, that Mistral reproduction and both Llama/Mistral accuracy attempt1
runs were stopped for ownership-monitor gaps above 60 seconds; their original
launch records and partial outputs remain unchanged and scientifically invalid.
See `results/MONITOR_CADENCE_INCIDENT.json`. First repairs use
`scripts/cadence_launch.py --entry gpu_run_wait.py` or `--entry accuracy_run.py`
followed by the original arguments and fresh attempt2 names. The wrapper pins
the unchanged adapter and records its own hash in `cadence_guard.json`; it adds
only fail-closed query-duration/interval checks. Four CPU fixtures test this
policy, not GPU numerical equivalence. Do not resubmit an existing live waiter.
Qwen's never-launched first repair was upgraded to the same guard under the
shared lease lock; `waiting_guard_upgrade.json` proves unchanged scientific
arguments and zero additional GPU attempts. Its new name is
`secondary_accuracy_qwen4b_attempt2_guarded`.

That Qwen repair completed and passed `accuracy_new_results.py`; do not rerun
it. Refresh the CPU-only, raw-array-checked point table after new admissions:

```bash
$PY "$SC/scripts/accuracy_table.py"
$PY "$SC/scripts/accuracy_plot.py"
```

Llama's never-launched guarded waiter was reassigned to GPU1 as
`secondary_accuracy_llama8b_attempt2_reassigned`; the original run retains
`guarded_waiting_reassignment.json`. Its scientific arguments/retry budget
are unchanged. A foreign process entered GPU1 after Qwen completed, and the
new preflight rejected launch. Inspect current state rather than duplicating
either waiting job.

Later terminal update: Llama's reassigned attempt2 obtained clean GPU1 preflight,
then failed the load-model phase check when a foreign PID entered. It is invalid;
no partial outcomes are admitted. The final permitted repair is now
`secondary_accuracy_llama8b_attempt3` on GPU0, using the same cadence wrapper,
accuracy plan and parameters. Inspect its live state; do not duplicate or create
attempt4. Mistral attempt2 remains separate on GPU3.

Llama attempt3 subsequently became invalid when a foreign process entered GPU0
after clean preflight. Its two-repair budget is exhausted; **do not launch a
fourth secondary-accuracy attempt** without explicit new authority and a
coordinated exclusive A6000 window. Mistral attempt2 has independently started
on GPU3. Read `results/GPU_RESOURCE_BLOCKERS.json` and live status before action.

For a later availability-only move of a **never-launched guarded waiter**, use
`scripts/reassign_guarded_waiting.py --inspect-only` with its exact PID,
`--start-ticks`, existing `--run`, fresh `--new-run`, and verified `--to-uuid`.
Only after the read-only snapshot passes may the same invocation omit
`--inspect-only`. The shared lease lock and repeated PID/UID/start-time checks
reject any running/leased/child-bearing process; three CPU tests exercise
these guards. No move has been required merely by preparing this helper.
It cannot extend scientific retry limits, reserve a GPU against other users,
or change any scientific argument. Every actual move retains
`guarded_waiting_reassignment.json` for independent comparison.

An additional availability-only recheck of the previously unstable A6000 failed
on a foreign compute PID (`results/availability_gpu0_recheck_01.json`). No GPU
work was launched by that check; it is not an evaluation retry. Existing
restricted-UUID scheduling therefore remains in effect.

The later GPU inventory timeout incident is recorded in
`results/HOST_QUERY_INCIDENT.json`. Do not resume stale waiting PIDs: Mistral
draw4 attempt2 and calibration attempt1 are invalid, and Qwen draw3 attempt2
exited before launch. Retain their records. After query recovery, schedule
only the remaining authorized repairs under unique attempt names, with fresh
preflight and uninterrupted monitoring. The continuation controller does not
automatically repair failed cells; check actual live processes and terminal
records before any manual launch.

`gpu_run_wait.py` is the new, separately hash-pinned restricted-UUID adapter
for future repair attempts. Its only extra behavior is to append prelaunch
inventory-query errors to `inventory_query_errors.jsonl` and wait without
allocating a lease or model. Three CPU tests verify the narrow transformation,
error logging and propagation of programming errors. In-run, phase-boundary
and postflight failures still invalidate attempts. Existing launchers were
not edited or replaced. This repair does not reset any attempt's retry budget.
