# Cluster research workflow

Start with `TASK_REORDER_HANDOFF.md` and preserve the user's research and spending constraints.
Heavy CPU/GPU work must run through Slurm on H200 compute nodes, using `gov113008`;
never run model workloads or heavy searches on the login node. Reuse completed controls.

On2026-09-20 the user accepted the measured both-axis result as good enough and
requested an explanation and publication. This supersedes the earlier instruction
to continue until90% recovery. The90% target was NOT achieved; retain that fact in
TARGET_90_PERCENT.json. Further GPU experimentation is stopped unless requested.
For future work, use at most4GPUs concurrently and retain all fresh-data gates.

Every submitted job or job array must have a monitor that reports completion or
failure back to the current session, so work can continue without another user prompt.
Run `scripts/watch_mixfp4_jobs.py` with the explicit job IDs (expanded array task
IDs), `--notify-thread "$CODEX_THREAD_ID"`, and an absolute `--codex-bin` path.
Use `setsid --fork` for a detached watcher, a unique output directory per batch,
and a 60-second interval. Check its PID and status after starting it. Register
successor jobs with another monitored batch before ending the turn.

The watcher queues same-session messages through `codex queue`; `notifications.json`
deduplicates terminal jobs across restarts. Queue acceptance is not proof of agent
receipt. On receipt, read the reports, record that receipt, and continue only the
next justified action. Keep quality gates and resource limits in force; a
notification does not authorize repeating jobs. No GPU allocation is needed to monitor.

If session notifications fail, keep the turn active with an attached monitor and
tool waits while jobs remain, and handle completion directly. Never describe a
detached file logger alone as a mechanism that will wake the agent. Report notification
failures explicitly rather than silently leaving completed jobs unhandled.

After positive-control diagnostic404650, the NEW prospective CE-primary branch uses pooledCE mean+2SE<0 against raw256 and matched identity on fresh64documents, with nonpositive domainCEmeans versusraw. KL is reported diagnostically; per-tilejointCE/KLk3 remains. Prior strict-gate failures remain failures. Never promote a candidate without its applicable frozen fresh-data gate. See ce_target_combinations/plan.json and scripts/ce_confirmation_gate.py.
