"""P80 final, fully data-derived analysis and artifact/GPU-policy audit.

This module consumes only completed run artifacts.  It never evaluates a model and
never edits an earlier artifact.  P81 performs the final redacted bundle build and
publishes the authoritative post-bundle matrix coverage.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from campaign import mapio
from campaign import records
from campaign import runtime
from campaign.extension_verify_bundle import amendment_chain_errors


CR = Path(os.environ["CAMPAIGN_ROOT"]).resolve()
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"]).resolve()
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PARENT_MANIFEST = "432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee"
MATRIX = CR / "handoff/agent_handoff/EXPERIMENT_MATRIX.csv"
ALLOWED_GPU_NAMES = {"NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation"}
PPL_ANALYSIS_NAMES = {
    "P10_P11_BREADTH_ANALYSIS.json", "P20_DENSITY_CONTROL_ANALYSIS.json",
    "P21_BREADTH_ANALYSIS.json", "P30_FIVE_DRAW_ANALYSIS.json",
    "P31_CALIBRATION_SIZE_ANALYSIS.json", "P32_CALIBRATION_SIZE_ANALYSIS.json",
    "P40_LOWER_K_ANALYSIS.json", "P41_DRAW_AGGREGATION_ANALYSIS.json",
    "P42_BREADTH_ANALYSIS.json", "P50_SELECTOR_ANALYSIS.json",
    "P52_BREADTH_ANALYSIS.json", "P60_ANALYSIS.json", "P61_ANALYSIS.json",
    "P63_COMBINATION_ANALYSIS.json", "P71_HELDOUT_PPL_ANALYSIS.json",
    "P74_PG19_ANALYSIS.json",
}
DIMENSION_KEYS = (
    "model", "family", "corpus", "domain", "context", "draw", "resolution",
    "scope", "method", "policy", "comparator", "selector", "candidate",
    "aggregation", "form", "task", "filter", "stage", "k", "arm",
)


def load(path: Path):
    return json.loads(Path(path).read_text())


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def logical(path: Path) -> str:
    path = Path(path)
    try:
        return path.absolute().relative_to(CR.absolute()).as_posix()
    except ValueError:
        try:
            return "parent:" + path.absolute().relative_to(PR.absolute()).as_posix()
        except ValueError:
            resolved = path.resolve()
            try:
                return resolved.relative_to(CR.resolve()).as_posix()
            except ValueError:
                try:
                    return "parent:" + resolved.relative_to(PR.resolve()).as_posix()
                except ValueError:
                    return "external:" + path.name


def attempts(job: str):
    rows = []
    for run in (CR / "runs").glob(f"{job}_attempt*"):
        try:
            n = int(run.name.rsplit("_attempt", 1)[1])
            launch = load(run / "launch_record.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        rows.append((n, run, launch))
    return sorted(rows)


def latest_complete(job: str) -> Path:
    accepted = []
    for n, run, launch in attempts(job):
        try:
            status = load(run / "job_status.json")
            valid = load(run / "run_record_validation.json")
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if launch.get("status") == status.get("status") == "complete" and valid.get("valid"):
            accepted.append((n, run))
    if not accepted:
        raise FileNotFoundError(job)
    return accepted[-1][1]


def artifact(job: str, relative: str):
    path = latest_complete(job) / relative
    value = load(path)
    return path, value


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accepted_run_dirs():
    result = []
    for run in sorted((CR / "runs").iterdir()):
        if not run.is_dir() or not (run / "launch_record.json").is_file():
            continue
        try:
            launch = load(run / "launch_record.json")
            status = load(run / "job_status.json")
            valid = load(run / "run_record_validation.json")
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if launch.get("status") == status.get("status") == "complete" and valid.get("valid"):
            result.append(run)
    return result


def collect_raw_ppl():
    rows, sources = [], {}
    for run in accepted_run_dirs():
        path = run / "ppl/ppl_report.json"
        if not path.is_file():
            continue
        report = load(path)
        if report.get("status") != "complete":
            continue
        launch = load(run / "launch_record.json")
        installs = {entry.get("name"): entry for entry in report.get("installs", [])}
        for policy, domains in report.get("evaluation", {}).items():
            install = installs.get(policy, {})
            for domain, result in domains.items():
                if domain == "seconds" or not isinstance(result, dict) or "ppl" not in result:
                    continue
                block = install.get("type_block")
                selected_tiles = install.get("selected_tiles")
                rows.append({
                    "record_type": "raw_ppl", "matrix_id": launch.get("matrix_id"),
                    "run_id": run.name, "model": report.get("model"), "policy": policy,
                    "corpus": domain, "raw_ppl": result.get("ppl"),
                    "mean_nll": result.get("mean_nll"), "tokens": result.get("tokens"),
                    "status": "complete", "map_sha256": install.get("map_sha256"),
                    "selected_tiles": selected_tiles,
                    "selected_weights": (selected_tiles * block[0] * block[1]
                                         if selected_tiles is not None and block else None),
                    "type_block": json.dumps(block),
                    "activation_policy": install.get("activation"),
                    "source_path": logical(path), "source_sha256": runtime.sha256_file(path),
                })
        sources[logical(path)] = runtime.sha256_file(path)

    # Bring the parent/carry-over raw cells into the same long table.  These files
    # were themselves hash-audited before protocol lock and remain clearly labeled.
    primary = CR / "handoff/agent_handoff/EXISTING_PRIMARY_PPL.csv"
    with primary.open(newline="") as handle:
        for source in csv.DictReader(handle):
            for key, policy in (("bf16_ppl", "bf16"), ("nvfp4_ppl", "nvfp4"),
                                ("four_over_six_ppl", "four_over_six"),
                                ("razer_native_rows_ppl", "razer_native_rows"),
                                ("n8_k3_ppl", "n8_k3"), ("n16_k3_ppl", "n16_k3")):
                value = source.get(key)
                rows.append({"record_type": "raw_ppl", "matrix_id": "P12_EXISTING_K2_PROVENANCE",
                             "run_id": "parent_handoff_table", "model": source["model"],
                             "policy": policy, "corpus": source["corpus"].lower(),
                             "raw_ppl": None if value in (None, "", "NA") else float(value),
                             "mean_nll": None if value in (None, "", "NA") else math.log(float(value)),
                             "status": "not_run" if value in (None, "", "NA") else "complete_parent_reuse",
                             "source_path": logical(primary), "source_sha256": runtime.sha256_file(primary)})
    sweep = CR / "handoff/agent_handoff/EXISTING_K_SWEEP.csv"
    with sweep.open(newline="") as handle:
        for source in csv.DictReader(handle):
            policy = source["tile_shape"].replace("K64", "").lower() + "_k" + source["k"]
            rows.append({"record_type": "raw_ppl", "matrix_id": "P12_EXISTING_K2_PROVENANCE",
                         "run_id": "parent_handoff_table", "model": source["model"],
                         "policy": policy, "corpus": source["corpus"].lower(),
                         "raw_ppl": float(source["method_ppl"]),
                         "mean_nll": math.log(float(source["method_ppl"])),
                         "status": "complete_parent_reuse", "selected_tiles": int(source["selected_tiles"]),
                         "source_path": logical(sweep), "source_sha256": runtime.sha256_file(sweep)})
    sources[logical(primary)] = runtime.sha256_file(primary)
    sources[logical(sweep)] = runtime.sha256_file(sweep)

    # P10/P11's table carries explicit unsupported/not-run baseline cells as
    # required by the fair-comparison protocol.
    try:
        p10_path, p10 = artifact("P10_P11_analysis", "analysis_breadth/P10_P11_BREADTH_ANALYSIS.json")
        for source in p10.get("raw_table", []):
            rows.append({"record_type": "raw_ppl", "matrix_id": "P10_K2_BREADTH+P11_BASELINE_COMPLETION",
                         "run_id": source.get("run_id"), "model": source.get("model"),
                         "policy": source.get("method"), "corpus": source.get("corpus"),
                         "raw_ppl": source.get("raw_ppl"), "mean_nll": source.get("mean_nll"),
                         "status": source.get("status"), "reason": source.get("reason"),
                         "source_path": logical(p10_path), "source_sha256": runtime.sha256_file(p10_path)})
        sources[logical(p10_path)] = runtime.sha256_file(p10_path)
    except FileNotFoundError:
        pass
    return rows, sources


def walk_contrasts(value, source_path: Path, pointer="", inherited=None):
    inherited = dict(inherited or {})
    rows = []
    if isinstance(value, dict):
        context = dict(inherited)
        for key in DIMENSION_KEYS:
            if key in value and isinstance(value[key], (str, int, float, bool)):
                context[key] = value[key]
        ci = value.get("ci95")
        estimate_key = ("estimate" if isinstance(value.get("estimate"), (int, float))
                        else "diff" if isinstance(value.get("diff"), (int, float)) else None)
        if estimate_key and isinstance(ci, list) and len(ci) == 2:
            is_ppl = estimate_key == "estimate"
            row = {"record_type": "paired_dlogppl" if is_ppl else "paired_accuracy", **context,
                   "effect": value[estimate_key],
                   "dlogppl": value[estimate_key] if is_ppl else None,
                   "accuracy_difference": value[estimate_key] if not is_ppl else None,
                   "ci95_low": ci[0], "ci95_high": ci[1],
                   "p_raw": value.get(
                       "p_raw",
                       value.get(
                           "p_value",
                           value.get(
                               "p_two_sided",
                               value.get("p_noninferiority", value.get("mcnemar_p")),
                           ),
                       ),
                   ),
                   "p_adjusted": value.get(
                       "p_adjusted",
                       value.get(
                           "holm_adjusted_p", value.get("p_holm", value.get("mcnemar_p_holm"))
                       ),
                   ),
                   "raw_ppl_method": value.get("raw_ppl_method"),
                   "raw_ppl_comparator": value.get("raw_ppl_comparator"),
                   "clusters": value.get("clusters"), "windows": value.get("windows"),
                   "stream_label": value.get("stream_label"),
                   "source_path": logical(source_path),
                   "source_sha256": runtime.sha256_file(source_path),
                   "json_pointer": pointer or "/"}
            rows.append(row)
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            rows.extend(walk_contrasts(item, source_path, pointer + "/" + escaped, context))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(walk_contrasts(item, source_path, pointer + f"/{index}", inherited))
    return rows


def collect_contrasts():
    rows, sources = [], {}
    for run in accepted_run_dirs():
        for path in sorted(run.glob("analysis_*/*.json")):
            if path.name not in PPL_ANALYSIS_NAMES:
                continue
            value = load(path)
            found = walk_contrasts(value, path)
            for row in found:
                row["matrix_id"] = load(run / "launch_record.json").get("matrix_id")
                row["run_id"] = run.name
            rows.extend(found)
            sources[logical(path)] = runtime.sha256_file(path)
    return rows, sources


def find_analysis(name: str):
    matches = []
    for run in accepted_run_dirs():
        for path in run.glob(f"analysis_*/{name}"):
            matches.append((run.name, path))
    if not matches:
        return None, None
    path = sorted(matches)[-1][1]
    return path, load(path)


def proven_post_exit_processes(launch: dict) -> set[tuple[int, str]] | None:
    """Return the exactly classified post-exit processes, or None if proof fails.

    A host after-check can observe a foreign process that started only after the
    container exited.  That is not co-tenancy with the completed attempt, but we
    accept the exception only when the launcher retained an unambiguous start-time
    classification for every process and no possibly overlapping process exists.
    """
    if not launch.get("host_after_post_exit_only"):
        return None
    if launch.get("late_foreign_processes"):
        return None
    rows = launch.get("post_exit_foreign_processes") or []
    if not rows:
        return None
    expected = set(launch.get("leased_uuids") or ())
    proven = set()
    for row in rows:
        pid, uuid = row.get("pid"), row.get("gpu_uuid")
        if (not isinstance(pid, int) or pid <= 0 or uuid not in expected
                or row.get("classification") != "post-exit (started after this run finished)"
                or not isinstance(row.get("seconds_after_container_exit"), (int, float))
                or row["seconds_after_container_exit"] <= 1.0):
            return None
        proven.add((pid, uuid))
    return proven


def audit_raw_gpu_evidence(run: Path, launch: dict, status: dict) -> dict:
    """Verify every referenced owner/PID/UUID snapshot, not only summary flags."""
    expected = tuple(launch.get("leased_uuids") or ())
    expected_set = set(expected)
    post_exit_processes = proven_post_exit_processes(launch)
    references = []
    for phase in ("host_preflight_before", "host_preflight_after"):
        if launch.get(phase):
            references.append((phase, launch[phase]))
    for item in status.get("preflight_records", []):
        references.append((item.get("phase") or "unknown", item))

    errors, owners, owner_uids, pids, phases = [], set(), set(), set(), []
    observed_models = set()
    for phase, reference in references:
        raw_path = reference.get("path")
        path = Path(raw_path) if raw_path else None
        phase_errors = []
        if path is None or not path.is_file():
            errors.append(f"{phase}: missing referenced preflight record")
            continue
        actual_sha = runtime.sha256_file(path)
        if actual_sha != reference.get("sha256"):
            phase_errors.append("reference SHA-256 mismatch")
        try:
            value = load(path)
        except Exception as exc:
            errors.append(f"{phase}: invalid JSON: {exc!r}")
            continue
        if value.get("schema") != "gpu_preflight/v1":
            phase_errors.append("wrong preflight schema")
        post_exit_exception = phase == "host_preflight_after" and post_exit_processes is not None
        if not value.get("passed") and not post_exit_exception:
            phase_errors.append("preflight did not pass")
        uid, user = value.get("uid"), value.get("user")
        if uid is None or not user:
            phase_errors.append("missing audit owner")
        else:
            owner_uids.add(uid)
            owners.add(user)
        visible = set((value.get("visible") or {}).get("resolved_uuids") or ())
        if visible != expected_set:
            phase_errors.append("visible UUIDs differ from lease")
        gpu_by_uuid = {gpu.get("uuid"): gpu for gpu in value.get("gpus", [])}
        for uuid in expected:
            gpu = gpu_by_uuid.get(uuid)
            if not gpu:
                phase_errors.append(f"leased UUID absent from snapshot: {uuid}")
            elif gpu.get("name") not in ALLOWED_GPU_NAMES:
                phase_errors.append(f"disallowed GPU model for {uuid}")
            else:
                observed_models.add(gpu.get("name"))
        raw_processes = set()
        for process in value.get("compute_processes", []):
            process_uuid = process.get("gpu_uuid")
            process_uid = process.get("owner_uid", process.get("uid"))
            process_user = process.get("owner", process.get("user"))
            pid = process.get("pid")
            if process_uuid not in expected_set:
                phase_errors.append("process reported outside leased UUIDs")
            if process_uid != uid or process_user != user:
                if not (post_exit_exception and (pid, process_uuid) in post_exit_processes):
                    phase_errors.append("foreign or unknown process owner")
            if phase not in ("host_preflight_before", "host_preflight_after", "before", "after") \
                    and process.get("belongs_to_current_job") is not True:
                phase_errors.append("phase process is not proven to belong to this job")
            if not isinstance(pid, int) or pid <= 0:
                phase_errors.append("invalid process PID")
            else:
                pids.add(pid)
                raw_processes.add((pid, process_uuid))
        if post_exit_exception and raw_processes != post_exit_processes:
            phase_errors.append("post-exit process classification differs from raw after-check")
        if phase_errors:
            errors.extend(f"{phase}: {message}" for message in phase_errors)
        phases.append({"phase": phase, "path": logical(path), "sha256": actual_sha,
                       "passed": not phase_errors, "process_count": len(value.get("compute_processes", []))})

    monitor = run / "gpu_monitor.jsonl"
    monitor_samples = 0
    monitor_max_gap = 0.0
    previous_stamp = None
    if not monitor.is_file():
        errors.append("missing ownership sidecar")
    else:
        for line_number, line in enumerate(monitor.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                sample = load_json_string(line)
                stamp = dt.datetime.fromisoformat(sample["utc"].replace("Z", "+00:00"))
            except Exception as exc:
                errors.append(f"sidecar line {line_number}: invalid JSON/timestamp: {exc!r}")
                continue
            monitor_samples += 1
            if previous_stamp is not None:
                monitor_max_gap = max(monitor_max_gap, (stamp - previous_stamp).total_seconds())
            previous_stamp = stamp
            if set((sample.get("gpus") or {}).keys()) != expected_set:
                errors.append(f"sidecar line {line_number}: GPU UUIDs differ from lease")
            for process in sample.get("processes", []):
                if process.get("gpu_uuid") not in expected_set:
                    errors.append(f"sidecar line {line_number}: process outside lease")
                if process.get("uid") not in owner_uids or process.get("owner") not in owners:
                    errors.append(f"sidecar line {line_number}: foreign or unknown owner")
                if process.get("in_container") is not True:
                    errors.append(f"sidecar line {line_number}: process not proven in container")
                pid = process.get("pid")
                if not isinstance(pid, int) or pid <= 0:
                    errors.append(f"sidecar line {line_number}: invalid PID")
                else:
                    pids.add(pid)
    reported_samples = int((launch.get("sidecar") or {}).get("samples") or 0)
    if monitor_samples != reported_samples:
        errors.append(f"sidecar sample count mismatch: file={monitor_samples}, launch={reported_samples}")
    if monitor_samples < 1:
        errors.append("sidecar has no samples")
    if monitor_max_gap > 60.0:
        errors.append(f"sidecar interval exceeded 60 seconds: {monitor_max_gap:.6f}")
    if len(owner_uids) != 1 or len(owners) != 1:
        errors.append("preflight owner identity is inconsistent")
    if len(observed_models) != 1:
        errors.append("preflight evidence is missing or mixes GPU models")
    return {
        "passed": not errors, "errors": errors, "leased_device_uuids": list(expected),
        "observed_device_models": sorted(observed_models), "owner_uids": sorted(owner_uids),
        "owners": sorted(owners), "observed_process_pids": sorted(pids),
        "preflight_record_count": len(references), "preflight_records": phases,
        "sidecar_path": logical(monitor), "sidecar_sha256": runtime.sha256_file(monitor) if monitor.is_file() else None,
        "sidecar_samples": monitor_samples, "sidecar_max_interval_seconds": monitor_max_gap,
    }


def table_bundle(names: list[str]):
    items = {}
    flat = []
    for name in names:
        path, value = find_analysis(name)
        if path is None:
            continue
        items[name] = {"path": logical(path), "sha256": runtime.sha256_file(path), "data": value}
        flat.extend(walk_contrasts(value, path))
        # Preserve draw-level intervals that intentionally separate between-draw
        # and within-evaluation uncertainty and therefore do not use `ci95`.
        def walk_special(node, pointer=""):
            if isinstance(node, dict):
                if "estimate_draw_mean" in node and "hierarchical_ci95" in node:
                    flat.append({"record_type": "draw_summary", "dlogppl": node["estimate_draw_mean"],
                                 "ci95_low": node["hierarchical_ci95"][0],
                                 "ci95_high": node["hierarchical_ci95"][1],
                                 "between_draw_variance": node.get("between_draw_variance"),
                                 "within_evaluation_variance_mean": node.get(
                                     "mean_within_evaluation_bootstrap_variance",
                                     node.get("within_evaluation_variance_mean")),
                                 **{key: node[key] for key in DIMENSION_KEYS if key in node},
                                 "stream_label": node.get("stream_label"),
                                 "source_path": logical(path), "source_sha256": runtime.sha256_file(path),
                                 "json_pointer": pointer or "/"})
                for k, v in node.items():
                    walk_special(v, pointer + "/" + str(k).replace("/", "~1"))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk_special(v, pointer + f"/{i}")
        walk_special(value)
    return items, flat


def audit_runs():
    registry_events = []
    for line_no, line in enumerate((CR / "registry/attempts.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            registry_events.append(load_json_string(line))
        except Exception as exc:
            raise RuntimeError(f"registry JSON error line {line_no}: {exc}")
    event_names = defaultdict(list)
    for event in registry_events:
        if event.get("run_id"):
            event_names[event["run_id"]].append(event.get("event"))

    run_rows, failures, manifest_errors, accepted_gpu, current_problems = [], [], [], [], []
    status_counts = Counter()
    gpu_hours_device, gpu_hours_model, gpu_hours_matrix = defaultdict(float), defaultdict(float), defaultdict(float)
    manifest_entries_verified = manifest_bytes_verified = 0
    newest_complete = {}
    all_dirs = []
    for run in sorted((CR / "runs").iterdir()):
        if not run.is_dir() or run.name == runtime.run_dir.name or not (run / "launch_record.json").is_file():
            continue
        launch = load(run / "launch_record.json")
        all_dirs.append((run, launch))
        job, _, anum = run.name.rpartition("_attempt")
        if launch.get("status") == "complete" and anum.isdigit():
            newest_complete[job] = max(newest_complete.get(job, 0), int(anum))
    for run, launch in all_dirs:
        status = launch.get("status", "unknown")
        job_status = load(run / "job_status.json") if (run / "job_status.json").is_file() else {}
        status_counts[status] += 1
        job, _, anum = run.name.rpartition("_attempt")
        superseded = bool(anum.isdigit() and newest_complete.get(job, 0) > int(anum))
        problems = []
        validation = None
        if (run / "run_record_validation.json").is_file():
            validation = load(run / "run_record_validation.json")
            if status == "complete" and not validation.get("valid"):
                problems.append("complete attempt has invalid run_record")
        elif status == "complete":
            problems.append("complete attempt lacks run_record_validation.json")
        if (run / "run_record.json").is_file():
            record = load(run / "run_record.json")
            schema_errors = records.validate_record(record)
            if status == "complete" and schema_errors:
                problems.append(f"run_record schema errors: {schema_errors[:3]}")
            sums = run / "SHA256SUMS_run.txt"
            if not sums.is_file():
                problems.append("run_record exists but SHA256SUMS_run.txt is absent")
            else:
                bad = []
                for line in sums.read_text().splitlines():
                    expected, rel = line.split("  ", 1)
                    target = run / rel
                    if not target.is_file():
                        bad.append(f"missing:{rel}")
                    else:
                        size = target.stat().st_size
                        actual = runtime.sha256_file(target)
                        manifest_entries_verified += 1
                        manifest_bytes_verified += size
                        if actual != expected:
                            bad.append(f"changed:{rel}")
                if bad:
                    manifest_errors.append({"run_id": run.name, "errors": bad})
                    problems.append(f"{len(bad)} run-manifest hash errors")
                if record.get("artifacts", {}).get("sha256_manifest") != runtime.sha256_file(sums):
                    problems.append("run-manifest digest differs from run_record")
            if status == "complete" and validation and validation.get("valid"):
                for policy in record.get("policies", []):
                    if policy.get("map_path") and policy.get("map_sha256"):
                        try:
                            mapio.read_map(policy["map_path"], policy["map_sha256"])
                        except Exception as exc:
                            problems.append(f"map {policy.get('name')}: {exc!r}")
                g = record.get("gpu_policy", {})
                names = g.get("device_models", [])
                if names:
                    raw_gpu_evidence = audit_raw_gpu_evidence(run, launch, job_status)
                    row = {"run_id": run.name, "matrix_id": launch.get("matrix_id"),
                           "device_models": names, "device_uuids": g.get("device_uuids", []),
                           "gpu_hours": float(launch.get("gpu_hours") or 0),
                           "preflight_passed": g.get("preflight_passed"),
                           "during_checks_passed": g.get("during_checks_passed"),
                           "postflight_passed": g.get("postflight_passed"),
                           "foreign_compute_processes_absent": g.get("foreign_compute_processes_absent"),
                           "sidecar_samples": (launch.get("sidecar") or {}).get("samples"),
                           "host_preflight_before": (launch.get("host_preflight_before") or {}).get("passed"),
                           "host_preflight_after": ((launch.get("host_preflight_after") or {}).get("passed")
                                                    or proven_post_exit_processes(launch) is not None),
                           "invalid_gpu_cotenancy": launch.get("invalid_gpu_cotenancy"),
                           "raw_owner_pid_uuid_evidence": raw_gpu_evidence}
                    row["homogeneous"] = len(set(names)) == 1
                    row["allowed_device"] = set(names) <= ALLOWED_GPU_NAMES
                    row["complete_evidence"] = all(row[k] is True for k in (
                        "preflight_passed", "during_checks_passed", "postflight_passed",
                        "foreign_compute_processes_absent", "host_preflight_before", "host_preflight_after"))
                    row["passed"] = (row["homogeneous"] and row["allowed_device"]
                                     and row["complete_evidence"] and not row["invalid_gpu_cotenancy"]
                                     and len(row["device_uuids"]) <= 3 and (row["sidecar_samples"] or 0) > 0
                                     and raw_gpu_evidence["passed"])
                    if not row["passed"]:
                        problems.append("accepted GPU attempt fails policy evidence audit")
                    accepted_gpu.append(row)
        if "created" not in event_names[run.name]:
            problems.append("registry lacks created event")
        if status in ("complete", "failed", "invalid") and "finished" not in event_names[run.name]:
            problems.append("registry lacks finished event")
        gpu_hours = float(launch.get("gpu_hours") or 0)
        if gpu_hours:
            model_name = "unknown"
            if (run / "run_record.json").is_file():
                model_name = load(run / "run_record.json").get("model", {}).get("name") or "unknown"
            device = (launch.get("requested") or {}).get("model", "unknown")
            gpu_hours_device[device] += gpu_hours
            gpu_hours_model[model_name] += gpu_hours
            gpu_hours_matrix[launch.get("matrix_id") or "unknown"] += gpu_hours
        if status != "complete":
            reason = launch.get("reason") or launch.get("reasons") or (launch.get("sidecar") or {}).get("invalid_reasons")
            if not reason and (run / "job_status.json").is_file():
                reason = load(run / "job_status.json").get("error")
            failures.append({"run_id": run.name, "matrix_id": launch.get("matrix_id"),
                             "status": status, "exit_code": launch.get("exit_code"),
                             "reason": reason, "superseded_by_later_complete_attempt": superseded,
                             "gpu_hours": gpu_hours})
        run_row = {"run_id": run.name, "matrix_id": launch.get("matrix_id"), "status": status,
                   "superseded_by_later_complete_attempt": superseded, "problems": problems}
        run_rows.append(run_row)
        if problems and not superseded:
            current_problems.append(run_row)

    concurrency = concurrency_audit(registry_events)
    maps = []
    map_errors = []
    for path in sorted((CR / "runs").rglob("*.mixfp4map")):
        try:
            header, _, digest = mapio.read_map(path)
            maps.append({"path": logical(path), "sha256": digest,
                         "type_block": header.get("type_block"),
                         "selected_tiles": header.get("totals", {}).get("selected_tiles")})
        except Exception as exc:
            map_errors.append({"path": logical(path), "error": repr(exc)})
    amendment_path = CR / "provenance/PROTOCOL_EXTENSION_AMENDMENTS.jsonl"
    amendment_rows = [load_json_string(line) for line in amendment_path.read_text().splitlines()
                      if line.strip()]
    amendment_errors = amendment_chain_errors(amendment_rows, PROTOCOL)
    policy_pass = (not current_problems and not manifest_errors and not map_errors
                   and not amendment_errors
                   and all(row["passed"] for row in accepted_gpu)
                   and concurrency["passed"])
    return {
        "schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
        "attempt_count": len(run_rows), "run_directories_excluding_current": len(run_rows),
        "by_status": dict(sorted(status_counts.items())), "runs": run_rows,
        "current_runs_with_problems": current_problems,
        "failures_oom_invalid_and_crashes": failures,
        "run_manifest_verification": {"entries": manifest_entries_verified,
                                      "bytes": manifest_bytes_verified,
                                      "errors": manifest_errors, "passed": not manifest_errors},
        "map_verification": {"maps": len(maps), "errors": map_errors, "passed": not map_errors},
        "protocol_amendment_chain": {
            "path": logical(amendment_path), "entries": len(amendment_rows),
            "tail_entry_sha256": amendment_rows[-1].get("entry_sha256") if amendment_rows else None,
            "errors": amendment_errors, "passed": not amendment_errors,
        },
        "accepted_gpu_attempts": accepted_gpu, "gpu_concurrency": concurrency,
        "gpu_hours": {"total": sum(gpu_hours_device.values()),
                      "by_device_family": dict(sorted(gpu_hours_device.items())),
                      "by_model": dict(sorted(gpu_hours_model.items())),
                      "by_matrix": dict(sorted(gpu_hours_matrix.items()))},
        "zero_policy_violations_among_accepted_runs": policy_pass,
        "passed": policy_pass,
    }


def load_json_string(text):
    return json.loads(text)


def concurrency_audit(events):
    gpu_runs = {event.get("run_id") for event in events
                if event.get("event") == "started" and event.get("uuids")}
    ordered = []
    for position, event in enumerate(events):
        if (event.get("event") not in ("started", "finished")
                or event.get("run_id") not in gpu_runs):
            continue
        stamp = event.get("logged_utc") or ""
        # A finish sorts before a start at an identical timestamp.
        order = 0 if event["event"] == "finished" else 1
        ordered.append((stamp, order, position, event))
    active, maximum, conflicts, timeline = {}, 0, [], []
    for stamp, _, _, event in sorted(ordered):
        run_id = event["run_id"]
        if event["event"] == "finished":
            active.pop(run_id, None)
        else:
            uuids = tuple(event.get("uuids") or ())
            used = {u: r for r, us in active.items() for u in us}
            overlap = sorted(set(uuids) & set(used))
            if overlap:
                conflicts.append({"timestamp_utc": stamp, "run_id": run_id,
                                  "uuids": overlap, "other_runs": sorted({used[x] for x in overlap})})
            active[run_id] = uuids
        count = len({u for us in active.values() for u in us})
        maximum = max(maximum, count)
        if event.get("uuids"):
            timeline.append({"timestamp_utc": stamp, "event": event["event"],
                             "run_id": run_id, "campaign_gpu_count": count})
    return {"max_concurrent_campaign_gpus": maximum, "limit": 3,
            "same_uuid_overlap_conflicts": conflicts,
            "events_with_gpu_identity": len(timeline), "timeline": timeline,
            "passed": maximum <= 3 and not conflicts and not active,
            "active_at_audit_end": {k: list(v) for k, v in active.items()}}


def build_artifact_manifest(out: Path, audit):
    lines = []
    roots = ("freeze", "provenance", "registry", "plans", "env",
             "handoff/agent_handoff", "queue/jobs", "queue/release")
    for root in roots:
        base = CR / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                lines.append(f"{runtime.sha256_file(path)}  {path.relative_to(CR).as_posix()}")
    for run in sorted((CR / "runs").iterdir()):
        sums = run / "SHA256SUMS_run.txt"
        if run.is_dir() and sums.is_file() and run.name != runtime.run_dir.name:
            lines.append(f"{runtime.sha256_file(sums)}  {sums.relative_to(CR).as_posix()}")
    path = out / "ARTIFACT_MANIFEST.sha256"
    path.write_text("\n".join(lines) + "\n")
    audit["artifact_manifest"] = {"path": logical(path), "entries": len(lines),
                                  "sha256": runtime.sha256_file(path),
                                  "scope": "stable campaign metadata plus each immutable run-level manifest"}
    return path


TERMINAL_JOBS = {
    "P00_INPUT_AUDIT": "P00_input_audit",
    "P01_ANCHOR_REPRO": "P01_anchor_acceptance",
    "P10_K2_BREADTH": "P10_P11_analysis", "P11_BASELINE_COMPLETION": "P10_P11_analysis",
    "P12_EXISTING_K2_PROVENANCE": "P12_existing_k2_provenance",
    "P20_K2_DENSITY_DEV": "P20_analysis", "P21_K2_DENSITY_BREADTH": "P21_terminal",
    "P30_K2_DRAWS": "P30_analysis", "P31_CAL_SIZE_128": "P31_analysis",
    "P32_CAL_SIZE_256": "P32_terminal", "P40_LOWER_K_SEARCH": "P40_analysis",
    "P41_DRAW_AGG_DEV": "P41_analysis", "P42_DRAW_AGG_BREADTH": "P42_terminal",
    "P50_SELECTOR_DEV": "P50_analysis", "P51_TILE_FIDELITY": "P51_analysis",
    "P52_SELECTOR_BREADTH": "P52_terminal", "P60_WEIGHT_SCALE": "P60_analysis",
    "P61_ACT_SCALE_CLIP": "P61_analysis", "P62_K64_LOCAL_TRANSFORM": "P62_gate",
    "P70_WINNER_FREEZE": "P70_winner_freeze", "P71_HELDOUT_PPL": "P71_analysis",
    "P75_PORTABILITY": "P75_analysis",
}


def matrix_coverage_prebundle():
    rows = []
    downstream = load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")
    p32 = load(CR / "provenance/P32_GATE_RESOLUTION.json")
    breadth = {
        matrix_id: load(CR / f"provenance/{stage}_GATE_RESOLUTION.json")
        for matrix_id, stage in (
            ("P21_K2_DENSITY_BREADTH", "P21"),
            ("P42_DRAW_AGG_BREADTH", "P42"),
            ("P52_SELECTOR_BREADTH", "P52"),
        )
    }
    with MATRIX.open(newline="") as handle:
        for spec in csv.DictReader(handle):
            matrix_id = spec["matrix_id"]
            status, evidence, reason = "blocked", [], None
            if matrix_id == "P02_CARRYOVER_FIXES":
                status, reason = "pending_P81_bundle_verification", "redaction-chain and Windows-safe clean-extract checks complete only in P81"
            elif matrix_id == "P62_K64_LOCAL_TRANSFORM":
                resolution_path = CR / "provenance/P62_GATE_RESOLUTION.json"
                resolution = load(resolution_path)
                status, reason = resolution["status"], resolution["reason"]
                evidence = [logical(resolution_path)]
            elif matrix_id == "P80_FINAL_ANALYSIS":
                status, evidence = "complete", [runtime.run_dir.name]
            elif matrix_id == "P81_ARTIFACT_BUNDLE":
                status, reason = "pending_P81_bundle_verification", "P81 follows this analysis"
            elif matrix_id in ("P72_ACCURACY", "P73_GSM8K", "P74_PG19"):
                if downstream["authorized"]:
                    job = {"P72_ACCURACY": "P72_analysis", "P73_GSM8K": "P73_analysis",
                           "P74_PG19": "P74_analysis"}[matrix_id]
                    evidence = [latest_complete(job).name]
                    status = "complete"
                else:
                    status, reason = "stopped_by_gate", downstream["stopped"][0]["reason"]
                    evidence = [logical(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")]
            elif matrix_id == "P32_CAL_SIZE_256" and not p32["continue_to_p32"]:
                status, reason = "stopped_by_gate", p32["reason"]
                evidence = [latest_complete("P32_terminal").name]
            elif matrix_id in breadth and not breadth[matrix_id]["authorized"]:
                status, reason = "stopped_by_gate", breadth[matrix_id]["reason"]
                evidence = [latest_complete(TERMINAL_JOBS[matrix_id]).name]
            else:
                job = TERMINAL_JOBS.get(matrix_id)
                if job:
                    evidence = [latest_complete(job).name]
                    status = "complete"
                else:
                    reason = "no terminal mapping"
            rows.append({**spec, "status": status, "evidence": evidence, "reason": reason})
    summary = dict(Counter(row["status"] for row in rows))
    unresolved = [row["matrix_id"] for row in rows
                  if row["status"] not in ("complete", "stopped_by_gate", "unsupported")]
    allowed = {"P02_CARRYOVER_FIXES", "P81_ARTIFACT_BUNDLE"}
    if set(unresolved) != allowed:
        raise RuntimeError(f"unexpected nonterminal matrix rows before P81: {unresolved}")
    return {"schema_version": "1.0", "status": "pre_bundle", "protocol_sha256": PROTOCOL,
            "matrix_source": logical(MATRIX), "matrix_source_sha256": runtime.sha256_file(MATRIX),
            "rows": rows, "summary": summary, "nonterminal_rows": unresolved,
            "authoritative_final_coverage": "P81 bundle_metadata/MATRIX_COVERAGE.json"}


def g5_continuation_gate(p31, p32, evidence):
    """Represent G5's scientific outcome separately from protocol compliance.

    G5 passes only when the frozen cross-model marginal-gain rule authorizes
    256+256 calibration. Correctly stopping P32 means the protocol was
    followed, but it does not turn the scientific continuation gate positive.
    """
    continued = bool(p32["continue_to_p32"])
    return {"passed": continued, "continued_to_256": continued,
            "decision_rule_applied": True, "evidence": evidence,
            "detail": p31["G5"]}


def gate_register():
    g0_path, g0 = artifact("P01_anchor_acceptance", "g0/G0_INTEGRITY_GATE.json")
    p10_path, p10 = artifact("P10_P11_analysis", "analysis_breadth/P10_P11_BREADTH_ANALYSIS.json")
    p20_path, p20 = artifact("P20_analysis", "analysis_p20/P20_DENSITY_CONTROL_ANALYSIS.json")
    p31_path, p31 = artifact("P31_analysis", "analysis_p31/P31_CALIBRATION_SIZE_ANALYSIS.json")
    p51_path, p51 = artifact("P51_analysis", "analysis_p51/P51_TILE_FIDELITY_ANALYSIS.json")
    p60_path, p60 = artifact("P60_analysis", "analysis_p60/P60_ANALYSIS.json")
    p61_path, p61 = artifact("P61_analysis", "analysis_p61/P61_ANALYSIS.json")
    p71_path, p71 = artifact("P71_analysis", "analysis_p71/P71_HELDOUT_PPL_ANALYSIS.json")
    p75_path, p75 = artifact("P75_analysis", "analysis_p75/P75_PORTABILITY_ANALYSIS.json")
    p32 = load(CR / "provenance/P32_GATE_RESOLUTION.json")
    downstream = load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")
    p72_path = p72 = None
    if downstream["authorized"]:
        p72_path, p72 = artifact("P72_analysis", "analysis_p72/P72_ACCURACY_ANALYSIS.json")
    g4_pass = all(p20["G4_matched"]["scope_n16_not_worse"].values())
    g9_component = p71["G9_heldout_component"]
    gates = {
        "G0": {"passed": bool(g0.get("passed")), "evidence": logical(g0_path), "detail": g0},
        "G1": {"passed": bool(p10["gates"]["G1"]["passed"]), "evidence": logical(p10_path), "detail": p10["gates"]["G1"]},
        "G2": {"passed": bool(p10["gates"]["G2"]["passed"]), "evidence": logical(p10_path), "detail": p10["gates"]["G2"]},
        "G3": {"passed": bool(p20["G3"]["passed"]), "evidence": logical(p20_path), "detail": p20["G3"]},
        "G4": {"passed": g4_pass, "evidence": logical(p20_path), "detail": p20["G4_matched"]},
        "G5": g5_continuation_gate(
            p31, p32, logical(CR / "provenance/P32_GATE_RESOLUTION.json")
        ),
        "G6": {"passed": bool(p51["G6"]["passed"]), "evidence": logical(p51_path), "detail": p51["G6"]},
        "G7": {"passed": bool(p60["selection"]["G7_compatible"]),
               "evidence": [logical(p60_path), logical(p61_path), "provenance/P63_COMBINATION_GATE.json"],
               "detail": {"weight_scale": p60["selection"], "activation": p61["selection"],
                          "local_transform": load(CR / "provenance/P62_GATE_RESOLUTION.json")}},
        "G8": {"passed": bool(p71["G8_heldout_quality"]["passed"] and p72 and p72["G8_accuracy_noninferiority"]["passed"]),
               "quality_passed": bool(p71["G8_heldout_quality"]["passed"]),
               "accuracy_passed": None if p72 is None else bool(p72["G8_accuracy_noninferiority"]["passed"]),
               "evidence": [logical(p71_path)] + ([logical(p72_path)] if p72_path else []),
               "detail": {"heldout_quality": p71["G8_heldout_quality"],
                          "accuracy": None if p72 is None else p72["G8_accuracy_noninferiority"]}},
        "G9": {"passed": False, "heldout_component_passed": bool(g9_component["winner_better_than_strongest_all_system_all_cells"]),
               "evidence": logical(p71_path), "detail": g9_component,
               "reason": "broad SOTA requires a corrected, scope-matched full-panel superiority family; held-out component alone is insufficient"},
        "PORTABILITY": {"passed": bool(p75["passed"]), "evidence": logical(p75_path), "detail": p75},
    }
    return {"schema_version": "1.0", "protocol_sha256": PROTOCOL, "gates": gates}


def g2_material_claim_classification(gate):
    detail = gate.get("detail") or {}
    if gate.get("passed") and not detail.get("practically_equivalent", False):
        return "supported"
    if detail.get("practically_equivalent", False):
        return "unsupported_practically_equivalent"
    return "unsupported"


def claim_register(gates):
    G = gates["gates"]
    def claim(cid, text, classification, evidence, limitation):
        return {"id": cid, "claim": text, "classification": classification,
                "evidence": evidence, "limitation": limitation}
    rows = [
        claim("C01", "N16 k2 broadly improves PPL versus FourOverSix in the frozen breadth panel",
              "supported_post_hoc" if G["G1"]["passed"] else "unsupported",
              G["G1"]["evidence"], "k2 was selected from prior results; this is a post-hoc robustness extension"),
        claim("C02", "N16 k2 materially improves over N16 k3",
              g2_material_claim_classification(G["G2"]),
              G["G2"]["evidence"], "development/breadth gate; no per-model threshold tuning"),
        claim("C03", "The k2 gain is selector-specific rather than selected-weight density",
              "supported" if G["G3"]["passed"] else "unsupported_density_driven",
              G["G3"]["evidence"], "requires both global and per-module selected-weight controls"),
        claim("C04", "N16 quality is no worse than N8 at matched selected-weight density",
              "supported" if G["G4"]["passed"] else "unsupported_tradeoff_only",
              G["G4"]["evidence"], "native deployment overhead was not measured"),
        claim("C05", "A replacement selector has better full-map PPL and tile-effect sign/rank fidelity",
              "supported_development" if G["G6"]["passed"] else "unsupported_old_selector_fallback",
              G["G6"]["evidence"], "tile interventions are the mechanistic inference unit"),
        claim("C06", "The frozen winner generalizes prospectively to two held-out families",
              "supported" if G["G8"]["quality_passed"] else "unsupported",
              G["G8"]["evidence"], "software/fake-quant PPL only"),
        claim("C07", "The frozen winner is downstream non-inferior on the frozen eight-task suite",
              "supported" if G["G8"]["accuracy_passed"] else "unsupported_or_stopped_by_gate",
              G["G8"]["evidence"], "GSM8K and PG19 remain secondary/exploratory transfers"),
        claim("C08", "The frozen winner beats the strongest fair prior-art baseline broadly enough for SOTA framing",
              "supported" if G["G9"]["passed"] else "unsupported",
              G["G9"]["evidence"], G["G9"]["reason"]),
        claim("C09", "Exact-map fake-quant quality is portable between A6000 and RTX 6000 Ada",
              "supported" if G["PORTABILITY"]["passed"] else "unsupported",
              G["PORTABILITY"]["evidence"], "quality portability only; no native performance implication"),
        claim("C10", "The first-order tile score establishes tile-level causality or interpretability",
              "unsupported", "parent negative evidence plus P51", "old exact CE sign was 3/9; aggregate map CIs are conditional on the stored map"),
        claim("C11", "k is a per-tile significance guarantee based on independent CE/KL tests",
              "unsupported", "parent multiplicity audit", "CE/KL dependence invalidates the independence argument"),
        claim("C12", "MixFP4 executed native FP4/E0M3 Tensor Cores or establishes overhead, speedup, area, or power",
              "out_of_scope_unsupported", "protocol scope", "all experiments use BF16-dequantized software/fake quantization"),
    ]
    return {"schema_version": "1.0", "protocol_sha256": PROTOCOL,
            "claims": rows, "recommended_scope": narrowest_scope(rows, G)}


def narrowest_scope(claims, gates):
    if gates["G9"]["passed"] and gates["G8"]["passed"]:
        return "general positive MixFP4 selection method"
    if gates["G1"]["passed"] and gates["G8"]["quality_passed"]:
        return "robust post-hoc k2 quality extension without a native-performance or SOTA claim"
    if gates["G1"]["passed"] and not gates["G3"]["passed"]:
        return "density-driven PPL improvement"
    if not gates["G4"]["passed"]:
        return "N16/N8 engineering trade-off study with deployment benefit unmeasured"
    return "map-level robustness plus tile-level negative-result/workshop study"


def risk_rows(gates, audit):
    G = gates["gates"]
    data = {
        "R01": ("mitigated", "k2 is explicitly labeled post-hoc; only the later held-out winner is prospective"),
        "R02": ("closed" if G["G3"]["passed"] else "confirmed_confound",
                "matched selected-weight controls completed; interpret by G3"),
        "R03": ("closed_for_reporting", "Qwen-outlier-excluded summaries are retained"),
        "R04": ("closed" if G["G6"]["passed"] else "open_negative_result", "P51 exact-tile G6 result"),
        "R05": ("closed_by_claim_removal", "no CE/KL independence or per-tile significance claim"),
        "R06": ("mitigated", "exact hashes and five-draw/aggregation evidence; map identity is not overclaimed"),
        "R07": ("closed" if G["G9"]["passed"] else "open_blocks_SOTA", "full baseline table retained; G9 controls framing"),
        "R08": ("closed_as_evidence", "natural and matched-density N16/N8 contrasts reported under G4"),
        "R09": ("closed_as_quality_screen", "format-preserving activation candidates screened; incompatible unfused work excluded"),
        "R10": ("closed" if G["G8"]["passed"] else "open_blocks_general_quality_claim", "accuracy/GSM8K/PG19 follow frozen gate"),
        "R11": ("closed", "five draws report within-evaluation and between-draw uncertainty separately"),
        "R12": ("closed", "two reserved new families evaluated only after winner hash lock"),
        "R13": ("open_out_of_scope", "no native GPU, overhead, speedup, area, or power claim is made"),
        "R14": ("closed" if G["G7"]["passed"] else "open", "G7 excludes hidden metadata/unfused/high-precision scope drift"),
        "R15": ("closed", "all development candidates and staged selection rules are retained"),
        "R16": ("pending_P81", "redacted chain, Windows-safe archive, and clean extraction are independently verified in P81"),
        "R17": ("closed" if audit["zero_policy_violations_among_accepted_runs"] else "open_blocker", "GPU ownership/concurrency audit"),
        "R18": ("closed_as_preserved_evidence", "all Qwen OOM/failure attempts remain in the append-only inventory"),
    }
    return [{"id": key, "disposition": value[0], "evidence_or_reason": value[1]} for key, value in data.items()]


def markdown_reports(out, gates, claims, risks, audit, coverage, final_ppl):
    G = gates["gates"]
    stat = ["# Statistical validity report", "",
            f"Protocol SHA-256: `{PROTOCOL}`.", "",
            "All paired PPL estimands are token-weighted `dlogPPL = mean_NLL(method) - mean_NLL(comparator)`; negative favors the method. Final intervals use 10,000 deterministic document/article-cluster bootstrap replicates unless an artifact explicitly labels an exploratory count. Finite Monte Carlo tests use the plus-one rule, and frozen endpoint families use Holm adjustment.", "",
            f"The complete machine-readable table contains {len(final_ppl['raw_rows'])} raw PPL rows and {len(final_ppl['paired_dlogppl_rows'])} paired dlogPPL/CI rows. Each row names its source path, full source SHA-256, and (for nested analyses) JSON pointer.", "",
            "Calibration-map uncertainty is not folded into an ordinary evaluation-window CI. `CALIBRATION_ROBUSTNESS.json` separately reports each of five maps, between-draw SD/small-sample intervals, within-evaluation bootstrap variance, pairwise map overlap, and hierarchical sensitivity.", "",
            "## Gate outcomes", "", "| gate | passed | interpretation |", "|---|---:|---|"]
    for key in [f"G{i}" for i in range(10)]:
        row = G[key]
        stat.append(f"| {key} | {row.get('passed')} | {json.dumps({k:v for k,v in row.items() if k not in ('detail','evidence')}, sort_keys=True)} |")
    stat += ["", "Development searches (k, aggregation, selector, scale/clipping) are explicitly post-selection evidence; naive winner CIs are not treated as confirmatory. Held-out P71 inference begins only after the exact global configuration and maps were hash-locked.", "",
             "Raw tables: `FINAL_PPL_TABLE.csv/json`, `CALIBRATION_ROBUSTNESS.csv/json`, `MATCHED_BUDGET_CONTROLS.csv/json`, and `DOWNSTREAM_RESULTS.csv/json`."]
    (out / "STATISTICAL_VALIDITY_REPORT.md").write_text("\n".join(stat) + "\n")

    repro = ["# Reproduction report", "",
             f"Extension protocol: `freeze/PROTOCOL_EXTENSION.json`, SHA-256 `{PROTOCOL}`.",
             f"Parent authoritative artifact manifest: `{PARENT_MANIFEST}`; parent campaign remains read-only.", "",
             "This campaign is append-only and uses exact stored maps. Every accepted GPU run includes host and in-container before/during/phase/post ownership evidence, UUID-scoped device isolation, a sidecar no longer than 60 seconds, and a run-level checksum manifest.", "",
             f"Attempts audited before P80 finalization: {audit['attempt_count']}; GPU-hours (including failed/invalid attempts): {audit['gpu_hours']['total']:.9f}; maximum concurrent campaign GPUs: {audit['gpu_concurrency']['max_concurrent_campaign_gpus']}.", "",
             "No-GPU reproduction from the extracted reviewer ZIP:", "",
             "```bash", "python tools/verify_bundle.py .", "python tools/recompute_tables.py .", "```", "",
             "These commands verify packaged hashes, parse JSON/JSONL/gzip members, validate exact-map binaries, and independently rebuild/check the four machine-readable result tables. Model caches and raw selector tensor workspaces are excluded from the core ZIP and are not needed for these checks.", "",
             "Optional GPU reproduction uses the frozen model revisions, calibration/evaluation manifests, exact maps, and the launcher policy in the bundle. It requires one homogeneous A6000 or RTX 6000 Ada allocation; portability uses the same map separately on each family. No native FP4/E0M3 performance result is reproduced or implied."]
    (out / "REPRODUCTION_REPORT.md").write_text("\n".join(repro) + "\n")

    risk = ["# Final submission risk audit", "", f"Recommended scope: **{claims['recommended_scope']}**.", "",
            "This is a software/fake-quant quality campaign. Native FP4/E0M3 Tensor Core execution, the external N8 ~13% / N16 ~1.5% overhead estimates, latency/speedup, area, and power are not claims of this work.", "",
            "| risk | disposition | evidence/reason |", "|---|---|---|"]
    risk += [f"| {row['id']} | {row['disposition']} | {row['evidence_or_reason']} |" for row in risks]
    risk += ["", "## Claim classifications", "", "| claim | classification | limitation |", "|---|---|---|"]
    risk += [f"| {row['id']} | {row['classification']} | {row['limitation']} |" for row in claims["claims"]]
    risk += ["", "All adverse baselines, negative results, OOM/crash/invalid attempts, and stopped-by-gate cells remain in `FAILED_OR_SKIPPED_RUNS.json` and the append-only registry. P81 changes R16 from pending to closed only after an independent clean-extraction verification passes."]
    (out / "FINAL_SUBMISSION_RISK_AUDIT.md").write_text("\n".join(risk) + "\n")


def parent_diff(gates, audit):
    return {
        "schema_version": "1.0", "relationship": "append-only post-hoc robustness extension",
        "parent": {"logical_id": "parent:mixfp4_n16k64_full_validation_20260911T065444Z",
                   "artifact_manifest_sha256": PARENT_MANIFEST, "run_directories": 338,
                   "manifest_entries": 78705, "matrix": {"complete": 39, "partial": 0, "missing": 0}},
        "extension": {"logical_id": "campaign:mixfp4_n16k64_ppl_improvement_20260914T162041Z",
                      "protocol_sha256": PROTOCOL, "attempts_before_P80": audit["attempt_count"],
                      "gpu_hours_before_P80": audit["gpu_hours"]["total"],
                      "new_models": ["granite8b", "falcon3_10b"],
                      "added_work": ["N16/N8 k2 breadth", "selected-weight-density controls", "five calibration draws",
                                     "128+128 and gated 256+256 calibration", "development global-k search",
                                     "cross-draw aggregation", "selector and exact-tile fidelity", "format-preserving scale/clipping",
                                     "held-out families", "accuracy/GSM8K/PG19", "A6000/Ada exact-map portability"],
                      "gates": {k: {x: v for x, v in row.items() if x in ("passed", "quality_passed", "accuracy_passed")}
                                for k, row in gates["gates"].items()}},
        "unchanged_scope": ["software/fake-quant quality only", "exact-map evidence is authoritative",
                            "negative/OOM/invalid evidence retained", "no native performance claim"]}


def test_audit(out):
    source = Path(__file__).resolve().parents[1]
    tests = source / "campaign/tests"
    targets = [str(path.relative_to(source)) for path in sorted(tests.glob("test_extension*.py"))]
    targets += [f"campaign/tests/{name}" for name in (
        "test_e2e_cpu.py", "test_gpu_preflight.py", "test_v10_quantizer.py",
        "test_v11_tiles.py", "test_v12_scores.py", "test_v13_maps.py")]
    command = [sys.executable, "-m", "pytest", "-q", *targets,
               "-k", "not audited_tile_totals_match_known_results"]
    proc = subprocess.run(command, cwd=source, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=3600)
    path = out / "CPU_TEST_REPORT.txt"
    path.write_text("$ " + " ".join(command) + "\n" + proc.stdout)
    result = {"command": command, "returncode": proc.returncode,
              "output_path": logical(path), "output_sha256": runtime.sha256_file(path),
              "passed": proc.returncode == 0,
              "legacy_parent_harness_excluded": {
                  "files": ["test_fidelity_noise.py", "test_final_reports_accuracy.py"],
                  "single_test": "test_v11_tiles.py::test_audited_tile_totals_match_known_results",
                  "reason": "these inherited tests import parent-only PROTOCOL_FREEZE.json or KNOWN_RESULTS.json absent by design from the extension handoff; the authoritative parent test artifacts were hash-audited by P00"}}
    if proc.returncode:
        raise RuntimeError(f"CPU test suite failed; see {path}")
    return result


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    out = runtime.out_dir("final")
    tests = test_audit(out)
    audit = audit_runs()
    if not audit["passed"]:
        runtime.atomic_json(out / "ARTIFACT_VALIDATION_FAILED.json", audit)
        raise RuntimeError("artifact/GPU-policy audit failed")
    manifest_path = build_artifact_manifest(out, audit)
    runtime.atomic_json(out / "ARTIFACT_VALIDATION.json", audit)
    runtime.atomic_json(out / "GPU_POLICY_AUDIT.json", {
        key: audit[key] for key in ("accepted_gpu_attempts", "gpu_concurrency", "gpu_hours",
                                    "zero_policy_violations_among_accepted_runs")})
    runtime.atomic_json(out / "FAILED_OR_SKIPPED_RUNS.json", {
        "schema_version": "1.0", "attempts": audit["failures_oom_invalid_and_crashes"],
        "all_preserved_in_registry": True, "registry_sha256": runtime.sha256_file(CR / "registry/attempts.jsonl")})

    raw, raw_sources = collect_raw_ppl()
    contrasts, contrast_sources = collect_contrasts()
    final_ppl = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                 "definition": "dlogPPL = mean_NLL(method) - mean_NLL(comparator); negative favors method",
                 "raw_rows": raw, "paired_dlogppl_rows": contrasts,
                 "sources": {**raw_sources, **contrast_sources}}
    runtime.atomic_json(out / "FINAL_PPL_TABLE.json", final_ppl)
    combined = raw + contrasts
    write_csv(out / "FINAL_PPL_TABLE.csv", combined)

    cal_items, cal_flat = table_bundle(["P30_FIVE_DRAW_ANALYSIS.json",
                                        "P31_CALIBRATION_SIZE_ANALYSIS.json",
                                        "P32_CALIBRATION_SIZE_ANALYSIS.json",
                                        "P41_DRAW_AGGREGATION_ANALYSIS.json"])
    cal = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
           "within_and_between_uncertainty_separated": True, "artifacts": cal_items, "flat_rows": cal_flat,
           "P32_gate": load(CR / "provenance/P32_GATE_RESOLUTION.json")}
    runtime.atomic_json(out / "CALIBRATION_ROBUSTNESS.json", cal)
    write_csv(out / "CALIBRATION_ROBUSTNESS.csv", cal_flat)

    budget_items, budget_flat = table_bundle(["P20_DENSITY_CONTROL_ANALYSIS.json",
                                              "P21_BREADTH_ANALYSIS.json"])
    budget = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
              "budget_unit": "selected weights, with global and per-module scopes separate",
              "artifacts": budget_items, "flat_rows": budget_flat,
              "P21_gate": load(CR / "provenance/P21_GATE_RESOLUTION.json")}
    runtime.atomic_json(out / "MATCHED_BUDGET_CONTROLS.json", budget)
    write_csv(out / "MATCHED_BUDGET_CONTROLS.csv", budget_flat)

    downstream_names = ["P72_ACCURACY_ANALYSIS.json", "P73_GSM8K_ANALYSIS.json",
                        "P74_PG19_ANALYSIS.json", "P75_PORTABILITY_ANALYSIS.json"]
    downstream_items, downstream_flat = table_bundle(downstream_names)
    downstream = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                  "gate": load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json"),
                  "artifacts": downstream_items, "flat_rows": downstream_flat}
    runtime.atomic_json(out / "DOWNSTREAM_RESULTS.json", downstream)
    write_csv(out / "DOWNSTREAM_RESULTS.csv", downstream_flat)

    coverage = matrix_coverage_prebundle()
    runtime.atomic_json(out / "MATRIX_COVERAGE_PRE_BUNDLE.json", coverage)
    gates = gate_register()
    runtime.atomic_json(out / "GATE_REGISTER.json", gates)
    claims = claim_register(gates)
    runtime.atomic_json(out / "CLAIM_REGISTER.json", claims)
    risks = risk_rows(gates, audit)
    runtime.atomic_json(out / "PAPER_RISK_REGISTER_FINAL.json", {"schema_version": "1.0", "risks": risks})
    runtime.atomic_json(out / "PARENT_CAMPAIGN_DIFF.json", parent_diff(gates, audit))
    markdown_reports(out, gates, claims, risks, audit, coverage, final_ppl)

    # Each final file gets a conventional sidecar; the index is written after the
    # sidecars so it cannot accidentally omit them.
    for path in sorted(out.iterdir()):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "FINAL_DELIVERABLES_INDEX.json":
            (out / f"{path.name}.sha256").write_text(f"{runtime.sha256_file(path)}  {path.name}\n")
    index = {path.name: {"sha256": runtime.sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file() and path.name != "FINAL_DELIVERABLES_INDEX.json"}
    final_index = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                   "created_utc": utc_now(), "files": index,
                   "artifact_manifest": {"path": logical(manifest_path),
                                         "sha256": runtime.sha256_file(manifest_path)},
                   "tests": tests, "coverage": coverage["summary"],
                   "claim_classifications": {row["id"]: row["classification"] for row in claims["claims"]},
                   "P81_required": True}
    runtime.atomic_json(out / "FINAL_DELIVERABLES_INDEX.json", final_index)
    (out / "FINAL_DELIVERABLES_INDEX.json.sha256").write_text(
        f"{runtime.sha256_file(out / 'FINAL_DELIVERABLES_INDEX.json')}  FINAL_DELIVERABLES_INDEX.json\n")
    launch = load(runtime.run_dir / "launch_record.json")
    outputs = [str(path) for path in sorted(out.iterdir()) if path.is_file()]
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P80-final-analysis", "model_revision": runtime.sha256_file(out / "FINAL_DELIVERABLES_INDEX.json"),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per-run", "evaluation_manifest_sha256": "per-run",
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": outputs,
                    "summary": {"matrix": coverage["summary"], "gates": {k: v["passed"] for k, v in gates["gates"].items()},
                                "attempts": audit["attempt_count"], "gpu_hours": audit["gpu_hours"]["total"]},
                    "uncertainty": {"primary_bootstrap_replicates": 10000},
                    "attempted_endpoints": ["P80_FINAL_ANALYSIS"],
                    "missing_endpoints": ["P81_ARTIFACT_BUNDLE"]}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "attempts": audit["attempt_count"],
                      "gpu_hours": audit["gpu_hours"]["total"], "matrix": coverage["summary"],
                      "recommended_scope": claims["recommended_scope"]}, sort_keys=True))


if __name__ == "__main__":
    main()
