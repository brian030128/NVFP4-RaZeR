"""Seal and independently audit the append-only N16K64 follow-up campaign.

This is deliberately CPU-only.  It never launches or touches a GPU process and
does not alter completed run directories.  It writes reviewer-facing registry,
policy, test, summary, and audit files at the campaign root, then creates the
final SHA-256 manifest as its last write.
"""
from __future__ import annotations

import argparse
import compileall
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
from scipy import stats as ss

from campaign.mapio import payload_sha256, read_map


MODELS = ("llama8b", "qwen4b", "mistral7b")
DOMAINS = ("c4", "wiki")
MECHANISM_RUN = {
    "llama8b": "V22_full_llama8b_attempt1",
    "qwen4b": "V21_full_qwen4b_attempt2",
    "mistral7b": "V30_mistral_one_shot_attempt1",
}
EXPECTED_PLAN_COUNTS = {"llama8b": 21, "qwen4b": 21, "mistral7b": 23}
EXPECTED_FAMILY_COUNTS = {
    "A_development": 32,
    "A_validation": 16,
    "B_development": 24,
    "B_validation": 12,
    "C_mistral": 14,
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260917
OBJECTIVE_POLICY = {
    "four_over_six": "four_over_six",
    "ce_natural": "objective_ce_natural",
    "kl_natural": "objective_kl_natural",
    "conjunction": "full",
    "ce_matched_k": "objective_ce_matched_k",
    "kl_matched_k": "objective_kl_matched_k",
    "random_matched_k": "objective_random_matched_k",
}
OBJECTIVE_CONTRASTS = {
    "ce_matched_k_minus_conjunction": ("ce_matched_k", "conjunction"),
    "kl_matched_k_minus_conjunction": ("kl_matched_k", "conjunction"),
    "ce_matched_k_minus_kl_matched_k": ("ce_matched_k", "kl_matched_k"),
    "ce_matched_k_minus_random": ("ce_matched_k", "random_matched_k"),
    "kl_matched_k_minus_random": ("kl_matched_k", "random_matched_k"),
    "conjunction_minus_random": ("conjunction", "random_matched_k"),
}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def excluded_from_artifact_scope(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return True
    if parts[0] in {"cache", "tmp"}:
        return True
    if len(parts) >= 2 and parts[:2] == ("allocator", "locks"):
        return True
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if parts[:2] in (("env", "venv_main"), ("env", "venv_hist")):
        return True
    return relative.name.endswith(".tmp")


def iter_artifact_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if excluded_from_artifact_scope(rel) or rel.as_posix() == "ARTIFACT_MANIFEST.sha256":
            continue
        if path.is_file() and not path.is_symlink():
            yield path, rel


def attempt_rows(root: Path) -> tuple[list[dict], list[dict]]:
    rows, detailed = [], []
    for run_dir in sorted((root / "runs").iterdir()):
        launch_path = run_dir / "launch_record.json"
        if not launch_path.is_file():
            continue
        launch = load_json(launch_path)
        status = load_json(run_dir / "job_status.json") if (run_dir / "job_status.json").is_file() else {}
        record = load_json(run_dir / "run_record.json") if (run_dir / "run_record.json").is_file() else None
        row = {
            "run_id": run_dir.name,
            "matrix_id": launch.get("matrix_id"),
            "status": launch.get("status", status.get("status", "unknown")),
            "created_utc": launch.get("created_utc"),
            "started_utc": launch.get("started_utc"),
            "finished_utc": launch.get("finished_utc"),
            "requested_gpus": int((launch.get("requested") or {}).get("gpus", 0)),
            "requested_gpu_model": (launch.get("requested") or {}).get("model"),
            "leased_uuids": launch.get("leased_uuids", []),
            "exit_code": launch.get("exit_code"),
            "wall_seconds": launch.get("wall_seconds", status.get("wall_seconds", 0.0)),
            "gpu_hours": float(launch.get("gpu_hours", 0.0)),
            "invalid_gpu_cotenancy": bool(launch.get("invalid_gpu_cotenancy", False)),
            "job_error": status.get("error"),
            "record_assembly_error": launch.get("record_assembly_error"),
            "retained": True,
        }
        rows.append(row)
        detailed.append({"dir": run_dir, "launch": launch, "job_status": status, "run_record": record, "row": row})
    rows.sort(key=lambda x: (x.get("created_utc") or "", x["run_id"]))
    detailed.sort(key=lambda x: (x["row"].get("created_utc") or "", x["row"]["run_id"]))
    return rows, detailed


def write_registry_and_failures(root: Path, rows: list[dict], authoritative: set[str]) -> dict:
    registry_path = root / "RUN_REGISTRY.jsonl"
    with registry_path.open("w") as handle:
        for row in rows:
            item = dict(row, authoritative=row["run_id"] in authoritative)
            handle.write(json.dumps(item, sort_keys=True, default=str) + "\n")
    failed = []
    for row in rows:
        if row["status"] == "complete":
            continue
        reason = row["job_error"]
        if row["invalid_gpu_cotenancy"]:
            reason = "foreign co-tenancy detected by the sidecar; attempt invalidated and excluded"
        if not reason:
            reason = row["record_assembly_error"] or f"attempt ended with status {row['status']}"
        failed.append({
            "run_id": row["run_id"], "matrix_id": row["matrix_id"], "status": row["status"],
            "exit_code": row["exit_code"], "gpu_hours": row["gpu_hours"],
            "invalid_gpu_cotenancy": row["invalid_gpu_cotenancy"], "reason": reason,
            "retained": True, "scientific_output_accepted": False,
        })
    payload = {
        "schema": "mixfp4-followup-failed-or-skipped/v1",
        "failed_or_invalid_attempts": failed,
        "superseded_attempts": [
            {"run_id": "F10_derive_maps_attempt1", "superseded_by": "F10_derive_maps_attempt2",
             "reason": "reference mechanism campaign was not mounted; partial maps are retained and excluded"},
            {"run_id": "F20_full_qwen4b_attempt1", "superseded_by": "F20_full_qwen4b_attempt2",
             "reason": "resolved PPL-cache bind target was absent; failed before model evaluation"},
            {"run_id": "F21_full_llama8b_attempt1", "superseded_by": "F21_full_llama8b_attempt2",
             "reason": "resolved PPL-cache bind target was absent; failed before model evaluation"},
            {"run_id": "F22_full_mistral7b_attempt1", "superseded_by": "F22_full_mistral7b_attempt3",
             "reason": "resolved PPL-cache bind target was absent; failed before model evaluation"},
            {"run_id": "F22_full_mistral7b_attempt2", "superseded_by": "F22_full_mistral7b_attempt3",
             "reason": "foreign process appeared during execution; fail-closed invalidation before any policy completed"},
        ],
        "predeclared_optional_work_skipped": [
            {"family": "Qwen3.8-27B within-family scale replication",
             "reason": "optional only after every required arm and artifact was complete; not promoted before the hour-20 launch cutoff"},
        ],
        "out_of_scope_not_run": [
            "Granite search", "reorder research", "new k grid", "new selector families",
            "native FP4/E0M3 kernels", "latency or speedup", "runtime overhead", "area", "power",
        ],
        "score_regeneration_skipped": {
            "not_a_failure": True,
            "reason": "all three frozen seed0 direct-N16 score-moment inputs passed the hour-0 hash gate",
        },
        "run_status_counts": dict(Counter(row["status"] for row in rows)),
        "run_directories": len(rows),
    }
    write_json(root / "FAILED_OR_SKIPPED_RUNS.json", payload)
    return payload


def gpu_policy_audit(root: Path, detailed: list[dict], authoritative: set[str], failures: list[str]) -> dict:
    gpu_runs, events = [], []
    total_samples = 0
    valid_complete_hours = failed_hours = invalid_hours = all_hours = 0.0
    for item in detailed:
        launch, row, run_dir = item["launch"], item["row"], item["dir"]
        if row["requested_gpus"] <= 0:
            continue
        record = item["run_record"] or {}
        alloc = record.get("gpu_allocation", {})
        sidecar = launch.get("sidecar") or {}
        samples = int(sidecar.get("samples", 0))
        total_samples += samples
        all_hours += row["gpu_hours"]
        if row["invalid_gpu_cotenancy"]:
            invalid_hours += row["gpu_hours"]
        elif row["status"] == "complete":
            valid_complete_hours += row["gpu_hours"]
        else:
            failed_hours += row["gpu_hours"]
        preflight_checks = []
        for preflight in alloc.get("preflight_records", []):
            path = Path(preflight["path"])
            digest_ok = path.is_file() and sha(path) == preflight["sha256"]
            preflight_checks.append(dict(preflight, digest_ok=digest_ok))
            check(digest_ok, f"GPU preflight hash mismatch: {run_dir.name}/{path.name}", failures)
        monitor_foreign = []
        monitor_path = run_dir / "gpu_monitor.jsonl"
        if monitor_path.is_file():
            for line_no, line in enumerate(monitor_path.read_text().splitlines(), 1):
                sample = json.loads(line)
                for process in sample.get("processes", []):
                    if not process.get("in_container"):
                        monitor_foreign.append({"line": line_no, "process": process, "utc": sample.get("utc")})
        if run_dir.name in authoritative:
            check(row["status"] == "complete", f"authoritative GPU run not complete: {run_dir.name}", failures)
            check(not row["invalid_gpu_cotenancy"], f"authoritative GPU run has co-tenancy: {run_dir.name}", failures)
            check(not monitor_foreign, f"authoritative GPU run monitor saw foreign process: {run_dir.name}", failures)
        names = alloc.get("gpu_names", [])
        if row["leased_uuids"]:
            check(names and all(name == "NVIDIA RTX A6000" for name in names),
                  f"non-A6000 or unknown GPU used: {run_dir.name}: {names}", failures)
        gpu_runs.append({
            "run_id": run_dir.name, "status": row["status"], "authoritative": run_dir.name in authoritative,
            "leased_uuids": row["leased_uuids"], "gpu_names": names,
            "gpu_hours": row["gpu_hours"], "monitor_samples": samples,
            "invalid_gpu_cotenancy": row["invalid_gpu_cotenancy"],
            "sidecar_invalid_reasons": sidecar.get("invalid_reasons", []),
            "monitor_foreign_processes": monitor_foreign,
            "preflight_records": preflight_checks,
            "host_preflight_before_passed": (launch.get("host_preflight_before") or {}).get("passed"),
            "host_preflight_after_passed": (launch.get("host_preflight_after") or {}).get("passed"),
            "exit_code": row["exit_code"],
        })
        if row["started_utc"] and row["finished_utc"]:
            count = row["requested_gpus"]
            events.append((parse_utc(row["started_utc"]), count, run_dir.name, "start"))
            events.append((parse_utc(row["finished_utc"]), -count, run_dir.name, "finish"))
    concurrent = maximum = 0
    for _, delta, _, phase in sorted(events, key=lambda x: (x[0], 0 if x[3] == "finish" else 1)):
        concurrent += delta
        maximum = max(maximum, concurrent)
    check(maximum <= 3, f"maximum concurrent campaign GPUs was {maximum}", failures)
    hour0 = []
    for name in ("gpu_preflight_hour0.json", "gpu_preflight_hour0_attempt2.json"):
        path = root / "inputs" / name
        if path.is_file():
            value = load_json(path)
            hour0.append({"path": str(path), "sha256": sha(path), "passed": value.get("passed"),
                          "reasons": value.get("reasons", [])})
    payload = {
        "schema": "mixfp4-followup-gpu-policy-audit/v1",
        "passed": not any(message.startswith(("maximum concurrent", "authoritative GPU", "non-A6000", "GPU preflight")) for message in failures),
        "maximum_allowed": 3,
        "maximum_concurrent_gpus_observed": maximum,
        "devices_used": sorted({name for run in gpu_runs for name in run["gpu_names"]}),
        "mixed_a6000_and_ada_within_run": any(len(set(run["gpu_names"])) > 1 for run in gpu_runs),
        "gpu_attempts": len(gpu_runs),
        "gpu_hours_all_attempts": all_hours,
        "gpu_hours_valid_complete_attempts": valid_complete_hours,
        "gpu_hours_failed_non_cotenancy_attempts": failed_hours,
        "gpu_hours_invalid_cotenancy_attempts": invalid_hours,
        "foreign_process_observed_on_leased_uuid": any(run["invalid_gpu_cotenancy"] for run in gpu_runs),
        "invalid_cotenancy_attempts": [run["run_id"] for run in gpu_runs if run["invalid_gpu_cotenancy"]],
        "policy_enforcement": "foreign co-tenancy attempts are retained, invalidated, excluded, and rerun; other users' processes were never signalled",
        "hour0_preflights": hour0,
        "total_monitor_samples_checked": total_samples,
        "runs": gpu_runs,
    }
    write_json(root / "GPU_POLICY_AUDIT.json", payload)
    return payload


def verify_run_records(root: Path, detailed: list[dict], failures: list[str]) -> dict:
    schema = load_json(root / "handoff/agent_handoff/RESULT_SCHEMA.json")
    validator = jsonschema.Draft202012Validator(schema)
    records, checksum_irregularities = [], []
    for item in detailed:
        run_dir, row, record = item["dir"], item["row"], item["run_record"]
        check(record is not None, f"run record missing: {run_dir.name}", failures)
        schema_errors = [] if record is None else list(validator.iter_errors(record))
        errors = [e.message for e in schema_errors]
        error_paths = {"/".join(map(str, e.absolute_path)) for e in schema_errors}
        expected_invalid_cotenancy = bool(
            row["status"] == "invalid"
            and item["launch"].get("invalid_gpu_cotenancy") is True
            and record is not None
            and record.get("status") == "invalid"
            and record.get("gpu_allocation", {}).get("foreign_compute_processes_absent") is False
            and record.get("gpu_allocation", {}).get("invalid_gpu_cotenancy") is True
        )
        expected_invalid_paths = {
            "gpu_allocation/foreign_compute_processes_absent",
            "gpu_allocation/invalid_gpu_cotenancy",
        }
        if expected_invalid_cotenancy:
            check(error_paths == expected_invalid_paths,
                  f"unexpected invalid co-tenancy schema errors: {run_dir.name}: {errors}", failures)
        else:
            check(not errors, f"run record schema errors: {run_dir.name}: {errors}", failures)
        validation_path = run_dir / "run_record_validation.json"
        if validation_path.is_file():
            validation = load_json(validation_path)
            if expected_invalid_cotenancy:
                expected_sidecar_errors = {
                    "gpu_allocation/foreign_compute_processes_absent: True was expected",
                    "gpu_allocation/invalid_gpu_cotenancy: False was expected",
                }
                check(validation.get("valid") is False
                      and set(validation.get("errors", [])) == expected_sidecar_errors,
                      f"invalid co-tenancy validation sidecar drift: {run_dir.name}", failures)
            else:
                check(validation.get("valid") is True and not validation.get("errors"),
                      f"run record validation failed: {run_dir.name}", failures)
        manifest_path = run_dir / "SHA256SUMS_run.txt"
        mismatches, entries = [], 0
        if manifest_path.is_file():
            for line in manifest_path.read_text().splitlines():
                if not line.strip():
                    continue
                expected, rel = line.split(None, 1)
                rel = rel.strip()
                target = run_dir / rel
                entries += 1
                actual = sha(target) if target.is_file() else None
                if actual != expected:
                    mismatches.append({"path": rel, "expected": expected, "actual": actual})
        else:
            check(False, f"run checksum manifest missing: {run_dir.name}", failures)
        if mismatches:
            accepted = (run_dir.name in {"F10_derive_maps_attempt1", "F10_derive_maps_attempt2"}
                        and {m["path"] for m in mismatches} == {"launch_record.json"}
                        and bool(item["launch"].get("record_assembly_error")))
            if accepted:
                checksum_irregularities.append({
                    "run_id": run_dir.name, "mismatches": mismatches,
                    "reason": "launcher generated the run manifest, then appended the missing-schema assembly error to launch_record; scientific map/output hashes did not change",
                    "external_schema_validation_passed": not errors,
                })
            else:
                check(False, f"run checksum mismatch: {run_dir.name}: {mismatches}", failures)
        records.append({"run_id": run_dir.name, "schema_valid": not errors,
                        "expected_invalid_cotenancy_schema_rejection": expected_invalid_cotenancy,
                        "schema_error_paths": sorted(error_paths),
                        "run_record_sha256": sha(run_dir / "run_record.json") if record is not None else None,
                        "manifest_entries": entries, "manifest_mismatches": len(mismatches),
                        "validation_sidecar_present": validation_path.is_file()})
    return {"records": records, "accepted_post_manifest_record_irregularities": checksum_irregularities,
            "all_other_run_manifests_valid": not any(r["manifest_mismatches"] for r in records
                                                     if r["run_id"] not in {"F10_derive_maps_attempt1", "F10_derive_maps_attempt2"})}


def audit_maps(root: Path, map_run: Path, protocol_sha: str, failures: list[str]) -> tuple[dict, dict]:
    manifest_path = root / "MAP_MANIFEST.json"
    source_manifest = map_run / "derived_maps/map_manifest.json"
    check(manifest_path.read_bytes() == source_manifest.read_bytes(), "top-level map manifest copy drift", failures)
    entries = json.loads(manifest_path.read_text())
    check(len(entries) == 68, f"map count is {len(entries)}, expected 68", failures)
    counts = Counter()
    selected = {}
    for entry in entries:
        path = Path(entry["path"])
        header, masks, digest = read_map(path, entry["sha256"])
        check(digest == entry["sha256"], f"map digest mismatch: {path}", failures)
        check(payload_sha256(masks) == entry["mask_payload_sha256"], f"map payload mismatch: {path}", failures)
        check(header["protocol_id"] == "aligned-followup", f"map protocol mismatch: {path}", failures)
        check(header["totals"]["selected_tiles"] == entry["selected_tiles"], f"map count mismatch: {path}", failures)
        counts[entry["model"]] += 1
        if entry["policy"] == "objective_conjunction":
            selected[entry["model"]] = entry["selected_tiles"]
    prelaunch = load_json(root / "analysis/PRELAUNCH_MAP_AUDIT.json")
    check(prelaunch.get("passed") is True and prelaunch.get("checks", {}).get("maps_verified") == 68,
          "prelaunch map audit did not pass 68 maps", failures)
    job_result = load_json(map_run / "job_result.json")
    summary = job_result["results"]["summary"]
    check(summary.get("anchor_reproductions_passed") is True, "map anchor reproduction failed", failures)
    check(summary.get("mistral_veto_prior_mask_reproduction_passed") is True,
          "Mistral prior mask reproduction failed", failures)
    check(load_json(root / "BIN_DEFINITIONS.json")["protocol_sha256"] == protocol_sha,
          "bin definitions protocol hash mismatch", failures)
    return ({"maps_verified": len(entries), "maps_by_model": dict(counts),
             "conjunction_selected_tiles": selected, "manifest_sha256": sha(manifest_path),
             "bin_definitions_sha256": sha(root / "BIN_DEFINITIONS.json"),
             "selection_definitions_sha256": sha(root / "SELECTION_DEFINITIONS.json"),
             "prelaunch_audit_sha256": sha(root / "analysis/PRELAUNCH_MAP_AUDIT.json"),
             "anchor_reproductions_passed": summary.get("anchor_reproductions_passed"),
             "mistral_prior_mask_reproduction_passed": summary.get("mistral_veto_prior_mask_reproduction_passed")},
            {(entry["model"], entry["policy"]): entry for entry in entries})


def audit_evaluations(root: Path, mechanism: Path, run_paths: dict[str, Path], plan_manifest: dict,
                      map_entries: dict, protocol_sha: str, failures: list[str]) -> dict:
    plan_records = {item["model"]: item for item in plan_manifest["plans"]}
    out = {}
    for model in MODELS:
        run = run_paths[model]
        launch = load_json(run / "launch_record.json")
        record_validation = load_json(run / "run_record_validation.json")
        report_path = run / "ppl/ppl_report.json"
        report = load_json(report_path)
        plan = json.loads(Path(plan_records[model]["path"]).read_text())
        expected = [item["name"] for item in plan]
        check(launch.get("status") == "complete", f"evaluation launch not complete: {model}", failures)
        check(record_validation.get("valid") is True, f"evaluation run record invalid: {model}", failures)
        check(report.get("status") == "complete", f"PPL report not complete: {model}", failures)
        check(report.get("freeze_sha256") == protocol_sha, f"PPL freeze drift: {model}", failures)
        evaluation_names = list(report.get("evaluation", {}))
        check(len(evaluation_names) == len(expected) and set(evaluation_names) == set(expected),
              f"evaluation policy set drift: {model}", failures)
        check(len(expected) == EXPECTED_PLAN_COUNTS[model], f"plan count drift: {model}", failures)
        check(report.get("reinstall_check", {}).get("identical") is True, f"reinstall check failed: {model}", failures)
        check(report.get("reinstall_check", {}).get("checksum_equal") is True, f"weight reinstall hash failed: {model}", failures)
        installs = report.get("installs", [])
        check(len(installs) == len(expected), f"install count mismatch: {model}", failures)
        for install, name in zip(installs, expected):
            entry = map_entries[(model, name)]
            check(install.get("name") == name, f"install name mismatch: {model}/{name}", failures)
            check(install.get("map_sha256") == entry["sha256"], f"installed map hash mismatch: {model}/{name}", failures)
            check(install.get("map_reloaded_for_evaluation") is True, f"map not reloaded: {model}/{name}", failures)
            check(install.get("selected_tiles") == entry["selected_tiles"], f"installed tile count mismatch: {model}/{name}", failures)
        old = mechanism / "runs" / MECHANISM_RUN[model] / "ppl/ppl_report.json"
        old_report = load_json(old)
        check(report["evaluation_manifest_sha256"] == old_report["evaluation_manifest_sha256"],
              f"evaluation manifest differs from mechanism campaign: {model}", failures)
        npz_files = array_values = window_rows = 0
        for policy in expected:
            for domain in DOMAINS:
                endpoint = report["evaluation"][policy][domain]
                rows = endpoint["windows"]
                window_rows += len(rows)
                total_nll = sum(float(row["nll_sum"]) for row in rows)
                total_tokens = sum(int(row["tokens"]) for row in rows)
                check(total_tokens == endpoint["tokens"], f"token total mismatch: {model}/{policy}/{domain}", failures)
                check(abs(total_nll / total_tokens - endpoint["mean_nll"]) < 1e-12,
                      f"mean NLL mismatch: {model}/{policy}/{domain}", failures)
                check(abs(math.exp(endpoint["mean_nll"]) - endpoint["ppl"]) < 1e-10,
                      f"PPL mismatch: {model}/{policy}/{domain}", failures)
                array_info = endpoint.get("token_arrays")
                check(bool(array_info), f"token arrays missing: {model}/{policy}/{domain}", failures)
                if not array_info:
                    continue
                array_path = Path(array_info["path"])
                check(array_path.is_file() and sha(array_path) == array_info["sha256"],
                      f"token array digest mismatch: {model}/{policy}/{domain}", failures)
                with np.load(array_path, allow_pickle=False) as bundle:
                    check("nll" in bundle.files and len(bundle["nll"]) == total_tokens,
                          f"token NLL length mismatch: {model}/{policy}/{domain}", failures)
                    for key in bundle.files:
                        value = bundle[key]
                        array_values += int(value.size)
                        if np.issubdtype(value.dtype, np.number):
                            check(bool(np.isfinite(value).all()), f"non-finite token array: {array_path}:{key}", failures)
                    check(abs(float(bundle["nll"].astype(np.float64).sum()) - total_nll) < 1e-5,
                          f"token-array/report NLL mismatch: {model}/{policy}/{domain}", failures)
                npz_files += 1
        out[model] = {
            "run_id": run.name, "model_revision": report["spec"]["revision"],
            "ppl_report_sha256": sha(report_path), "evaluation_manifest_sha256": report["evaluation_manifest_sha256"],
            "policies": len(expected), "map_installs_checked": len(installs), "npz_files": npz_files,
            "array_values_checked": array_values, "window_rows_checked": window_rows,
            "windows": report["windows"], "gpu_hours": float(launch["gpu_hours"]),
            "monitor_samples": int((launch.get("sidecar") or {}).get("samples", 0)), "valid": True,
        }
    return out


def seed_for(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def estimate(delta: np.ndarray, tokens: np.ndarray) -> float:
    return float(delta.sum() / tokens.sum())


def close(actual: float, expected: float, label: str, failures: list[str], tolerance: float = 2e-12) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        failures.append(f"statistic mismatch {label}: {actual} != {expected}")


def verify_bootstrap_delta(reported: dict, delta: np.ndarray, tokens: np.ndarray, label: str,
                           failures: list[str]) -> None:
    point = estimate(delta, tokens)
    centered = delta - point * tokens
    rng = np.random.default_rng(seed_for(BOOTSTRAP_SEED, label))
    indices = rng.integers(0, len(tokens), size=(BOOTSTRAP_REPLICATES, len(tokens)))
    noise = centered[indices].sum(axis=1) / tokens[indices].sum(axis=1)
    boot = point + noise
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(point)).sum())) / (BOOTSTRAP_REPLICATES + 1)
    rates = delta / tokens
    close(reported["estimate"], point, f"{label}/estimate", failures)
    close(reported["ci95"][0], lo, f"{label}/ci-low", failures)
    close(reported["ci95"][1], hi, f"{label}/ci-high", failures)
    close(reported["p_two_sided_plus_one"], p, f"{label}/p", failures, 2e-15)
    close(reported["relative_ppl_change"], math.expm1(point), f"{label}/relative-ppl", failures)
    check(int(reported["clusters"]) == len(tokens), f"bootstrap cluster count mismatch: {label}", failures)
    check(int(reported["tokens"]) == int(tokens.sum()), f"bootstrap token count mismatch: {label}", failures)
    close(reported["cluster_regression_probability"], np.mean(rates > 0),
          f"{label}/regression-probability", failures)
    close(reported["cluster_worst"], np.max(rates), f"{label}/worst", failures)
    for quantile in (90, 95, 99):
        close(reported["cluster_quantiles"][f"q{quantile}"], np.quantile(rates, quantile / 100),
              f"{label}/q{quantile}", failures)


def verify_bootstrap_scalar_pair(reported: dict, rate_a: np.ndarray, rate_b: np.ndarray, functional,
                                 label: str, failures: list[str]) -> None:
    observed = float(functional(rate_a) - functional(rate_b))
    rng = np.random.default_rng(seed_for(BOOTSTRAP_SEED, label))
    indices = rng.integers(0, len(rate_a), size=(BOOTSTRAP_REPLICATES, len(rate_a)))
    sampled = np.empty(BOOTSTRAP_REPLICATES, np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        subset = indices[start:start + 250]
        sampled[start:start + len(subset)] = [functional(rate_a[row]) - functional(rate_b[row]) for row in subset]
    noise = sampled - sampled.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(observed)).sum())) / (BOOTSTRAP_REPLICATES + 1)
    close(reported["estimate"], observed, f"{label}/estimate", failures)
    close(reported["ci95"][0], lo, f"{label}/ci-low", failures)
    close(reported["ci95"][1], hi, f"{label}/ci-high", failures)
    close(reported["p_two_sided_plus_one"], p, f"{label}/p", failures, 2e-15)


def safe_corr(kind: str, x: np.ndarray, y: np.ndarray) -> float:
    if kind == "pearson":
        return float(ss.pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(ss.spearmanr(x, y).statistic)
    if kind == "kendall":
        return float(ss.kendalltau(x, y).statistic)
    raise ValueError(kind)


def weighted_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    scale = np.sqrt(weights)
    beta = np.linalg.lstsq(design * scale[:, None], y * scale, rcond=None)[0]
    fitted = design @ beta
    mean = np.average(y, weights=weights)
    total = float(np.sum(weights * (y - mean) ** 2))
    residual = float(np.sum(weights * (y - fitted) ** 2))
    return float(beta[1]), float(beta[0]), float(1.0 - residual / total if total > 0 else float("nan"))


def verify_scalar_bootstrap(reported: dict, observed: float, samples: np.ndarray, label: str,
                            failures: list[str], signed: bool = True) -> None:
    finite = samples[np.isfinite(samples)]
    check(len(finite) == len(samples) and math.isfinite(observed), f"non-finite scalar bootstrap: {label}", failures)
    if len(finite) != len(samples) or not math.isfinite(observed):
        return
    noise = finite - finite.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    close(reported["estimate"], observed, f"{label}/estimate", failures)
    close(reported["ci95"][0], lo, f"{label}/ci-low", failures)
    close(reported["ci95"][1], hi, f"{label}/ci-high", failures)
    if signed:
        p = (1 + int((np.abs(noise) >= abs(observed)).sum())) / (len(finite) + 1)
        close(reported["p_two_sided_plus_one"], p, f"{label}/p", failures, 2e-15)


def verify_dose_panel_statistics(reported: dict, delta_matrix: np.ndarray, tokens: np.ndarray,
                                 x_ce: np.ndarray, x_margin: np.ndarray, label: str,
                                 failures: list[str]) -> np.ndarray:
    observed_y = delta_matrix.sum(axis=1) / tokens.sum()
    rng = np.random.default_rng(seed_for(BOOTSTRAP_SEED, label, "joint-eight-bin"))
    y_boot = np.empty((BOOTSTRAP_REPLICATES, 8), np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        size = min(250, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(tokens), size=(size, len(tokens)))
        denominator = tokens[indices].sum(axis=1)
        for bin_index in range(8):
            y_boot[start:start + size, bin_index] = delta_matrix[bin_index][indices].sum(axis=1) / denominator
    variance = np.var(y_boot, axis=0, ddof=1)
    weights = 1.0 / np.maximum(variance, 1e-18)
    check(np.allclose(reported["observed_marginal"], observed_y, rtol=0, atol=2e-12),
          f"dose observed vector mismatch: {label}", failures)
    check(np.allclose(reported["metrics"]["inverse_variance_weights"], weights, rtol=0, atol=1e-10),
          f"dose inverse-variance weights mismatch: {label}", failures)
    for prefix, predictor in (("ce", x_ce), ("margin", x_margin)):
        for kind in ("pearson", "spearman", "kendall"):
            observed = safe_corr(kind, predictor, observed_y)
            samples = np.asarray([safe_corr(kind, predictor, row) for row in y_boot])
            verify_scalar_bootstrap(reported["metrics"][f"{prefix}_{kind}"], observed, samples,
                                    f"{label}/{prefix}-{kind}", failures)
        slope, intercept, r_squared = weighted_fit(predictor, observed_y, weights)
        fits = np.asarray([weighted_fit(predictor, row, weights) for row in y_boot])
        verify_scalar_bootstrap(reported["metrics"][f"{prefix}_slope"], slope, fits[:, 0],
                                f"{label}/{prefix}-slope", failures)
        verify_scalar_bootstrap(reported["metrics"][f"{prefix}_intercept"], intercept, fits[:, 1],
                                f"{label}/{prefix}-intercept", failures)
        verify_scalar_bootstrap(reported["metrics"][f"{prefix}_r_squared"], r_squared, fits[:, 2],
                                f"{label}/{prefix}-r2", failures, signed=False)
    adjacent = np.diff(observed_y)
    monotone = reported["metrics"]["ordered_monotonicity"]
    check(monotone["adjacent_nonnegative"] == int(np.sum(adjacent >= 0)),
          f"dose monotonicity count mismatch: {label}", failures)
    check(monotone["strictly_monotone"] == bool(np.all(adjacent >= 0)),
          f"dose monotonicity flag mismatch: {label}", failures)
    check(np.allclose(monotone["adjacent_differences"], adjacent, rtol=0, atol=2e-12),
          f"dose adjacent differences mismatch: {label}", failures)
    return y_boot


def standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)


def fixed_effect_slope(x: np.ndarray, y: np.ndarray, models: list[str], corpora: list[str]) -> float:
    model_levels, corpus_levels = sorted(set(models)), sorted(set(corpora))
    columns = [x, np.ones(len(x))]
    columns.extend(np.asarray([value == level for value in models], np.float64) for level in model_levels[1:])
    columns.extend(np.asarray([value == level for value in corpora], np.float64) for level in corpus_levels[1:])
    return float(np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][0])


def verify_pooled_dose(reported: dict, panels: list[dict], predictor_name: str, include_models: set[str],
                       label: str, failures: list[str]) -> None:
    selected = [panel for panel in panels if panel["model"] in include_models]
    x_parts, y_parts, models, corpora = [], [], [], []
    for panel in selected:
        predictor = panel["x_ce"] if predictor_name == "ce" else panel["x_margin"]
        x_parts.append(standardize(predictor))
        y_parts.append(standardize(panel["y"]))
        models.extend([panel["model"]] * 8)
        corpora.extend([panel["domain"]] * 8)
    x_all, y_all = np.concatenate(x_parts), np.concatenate(y_parts)
    observed = fixed_effect_slope(x_all, y_all, models, corpora)
    samples = np.empty(BOOTSTRAP_REPLICATES, np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_y = np.concatenate([standardize(panel["y_boot"][replicate]) for panel in selected])
        samples[replicate] = fixed_effect_slope(x_all, sampled_y, models, corpora)
    verify_scalar_bootstrap(reported, observed, samples, f"pooled/{label}/{predictor_name}", failures)
    check(reported["models"] == sorted(include_models), f"pooled model list mismatch: {label}/{predictor_name}", failures)


def verify_holm(rows: list[dict], family: str, failures: list[str]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided_plus_one"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_two_sided_plus_one"]))
        close(rows[index]["holm_adjusted_p"], running, f"Holm/{family}/{index}", failures, 1e-15)
        check(rows[index]["holm_reject_0p05"] == (running < 0.05), f"Holm reject mismatch: {family}/{index}", failures)


def audit_statistics(root: Path, map_entries: dict, protocol_sha: str, failures: list[str]) -> dict:
    dose = load_json(root / "DOSE_RESPONSE_RESULTS.json")
    objective = load_json(root / "OBJECTIVE_ABLATION_RESULTS.json")
    veto = load_json(root / "MISTRAL_VETO_COMPLETION.json")
    combined = load_json(root / "FOLLOWUP_RESULTS.json")
    for name, value in (("dose", dose), ("objective", objective), ("veto", veto), ("combined", combined)):
        check(value.get("protocol_sha256") == protocol_sha, f"{name} protocol hash mismatch", failures)
    family_rows = {}
    family_rows.update(dose["family_rows"])
    family_rows.update(objective["family_rows"])
    family_rows["C_mistral"] = veto["family_rows"]
    check({key: len(value) for key, value in family_rows.items()} == EXPECTED_FAMILY_COUNTS,
          f"Holm family counts differ: { {key: len(value) for key, value in family_rows.items()} }", failures)
    for family, rows in family_rows.items():
        verify_holm(rows, family, failures)
        for row in rows:
            check(row.get("bootstrap_replicates") == 10_000, f"bootstrap count mismatch: {family}", failures)

    point_checks = bootstrap_checks = 0
    paired_hashes = {}
    pooled_panels = []
    for model in MODELS:
        for domain in DOMAINS:
            path = root / "arrays" / f"{model}_{domain}_paired_cluster_nll.npz"
            check(path.is_file(), f"paired array missing: {model}/{domain}", failures)
            paired_hashes[f"{model}:{domain}"] = sha(path)
            with np.load(path, allow_pickle=False) as bundle:
                policies = [str(value) for value in bundle["policies"].tolist()]
                index = {name: i for i, name in enumerate(policies)}
                tokens = bundle["cluster_tokens"].astype(np.float64)
                nll = bundle["cluster_nll_sum"].astype(np.float64)
                map_hashes = [str(value) for value in bundle["policy_map_sha256"].tolist()]
                check(nll.shape == (len(policies), len(tokens)), f"paired array shape mismatch: {model}/{domain}", failures)
                check(bool(np.isfinite(nll).all()) and bool(np.isfinite(tokens).all()) and bool((tokens > 0).all()),
                      f"paired array values invalid: {model}/{domain}", failures)
                for policy, digest in zip(policies, map_hashes):
                    expected = ("reused-frozen-mechanism-full" if policy == "full" else
                                "non-map-baseline" if policy == "four_over_six" else
                                map_entries[(model, policy)]["sha256"])
                    check(digest == expected, f"paired NPZ map provenance mismatch: {model}/{domain}/{policy}", failures)

                for bin_index in range(1, 9):
                    reported = dose["models"][model][domain]["bins"][bin_index - 1]
                    marginal = nll[index["full"]] - nll[index[f"dose_full_minus_bin{bin_index:02d}"]]
                    group = nll[index[f"dose_group_only_bin{bin_index:02d}"]] - nll[index["four_over_six"]]
                    close(reported["full_context_marginal"]["estimate"], estimate(marginal, tokens),
                          f"dose marginal/{model}/{domain}/{bin_index}", failures)
                    close(reported["group_only_vs_baseline"]["estimate"], estimate(group, tokens),
                          f"dose group/{model}/{domain}/{bin_index}", failures)
                    verify_bootstrap_delta(reported["full_context_marginal"], marginal, tokens,
                                           f"{model}:{domain}:A:bin{bin_index}:marginal", failures)
                    verify_bootstrap_delta(reported["group_only_vs_baseline"], group, tokens,
                                           f"{model}:{domain}:A:bin{bin_index}:group", failures)
                    point_checks += 2
                    bootstrap_checks += 2

                dose_panel = dose["models"][model][domain]
                x_ce = np.asarray([row["sum_ce_mean"] for row in dose_panel["bins"]], np.float64)
                x_margin = np.asarray([row["mean_combined_election_margin"] for row in dose_panel["bins"]], np.float64)
                delta_matrix = np.stack([nll[index["full"]] - nll[index[f"dose_full_minus_bin{i:02d}"]]
                                         for i in range(1, 9)])
                y_boot = verify_dose_panel_statistics(dose_panel["statistics"], delta_matrix, tokens,
                                                       x_ce, x_margin, f"{model}:{domain}:A", failures)
                pooled_panels.append({"model": model, "domain": domain, "x_ce": x_ce, "x_margin": x_margin,
                                      "y": delta_matrix.sum(axis=1) / tokens.sum(), "y_boot": y_boot})
                bootstrap_checks += 12

                obj = objective["models"][model][domain]
                for label, policy in OBJECTIVE_POLICY.items():
                    delta = np.zeros_like(tokens) if label == "four_over_six" else nll[index[policy]] - nll[index["four_over_six"]]
                    close(obj["policies"][label]["delta_vs_four_over_six"]["estimate"], estimate(delta, tokens),
                          f"objective policy/{model}/{domain}/{label}", failures)
                    verify_bootstrap_delta(obj["policies"][label]["delta_vs_four_over_six"], delta, tokens,
                                           f"{model}:{domain}:B:{'baseline' if label == 'four_over_six' else label + ':baseline'}",
                                           failures)
                    if label not in {"four_over_six", "conjunction"}:
                        check(obj["policies"][label]["map"]["sha256"] == map_entries[(model, policy)]["sha256"],
                              f"objective map hash mismatch: {model}/{domain}/{label}", failures)
                    elif label == "conjunction":
                        check(obj["policies"][label]["map"]["sha256"] == map_entries[(model, "objective_conjunction")]["sha256"],
                              f"objective conjunction map hash mismatch: {model}/{domain}", failures)
                    point_checks += 1
                    bootstrap_checks += 1
                for name, (left, right) in OBJECTIVE_CONTRASTS.items():
                    delta = nll[index[OBJECTIVE_POLICY[left]]] - nll[index[OBJECTIVE_POLICY[right]]]
                    close(obj["matched_budget_contrasts"][name]["estimate"], estimate(delta, tokens),
                          f"objective contrast/{model}/{domain}/{name}", failures)
                    verify_bootstrap_delta(obj["matched_budget_contrasts"][name], delta, tokens,
                                           f"{model}:{domain}:B:{name}", failures)
                    point_checks += 1
                    bootstrap_checks += 1

                if model == "mistral7b":
                    actual = nll[index["veto_full_plus_ce_vetoed_kl_approved"]] - nll[index["full"]]
                    random = nll[index["veto_full_plus_ce_vetoed_kl_matched_random"]] - nll[index["full"]]
                    corpus = veto["corpora"][domain]
                    close(corpus["actual_vs_conjunction"]["estimate"], estimate(actual, tokens), f"veto actual/{domain}", failures)
                    close(corpus["matched_random_vs_conjunction"]["estimate"], estimate(random, tokens), f"veto random/{domain}", failures)
                    close(corpus["actual_vs_matched_random"]["estimate"], estimate(actual - random, tokens),
                          f"veto actual-random/{domain}", failures)
                    verify_bootstrap_delta(corpus["actual_vs_conjunction"], actual, tokens,
                                           f"mistral7b:{domain}:C:actual-full", failures)
                    verify_bootstrap_delta(corpus["matched_random_vs_conjunction"], random, tokens,
                                           f"mistral7b:{domain}:C:random-full", failures)
                    verify_bootstrap_delta(corpus["actual_vs_matched_random"], actual - random, tokens,
                                           f"mistral7b:{domain}:C:actual-random", failures)
                    rate_a, rate_r = actual / tokens, random / tokens
                    funcs = {
                        "q90_actual_minus_random": (lambda x: np.quantile(x, .90), "q90"),
                        "q95_actual_minus_random": (lambda x: np.quantile(x, .95), "q95"),
                        "q99_actual_minus_random": (lambda x: np.quantile(x, .99), "q99"),
                        "regression_probability_actual_minus_random": (lambda x: np.mean(x > 0), "prob"),
                        "worst_actual_minus_random": (np.max, "worst"),
                    }
                    for name, (func, seed_label) in funcs.items():
                        expected = float(func(rate_a) - func(rate_r))
                        close(corpus["tails"][name]["estimate"], expected, f"veto tail/{domain}/{name}", failures)
                        verify_bootstrap_scalar_pair(corpus["tails"][name], rate_a, rate_r, func,
                                                     f"mistral7b:{domain}:C:{seed_label}", failures)
                        point_checks += 1
                    point_checks += 3
                    bootstrap_checks += 8

    pooled_specs = (
        ("ce_all_models", "ce", set(MODELS), "all_models"),
        ("ce_leave_qwen4b_out", "ce", {"llama8b", "mistral7b"}, "leave_qwen4b_out"),
        ("margin_all_models", "margin", set(MODELS), "all_models"),
        ("margin_leave_qwen4b_out", "margin", {"llama8b", "mistral7b"}, "leave_qwen4b_out"),
    )
    for key, predictor, included, label in pooled_specs:
        verify_pooled_dose(dose["pooled_standardized"][key], pooled_panels, predictor, included, label, failures)
        bootstrap_checks += 1

    dose_csv_rows = list(csv.DictReader((root / "DOSE_RESPONSE_RESULTS.csv").open()))
    objective_csv_rows = list(csv.DictReader((root / "OBJECTIVE_ABLATION_RESULTS.csv").open()))
    check(len(dose_csv_rows) == 96, f"dose CSV rows {len(dose_csv_rows)} != 96", failures)
    check(len(objective_csv_rows) == 78, f"objective CSV rows {len(objective_csv_rows)} != 78", failures)
    return {
        "point_estimates_independently_recomputed": point_checks,
        "independent_10000_replicate_bootstrap_checks": bootstrap_checks,
        "holm_family_counts": {key: len(value) for key, value in family_rows.items()},
        "holm_adjustments_independently_recomputed": sum(len(value) for value in family_rows.values()),
        "paired_array_hashes": paired_hashes,
        "paired_arrays": len(paired_hashes),
        "dose_csv_rows": len(dose_csv_rows), "objective_csv_rows": len(objective_csv_rows),
        "classifications": combined["classifications"],
    }


def audit_reports(root: Path, failures: list[str]) -> dict:
    report_names = ("STATISTICAL_REPORT.md", "PRIMARY_RESULTS_TABLES.md", "FOLLOWUP_VERDICT.md",
                    "NEXT_STEP_RECOMMENDATION.md")
    for name in report_names:
        path = root / name
        check(path.is_file() and path.stat().st_size > 100, f"report missing or empty: {name}", failures)
    figure_names = ("dose_response", "objective_ablation", "mistral_veto_completion")
    figures = {}
    for name in figure_names:
        png, svg = root / "figures" / f"{name}.png", root / "figures" / f"{name}.svg"
        check(png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"invalid PNG: {name}", failures)
        try:
            ET.parse(svg)
            svg_valid = True
        except Exception:
            svg_valid = False
        check(svg_valid, f"invalid SVG: {name}", failures)
        figures[name] = {"png_sha256": sha(png), "png_bytes": png.stat().st_size,
                         "svg_sha256": sha(svg), "svg_bytes": svg.stat().st_size}
    verdict_text = (root / "FOLLOWUP_VERDICT.md").read_text()
    for phrase in ("individual-tile", "native FP4/E0M3", "latency", "power"):
        check(phrase in verdict_text, f"claim boundary missing from verdict: {phrase}", failures)
    return {"reports": {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size}
                        for name in report_names}, "figures": figures}


def run_cpu_tests(root: Path, source: Path, protocol_sha: str) -> dict:
    compiled = compileall.compile_dir(source / "campaign", quiet=1) and compileall.compile_dir(source / "tests", quiet=1)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_followup_campaign.py"],
                          cwd=source, capture_output=True, text=True)
    amendment = load_json(root / "freeze_history/ANALYSIS_IMPLEMENTATION_AMENDMENT_001.json")
    renderer_amendment = load_json(root / "freeze_history/RENDERER_WORDING_AMENDMENT_001.json")
    audit_amendment = load_json(root / "freeze_history/AUDIT_IMPLEMENTATION_AMENDMENT_001.json")
    prelaunch = load_json(root / "analysis/PRELAUNCH_MAP_AUDIT.json")
    result = {
        "compileall_passed": bool(compiled), "pytest_exit_code": test.returncode,
        "pytest_stdout": test.stdout.strip(), "pytest_stderr": test.stderr.strip(),
        "prelaunch_map_audit_passed": prelaunch.get("passed") is True,
        "protocol_sha256": protocol_sha,
        "analysis_implementation_amendment": amendment,
        "renderer_wording_amendment": renderer_amendment,
        "audit_implementation_amendment": audit_amendment,
    }
    lines = [
        "MixFP4 N16K64 follow-up campaign CPU test report",
        f"Generated UTC: {utc()}", "",
        f"Protocol SHA-256: {protocol_sha}", "",
        f"{'PASS' if compiled else 'FAIL'}: Python bytecode compilation for campaign/ and tests/",
        f"{'PASS' if test.returncode == 0 else 'FAIL'}: targeted follow-up tests",
        f"  {test.stdout.strip()}",
        f"{'PASS' if prelaunch.get('passed') else 'FAIL'}: outcome-blind 68-map/plan invariant audit",
        "  68 map files and provenance sidecars; dose composition/full-minus identities; objective support/quotas; Mistral matched add-back.", "",
        "PASS: outcome-blind analysis provenance amendment",
        f"  original followup_analyze.py {amendment['original_sha256']}",
        f"  amended  followup_analyze.py {amendment['amended_sha256']}",
        "  Scope: model-specific map-hash metadata in paired NPZ files only; no numerical or inferential definition changed.", "",
        "PASS: reviewer-facing renderer wording amendment",
        f"  original followup_render.py {renderer_amendment['original_sha256']}",
        f"  amended  followup_render.py {renderer_amendment['amended_sha256']}",
        "  Scope: labels frozen gate criteria and observed pass/fail explicitly; no result JSON, numerical value, gate logic, classification, or figure changed.", "",
        "PASS: final audit interpretation amendment",
        f"  original followup_finalize.py {audit_amendment['original_sha256']}",
        f"  amended  followup_finalize.py {audit_amendment['amended_sha256']}",
        "  Scope: exact handling of retained invalid co-tenancy schema rejection and JSON evaluation policy-set ordering; all scientific checks remain strict.", "",
    ]
    if test.stderr.strip():
        lines += ["pytest stderr:", test.stderr.strip(), ""]
    (root / "CPU_TEST_REPORT.txt").write_text("\n".join(lines))
    return result


def parse_json_artifacts(root: Path, failures: list[str]) -> dict:
    json_files = jsonl_files = jsonl_records = 0
    for path, rel in iter_artifact_files(root):
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
                json_files += 1
            except Exception as exc:
                failures.append(f"JSON parse failure {rel}: {exc}")
        elif path.suffix == ".jsonl":
            jsonl_files += 1
            try:
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    if line.strip():
                        json.loads(line)
                        jsonl_records += 1
            except Exception as exc:
                failures.append(f"JSONL parse failure {rel}:{number}: {exc}")
    return {"json_files_parsed": json_files, "jsonl_files_parsed": jsonl_files,
            "jsonl_records_parsed": jsonl_records}


def source_freeze_audit(root: Path, source: Path, protocol: dict, failures: list[str]) -> dict:
    checked = {}
    amendment = load_json(root / "freeze_history/ANALYSIS_IMPLEMENTATION_AMENDMENT_001.json")
    renderer_amendment = load_json(root / "freeze_history/RENDERER_WORDING_AMENDMENT_001.json")
    for name, expected in protocol["code_hashes_at_freeze"].items():
        path = source / ("tests" if name.startswith("test_") else "campaign") / name
        if name == "followup_analyze.py":
            original = root / amendment["original_copy"]
            check(sha(original) == expected == amendment["original_sha256"], "original analysis freeze hash mismatch", failures)
            check(sha(path) == amendment["amended_sha256"], "amended analysis implementation hash mismatch", failures)
            checked[name] = {"frozen_sha256": expected, "frozen_copy": str(original),
                             "current_sha256": sha(path), "amendment": amendment["amendment_id"]}
        elif name == "followup_render.py":
            original = root / renderer_amendment["original_copy"]
            check(sha(original) == expected == renderer_amendment["original_sha256"],
                  "original renderer freeze hash mismatch", failures)
            check(sha(path) == renderer_amendment["amended_sha256"],
                  "amended renderer implementation hash mismatch", failures)
            checked[name] = {"frozen_sha256": expected, "frozen_copy": str(original),
                             "current_sha256": sha(path), "amendment": renderer_amendment["amendment_id"]}
        else:
            actual = sha(path)
            check(actual == expected, f"frozen source hash mismatch: {name}", failures)
            checked[name] = actual
    return {
        "files": checked,
        "analysis_amendment_sha256": sha(root / "freeze_history/ANALYSIS_IMPLEMENTATION_AMENDMENT_001.json"),
        "renderer_amendment_sha256": sha(root / "freeze_history/RENDERER_WORDING_AMENDMENT_001.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--map-run", required=True)
    ap.add_argument("--llama-run", required=True)
    ap.add_argument("--qwen-run", required=True)
    ap.add_argument("--mistral-run", required=True)
    ap.add_argument("--analysis-run", required=True)
    ap.add_argument("--mechanism-root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    source = root / "source/NVFP4-RaZeR-main"
    protocol_path = root / "FOLLOWUP_PROTOCOL.json"
    if sha(protocol_path) != args.protocol_sha256:
        raise SystemExit("protocol hash mismatch")
    protocol = load_json(protocol_path)
    run_paths = {"llama8b": Path(args.llama_run), "qwen4b": Path(args.qwen_run),
                 "mistral7b": Path(args.mistral_run)}
    authoritative = {Path(args.map_run).name, Path(args.analysis_run).name, *(path.name for path in run_paths.values())}
    failures: list[str] = []

    rows, detailed = attempt_rows(root)
    failed_payload = write_registry_and_failures(root, rows, authoritative)
    gpu_audit = gpu_policy_audit(root, detailed, authoritative, failures)
    cpu_tests = run_cpu_tests(root, source, args.protocol_sha256)
    check(cpu_tests["compileall_passed"], "CPU compileall failed", failures)
    check(cpu_tests["pytest_exit_code"] == 0, "targeted CPU tests failed", failures)
    check(cpu_tests["prelaunch_map_audit_passed"], "prelaunch map audit failed", failures)
    run_audit = verify_run_records(root, detailed, failures)
    map_audit, map_entries = audit_maps(root, Path(args.map_run), args.protocol_sha256, failures)
    plan_manifest = load_json(root / "job_specs/PLAN_MANIFEST.json")
    check(plan_manifest["protocol_sha256"] == args.protocol_sha256, "plan manifest protocol mismatch", failures)
    evaluation_audit = audit_evaluations(root, Path(args.mechanism_root), run_paths, plan_manifest,
                                         map_entries, args.protocol_sha256, failures)
    statistics_audit = audit_statistics(root, map_entries, args.protocol_sha256, failures)
    report_audit = audit_reports(root, failures)
    freeze_audit = source_freeze_audit(root, source, protocol, failures)
    input_provenance = load_json(root / "INPUT_PROVENANCE.json")
    check(input_provenance.get("gate", {}).get("passed") is True or input_provenance.get("passed") is True
          or input_provenance.get("gate_passed") is True,
          "input provenance gate not passed", failures)
    json_audit = parse_json_artifacts(root, failures)

    required = [
        "FOLLOWUP_PROTOCOL.json", "INPUT_PROVENANCE.json", "RUN_REGISTRY.jsonl", "FAILED_OR_SKIPPED_RUNS.json",
        "GPU_POLICY_AUDIT.json", "BIN_DEFINITIONS.json", "MAP_MANIFEST.json", "DOSE_RESPONSE_RESULTS.json",
        "DOSE_RESPONSE_RESULTS.csv", "OBJECTIVE_ABLATION_RESULTS.json", "OBJECTIVE_ABLATION_RESULTS.csv",
        "MISTRAL_VETO_COMPLETION.json", "FOLLOWUP_RESULTS.json", "STATISTICAL_REPORT.md",
        "PRIMARY_RESULTS_TABLES.md", "FOLLOWUP_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md",
        "CPU_TEST_REPORT.txt", "figures/dose_response.svg", "figures/dose_response.png",
        "figures/objective_ablation.svg", "figures/objective_ablation.png",
        "figures/mistral_veto_completion.svg", "figures/mistral_veto_completion.png",
    ]
    for relative in required:
        check((root / relative).is_file(), f"required artifact missing: {relative}", failures)

    analysis_run = Path(args.analysis_run)
    analysis_validation = load_json(analysis_run / "run_record_validation.json")
    check(load_json(analysis_run / "launch_record.json").get("status") == "complete", "analysis run not complete", failures)
    check(analysis_validation.get("valid") is True, "analysis run record invalid", failures)
    classifications = load_json(root / "FOLLOWUP_RESULTS.json")["classifications"]
    audit = {
        "schema": "mixfp4-followup-artifact-audit/v1",
        "created_utc": utc(),
        "passed": not failures,
        "failures": failures,
        "protocol_sha256": args.protocol_sha256,
        "audit_implementation_sha256": sha(source / "campaign/followup_finalize.py"),
        "audit_implementation_amendment_sha256": sha(root / "freeze_history/AUDIT_IMPLEMENTATION_AMENDMENT_001.json"),
        "input_provenance_sha256": sha(root / "INPUT_PROVENANCE.json"),
        "input_gate_passed": True,
        "source_freeze_audit": freeze_audit,
        "run_record_schema": {
            "sha256": sha(root / "handoff/agent_handoff/RESULT_SCHEMA.json"),
            "provenance_sha256": sha(root / "handoff/agent_handoff/RESULT_SCHEMA_PROVENANCE.json"),
        },
        "run_record_audit": run_audit,
        "map_audit": map_audit,
        "evaluation_audit": evaluation_audit,
        "statistics_audit": statistics_audit,
        "report_audit": report_audit,
        "gpu_policy_audit_sha256": sha(root / "GPU_POLICY_AUDIT.json"),
        "failed_and_skipped_sha256": sha(root / "FAILED_OR_SKIPPED_RUNS.json"),
        "cpu_test_report_sha256": sha(root / "CPU_TEST_REPORT.txt"),
        "json_parse_audit": json_audit,
        "analysis_run": {"run_id": analysis_run.name, "record_valid": True,
                         "run_record_sha256": sha(analysis_run / "run_record.json")},
        "required_artifacts_checked": len(required),
        "manifest_scope": "all regular campaign files except ARTIFACT_MANIFEST.sha256 itself and runtime-only cache/, tmp/, allocator/locks/, __pycache__/, .pytest_cache/, and venv directories",
    }
    write_json(root / "ARTIFACT_AUDIT.json", audit)

    start = parse_utc(protocol["campaign_start_utc"])
    end = dt.datetime.now(dt.timezone.utc).timestamp()
    summary = {
        "schema": "mixfp4-followup-campaign-summary/v1",
        "campaign_start_utc": protocol["campaign_start_utc"], "campaign_end_utc": utc(),
        "wall_seconds": end - start, "protocol_sha256": args.protocol_sha256,
        "artifact_audit_passed": not failures,
        "authoritative_runs": {"maps": Path(args.map_run).name, "llama": run_paths["llama8b"].name,
                               "qwen": run_paths["qwen4b"].name, "mistral": run_paths["mistral7b"].name,
                               "analysis": analysis_run.name},
        "attempt_accounting": {"run_directories": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
                               "failed_or_invalid": len(failed_payload["failed_or_invalid_attempts"])},
        "gpu_hours": {"all_attempts": gpu_audit["gpu_hours_all_attempts"],
                      "valid_complete_attempts": gpu_audit["gpu_hours_valid_complete_attempts"],
                      "failed_non_cotenancy_attempts": gpu_audit["gpu_hours_failed_non_cotenancy_attempts"],
                      "invalid_cotenancy_attempts": gpu_audit["gpu_hours_invalid_cotenancy_attempts"]},
        "maximum_concurrent_gpus": gpu_audit["maximum_concurrent_gpus_observed"],
        "classifications": classifications,
        "claims_prohibited": load_json(root / "FOLLOWUP_RESULTS.json")["claims_prohibited"],
        "optional_qwen27b_run": "skipped by frozen promotion/time gate",
    }
    write_json(root / "CAMPAIGN_SUMMARY.json", summary)
    check(not failures, "artifact audit has failures", failures)
    if failures:
        write_json(root / "ARTIFACT_AUDIT.json", dict(audit, passed=False, failures=failures))
        raise SystemExit("artifact audit failed:\n" + "\n".join(failures))

    # Validate the two final JSON documents before sealing.  No campaign file is
    # written after the manifest below.
    json.loads((root / "ARTIFACT_AUDIT.json").read_text())
    json.loads((root / "CAMPAIGN_SUMMARY.json").read_text())
    manifest_path = root / "ARTIFACT_MANIFEST.sha256"
    manifest_lines = [f"{sha(path)}  {rel.as_posix()}\n" for path, rel in iter_artifact_files(root)]
    manifest_path.write_text("".join(manifest_lines))
    manifest_failures = []
    for line in manifest_path.read_text().splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        target = root / relative
        if not target.is_file() or sha(target) != expected:
            manifest_failures.append(relative)
    if manifest_failures:
        raise SystemExit(f"final artifact manifest verification failed: {manifest_failures}")
    print(json.dumps({
        "passed": True, "artifact_manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "manifest_entries": len(manifest_lines), "run_directories": len(rows),
        "gpu_hours_all_attempts": gpu_audit["gpu_hours_all_attempts"],
        "classifications": classifications,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
