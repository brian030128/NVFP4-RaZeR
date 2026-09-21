"""Consolidate, independently audit, seal, and verify the completed campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from campaign import mapio
from campaign.boundary_common import (CAMPAIGN_START_UTC, DOMAINS, MECHANISM_ROOT, MECHANISM_RUN,
                                      MODELS, SEED_ROOT, atomic_json, sha256_file)


BOOTSTRAPS = 10_000
EXCLUDED_TOP = {"ARTIFACT_MANIFEST.sha256", "POST_SEAL_VERIFICATION.txt"}
EXCLUDED_PARTS = {"cache", "tmp", "allocator/locks", "__pycache__", ".pytest_cache"}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def close(a: float, b: float, tolerance: float = 2e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def seed_for(*parts: object) -> int:
    return int(hashlib.sha256(":".join([str(SEED_ROOT), *map(str, parts)]).encode()).hexdigest()[:16], 16)


def fixed_effect_slope(x: np.ndarray, y: np.ndarray, group: list[int]) -> float:
    levels = sorted(set(group))
    columns = [np.ones(len(x)), x]
    for level in levels[1:]:
        columns.append(np.asarray([value == level for value in group], np.float64))
    return float(np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][1])


def independent_delta(delta: np.ndarray, tokens: np.ndarray, label: str) -> dict:
    estimate = float(delta.sum() / tokens.sum())
    centered = delta - estimate * tokens
    rng = np.random.default_rng(seed_for(label))
    idx = rng.integers(0, len(tokens), size=(BOOTSTRAPS, len(tokens)))
    noise = centered[idx].sum(axis=1) / tokens[idx].sum(axis=1)
    samples = estimate + noise
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(estimate)).sum())) / (BOOTSTRAPS + 1)
    return {"estimate": estimate, "ci95": [float(lo), float(hi)], "p_two_sided_plus_one": float(p)}


def independent_joint(delta: np.ndarray, tokens: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    estimates = delta.sum(axis=1) / tokens.sum()
    centered = delta - estimates[:, None] * tokens[None, :]
    rng = np.random.default_rng(seed_for(label))
    samples = np.empty((BOOTSTRAPS, delta.shape[0]), np.float64)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        idx = rng.integers(0, len(tokens), size=(size, len(tokens)))
        denominator = tokens[idx].sum(axis=1)
        for row in range(delta.shape[0]):
            samples[start:start + size, row] = estimates[row] + centered[row][idx].sum(axis=1) / denominator
    return estimates, samples


def independent_scalar(observed: float, samples: np.ndarray) -> dict:
    noise = samples - samples.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(observed)).sum())) / (len(samples) + 1)
    return {"estimate": observed, "ci95": [float(lo), float(hi)], "p_two_sided_plus_one": float(p)}


def compare_result(expected: dict, actual: dict, label: str, failures: list[str]) -> None:
    for key in ("estimate", "p_two_sided_plus_one"):
        if not close(float(expected[key]), float(actual[key])):
            failures.append(f"{label}: {key} mismatch {expected[key]} vs {actual[key]}")
    for index in (0, 1):
        if not close(float(expected["ci95"][index]), float(actual["ci95"][index])):
            failures.append(f"{label}: ci95[{index}] mismatch")


def build_registry(root: Path, authoritative: set[str]) -> list[dict]:
    records = []
    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir() or not (run / "launch_record.json").is_file():
            continue
        launch = json.loads((run / "launch_record.json").read_text())
        status = json.loads((run / "job_status.json").read_text()) if (run / "job_status.json").is_file() else {}
        policy_marker_path = run / "INVALID_GPU_POLICY.json"
        strict_marker_path = run / "STRICT_MONITOR_INVALID.json"
        policy_marker = json.loads(policy_marker_path.read_text()) if policy_marker_path.is_file() else None
        strict_marker = json.loads(strict_marker_path.read_text()) if strict_marker_path.is_file() else None
        effective_marker = policy_marker or strict_marker
        invalid_reasons = launch.get("sidecar", {}).get("invalid_reasons", [])
        effective_status = "invalid" if effective_marker else launch.get("status")
        records.append({"run_id": run.name, "matrix_id": launch.get("matrix_id"),
                        "status": effective_status, "launcher_status": launch.get("status"),
                        "authoritative": run.name in authoritative,
                        "retained": True, "requested_gpus": launch.get("requested", {}).get("gpus", 0),
                        "requested_gpu_model": launch.get("requested", {}).get("model"),
                        "leased_uuids": launch.get("leased_uuids", []),
                        "created_utc": launch.get("created_utc"), "started_utc": launch.get("started_utc"),
                        "finished_utc": launch.get("finished_utc"), "wall_seconds": launch.get("wall_seconds", 0),
                        "gpu_hours": launch.get("gpu_hours", 0),
                        "invalid_gpu_cotenancy": launch.get("invalid_gpu_cotenancy", False),
                        "invalid_gpu_policy": bool(effective_marker),
                        "invalid_gpu_policy_record": str(policy_marker_path) if policy_marker else (
                            str(strict_marker_path) if strict_marker else None),
                        "strict_monitor_invalid_record": str(strict_marker_path) if strict_marker else None,
                        "invalid_gpu_policy_reason": effective_marker.get("reason") if effective_marker else None,
                        "scientific_outputs_valid": effective_marker.get("scientific_outputs_valid") if effective_marker else None,
                        "exit_code": launch.get("exit_code"), "job_error": status.get("error"),
                        "launcher_error": launch.get("docker_stderr"),
                        "invalid_reasons": invalid_reasons,
                        "late_foreign_processes": launch.get("late_foreign_processes", []),
                        "record_assembly_error": launch.get("record_assembly_error")})
    (root / "RUN_REGISTRY.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    return records


def gpu_audit(root: Path, registry: list[dict]) -> dict:
    attempts = []
    samples_total = 0
    all_compliant = True
    intervals = []
    model_names = set()
    for record in registry:
        if not record["requested_gpus"]:
            continue
        run = root / "runs" / record["run_id"]
        launch = json.loads((run / "launch_record.json").read_text())
        preflight_files = sorted((run / "preflight").glob("*.json"))
        preflights = [(path, json.loads(path.read_text())) for path in preflight_files]
        preflight_passed = all(item.get("passed") for _, item in preflights)
        before_and_phase = [(path, item) for path, item in preflights
                            if not path.name.endswith("_after.json")]
        launch_preflights_passed = bool(before_and_phase) and all(
            item.get("passed") for _, item in before_and_phase)
        selected = set(record["leased_uuids"])
        for _, item in preflights:
            for gpu in item.get("gpus", []):
                if gpu.get("uuid") in selected:
                    model_names.add(gpu.get("name"))
        monitor_path = run / "gpu_monitor.jsonl"
        monitor = [json.loads(line) for line in monitor_path.read_text().splitlines() if line] if monitor_path.is_file() else []
        strict_path = run / "strict_gpu_monitor.jsonl"
        strict = [json.loads(line) for line in strict_path.read_text().splitlines() if line] if strict_path.is_file() else []
        strict_samples = [row for row in strict if row.get("event") == "sample"]
        samples_total += len(monitor) + len(strict_samples)
        timestamps = [parse_utc(row["utc"]) for row in monitor if row.get("utc") and not row.get("error")]
        gaps = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
        max_gap = max(gaps, default=0.0)
        foreign = [process for row in monitor for process in row.get("processes", []) if not process.get("in_container")]
        monitor_errors = [row.get("error") for row in monitor if row.get("error")]
        strict_timestamps = [parse_utc(row["utc"]) for row in strict_samples if row.get("utc")]
        strict_terminal = [row for row in strict if row.get("event") == "container_terminal"]
        strict_gaps = [(right - left).total_seconds()
                       for left, right in zip(strict_timestamps, strict_timestamps[1:])]
        if strict_timestamps and launch.get("started_utc"):
            strict_gaps.append((strict_timestamps[0] - parse_utc(launch["started_utc"])).total_seconds())
        if strict_timestamps and strict_terminal:
            strict_gaps.append((parse_utc(strict_terminal[-1]["utc"]) - strict_timestamps[-1]).total_seconds())
        strict_max_gap = max(strict_gaps, default=0.0)
        strict_foreign = [process for row in strict_samples for process in row.get("processes", [])
                          if not process.get("in_container")]
        strict_errors = [row for row in strict if row.get("error") or row.get("event") == "invalidated"]
        strict_invalid = (run / "STRICT_MONITOR_INVALID.json").is_file()
        strict_available = bool(strict and any(row.get("event") == "watchdog_started" for row in strict))
        strict_clean = bool(strict_available and strict_samples and strict_terminal and strict_max_gap <= 60.0 and
                            not strict_foreign and not strict_errors and not strict_invalid)
        built_in_clean = bool(max_gap <= 60.0 and not foreign and not monitor_errors)
        monitoring_clean = strict_clean if strict_available else built_in_clean
        clean = (preflight_passed and monitoring_clean and
                 not launch.get("invalid_gpu_cotenancy", False) and not record.get("invalid_gpu_policy"))
        cotenancy_fail_closed = bool(
            launch.get("invalid_gpu_cotenancy", False) and record["status"] == "invalid" and
            launch_preflights_passed and foreign and not monitor_errors and max_gap <= 60.0 and
            not record["authoritative"] and launch.get("exit_code") not in (None, 0))
        monitoring_fail_closed = bool(
            record.get("invalid_gpu_policy") and record["status"] == "invalid" and
            record.get("scientific_outputs_valid") is False and launch_preflights_passed and
            not record["authoritative"] and launch.get("exit_code") not in (None, 0))
        fail_closed_compliant = cotenancy_fail_closed or monitoring_fail_closed
        policy_compliant = clean or fail_closed_compliant
        all_compliant &= policy_compliant
        attempts.append({"run_id": record["run_id"], "status": record["status"],
                         "leased_uuids": record["leased_uuids"], "preflight_records": len(preflights),
                         "all_preflights_passed": preflight_passed,
                         "launch_and_phase_preflights_passed": launch_preflights_passed,
                         "monitor_samples": len(monitor),
                         "maximum_monitor_gap_seconds": max_gap, "foreign_monitor_processes": foreign,
                         "monitor_errors": monitor_errors, "invalid_cotenancy": record["invalid_gpu_cotenancy"],
                         "monitoring_basis": "strict_independent_watchdog" if strict_available else "launcher_sidecar",
                         "strict_monitor_samples": len(strict_samples),
                         "strict_monitor_terminal_recorded": bool(strict_terminal),
                         "strict_maximum_monitor_gap_seconds": strict_max_gap,
                         "strict_foreign_monitor_processes": strict_foreign,
                         "strict_monitor_errors": strict_errors,
                         "strict_monitor_invalid_marker": strict_invalid,
                         "invalid_gpu_policy": record.get("invalid_gpu_policy", False),
                         "invalid_gpu_policy_reason": record.get("invalid_gpu_policy_reason"),
                         "invalid_reasons": record.get("invalid_reasons", []),
                         "exit_code": record.get("exit_code"), "authoritative": record["authoritative"],
                         "gpu_hours": record["gpu_hours"], "policy_clean": clean,
                         "fail_closed_compliant": fail_closed_compliant,
                         "policy_compliant": policy_compliant})
        if record.get("started_utc") and record.get("finished_utc"):
            intervals.extend([(parse_utc(record["started_utc"]), 1), (parse_utc(record["finished_utc"]), -1)])
    active = maximum = 0
    for _, change in sorted(intervals, key=lambda row: (row[0], row[1])):
        active += change
        maximum = max(maximum, active)
    valid = sum(row["gpu_hours"] for row in registry if row["requested_gpus"] and row["status"] == "complete")
    invalid = sum(row["gpu_hours"] for row in registry if row["requested_gpus"] and row["invalid_gpu_cotenancy"])
    invalid_policy = sum(row["gpu_hours"] for row in registry if row["requested_gpus"] and row.get("invalid_gpu_policy"))
    failed = sum(row["gpu_hours"] for row in registry if row["requested_gpus"] and row["status"] == "failed" and
                 not row["invalid_gpu_cotenancy"] and not row.get("invalid_gpu_policy"))
    all_hours = sum(row["gpu_hours"] for row in registry if row["requested_gpus"])
    process_output = subprocess.run(["ps", "-eo", "pid,user,cmd"], capture_output=True, text=True).stdout
    residual = [line for line in process_output.splitlines()
                if "mixfp4_n16k64_boundary_corruption_20260918T080303Z" in line and
                any(token in line for token in ("campaign.evaluate_ppl", "campaign.launcher")) and
                "boundary_finalize" not in line]
    return {"schema": "mixfp4-boundary-corruption-gpu-policy-audit/v1", "passed": all_compliant and maximum <= 3 and not residual,
            "maximum_allowed_concurrent_physical_gpus": 3, "maximum_observed_concurrent_gpus": maximum,
            "device_models_used": sorted(name for name in model_names if name),
            "homogeneous_a6000_only": model_names == {"NVIDIA RTX A6000"},
            "monitor_interval_requirement_seconds": 60, "monitor_samples": samples_total,
            "attempts": attempts, "co_tenancy_attempts": sum(row["invalid_cotenancy"] for row in attempts),
            "fail_closed_compliant_attempts": sum(row["fail_closed_compliant"] for row in attempts),
            "foreign_processes_signalled": 0, "residual_campaign_gpu_processes": residual,
            "gpu_hours": {"valid_complete_attempts": valid, "failed_non_cotenancy_attempts": failed,
                          "invalid_cotenancy_attempts": invalid,
                          "invalid_monitoring_policy_attempts": invalid_policy,
                          "all_attempts": all_hours}}


def write_operational_outputs(root: Path, map_run: Path, authoritative: set[str]) -> tuple[list[dict], dict]:
    for source, destination in (
        (map_run / "derived_maps/map_manifest.json", root / "MAP_MANIFEST.json"),
        (map_run / "derived_maps/anchor_checks.json", root / "ANCHOR_CHECKS.json"),
        (map_run / "derived_maps/p1_distinctness_checks.json", root / "P1_DISTINCTNESS_CHECKS.json"),
    ):
        if destination.exists():
            if sha256_file(destination) == sha256_file(source):
                continue
            raise FileExistsError(f"refusing to overwrite nonidentical artifact: {destination}")
        shutil.copyfile(source, destination)
    registry = build_registry(root, authoritative)
    gpu = gpu_audit(root, registry)
    atomic_json(root / "GPU_POLICY_AUDIT.json", gpu)
    source = root / "source/NVFP4-RaZeR-main/campaign"
    atomic_json(root / "POSTPROCESSING_PROVENANCE.json", {
        "schema": "mixfp4-boundary-corruption-postprocessing-provenance/v1",
        "created_utc": utc(),
        "scientific_rules_source": "PROTOCOL.json code_hashes and PRE_GPU_FREEZE.sha256",
        "gpu_evaluator_source_manifest": "verified independently from each authoritative GPU launch",
        "postlaunch_non_scientific_scripts": {
            "campaign/boundary_analyze_postfreeze.py": sha256_file(source / "boundary_analyze_postfreeze.py"),
            "campaign/boundary_render.py": sha256_file(source / "boundary_render.py"),
            "campaign/boundary_finalize.py": sha256_file(source / "boundary_finalize.py")
        },
        "analysis_correction_record": {
            "path": str(root / "ANALYSIS_CORRECTIONS.json"),
            "sha256": sha256_file(root / "ANALYSIS_CORRECTIONS.json"),
            "freeze_manifest": str(root / "ANALYSIS_CORRECTION_FREEZE.sha256"),
            "freeze_manifest_sha256": sha256_file(root / "ANALYSIS_CORRECTION_FREEZE.sha256")
        },
        "scope": "outcome-blind evaluation-identity plumbing correction, rendering, operational consolidation, independent recomputation, artifact audit, and checksum sealing only",
        "changes_to_frozen_maps_or_statistics": False
    })
    failures = [{"run_id": row["run_id"], "matrix_id": row["matrix_id"], "status": row["status"],
                 "reason": row.get("invalid_gpu_policy_reason") or row["job_error"] or row.get("launcher_error") or
                 "; ".join(row.get("invalid_reasons", [])),
                 "gpu_hours": row["gpu_hours"],
                 "invalid_gpu_cotenancy": row["invalid_gpu_cotenancy"],
                 "invalid_gpu_policy": row.get("invalid_gpu_policy", False), "retained": True,
                 "scientific_outputs_used": False}
                for row in registry if row["status"] not in ("complete",)]
    assembly_warnings = [{"run_id": row["run_id"], "status": row["status"],
                          "record_assembly_error": row["record_assembly_error"]}
                         for row in registry if row.get("record_assembly_error")]
    skipped = [{"arm": "score regeneration", "reason": "verified frozen sufficient statistics were complete"},
               {"arm": "optional/exploratory experiments", "reason": "none were predeclared; campaign remained tightly scoped"},
               {"arm": "native FP4/E0M3 performance", "reason": "explicitly out of scope"}]
    atomic_json(root / "FAILED_OR_SKIPPED_RUNS.json", {
        "schema": "mixfp4-boundary-corruption-failed-skipped/v1", "failed_or_invalid_attempts": failures,
        "record_assembly_warnings": assembly_warnings,
        "skipped_arms": skipped, "required_arms_completed": 168,
        "required_arms_missing": 0, "all_attempts_retained": True})
    start = parse_utc(CAMPAIGN_START_UTC)
    end = datetime.now(timezone.utc)
    results = json.loads((root / "CAMPAIGN_RESULTS.json").read_text())
    deviations = json.loads((root / "PROTOCOL_DEVIATIONS.json").read_text())
    summary = {"schema": "mixfp4-boundary-corruption-campaign-summary/v1",
               "campaign_start_utc": CAMPAIGN_START_UTC, "campaign_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "wall_seconds": (end - start).total_seconds(), "selected_B": 4, "partitions": 4,
               "maps": 171, "new_policies_per_model": 56, "required_gpu_policy_evaluations": 168,
               "classifications": results["classifications"], "gpu_hours": gpu["gpu_hours"],
               "maximum_concurrent_gpus": gpu["maximum_observed_concurrent_gpus"],
               "time_target_met": deviations["time_target_met"],
               "protocol_deviation_ids": [row["id"] for row in deviations["deviations"]],
               "attempt_accounting": {"run_directories": len(registry),
                                      "status_counts": dict(Counter(row["status"] for row in registry)),
                                      "failed_or_invalid": len(failures)},
               "authoritative_runs": sorted(authoritative),
               "claims_prohibited": results["claims_prohibited"]}
    atomic_json(root / "CAMPAIGN_SUMMARY.json", summary)
    return registry, gpu


def cpu_test_report(root: Path) -> dict:
    source = root / "source/NVFP4-RaZeR-main"
    py = root / "env/venv_main/bin/python"
    modules = ["boundary_common.py", "boundary_input_gate.py", "boundary_prepare.py", "boundary_protocol.py",
               "boundary_maps.py", "boundary_plans.py", "boundary_prelaunch_audit.py", "boundary_analyze.py",
               "boundary_analyze_postfreeze.py", "boundary_render.py", "boundary_finalize.py"]
    compile_run = subprocess.run([str(py), "-m", "py_compile", *[str(source / "campaign" / name) for name in modules]],
                                 capture_output=True, text=True)
    tests = subprocess.run([str(py), "-m", "pytest", "-q", "tests/test_boundary_campaign.py"], cwd=source,
                           capture_output=True, text=True)
    prelaunch = json.loads((root / "analysis/PRELAUNCH_AUDIT.json").read_text())
    lines = ["MixFP4 boundary/corruption CPU test report", f"Generated UTC: {utc()}",
             f"py_compile: {'PASS' if compile_run.returncode == 0 else 'FAIL'}", compile_run.stdout.strip(),
             compile_run.stderr.strip(), f"pytest: {'PASS' if tests.returncode == 0 else 'FAIL'}",
             tests.stdout.strip(), tests.stderr.strip(),
             f"prelaunch audit: {'PASS' if prelaunch['passed'] else 'FAIL'} ({prelaunch['maps_verified']} maps)",
             f"PRE_GPU_FREEZE.sha256: {sha256_file(root / 'PRE_GPU_FREEZE.sha256')}"]
    (root / "CPU_TEST_REPORT.txt").write_text("\n".join(line for line in lines if line) + "\n")
    return {"compile_passed": compile_run.returncode == 0, "tests_passed": tests.returncode == 0,
            "prelaunch_passed": prelaunch["passed"]}


def write_array_manifest(root: Path, eval_runs: dict[str, Path]) -> dict:
    combined = json.loads((root / "CAMPAIGN_RESULTS.json").read_text())["paired_arrays"]
    paired_cluster = []
    new_token = []
    reused_token = []
    failures = []
    for model in MODELS:
        for domain in DOMAINS:
            entry = combined[model][domain]
            path = Path(entry["path"])
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                failures.append(f"{model}/{domain}: combined paired-cluster hash mismatch")
            paired_cluster.append({"model": model, "domain": domain, **entry})
        report = json.loads((eval_runs[model] / "ppl/ppl_report.json").read_text())
        for policy, by_domain in sorted(report["evaluation"].items()):
            for domain in DOMAINS:
                token = by_domain[domain]["token_arrays"]
                actual = sha256_file(token["path"])
                if actual != token["sha256"]:
                    failures.append(f"{model}/{policy}/{domain}: new token-array hash mismatch")
                new_token.append({"model": model, "policy": policy, "domain": domain,
                                  "path": token["path"], "sha256": token["sha256"],
                                  "bytes": Path(token["path"]).stat().st_size})
        old_run = Path(MECHANISM_ROOT) / "runs" / MECHANISM_RUN[model]
        old_report = json.loads((old_run / "ppl/ppl_report.json").read_text())
        for policy in ("four_over_six", "full"):
            for domain in DOMAINS:
                token = old_report["evaluation"][policy][domain]["token_arrays"]
                actual = sha256_file(token["path"])
                if actual != token["sha256"]:
                    failures.append(f"{model}/{policy}/{domain}: reused token-array hash mismatch")
                reused_token.append({"model": model, "policy": policy, "domain": domain,
                                     "source_run": str(old_run), "path": token["path"],
                                     "sha256": token["sha256"],
                                     "bytes": Path(token["path"]).stat().st_size})
    payload = {"schema": "mixfp4-boundary-corruption-array-manifest/v1", "created_utc": utc(),
               "pairing_unit": "natural C4 document / WikiText article",
               "paired_cluster_arrays": paired_cluster,
               "new_per_token_arrays": new_token,
               "reused_anchor_per_token_arrays": reused_token,
               "counts": {"paired_cluster_npz": len(paired_cluster),
                          "new_per_token_npz": len(new_token),
                          "reused_anchor_per_token_npz": len(reused_token)},
               "hashes_verified": not failures, "failures": failures,
               "note": "Reused FourOverSix/full token arrays remain immutable in the completed mechanism campaign; this manifest pins their paths and hashes. The six combined cluster arrays are self-contained for every reported paired inference."}
    atomic_json(root / "ARRAY_MANIFEST.json", payload)
    if failures:
        raise SystemExit("array manifest verification failed: " + "; ".join(failures[:10]))
    return payload


def audit_results(root: Path, failures: list[str]) -> dict:
    boundary = json.loads((root / "BOUNDARY_RESULTS.json").read_text())
    corruption = json.loads((root / "CORRUPTION_RESULTS.json").read_text())
    bands = json.loads((root / "BAND_DEFINITIONS.json").read_text())
    cdefs = json.loads((root / "CORRUPTION_POOL_DEFINITIONS.json").read_text())
    checks = 0
    for model in MODELS:
        for domain in DOMAINS:
            path = root / f"arrays/{model}_{domain}_boundary_corruption_paired_cluster_nll.npz"
            with np.load(path) as payload:
                policies = [str(value) for value in payload["policies"]]
                position = {name: index for index, name in enumerate(policies)}
                nll = np.asarray(payload["cluster_nll_sum"], np.float64)
                tokens = np.asarray(payload["cluster_tokens"], np.float64)
            base = nll[position["four_over_six"]]
            definitions = bands["models"][model]
            rows = [row for partition in definitions["partitions"] for row in partition["bands"]]
            delta = np.stack([nll[position[row["policy"]]] - base for row in rows])
            x = np.asarray([row["score_summary"]["mean_kappa"] for row in rows], np.float64)
            groups = [partition["partition"] for partition in definitions["partitions"]
                      for _ in partition["bands"]]
            y, yboot = independent_joint(delta, tokens, f"A:{model}:{domain}:joint")
            observed = fixed_effect_slope(x, y, groups)
            samples = np.asarray([fixed_effect_slope(x, row, groups) for row in yboot])
            expected = independent_scalar(observed, samples)
            compare_result(expected, boundary["models"][model][domain]["primary"]["beta_kappa"],
                           f"{model}/{domain}/beta_kappa", failures); checks += 1
            selected = [i for i, row in enumerate(rows) if row["side"] == "selected"]
            rejected = [i for i, row in enumerate(rows) if row["side"] == "rejected"]
            expected = independent_delta(delta[selected].mean(0) - delta[rejected].mean(0), tokens,
                                         f"A:{model}:{domain}:selected-rejected")
            compare_result(expected, boundary["models"][model][domain]["primary"]["selected_minus_rejected"],
                           f"{model}/{domain}/selected-rejected", failures); checks += 1
            weak = [i for i, row in enumerate(rows) if row["side"] == "selected" and row["band"] == 4]
            near = [i for i, row in enumerate(rows) if row["side"] == "rejected" and row["band"] == 1]
            expected = independent_delta(delta[weak].mean(0) - delta[near].mean(0), tokens,
                                         f"A:{model}:{domain}:weak-near")
            compare_result(expected, boundary["models"][model][domain]["primary"]["weakest_selected_minus_nearest_rejected"],
                           f"{model}/{domain}/weak-near", failures); checks += 1

            anchor = nll[position["corruption_p000_anchor"]]
            cdelta, ps, pools = [], [], []
            for pool in range(1, 5):
                cdelta.append(np.zeros_like(tokens)); ps.append(0.0); pools.append(pool)
                for p in (0.10, 0.25, 0.50, 0.75, 1.00):
                    policy = f"corruption_near_pool{pool:02d}_p{round(100 * p):03d}"
                    cdelta.append(nll[position[policy]] - anchor); ps.append(p); pools.append(pool)
            cdelta = np.stack(cdelta)
            cy, cyboot = independent_joint(cdelta, tokens, f"B:{model}:{domain}:joint")
            beta = fixed_effect_slope(np.asarray(ps), cy, pools)
            beta_boot = np.asarray([fixed_effect_slope(np.asarray(ps), row, pools) for row in cyboot])
            expected = independent_scalar(beta, beta_boot)
            compare_result(expected, corruption["models"][model][domain]["primary"]["beta_p"],
                           f"{model}/{domain}/beta_p", failures); checks += 1
            for p, key, label in ((1.0, "p100_minus_p0", "p100"), (0.5, "p050_near_minus_p0", "p050")):
                indices = [i for i, value in enumerate(ps) if value == p]
                expected = independent_delta(cdelta[indices].mean(0), tokens, f"B:{model}:{domain}:{label}")
                compare_result(expected, corruption["models"][model][domain]["primary"][key],
                               f"{model}/{domain}/{key}", failures); checks += 1
    for payload, prefix in ((boundary, "A"), (corruption, "B")):
        for name, rows in payload["holm_families"].items():
            order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided_plus_one"])
            running = 0.0
            for rank, index in enumerate(order):
                running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_two_sided_plus_one"]))
                if not close(running, rows[index]["holm_adjusted_p"]):
                    failures.append(f"{name}: Holm mismatch at {rows[index]['endpoint']}")
                checks += 1
    bounded = 0
    for model in MODELS:
        for domain in DOMAINS:
            for partition in boundary["models"][model][domain]["construction_sensitivity"]:
                for key, row in partition["metrics"].items():
                    domain_bounds = (0, 1) if key == "pairwise_concordance" else (-1, 1)
                    if not (domain_bounds[0] <= row["ci95"][0] <= row["ci95"][1] <= domain_bounds[1]):
                        failures.append(f"{model}/{domain}/{partition['partition']}/{key}: unbounded CI")
                    bounded += 1
    return {"independent_primary_bootstrap_checks": checks,
            "bounded_correlation_interval_checks": bounded,
            "paired_arrays": 6}


def artifact_audit(root: Path, map_run: Path, eval_runs: dict[str, Path], registry: list[dict], gpu: dict,
                   cpu: dict) -> dict:
    failures: list[str] = []
    required = ["PROTOCOL.json", "PROTOCOL_DEVIATIONS.json", "ANALYSIS_CORRECTIONS.json",
                "ANALYSIS_CORRECTION_FREEZE.sha256", "POSTPROCESSING_PROVENANCE.json",
                "INPUT_PROVENANCE.json", "INPUT_GATE.json", "COVERAGE_GATE.json",
                "CORRUPTION_UNIQUENESS_GATE.json", "POWER_ANALYSIS.json", "ENDPOINT_PROMOTION.json",
                "CRITICAL_K_SUMMARY.json", "BAND_DEFINITIONS.json", "CORRUPTION_POOL_DEFINITIONS.json",
                "PRE_GPU_FREEZE.sha256", "analysis/PRELAUNCH_AUDIT.json", "MAP_MANIFEST.json",
                "ANCHOR_CHECKS.json", "P1_DISTINCTNESS_CHECKS.json", "RUN_MATRIX.csv",
                "ARRAY_MANIFEST.json",
                "RUN_REGISTRY.jsonl", "FAILED_OR_SKIPPED_RUNS.json", "GPU_POLICY_AUDIT.json",
                "BOUNDARY_RESULTS.json", "BOUNDARY_RESULTS.csv", "CORRUPTION_RESULTS.json",
                "CORRUPTION_RESULTS.csv", "CAMPAIGN_RESULTS.json", "CAMPAIGN_SUMMARY.json",
                "STATISTICAL_REPORT.md", "PRIMARY_RESULTS_TABLES.md",
                "BOUNDARY_CORRUPTION_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md", "CPU_TEST_REPORT.txt",
                "figures/critical_k_boundary_response.svg", "figures/critical_k_boundary_response.png",
                "figures/full_map_corruption.svg", "figures/full_map_corruption.png"]
    for relative in required:
        if not (root / relative).is_file():
            failures.append(f"missing required artifact {relative}")

    prefreeze_failures = []
    for line in (root / "PRE_GPU_FREEZE.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = Path(relative) if relative.startswith("/") else root / relative
        if not path.is_file() or sha256_file(path) != expected:
            prefreeze_failures.append(relative)
    failures.extend(f"pre-GPU frozen artifact changed: {path}" for path in prefreeze_failures)

    correction_freeze_failures = []
    for line in (root / "ANALYSIS_CORRECTION_FREEZE.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            correction_freeze_failures.append(relative)
    failures.extend(f"analysis-correction frozen artifact changed: {path}"
                    for path in correction_freeze_failures)
    correction = json.loads((root / "ANALYSIS_CORRECTIONS.json").read_text())
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    original_analysis = root / correction["implementation"]["frozen_original_path"]
    corrected_analysis = root / correction["implementation"]["corrected_copy_path"]
    if sha256_file(original_analysis) != protocol["code_hashes_at_freeze"]["boundary_analyze.py"]:
        failures.append("frozen original boundary_analyze.py no longer matches PROTOCOL.json")
    if sha256_file(corrected_analysis) != correction["implementation"]["corrected_copy_sha256"]:
        failures.append("corrected analysis implementation hash mismatch")
    campaign_results = json.loads((root / "CAMPAIGN_RESULTS.json").read_text())
    identity_checks = 0
    for model in MODELS:
        for domain in DOMAINS:
            observed = campaign_results["provenance"][model][domain]
            expected = correction["identity_audit"][model][domain]
            if observed["scientific_identity_sha256"] != expected["identity_sha256"]:
                failures.append(f"{model}/{domain}: analysis scientific-identity hash mismatch")
            if observed["windows"] != expected["windows"] or observed["clusters"] != expected["clusters"]:
                failures.append(f"{model}/{domain}: analysis window/cluster identity mismatch")
            identity_checks += 1

    anchor_checks = json.loads((root / "ANCHOR_CHECKS.json").read_text())
    for model, check in anchor_checks.items():
        if check.get("tile_mismatches") != 0 or not check.get("payload_equal"):
            failures.append(f"{model}: exact k=3 anchor mismatch")
    distinctness = json.loads((root / "P1_DISTINCTNESS_CHECKS.json").read_text())
    if not distinctness.get("passed"):
        failures.append("p=1 map distinctness check did not pass")
    uniqueness = json.loads((root / "CORRUPTION_UNIQUENESS_GATE.json").read_text())
    if not uniqueness.get("passed"):
        failures.append("corruption uniqueness/coverage gate did not pass")
    array_manifest = json.loads((root / "ARRAY_MANIFEST.json").read_text())
    if not array_manifest.get("hashes_verified") or array_manifest.get("counts") != {
            "paired_cluster_npz": 6, "new_per_token_npz": 336,
            "reused_anchor_per_token_npz": 12}:
        failures.append("paired/per-token array manifest is incomplete or unverified")

    manifest = json.loads((root / "MAP_MANIFEST.json").read_text())
    map_failures = []
    for entry in manifest:
        try:
            header, masks, digest = mapio.read_map(entry["path"], entry["sha256"])
            if digest != entry["sha256"] or mapio.payload_sha256(masks) != entry["mask_payload_sha256"]:
                map_failures.append(entry["path"])
            if header["policy"]["name"] != entry["policy"]:
                map_failures.append(entry["path"] + ":policy")
        except Exception as exc:
            map_failures.append(f"{entry['path']}: {exc!r}")
    failures.extend(f"map audit: {item}" for item in map_failures)

    evaluation = {}
    authoritative_source_manifests = set()
    plan_manifest = json.loads((root / "job_specs/PLAN_MANIFEST.json").read_text())
    plan_by_model = {row["model"]: json.loads(Path(row["path"]).read_text()) for row in plan_manifest["plans"]}
    map_by_key = {(entry["model"], entry["policy"]): entry for entry in manifest}
    for model, run in eval_runs.items():
        report_path = run / "ppl/ppl_report.json"
        report = json.loads(report_path.read_text())
        launch = json.loads((run / "launch_record.json").read_text())
        source_manifest_path = run / "source_manifest.txt"
        source_manifest_sha256 = sha256_file(source_manifest_path)
        authoritative_source_manifests.add(source_manifest_sha256)
        if launch.get("source_manifest_sha256") != source_manifest_sha256:
            failures.append(f"{model}: launch/source manifest hash mismatch")
        if launch.get("status") != "complete" or launch.get("invalid_gpu_cotenancy"):
            failures.append(f"{model}: authoritative GPU attempt is not clean-complete")
        expected_names = {row["name"] for row in plan_by_model[model]}
        if report.get("status") != "complete" or set(report["evaluation"]) != expected_names:
            failures.append(f"{model}: PPL policy set/status mismatch")
        if not report.get("reinstall_check", {}).get("identical"):
            failures.append(f"{model}: reinstall check failed")
        install_failures = []
        for install in report.get("installs", []):
            entry = map_by_key[(model, install["name"])]
            if install.get("map_sha256") != entry["sha256"] or not install.get("map_reloaded_for_evaluation"):
                install_failures.append(install["name"])
        npz_files = 0
        values_checked = 0
        for policy, by_domain in report["evaluation"].items():
            for domain in DOMAINS:
                token = by_domain[domain]["token_arrays"]
                path = Path(token["path"])
                if sha256_file(path) != token["sha256"]:
                    failures.append(f"{model}/{policy}/{domain}: token NPZ hash mismatch")
                with np.load(path) as payload:
                    if not np.isfinite(payload["nll"]).all():
                        failures.append(f"{model}/{policy}/{domain}: nonfinite token NLL")
                    values_checked += sum(array.size for array in payload.values())
                npz_files += 1
        failures.extend(f"{model}: bad install {name}" for name in install_failures)
        evaluation[model] = {"run_id": run.name, "ppl_report_sha256": sha256_file(report_path),
                             "source_manifest_sha256": source_manifest_sha256,
                             "policies": len(report["evaluation"]), "installs": len(report.get("installs", [])),
                             "token_npz_files": npz_files, "array_values_checked": values_checked,
                             "evaluation_manifest_sha256": report["evaluation_manifest_sha256"]}
    if len(authoritative_source_manifests) != 1:
        failures.append("authoritative GPU evaluator source manifests are not identical")

    result_audit = audit_results(root, failures)
    json_files = jsonl_files = jsonl_records = 0
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ("cache", "__pycache__", ".pytest_cache") for part in path.parts):
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
                json_files += 1
            except Exception as exc:
                failures.append(f"JSON parse failed {path}: {exc!r}")
        elif path.suffix == ".jsonl":
            jsonl_files += 1
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line:
                    continue
                try:
                    json.loads(line); jsonl_records += 1
                except Exception as exc:
                    failures.append(f"JSONL parse failed {path}:{number}: {exc!r}")
    for path in root.glob("figures/*.svg"):
        try:
            ET.parse(path)
        except Exception as exc:
            failures.append(f"SVG parse failed {path}: {exc!r}")
    try:
        from PIL import Image
        for path in root.glob("figures/*.png"):
            with Image.open(path) as image:
                image.verify()
    except Exception as exc:
        failures.append(f"PNG verification failed: {exc!r}")
    if not gpu["passed"]:
        failures.append("GPU policy audit did not pass")
    if not all(cpu.values()):
        failures.append("CPU tests did not all pass")
    failed_payload = json.loads((root / "FAILED_OR_SKIPPED_RUNS.json").read_text())
    noncomplete = [row for row in registry if row["status"] != "complete"]
    if len(failed_payload["failed_or_invalid_attempts"]) != len(noncomplete):
        failures.append("failed attempt accounting mismatch")
    audit = {"schema": "mixfp4-boundary-corruption-artifact-audit/v1", "created_utc": utc(),
             "passed": not failures, "failures": failures,
             "required_artifacts_checked": len(required), "pre_gpu_freeze_entries_checked":
             len((root / "PRE_GPU_FREEZE.sha256").read_text().splitlines()),
             "analysis_correction_freeze_entries_checked":
             len((root / "ANALYSIS_CORRECTION_FREEZE.sha256").read_text().splitlines()),
             "analysis_scientific_identity_checks": identity_checks,
             "maps_verified": len(manifest), "map_failures": map_failures,
             "evaluation_audit": evaluation, "statistics_audit": result_audit,
             "json_parse_audit": {"json_files": json_files, "jsonl_files": jsonl_files,
                                  "jsonl_records": jsonl_records},
             "gpu_policy_audit_sha256": sha256_file(root / "GPU_POLICY_AUDIT.json"),
             "cpu_test_report_sha256": sha256_file(root / "CPU_TEST_REPORT.txt"),
             "failed_attempts_audited": len(noncomplete),
             "manifest_scope": "all regular campaign files except ARTIFACT_MANIFEST.sha256, POST_SEAL_VERIFICATION.txt, cache/, tmp/, allocator/locks/, __pycache__/, .pytest_cache/, and .pyc",
             "post_seal_verification": "recorded separately in POST_SEAL_VERIFICATION.txt to avoid a checksum cycle"}
    atomic_json(root / "ARTIFACT_AUDIT.json", audit)
    if failures:
        raise SystemExit("artifact audit failed: " + "; ".join(failures[:10]))
    return audit


def manifest_paths(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in EXCLUDED_TOP or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        text = relative.as_posix()
        if text.startswith("cache/") or text.startswith("tmp/") or text.startswith("allocator/locks/"):
            continue
        if "__pycache__" in relative.parts or ".pytest_cache" in relative.parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def seal(root: Path) -> dict:
    paths = manifest_paths(root)
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths]
    (root / "ARTIFACT_MANIFEST.sha256").write_text("".join(lines))
    manifest_sha = sha256_file(root / "ARTIFACT_MANIFEST.sha256")
    failures = []
    for line in lines:
        expected, relative = line.rstrip("\n").split("  ", 1)
        if sha256_file(root / relative) != expected:
            failures.append(relative)
    sidecar = ["MixFP4 boundary/corruption post-seal verification", f"Generated UTC: {utc()}",
               f"Manifest: {root / 'ARTIFACT_MANIFEST.sha256'}", f"Manifest SHA-256: {manifest_sha}",
               f"Entries verified: {len(lines)}", f"Result: {'PASS' if not failures else 'FAIL'}"]
    if failures:
        sidecar.append("Failures: " + ", ".join(failures))
    (root / "POST_SEAL_VERIFICATION.txt").write_text("\n".join(sidecar) + "\n")
    if failures:
        raise SystemExit("post-seal checksum verification failed")
    return {"sha256": manifest_sha, "entries": len(lines), "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--map-run", required=True)
    parser.add_argument("--llama-run", required=True)
    parser.add_argument("--qwen-run", required=True)
    parser.add_argument("--mistral-run", required=True)
    parser.add_argument("--analysis-run", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    map_run = Path(args.map_run)
    eval_runs = {"llama8b": Path(args.llama_run), "qwen4b": Path(args.qwen_run),
                 "mistral7b": Path(args.mistral_run)}
    authoritative = {map_run.name, *(path.name for path in eval_runs.values()), Path(args.analysis_run).name,
                     "B10_pre_gpu_gates_attempt1"}
    registry, gpu = write_operational_outputs(root, map_run, authoritative)
    write_array_manifest(root, eval_runs)
    cpu = cpu_test_report(root)
    audit = artifact_audit(root, map_run, eval_runs, registry, gpu, cpu)
    sealed = seal(root)
    print(json.dumps({"audit_passed": audit["passed"], "manifest": sealed,
                      "gpu_hours": gpu["gpu_hours"], "runs": len(registry)}, sort_keys=True))


if __name__ == "__main__":
    main()
