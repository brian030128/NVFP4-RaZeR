#!/usr/bin/env python3
"""Independent, outcome-agnostic final audit for the frozen mechanism campaign.

This file is deliberately outside the frozen source snapshot.  It never creates
maps, chooses arms, or changes statistical rules.  It validates completed run
evidence and independently recomputes reported point estimates and Holm values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROTOCOL_SHA256 = "9e7c3d1dcb6199ab9bbedf9e0947f6198304f26466f2fd2c58e5533fb747926c"
POWER_ANALYSIS_SHA256 = "9310ca2af0ef707cb83cd1874677b49814a8861c2e30db07746304e1d56a1da4"
REPORT_RENDERER_SHA256 = "920d86d02df16de7426481d46be3427b9e8ca1b4c1fcdcf543c63de397f3505a"
REPORTING_PLAN_SHA256 = "527ec0cb608b3b3f5bf7ed03b9aa02ff72aff9ba939c7a65ea0072cd5580b475"
EXPECTED_CODE_HASHES = {
    "mechanism_maps.py": "8c7b060682a7d6882ab4a1a39debc9448629656d784d1ad2275e62f2efebd336",
    "mechanism_analyze.py": "5352a7d726445c508344b686866858e7d6f3fdd2ffc4e47f859793e3c9fd4bc5",
    "mechanism_power.py": "f0f614e6070da271d6d2990d78402a9a7116f423d13ad9bd333f3b9239a9414f",
    "evaluate_ppl.py": "de657a20cc33da25a8cad3bf26bd0b08b195bfb8c20d1ed0ed41ffd93122ab4b",
    "launcher.py": "b3b0f3f003f121b5bed4962dc3c4a32974d79c1961b50292eb9b2054aa2df872",
    "mapio.py": "c8627a8345e70cb1727082f9ec26009d4d6eaa18a8cfa01d21b6564feed53afe",
    "stats.py": "2d554a8fc38faa3b2b179c425faca254bfd2bf55c8186219f4b4ff6c03315cba",
}
MODEL_RUN_EXPECTATIONS = {
    "llama8b": {"policies": 17, "windows": {"wiki": 141, "c4": 256},
                "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"},
    "qwen4b": {"policies": 17, "windows": {"wiki": 146, "c4": 256},
               "revision": "1cfa9a7208912126459214e8b04321603b3df60c"},
    "mistral7b": {"policies": 13, "windows": {"wiki": 163, "c4": 256},
                  "revision": "caa1feb0e54d415e2df31207e5f4e273e33509b1"},
}
REQUIRED_TOP_LEVEL = [
    "MECHANISM_PROTOCOL.json",
    "INPUT_PROVENANCE.json",
    "RUN_REGISTRY.jsonl",
    "FAILED_OR_SKIPPED_RUNS.json",
    "MAP_MANIFEST.json",
    "MISTRAL_PROMOTION_GATE.json",
    "MECHANISM_RESULTS.json",
    "MECHANISM_RESULTS.csv",
    "SCORE_MARGIN_ANALYSIS.json",
    "CE_KL_VETO_ANALYSIS.json",
    "MODULE_INTERACTION_ANALYSIS.json",
    "STATISTICAL_REPORT.md",
    "PRIMARY_RESULTS_TABLES.md",
    "MECHANISM_VERDICT.md",
    "NEXT_STEP_RECOMMENDATION.md",
    "CPU_TEST_REPORT.txt",
    "GPU_POLICY_AUDIT.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def assert_close(a: float, b: float, *, atol: float = 2e-12, label: str = "") -> None:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a) - float(b)) > atol:
        raise AssertionError(f"{label}: {a!r} != {b!r} within {atol}")


def _seed_for(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def verify_bootstrap_delta(reported: dict, delta_sum: np.ndarray, token_count: np.ndarray, label: str) -> None:
    estimate = float(delta_sum.sum() / token_count.sum())
    centered = delta_sum - estimate * token_count
    rng = np.random.default_rng(_seed_for(20260917, label))
    indices = rng.integers(0, len(delta_sum), size=(10_000, len(delta_sum)))
    noise = centered[indices].sum(axis=1) / token_count[indices].sum(axis=1)
    boot = estimate + noise
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(estimate)).sum())) / 10_001
    rates = delta_sum / token_count
    assert_close(reported["estimate"], estimate, label=f"{label}:estimate")
    assert_close(reported["ci95"][0], lo, label=f"{label}:ci-low")
    assert_close(reported["ci95"][1], hi, label=f"{label}:ci-high")
    assert_close(reported["p_two_sided_plus_one"], p, atol=2e-15, label=f"{label}:p")
    assert_close(reported["relative_ppl_change"], math.expm1(estimate), label=f"{label}:relative-ppl")
    if int(reported["clusters"]) != len(delta_sum) or int(reported["tokens"]) != int(token_count.sum()):
        raise AssertionError(f"{label}: cluster/token count")
    assert_close(reported["cluster_regression_probability"], np.mean(rates > 0), label=f"{label}:regression-prob")
    assert_close(reported["cluster_worst"], np.max(rates), label=f"{label}:worst")
    for q in (90, 95, 99):
        assert_close(reported["cluster_quantiles"][f"q{q}"], np.quantile(rates, q / 100),
                     label=f"{label}:q{q}")


def verify_bootstrap_scalar_pair(reported: dict, rate_a: np.ndarray, rate_b: np.ndarray,
                                 functional, label: str) -> None:
    observed = float(functional(rate_a) - functional(rate_b))
    rng = np.random.default_rng(_seed_for(20260917, label))
    indices = rng.integers(0, len(rate_a), size=(10_000, len(rate_a)))
    values = np.empty(10_000, np.float64)
    for start in range(0, 10_000, 500):
        subset = indices[start:start + 500]
        values[start:start + len(subset)] = np.asarray(
            [functional(rate_a[row]) - functional(rate_b[row]) for row in subset])
    centered = values - values.mean()
    lo, hi = np.percentile(observed + centered, [2.5, 97.5])
    p = (1 + int((np.abs(centered) >= abs(observed)).sum())) / 10_001
    assert_close(reported["estimate"], observed, label=f"{label}:estimate")
    assert_close(reported["ci95"][0], lo, label=f"{label}:ci-low")
    assert_close(reported["ci95"][1], hi, label=f"{label}:ci-high")
    assert_close(reported["p_two_sided_plus_one"], p, atol=2e-15, label=f"{label}:p")


def validate_eval_run(root: Path, model: str, run_id: str) -> dict:
    expected = MODEL_RUN_EXPECTATIONS[model]
    run = root / "runs" / run_id
    report = load_json(run / "ppl" / "ppl_report.json")
    plan_path = root / "job_specs" / f"{model}_mechanism_full.json"
    plan = load_json(plan_path)
    names = [p["name"] for p in plan]
    if report["status"] != "complete" or set(report["evaluation"]) != set(names):
        raise AssertionError(f"{run_id}: incomplete or policy mismatch")
    if len(names) != expected["policies"] or len(report["installs"]) != len(names):
        raise AssertionError(f"{run_id}: wrong policy count")
    if (report["freeze_sha256"] != PROTOCOL_SHA256 or report["windows"] != expected["windows"] or
            report["teacher"] != "none" or report["spec"]["revision"] != expected["revision"]):
        raise AssertionError(f"{run_id}: freeze or window mismatch")
    array_keys = {"nll", "correct", "entropy", "confidence", "argmax"}
    array_files = 0
    array_values = 0
    map_installs = 0
    for name, install in zip(names, report["installs"]):
        if install["name"] != name:
            raise AssertionError(f"{run_id}: install order mismatch")
        if install.get("map_path"):
            map_path = Path(install["map_path"])
            if sha256(map_path) != install["map_sha256"] or install.get("map_reloaded_for_evaluation") is not True:
                raise AssertionError(f"{run_id}:{name}: map reload/hash mismatch")
            if not 0 <= int(install["selected_tiles"]) <= int(install["total_tiles"]):
                raise AssertionError(f"{run_id}:{name}: tile accounting invalid")
            if install["type_block"] != [16, 64] or install["activation"] != "four_over_six_rows":
                raise AssertionError(f"{run_id}:{name}: format/protocol drift")
            map_installs += 1
        for domain, expected_windows in expected["windows"].items():
            ev = report["evaluation"][name][domain]
            if len(ev["windows"]) != expected_windows:
                raise AssertionError(f"{run_id}:{name}:{domain}: window count")
            tokens = sum(int(w["tokens"]) for w in ev["windows"])
            if tokens != int(ev["tokens"]):
                raise AssertionError(f"{run_id}:{name}:{domain}: token accounting")
            for row in ev["windows"]:
                for key in ("nll_sum", "nll_mean", "token_accuracy", "entropy_mean", "confidence_mean"):
                    if not math.isfinite(float(row[key])):
                        raise AssertionError(f"{run_id}:{name}:{domain}: nonfinite {key}")
            arrays = Path(ev["token_arrays"]["path"])
            if sha256(arrays) != ev["token_arrays"]["sha256"]:
                raise AssertionError(f"{run_id}:{name}:{domain}: token array hash")
            with np.load(arrays, allow_pickle=False) as z:
                if set(z.files) != array_keys:
                    raise AssertionError(f"{run_id}:{name}:{domain}: token array keys")
                for key in z.files:
                    a = z[key]
                    if a.ndim != 1 or a.size != tokens:
                        raise AssertionError(f"{run_id}:{name}:{domain}:{key}: shape")
                    if a.dtype.kind == "f" and not np.isfinite(a).all():
                        raise AssertionError(f"{run_id}:{name}:{domain}:{key}: nonfinite")
                    array_values += int(a.size)
            array_files += 1
    reinstall = report["reinstall_check"]
    if reinstall["identical"] is not True or reinstall["checksum_equal"] is not True:
        raise AssertionError(f"{run_id}: reinstall check")
    launch = load_json(run / "launch_record.json")
    job = load_json(run / "job_status.json")
    validation = load_json(run / "run_record_validation.json")
    run_record = load_json(run / "run_record.json")
    if launch["status"] != "complete" or launch["exit_code"] != 0 or launch["invalid_gpu_cotenancy"]:
        raise AssertionError(f"{run_id}: launcher status")
    if job["status"] != "complete" or validation != {"valid": True, "errors": []}:
        raise AssertionError(f"{run_id}: job/schema status")
    if not all(policy["scale_block"] == 16 for policy in run_record["policies"]):
        raise AssertionError(f"{run_id}: K16 scale-block drift")
    if not launch["host_preflight_before"]["passed"] or not launch["host_preflight_after"]["passed"]:
        raise AssertionError(f"{run_id}: host pre/postflight")
    if launch["sidecar"]["invalid_reasons"]:
        raise AssertionError(f"{run_id}: sidecar invalid")
    if not all(p["passed"] for p in job["preflight_records"]):
        raise AssertionError(f"{run_id}: in-container preflight")
    leased = set(launch["leased_uuids"])
    samples = 0
    with (run / "gpu_monitor.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            samples += 1
            if set(sample["gpus"]) != leased:
                raise AssertionError(f"{run_id}: sidecar UUID mismatch")
            for proc in sample["processes"]:
                if not (proc["gpu_uuid"] in leased and proc["owner"] == "JaaaaaA_l" and
                        int(proc["uid"]) == 1011 and proc["in_container"] is True):
                    raise AssertionError(f"{run_id}: foreign sidecar process {proc}")
    return {
        "run_id": run_id,
        "model": model,
        "valid": True,
        "policies": len(names),
        "windows": expected["windows"],
        "npz_files": array_files,
        "array_values_checked": array_values,
        "map_installs_checked": map_installs,
        "gpu_monitor_samples": samples,
        "ppl_report_sha256": sha256(run / "ppl" / "ppl_report.json"),
        "run_record_sha256": sha256(run / "run_record.json"),
        "wall_seconds": launch["wall_seconds"],
        "gpu_hours": launch["gpu_hours"],
        "model_revision": expected["revision"],
        "execution_scope": "weight-only N16K64 map mixing with causal per-token FourOverSix activations; fake-quantized/dequantized BF16 quality evaluation",
    }


def validate_llama_screen(root: Path, run_id: str = "V20_screen_llama8b_attempt2") -> dict:
    run = root / "runs" / run_id
    report = load_json(run / "ppl" / "ppl_report.json")
    plan = load_json(root / "job_specs" / "llama8b_mechanism_full.json")
    names = [p["name"] for p in plan]
    expected_windows = {"wiki": 94, "c4": 105}
    if (report["status"] != "complete" or report["evaluation_stage"] != "screening" or
            report["windows"] != expected_windows or set(report["evaluation"]) != set(names) or
            len(report["installs"]) != 17 or report["teacher"] != "none" or
            report["freeze_sha256"] != PROTOCOL_SHA256):
        raise AssertionError(f"{run_id}: screening report")
    finite_rows = 0
    map_installs = 0
    for name, install in zip(names, report["installs"]):
        if install["name"] != name:
            raise AssertionError(f"{run_id}: install order")
        if install.get("map_path"):
            if (sha256(Path(install["map_path"])) != install["map_sha256"] or
                    not install["map_reloaded_for_evaluation"]):
                raise AssertionError(f"{run_id}:{name}: map")
            map_installs += 1
        for domain, windows in expected_windows.items():
            ev = report["evaluation"][name][domain]
            if len(ev["windows"]) != windows or "token_arrays" in ev:
                raise AssertionError(f"{run_id}:{name}:{domain}: screen window/array")
            for row in ev["windows"]:
                if not all(math.isfinite(float(row[key])) for key in
                           ("nll_sum", "nll_mean", "token_accuracy", "entropy_mean", "confidence_mean")):
                    raise AssertionError(f"{run_id}:{name}:{domain}: nonfinite")
                finite_rows += 1
    if not report["reinstall_check"]["identical"] or not report["reinstall_check"]["checksum_equal"]:
        raise AssertionError(f"{run_id}: reinstall")
    launch = load_json(run / "launch_record.json")
    job = load_json(run / "job_status.json")
    validation = load_json(run / "run_record_validation.json")
    if (launch["status"] != "complete" or launch["exit_code"] != 0 or
            launch["invalid_gpu_cotenancy"] or launch["sidecar"]["invalid_reasons"] or
            not launch["host_preflight_before"]["passed"] or not launch["host_preflight_after"]["passed"] or
            job["status"] != "complete" or len(job["preflight_records"]) != 21 or
            not all(x["passed"] for x in job["preflight_records"]) or
            validation != {"valid": True, "errors": []}):
        raise AssertionError(f"{run_id}: run/GPU record")
    leased = set(launch["leased_uuids"])
    samples = 0
    for line in (run / "gpu_monitor.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line); samples += 1
        if set(sample["gpus"]) != leased:
            raise AssertionError(f"{run_id}: monitor UUID")
        for proc in sample["processes"]:
            if proc["owner"] != "JaaaaaA_l" or int(proc["uid"]) != 1011 or not proc["in_container"]:
                raise AssertionError(f"{run_id}: foreign process")
    return {
        "run_id": run_id,
        "model": "llama8b",
        "stage": "screening-only; never used to select or discard primary arms",
        "valid": True,
        "policies": 17,
        "windows": expected_windows,
        "finite_window_rows": finite_rows,
        "map_installs_checked": map_installs,
        "gpu_monitor_samples": samples,
        "ppl_report_sha256": sha256(run / "ppl" / "ppl_report.json"),
        "run_record_sha256": sha256(run / "run_record.json"),
        "gpu_hours": launch["gpu_hours"],
    }


def validate_screen_full_consistency(root: Path, full_run_id: str) -> dict:
    screen_dir = root / "runs" / "V20_screen_llama8b_attempt2" / "ppl"
    full_dir = root / "runs" / full_run_id / "ppl"
    screen = load_json(screen_dir / "ppl_report.json")
    full = load_json(full_dir / "ppl_report.json")
    power = load_json(root / "POWER_ANALYSIS.json")
    fields = ("nll_sum", "nll_mean", "token_accuracy", "entropy_mean", "confidence_mean", "tokens")
    rows = 0
    values = 0
    for domain in ("wiki", "c4"):
        indices = power["models"]["llama8b"]["corpora"][domain]["selected_window_indices"]
        screen_meta = load_json(screen_dir / f"windows_{domain}.json")
        full_meta = load_json(full_dir / f"windows_{domain}.json")
        if screen_meta["token_sha256"] != [full_meta["token_sha256"][i] for i in indices]:
            raise AssertionError(f"screen/full token hash mismatch: {domain}")
        for policy in screen["evaluation"]:
            a = screen["evaluation"][policy][domain]["windows"]
            b = [full["evaluation"][policy][domain]["windows"][i] for i in indices]
            if len(a) != len(b):
                raise AssertionError(f"screen/full row count: {policy}:{domain}")
            for left, right in zip(a, b):
                for field in fields:
                    if left[field] != right[field]:
                        raise AssertionError(f"screen/full non-identical {field}: {policy}:{domain}")
                    values += 1
                rows += 1
    return {"valid": True, "bitwise_equal_reported_values": True, "policies": len(screen["evaluation"]),
            "screen_window_rows_checked": rows, "scalar_values_checked": values,
            "power_analysis_sha256": sha256(root / "POWER_ANALYSIS.json")}


def validate_power_freeze(root: Path) -> dict:
    path = root / "POWER_ANALYSIS.json"
    report = load_json(path)
    if (sha256(path) != POWER_ANALYSIS_SHA256 or report["seed"] != 20260917 or
            report["bootstrap_replicates"] != 10_000 or report["effects_delta_log_ppl"] != [0.001, 0.003, 0.005]):
        raise AssertionError("power-analysis freeze")
    expected = {
        "llama8b": {"wiki": (32, 94, False), "c4": (96, 105, False)},
        "qwen4b": {"wiki": (53, 146, True), "c4": (231, 256, True)},
    }
    summary = {}
    for model, corpora in expected.items():
        summary[model] = {}
        for corpus, triple in corpora.items():
            value = report["models"][model]["corpora"][corpus]
            got = (value["selected_cluster_count"], value["selected_window_count"], value["descriptive_only"])
            if got != triple or value["calibration_document_overlap"] != 0:
                raise AssertionError(f"power selection drift: {model}:{corpus}")
            if len(value["selected_cluster_ids"]) != triple[0] or len(value["selected_window_indices"]) != triple[1]:
                raise AssertionError(f"power selected IDs/indices: {model}:{corpus}")
            summary[model][corpus] = {"clusters": triple[0], "windows": triple[1],
                                      "descriptive_only": triple[2], "calibration_overlap": 0}
    return {"valid": True, "sha256": POWER_ANALYSIS_SHA256, "selection": summary,
            "screening_never_selects_primary_arms": True}


def validate_maps(root: Path) -> dict:
    manifest = load_json(root / "MAP_MANIFEST.json")
    by_model = Counter()
    anchors = {}
    for entry in manifest:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise AssertionError(f"map hash mismatch: {path}")
        if not 0 <= int(entry["selected_tiles"]) <= int(entry["total_tiles"]):
            raise AssertionError(f"map tile accounting: {path}")
        if entry["type_block"] != [16, 64]:
            raise AssertionError(f"map type block: {path}")
        by_model[entry["model"]] += 1
        if entry["policy"] == "full":
            anchors[entry["model"]] = {
                "map_sha256": entry["sha256"],
                "selected_tiles": entry["selected_tiles"],
                "total_tiles": entry["total_tiles"],
            }
    expected_anchors = {
        "llama8b": ("ebb1122f74d14784a4a5040cbe92e74a51ad3db80ff391b2ca454c72d2724ab2", 1781, 6815744),
        "qwen4b": ("c43d9ba98ed6d1c25a185e81ce28c2930a520f0e26356f5defb6025743f95bce", 4077, 3548160),
        "mistral7b": ("7061a82ec353f477ab2e5107b65c92916e3e09067729abb620f37f4b01809938", 4179, 6815744),
    }
    for model, (digest, selected, total) in expected_anchors.items():
        got = anchors[model]
        if (got["map_sha256"], got["selected_tiles"], got["total_tiles"]) != (digest, selected, total):
            raise AssertionError(f"{model}: mechanism full-map anchor mismatch")
    if sum(by_model.values()) != 48:
        raise AssertionError("wrong mechanism map count")
    definitions_path = root / "analysis" / "GROUP_DEFINITIONS.json"
    definitions = load_json(definitions_path)
    composition = {}
    for model, value in definitions.items():
        anchor = value["anchor_reproduction"]
        ranking = value["ranking"]
        interaction = value["interaction"]
        if not anchor["passed"] or anchor["tile_mismatches"] != 0:
            raise AssertionError(f"{model}: source anchor reproduction")
        if (not ranking["groups_disjoint"] or
                sum(ranking["per_module_allocation"].values()) != ranking["actual_tiles_each"]):
            raise AssertionError(f"{model}: ranking composition/disjointness")
        if (not interaction["full_union_exact"] or
                interaction["attention_tiles"] != interaction["random_attention_tiles"] or
                interaction["mlp_tiles"] != interaction["random_mlp_tiles"]):
            raise AssertionError(f"{model}: interaction composition")
        veto = {}
        for label, block in value["veto"].items():
            if block["bins"] != 5 or block["actual_tiles"] != block["random_tiles"] or block["actual_tiles"] <= 0:
                raise AssertionError(f"{model}:{label}: veto composition")
            veto[label] = {"actual_tiles": block["actual_tiles"], "random_tiles": block["random_tiles"],
                           "bins": block["bins"], "reductions": len(block["reductions"])}
        composition[model] = {
            "anchor_tile_mismatches": 0,
            "ranking_tiles_each": ranking["actual_tiles_each"],
            "ranking_groups_disjoint": True,
            "veto": veto,
            "interaction": interaction,
        }
    return {"valid": True, "maps": len(manifest), "by_model": dict(by_model), "full_maps": anchors,
            "composition_invariants": composition,
            "group_definitions_sha256": sha256(definitions_path),
            "manifest_sha256": sha256(root / "MAP_MANIFEST.json")}


def validate_immutable_inputs(root: Path) -> dict:
    provenance = load_json(root / "INPUT_PROVENANCE.json")
    if not provenance["gate_passed"] or provenance["score_gate"]["regeneration_required"]:
        raise AssertionError("hour-0 input/score gate drift")
    handoff = provenance["handoff"]
    outer = Path(handoff["outer_zip"])
    if sha256(outer) != handoff["outer_sha256"] or handoff["outer_sha256"] != handoff["outer_expected_sha256"]:
        raise AssertionError("outer handoff ZIP hash drift")
    source = provenance["source"]
    source_archive = Path(source["archive"])
    if sha256(source_archive) != source["archive_sha256"]:
        raise AssertionError("source archive hash drift")
    readonly_root = Path(handoff["extracted_read_only_root"])
    readonly_checks = [readonly_root, readonly_root / "README_FIRST.md", readonly_root / "RESEARCH_PROTOCOL.md",
                       readonly_root / "EVIDENCE_LINEAGE.md", source_archive]
    if any(path.stat().st_mode & 0o222 for path in readonly_checks):
        raise AssertionError("supplied extracted evidence is not read-only")
    checked = []

    def visit(value, label: str) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                digest = sha256(path)
                if digest != value["sha256"] or (value.get("expected_sha256") and digest != value["expected_sha256"]):
                    raise AssertionError(f"immutable input hash drift: {label}")
                checked.append({"logical_input": label, "sha256": digest, "bytes": path.stat().st_size})
            for key, child in value.items():
                visit(child, f"{label}.{key}" if label else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")

    visit(provenance["score_and_map_inputs"], "score_and_map_inputs")
    for name, cache in provenance["caches"].items():
        if not Path(cache["path"]).exists() or not cache["present"]:
            raise AssertionError(f"pinned cache missing at final audit: {name}")
    return {
        "valid": True,
        "input_provenance_sha256": sha256(root / "INPUT_PROVENANCE.json"),
        "outer_handoff_zip_sha256": handoff["outer_sha256"],
        "source_archive_sha256": source["archive_sha256"],
        "read_only_evidence_paths_checked": [str(path) for path in readonly_checks],
        "score_map_files_rehashed": len(checked),
        "score_map_inputs": checked,
        "pinned_caches_present": sorted(provenance["caches"]),
    }


def _policy_index(policies: np.ndarray) -> dict[str, int]:
    return {str(v): i for i, v in enumerate(policies.tolist())}


def _estimate(nll: np.ndarray, tokens: np.ndarray, at: dict[str, int], a: str, b: str) -> float:
    return float((nll[at[a]] - nll[at[b]]).sum() / tokens.sum())


def _delta(nll: np.ndarray, at: dict[str, int], a: str, b: str) -> np.ndarray:
    return nll[at[a]] - nll[at[b]]


def validate_statistics(root: Path) -> dict:
    results = load_json(root / "MECHANISM_RESULTS.json")
    if results["protocol_sha256"] != PROTOCOL_SHA256 or results["bootstrap_replicates"] != 10_000:
        raise AssertionError("results freeze/bootstrap mismatch")
    if set(results["models"]) != {"llama8b", "qwen4b", "mistral7b"}:
        raise AssertionError("results model set")
    checks = 0
    bootstrap_checks = 0
    paired_hashes = {}
    for model, model_result in results["models"].items():
        for domain, domain_result in model_result["domains"].items():
            paired = Path(domain_result["paired_array"]["path"])
            digest = sha256(paired)
            if digest != domain_result["paired_array"]["sha256"]:
                raise AssertionError(f"{model}:{domain}: paired array hash")
            paired_hashes[f"{model}:{domain}"] = digest
            with np.load(paired, allow_pickle=False) as z:
                policies = z["policies"]
                tokens = z["cluster_tokens"].astype(np.float64)
                nll = z["cluster_nll_sum"].astype(np.float64)
                ids = z["cluster_ids"]
            if nll.shape != (len(policies), len(ids)) or tokens.shape != (len(ids),) or np.any(tokens <= 0):
                raise AssertionError(f"{model}:{domain}: paired shape/tokens")
            if not np.isfinite(nll).all() or not np.isfinite(tokens).all():
                raise AssertionError(f"{model}:{domain}: paired nonfinite")
            at = _policy_index(policies)
            for policy, reported_ppl in domain_result["absolute_ppl"].items():
                expected = math.exp(float(nll[at[policy]].sum() / tokens.sum()))
                assert_close(expected, reported_ppl, atol=2e-12, label=f"{model}:{domain}:{policy}:ppl")
                checks += 1
            groups = ("strongest", "weakest") if model == "mistral7b" else ("strongest", "weakest", "random")
            group_only = {}
            marginal = {}
            for group in groups:
                go_delta = _delta(nll, at, f"group_only_{group}", "four_over_six")
                marginal_delta = _delta(nll, at, "full", f"full_minus_{group}")
                group_only[group] = float(go_delta.sum() / tokens.sum())
                marginal[group] = float(marginal_delta.sum() / tokens.sum())
                go_result = domain_result["ranking"][group]["group_only_vs_baseline"]
                marginal_result = domain_result["ranking"][group]["full_context_marginal"]
                assert_close(group_only[group], go_result["estimate"],
                             label=f"{model}:{domain}:{group}:group-only")
                assert_close(marginal[group], marginal_result["estimate"],
                             label=f"{model}:{domain}:{group}:marginal")
                verify_bootstrap_delta(go_result, go_delta, tokens, f"{model}:{domain}:A:group:{group}")
                verify_bootstrap_delta(marginal_result, marginal_delta, tokens, f"{model}:{domain}:A:marginal:{group}")
                bootstrap_checks += 2
                checks += 2
            pairs = [("strongest", "weakest")]
            if model != "mistral7b":
                pairs += [("strongest", "random"), ("weakest", "random")]
            for x, y in pairs:
                group_pair = (_delta(nll, at, f"group_only_{x}", "four_over_six") -
                              _delta(nll, at, f"group_only_{y}", "four_over_six"))
                marginal_pair = (_delta(nll, at, "full", f"full_minus_{x}") -
                                 _delta(nll, at, "full", f"full_minus_{y}"))
                group_result = domain_result["ranking"][f"group_only:{x}-{y}"]
                marginal_result = domain_result["ranking"][f"full_context_marginal:{x}-{y}"]
                assert_close(group_only[x] - group_only[y], group_result["estimate"],
                             label=f"{model}:{domain}:group-only:{x}-{y}")
                assert_close(marginal[x] - marginal[y], marginal_result["estimate"],
                             label=f"{model}:{domain}:marginal:{x}-{y}")
                verify_bootstrap_delta(group_result, group_pair, tokens,
                                       f"{model}:{domain}:A:group_only:{x}-{y}")
                verify_bootstrap_delta(marginal_result, marginal_pair, tokens,
                                       f"{model}:{domain}:A:full_context_marginal:{x}-{y}")
                bootstrap_checks += 2
                checks += 2
            veto_defs = [("kl_vetoed_ce_approved", "full_plus_kl_vetoed_ce_approved",
                          "full_plus_kl_vetoed_matched_random")]
            if model != "mistral7b":
                veto_defs.append(("ce_vetoed_kl_approved", "full_plus_ce_vetoed_kl_approved",
                                  "full_plus_ce_vetoed_matched_random"))
            for label, actual, random in veto_defs:
                da_sum = _delta(nll, at, actual, "full")
                dr_sum = _delta(nll, at, random, "full")
                dar_sum = da_sum - dr_sum
                actual_full = float(da_sum.sum() / tokens.sum())
                random_full = float(dr_sum.sum() / tokens.sum())
                actual_random = float(dar_sum.sum() / tokens.sum())
                block = domain_result["veto"][label]
                assert_close(actual_full, block["actual_vs_full"]["estimate"], label=f"{model}:{domain}:{label}:full")
                assert_close(random_full, block["matched_random_vs_full"]["estimate"], label=f"{model}:{domain}:{label}:random")
                assert_close(actual_random, block["actual_vs_matched_random"]["estimate"], label=f"{model}:{domain}:{label}:diff")
                verify_bootstrap_delta(block["actual_vs_full"], da_sum, tokens,
                                       f"{model}:{domain}:B:{label}:actual-full")
                verify_bootstrap_delta(block["matched_random_vs_full"], dr_sum, tokens,
                                       f"{model}:{domain}:B:{label}:random-full")
                verify_bootstrap_delta(block["actual_vs_matched_random"], dar_sum, tokens,
                                       f"{model}:{domain}:B:{label}:actual-random")
                bootstrap_checks += 3
                da = da_sum / tokens
                dr = dr_sum / tokens
                tail_expected = {
                    "q90_actual_minus_random": np.quantile(da, .90) - np.quantile(dr, .90),
                    "q95_actual_minus_random": np.quantile(da, .95) - np.quantile(dr, .95),
                    "q99_actual_minus_random": np.quantile(da, .99) - np.quantile(dr, .99),
                    "worst_actual_minus_random": np.max(da) - np.max(dr),
                    "regression_probability_actual_minus_random": np.mean(da > 0) - np.mean(dr > 0),
                }
                for endpoint, expected_value in tail_expected.items():
                    assert_close(expected_value, block["tails"][endpoint]["estimate"],
                                 label=f"{model}:{domain}:{label}:{endpoint}")
                    checks += 1
                tail_specs = (
                    ("q90_actual_minus_random", lambda x: np.quantile(x, .90), "q90"),
                    ("q95_actual_minus_random", lambda x: np.quantile(x, .95), "q95"),
                    ("q99_actual_minus_random", lambda x: np.quantile(x, .99), "q99"),
                    ("worst_actual_minus_random", np.max, "worst"),
                    ("regression_probability_actual_minus_random", lambda x: np.mean(x > 0), "prob"),
                )
                for endpoint, functional, seed_label in tail_specs:
                    verify_bootstrap_scalar_pair(block["tails"][endpoint], da, dr, functional,
                                                 f"{model}:{domain}:B:{label}:{seed_label}")
                    bootstrap_checks += 1
                checks += 3
            actual_i_sum = (_delta(nll, at, "full", "four_over_six") -
                            _delta(nll, at, "attention_only", "four_over_six") -
                            _delta(nll, at, "mlp_only", "four_over_six"))
            random_i_sum = (_delta(nll, at, "matched_random_union", "four_over_six") -
                            _delta(nll, at, "matched_random_attention", "four_over_six") -
                            _delta(nll, at, "matched_random_mlp", "four_over_six"))
            actual_i = float(actual_i_sum.sum() / tokens.sum())
            random_i = float(random_i_sum.sum() / tokens.sum())
            interaction = domain_result["interaction"]
            assert_close(actual_i, interaction["actual_residual"]["estimate"], label=f"{model}:{domain}:interaction")
            assert_close(random_i, interaction["matched_random_residual"]["estimate"], label=f"{model}:{domain}:random-interaction")
            assert_close(actual_i - random_i, interaction["actual_minus_matched_random"]["estimate"],
                         label=f"{model}:{domain}:interaction-diff")
            verify_bootstrap_delta(interaction["actual_residual"], actual_i_sum, tokens,
                                   f"{model}:{domain}:C:actual")
            verify_bootstrap_delta(interaction["matched_random_residual"], random_i_sum, tokens,
                                   f"{model}:{domain}:C:random")
            verify_bootstrap_delta(interaction["actual_minus_matched_random"], actual_i_sum - random_i_sum,
                                   tokens, f"{model}:{domain}:C:actual-random")
            bootstrap_checks += 3
            checks += 3
    family_counts = {}
    for family, rows in results["holm_families"].items():
        family_counts[family] = len(rows)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided_plus_one"])
        running = 0.0
        for rank, index in enumerate(order):
            row = rows[index]
            p = float(row["p_two_sided_plus_one"])
            if not 1 / 10001 <= p <= 1:
                raise AssertionError(f"{family}: invalid plus-one p")
            running = max(running, min(1.0, (len(rows) - rank) * p))
            assert_close(running, row["holm_adjusted_p"], label=f"{family}:Holm", atol=2e-15)
            if bool(running < .05) != row["holm_reject_0p05"]:
                raise AssertionError(f"{family}: Holm decision")
            if row["bootstrap_replicates"] != 10_000:
                raise AssertionError(f"{family}: bootstrap count")
            lo, hi = row["ci95"]
            if not all(math.isfinite(float(v)) for v in (lo, hi, row["estimate"])) or lo > hi:
                raise AssertionError(f"{family}: CI")
            checks += 1
    with (root / "MECHANISM_RESULTS.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != 70:
        raise AssertionError(f"results CSV row count {len(csv_rows)} != 70")
    for row in csv_rows:
        float(row["estimate"]); float(row["p_two_sided_plus_one"]); float(row["relative_ppl_change"])
        json.loads(row["ci95"].replace("'", '"'))
    a = [r for r in results["holm_families"]["A_development"]
         if r["endpoint"] == "full_context_marginal:strongest-weakest"]
    a_group = [r for r in results["holm_families"]["A_development"]
               if r["endpoint"] == "group_only:strongest-weakest"]
    ranking_pass = (len(a) == 4 and all(r["estimate"] < 0 for r in a) and
                    sum(r["holm_reject_0p05"] for r in a) >= 3 and
                    sum(r["estimate"] < 0 for r in a_group) >= 3)
    veto_classes = {}
    for label in ("kl_vetoed_ce_approved", "ce_vetoed_kl_approved"):
        full = [r for r in results["holm_families"]["B_development"]
                if r["endpoint"] == f"{label}:mean_actual_minus_full"]
        random = [r for r in results["holm_families"]["B_development"]
                  if r["endpoint"] == f"{label}:mean_actual_minus_random"]
        veto_classes[label] = (len(full) == 4 and len(random) == 4 and
                               sum(r["estimate"] > 0 for r in full) >= 3 and
                               sum(r["estimate"] > 0 for r in random) >= 3 and
                               sum(r["holm_reject_0p05"] for r in full + random) >= 2)
    interaction_rows = [r for r in results["holm_families"]["C_development"]
                        if r["endpoint"] == "actual_residual"]
    signs = [int(np.sign(r["estimate"])) for r in interaction_rows]
    interaction_pass = (len(interaction_rows) == 4 and
                        (signs.count(1) == 4 or signs.count(-1) == 4) and
                        sum(r["holm_reject_0p05"] for r in interaction_rows) >= 3)
    independent_gates = {
        "ranking_not_calibration": ranking_pass,
        "ce_kl_veto": all(veto_classes.values()),
        "attention_mlp_interaction": interaction_pass,
    }
    reported_gates = {k: bool(v["passed"]) for k, v in results["hypothesis_classification"].items()}
    if independent_gates != reported_gates:
        raise AssertionError(f"hypothesis classification mismatch: {independent_gates} != {reported_gates}")
    actual_classes = {k: ("passed" if v else "not_passed") for k, v in independent_gates.items()}
    return {
        "valid": True,
        "independent_point_estimate_checks": checks,
        "independent_10000_replicate_bootstrap_checks": bootstrap_checks,
        "paired_array_hashes": paired_hashes,
        "holm_family_counts": family_counts,
        "csv_rows": len(csv_rows),
        "hypothesis_gates": actual_classes,
        "veto_subclass_gates": veto_classes,
        "results_sha256": sha256(root / "MECHANISM_RESULTS.json"),
    }


def validate_score_and_question_files(root: Path) -> dict:
    main = load_json(root / "MECHANISM_RESULTS.json")
    score_path = root / "SCORE_MARGIN_ANALYSIS.json"
    veto_path = root / "CE_KL_VETO_ANALYSIS.json"
    interaction_path = root / "MODULE_INTERACTION_ANALYSIS.json"
    score = load_json(score_path)
    veto = load_json(veto_path)
    interaction = load_json(interaction_path)
    for name, value, key in (
            ("score", score, "ranking_not_calibration"),
            ("veto", veto, "ce_kl_veto"),
            ("interaction", interaction, "attention_mlp_interaction")):
        if value["protocol_sha256"] != PROTOCOL_SHA256:
            raise AssertionError(f"{name}: protocol hash")
        if value["classification"] != main["hypothesis_classification"][key]:
            raise AssertionError(f"{name}: classification mismatch")
    required_components = {
        "sum_ce_mean", "mean_ce_mean", "mean_ce_se", "mean_ce_standardized", "mean_ce_margin",
        "mean_kl_mean", "mean_kl_se", "mean_kl_standardized", "mean_kl_margin", "mean_combined_margin",
    }
    models = {}
    for model, block in score["models"].items():
        groups = {name: value for name, value in block["group_components"].items()}
        for group, values in groups.items():
            if not required_components.issubset(values) or not all(
                    math.isfinite(float(values[key])) for key in required_components):
                raise AssertionError(f"{model}:{group}: score-component decomposition")
        for observed_kind in ("group_only", "marginal"):
            correlations = block["component_correlations"][observed_kind]
            if set(correlations) != required_components:
                raise AssertionError(f"{model}:{observed_kind}: component correlations")
            for values in correlations.values():
                if not all(math.isfinite(float(values[key])) for key in ("pearson", "spearman")):
                    raise AssertionError(f"{model}:{observed_kind}: nonfinite component correlation")
        component_npz = Path(block["score_components_npz"]["path"])
        if sha256(component_npz) != block["score_components_npz"]["sha256"]:
            raise AssertionError(f"{model}: score component NPZ hash")
        fidelity = block["individual_tile_existing_frozen_evidence"]
        if int(fidelity["n"]) <= 0 or not all(math.isfinite(float(v)) for k, v in fidelity.items() if k != "n"):
            raise AssertionError(f"{model}: individual-tile counterevidence")
        models[model] = {"groups": {group: values["tiles"] for group, values in groups.items()},
                         "individual_tile_n": fidelity["n"]}
    fidelity_path = Path(score["source_first_order_fidelity"])
    if sha256(fidelity_path) != score["source_sha256"]:
        raise AssertionError("frozen attempt7 fidelity hash")
    prohibited = set(main["claims_prohibited"])
    expected_prohibited = {"individual-tile causality", "native FP4/E0M3 Tensor Core execution",
                           "latency or speedup", "area", "power"}
    if prohibited != expected_prohibited:
        raise AssertionError("prohibited-claims boundary drift")
    return {
        "valid": True,
        "models": models,
        "required_score_components": sorted(required_components),
        "frozen_first_order_fidelity_sha256": score["source_sha256"],
        "question_file_sha256": {
            "score_margin": sha256(score_path),
            "ce_kl_veto": sha256(veto_path),
            "module_interaction": sha256(interaction_path),
        },
        "claims_prohibited": sorted(prohibited),
    }


def validate_figures_and_documents(root: Path) -> dict:
    figures = {}
    for stem in ("score_margin_ranking", "ce_kl_veto", "module_interaction"):
        png = root / "figures" / f"{stem}.png"
        svg = root / "figures" / f"{stem}.svg"
        if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"bad PNG signature: {png}")
        with png.open("rb") as f:
            f.seek(16)
            width, height = struct.unpack(">II", f.read(8))
        if width < 1000 or height < 500:
            raise AssertionError(f"figure too small: {png}")
        ET.parse(svg)
        figures[stem] = {
            "png": {"sha256": sha256(png), "bytes": png.stat().st_size, "width": width, "height": height},
            "svg": {"sha256": sha256(svg), "bytes": svg.stat().st_size},
        }
    docs = {}
    for name in ("STATISTICAL_REPORT.md", "PRIMARY_RESULTS_TABLES.md", "MECHANISM_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md"):
        path = root / name
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 100:
            raise AssertionError(f"document too short: {name}")
        docs[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    verdict = (root / "MECHANISM_VERDICT.md").read_text(encoding="utf-8").lower()
    if not any(phrase in verdict for phrase in ("individual-tile causality", "individual-tile causal")):
        raise AssertionError("verdict missing scope boundary: individual-tile causality")
    for phrase in ("native fp4/e0m3", "latency", "speedup", "area", "power"):
        if phrase not in verdict:
            raise AssertionError(f"verdict missing scope boundary: {phrase}")
    return {"valid": True, "figures": figures, "documents": docs}


def audit_gpu(root: Path) -> dict:
    runs = []
    intervals = []
    total_gpu_hours = 0.0
    valid_gpu_hours = 0.0
    failed_gpu_hours = 0.0
    total_samples = 0
    gpu_model_targets = []
    mistral_launches = []
    for run_dir in sorted((root / "runs").iterdir()):
        launch_path = run_dir / "launch_record.json"
        if not launch_path.exists():
            continue
        launch = load_json(launch_path)
        requested = launch.get("requested", {})
        if int(requested.get("gpus", 0)) <= 0:
            continue
        command = launch.get("command") or []
        if "--model" in command:
            target = command[command.index("--model") + 1]
            gpu_model_targets.append(target)
            if target == "mistral7b":
                mistral_launches.append(run_dir.name)
        hours = float(launch.get("gpu_hours") or 0.0)
        total_gpu_hours += hours
        if launch.get("status") == "complete" and not launch.get("invalid_gpu_cotenancy"):
            valid_gpu_hours += hours
        else:
            failed_gpu_hours += hours
        uuids = launch.get("leased_uuids") or []
        run_record_path = run_dir / "run_record.json"
        gpu_names = []
        if run_record_path.exists():
            gpu_names = load_json(run_record_path).get("gpu_allocation", {}).get("gpu_names", [])
        if len(uuids) > 3 or (gpu_names and len(set(gpu_names)) != 1):
            raise AssertionError(f"{run_dir.name}: GPU count/homogeneity")
        if gpu_names and any("A6000" not in name or "Ada" in name for name in gpu_names):
            raise AssertionError(f"{run_dir.name}: unexpected GPU model {gpu_names}")
        invalid_reasons = (launch.get("sidecar") or {}).get("invalid_reasons") or []
        if invalid_reasons or launch.get("invalid_gpu_cotenancy"):
            raise AssertionError(f"{run_dir.name}: invalid co-tenancy")
        before_passed = bool((launch.get("host_preflight_before") or {}).get("passed"))
        after_passed = bool((launch.get("host_preflight_after") or {}).get("passed"))
        if not before_passed or (not after_passed and not launch.get("host_after_post_exit_only")):
            raise AssertionError(f"{run_dir.name}: missing/failed host preflight or postflight")
        samples = 0
        monitor = run_dir / "gpu_monitor.jsonl"
        if monitor.exists():
            with monitor.open("r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line); samples += 1
                    if set(sample["gpus"]) != set(uuids):
                        raise AssertionError(f"{run_dir.name}: monitor UUID")
                    for proc in sample["processes"]:
                        if proc["owner"] != "JaaaaaA_l" or int(proc["uid"]) != 1011 or not proc["in_container"]:
                            raise AssertionError(f"{run_dir.name}: foreign process")
        total_samples += samples
        if launch.get("started_utc") and launch.get("finished_utc"):
            intervals.append((parse_utc(launch["started_utc"]), parse_utc(launch["finished_utc"]), len(uuids), run_dir.name))
        runs.append({
            "run_id": run_dir.name,
            "status": launch.get("status"),
            "exit_code": launch.get("exit_code"),
            "gpu_hours": hours,
            "leased_uuids": uuids,
            "gpu_names": gpu_names,
            "monitor_samples": samples,
            "invalid_gpu_cotenancy": bool(launch.get("invalid_gpu_cotenancy")),
            "sidecar_invalid_reasons": invalid_reasons,
            "host_preflight_before_passed": before_passed,
            "host_preflight_after_passed": after_passed,
            "host_after_post_exit_only": bool(launch.get("host_after_post_exit_only")),
        })
    events = []
    for start, end, count, run_id in intervals:
        events.append((start, 1, count, run_id))
        events.append((end, 0, -count, run_id))
    concurrent = 0
    maximum = 0
    for _, _, delta, _ in sorted(events):
        concurrent += delta
        maximum = max(maximum, concurrent)
    if maximum > 3 or concurrent != 0:
        raise AssertionError(f"GPU concurrency invalid: max={maximum}, final={concurrent}")
    unfinished = [r["run_id"] for r in runs if r["status"] not in {"complete", "failed", "invalid"}]
    if unfinished:
        raise AssertionError(f"unfinished GPU attempts: {unfinished}")
    if set(gpu_model_targets) - {"llama8b", "qwen4b", "mistral7b"}:
        raise AssertionError(f"out-of-scope GPU model target: {gpu_model_targets}")
    if len(mistral_launches) != 1:
        raise AssertionError(f"Mistral must be launched exactly once: {mistral_launches}")
    protocol = load_json(root / "MECHANISM_PROTOCOL.json")
    launch_cutoff = parse_utc(protocol["hard_cutoffs_utc"]["new_gpu_launches"])
    campaign_end = parse_utc(protocol["hard_cutoffs_utc"]["campaign_end"])
    late_launches = [run_id for start, _, _, run_id in intervals if start > launch_cutoff]
    late_finishes = [run_id for _, end, _, run_id in intervals if end > campaign_end]
    if late_launches or late_finishes:
        raise AssertionError(f"campaign time gate violation: launches={late_launches}, finishes={late_finishes}")
    return {
        "schema": "mixfp4-gpu-policy-audit/v1",
        "passed": True,
        "maximum_concurrent_gpus_observed": maximum,
        "maximum_allowed": 3,
        "devices_used": "NVIDIA RTX A6000 only",
        "mixed_a6000_and_ada_within_run": False,
        "foreign_process_observed_on_leased_uuid": False,
        "total_monitor_samples_checked": total_samples,
        "gpu_attempts": len(runs),
        "gpu_model_targets": gpu_model_targets,
        "mistral_one_shot_launches": mistral_launches,
        "gpu_hours_all_attempts": total_gpu_hours,
        "gpu_hours_valid_complete_attempts": valid_gpu_hours,
        "gpu_hours_failed_attempts": failed_gpu_hours,
        "new_gpu_launch_cutoff_utc": protocol["hard_cutoffs_utc"]["new_gpu_launches"],
        "campaign_end_utc": protocol["hard_cutoffs_utc"]["campaign_end"],
        "gpu_runs_launched_after_cutoff": late_launches,
        "gpu_runs_finished_after_campaign_end": late_finishes,
        "runs": runs,
    }


def collect_failures_and_registry(root: Path) -> dict:
    shutil.copyfile(root / "registry" / "attempts.jsonl", root / "RUN_REGISTRY.jsonl")
    registry_lines = (root / "RUN_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines()
    for line in registry_lines:
        json.loads(line)
    failures = []
    status_counts = Counter()
    for run_dir in sorted((root / "runs").iterdir()):
        launch_path = run_dir / "launch_record.json"
        if not launch_path.exists():
            continue
        launch = load_json(launch_path)
        status = launch.get("status", "unknown")
        status_counts[status] += 1
        if status in {"failed", "invalid", "preflight_failed", "docker_failed", "not_started"}:
            job = load_json(run_dir / "job_status.json") if (run_dir / "job_status.json").exists() else {}
            failures.append({
                "run_id": run_dir.name,
                "matrix_id": launch.get("matrix_id"),
                "status": status,
                "exit_code": launch.get("exit_code"),
                "reason": job.get("error") or launch.get("reasons"),
                "oom": bool(job.get("oom", False)),
                "invalid_gpu_cotenancy": bool(launch.get("invalid_gpu_cotenancy")),
                "gpu_hours": float(launch.get("gpu_hours") or 0.0),
                "retained": True,
                "scientific_outcome_produced": False,
            })
    out = {
        "schema": "mixfp4-failed-or-skipped/v1",
        "failed_attempts": failures,
        "superseded_attempts": [{
            "run_id": "M10_derive_maps_attempt1",
            "reason": "CPU-only map attempt exposed RESULT_SCHEMA protocol/matrix enum incompatibility; all anchors passed but the record layer was superseded by amendment A01 before new GPU outcomes.",
            "scientific_definitions_changed": False,
            "retained": True,
        }],
        "predeclared_optional_work_skipped": [
            {"family": "quartile_sweeps", "reason": "optional exploration disabled in frozen protocol"},
            {"family": "complete_cumulative_curves", "reason": "optional exploration disabled in frozen protocol"},
            {"family": "early_middle_late_layer_thirds", "reason": "optional exploration disabled in frozen protocol"},
        ],
        "score_regeneration_skipped": {
            "reason": "verified frozen seed0 N16 moments/raw score evidence and exact maps were available at the hour-0 gate",
            "not_a_failure": True,
        },
        "out_of_scope_not_run": [
            "Granite-specific grid search",
            "reorder research",
            "new selector families",
            "new k sweep",
            "additional model families",
            "Blackwell fake-quant reruns",
            "native FP4/E0M3 kernels or latency/speedup/area/power measurements"
        ],
        "run_directory_status_counts": dict(status_counts),
        "registry_events": len(registry_lines),
    }
    write_json(root / "FAILED_OR_SKIPPED_RUNS.json", out)
    return out


def validate_json_and_jsonl(root: Path) -> dict:
    json_count = 0
    jsonl_count = 0
    jsonl_records = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "cache" in path.relative_to(root).parts:
            continue
        if path.suffix == ".json":
            load_json(path); json_count += 1
        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line); jsonl_records += 1
            jsonl_count += 1
    return {"json_files_parsed": json_count, "jsonl_files_parsed": jsonl_count, "jsonl_records_parsed": jsonl_records}


def write_manifest(root: Path) -> dict:
    manifest = root / "ARTIFACT_MANIFEST.sha256"
    excluded_top_level = {"cache", "tmp"}
    files = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rel = path.relative_to(root)
        if (rel.parts and rel.parts[0] in excluded_top_level) or "__pycache__" in rel.parts:
            continue
        digest = sha256(path)
        files.append(f"{digest}  {rel.as_posix()}\n")
        total_bytes += path.stat().st_size
    manifest.write_text("".join(files), encoding="utf-8")
    return {"entries": len(files), "represented_bytes": total_bytes, "sha256": sha256(manifest),
            "excluded_runtime_only_top_level_directories": sorted(excluded_top_level),
            "excluded_directory_name_anywhere": "__pycache__"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--llama-run", required=True)
    parser.add_argument("--qwen-run", required=True)
    parser.add_argument("--mistral-run", required=True)
    parser.add_argument("--analysis-run", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root).resolve()
    if sha256(root / "MECHANISM_PROTOCOL.json") != PROTOCOL_SHA256:
        raise SystemExit("protocol hash mismatch")
    protocol = load_json(root / "MECHANISM_PROTOCOL.json")
    source = root / "source" / "NVFP4-RaZeR-main" / "campaign"
    code_hashes = {name: sha256(source / name) for name in EXPECTED_CODE_HASHES}
    if code_hashes != EXPECTED_CODE_HASHES or protocol["code_hashes"] != EXPECTED_CODE_HASHES:
        raise SystemExit("frozen code hash mismatch")
    renderer = root / "analysis" / "render_reports.py"
    reporting_plan_path = root / "analysis" / "REPORTING_PLAN.json"
    reporting_plan = load_json(reporting_plan_path)
    if (sha256(renderer) != REPORT_RENDERER_SHA256 or
            sha256(reporting_plan_path) != REPORTING_PLAN_SHA256 or
            reporting_plan["renderer_sha256"] != REPORT_RENDERER_SHA256):
        raise SystemExit("pre-Mistral reporting renderer hash mismatch")
    failures = collect_failures_and_registry(root)
    gpu = audit_gpu(root)
    write_json(root / "GPU_POLICY_AUDIT.json", gpu)
    run_audits = [
        validate_llama_screen(root),
        validate_eval_run(root, "llama8b", args.llama_run),
        validate_eval_run(root, "qwen4b", args.qwen_run),
        validate_eval_run(root, "mistral7b", args.mistral_run),
    ]
    screen_full_audit = validate_screen_full_consistency(root, args.llama_run)
    power_audit = validate_power_freeze(root)
    input_audit = validate_immutable_inputs(root)
    map_audit = validate_maps(root)
    stats_audit = validate_statistics(root)
    question_file_audit = validate_score_and_question_files(root)
    presentation_audit = validate_figures_and_documents(root)
    promotion_gate_path = root / "MISTRAL_PROMOTION_GATE.json"
    promotion_gate = load_json(promotion_gate_path)
    if (not promotion_gate["passed"] or promotion_gate["decision"] != "launch_exactly_once" or
            promotion_gate["prior_mistral_launches"] or
            promotion_gate["promotion_gate_code_sha256"] != sha256(root / "analysis" / "check_mistral_promotion.py")):
        raise AssertionError("Mistral promotion gate invalid")
    analysis_run = root / "runs" / args.analysis_run
    analysis_record = load_json(analysis_run / "run_record_validation.json")
    analysis_launch = load_json(analysis_run / "launch_record.json")
    map_launch = load_json(root / "runs" / "V10_derive_maps_attempt2" / "launch_record.json")
    if (analysis_record != {"valid": True, "errors": []} or
            analysis_launch["requested"]["gpus"] != 0 or map_launch["requested"]["gpus"] != 0):
        raise AssertionError("analysis run record invalid")
    for name in REQUIRED_TOP_LEVEL:
        if not (root / name).is_file():
            raise AssertionError(f"missing required deliverable: {name}")
    parse_audit = validate_json_and_jsonl(root)
    audit = {
        "schema": "mixfp4-mechanism-artifact-audit/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "frozen_code_hashes": code_hashes,
        "pre_mistral_reporting_renderer_sha256": REPORT_RENDERER_SHA256,
        "pre_mistral_reporting_plan_sha256": REPORTING_PLAN_SHA256,
        "evaluation_runs": run_audits,
        "screen_full_consistency": screen_full_audit,
        "power_and_screening_audit": power_audit,
        "immutable_input_audit": input_audit,
        "map_audit": map_audit,
        "statistics_audit": stats_audit,
        "score_and_question_file_audit": question_file_audit,
        "presentation_audit": presentation_audit,
        "mistral_promotion_gate": {"passed": True, "sha256": sha256(promotion_gate_path),
                                   "checked_utc": promotion_gate["checked_utc"]},
        "gpu_policy_audit_sha256": sha256(root / "GPU_POLICY_AUDIT.json"),
        "failed_and_skipped_audit_sha256": sha256(root / "FAILED_OR_SKIPPED_RUNS.json"),
        "run_registry_sha256": sha256(root / "RUN_REGISTRY.jsonl"),
        "analysis_run": {"run_id": args.analysis_run, "record_valid": True,
                         "run_record_sha256": sha256(analysis_run / "run_record.json")},
        "json_parse_audit": parse_audit,
        "required_deliverables_present": REQUIRED_TOP_LEVEL,
        "manifest_scope": "all regular campaign files except ARTIFACT_MANIFEST.sha256 itself and runtime-only cache/, tmp/, and __pycache__/ directories",
        "failures_and_skips": {"failed_attempts": len(failures["failed_attempts"]),
                               "optional_families_skipped": len(failures["predeclared_optional_work_skipped"])},
    }
    write_json(root / "ARTIFACT_AUDIT.json", audit)
    summary = {
        "schema": "mixfp4-mechanism-campaign-summary/v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_gate": {
            "decision": load_json(root / "INPUT_PROVENANCE.json")["score_gate"]["decision"],
            "input_provenance_sha256": sha256(root / "INPUT_PROVENANCE.json"),
            "power_analysis_sha256": sha256(root / "POWER_ANALYSIS.json"),
            "score_regeneration_performed": False,
        },
        "map_manifest_sha256": sha256(root / "MAP_MANIFEST.json"),
        "authoritative_runs": {
            "maps": "V10_derive_maps_attempt2",
            "llama_screen": "V20_screen_llama8b_attempt2",
            "qwen_full": args.qwen_run,
            "llama_full": args.llama_run,
            "mistral_one_shot": args.mistral_run,
            "analysis": args.analysis_run,
        },
        "gpu_hours": {
            "all_attempts": gpu["gpu_hours_all_attempts"],
            "valid_complete_attempts": gpu["gpu_hours_valid_complete_attempts"],
            "failed_attempts": gpu["gpu_hours_failed_attempts"],
        },
        "attempt_accounting": {
            "run_directory_status_counts": failures["run_directory_status_counts"],
            "total_run_directories_with_launch_records": sum(failures["run_directory_status_counts"].values()),
            "registry_events": failures["registry_events"],
            "gpu_attempts": gpu["gpu_attempts"],
            "failed_attempts": len(failures["failed_attempts"]),
            "superseded_attempts": len(failures["superseded_attempts"]),
        },
        "hypothesis_classification": load_json(root / "MECHANISM_RESULTS.json")["hypothesis_classification"],
        "claims_prohibited": load_json(root / "MECHANISM_RESULTS.json")["claims_prohibited"],
        "artifact_audit_passed": True,
    }
    write_json(root / "CAMPAIGN_SUMMARY.json", summary)
    # Reparse newly written metadata, then write the checksum manifest last.
    validate_json_and_jsonl(root)
    manifest = write_manifest(root)
    print(json.dumps({"passed": True, "artifact_audit": str(root / "ARTIFACT_AUDIT.json"),
                      "manifest": manifest, "gpu_hours": summary["gpu_hours"]}, indent=2))


if __name__ == "__main__":
    main()
