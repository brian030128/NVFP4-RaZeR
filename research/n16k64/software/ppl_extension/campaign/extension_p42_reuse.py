"""Reuse P21 quality arrays for the degenerate P42 seed0 fallback.

P41 selected ``seed0_k2_fallback``.  Consequently the P42 aggregation mask is
tile-for-tile identical to the seed0 N16 k2 reference, while the frozen P20
control is tile-for-tile identical to the already evaluated P21 control.  This
module proves those identities before relabelling the P21 per-window arrays.  It
exists to avoid rerunning an already completed GPU quality experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import torch

from campaign import mapio
from campaign import runtime
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("qwen27b", "phi4", "olmo2_13b")
P42_ORDER = ("seed0_n16_k2_reference", "p42_frozen_aggregation",
             "p42_strong_control_budget_matched")
P21_ORDER = ("n16_k2_reference", "p21_n8_u2_matched",
             "p21_frozen_strongest_control")
SOURCE_POLICY = {
    "seed0_n16_k2_reference": "n16_k2_reference",
    "p42_frozen_aggregation": "n16_k2_reference",
    "p42_strong_control_budget_matched": "p21_frozen_strongest_control",
}
SNAPSHOT = CR / "provenance/P42_ORIGINAL_GPU_QUALITY_JOB_SPECS.json"
RESOLUTION = CR / "provenance/P42_EXACT_PAYLOAD_REUSE_RESOLUTION.json"


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, valid = (load(run / name) for name in
                                     ("launch_record.json", "job_status.json",
                                      "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and valid.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def normalize_eval_command(command):
    """Remove only the plan pathname from an evaluate_ppl command."""
    command = list(command)
    if command[:2] != ["-m", "campaign.evaluate_ppl"]:
        raise RuntimeError("quality snapshot is not an evaluate_ppl command")
    try:
        position = command.index("--plan")
    except ValueError as exc:
        raise RuntimeError("quality command has no --plan") from exc
    command[position + 1] = "<FROZEN_PLAN>"
    return command


def payload_identity(left, right):
    left_header, left_masks, left_sha = mapio.read_map(
        left["map_path"], left["map_sha256"])
    right_header, right_masks, right_sha = mapio.read_map(
        right["map_path"], right["map_sha256"])
    same_order = list(left_masks) == list(right_masks)
    same_masks = same_order and all(
        torch.equal(left_masks[name], right_masks[name]) for name in left_masks)
    left_selected = sum(int(mask.sum()) for mask in left_masks.values())
    right_selected = sum(int(mask.sum()) for mask in right_masks.values())
    return {
        "left_map_sha256": left_sha,
        "right_map_sha256": right_sha,
        "module_order_identical": same_order,
        "tile_masks_identical": same_masks,
        "left_selected_tiles": left_selected,
        "right_selected_tiles": right_selected,
        "type_block_identical": left_header.get("type_block") == right_header.get("type_block"),
    }


def source_context(model):
    run = latest_complete(f"P21_quality_{model}")
    report_path = run / "ppl/ppl_report.json"
    result_path = run / "job_result.json"
    report, result = load(report_path), load(result_path)
    if (report.get("status") != "complete"
            or report.get("protocol_id") != "ppl-improvement-extension-v1"
            or report.get("freeze_sha256") != PROTOCOL
            or report.get("length") != 2048
            or report.get("domains") != ["wiki", "c4"]
            or report.get("teacher") != "none"
            or report.get("attention_backend") != "sdpa"
            or not report.get("reinstall_check", {}).get("identical")):
        raise RuntimeError(f"P21 source quality contract failed: {model}")
    validate_policy_contract(report, P21_ORDER, f"P21 reuse source {model}")
    return run, report_path, report, result


def model_identity(model, snapshot):
    p42_plan = load(CR / f"plans/P42_aggregation_breadth_{model}.json")
    p21_plan = load(CR / f"plans/P21_density_breadth_{model}.json")
    if tuple(row["name"] for row in p42_plan) != P42_ORDER:
        raise RuntimeError(f"P42 plan order drift: {model}")
    if tuple(row["name"] for row in p21_plan) != P21_ORDER:
        raise RuntimeError(f"P21 plan order drift: {model}")
    p42 = {row["name"]: row for row in p42_plan}
    p21 = {row["name"]: row for row in p21_plan}
    aggregation = payload_identity(p42["p42_frozen_aggregation"],
                                   p42["seed0_n16_k2_reference"])
    control = payload_identity(p42["p42_strong_control_budget_matched"],
                               p21["p21_frozen_strongest_control"])
    if not all((aggregation["module_order_identical"],
                aggregation["tile_masks_identical"],
                aggregation["type_block_identical"],
                control["module_order_identical"],
                control["tile_masks_identical"],
                control["type_block_identical"])):
        raise RuntimeError(f"P42 exact-payload reuse proof failed: {model}")
    if aggregation["left_selected_tiles"] != control["left_selected_tiles"]:
        raise RuntimeError(f"P42 matched budget drift: {model}")
    original = snapshot["original_job_specs"][model]["spec"]
    p21_job = load(CR / f"queue/jobs/P21_quality_{model}.json")
    command_equal = normalize_eval_command(original["command"]) == normalize_eval_command(
        p21_job["command"])
    if not command_equal:
        raise RuntimeError(f"P21/P42 evaluation command drift: {model}")
    run, report_path, report, result = source_context(model)
    installs = {row["name"]: row for row in report["installs"]}
    return {
        "model": model,
        "aggregation_vs_seed0": aggregation,
        "strong_control_vs_p21_control": control,
        "evaluation_command_identical_except_plan": command_equal,
        "evaluation_manifest_sha256": report["evaluation_manifest_sha256"],
        "source_run_id": run.name,
        "source_report_path": str(report_path),
        "source_report_sha256": runtime.sha256_file(report_path),
        "source_reinstall_identical": report["reinstall_check"]["identical"],
        "source_installed_weight_sha256": {
            "seed0_and_aggregation": installs["n16_k2_reference"]["installed_weight_sha256"],
            "strong_control": installs["p21_frozen_strongest_control"]["installed_weight_sha256"],
        },
        "original_p42_job_spec_file_sha256": snapshot["original_job_specs"][model]["file_sha256"],
        "p21_job_spec_file_sha256": runtime.sha256_file(CR / f"queue/jobs/P21_quality_{model}.json"),
    }


def ensure_resolution():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise RuntimeError("protocol digest mismatch")
    selection_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    selection = load(selection_path)
    if not (selection.get("fallback_used") is True
            and selection.get("aggregation") == "seed0_k2_fallback"
            and selection.get("form") == "natural"
            and float(selection.get("global_k")) == 2.0):
        raise RuntimeError("P42 reuse is legal only for the frozen seed0 k2 fallback")
    snapshot = load(SNAPSHOT)
    if snapshot.get("status") != "CAPTURED_BEFORE_P42_GPU_LAUNCH":
        raise RuntimeError("original P42 GPU job snapshot missing")
    identities = {model: model_identity(model, snapshot) for model in MODELS}
    resolution = {
        "schema_version": "1.0",
        "status": "FROZEN_EXACT_PAYLOAD_REUSE_BEFORE_P42_QUALITY",
        "matrix_id": "P42_DRAW_AGG_BREADTH",
        "protocol_sha256": PROTOCOL,
        "P41_selection_path": str(selection_path),
        "P41_selection_sha256": runtime.sha256_file(selection_path),
        "original_job_spec_snapshot_path": str(SNAPSHOT),
        "original_job_spec_snapshot_sha256": runtime.sha256_file(SNAPSHOT),
        "identities": identities,
        "gpu_quality_rerun_avoided": True,
        "reason": "P41 selected seed0_k2_fallback; every P42 method payload is tile-identical to a completed P21 evaluation under the same frozen evaluation command and manifest",
        "scientific_interpretation": "aggregation_vs_seed0 is exactly zero by construction; aggregation_vs_strong_control reuses the already measured paired P21 arrays",
        "post_hoc_robustness_extension": True,
    }
    if RESOLUTION.exists():
        if load(RESOLUTION) != resolution:
            raise RuntimeError("existing P42 reuse resolution differs")
    else:
        runtime.atomic_json(RESOLUTION, resolution)
        RESOLUTION.chmod(0o444)
    return resolution


def target_install(plan_row, source_install):
    header, masks, digest = mapio.read_map(plan_row["map_path"], plan_row["map_sha256"])
    install = copy.deepcopy(source_install)
    install.update({
        "name": plan_row["name"],
        "map_path": plan_row["map_path"],
        "map_sha256": digest,
        "map_header_policy": header["policy"],
        "expected_protocol_id": plan_row["protocol_id"],
        "map_reloaded_for_evaluation": False,
        "source_equivalent_map_reloaded_for_evaluation": bool(
            source_install.get("map_reloaded_for_evaluation")),
        "evaluation_reused_from_exact_payload": True,
        "selected_tiles": sum(int(mask.sum()) for mask in masks.values()),
        "total_tiles": sum(int(mask.numel()) for mask in masks.values()),
        "type_block": header["type_block"],
    })
    return install


def remap_report(source_report, plan, source_run, source_report_sha, identity):
    source_eval = source_report["evaluation"]
    source_installs = {row["name"]: row for row in source_report["installs"]}
    report = copy.deepcopy(source_report)
    report["plan"] = copy.deepcopy(plan)
    report["evaluation"] = {
        name: copy.deepcopy(source_eval[SOURCE_POLICY[name]]) for name in P42_ORDER
    }
    report["installs"] = [target_install(row, source_installs[SOURCE_POLICY[row["name"]]])
                          for row in plan]
    original_check = copy.deepcopy(source_report["reinstall_check"])
    report["reinstall_check"] = {
        "policy": P42_ORDER[0],
        "identical": bool(original_check.get("identical")),
        "checksum_equal": bool(original_check.get("checksum_equal")),
        "exact_payload_reuse": True,
        "source_policy": original_check.get("policy"),
        "source_reinstall_check": original_check,
    }
    report["exact_payload_reuse"] = {
        "source_run_id": source_run,
        "source_report_sha256": source_report_sha,
        "resolution_path": str(RESOLUTION),
        "resolution_sha256": runtime.sha256_file(RESOLUTION),
        "identity": identity,
        "gpu_rerun_performed": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    model = args.model
    resolution = ensure_resolution()
    source_run, source_path, source_report, source_result = source_context(model)
    plan_path = CR / f"plans/P42_aggregation_breadth_{model}.json"
    plan = load(plan_path)
    report = remap_report(source_report, plan, source_run.name,
                          runtime.sha256_file(source_path), resolution["identities"][model])
    validate_policy_contract(report, P42_ORDER, f"P42 exact-payload reuse {model}")
    out = runtime.out_dir("ppl")
    for domain in ("wiki", "c4"):
        shutil.copyfile(source_run / f"ppl/windows_{domain}.json",
                        out / f"windows_{domain}.json")
    report_path = out / "ppl_report.json"
    runtime.atomic_json(report_path, report)
    reuse_path = out / "P42_EXACT_PAYLOAD_REUSE.json"
    runtime.atomic_json(reuse_path, {
        "schema_version": "1.0", "status": "complete", "model": model,
        "protocol_sha256": PROTOCOL,
        "source_run_id": source_run.name,
        "source_report_path": str(source_path),
        "source_report_sha256": runtime.sha256_file(source_path),
        "output_report_path": str(report_path),
        "output_report_sha256": runtime.sha256_file(report_path),
        "resolution_path": str(RESOLUTION),
        "resolution_sha256": runtime.sha256_file(RESOLUTION),
        "identity": resolution["identities"][model],
        "gpu_rerun_performed": False,
    })
    launch = load(runtime.run_dir / "launch_record.json")
    summary = {name: {domain: {
        "ppl": report["evaluation"][name][domain]["ppl"],
        "mean_nll": report["evaluation"][name][domain]["mean_nll"],
    } for domain in ("wiki", "c4")} for name in P42_ORDER}
    policies = []
    for install in report["installs"]:
        policies.append({
            "name": install["name"], "weight_format": "map",
            "activation_format": install["activation"], "scale_block": 16,
            "map_path": install["map_path"], "map_sha256": install["map_sha256"],
            "type_block": install["type_block"],
            "selected_tiles": install["selected_tiles"],
            "total_tiles": install["total_tiles"],
            "weight_scale_multiplier": install.get("weight_scale_multiplier", 1),
            "map_reloaded_for_evaluation": False,
            "source_equivalent_map_reloaded_for_evaluation": True,
            "evaluation_reused_from_exact_payload": True,
        })
    source = copy.deepcopy(source_result["source"])
    source["source_manifest_sha256"] = launch["source_manifest_sha256"]
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL,
        "source": source,
        "environment": runtime.environment(attention_backend="sdpa",
                                           activation_quantizer="reused exact P21 payload"),
        "data": copy.deepcopy(source_result["data"]),
        "policies": policies,
        "results": {
            "raw_outputs": [str(report_path), str(reuse_path),
                            str(out / "windows_wiki.json"), str(out / "windows_c4.json"),
                            str(RESOLUTION), str(source_path)],
            "summary": summary,
            "uncertainty": {"note": "paired per-window arrays are byte-value-identical relabellings of the exact-payload P21 evaluation; bootstrap is recomputed in P42 analysis"},
            "attempted_endpoints": [f"{name}:{domain}" for name in P42_ORDER
                                    for domain in ("wiki", "c4")],
            "missing_endpoints": [],
            "gpu_quality_rerun_avoided": True,
        },
        "logs": [], "failures": [],
    })
    print(json.dumps({"status": "complete", "model": model,
                      "source_run": source_run.name,
                      "report_sha256": runtime.sha256_file(report_path),
                      "gpu_rerun_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
