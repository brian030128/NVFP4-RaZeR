"""Record the frozen-selector reuse decision for P75 without redundant GPU work.

The P75 portability map builder needs fresh A6000 selector statistics only when
P50 selected a replacement selector. When the hash-locked global winner falls
back to the old first-order selector, it instead consumes the already frozen
parent exact map, which was generated in a valid A6000 run. This CPU sentinel
checks that branch and preserves the redundant planned GPU job as a
machine-readable reuse decision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio
from campaign import runtime
from campaign.extension_heldout_maps import latest_complete, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
DECISION = CR / "provenance/P75_SELECTOR_STATS_REUSE_DECISION.json"


def selector_semantics(command: list[str]) -> dict:
    """Return the scientific arguments of an extension_selector_stats command."""
    if command[:2] != ["-m", "campaign.extension_selector_stats"]:
        raise ValueError("not an extension_selector_stats command")
    values = {
        "model": None,
        "freeze": None,
        "freeze_sha256": None,
        "attn": "sdpa",
        "max_memory_gib": None,
        "reconstruction_chunk_rows": 512,
    }
    mapping = {
        "--model": ("model", str),
        "--freeze": ("freeze", str),
        "--freeze-sha256": ("freeze_sha256", str),
        "--attn": ("attn", str),
        "--max-memory-gib": ("max_memory_gib", float),
        "--reconstruction-chunk-rows": ("reconstruction_chunk_rows", int),
    }
    index = 2
    while index < len(command):
        flag = command[index]
        if flag not in mapping or index + 1 >= len(command):
            raise ValueError(f"unexpected selector command argument: {flag}")
        key, convert = mapping[flag]
        values[key] = convert(command[index + 1])
        index += 2
    if values["model"] is None or values["freeze"] is None or values["freeze_sha256"] is None:
        raise ValueError("selector command is missing a required scientific argument")
    return values


def validate_fallback(winner: dict, p50: dict, p41: dict) -> dict:
    config = winner["global_configuration"]
    if not config.get("selector_fallback_to_old"):
        raise RuntimeError("P75 requires fresh A6000 statistics for a replacement selector")
    if not p50.get("fallback_to_old") or p50.get("selected_replacement") is not None:
        raise RuntimeError("P50 selection does not certify old-selector fallback")
    if config.get("selector") != "first_order_ce_kl" or p50.get("selected_policy") != "p50_old_first_order_ce_kl":
        raise RuntimeError("winner/P50 old-selector identity mismatch")
    p50_map = p50["selected_maps"]["llama8b"]
    p41_map = p41["selected_maps"]["llama8b"]
    for key in ("map_path", "map_sha256", "type_block"):
        if p50_map[key] != p41_map[key]:
            raise RuntimeError(f"P50/P41 Llama fallback map mismatch: {key}")
    return p50_map


def validated_module_manifest(header: dict, selector_report: dict,
                              parent_calibration_report: dict) -> str:
    """Resolve the module manifest from reports, not the exact-map header.

    The ``mixfp4map/v1`` header intentionally records the calibration manifest
    and module geometry but not a module-manifest digest.  The completed P50
    selector report and the parent calibration report are the two authoritative
    sources for that digest.  If a future map schema also carries the digest,
    require it to agree rather than silently preferring one source.
    """
    selector_sha = selector_report.get("module_manifest_sha256")
    parent_sha = parent_calibration_report.get("module_manifest_sha256")
    if (not isinstance(selector_sha, str) or len(selector_sha) != 64
            or not isinstance(parent_sha, str) or len(parent_sha) != 64):
        raise RuntimeError("module-manifest digest missing from authoritative reports")
    if selector_sha != parent_sha:
        raise RuntimeError("P50 selector and parent calibration module manifests differ")
    header_sha = header.get("module_manifest_sha256")
    if header_sha is not None and header_sha != selector_sha:
        raise RuntimeError("exact-map header and report module manifests differ")
    return selector_sha


def main() -> None:
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    decision = load(DECISION)
    if decision.get("protocol_sha256") != PROTOCOL:
        raise RuntimeError("P75 selector-reuse decision protocol mismatch")
    winner_path = CR / "freeze/P70_GLOBAL_WINNER.json"
    p50_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    fallback = validate_fallback(load(winner_path), load(p50_path), load(p41_path))

    header, _, map_sha = mapio.read_map(fallback["map_path"], fallback["map_sha256"])
    parent_run = Path(fallback["map_path"]).parents[1]
    try:
        parent_run.relative_to(PR / "runs")
    except ValueError as exc:
        raise RuntimeError("fallback map is not from the read-only parent campaign") from exc
    parent_launch = load(parent_run / "launch_record.json")
    parent_validation = load(parent_run / "run_record_validation.json")
    parent_calibration_report_path = parent_run / "calibration/calibration_report.json"
    parent_calibration_report = load(parent_calibration_report_path)
    if (parent_launch.get("status") != "complete"
            or parent_launch.get("requested", {}).get("model") != "a6000"
            or len(parent_launch.get("leased_uuids") or []) != 1
            or not parent_validation.get("valid")):
        raise RuntimeError("fallback exact map lacks a valid one-A6000 parent derivation")

    prior_run = latest_complete("P50_selector_stats_llama8b")
    prior_launch = load(prior_run / "launch_record.json")
    original_command = decision["original_planned_gpu_job"]["command"]
    if selector_semantics(prior_launch["command"]) != selector_semantics(original_command):
        raise RuntimeError("planned P75 selector statistics differ from completed P50 statistics")
    stats_path = prior_run / "selector_stats/selector_statistics.pt"
    stats_report = prior_run / "selector_stats/selector_stats_report.json"
    stats_report_data = load(stats_report)
    if stats_report_data.get("status") != "complete":
        raise RuntimeError("prior P50 selector-statistics report is incomplete")
    module_manifest_sha256 = validated_module_manifest(
        header, stats_report_data, parent_calibration_report)

    report = {
        "schema_version": "1.0",
        "status": "complete_reuse_no_gpu",
        "protocol_sha256": PROTOCOL,
        "reason": "the frozen winner uses the old first-order selector, so P75 consumes the parent exact map generated on one clean A6000; fresh selector statistics cannot affect that map",
        "winner": {"path": str(winner_path), "sha256": runtime.sha256_file(winner_path)},
        "P50_selection": {"path": str(p50_path), "sha256": runtime.sha256_file(p50_path)},
        "P41_selection": {"path": str(p41_path), "sha256": runtime.sha256_file(p41_path)},
        "reused_exact_map": {
            "path": fallback["map_path"], "sha256": map_sha,
            "policy": header["policy"], "type_block": header["type_block"],
            "selected_tiles": header["totals"]["selected_tiles"],
            "selected_weights": header["totals"]["selected_weights"],
            "parent_run_id": parent_run.name,
            "parent_run_requested_device": parent_launch["requested"]["model"],
            "module_manifest_sha256": module_manifest_sha256,
            "parent_calibration_report_path": str(parent_calibration_report_path),
            "parent_calibration_report_sha256": runtime.sha256_file(parent_calibration_report_path),
        },
        "completed_equivalent_selector_statistics": {
            "run_id": prior_run.name,
            "scientific_command": selector_semantics(prior_launch["command"]),
            "statistics_path": str(stats_path),
            "statistics_sha256": runtime.sha256_file(stats_path),
            "report_path": str(stats_report),
            "report_sha256": runtime.sha256_file(stats_report),
            "note": "retained as development evidence but not used to regenerate the fallback portability map",
        },
        "original_planned_gpu_job_sha256": decision["original_planned_gpu_job_sha256"],
        "gpu_hours_added": 0.0,
        "scientific_result_changed": False,
    }
    out = runtime.out_dir("selector_reuse")
    path = out / "P75_SELECTOR_STATS_REUSE.json"
    runtime.atomic_json(path, report)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P75-selector-reuse-sentinel",
                   "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "not_applicable", "model_class": "reuse",
                   "module_manifest_sha256": module_manifest_sha256,
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": header["calibration_manifest_sha256"],
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(path)], "summary": report,
                    "uncertainty": {}, "attempted_endpoints": ["P75_SELECTOR_STATS_REUSE"],
                    "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps({"status": "complete", "reuse": True,
                      "map_sha256": map_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
