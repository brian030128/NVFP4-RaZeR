"""Derive P50 fixed-budget selector maps after the P20/P41 freezes.

The P41-selected exact map supplies the model-specific selected-weight budget and
is carried into P50 byte-for-byte.  Replacement selector maps are global N16K64
rankings at that exact budget.  This module consumes calibration artifacts only;
it never reads WikiText/C4 outcomes except the immutable P20/P41 selection files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_maps import Writer, select_global


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("llama8b", "qwen4b", "mistral7b")
PARENT = {
    "llama8b": "V30_calib_llama8b_seed0_attempt1",
    "qwen4b": "V30_calib_qwen4b_seed0_attempt1",
    "mistral7b": "V61_calib_mistral7b_seed0_attempt2",
}
CANDIDATES = (
    "activation_weighted_error",
    "layer_output_reconstruction",
    "diagonal_hessian_gptq_proxy",
)


def load(path: Path):
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    return runtime.sha256_file(path)


def latest_complete(prefix: str) -> Path:
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            lr = load(run / "launch_record.json")
            st = load(run / "job_status.json")
            vr = load(run / "run_record_validation.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if lr.get("status") == st.get("status") == "complete" and vr.get("valid"):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def transformed_strong_scores(selector: str, weight_stats: dict, selector_stats: dict):
    """Return the P20 family as an ascending score, exactly as predeclared."""
    if selector == "activation_weighted":
        return selector_stats["scores"]["activation_weighted_error"], False
    keys = {"weight_mse": "mse_gain16", "magnitude": "l2_16", "change_norm": "dnorm16"}
    if selector not in keys:
        raise ValueError(f"unsupported frozen P20 selector family: {selector}")
    # P20 selects these three scores descending.  Negation makes the common
    # P50 ranking ascending without changing the deterministic tie order.
    return {name: -value.double() for name, value in weight_stats[keys[selector]].items()}, True


def validate_scores(names, shapes, groups):
    for label, group in groups.items():
        if list(group) != names:
            raise RuntimeError(f"{label}: module names/order differ")
        for name in names:
            expected = (shapes[name][0] // 16, shapes[name][1] // 64)
            # Parent weight statistics and the new selector artifact both use
            # one value per tile, but one stores a flat vector while the other
            # is free to store the grid.  Map serialization canonicalizes both
            # through Writer.write(...).reshape(expected).
            if int(group[name].numel()) != expected[0] * expected[1]:
                raise RuntimeError(f"{label}/{name}: score size {group[name].numel()} != {expected}")
            if not torch.isfinite(group[name]).all():
                raise RuntimeError(f"{label}/{name}: nonfinite score")


def derive(model: str):
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    p20_path = CR / "provenance/P20_STRONGEST_CONTROL_SELECTION.json"
    resolution_path = CR / "provenance/P50_STRONG_HEURISTIC_RESOLUTION.json"
    gate_path = CR / "provenance/P50_G6_OPERATIONALIZATION.json"
    for path in (p41_path, p20_path, resolution_path, gate_path):
        if not path.exists():
            raise FileNotFoundError(path)
    p41, p20 = load(p41_path), load(p20_path)
    old = p41["selected_maps"][model]
    old_header, old_masks, old_sha = MIO.read_map(old["map_path"], old["map_sha256"])
    if old_sha != old["map_sha256"] or old_header["type_block"] != [16, 64]:
        raise RuntimeError("P41 selected map identity/type-block mismatch")
    target = int(old_header["totals"]["selected_tiles"])
    if target * 1024 != int(old_header["totals"]["selected_weights"]):
        raise RuntimeError("P41 selected map has non-N16 selected-weight accounting")

    stats_run = latest_complete(f"P50_selector_stats_{model}")
    stats_path = stats_run / "selector_stats/selector_statistics.pt"
    stats_report_path = stats_run / "selector_stats/selector_stats_report.json"
    stats_report = load(stats_report_path)
    if stats_report.get("status") != "complete" or stats_report["statistic_artifact"]["sha256"] != sha(stats_path):
        raise RuntimeError("selector-statistics artifact is not accepted")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if stats.get("model") != model or stats.get("protocol_sha256") != PROTOCOL:
        raise RuntimeError("selector-statistics identity mismatch")
    names = list(stats["names"])
    shapes = {name: tuple(stats["shapes"][name]) for name in names}
    if list(old_masks) != names:
        raise RuntimeError("P41 map and selector-statistics module order differ")
    if any(tuple(old_masks[name].shape) != (shapes[name][0] // 16, shapes[name][1] // 64) for name in names):
        raise RuntimeError("P41 map and selector-statistics geometry differ")

    parent = PR / "runs" / PARENT[model]
    weight_path = parent / "calibration/moments/weight_tile_stats.pt"
    p20_spec = load(CR / "provenance/P20_CONTROL_DERIVATION_SPEC.json")
    expected_weight_sha = p20_spec["parent_moment_inputs"][model]["weight_tile_stats_sha256"]
    if sha(weight_path) != expected_weight_sha:
        raise RuntimeError("parent weight-statistics digest mismatch")
    weight_stats = torch.load(weight_path, map_location="cpu", weights_only=False)
    strong_selector = p20["selector"]
    strong_scores, was_negated = transformed_strong_scores(strong_selector, weight_stats, stats)
    groups = {name: stats["scores"][name] for name in CANDIDATES}
    groups["strongest_p20_selector_global_rerank"] = strong_scores
    validate_scores(names, shapes, groups)

    writer = Writer(model, names, shapes, stats["calibration_manifest_sha256"],
                    stats_report["model_class"], sha(CR / "provenance/P50_P51_SELECTOR_FIDELITY_SPEC.json"))
    map_names = {}
    for score_name in CANDIDATES:
        policy = f"p50_{score_name}"
        masks = select_global(names, groups[score_name], target)
        entry = writer.write(policy, masks,
                             {"selector": score_name, "scope": "global", "target_tiles": target,
                              "matched_to": "P41_selected_map"},
                             f"ascending {score_name}; exact P41 selected-weight budget")
        map_names[score_name] = entry["policy"]
    strong_policy = "p50_strongest_p20_selector_global_rerank"
    strong_masks = select_global(names, strong_scores, target)
    entry = writer.write(strong_policy, strong_masks,
                         {"selector": strong_selector, "scope": "global", "target_tiles": target,
                          "transformed_to_ascending_by_negation": was_negated,
                          "matched_to": "P41_selected_map"},
                         "frozen P20 selector family reranked globally at exact P41 budget")
    map_names["strongest_p20_selector_global_rerank"] = entry["policy"]

    summaries = {}
    for label, group in groups.items():
        flat = torch.cat([group[name].reshape(-1).double() for name in sorted(names)])
        summaries[label] = {"tiles": int(flat.numel()), "minimum": float(flat.min()),
                            "maximum": float(flat.max()), "mean": float(flat.mean()),
                            "finite": bool(torch.isfinite(flat).all())}
    report = {
        "schema_version": "1.0", "mode": "P50", "model": model, "status": "running",
        "protocol_sha256": PROTOCOL,
        "selector_spec_sha256": sha(CR / "provenance/P50_P51_SELECTOR_FIDELITY_SPEC.json"),
        "strong_resolution_sha256": sha(resolution_path), "gate_spec_sha256": sha(gate_path),
        "p41_selection_sha256": sha(p41_path), "p20_selection_sha256": sha(p20_path),
        "old_first_order_map": {**old, "header_protocol_id": old_header["protocol_id"],
                                "selected_tiles": target, "selected_weights": target * 1024},
        "target_selected_tiles": target, "target_selected_weights": target * 1024,
        "selector_statistics": {"run_id": stats_run.name, "path": str(stats_path),
                                "sha256": sha(stats_path)},
        "weight_statistics": {"path": str(weight_path), "sha256": expected_weight_sha},
        "strongest_p20_selector_family": strong_selector,
        "strongest_score_negated_for_ascending_order": was_negated,
        "score_summaries": summaries, "map_policy_names": map_names,
        "module_manifest_sha256": stats["module_manifest_sha256"],
    }
    writer.finish(report)
    result_path = runtime.run_dir / "job_result.json"
    result = load(result_path)
    result["results"]["raw_outputs"].extend([str(stats_path), str(stats_report_path), str(weight_path)])
    result["results"]["summary"].update(target_selected_tiles=target,
                                           target_selected_weights=target * 1024,
                                           strongest_p20_selector_family=strong_selector,
                                           old_map_sha256=old_sha)
    result["results"]["attempted_endpoints"] = ["P50_SELECTOR_MAP_DERIVATION"]
    runtime.atomic_json(result_path, result)
    print(json.dumps({"status": "complete", "model": model, "maps": len(writer.entries),
                      "target_tiles": target}, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    args = ap.parse_args()
    if sha(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    derive(args.model)


if __name__ == "__main__":
    main()
