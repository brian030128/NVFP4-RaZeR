"""Freeze exact maps/plans for A6000-to-Ada quality portability anchors."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_maps import Writer, select_global
from campaign.extension_heldout_maps import latest_complete, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
A6000 = "NVIDIA RTX A6000"


def map_plan_entry(row, name, multiplier):
    path = row.get("map_path", row.get("path")); digest = row.get("map_sha256", row.get("sha256"))
    header, _, got = MIO.read_map(path, digest)
    return {"name": name, "kind": "map", "map_path": path, "map_sha256": got,
            "map_policy": row.get("map_policy", row.get("policy")),
            "type_block": row["type_block"], "protocol_id": header["protocol_id"],
            "expected_total_tiles": row.get("expected_total_tiles", row.get("total_tiles")),
            "weight_scale_multiplier": float(multiplier)}


def a6000_selector_map(config, p50):
    if config["selector_fallback_to_old"]:
        p41 = load(CR / "provenance/P41_AGGREGATION_SELECTION.json")
        return p41["selected_maps"]["llama8b"], {
            "source": "P41 first-order map from A6000 parent calibration sufficient statistics",
            "selector_stats_run": None}
    run = latest_complete("P75_selector_stats_llama8b")
    launch = load(run / "launch_record.json")
    if launch["requested"]["model"] != "a6000" or len(launch["leased_uuids"]) != 1:
        raise RuntimeError("P75 Llama selector statistics were not produced on one A6000")
    stats_path = run / "selector_stats/selector_statistics.pt"
    import torch
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    key = config["selector"]
    score = stats["scores"][key]
    old = p50["selected_maps"]["llama8b"]
    old_header, _, _ = MIO.read_map(old["map_path"], old["map_sha256"])
    target = int(old_header["totals"]["selected_tiles"])
    masks = select_global(stats["names"], score, target)
    writer = Writer("llama8b", stats["names"],
                    {name: tuple(stats["shapes"][name]) for name in stats["names"]},
                    stats["calibration_manifest_sha256"], load(run / "selector_stats/selector_stats_report.json")["model_class"],
                    runtime.sha256_file(CR / "provenance/P70_P75_WINNER_HELDOUT_OPERATIONALIZATION.json"))
    entry = writer.write("p75_a6000_frozen_winner", masks,
                         {"selector": key, "target_tiles": target,
                          "weight_scale_multiplier": config["weight_scale_multiplier"],
                          "activation": "four_over_six_rows", "portability_anchor": True},
                         "frozen selector replayed from A6000 calibration inputs at exact development budget")
    writer.finish({"schema_version": "1.0", "mode": "P75_A6000_MAP", "model": "llama8b",
                   "status": "running", "protocol_sha256": PROTOCOL,
                   "module_manifest_sha256": stats["module_manifest_sha256"],
                   "selector_stats_path": str(stats_path),
                   "selector_stats_sha256": runtime.sha256_file(stats_path),
                   "target_selected_tiles": target})
    return entry, {"source": "frozen selector statistics captured on one clean A6000",
                   "selector_stats_run": run.name,
                   "selector_stats_sha256": runtime.sha256_file(stats_path)}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    winner_path = CR / "freeze/P70_GLOBAL_WINNER.json"
    winner = load(winner_path); config = winner["global_configuration"]
    p50 = load(CR / "provenance/P50_SELECTOR_SELECTION.json")
    llama_map, llama_source = a6000_selector_map(config, p50)
    granite = next(row for row in winner["heldout_models"] if row["model"] == "granite8b")
    granite_report = load(granite["map_derivation_report"])
    granite_map = {"map_path": granite["winner_map_path"], "map_sha256": granite["winner_map_sha256"],
                   "map_policy": "p70_frozen_winner", "type_block": [16, 64],
                   "expected_total_tiles": None}
    multiplier = config["weight_scale_multiplier"]
    models = {
        "llama8b": {"map": llama_map, "generation": llama_source},
        "granite8b": {"map": granite_map,
                       "generation": {"source": "P70 five-draw heldout calibration/selector path scheduled on A6000",
                                      "seed0_calibration_run": next((x.get("run_id") for x in granite_report[
                                          "calibration_inputs"] if x["draw"] == "seed0"), None),
                                      "selector_stats": granite_report.get("selector_statistics")}}
    }
    rows = []
    for model, obj in models.items():
        entry = map_plan_entry(obj["map"], "frozen_winner", multiplier)
        plan = [{"name": "four_over_six", "kind": "four_over_six"}, entry]
        path = CR / "plans" / f"P75_portability_{model}.json"
        if path.exists():
            raise FileExistsError(path)
        runtime.atomic_json(path, plan); path.chmod(0o444)
        rows.append({"model": model, "plan_path": str(path),
                     "plan_sha256": runtime.sha256_file(path),
                     "map_path": entry["map_path"], "map_sha256": entry["map_sha256"],
                     "map_generation": obj["generation"]})
    freeze = {"schema_version": "1.0", "status": "LOCKED_BEFORE_P75_QUALITY",
              "protocol_sha256": PROTOCOL,
              "winner_freeze_sha256": runtime.sha256_file(winner_path),
              "devices": ["NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation"],
              "windows": {"wiki": 16, "c4": 16}, "models": rows,
              "same_exact_map_both_devices": True, "Ada_map_regeneration": "forbidden",
              "native_performance_claim": "forbidden"}
    freeze_path = CR / "freeze/P75_PORTABILITY_MAPS.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    runtime.atomic_json(freeze_path, freeze); freeze_path.chmod(0o444)
    # Writer.finish may already have emitted the Llama-map job result.  Replace
    # it with the complete two-model portability freeze record.
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P75-portability-map-freeze",
                   "model_revision": runtime.sha256_file(freeze_path),
                   "tokenizer_revision": "multiple", "model_class": "map_freeze",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per model",
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(freeze_path)] + [row["plan_path"] for row in rows],
                    "summary": {"models": 2, "same_exact_map_both_devices": True},
                    "uncertainty": {}, "attempted_endpoints": ["P75_MAP_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "maps": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
