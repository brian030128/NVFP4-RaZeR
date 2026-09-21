"""Freeze five-arm P50 exact-map plans after all map derivations complete."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_p50_maps import CANDIDATES, CR, MODELS, PROTOCOL, latest_complete, load


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def plan_entry(name, row):
    header, _, digest = MIO.read_map(row["path"], row["sha256"])
    return {"name": name, "kind": "map", "map_policy": row["policy"],
            "map_path": row["path"], "map_sha256": digest,
            "type_block": row["type_block"], "protocol_id": header["protocol_id"],
            "expected_total_tiles": row["total_tiles"]}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    p41 = load(p41_path)
    rows = []
    order = ["p50_strongest_p20_selector_global_rerank", *[f"p50_{x}" for x in CANDIDATES]]
    for model in MODELS:
        run = latest_complete(f"P50_maps_{model}")
        manifest_path = run / "derived_maps/map_manifest.json"
        manifest = {row["policy"]: row for row in load(manifest_path)}
        old = p41["selected_maps"][model]
        old_header, _, old_digest = MIO.read_map(old["map_path"], old["map_sha256"])
        old_entry = {"policy": old["map_policy"], "path": old["map_path"], "sha256": old_digest,
                     "type_block": old["type_block"], "total_tiles": old["expected_total_tiles"]}
        plan = [plan_entry("p50_old_first_order_ce_kl", old_entry)]
        plan.extend(plan_entry(name, manifest[name]) for name in order)
        selected = [int(MIO.read_map(e["map_path"], e["map_sha256"])[0]["totals"]["selected_weights"])
                    for e in plan]
        if len(set(selected)) != 1:
            raise RuntimeError(f"P50 selected-weight budget mismatch: {model}")
        path = CR / "plans" / f"P50_selector_{model}.json"
        write_new(path, plan)
        rows.append({"model": model, "map_run_id": run.name,
                     "map_manifest_path": str(manifest_path),
                     "map_manifest_sha256": runtime.sha256_file(manifest_path),
                     "p41_selected_map_sha256": old_digest, "selected_weights": selected[0],
                     "plan_path": str(path), "plan_sha256": runtime.sha256_file(path),
                     "entries": len(plan), "policy_order": [e["name"] for e in plan]})
    manifest = {"schema_version": "1.0", "status": "LOCKED_BEFORE_P50_QUALITY",
                "protocol_sha256": PROTOCOL,
                "selector_spec_sha256": runtime.sha256_file(CR / "provenance/P50_P51_SELECTOR_FIDELITY_SPEC.json"),
                "gate_spec_sha256": runtime.sha256_file(CR / "provenance/P50_G6_OPERATIONALIZATION.json"),
                "p41_selection_sha256": runtime.sha256_file(p41_path), "plans": rows}
    manifest_path = CR / "provenance/P50_PLAN_MANIFEST.json"
    write_new(manifest_path, manifest)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P50-plan-freeze", "model_revision": runtime.sha256_file(manifest_path),
                   "tokenizer_revision": "multiple", "model_class": "plan_builder",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(manifest_path)] + [row["plan_path"] for row in rows],
                    "summary": {"plans": len(rows), "entries": sum(row["entries"] for row in rows)},
                    "uncertainty": {}, "attempted_endpoints": ["P50_PLAN_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "manifest": str(manifest_path),
                      "plans": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
