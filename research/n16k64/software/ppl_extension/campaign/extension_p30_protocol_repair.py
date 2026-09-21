"""Repair only the protocol labels in the immutable P30 development draw plans.

The parent draw1--draw4 maps are byte-identical inputs whose embedded protocol is
``aligned-robustness``.  The first P30 quality attempt exposed that the hand-built
plan copies labelled those entries ``aligned-primary``.  This job reads every map
header and creates new plans; it never edits a map or the original frozen plans.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL_SHA256 = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("llama8b", "qwen4b", "mistral7b")
OLD_PROTOCOL = "aligned-primary"
HEADER_PROTOCOL = "aligned-robustness"


def load(path):
    return json.loads(Path(path).read_text())


def write_immutable(path, value):
    path = Path(path)
    if path.exists():
        if load(path) != value:
            raise FileExistsError(f"immutable output differs: {path}")
        return
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def corrected_entry(entry, header, digest):
    """Return a copy with only the false protocol label corrected."""
    if entry.get("kind") != "map":
        return dict(entry), None
    if digest != entry.get("map_sha256"):
        raise RuntimeError("P30 map digest differs")
    if header.get("protocol_id") != HEADER_PROTOCOL:
        raise RuntimeError(f"unexpected P30 map-header protocol: {header.get('protocol_id')}")
    if entry.get("protocol_id") != OLD_PROTOCOL:
        raise RuntimeError(f"unexpected P30 plan protocol: {entry.get('protocol_id')}")
    if header.get("policy", {}).get("name") != entry.get("map_policy"):
        raise RuntimeError("P30 map policy differs")
    if header.get("type_block") != entry.get("type_block"):
        raise RuntimeError("P30 map type block differs")
    if header.get("totals", {}).get("total_tiles") != entry.get("expected_total_tiles"):
        raise RuntimeError("P30 map total differs")
    out = dict(entry)
    out["protocol_id"] = HEADER_PROTOCOL
    changed = {key for key in set(entry) | set(out) if entry.get(key) != out.get(key)}
    if changed != {"protocol_id"}:
        raise RuntimeError(f"repair changed scientific fields: {sorted(changed)}")
    return out, {
        "name": entry["name"], "map_sha256": digest,
        "old_plan_protocol_id": OLD_PROTOCOL,
        "map_header_protocol_id": HEADER_PROTOCOL,
        "corrected_plan_protocol_id": HEADER_PROTOCOL,
    }


def repair_model(model, locked):
    old_path = CR / locked["path"]
    if runtime.sha256_file(old_path) != locked["sha256"]:
        raise RuntimeError(f"locked P30 plan digest differs: {model}")
    old_plan = load(old_path)
    if len(old_plan) != 9 or old_plan[0] != {"name": "four_over_six", "kind": "four_over_six"}:
        raise RuntimeError(f"P30 plan shape differs: {model}")
    spec = MOD.REGISTRY[model]
    new_plan, corrections = [], []
    for entry in old_plan:
        if entry.get("kind") != "map":
            new_plan.append(dict(entry))
            continue
        header, _, digest = MIO.read_map(entry["map_path"], entry["map_sha256"])
        ident = header.get("model", {})
        if (ident.get("model_id"), ident.get("revision"), ident.get("tokenizer_revision")) != (
                spec["model_id"], spec["revision"], spec["revision"]):
            raise RuntimeError(f"P30 map model identity differs: {model}/{entry['name']}")
        fixed, correction = corrected_entry(entry, header, digest)
        new_plan.append(fixed)
        corrections.append(correction)
    if len(corrections) != 8:
        raise RuntimeError(f"P30 correction cardinality differs: {model}")
    new_path = CR / "plans" / f"P30_k2_draws_{model}_protocol_repaired.json"
    write_immutable(new_path, new_plan)
    return {
        "model": model,
        "original_plan_path": old_path.relative_to(CR).as_posix(),
        "original_plan_sha256": runtime.sha256_file(old_path),
        "corrected_plan_path": new_path.relative_to(CR).as_posix(),
        "corrected_plan_sha256": runtime.sha256_file(new_path),
        "entry_count": len(new_plan), "protocol_corrections": corrections,
        "scientific_fields_changed": False,
        "binary_maps_rewritten": False,
    }


def main():
    freeze = CR / "freeze/PROTOCOL_EXTENSION.json"
    if runtime.sha256_file(freeze) != PROTOCOL_SHA256:
        raise SystemExit("protocol digest mismatch")
    locked_path = CR / "provenance/P30_DEV_PLAN_MANIFEST.json"
    locked = load(locked_path)
    models = [repair_model(model, locked["plans"][model]) for model in MODELS]
    manifest = {
        "schema_version": "1.0",
        "status": "LOCKED_AFTER_METADATA_REPAIR_BEFORE_P30_RETRY",
        "protocol_sha256": PROTOCOL_SHA256,
        "original_manifest_path": locked_path.relative_to(CR).as_posix(),
        "original_manifest_sha256": runtime.sha256_file(locked_path),
        "failure_run_id": "P30_draw_quality_llama8b_attempt1",
        "failure": "MapVerificationError: protocol aligned-robustness != aligned-primary",
        "result_exposure_before": "Only the same-run FourOverSix Wiki/C4 baseline completed; the first map was rejected before installation or evaluation.",
        "repair_rule": "For each exact map, copy the embedded protocol_id into a new plan after SHA-256, policy, type-block, total-tile, and model-identity validation; change no other field.",
        "models": models,
        "protocol_corrections": sum(len(row["protocol_corrections"]) for row in models),
        "binary_maps_rewritten": False,
        "original_plans_modified": False,
    }
    manifest_path = CR / "provenance/P30_DEV_PLAN_PROTOCOL_REPAIR.json"
    write_immutable(manifest_path, manifest)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL_SHA256,
        "source": {"model_id": "P30-development-plan-protocol-repair",
                   "model_revision": runtime.sha256_file(manifest_path),
                   "tokenizer_revision": "multiple", "model_class": "metadata_repair",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(manifest_path)] +
                                    [str(CR / row["corrected_plan_path"]) for row in models],
                    "summary": {"plans": 3, "entries": 27,
                                "protocol_corrections": 24,
                                "binary_maps_rewritten": False,
                                "scientific_fields_changed": False},
                    "uncertainty": {},
                    "attempted_endpoints": ["P30_PLAN_PROTOCOL_METADATA_REPAIR"],
                    "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps({"status": "complete", "manifest": str(manifest_path),
                      "protocol_corrections": 24}, sort_keys=True))


if __name__ == "__main__":
    main()
