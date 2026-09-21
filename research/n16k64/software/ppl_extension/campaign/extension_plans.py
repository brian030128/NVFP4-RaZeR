"""Build immutable exact-map evaluation plans from accepted derivation attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("llama8b", "qwen4b", "mistral7b")


def load(path):
    return json.loads(Path(path).read_text())


def latest(prefix):
    found = []
    for d in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            a = int(d.name.rsplit("_attempt", 1)[1])
            lr = load(d / "launch_record.json")
            st = load(d / "job_status.json")
            rv = load(d / "run_record_validation.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if lr.get("status") == st.get("status") == "complete" and rv.get("valid"):
            found.append((a, d))
    if not found:
        raise FileNotFoundError(prefix)
    return max(found)[1]


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def map_entry(row):
    return {"name": row["policy"], "kind": "map", "map_policy": row["policy"],
            "map_path": row["path"], "map_sha256": row["sha256"],
            "type_block": row["type_block"], "protocol_id": "ppl-improvement-extension-v1",
            "expected_total_tiles": row["total_tiles"]}


def stage_config(stage, model):
    if stage == "p40":
        return (f"P40_maps_{model}", f"P40_lower_k_{model}.json",
                 [{"name": "four_over_six", "kind": "four_over_six"},
                 {"name": "all_e0m3", "kind": "all_e0m3"}], "derived_maps/map_manifest.json")
    if stage == "p20":
        return (f"P20_maps_{model}", f"P20_density_{model}.json",
                [{"name": "n16_k2_reference", "kind": "map", "from_parent": True}], "derived_maps/map_manifest.json")
    if stage == "p41":
        return (f"P41_maps_{model}", f"P41_aggregation_{model}.json",
                [{"name": "n16_k2_reference", "kind": "map", "from_parent": True}], "derived_maps/map_manifest.json")
    if stage in ("p31", "p32"):
        return (f"{stage.upper()}_calib_{model}", f"{stage.upper()}_calibration_size_{model}.json",
                [{"name": "four_over_six", "kind": "four_over_six"}], "calibration_size/map_manifest.json")
    raise ValueError(stage)


def parent_reference(model):
    runs = {"llama8b": "V30_calib_llama8b_seed0_attempt1",
            "qwen4b": "V30_calib_qwen4b_seed0_attempt1",
            "mistral7b": "V61_calib_mistral7b_seed0_attempt2"}
    root = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
    manifest = {x["policy"]: x for x in load(root / "runs" / runs[model] / "calibration/map_manifest.json")}
    row = manifest["n16_k2"]
    return {"name": "n16_k2_reference", "kind": "map", "map_policy": "n16_k2",
            "map_path": row["path"], "map_sha256": row["sha256"], "type_block": row["type_block"],
            "protocol_id": "aligned-primary", "expected_total_tiles": row["total_tiles"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("p20", "p31", "p32", "p40", "p41"))
    a = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    rows = []
    for model in MODELS:
        prefix, filename, base, manifest_relative = stage_config(a.stage, model)
        run = latest(prefix)
        manifest_path = run / manifest_relative
        manifest = load(manifest_path)
        plan = []
        for e in base:
            plan.append(parent_reference(model) if e.get("from_parent") else e)
        plan.extend(map_entry(row) for row in manifest)
        path = CR / "plans" / filename
        write_new(path, plan)
        rows.append({"model": model, "map_run_id": run.name,
                     "map_manifest_path": str(manifest_path), "map_manifest_sha256": runtime.sha256_file(manifest_path),
                     "plan_path": str(path), "plan_sha256": runtime.sha256_file(path), "entries": len(plan)})
    manifest = {"schema_version": "1.0", "status": "LOCKED_BEFORE_QUALITY",
                "stage": a.stage.upper(), "protocol_sha256": PROTOCOL, "plans": rows}
    manifest_path = CR / "provenance" / f"{a.stage.upper()}_PLAN_MANIFEST.json"
    write_new(manifest_path, manifest)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{a.stage}-plan-freeze", "model_revision": runtime.sha256_file(manifest_path),
                   "tokenizer_revision": "multiple", "model_class": "plan_builder",
                   "module_manifest_sha256": "not_applicable", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(manifest_path)] + [x["plan_path"] for x in rows],
                    "summary": {"stage": a.stage, "plans": len(rows), "entries": sum(x["entries"] for x in rows)},
                    "uncertainty": {}, "attempted_endpoints": [f"{a.stage.upper()}_PLAN_FREEZE"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"stage": a.stage, "manifest": str(manifest_path), "plans": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
