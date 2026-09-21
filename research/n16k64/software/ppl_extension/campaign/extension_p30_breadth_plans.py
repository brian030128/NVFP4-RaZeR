"""Freeze breadth-model P30 draw-evaluation plans from accepted calibrations.

The four calibration runs are consumed only as score/map artifacts.  This module
checks their identities, exact map hashes, module geometry, and cross-draw document
disjointness before creating the single nine-policy PPL plan for each model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PROTOCOL_ID = "ppl-improvement-extension-v1"
MODELS = ("qwen27b", "phi4", "olmo2_13b")
DRAWS = ("draw1", "draw2", "draw3", "draw4")
SEED_RUNS = {"qwen27b": "V30_calib_qwen27b_seed0_attempt1",
             "phi4": "V61_calib_phi4_seed0_attempt1",
             "olmo2_13b": "V61_calib_olmo2_13b_seed0_attempt2"}


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, validation = (load(run / name) for name in
                                           ("launch_record.json", "job_status.json",
                                            "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and validation.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def document_hashes(meta):
    return {row["document_sha256"] for domain in ("math", "code")
            for row in meta[domain]["documents"]}


def build_model(model):
    plan = [{"name": "four_over_six", "kind": "four_over_six"}]
    seed_manifest_path = PR / "runs" / SEED_RUNS[model] / "calibration/calibration_manifest.json"
    seed_report_path = PR / "runs" / SEED_RUNS[model] / "calibration/calibration_report.json"
    seed_manifest, seed_report = load(seed_manifest_path), load(seed_report_path)
    seed_documents = document_hashes(seed_manifest)
    if len(seed_documents) != 128:
        raise RuntimeError(f"P30 seed0 cardinality differs: {model}")
    inputs, all_documents, module_sha = [], set(seed_documents), seed_report["module_manifest_sha256"]
    for draw in DRAWS:
        run = latest_complete(f"P30_calib_{model}_{draw}")
        report_path = run / "calibration/calibration_report.json"
        manifest_path = run / "calibration/calibration_manifest.json"
        map_manifest_path = run / "calibration/map_manifest.json"
        report, manifest, map_rows = load(report_path), load(manifest_path), load(map_manifest_path)
        if report.get("status") != "complete" or report.get("draw") != draw:
            raise RuntimeError(f"unaccepted P30 calibration report: {model}/{draw}")
        if report.get("model") != model or report.get("freeze_sha256") != PROTOCOL:
            raise RuntimeError(f"P30 calibration identity mismatch: {model}/{draw}")
        if report.get("calibration_manifest_sha256") != runtime.sha256_file(manifest_path):
            # Calibration uses canonical-json hashing, so also accept its explicit
            # canonical digest below; file SHA is deliberately recorded separately.
            from campaign import data as D
            if report.get("calibration_manifest_sha256") != D.manifest_sha256(manifest):
                raise RuntimeError(f"P30 calibration manifest mismatch: {model}/{draw}")
        docs = document_hashes(manifest)
        overlap = docs & all_documents
        if overlap:
            raise RuntimeError(f"P30 calibration documents overlap across draws: {model}/{draw}")
        all_documents |= docs
        current_module_sha = report.get("module_manifest_sha256")
        if module_sha != current_module_sha:
            raise RuntimeError(f"P30 module identity differs across draws: {model}")
        by_policy = {row["policy"]: row for row in map_rows}
        if set(by_policy) < {"n8_k2", "n16_k2"}:
            raise RuntimeError(f"P30 maps missing: {model}/{draw}")
        for policy, type_block in (("n8_k2", [8, 64]), ("n16_k2", [16, 64])):
            row = by_policy[policy]
            header, _, digest = MIO.read_map(row["path"], row["sha256"])
            if digest != row["sha256"] or header["type_block"] != type_block:
                raise RuntimeError(f"P30 map identity mismatch: {model}/{draw}/{policy}")
            plan.append({"name": f"{policy}_{draw}", "kind": "map",
                         "map_path": row["path"], "map_sha256": digest,
                         "map_policy": policy, "protocol_id": header["protocol_id"],
                         "type_block": type_block,
                         "expected_total_tiles": row["total_tiles"]})
        inputs.append({"draw": draw, "run_id": run.name,
                       "calibration_report_sha256": runtime.sha256_file(report_path),
                       "calibration_manifest_sha256": runtime.sha256_file(manifest_path),
                       "calibration_manifest_canonical_sha256": report["calibration_manifest_sha256"],
                       "map_manifest_sha256": runtime.sha256_file(map_manifest_path),
                       "document_count": len(docs)})
    if len(plan) != 9 or len(all_documents) != 5 * 128:
        raise RuntimeError(f"P30 breadth plan cardinality differs: {model}")
    path = CR / "plans" / f"P30_k2_draws_{model}.json"
    write_new(path, plan)
    return {"model": model, "plan_path": str(path),
            "plan_sha256": runtime.sha256_file(path), "entries": len(plan),
            "unique_seed0_and_draw_documents": len(all_documents),
            "seed0_input": {"run_id": SEED_RUNS[model],
                            "calibration_report_sha256": runtime.sha256_file(seed_report_path),
                            "calibration_manifest_sha256": runtime.sha256_file(seed_manifest_path)},
            "module_manifest_sha256": module_sha, "calibration_inputs": inputs}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    rows = [build_model(model) for model in MODELS]
    manifest = {"schema_version": "1.0", "status": "LOCKED_BEFORE_P30_BREADTH_QUALITY",
                "protocol_sha256": PROTOCOL,
                "calibration_plan_sha256": runtime.sha256_file(
                    CR / "provenance/P30_BREADTH_CALIBRATION_PLAN.json"),
                "models": rows}
    manifest_path = CR / "provenance/P30_BREADTH_PLAN_MANIFEST.json"
    write_new(manifest_path, manifest)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": PROTOCOL_ID, "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P30-breadth-plan-freeze",
                   "model_revision": runtime.sha256_file(manifest_path),
                   "tokenizer_revision": "multiple", "model_class": "plan_builder",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "multiple",
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": {"passed": True,
                                   "unique_seed0_and_draw_documents_per_model": 640}},
        "policies": [],
        "results": {"raw_outputs": [str(manifest_path)] + [row["plan_path"] for row in rows],
                    "summary": {"plans": 3, "entries": 27,
                                "unique_seed0_and_draw_documents_per_model": 640},
                    "uncertainty": {}, "attempted_endpoints": ["P30_BREADTH_PLAN_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "plans": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
