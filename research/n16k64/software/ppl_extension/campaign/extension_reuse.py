"""Resolve and verify every parent artifact reused by the extension."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from campaign import mapio
from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])

MODELS = {
    "llama8b": {"map_run": "V30_calib_llama8b_seed0_attempt1", "ppl_run": "V31_ppl_primary_llama8b_attempt1", "k2_run": "V40_ppl_ksweep_llama8b_attempt1"},
    "qwen4b": {"map_run": "V30_calib_qwen4b_seed0_attempt1", "ppl_run": "V31_ppl_primary_qwen4b_attempt1", "k2_run": "V40_ppl_ksweep_qwen4b_attempt1"},
    "mistral7b": {"map_run": "V61_calib_mistral7b_seed0_attempt2", "ppl_run": "V62_ppl_primary_mistral7b_attempt1", "k2_run": "V40_ppl_ksweep_mistral7b_attempt1"},
    "qwen27b": {"map_run": "V30_calib_qwen27b_seed0_attempt1", "ppl_run": "V31_ppl_primary_qwen27b_attempt1", "k2_run": None},
    "phi4": {"map_run": "V61_calib_phi4_seed0_attempt1", "ppl_run": "V62_ppl_primary_phi4_attempt3", "k2_run": None},
    "olmo2_13b": {"map_run": "V61_calib_olmo2_13b_seed0_attempt2", "ppl_run": "V62_ppl_primary_olmo2_13b_attempt3", "k2_run": None},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def manifest_rows(path: Path) -> dict:
    return {row["policy"]: row for row in load(path)}


def inspect_model(model: str, spec: dict) -> dict:
    map_run = PR / "runs" / spec["map_run"]
    map_manifest = manifest_rows(map_run / "calibration/map_manifest.json")
    maps = {}
    for resolution in ("n8", "n16"):
        for kval in (2, 3):
            policy = f"{resolution}_k{kval}"
            row = map_manifest[policy]
            path = Path(row["path"])
            header, _, digest = mapio.read_map(path, expected_sha256=row["sha256"])
            maps[policy] = {
                "logical_path": f"parent:{path.relative_to(PR).as_posix()}", "source_path": str(path),
                "sha256": digest, "bytes": path.stat().st_size, "protocol_id": header["protocol_id"],
                "policy": header["policy"], "type_block": header["type_block"], "totals": header["totals"],
                "source_manifest_sha256": header["source_manifest_sha256"],
                "manifest_row_matches": (digest == row["sha256"] and header["totals"]["total_tiles"] == row["total_tiles"]),
            }
    ppl_run = PR / "runs" / spec["ppl_run"]
    ppl = load(ppl_run / "ppl/ppl_report.json")
    ppl_record = load(ppl_run / "run_record.json")
    required = ["bf16", "nvfp4", "four_over_six", "all_e0m3", "n8_k3", "n16_k3"]
    baseline = {
        "run_id": spec["ppl_run"], "status": ppl["status"],
        "report_path": f"parent:runs/{spec['ppl_run']}/ppl/ppl_report.json",
        "report_sha256": sha(ppl_run / "ppl/ppl_report.json"),
        "evaluation_manifest_sha256": ppl["evaluation_manifest_sha256"],
        "source": ppl_record["source"], "environment": ppl_record["environment"],
        "required_policies": required, "all_required_present": all(x in ppl["evaluation"] for x in required),
        "raw": {policy: {domain: {"raw_ppl": ppl["evaluation"][policy][domain]["ppl"],
                                           "mean_nll": ppl["evaluation"][policy][domain]["mean_nll"],
                                           "windows": len(ppl["evaluation"][policy][domain]["windows"])}
                           for domain in ("wiki", "c4")} for policy in required},
        "window_metadata_sha256": {domain: sha(ppl_run / f"ppl/windows_{domain}.json") for domain in ("wiki", "c4")},
    }
    existing_k2 = None
    if spec["k2_run"]:
        k2_run = PR / "runs" / spec["k2_run"]
        k2 = load(k2_run / "ppl/ppl_report.json")
        k2_record = load(k2_run / "run_record.json")
        exact_pairing = (k2["evaluation_manifest_sha256"] == ppl["evaluation_manifest_sha256"]
                         and all(load(k2_run / f"ppl/windows_{d}.json") == load(ppl_run / f"ppl/windows_{d}.json")
                                 for d in ("wiki", "c4")))
        existing_k2 = {
            "run_id": spec["k2_run"], "status": k2["status"],
            "report_path": f"parent:runs/{spec['k2_run']}/ppl/ppl_report.json",
            "report_sha256": sha(k2_run / "ppl/ppl_report.json"),
            "evaluation_manifest_sha256": k2["evaluation_manifest_sha256"],
            "exact_pairing_with_primary": exact_pairing,
            "source_identity_equal": k2_record["source"]["model_id"] == ppl_record["source"]["model_id"]
                                     and k2_record["source"]["model_revision"] == ppl_record["source"]["model_revision"]
                                     and k2_record["source"]["module_manifest_sha256"] == ppl_record["source"]["module_manifest_sha256"],
            "raw": {policy: {domain: {"raw_ppl": k2["evaluation"][policy][domain]["ppl"],
                                       "mean_nll": k2["evaluation"][policy][domain]["mean_nll"]}
                             for domain in ("wiki", "c4")} for policy in ("four_over_six", "n8_k2", "n16_k2")},
        }
    passed = (ppl["status"] == ppl_record["status"] == "complete" and baseline["all_required_present"]
              and all(m["manifest_row_matches"] for m in maps.values())
              and (existing_k2 is None or (existing_k2["status"] == "complete"
                                           and existing_k2["exact_pairing_with_primary"]
                                           and existing_k2["source_identity_equal"])))
    return {"model": model, "role": "development" if spec["k2_run"] else "breadth_replication",
            "map_run": spec["map_run"], "maps": maps, "parent_baselines": baseline,
            "existing_k2_evidence": existing_k2, "passed": passed}


def main() -> None:
    protocol_sha = (CR / "freeze/PROTOCOL_EXTENSION.sha256").read_text().split()[0]
    if sha(CR / "freeze/PROTOCOL_EXTENSION.json") != protocol_sha:
        raise RuntimeError("protocol lock mismatch")
    models = [inspect_model(name, spec) for name, spec in MODELS.items()]
    report = {
        "schema_version": "1.0", "status": "complete", "protocol_sha256": protocol_sha,
        "parent_logical_path": "research_runs/mixfp4_n16k64_full_validation_20260911T065444Z",
        "reuse_rule": "reuse only identity-exact parent artifacts; every extension evaluation must verify exact maps and prove shared evaluation-window/FourOverSix identity before a cross-run paired contrast",
        "models": models, "model_count": len(models), "all_passed": all(x["passed"] for x in models),
        "development_k2_models_reused": sum(x["existing_k2_evidence"] is not None for x in models),
        "breadth_k2_models_requiring_evaluation": [x["model"] for x in models if x["existing_k2_evidence"] is None],
        "parent_failures_preserved": True,
    }
    out = runtime.out_dir("reuse_inventory")
    path = out / "PARENT_REUSE_INVENTORY.json"
    runtime.atomic_json(path, report)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": protocol_sha,
        "source": {"model_id": "parent-reuse-inventory", "model_revision": sha(PR / "freeze/PROTOCOL_FREEZE.json"),
                   "tokenizer_revision": "multiple", "model_class": "multiple",
                   "module_manifest_sha256": hashlib.sha256("".join(x["parent_baselines"]["source"]["module_manifest_sha256"] for x in models).encode()).hexdigest(),
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path)],
                    "summary": {"all_passed": report["all_passed"], "model_count": len(models),
                                "development_k2_models_reused": report["development_k2_models_reused"],
                                "breadth_k2_models_requiring_evaluation": report["breadth_k2_models_requiring_evaluation"]},
                    "uncertainty": {}, "attempted_endpoints": ["P12_EXISTING_K2_PROVENANCE"],
                    "missing_endpoints": [] if report["all_passed"] else ["P12_EXISTING_K2_PROVENANCE"]},
        "logs": [], "failures": [] if report["all_passed"] else [{"stage": "reuse_inventory", "error": "one or more identity checks failed"}],
    })
    print(json.dumps(report["results"] if "results" in report else {
        "all_passed": report["all_passed"], "development_reused": report["development_k2_models_reused"],
        "breadth_missing": report["breadth_k2_models_requiring_evaluation"]}, sort_keys=True), flush=True)
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
