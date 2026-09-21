"""Independently assemble the extension G0 integrity/anchor gate."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def preflight_summary(run: Path) -> dict:
    rows = []
    for path in sorted((run / "preflight").glob("*.json")):
        record = load(path)
        rows.append({"path": path.name, "phase": record["phase"], "passed": record["passed"],
                     "reasons": record["reasons"]})
    launch = load(run / "launch_record.json")
    return {"records": rows, "record_count": len(rows), "all_passed": bool(rows) and all(x["passed"] for x in rows),
            "sidecar": launch.get("sidecar"), "invalid_gpu_cotenancy": launch.get("invalid_gpu_cotenancy"),
            "leased_uuids": launch.get("leased_uuids"), "gpu_hours": launch.get("gpu_hours")}


def anchor(model: str, subset_run: str, full_run: str, parent_run: str, expected_map: str,
           refs: dict) -> dict:
    parent = PR / "runs" / parent_run
    parent_report = load(parent / "ppl/ppl_report.json")
    parent_windows = load(parent / "ppl/windows_wiki.json")
    stages = {}
    for stage, run_id in (("first4", subset_run), ("full", full_run)):
        run = CR / "runs" / run_id
        report = load(run / "ppl/ppl_report.json")
        record = load(run / "run_record.json")
        windows = load(run / "ppl/windows_wiki.json")
        expected_tokens = parent_windows["token_sha256"][:4] if stage == "first4" else parent_windows["token_sha256"]
        token_hashes_equal = windows["token_sha256"] == expected_tokens
        static_meta_equal = all(windows[k] == parent_windows[k] for k in parent_windows if k != "token_sha256")
        policies = {}
        for policy in ("four_over_six", "n16_k3"):
            got = report["evaluation"][policy]["wiki"]
            ref = refs[stage][policy]
            relative = got["ppl"] / ref["raw_ppl"] - 1.0
            policies[policy] = {"mean_nll": got["mean_nll"], "raw_ppl": got["ppl"],
                                "parent_mean_nll": ref["mean_nll"], "parent_raw_ppl": ref["raw_ppl"],
                                "nll_difference": got["mean_nll"] - ref["mean_nll"],
                                "relative_ppl_difference": relative, "passed": abs(relative) <= 0.005}
        effect = report["evaluation"]["n16_k3"]["wiki"]["mean_nll"] - report["evaluation"]["four_over_six"]["wiki"]["mean_nll"]
        parent_effect = refs[stage]["dlogppl"]
        tolerance = max(0.25 * abs(parent_effect), 0.002)
        effect_pass = math.copysign(1, effect) == math.copysign(1, parent_effect) and abs(effect - parent_effect) <= tolerance
        map_install = next(x for x in report["installs"] if x["name"] == "n16_k3")
        pf = preflight_summary(run)
        passed = (report["status"] == record["status"] == "complete" and token_hashes_equal and static_meta_equal
                  and all(x["passed"] for x in policies.values()) and effect_pass
                  and map_install["map_sha256"] == expected_map and map_install["map_reloaded_for_evaluation"]
                  and pf["all_passed"] and not pf["invalid_gpu_cotenancy"]
                  and load(run / "run_record_validation.json")["valid"])
        stages[stage] = {
            "run_id": run_id, "status": report["status"], "run_record_status": record["status"],
            "report_sha256": sha(run / "ppl/ppl_report.json"),
            "evaluation_manifest_sha256": report["evaluation_manifest_sha256"],
            "parent_combined_wiki_c4_manifest_sha256": parent_report["evaluation_manifest_sha256"],
            "manifest_scope_note": "anchor manifest is WikiText-only; parent manifest combines WikiText+C4, so component token hashes rather than the combined digest are compared",
            "token_hashes_equal_to_parent_component": token_hashes_equal,
            "static_window_metadata_equal_to_parent": static_meta_equal,
            "map_sha256": map_install["map_sha256"], "expected_map_sha256": expected_map,
            "map_reloaded_for_evaluation": map_install["map_reloaded_for_evaluation"],
            "policies": policies,
            "paired_effect": {"dlogppl": effect, "parent_dlogppl": parent_effect,
                              "absolute_difference": abs(effect - parent_effect), "tolerance": tolerance,
                              "sign_equal": math.copysign(1, effect) == math.copysign(1, parent_effect),
                              "passed": effect_pass},
            "gpu_evidence": pf, "passed": passed,
        }
    return {"model": model, "map_sha256": expected_map, "stages": stages,
            "passed": all(x["passed"] for x in stages.values())}


def main() -> None:
    protocol_path = CR / "freeze/PROTOCOL_EXTENSION.json"
    protocol_sha = sha(protocol_path)
    sidecar_sha = (CR / "freeze/PROTOCOL_EXTENSION.sha256").read_text().split()[0]
    if protocol_sha != sidecar_sha:
        raise RuntimeError("protocol detached hash mismatch")
    input_audit = load(CR / "runs/P00_input_audit_attempt3/input_audit/READ_ONLY_INPUT_AUDIT.json")
    heldout_audit = load(CR / "runs/P00_heldout_input_audit_attempt2/heldout_input_audit/HELDOUT_INPUT_AUDIT.json")
    gpu_probe_run = CR / "runs/P00_gpu_preflight_a6000_attempt2"
    probe_pf = preflight_summary(gpu_probe_run)
    anchors = [
        anchor("llama8b", "P01_anchor_llama8b_subset_attempt1", "P01_anchor_llama8b_full_attempt1",
               "V31_ppl_primary_llama8b_attempt1",
               "0920f55ddc053a5f5a8b0d046d0b74a62d4abafcad5e36051d7682da961e1f2b",
               {"first4": {"four_over_six": {"mean_nll": 2.0610900399817313, "raw_ppl": 7.854526894041561},
                            "n16_k3": {"mean_nll": 2.0593648955835695, "raw_ppl": 7.840988382271653},
                            "dlogppl": -0.0017251443981618486},
                "full": {"four_over_six": {"mean_nll": 1.9279499304938719, "raw_ppl": 6.8754007343996815},
                         "n16_k3": {"mean_nll": 1.92326091483662, "raw_ppl": 6.843237338969587},
                         "dlogppl": -0.004689015657251883}}),
        anchor("mistral7b", "P01_anchor_mistral7b_subset_attempt1", "P01_anchor_mistral7b_full_attempt1",
               "V62_ppl_primary_mistral7b_attempt1",
               "0c3d822a18d0480ca2aeff0abd78cf6e4ca3155bb207156a400fde124e1cf13d",
               {"first4": {"four_over_six": {"mean_nll": 1.8626170782621378, "raw_ppl": 6.440570210901774},
                            "n16_k3": {"mean_nll": 1.8609998059798598, "raw_ppl": 6.430162473558694},
                            "dlogppl": -0.0016172722822780372},
                "full": {"four_over_six": {"mean_nll": 1.708977147744794, "raw_ppl": 5.5233090581954585},
                         "n16_k3": {"mean_nll": 1.7050881772071924, "raw_ppl": 5.50187078542319},
                         "dlogppl": -0.0038889705376015105}}),
    ]
    probe_record = load(gpu_probe_run / "run_record.json")
    probe_passed = (probe_record["status"] == "complete" and probe_pf["all_passed"]
                    and not probe_pf["invalid_gpu_cotenancy"]
                    and load(gpu_probe_run / "run_record_validation.json")["valid"])
    failed_preserved = [
        {"run_id": "P00_input_audit_attempt1", "reason": "Docker CLI was unavailable inside the audit container"},
        {"run_id": "P00_input_audit_attempt2", "reason": "Mistral tokenizer.model.v3 had not yet been rehydrated when scanned"},
        {"run_id": "P00_heldout_input_audit_attempt1", "reason": "network audit was launched with HF_HUB_OFFLINE=1"},
        {"run_id": "P00_gpu_preflight_a6000_attempt1", "reason": "result writer referenced a nonexistent runtime attribute after all GPU checks passed"},
    ]
    passed = (protocol_sha == sidecar_sha and input_audit["g0_input_audit"]["passed"]
              and heldout_audit["all_compatibility_passed"] and probe_passed
              and all(x["passed"] for x in anchors))
    report = {
        "schema_version": "1.0", "gate": "G0", "status": "pass" if passed else "fail",
        "passed": passed, "protocol_sha256": protocol_sha,
        "inputs": {"parent_and_handoff_passed": input_audit["g0_input_audit"]["passed"],
                   "heldout_compatibility_passed": heldout_audit["all_compatibility_passed"],
                   "heldout_reservation_sha256": heldout_audit["reservation_sha256"]},
        "gpu_probe": {"run_id": gpu_probe_run.name, "passed": probe_passed, "evidence": probe_pf},
        "anchors": anchors, "failed_attempts_preserved": failed_preserved,
        "large_matrix_authorized": passed,
    }
    out = runtime.out_dir("g0")
    report_path = out / "G0_INTEGRITY_GATE.json"
    runtime.atomic_json(report_path, report)
    lines = ["# G0 integrity gate", "", f"Status: **{'PASS' if passed else 'FAIL'}**", "",
             f"Protocol SHA-256: `{protocol_sha}`", "",
             f"Input audit: `{input_audit['g0_input_audit']['passed']}`; held-out compatibility: `{heldout_audit['all_compatibility_passed']}`; GPU probe: `{probe_passed}`.", ""]
    for item in anchors:
        lines.append(f"- {item['model']}: subset `{item['stages']['first4']['passed']}`, full `{item['stages']['full']['passed']}`, exact map `{item['map_sha256']}`")
    lines += ["", "All earlier failed attempts remain in the append-only registry.", ""]
    md_path = out / "G0_INTEGRITY_GATE.md"
    md_path.write_text("\n".join(lines))
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": protocol_sha,
        "source": {"model_id": "g0-integrity-analysis", "model_revision": protocol_sha,
                   "tokenizer_revision": "not_applicable", "model_class": "not_applicable",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(report_path), str(md_path)],
                    "summary": {"gate": "G0", "passed": passed, "large_matrix_authorized": passed},
                    "uncertainty": {}, "attempted_endpoints": ["G0"],
                    "missing_endpoints": [] if passed else ["G0"]},
        "logs": [], "failures": [] if passed else [{"stage": "G0", "error": "integrity gate failed"}],
    })
    print(json.dumps({"gate": "G0", "passed": passed, "report": str(report_path)}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
