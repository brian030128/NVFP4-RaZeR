"""Validate that a conditional branch has reached its declared terminal state.

These lightweight CPU sentinels let the unattended queue express dependencies on
jobs that are themselves created only after a frozen gate is evaluated.  A branch
is terminal either when it was stopped by that gate, or when every authorized
analysis artifact is present in a valid completed run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from campaign import runtime
from campaign.extension_heldout_maps import latest_complete, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"

STAGES = {
    "p21": {
        "resolution": "provenance/P21_GATE_RESOLUTION.json",
        "analyses": [("P21_analysis", "analysis_p21/P21_BREADTH_ANALYSIS.json")],
    },
    "p32": {
        "resolution": "provenance/P32_GATE_RESOLUTION.json",
        "analyses": [("P32_analysis", "analysis_p32/P32_CALIBRATION_SIZE_ANALYSIS.json")],
    },
    "p42": {
        "resolution": "provenance/P42_GATE_RESOLUTION.json",
        "analyses": [("P42_analysis", "analysis_p42/P42_BREADTH_ANALYSIS.json")],
    },
    "p52": {
        "resolution": "provenance/P52_GATE_RESOLUTION.json",
        "analyses": [("P52_analysis", "analysis_p52/P52_BREADTH_ANALYSIS.json")],
    },
    "p63": {
        "resolution": "provenance/P63_COMBINATION_GATE.json",
        "analyses": [("P63_analysis", "analysis_p63/P63_COMBINATION_ANALYSIS.json")],
    },
    "downstream": {
        "resolution": "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json",
        "analyses": [
            ("P72_analysis", "analysis_p72/P72_ACCURACY_ANALYSIS.json"),
            ("P73_analysis", "analysis_p73/P73_GSM8K_ANALYSIS.json"),
            ("P74_analysis", "analysis_p74/P74_PG19_ANALYSIS.json"),
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    cfg = STAGES[args.stage]
    resolution_path = CR / cfg["resolution"]
    resolution = load(resolution_path)
    authorized = bool(resolution.get("authorized", resolution.get("continue_to_p32", False)))
    if resolution.get("status") not in ("authorized_enqueued", "stopped_by_gate"):
        raise RuntimeError(f"nonterminal gate resolution: {resolution.get('status')}")
    artifacts = []
    if authorized:
        for job, rel in cfg["analyses"]:
            run = latest_complete(job)
            path = run / rel
            value = load(path)
            if value.get("status") != "complete":
                raise RuntimeError(f"authorized analysis is not complete: {path}")
            artifacts.append({"job_id": job, "run_id": run.name,
                              "path": str(path), "sha256": runtime.sha256_file(path)})
    elif not resolution.get("status") == "stopped_by_gate":
        raise RuntimeError("unauthorized branch lacks stopped_by_gate resolution")
    report = {
        "schema_version": "1.0", "status": "complete", "stage": args.stage,
        "protocol_sha256": PROTOCOL, "authorized": authorized,
        "terminal_status": "complete" if authorized else "stopped_by_gate",
        "resolution": {"path": str(resolution_path),
                       "sha256": runtime.sha256_file(resolution_path)},
        "analysis_artifacts": artifacts,
    }
    out = runtime.out_dir("terminal")
    path = out / f"{args.stage.upper()}_TERMINAL.json"
    runtime.atomic_json(path, report)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{args.stage}-terminal-sentinel",
                   "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "not_applicable", "model_class": "terminal",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path)], "summary": report,
                    "uncertainty": {}, "attempted_endpoints": [f"{args.stage.upper()}_TERMINAL"],
                    "missing_endpoints": []}, "logs": [], "failures": [],
    })
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
