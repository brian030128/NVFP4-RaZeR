#!/usr/bin/env python3
"""Outcome-blind one-shot Mistral promotion gate, run immediately before launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pwd
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROTOCOL_SHA256 = "9e7c3d1dcb6199ab9bbedf9e0947f6198304f26466f2fd2c58e5533fb747926c"
MISTRAL_PLAN_SHA256 = "9574bccfa1cb946dafa16521f02decd84d7d88ae5b6edf732c980a6f27917c21"
REPORT_RENDERER_SHA256 = "920d86d02df16de7426481d46be3427b9e8ca1b4c1fcdcf543c63de397f3505a"
REPORTING_PLAN_SHA256 = "527ec0cb608b3b3f5bf7ed03b9aa02ff72aff9ba939c7a65ea0072cd5580b475"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(command: list[str]) -> list[list[str]]:
    text = subprocess.check_output(command, text=True)
    return [[field.strip() for field in row] for row in csv.reader(io.StringIO(text)) if row]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--gpu-uuid", required=True)
    ap.add_argument("--projected-hours", type=float, default=3.5)
    args = ap.parse_args()
    root = Path(args.campaign_root).resolve()
    failures = []
    protocol = load(root / "MECHANISM_PROTOCOL.json")
    if sha256(root / "MECHANISM_PROTOCOL.json") != PROTOCOL_SHA256:
        failures.append("protocol hash mismatch")
    source = root / "source" / "NVFP4-RaZeR-main" / "campaign"
    code_hashes = {name: sha256(source / name) for name in protocol["code_hashes"]}
    if code_hashes != protocol["code_hashes"]:
        failures.append("frozen analysis/source code hash drift")
    if sha256(root / "job_specs" / "mistral7b_mechanism_full.json") != MISTRAL_PLAN_SHA256:
        failures.append("Mistral plan hash drift")
    if sha256(root / "analysis" / "render_reports.py") != REPORT_RENDERER_SHA256:
        failures.append("predeclared report renderer hash drift")
    if sha256(root / "analysis" / "REPORTING_PLAN.json") != REPORTING_PLAN_SHA256:
        failures.append("predeclared reporting plan hash drift")
    development = {}
    for model, run_id, policies, windows in (
            ("llama8b", "V22_full_llama8b_attempt1", 17, {"wiki": 141, "c4": 256}),
            ("qwen4b", "V21_full_qwen4b_attempt2", 17, {"wiki": 146, "c4": 256})):
        run = root / "runs" / run_id
        launch = load(run / "launch_record.json")
        validation = load(run / "run_record_validation.json")
        report = load(run / "ppl" / "ppl_report.json")
        valid = (launch["status"] == "complete" and launch["exit_code"] == 0 and
                 not launch["invalid_gpu_cotenancy"] and validation == {"valid": True, "errors": []} and
                 report["status"] == "complete" and len(report["evaluation"]) == policies and
                 report["windows"] == windows and report["freeze_sha256"] == PROTOCOL_SHA256 and
                 report["reinstall_check"]["identical"] and report["reinstall_check"]["checksum_equal"])
        if not valid:
            failures.append(f"development run not valid: {run_id}")
        development[model] = {
            "run_id": run_id,
            "valid": valid,
            "ppl_report_sha256": sha256(run / "ppl" / "ppl_report.json"),
            "run_record_sha256": sha256(run / "run_record.json"),
            "gpu_hours": launch["gpu_hours"],
        }
    definitions = load(root / "analysis" / "GROUP_DEFINITIONS.json")
    map_invariants = {}
    for model, value in definitions.items():
        ok = (value["anchor_reproduction"] == {"passed": True, "tile_mismatches": 0} and
              value["ranking"]["groups_disjoint"] and value["interaction"]["full_union_exact"] and
              all(v["actual_tiles"] == v["random_tiles"] and v["bins"] == 5
                  for v in value["veto"].values()))
        if not ok:
            failures.append(f"map/accounting invariants failed: {model}")
        map_invariants[model] = ok
    prior_mistral_launches = []
    for run in (root / "runs").iterdir():
        launch_path = run / "launch_record.json"
        if not launch_path.exists():
            continue
        command = load(launch_path).get("command") or []
        if "--model" in command and command[command.index("--model") + 1] == "mistral7b":
            prior_mistral_launches.append(run.name)
    if prior_mistral_launches:
        failures.append(f"Mistral was already launched: {prior_mistral_launches}")
    inventory_rows = parse_csv([
        "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"])
    inventory = {row[1]: {"index": int(row[0]), "uuid": row[1], "name": row[2]} for row in inventory_rows}
    selected = inventory.get(args.gpu_uuid)
    if selected is None or "A6000" not in selected["name"] or "Ada" in selected["name"]:
        failures.append("selected UUID is not an RTX A6000")
    process_rows = parse_csv([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"])
    selected_processes = []
    for row in process_rows:
        if row[0] != args.gpu_uuid:
            continue
        pid = int(row[1])
        try:
            owner = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except (FileNotFoundError, ProcessLookupError):
            owner = "process_exited_during_check"
        selected_processes.append({"gpu_uuid": row[0], "pid": pid, "process_name": row[2],
                                   "used_memory_mib": int(row[3]), "owner": owner})
    if selected_processes:
        failures.append("selected UUID has a compute process; fail closed")
    now = datetime.now(timezone.utc)
    launch_cutoff = datetime.fromisoformat(protocol["hard_cutoffs_utc"]["new_gpu_launches"].replace("Z", "+00:00"))
    projected = now + timedelta(hours=args.projected_hours)
    if now >= launch_cutoff or projected >= launch_cutoff:
        failures.append("projected one-shot completion does not precede hour-20 launch cutoff")
    record = {
        "schema": "mixfp4-mistral-promotion-gate/v1",
        "checked_utc": now.isoformat().replace("+00:00", "Z"),
        "passed": not failures,
        "failures": failures,
        "protocol_sha256": PROTOCOL_SHA256,
        "promotion_gate_code_sha256": sha256(Path(__file__)),
        "frozen_code_hashes": code_hashes,
        "mistral_plan_sha256": MISTRAL_PLAN_SHA256,
        "report_renderer_sha256": REPORT_RENDERER_SHA256,
        "reporting_plan_sha256": REPORTING_PLAN_SHA256,
        "development_runs": development,
        "map_accounting_invariants": map_invariants,
        "group_definitions_sha256": sha256(root / "analysis" / "GROUP_DEFINITIONS.json"),
        "prior_mistral_launches": prior_mistral_launches,
        "selected_gpu": selected,
        "selected_gpu_compute_processes": selected_processes,
        "projected_hours": args.projected_hours,
        "projected_completion_utc": projected.isoformat().replace("+00:00", "Z"),
        "new_gpu_launch_cutoff_utc": protocol["hard_cutoffs_utc"]["new_gpu_launches"],
        "decision": "launch_exactly_once" if not failures else "fail_closed_do_not_launch",
    }
    path = root / "MISTRAL_PROMOTION_GATE.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(json.dumps(record, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
