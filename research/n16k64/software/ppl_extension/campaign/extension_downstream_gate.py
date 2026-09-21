"""Freeze downstream plans and enqueue only P71-authorized conditional work."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_heldout_maps import MODELS as HELDOUT, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PARENT_CAL = {"llama8b": "V30_calib_llama8b_seed0_attempt1",
              "mistral7b": "V61_calib_mistral7b_seed0_attempt2"}


def write_new(path, value, readonly=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    if readonly:
        path.chmod(0o444)


def renamed(entry, name):
    row = dict(entry); row["name"] = name
    return row


def system(name):
    return {"name": "strongest_fair_baseline", "kind": name,
            "baseline_source_policy": name}


def exact_map(name, row, multiplier=1.0):
    path = row.get("map_path", row.get("path")); digest = row.get("map_sha256", row.get("sha256"))
    header, _, got = MIO.read_map(path, digest)
    return {"name": name, "kind": "map", "map_path": path, "map_sha256": got,
            "map_policy": row.get("map_policy", row.get("policy")),
            "type_block": row["type_block"], "protocol_id": header["protocol_id"],
            "expected_total_tiles": row.get("expected_total_tiles", row.get("total_tiles")),
            "weight_scale_multiplier": float(multiplier)}


def heldout_plan(model, decision, purpose):
    source = load(CR / "plans" / f"P71_heldout_ppl_{model}.json")
    by_name = {row["name"]: row for row in source}
    if purpose == "accuracy":
        return [renamed(by_name["frozen_winner"], "frozen_winner"),
                renamed(by_name["four_over_six"], "four_over_six"),
                renamed(by_name["n8_k2"], "n8_k2"),
                renamed(by_name["n16_k3"], "n16_k3"),
                system(decision["strongest_model_baselines"][model])]
    if purpose == "gsm8k":
        return [renamed(by_name["frozen_winner"], "frozen_winner"),
                renamed(by_name["four_over_six"], "four_over_six"),
                system(decision["strongest_model_baselines"][model])]
    if purpose == "pg19":
        return [renamed(by_name["frozen_winner"], "frozen_winner"),
                renamed(by_name["four_over_six"], "four_over_six"),
                renamed(by_name["n8_k2"], "n8_k2"),
                renamed(by_name["n16_k3"], "n16_k3")]
    raise ValueError(purpose)


def existing_pg19_plan(model, winner, p50, p60):
    manifest = {row["policy"]: row for row in load(
        PR / "runs" / PARENT_CAL[model] / "calibration/map_manifest.json")}
    selected = p50["selected_maps"][model]
    return [exact_map("frozen_winner", selected, p60["selected_multiplier"]),
            {"name": "four_over_six", "kind": "four_over_six"},
            exact_map("n8_k2", manifest["n8_k2"]),
            exact_map("n16_k3", manifest["n16_k3"])]


def save_plan(matrix, purpose, model, plan):
    path = CR / "plans" / f"{matrix}_{purpose}_{model}.json"
    write_new(path, plan, readonly=True)
    return {"matrix": matrix, "purpose": purpose, "model": model, "path": str(path),
            "sha256": runtime.sha256_file(path), "policy_order": [row["name"] for row in plan]}


def common_lmeval(model, plan, suite, batch):
    return ["-m", "campaign.evaluate_lmeval", "--model", model, "--plan", plan,
            "--suite", suite, "--batch-size", str(batch),
            "--protocol-id", "ppl-improvement-extension-v1",
            "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
            "--freeze-sha256", PROTOCOL]


def common_pg19(model, plan):
    return ["-m", "campaign.evaluate_ppl", "--model", model, "--plan", plan,
            "--domains", "pg19_4k,pg19_8k", "--length", "2048",
            "--protocol-id", "ppl-improvement-extension-v1",
            "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
            "--freeze-sha256", PROTOCOL, "--teacher", "none", "--attn", "sdpa",
            "--no-token-arrays"]


def gpu_job(jid, matrix, model, dependency, command, priority):
    return {"job_id": jid, "matrix_id": matrix,
            "protocol_id": "ppl-improvement-extension-v1", "gpus": 1,
            "gpu_model": "a6000", "env": "main", "cpus": 16,
            "memory": "170g", "priority": priority, "max_invalid_retries": 3,
            "depends_on": [dependency], "command": command}


def cpu_job(jid, matrix, dependency, module, priority):
    return {"job_id": jid, "matrix_id": matrix,
            "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
            "gpu_model": "any", "env": "main", "cpus": 16, "memory": "48g",
            "priority": priority, "max_invalid_retries": 0,
            "depends_on": [dependency], "command": ["-m", module]}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    decision_path = CR / "provenance/P71_G8_QUALITY_DECISION.json"
    decision = load(decision_path)
    winner_path = CR / "freeze/P70_GLOBAL_WINNER.json"
    winner = load(winner_path)
    p50 = load(CR / "provenance/P50_SELECTOR_SELECTION.json")
    p60 = load(CR / "provenance/P60_SELECTION.json")
    eligible = list(decision["downstream_eligible_models"])
    strong_outliers = list(decision.get("strong_PPL_outlier_models", []))
    gsm_models = list(dict.fromkeys([*eligible, *strong_outliers]))
    authorized = bool(decision["passed"] and eligible)
    plans, jobs, stopped = [], [], []
    if authorized:
        for model in eligible:
            plans.append(save_plan("P72", "accuracy", model,
                                   heldout_plan(model, decision, "accuracy")))
        for model in gsm_models:
            plans.append(save_plan("P73", "gsm8k", model,
                                   heldout_plan(model, decision, "gsm8k")))
        pg_models = ["llama8b", "mistral7b", *eligible]
        for model in pg_models:
            plan = (heldout_plan(model, decision, "pg19") if model in HELDOUT else
                    existing_pg19_plan(model, winner, p50, p60))
            plans.append(save_plan("P74", "pg19", model, plan))
        lookup = {(row["matrix"], row["model"]): row["path"] for row in plans}
        previous = "P71_downstream_gate"
        # Accuracy is restricted to the held-out models that pass the frozen
        # PPL eligibility gate.  GSM8K additionally includes every strong PPL
        # outlier, even when that model is not accuracy-eligible.
        for i, model in enumerate(eligible):
            jid = f"P72_accuracy_{model}"
            jobs.append(gpu_job(jid, "P72_ACCURACY", model, previous,
                                common_lmeval(model, lookup[("P72", model)], "full8", 16), 132 + i))
            previous = jid
        jobs.append(cpu_job("P72_analysis", "P72_ACCURACY", previous,
                            "campaign.extension_downstream_analysis", 134))
        previous = "P72_analysis"
        for i, model in enumerate(gsm_models):
            jid = f"P73_gsm8k_{model}"
            jobs.append(gpu_job(jid, "P73_GSM8K", model, previous,
                                common_lmeval(model, lookup[("P73", model)], "gsm8k", 4), 135 + i))
            previous = jid
        jobs.append(cpu_job("P73_analysis", "P73_GSM8K", previous,
                            "campaign.extension_downstream_analysis", 137))
        # Add the stage selector expected by the shared analysis module.
        jobs[-1]["command"].extend(["--stage", "p73"])
        previous = "P73_analysis"
        for i, model in enumerate(pg_models):
            jid = f"P74_pg19_{model}"
            jobs.append(gpu_job(jid, "P74_PG19", model, previous,
                                common_pg19(model, lookup[("P74", model)]), 138 + i))
            previous = jid
        jobs.append(cpu_job("P74_analysis", "P74_PG19", previous,
                            "campaign.extension_downstream_analysis", 143))
        jobs[-1]["command"].extend(["--stage", "p74"])
        jobs.append(cpu_job("P71_downstream_terminal", "P72_ACCURACY+P73_GSM8K+P74_PG19",
                            "P74_analysis", "campaign.extension_terminal", 144))
        jobs[-1]["command"].extend(["--stage", "downstream"])
        # P72 needs its explicit stage as well.
        next(row for row in jobs if row["job_id"] == "P72_analysis")["command"].extend(["--stage", "p72"])
        for spec in jobs:
            path = CR / "queue/jobs" / f"{spec['job_id']}.json"
            write_new(path, spec)
    else:
        stopped = [{"matrix_id": matrix, "status": "stopped_by_gate",
                    "reason": "P71 heldout quality gate did not authorize downstream execution"}
                   for matrix in ("P72_ACCURACY", "P73_GSM8K", "P74_PG19")]
        jobs.append(cpu_job("P71_downstream_terminal",
                            "P72_ACCURACY+P73_GSM8K+P74_PG19",
                            "P71_downstream_gate", "campaign.extension_terminal", 144))
        jobs[-1]["command"].extend(["--stage", "downstream"])
        spec = jobs[-1]
        write_new(CR / "queue/jobs" / f"{spec['job_id']}.json", spec)
    resolution = {"schema_version": "1.0", "status": "authorized_enqueued" if authorized else "stopped_by_gate",
                  "protocol_sha256": PROTOCOL, "authorized": authorized,
                  "P71_decision_sha256": runtime.sha256_file(decision_path),
                  "winner_freeze_sha256": runtime.sha256_file(winner_path),
                  "eligible_heldout_models": eligible,
                  "strong_PPL_outlier_models": strong_outliers,
                  "gsm8k_models": gsm_models if authorized else [], "plans": plans,
                  "queue_jobs": [{"job_id": row["job_id"],
                                  "sha256": runtime.sha256_file(CR / "queue/jobs" / f"{row['job_id']}.json")}
                                 for row in jobs],
                  "stopped": stopped}
    path = CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json"
    write_new(path, resolution, readonly=True)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P71-downstream-gate",
                   "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "multiple", "model_class": "gate",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path)] + [row["path"] for row in plans],
                    "summary": {"authorized": authorized, "eligible_models": eligible,
                                "jobs": len(jobs)}, "uncertainty": {},
                    "attempted_endpoints": ["P72_P73_P74_GATE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps(resolution, sort_keys=True))


if __name__ == "__main__":
    main()
