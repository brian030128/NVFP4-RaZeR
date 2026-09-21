"""Gate, build, enqueue, and analyze the single authorized P60/P61 combination."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import mapio as MIO
from campaign import runtime
from campaign import stats as S
from campaign.extension_p50_maps import MODELS, latest_complete, load
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
DOMAINS = ("wiki", "c4")
ARMS = ("p63_baseline", "p63_weight_only", "p63_activation_only", "p63_combined")


def write_new(path, value, readonly=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    if readonly:
        path.chmod(0o444)


def map_entry(name, row, multiplier, activation):
    header, _, digest = MIO.read_map(row["map_path"], row["map_sha256"])
    return {"name": name, "kind": "map", "map_policy": row["map_policy"],
            "map_path": row["map_path"], "map_sha256": digest,
            "type_block": row["type_block"], "protocol_id": header["protocol_id"],
            "expected_total_tiles": row["expected_total_tiles"],
            "weight_scale_multiplier": float(multiplier),
            "activation_kind": activation}


def build_plans(p60, p61, selector):
    multiplier = float(p60["selected_multiplier"])
    activation = p61["selected_activation_kind"]
    rows = []
    for model in MODELS:
        m = selector["selected_maps"][model]
        plan = [map_entry("p63_baseline", m, 1.0, "four_over_six_rows"),
                map_entry("p63_weight_only", m, multiplier, "four_over_six_rows"),
                map_entry("p63_activation_only", m, 1.0, activation),
                map_entry("p63_combined", m, multiplier, activation)]
        path = CR / "plans" / f"P63_combination_{model}.json"
        write_new(path, plan, readonly=True)
        rows.append({"model": model, "path": str(path),
                     "sha256": runtime.sha256_file(path), "entries": 4,
                     "map_sha256": m["map_sha256"]})
    return rows


def quality_spec(model, dependency):
    return {"job_id": f"P63_quality_{model}", "matrix_id": "P60_WEIGHT_SCALE+P61_ACT_SCALE_CLIP",
            "protocol_id": "ppl-improvement-extension-v1", "gpus": 1,
            "gpu_model": "ada", "env": "main", "cpus": 16, "memory": "150g",
            "priority": 68, "max_invalid_retries": 3, "depends_on": [dependency],
            "command": ["-m", "campaign.evaluate_ppl", "--model", model,
                        "--plan", str(CR / "plans" / f"P63_combination_{model}.json"),
                        "--domains", "wiki,c4", "--length", "2048",
                        "--protocol-id", "ppl-improvement-extension-v1",
                        "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
                        "--freeze-sha256", PROTOCOL, "--teacher", "none",
                        "--attn", "sdpa", "--no-token-arrays"]}


def gate():
    p60_path, p61_path = (CR / "provenance/P60_SELECTION.json",
                          CR / "provenance/P61_SELECTION.json")
    selector_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    p60, p61, selector = load(p60_path), load(p61_path), load(selector_path)
    authorized = (float(p60["selected_multiplier"]) != 1.0
                  and p61["selected_label"] != "baseline_absmax")
    plans, generated = [], []
    if authorized:
        plans = build_plans(p60, p61, selector)
        previous = "P63_gate"
        for model in MODELS:
            spec = quality_spec(model, previous)
            path = CR / "queue/jobs" / f"{spec['job_id']}.json"
            write_new(path, spec)
            generated.append({"path": str(path), "sha256": runtime.sha256_file(path)})
            previous = spec["job_id"]
        spec = {"job_id": "P63_analysis", "matrix_id": "P60_WEIGHT_SCALE+P61_ACT_SCALE_CLIP",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 16, "memory": "32g",
                "priority": 69, "max_invalid_retries": 0, "depends_on": [previous],
                "command": ["-m", "campaign.extension_p63_combination", "analyze"]}
        path = CR / "queue/jobs/P63_analysis.json"
        write_new(path, spec); generated.append({"path": str(path),
                                                  "sha256": runtime.sha256_file(path)})
        spec = {"job_id": "P63_terminal", "matrix_id": "P60_WEIGHT_SCALE+P61_ACT_SCALE_CLIP",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 2, "memory": "4g",
                "priority": 70, "max_invalid_retries": 0,
                "depends_on": ["P63_analysis"],
                "command": ["-m", "campaign.extension_terminal", "--stage", "p63"]}
        path = CR / "queue/jobs/P63_terminal.json"
        write_new(path, spec); generated.append({"path": str(path),
                                                  "sha256": runtime.sha256_file(path)})
        status = "authorized_enqueued"
    else:
        status = "stopped_by_gate"
        spec = {"job_id": "P63_terminal", "matrix_id": "P60_WEIGHT_SCALE+P61_ACT_SCALE_CLIP",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 2, "memory": "4g",
                "priority": 70, "max_invalid_retries": 0,
                "depends_on": ["P63_gate"],
                "command": ["-m", "campaign.extension_terminal", "--stage", "p63"]}
        path = CR / "queue/jobs/P63_terminal.json"
        write_new(path, spec); generated.append({"path": str(path),
                                                  "sha256": runtime.sha256_file(path)})
    out = {"schema_version": "1.0", "status": status,
           "protocol_sha256": PROTOCOL, "authorized": authorized,
           "P60_selection_sha256": runtime.sha256_file(p60_path),
           "P61_selection_sha256": runtime.sha256_file(p61_path),
           "selector_selection_sha256": runtime.sha256_file(selector_path),
           "operationalization_sha256": runtime.sha256_file(
               CR / "provenance/P63_COMBINATION_OPERATIONALIZATION.json"),
           "plans": plans, "generated_queue_specs": generated,
           "reason": ("Both single axes passed their predeclared quality rules"
                      if authorized else
                      "At least one single axis did not pass its predeclared quality rule")}
    path = CR / "provenance/P63_COMBINATION_GATE.json"
    write_new(path, out, readonly=True)
    job_result("P63-combination-gate", [path] + [Path(x["path"]) for x in plans]
               + [Path(x["path"]) for x in generated], out)
    print(json.dumps(out, sort_keys=True))


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, method, comparator, meta, model, domain):
    a, ta = arrays(report, method, domain)
    b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P63 pairing failed: {model}/{domain}/{method}")
    label = f"P63:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def choose(summaries, tolerance=0.0001):
    eligible = [row for row in summaries if row["safe"]]
    if not eligible:
        return next(row for row in summaries if row["arm"] == "p63_baseline")
    best = min(row["median_six_cells"] for row in eligible)
    pool = [row for row in eligible if row["median_six_cells"] <= best + tolerance]
    worst = min(row["worst_cell"] for row in pool)
    pool = [row for row in pool if row["worst_cell"] == worst]
    order = {name: index for index, name in enumerate(ARMS)}
    return min(pool, key=lambda row: order[row["arm"]])


def analyze():
    gate_path = CR / "provenance/P63_COMBINATION_GATE.json"
    gate_info = load(gate_path)
    if not gate_info.get("authorized"):
        raise RuntimeError("P63 analysis cannot run after a stopped gate")
    reports, runs, raw = {}, {}, []
    for model in MODELS:
        run = latest_complete(f"P63_quality_{model}")
        report = load(run / "ppl/ppl_report.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"unaccepted P63 report: {model}")
        validate_policy_contract(report, ARMS, f"P63 {model}")
        reports[model], runs[model] = report, run
        for arm in ARMS:
            for domain in DOMAINS:
                ev = report["evaluation"][arm][domain]
                raw.append({"model": model, "corpus": domain, "policy": arm,
                            "raw_ppl": ev["ppl"], "mean_nll": ev["mean_nll"],
                            "run_id": run.name})
    summaries = []
    for arm in ARMS:
        cells = []
        if arm != "p63_baseline":
            for model in MODELS:
                for domain in DOMAINS:
                    cells.append(contrast(reports[model], arm, "p63_baseline",
                                          load(runs[model] / f"ppl/windows_{domain}.json"),
                                          model, domain))
        values = [row["estimate"] for row in cells] or [0.0] * 6
        summaries.append({"arm": arm, "safe": max(values) <= 0.001,
                          "median_six_cells": float(np.median(values)),
                          "worst_cell": float(max(values)), "contrasts": cells})
    selected = choose(summaries)
    deployment_compatible = selected["arm"] in ("p63_baseline", "p63_weight_only")
    analysis = {"schema_version": "1.0", "status": "complete",
                "protocol_sha256": PROTOCOL,
                "gate_sha256": runtime.sha256_file(gate_path),
                "bootstrap_replicates": B, "candidate_summaries": summaries,
                "selected_quality_arm": selected["arm"],
                "selected_quality_arm_G7_compatible": deployment_compatible,
                "heldout_eligible_arm": "p63_weight_only",
                "heldout_eligibility_reason": "P61-added operations are unfused; the P60 selection independently passed and is the only non-baseline compatible axis.",
                "development_only": True, "native_performance_claim": "forbidden"}
    out = runtime.out_dir("analysis_p63")
    path = out / "P63_COMBINATION_ANALYSIS.json"; runtime.atomic_json(path, analysis)
    csv_path = out / "P63_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    freeze = {"schema_version": "1.0", "status": "FROZEN_AFTER_P63_DEVELOPMENT",
              "protocol_sha256": PROTOCOL,
              "selected_quality_arm": selected["arm"],
              "selected_quality_arm_G7_compatible": deployment_compatible,
              "heldout_eligible_arm": "p63_weight_only",
              "analysis_path": str(path), "analysis_sha256": runtime.sha256_file(path)}
    freeze_path = CR / "provenance/P63_COMBINATION_SELECTION.json"
    write_new(freeze_path, freeze, readonly=True)
    job_result("P63-combination-analysis", [path, csv_path, freeze_path], analysis)
    print(json.dumps(analysis, sort_keys=True))


def job_result(model_id, paths, summary):
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": model_id,
                   "model_revision": runtime.sha256_file(paths[0]),
                   "tokenizer_revision": "not_applicable", "model_class": "gate_or_analysis",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path) for path in paths], "summary": summary,
                    "uncertainty": {"bootstrap_replicates": B if "analysis" in model_id else 0},
                    "attempted_endpoints": ["P60_P61_COMBINATION"],
                    "missing_endpoints": []}, "logs": [], "failures": []})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("action", choices=("gate", "analyze")); a = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    gate() if a.action == "gate" else analyze()


if __name__ == "__main__":
    main()
