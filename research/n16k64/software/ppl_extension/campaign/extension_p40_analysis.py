"""Frozen development-only P40 lower-k analysis and one-global-k selection."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import data as D
from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b")
K_TO_POLICY = {
    0.5: "p40_n16_k0p5_seed0", 1.0: "p40_n16_k1p0_seed0",
    1.5: "p40_n16_k1p5_seed0", 2.0: "p40_n16_k2p0_seed0",
    2.5: "p40_n16_k2p5_seed0", 3.0: "p40_n16_k3p0_seed0",
}
CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5)


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            lr, st, vr = (load(run / name) for name in
                          ("launch_record.json", "job_status.json", "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if lr.get("status") == st.get("status") == "complete" and vr.get("valid"):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def window_arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, method, comparator, meta, model, domain):
    a, ta = window_arrays(report, method, domain)
    b, tb = window_arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"unpaired P40 windows: {model}/{domain}/{method}")
    label = f"P40:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def select_k(summaries, tolerance=0.0001):
    """Apply the frozen staged lexicographic rule, never per-model tuning."""
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise RuntimeError("k=2 anchor must make at least one P40 candidate eligible")
    best_median = min(row["median_six_cells"] for row in eligible)
    stage = [row for row in eligible if row["median_six_cells"] <= best_median + tolerance]
    best_worst = min(row["worst_cell"] for row in stage)
    stage = [row for row in stage if row["worst_cell"] == best_worst]
    least_weights = min(row["total_selected_weights"] for row in stage)
    stage = [row for row in stage if row["total_selected_weights"] == least_weights]
    return max(stage, key=lambda row: row["k"])


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    out = runtime.out_dir("analysis_p40")
    reports, runs, raw_rows, rows = {}, {}, [], []
    selected_weights = {k: 0 for k in K_TO_POLICY}
    for model in MODELS:
        run = latest_complete(f"P40_quality_{model}")
        report_path = run / "ppl/ppl_report.json"
        report = load(report_path)
        if report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P40 PPL report for {model}")
        reports[model], runs[model] = report, run
        installs = {row["name"]: row for row in report["installs"]}
        for k, policy in K_TO_POLICY.items():
            selected_weights[k] += int(installs[policy]["selected_tiles"]) * 1024
        for domain in ("wiki", "c4"):
            meta = load(run / "ppl" / f"windows_{domain}.json")
            for k, policy in K_TO_POLICY.items():
                row = contrast(report, policy, K_TO_POLICY[2.0], meta, model, domain)
                row["k"] = k
                rows.append(row)
            for policy, domains in report["evaluation"].items():
                raw_rows.append({"model": model, "corpus": domain, "policy": policy,
                                 "raw_ppl": domains[domain]["ppl"],
                                 "mean_nll": domains[domain]["mean_nll"], "run_id": run.name})
    summaries = []
    for k in CANDIDATES:
        cells = [row for row in rows if row["k"] == k]
        values = [row["estimate"] for row in cells]
        summaries.append({"k": k, "eligible": all(value <= 0.001 for value in values),
                          "median_six_cells": float(np.median(values)), "worst_cell": float(max(values)),
                          "total_selected_weights": selected_weights[k], "cells": cells})
    chosen = select_k(summaries)
    analysis = {
        "schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
        "derivation_spec_sha256": runtime.sha256_file(CR / "provenance/P40_P41_DERIVATION_SPEC.json"),
        "bootstrap_replicates": B, "development_only": True,
        "run_inputs": {model: {"run_id": runs[model].name,
                                 "report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json")}
                       for model in MODELS},
        "candidate_summaries": summaries,
        "anchors": {"k3": {"total_selected_weights": selected_weights[3.0]},
                    "all_e0m3_and_four_over_six": "raw table"},
        "selection_rule": ["eligible iff all six cells <= +0.001", "smallest median dlogPPL vs k2",
                           "within 0.0001 choose smallest worst cell", "then fewer selected weights",
                           "then larger k"],
        "selected_global_k": chosen["k"], "selected_summary": chosen,
        "post_selection_inference": "No naive confirmatory CI is attached to the development-selected winner.",
    }
    analysis_path = out / "P40_LOWER_K_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P40_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader(); writer.writerows(raw_rows)
    selection = {
        "schema_version": "1.0", "status": "FROZEN_AFTER_DEVELOPMENT_P40_BEFORE_P41_DERIVATION",
        "protocol_sha256": PROTOCOL, "global_k": chosen["k"], "per_model_tuning": False,
        "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path),
        "eligible_candidates": [row["k"] for row in summaries if row["eligible"]],
        "selected_median_six_cells": chosen["median_six_cells"],
        "selected_worst_cell": chosen["worst_cell"],
        "selected_total_weights_across_models": chosen["total_selected_weights"],
    }
    selection_path = CR / "provenance/P40_GLOBAL_K_SELECTION.json"
    write_new(selection_path, selection)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P40-development-analysis", "model_revision": selection["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(selection_path)],
                    "summary": {"selected_global_k": chosen["k"],
                                "eligible_candidates": selection["eligible_candidates"]},
                    "uncertainty": {"bootstrap_replicates": B, "selection_adjusted": False},
                    "attempted_endpoints": ["P40_LOWER_K_SEARCH", "GLOBAL_K_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "selected_global_k": chosen["k"],
                      "selection_sha256": runtime.sha256_file(selection_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
