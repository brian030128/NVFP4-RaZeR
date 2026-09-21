"""Prospective held-out PPL analysis after the P70 global-winner freeze."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.extension_heldout_maps import MODELS, latest_complete, load
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
DOMAINS = ("wiki", "c4")
MATERIAL = math.log(1.005)
SYSTEM_ARMS = ("nvfp4", "four_over_six", "all_e0m3", "razer_wonly_shared_act",
               "razer_native_rows", "nover6_wonly_shared_act", "nover6_native_rows")
SHARED_ARMS = ("four_over_six", "all_e0m3", "razer_wonly_shared_act",
               "nover6_wonly_shared_act")


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, method, comparator, meta, model, domain, family):
    a, ta = arrays(report, method, domain)
    b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P71 pairing failed: {model}/{domain}/{method}/{comparator}")
    label = f"P71:{family}:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               family=family, stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def holm(rows):
    adjusted, rejected = S.holm([row["p_two_sided"] for row in rows])
    for row, p, reject in zip(rows, adjusted, rejected):
        row["p_holm"] = p; row["holm_reject_0_05"] = reject


def strongest(report, domain, arms):
    order = {arm: i for i, arm in enumerate(arms)}
    return min(arms, key=lambda arm: (report["evaluation"][arm][domain]["mean_nll"], order[arm]))


def strongest_model(report, arms=SYSTEM_ARMS):
    order = {arm: i for i, arm in enumerate(arms)}
    return min(arms, key=lambda arm: (
        np.mean([report["evaluation"][arm][domain]["mean_nll"] for domain in DOMAINS]),
        order[arm]))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value); path.chmod(0o444)


def main():
    winner_path = CR / "freeze/P70_GLOBAL_WINNER.json"
    winner_sha_path = CR / "freeze/P70_GLOBAL_WINNER.sha256"
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    winner = load(winner_path)
    expected = winner_sha_path.read_text().split()[0]
    if runtime.sha256_file(winner_path) != expected or winner["status"] != "HASH_LOCKED_BEFORE_HELDOUT_QUALITY":
        raise RuntimeError("P70 winner freeze identity failed")
    out = runtime.out_dir("analysis_p71")
    reports, runs, raw = {}, {}, []
    expected_order = ("four_over_six", "nvfp4", "all_e0m3", "n8_k2", "n16_k2",
                      "n16_k3", "frozen_winner", "razer_wonly_shared_act",
                      "razer_native_rows", "nover6_wonly_shared_act", "nover6_native_rows")
    for model in MODELS:
        run = latest_complete(f"P71_quality_{model}")
        report = load(run / "ppl/ppl_report.json")
        if (report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical")):
            raise RuntimeError(f"unaccepted P71 heldout report: {model}")
        validate_policy_contract(report, expected_order, f"P71 {model}")
        frozen = next(row for row in winner["heldout_models"] if row["model"] == model)
        install = next(row for row in report["installs"] if row["name"] == "frozen_winner")
        if install["map_sha256"] != frozen["winner_map_sha256"]:
            raise RuntimeError(f"P71 did not install frozen winner map: {model}")
        reports[model], runs[model] = report, run
        for policy, domains in report["evaluation"].items():
            for domain in DOMAINS:
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": domains[domain]["ppl"],
                            "mean_nll": domains[domain]["mean_nll"], "run_id": run.name})

    fos_rows, system_rows, shared_rows = [], [], []
    selected = {}
    for model in MODELS:
        report, run = reports[model], runs[model]
        selected[model] = {"all_system": strongest_model(report),
                           "shared_activation": strongest_model(report, SHARED_ARMS),
                           "per_corpus": {}}
        for domain in DOMAINS:
            meta = load(run / f"ppl/windows_{domain}.json")
            fos_rows.append(contrast(report, "frozen_winner", "four_over_six", meta,
                                     model, domain, "heldout_quality"))
            sys_arm = strongest(report, domain, SYSTEM_ARMS)
            shr_arm = strongest(report, domain, SHARED_ARMS)
            selected[model]["per_corpus"][domain] = {"all_system": sys_arm,
                                                        "shared_activation": shr_arm}
            system_rows.append(contrast(report, "frozen_winner", sys_arm, meta,
                                        model, domain, "heldout_strongest_all_system"))
            shared_rows.append(contrast(report, "frozen_winner", shr_arm, meta,
                                        model, domain, "heldout_strongest_shared_activation"))
    for family in (fos_rows, system_rows, shared_rows):
        holm(family)
    values = [row["estimate"] for row in fos_rows]
    per_model = {model: float(np.mean([row["estimate"] for row in fos_rows
                                       if row["model"] == model])) for model in MODELS}
    downstream = [model for model in MODELS
                  if per_model[model] < 0 and all(row["estimate"] < MATERIAL for row in fos_rows
                                                  if row["model"] == model)]
    strong_outliers = [model for model in MODELS
                       if any(abs(row["estimate"]) >= MATERIAL for row in fos_rows
                              if row["model"] == model)]
    overall = float(np.median(values)) < 0 and all(value < MATERIAL for value in values)
    strongest_values = [row["estimate"] for row in system_rows]
    analysis = {
        "schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
        "winner_freeze": {"path": str(winner_path), "sha256": expected},
        "operationalization_sha256": runtime.sha256_file(
            CR / "provenance/P70_P75_WINNER_HELDOUT_OPERATIONALIZATION.json"),
        "bootstrap_replicates": B,
        "run_inputs": {model: {"run_id": runs[model].name,
                                 "report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json")}
                       for model in MODELS},
        "contrasts": {"winner_vs_four_over_six": fos_rows,
                      "winner_vs_strongest_all_system": system_rows,
                      "winner_vs_strongest_shared_activation": shared_rows},
        "baseline_selection_without_winner_peeking": selected,
        "G8_heldout_quality": {"passed": overall,
                               "median_four_cells": float(np.median(values)),
                               "worst_cell": float(max(values)),
                               "per_model_cross_corpus_mean": per_model,
                               "downstream_eligible_models": downstream,
                               "strong_PPL_outlier_models": strong_outliers,
                               "strong_PPL_outlier_rule": "any corpus abs(dlogPPL) >= log(1.005)",
                               "material_regression_threshold": MATERIAL},
        "G9_heldout_component": {"winner_better_than_strongest_all_system_all_cells":
                                  all(value < 0 for value in strongest_values),
                                  "median_four_cells": float(np.median(strongest_values)),
                                  "worst_cell": float(max(strongest_values)),
                                  "broad_SOTA_not_decided_here": True},
        "prospective": True, "native_performance_claim": "forbidden"}
    analysis_path = out / "P71_HELDOUT_PPL_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P71_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    decision = {"schema_version": "1.0", "status": "FROZEN_AFTER_COMPLETE_P71",
                "protocol_sha256": PROTOCOL, **analysis["G8_heldout_quality"],
                "strongest_model_baselines": {model: selected[model]["all_system"] for model in MODELS},
                "strongest_shared_activation_baselines": {
                    model: selected[model]["shared_activation"] for model in MODELS},
                "analysis_path": str(analysis_path),
                "analysis_sha256": runtime.sha256_file(analysis_path)}
    decision_path = CR / "provenance/P71_G8_QUALITY_DECISION.json"
    write_new(decision_path, decision)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P71-heldout-analysis",
                   "model_revision": runtime.sha256_file(analysis_path),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per-model winner freeze",
                 "evaluation_manifest_sha256": "per-model exact windows",
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(decision_path)],
                    "summary": {"G8_quality": overall, "eligible_models": downstream},
                    "uncertainty": {"bootstrap_replicates": B, "correction": "Holm"},
                    "attempted_endpoints": ["P71_HELDOUT_PPL", "G8_QUALITY", "G9_HELDOUT_COMPONENT"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "G8_quality": overall,
                      "eligible_models": downstream}, sort_keys=True))


if __name__ == "__main__":
    main()
