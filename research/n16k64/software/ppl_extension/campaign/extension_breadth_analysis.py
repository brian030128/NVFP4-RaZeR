"""Paired analyses for the gate-authorized P21, P42, and P52 breadth panels."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("qwen27b", "phi4", "olmo2_13b")
DOMAINS = ("wiki", "c4")
CONFIG = {
    "p21": {
        "job": "P21_quality_{}",
        "order": ("n16_k2_reference", "p21_n8_u2_matched",
                  "p21_frozen_strongest_control"),
        "contrasts": (("n16_k2_reference", "p21_n8_u2_matched", "matched_n16_vs_n8"),
                      ("n16_k2_reference", "p21_frozen_strongest_control", "natural_vs_strong_control")),
        "matrix": "P21_K2_DENSITY_BREADTH",
    },
    "p42": {
        "job": "P42_quality_{}",
        "order": ("seed0_n16_k2_reference", "p42_frozen_aggregation",
                  "p42_strong_control_budget_matched"),
        "contrasts": (("p42_frozen_aggregation", "seed0_n16_k2_reference", "aggregation_vs_seed0"),
                      ("p42_frozen_aggregation", "p42_strong_control_budget_matched", "aggregation_vs_strong_control")),
        "matrix": "P42_DRAW_AGG_BREADTH",
    },
    "p52": {
        "job": "P52_quality_{}",
        "order": ("p52_old_first_order", "p52_strongest_p20_selector_global_rerank",
                  "p52_frozen_replacement"),
        "contrasts": (("p52_frozen_replacement", "p52_old_first_order", "replacement_vs_old"),
                      ("p52_frozen_replacement", "p52_strongest_p20_selector_global_rerank", "replacement_vs_strong_control")),
        "matrix": "P52_SELECTOR_BREADTH",
    },
}


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, valid = (load(run / name) for name in
                                     ("launch_record.json", "job_status.json",
                                      "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and valid.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(stage, report, method, comparator, meta, model, domain, family):
    a, ta = arrays(report, method, domain)
    b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"{stage.upper()} pairing failed: {model}/{domain}/{family}")
    label = f"{stage.upper()}:{family}:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               family=family, stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def selected_weights(install):
    if install.get("selected_tiles") is None:
        return None
    return int(install["selected_tiles"]) * int(install["type_block"][0]) * int(install["type_block"][1])


def holm(rows):
    adjusted, rejected = S.holm([row["p_two_sided"] for row in rows])
    for row, p, reject in zip(rows, adjusted, rejected):
        row["p_holm"] = p
        row["holm_reject_0_05"] = reject


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=tuple(CONFIG), required=True)
    args = ap.parse_args()
    stage, cfg = args.stage, CONFIG[args.stage]
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    reports, runs, raw, budgets = {}, {}, [], {}
    for model in MODELS:
        run = latest_complete(cfg["job"].format(model))
        report_path = run / "ppl/ppl_report.json"
        report = load(report_path)
        if (report.get("status") != "complete"
                or not report.get("reinstall_check", {}).get("identical")):
            raise RuntimeError(f"unaccepted {stage.upper()} report: {model}")
        validate_policy_contract(report, cfg["order"], f"{stage.upper()} {model}")
        installs = {row["name"]: row for row in report["installs"]}
        weights = {name: selected_weights(installs[name]) for name in cfg["order"]}
        if stage == "p21":
            if len(set(weights.values())) != 1:
                raise RuntimeError(f"P21 selected-weight mismatch: {model}")
        elif stage == "p42":
            if weights["p42_frozen_aggregation"] != weights["p42_strong_control_budget_matched"]:
                raise RuntimeError(f"P42 control budget mismatch: {model}")
        elif len(set(weights.values())) != 1:
            raise RuntimeError(f"P52 selected-weight mismatch: {model}")
        budgets[model] = weights
        reports[model], runs[model] = report, run
        for policy in cfg["order"]:
            for domain in DOMAINS:
                ev = report["evaluation"][policy][domain]
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": ev["ppl"], "mean_nll": ev["mean_nll"],
                            "run_id": run.name})
    families = {}
    for method, comparator, family in cfg["contrasts"]:
        rows = []
        for model in MODELS:
            for domain in DOMAINS:
                rows.append(contrast(stage, reports[model], method, comparator,
                                     load(runs[model] / f"ppl/windows_{domain}.json"),
                                     model, domain, family))
        holm(rows)
        values = [row["estimate"] for row in rows]
        families[family] = {
            "rows": rows,
            "median_six_cells": float(np.median(values)),
            "worst_cell": float(max(values)),
            "beneficial_cells": sum(value < 0 for value in values),
            "per_model_cross_corpus_mean": {
                model: float(np.mean([row["estimate"] for row in rows
                                      if row["model"] == model]))
                for model in MODELS},
            "non_qwen_median_four_cells": float(np.median(
                [row["estimate"] for row in rows if row["model"] != "qwen27b"])),
        }
    analysis = {
        "schema_version": "1.0", "status": "complete",
        "matrix_id": cfg["matrix"], "protocol_sha256": PROTOCOL,
        "operationalization_sha256": runtime.sha256_file(
            CR / "provenance/P21_P42_P52_BREADTH_OPERATIONALIZATION.json"),
        "bootstrap_replicates": B,
        "run_inputs": {model: {"run_id": runs[model].name,
                                "report_sha256": runtime.sha256_file(
                                    runs[model] / "ppl/ppl_report.json")}
                       for model in MODELS},
        "selected_weight_identity": budgets,
        "families": families,
        "development_choices_frozen_before_breadth_quality": True,
        "post_hoc_robustness_extension": True,
    }
    out = runtime.out_dir(f"analysis_{stage}")
    analysis_path = out / f"{stage.upper()}_BREADTH_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    raw_path = out / f"{stage.upper()}_RAW_PPL.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    decision = {"schema_version": "1.0",
                "status": f"FROZEN_AFTER_{stage.upper()}_BREADTH",
                "matrix_id": cfg["matrix"], "protocol_sha256": PROTOCOL,
                "analysis_path": str(analysis_path),
                "analysis_sha256": runtime.sha256_file(analysis_path),
                "family_summaries": {name: {k: v for k, v in value.items()
                                                if k != "rows"}
                                     for name, value in families.items()}}
    decision_path = CR / "provenance" / f"{stage.upper()}_BREADTH_DECISION.json"
    write_new(decision_path, decision)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{stage}-breadth-analysis",
                   "model_revision": decision["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per model",
                 "evaluation_manifest_sha256": "per model",
                 "token_hashes": {}, "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(raw_path), str(decision_path)],
                    "summary": decision["family_summaries"],
                    "uncertainty": {"bootstrap_replicates": B,
                                    "correction": "Holm within each six-cell family"},
                    "attempted_endpoints": [cfg["matrix"]],
                    "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "stage": stage,
                      "families": decision["family_summaries"]}, sort_keys=True))


if __name__ == "__main__":
    main()
