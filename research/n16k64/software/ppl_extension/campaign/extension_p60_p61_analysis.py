"""Analyze the sequential P60/P61 quantizer screens and freeze each axis."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.extension_p50_maps import CR, MODELS, PROTOCOL, latest_complete, load
from campaign.extension_p60_p61_plans import ACTIVATIONS, SCALE
from campaign.report_contracts import validate_policy_contract


B = 10_000
DOMAINS = ("wiki", "c4")


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, method, comparator, meta, model, domain, stage):
    a, ta = arrays(report, method, domain); b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"{stage} pairing failed: {model}/{domain}/{method}/{comparator}")
    label = f"{stage}:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               stream_label=label, raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def select_summary(summaries, order, tolerance=0.0001):
    pool = [row for row in summaries if row["eligible"]]
    if not pool:
        return None
    best = min(row["winner_median"] for row in pool)
    stage = [row for row in pool if row["winner_median"] <= best + tolerance]
    worst = min(row["all_arm_worst"] for row in stage)
    stage = [row for row in stage if row["all_arm_worst"] == worst]
    return min(stage, key=lambda row: order[row["candidate"]])


def load_reports(stage):
    reports, runs, raw = {}, {}, []
    for model in MODELS:
        run = latest_complete(f"{stage}_quality_{model}")
        path = run / "ppl/ppl_report.json"; report = load(path)
        if report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted {stage} quality report: {model}")
        reports[model], runs[model] = report, run
        for policy, domains in report["evaluation"].items():
            for domain in DOMAINS:
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": domains[domain]["ppl"], "mean_nll": domains[domain]["mean_nll"],
                            "run_id": run.name})
    return reports, runs, raw


def summaries_for(reports, runs, stage, candidates, references):
    summaries = []
    for candidate, labels in candidates.items():
        winner, fos = [], []
        for model in MODELS:
            for domain in DOMAINS:
                meta = load(runs[model] / f"ppl/windows_{domain}.json")
                winner.append(contrast(reports[model], labels["winner"], references["winner"],
                                       meta, model, domain, stage))
                fos.append(contrast(reports[model], labels["fos"], references["fos"],
                                    meta, model, domain, stage))
        wv, fv = [row["estimate"] for row in winner], [row["estimate"] for row in fos]
        summaries.append({"candidate": candidate,
                          "eligible": float(np.median(wv)) <= -0.0005 and max(wv) <= 0.001 and max(fv) <= 0.001,
                          "winner_median": float(np.median(wv)), "winner_worst": float(max(wv)),
                          "fos_median": float(np.median(fv)), "fos_worst": float(max(fv)),
                          "all_arm_worst": float(max(wv + fv)),
                          "winner_contrasts": winner, "four_over_six_contrasts": fos})
    return summaries


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value); path.chmod(0o444)


def analyze_p60(reports, runs):
    expected = tuple([f"p60_fos_{label}" for label, _ in SCALE] +
                     [f"p60_winner_{label}" for label, _ in SCALE])
    for model, report in reports.items():
        validate_policy_contract(report, expected, f"P60 {model}")
    candidates = {label: {"fos": f"p60_fos_{label}", "winner": f"p60_winner_{label}"}
                  for label, value in SCALE if value != 1.0}
    refs = {"fos": "p60_fos_s1p0", "winner": "p60_winner_s1p0"}
    summaries = summaries_for(reports, runs, "P60", candidates, refs)
    multiplier = dict(SCALE)
    order = {label: (abs(value - 1), value) for label, value in SCALE if value != 1.0}
    chosen = select_summary(summaries, order)
    selected_label = "s1p0" if chosen is None else chosen["candidate"]
    return summaries, {"fallback": chosen is None, "selected_label": selected_label,
                       "selected_multiplier": multiplier[selected_label],
                       "selected_summary": chosen,
                       "G7_compatible": True,
                       "compatibility": "same E2M1/E0M3 four-bit codes, E4M3 scale-field count, type map and scope; one offline global configuration constant; no native-performance claim"}


def analyze_p61(reports, runs):
    expected = tuple([f"p61_{prefix}_{label}" for prefix in ("fos", "winner") for label, _ in ACTIVATIONS])
    for model, report in reports.items():
        validate_policy_contract(report, expected, f"P61 {model}")
    anchors = []
    for model in MODELS:
        report = reports[model]
        installs = {row["name"]: row for row in report["installs"]}
        for prefix in ("fos", "winner"):
            base, anchor = f"p61_{prefix}_baseline_absmax", f"p61_{prefix}_percentile_100_anchor"
            exact = installs[base]["installed_weight_sha256"] == installs[anchor]["installed_weight_sha256"]
            for domain in DOMAINS:
                a, ta = arrays(report, base, domain); b, tb = arrays(report, anchor, domain)
                exact = exact and np.array_equal(ta, tb) and np.array_equal(a, b)
            anchors.append({"model": model, "weight_arm": prefix, "baseline_policy": base,
                            "percentile_100_policy": anchor, "bit_exact_outputs": bool(exact)})
    if not all(row["bit_exact_outputs"] for row in anchors):
        raise RuntimeError("P61 percentile-100 anchor is not output exact")
    candidate_labels = ("mse_grid", "percentile_999", "percentile_9999")
    candidates = {label: {"fos": f"p61_fos_{label}", "winner": f"p61_winner_{label}"}
                  for label in candidate_labels}
    refs = {"fos": "p61_fos_baseline_absmax", "winner": "p61_winner_baseline_absmax"}
    summaries = summaries_for(reports, runs, "P61", candidates, refs)
    order = {"percentile_9999": 0, "percentile_999": 1, "mse_grid": 2}
    chosen = select_summary(summaries, order)
    selected_label = "baseline_absmax" if chosen is None else chosen["candidate"]
    kinds = dict(ACTIVATIONS)
    kind = "four_over_six_rows" if chosen is None else kinds[selected_label]
    compatible = chosen is None
    return summaries, {"fallback": chosen is None, "selected_label": selected_label,
                       "selected_activation_kind": kind, "selected_summary": chosen,
                       "anchor_checks": anchors, "G7_compatible": compatible,
                       "compatibility": ("baseline causal FourOverSix row quantizer; no extra operation" if compatible else
                                         "quality-only causal same-field W4A4 candidate with disclosed unfused inference-time operations; fails deployment-compatibility G7")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("p60", "p61")); args = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    stage = args.stage.upper(); reports, runs, raw = load_reports(stage)
    summaries, selection = analyze_p60(reports, runs) if args.stage == "p60" else analyze_p61(reports, runs)
    out = runtime.out_dir(f"analysis_{args.stage}")
    operational = CR / "provenance/P60_P61_ANALYSIS_OPERATIONALIZATION.json"
    analysis = {"schema_version": "1.0", "status": "complete", "stage": stage,
                "protocol_sha256": PROTOCOL,
                "quantizer_spec_sha256": runtime.sha256_file(CR / "provenance/P60_P62_QUANTIZER_SPEC.json"),
                "operationalization_sha256": runtime.sha256_file(operational),
                "bootstrap_replicates": B,
                "run_inputs": {model: {"run_id": runs[model].name,
                                       "report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json")}
                               for model in MODELS},
                "candidate_summaries": summaries, "selection": selection,
                "development_only": True, "native_performance_claim": "forbidden"}
    analysis_path = out / f"{stage}_ANALYSIS.json"; runtime.atomic_json(analysis_path, analysis)
    csv_path = out / f"{stage}_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    freeze = {"schema_version": "1.0", "status": f"FROZEN_AFTER_{stage}_DEVELOPMENT",
              "stage": stage, "protocol_sha256": PROTOCOL, **selection,
              "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path)}
    freeze_path = CR / "provenance" / f"{stage}_SELECTION.json"; write_new(freeze_path, freeze)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{stage}-development-analysis", "model_revision": freeze["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(freeze_path)],
                    "summary": selection, "uncertainty": {"bootstrap_replicates": B,
                                                            "selection_adjusted": False},
                    "attempted_endpoints": [stage, "G7"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "stage": stage, "selection": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
