"""P41 draw-aggregation development analysis and immutable winner freeze."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b")
AGGREGATIONS = ("seed0", "pooled_per_sequence_moments", "draw_mean", "draw_median", "consensus_3_of_5")
FORMS = ("natural", "fixed")
REFERENCE = "n16_k2_reference"


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


def klabel(k):
    return str(float(k)).replace(".", "p")


def policy_name(aggregation, form, k):
    return f"p41_{aggregation}_{form}_k{klabel(k)}"


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, policy, meta, model, domain):
    a, ta = arrays(report, policy, domain)
    b, tb = arrays(report, REFERENCE, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P41 pairing failed: {model}/{domain}/{policy}")
    label = f"P41:{model}:{domain}:{policy}:{REFERENCE}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=policy, comparator=REFERENCE,
               stream_label=label, raw_ppl_method=report["evaluation"][policy][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][REFERENCE][domain]["ppl"])
    return row


def select_candidate(summaries, tolerance=0.0001):
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        return None
    median = min(row["median_six_cells"] for row in eligible)
    stage = [row for row in eligible if row["median_six_cells"] <= median + tolerance]
    worst = min(row["worst_cell"] for row in stage)
    stage = [row for row in stage if row["worst_cell"] == worst]
    jaccard = max(row["median_lodo_jaccard"] for row in stage)
    stage = [row for row in stage if row["median_lodo_jaccard"] == jaccard]
    weights = min(row["total_selected_weights"] for row in stage)
    stage = [row for row in stage if row["total_selected_weights"] == weights]
    return min(stage, key=lambda row: (row["aggregation"], row["form"]))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    kfreeze = CR / "provenance/P40_GLOBAL_K_SELECTION.json"
    k = float(load(kfreeze)["global_k"])
    out = runtime.out_dir("analysis_p41")
    reports, runs, derivations, raw = {}, {}, {}, []
    for model in MODELS:
        run = latest_complete(f"P41_quality_{model}")
        derivation = latest_complete(f"P41_maps_{model}")
        report = load(run / "ppl/ppl_report.json")
        drep = load(derivation / "derived_maps/derivation_report.json")
        if report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P41 quality report: {model}")
        if float(drep["global_k"]) != k:
            raise RuntimeError("P41 map global-k drift")
        reports[model], runs[model], derivations[model] = report, run, drep
        for policy, domains in report["evaluation"].items():
            for domain in ("wiki", "c4"):
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": domains[domain]["ppl"], "mean_nll": domains[domain]["mean_nll"],
                            "run_id": run.name})
    summaries = []
    for aggregation in AGGREGATIONS:
        for form in FORMS:
            policy = policy_name(aggregation, form, k)
            cells, lodo, weights = [], [], 0
            for model in MODELS:
                report, run = reports[model], runs[model]
                install = next(row for row in report["installs"] if row["name"] == policy)
                weights += int(install["selected_tiles"]) * 1024
                lodo.append(float(derivations[model]["stability"][aggregation][form]["median_jaccard"]))
                for domain in ("wiki", "c4"):
                    cells.append(contrast(report, policy, load(run / f"ppl/windows_{domain}.json"), model, domain))
            values = [row["estimate"] for row in cells]
            summaries.append({"aggregation": aggregation, "form": form, "policy_name": policy,
                              "eligible": float(np.median(values)) <= -0.0005 and max(values) <= 0.001,
                              "median_six_cells": float(np.median(values)), "worst_cell": float(max(values)),
                              "median_lodo_jaccard": float(np.median(lodo)),
                              "per_model_lodo_jaccard": dict(zip(MODELS, lodo)),
                              "total_selected_weights": weights, "cells": cells})
    chosen = select_candidate(summaries)
    fallback = chosen is None
    if fallback:
        chosen = {"aggregation": "seed0_k2_fallback", "form": "natural", "policy_name": REFERENCE,
                  "eligible": True, "median_six_cells": 0.0, "worst_cell": 0.0,
                  "median_lodo_jaccard": 1.0,
                  "total_selected_weights": sum(int(next(row for row in reports[m]["installs"]
                                                          if row["name"] == REFERENCE)["selected_tiles"]) * 1024
                                                for m in MODELS), "cells": []}
    selected_maps = {}
    for model in MODELS:
        entry = next(row for row in reports[model]["plan"] if row["name"] == chosen["policy_name"])
        selected_maps[model] = {"map_path": entry["map_path"], "map_sha256": entry["map_sha256"],
                                "map_policy": entry["map_policy"], "type_block": entry["type_block"],
                                "expected_total_tiles": entry["expected_total_tiles"]}
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "global_k": k, "global_k_freeze_sha256": runtime.sha256_file(kfreeze),
                "derivation_spec_sha256": runtime.sha256_file(CR / "provenance/P40_P41_DERIVATION_SPEC.json"),
                "bootstrap_replicates": B,
                "run_inputs": {model: {"quality_run_id": runs[model].name,
                                        "quality_report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json"),
                                        "map_derivation_report_sha256": runtime.sha256_file(
                                            latest_complete(f"P41_maps_{model}") / "derived_maps/derivation_report.json")}
                               for model in MODELS},
                "candidate_summaries": summaries, "fallback_used": fallback,
                "selected_candidate": chosen,
                "post_selection_inference": "Development selection only; no naive confirmatory winner CI."}
    analysis_path = out / "P41_DRAW_AGGREGATION_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P41_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    freeze = {"schema_version": "1.0", "status": "FROZEN_AFTER_P41_DEVELOPMENT_BEFORE_P50_MAPS",
              "protocol_sha256": PROTOCOL, "global_k": k, "fallback_used": fallback,
              "aggregation": chosen["aggregation"], "form": chosen["form"],
              "policy_name": chosen["policy_name"], "selected_maps": selected_maps,
              "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path),
              "selection_rule": "eligibility; median; within-0.0001 worst; LODO Jaccard; weights; lexical"}
    freeze_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    write_new(freeze_path, freeze)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P41-development-analysis", "model_revision": freeze["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(freeze_path)],
                    "summary": {"selected_policy": freeze["policy_name"], "fallback": fallback},
                    "uncertainty": {"bootstrap_replicates": B, "selection_adjusted": False},
                    "attempted_endpoints": ["P41_DRAW_AGG_DEV", "AGGREGATION_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "selected_policy": freeze["policy_name"],
                      "fallback": fallback}, sort_keys=True))


if __name__ == "__main__":
    main()
