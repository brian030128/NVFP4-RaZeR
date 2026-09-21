"""P20 fixed-selected-weight density/control analysis under the frozen G3/G4 rule."""
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
SCOPES = ("global", "per_module")
DETERMINISTIC = ("weight_mse", "magnitude", "change_norm", "activation_weighted")
RANDOM_SEEDS = (2026091401, 2026091402, 2026091403, 2026091404, 2026091405)
NATURAL = "n16_k2_reference"


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


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def contrast(report, method, comparator, meta, model, domain, label_prefix):
    a, ta = arrays(report, method, domain)
    b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"unpaired P20 windows: {model}/{domain}/{method}/{comparator}")
    label = f"P20:{label_prefix}:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def median_random_policy(report, scope, domain):
    ranked = sorted((report["evaluation"][f"p20_{scope}_random_{seed}"][domain]["mean_nll"], seed)
                    for seed in RANDOM_SEEDS)
    return f"p20_{scope}_random_{ranked[2][1]}", ranked


def select_strongest(summaries, tolerance=0.0001):
    best = min(row["median_control_minus_natural"] for row in summaries)
    stage = [row for row in summaries if row["median_control_minus_natural"] <= best + tolerance]
    worst = min(row["worst_control_minus_natural"] for row in stage)
    stage = [row for row in stage if row["worst_control_minus_natural"] == worst]
    return min(stage, key=lambda row: (row["scope"], row["selector"]))


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
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    gate_spec = CR / "provenance/P20_G3_G4_OPERATIONALIZATION.json"
    if not gate_spec.exists():
        raise FileNotFoundError(gate_spec)
    out = runtime.out_dir("analysis_p20")
    reports, runs, raw = {}, {}, []
    identities = {}
    for model in MODELS:
        run = latest_complete(f"P20_quality_{model}")
        report = load(run / "ppl/ppl_report.json")
        if report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P20 PPL report: {model}")
        reports[model], runs[model] = report, run
        installs = {row["name"]: row for row in report["installs"]}
        natural_weights = int(installs[NATURAL]["selected_tiles"]) * 1024
        checks = {}
        for scope in SCOPES:
            for seed in RANDOM_SEEDS:
                name = f"p20_{scope}_random_{seed}"
                checks[name] = int(installs[name]["selected_tiles"]) * 1024 == natural_weights
            for selector in DETERMINISTIC:
                name = f"p20_{scope}_{selector}"
                checks[name] = int(installs[name]["selected_tiles"]) * 1024 == natural_weights
            n8 = f"p20_{scope}_n8_u2_matched"
            checks[n8] = int(installs[n8]["selected_tiles"]) * 512 == natural_weights
        if not all(checks.values()):
            raise RuntimeError(f"P20 selected-weight identity failed: {model}")
        identities[model] = {"natural_selected_weights": natural_weights, "controls_exact": checks}
        for policy, domains in report["evaluation"].items():
            for domain in ("wiki", "c4"):
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": domains[domain]["ppl"], "mean_nll": domains[domain]["mean_nll"],
                            "run_id": run.name})

    scope_results = {}
    all_deterministic_summaries = []
    for scope in SCOPES:
        random_rows, deterministic_rows, n8_rows, median_choices = [], [], [], []
        for model in MODELS:
            report, run = reports[model], runs[model]
            for domain in ("wiki", "c4"):
                meta = load(run / "ppl" / f"windows_{domain}.json")
                random_policy, ranking = median_random_policy(report, scope, domain)
                rr = contrast(report, NATURAL, random_policy, meta, model, domain, scope + ":median_random")
                rr.update(scope=scope, selected_median_random_policy=random_policy)
                random_rows.append(rr)
                median_choices.append({"model": model, "corpus": domain, "scope": scope,
                                       "selected_policy": random_policy,
                                       "ranking_mean_nll_then_seed": [{"mean_nll": x, "seed": y} for x, y in ranking]})
                for selector in DETERMINISTIC:
                    policy = f"p20_{scope}_{selector}"
                    row = contrast(report, NATURAL, policy, meta, model, domain, scope + ":deterministic")
                    row.update(scope=scope, selector=selector)
                    deterministic_rows.append(row)
                n8_policy = f"p20_{scope}_n8_u2_matched"
                row = contrast(report, NATURAL, n8_policy, meta, model, domain, scope + ":matched_n8")
                row.update(scope=scope)
                n8_rows.append(row)
        holm(random_rows); holm(deterministic_rows); holm(n8_rows)
        selector_summaries = []
        for selector in DETERMINISTIC:
            cells = [row["estimate"] for row in deterministic_rows if row["selector"] == selector]
            summary = {"scope": scope, "selector": selector,
                       "median_natural_minus_control": float(np.median(cells)),
                       "worst_natural_minus_control": float(max(cells)),
                       "median_control_minus_natural": float(np.median([-x for x in cells])),
                       "worst_control_minus_natural": float(max(-x for x in cells))}
            selector_summaries.append(summary)
            all_deterministic_summaries.append(summary)
        det_values = [row["estimate"] for row in deterministic_rows]
        random_pass = all(row["estimate"] < 0 for row in random_rows)
        det_pass = (sum(row["median_natural_minus_control"] < 0 for row in selector_summaries) >= 3 and
                    float(np.median(det_values)) < 0 and all(value <= 0.001 for value in det_values))
        n8_values = [row["estimate"] for row in n8_rows]
        scope_results[scope] = {
            "median_random_choices": median_choices, "median_random_contrasts": random_rows,
            "deterministic_contrasts": deterministic_rows, "deterministic_summaries": selector_summaries,
            "matched_n8_contrasts": n8_rows,
            "G3": {"median_random_pass": random_pass, "deterministic_consistency_pass": det_pass,
                   "passed": random_pass and det_pass, "deterministic_aggregate_median": float(np.median(det_values)),
                   "deterministic_worst_cell": float(max(det_values))},
            "G4_matched": {"median_six_cells": float(np.median(n8_values)),
                           "worst_cell": float(max(n8_values)),
                           "n16_not_worse": float(np.median(n8_values)) <= 0 and all(x <= 0.001 for x in n8_values)},
        }
    strongest = select_strongest(all_deterministic_summaries)
    g3 = all(scope_results[scope]["G3"]["passed"] for scope in SCOPES)
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "gate_spec_sha256": runtime.sha256_file(gate_spec), "bootstrap_replicates": B,
                "run_inputs": {model: {"run_id": runs[model].name,
                                        "report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json")}
                               for model in MODELS},
                "selected_weight_identity": identities, "scopes": scope_results,
                "G3": {"passed": g3, "scope_pass": {s: scope_results[s]["G3"]["passed"] for s in SCOPES},
                       "failure_interpretation": (None if g3 else
                                                  "Natural k2 improvement cannot be attributed to selector quality; density is the supported explanation.")},
                "G4_matched": {"scope_n16_not_worse": {s: scope_results[s]["G4_matched"]["n16_not_worse"]
                                                         for s in SCOPES}},
                "strongest_deterministic_control": strongest}
    analysis_path = out / "P20_DENSITY_CONTROL_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P20_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    selection = {"schema_version": "1.0", "status": "FROZEN_AFTER_P20_DEVELOPMENT",
                 "protocol_sha256": PROTOCOL, "gate_spec_sha256": runtime.sha256_file(gate_spec),
                 "scope": strongest["scope"], "selector": strongest["selector"],
                 "policy_name": f"p20_{strongest['scope']}_{strongest['selector']}",
                 "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path),
                 "selection_rule": "frozen median, within-0.0001 worst, then lexical scope/selector",
                 "development_only": True}
    selection_path = CR / "provenance/P20_STRONGEST_CONTROL_SELECTION.json"
    write_new(selection_path, selection)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P20-development-analysis", "model_revision": selection["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(selection_path)],
                    "summary": {"G3": g3, "strongest_control": selection["policy_name"],
                                "G4_matched": analysis["G4_matched"]},
                    "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": ["P20_K2_DENSITY_DEV", "G3", "G4_MATCHED"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "G3": g3,
                      "strongest_control": selection["policy_name"]}, sort_keys=True))


if __name__ == "__main__":
    main()
