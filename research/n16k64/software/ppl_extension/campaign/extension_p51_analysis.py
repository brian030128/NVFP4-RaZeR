"""Weighted P51 fidelity statistics and the frozen G6 decision."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as ss

from campaign import runtime
from campaign import stats as S
from campaign.extension_p50_maps import CANDIDATES, CR, MODELS, PROTOCOL, latest_complete, load


B = 10_000
OUTCOMES = {
    "full_model_CE": "actual_ce",
    "full_model_KL": "actual_kl",
    "layer_reconstruction": "actual_layer_reconstruction",
}


def exact_binomial_ci(successes, n, alpha=0.05):
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid binomial counts")
    lo = 0.0 if successes == 0 else float(ss.beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(ss.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return [lo, hi]


def weighted_mean(x, w):
    return float(np.sum(np.asarray(x) * np.asarray(w)) / np.sum(w))


def weighted_corr(x, y, w):
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, y, w))
    mx, my = weighted_mean(x, w), weighted_mean(y, w)
    dx, dy = x - mx, y - my
    den = math.sqrt(float(np.sum(w * dx * dx) * np.sum(w * dy * dy)))
    return float(np.sum(w * dx * dy) / den) if den else 0.0


def weighted_spearman(x, y, w):
    return weighted_corr(ss.rankdata(x, method="average"), ss.rankdata(y, method="average"), w)


def weighted_tau_b(x, y, w):
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, y, w))
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    wp = w[:, None] * w[None, :]
    upper = np.triu(np.ones_like(dx, dtype=bool), 1)
    num = float(np.sum(wp[upper] * dx[upper] * dy[upper]))
    denx = float(np.sum(wp[upper] * (dx[upper] != 0)))
    deny = float(np.sum(wp[upper] * (dy[upper] != 0)))
    return num / math.sqrt(denx * deny) if denx and deny else 0.0


def calibration(x, y, w):
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, y, w))
    mx = weighted_mean(x, w)
    sx = math.sqrt(weighted_mean((x - mx) ** 2, w))
    z = (x - mx) / sx if sx else np.zeros_like(x)
    design = np.column_stack([np.ones_like(z), z])
    beta = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    fitted = design @ beta
    my = weighted_mean(y, w)
    sse = float(np.sum(w * (y - fitted) ** 2)); sst = float(np.sum(w * (y - my) ** 2))
    return {"predictor_weighted_mean": mx, "predictor_weighted_sd": sx,
            "intercept_at_predictor_mean": float(beta[0]), "slope_per_predictor_sd": float(beta[1]),
            "weighted_r2": (1.0 - sse / sst if sst else 0.0)}


def metric_bundle(score, predicted_beneficial, actual, weight):
    score = np.asarray(score, dtype=np.float64)
    pred = np.asarray(predicted_beneficial, dtype=bool)
    actual = np.asarray(actual, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    correct = pred == (actual < 0)
    successes = int(correct.sum())
    return {"n": int(len(score)), "unweighted_sign_correct": successes,
            "unweighted_sign_accuracy": successes / len(score),
            "exact_binomial_ci95": exact_binomial_ci(successes, len(score)),
            "weighted_sign_accuracy": weighted_mean(correct.astype(float), weight),
            "weighted_spearman_rho": weighted_spearman(score, actual, weight),
            "weighted_kendall_tau_b": weighted_tau_b(score, actual, weight),
            "calibration": calibration(score, actual, weight)}


def strata_indices(labels):
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[str(label)].append(i)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def draw_indices(groups, rng):
    return np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])


def bootstrap_metrics(score, pred, actual, weight, labels, stream_label, B=B):
    score = np.asarray(score, dtype=np.float64); pred = np.asarray(pred, dtype=bool)
    actual = np.asarray(actual, dtype=np.float64); weight = np.asarray(weight, dtype=np.float64)
    groups = strata_indices(labels)
    rng = np.random.default_rng(S.stable_seed_sequence(20260914, stream_label))
    acc = np.empty(B); rho = np.empty(B); tau = np.empty(B)
    for b in range(B):
        idx = draw_indices(groups, rng)
        acc[b] = weighted_mean((pred[idx] == (actual[idx] < 0)).astype(float), weight[idx])
        rho[b] = weighted_spearman(score[idx], actual[idx], weight[idx])
        tau[b] = weighted_tau_b(score[idx], actual[idx], weight[idx])
    return {"B": B, "rng": "NumPy PCG64 with stable named SeedSequence",
            "stream_label": stream_label,
            "weighted_sign_accuracy_ci95": list(map(float, np.percentile(acc, [2.5, 97.5]))),
            "weighted_spearman_rho_ci95": list(map(float, np.percentile(rho, [2.5, 97.5]))),
            "weighted_kendall_tau_b_ci95": list(map(float, np.percentile(tau, [2.5, 97.5])))}


def paired_improvement(old_score, old_pred, new_score, new_pred, actual, weight, labels,
                       stream_label, B=B):
    values = [np.asarray(x) for x in (old_score, old_pred, new_score, new_pred, actual, weight)]
    old_score, old_pred, new_score, new_pred, actual, weight = values
    groups = strata_indices(labels)
    rng = np.random.default_rng(S.stable_seed_sequence(20260914, stream_label))
    acc = np.empty(B); rho = np.empty(B)
    for b in range(B):
        idx = draw_indices(groups, rng)
        acc[b] = (weighted_mean((new_pred[idx] == (actual[idx] < 0)).astype(float), weight[idx]) -
                  weighted_mean((old_pred[idx] == (actual[idx] < 0)).astype(float), weight[idx]))
        rho[b] = (weighted_spearman(new_score[idx], actual[idx], weight[idx]) -
                  weighted_spearman(old_score[idx], actual[idx], weight[idx]))
    return {"B": B, "stream_label": stream_label,
            "sign_accuracy_improvement_ci95": list(map(float, np.percentile(acc, [2.5, 97.5]))),
            "spearman_rho_improvement_ci95": list(map(float, np.percentile(rho, [2.5, 97.5])))}


def selector_arrays(rows, selector):
    if selector == "old_first_order_ce_kl":
        return (np.asarray([row["old_score"] for row in rows]),
                np.asarray([row["old_predicted_beneficial"] for row in rows], dtype=bool))
    return (np.asarray([row["candidate_scores"][selector] for row in rows]),
            np.asarray([row["candidate_scores"][selector] < 0 for row in rows], dtype=bool))


def panel_metrics(rows, selector, outcome, B=B):
    score, pred = selector_arrays(rows, selector)
    actual = np.asarray([row[OUTCOMES[outcome]] for row in rows])
    weight = np.asarray([row["sampling_weight"] for row in rows])
    labels = [(row["model"], tuple(row["collapsed_stratum"])) for row in rows]
    out = metric_bundle(score, pred, actual, weight)
    out["bootstrap"] = bootstrap_metrics(score, pred, actual, weight, labels,
                                          f"P51:{','.join(sorted({r['model'] for r in rows}))}:{selector}:{outcome}", B=B)
    return out


def point_strata(rows, selector, outcome, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: metric_bundle(*selector_arrays(group, selector),
                               np.asarray([row[OUTCOMES[outcome]] for row in group]),
                               np.asarray([row["sampling_weight"] for row in group]))
            for key, group in sorted(grouped.items())}


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value); path.chmod(0o444)


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    selection_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    sample_freeze_path = CR / "provenance/P51_SAMPLE_FREEZE.json"
    endpoint_path = CR / "provenance/P51_G6_ENDPOINT_RESOLUTION.json"
    selection, sample_freeze = load(selection_path), load(sample_freeze_path)
    audit_selector = selection["P51_audit_policy"].removeprefix("p50_")
    reports, all_rows = {}, []
    for model in MODELS:
        run = latest_complete(f"P51_effects_{model}")
        path = run / "tile_fidelity/P51_TILE_EFFECTS.json"
        report = load(path)
        frozen = next(row for row in sample_freeze["manifests"] if row["model"] == model)
        if report.get("status") != "complete" or report.get("completed_tiles") != 60:
            raise RuntimeError(f"unaccepted P51 effect report: {model}")
        if report["sample_manifest_sha256"] != frozen["sha256"]:
            raise RuntimeError(f"P51 sample/report digest mismatch: {model}")
        if not all(report["baseline_restoration"].get(k) for k in ("ce_exact", "kl_exact")):
            raise RuntimeError(f"P51 baseline restoration failed: {model}")
        reports[model] = {"run_id": run.name, "path": str(path), "sha256": runtime.sha256_file(path)}
        all_rows.extend(report["tiles"])
    if len(all_rows) != 180 or len({row["sample_id"] for row in all_rows}) != 180:
        raise RuntimeError("P51 combined sample size/identity mismatch")

    selectors = ("old_first_order_ce_kl", audit_selector)
    overall, per_model = {}, {}
    for outcome in OUTCOMES:
        overall[outcome] = {selector: panel_metrics(all_rows, selector, outcome) for selector in selectors}
        per_model[outcome] = {model: {selector: panel_metrics(
            [row for row in all_rows if row["model"] == model], selector, outcome) for selector in selectors}
                              for model in MODELS}
    exploratory = {outcome: {selector: metric_bundle(
        *selector_arrays(all_rows, selector),
        np.asarray([row[OUTCOMES[outcome]] for row in all_rows]),
        np.asarray([row["sampling_weight"] for row in all_rows]))
        for selector in CANDIDATES} for outcome in OUTCOMES}
    strata = {outcome: {selector: {
        "layer_quartile": point_strata(all_rows, selector, outcome, "layer_quartile"),
        "module_type": point_strata(all_rows, selector, outcome, "module_type")}
        for selector in selectors} for outcome in OUTCOMES}

    old_score, old_pred = selector_arrays(all_rows, "old_first_order_ce_kl")
    new_score, new_pred = selector_arrays(all_rows, audit_selector)
    actual = np.asarray([row["actual_ce"] for row in all_rows])
    weight = np.asarray([row["sampling_weight"] for row in all_rows])
    labels = [(row["model"], tuple(row["collapsed_stratum"])) for row in all_rows]
    paired = paired_improvement(old_score, old_pred, new_score, new_pred, actual, weight, labels,
                                f"P51:G6:{audit_selector}:full_model_CE", B=B)
    old_m, new_m = overall["full_model_CE"]["old_first_order_ce_kl"], overall["full_model_CE"][audit_selector]
    sign_delta = new_m["weighted_sign_accuracy"] - old_m["weighted_sign_accuracy"]
    rho_delta = new_m["weighted_spearman_rho"] - old_m["weighted_spearman_rho"]
    full_map_pass = not selection["fallback_to_old"] and selection["selected_replacement"] == selection["P51_audit_policy"]
    sign_pass = sign_delta >= 0.10 and paired["sign_accuracy_improvement_ci95"][0] > 0
    rank_pass = rho_delta >= 0.10 and paired["spearman_rho_improvement_ci95"][0] > 0
    g6 = {"passed": bool(full_map_pass and sign_pass and rank_pass),
          "full_map_replacement_pass": full_map_pass, "sign_fidelity_pass": bool(sign_pass),
          "rank_fidelity_pass": bool(rank_pass), "audit_selector": audit_selector,
          "weighted_sign_accuracy_improvement": float(sign_delta),
          "weighted_spearman_rho_improvement": float(rho_delta),
          "paired_bootstrap": paired,
          "interpretation": ("replacement selector improves fixed-budget full-map PPL and materially improves CE sign/rank fidelity"
                             if full_map_pass and sign_pass and rank_pass else
                             "G6 replacement claim not supported; apply the frozen full-map-only, fidelity-only, or old-selector fallback interpretation")}
    out = runtime.out_dir("analysis_p51")
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "sample_freeze_sha256": runtime.sha256_file(sample_freeze_path),
                "P50_selection_sha256": runtime.sha256_file(selection_path),
                "endpoint_resolution_sha256": runtime.sha256_file(endpoint_path),
                "bootstrap_replicates": B, "run_inputs": reports,
                "audit_selector": audit_selector, "overall": overall, "per_model": per_model,
                "all_candidate_exploratory_points": exploratory, "strata": strata, "G6": g6,
                "inference_unit": "frozen sampled tile intervention",
                "weighting": "inverse frozen microcell inclusion probability"}
    analysis_path = out / "P51_TILE_FIDELITY_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P51_RAW_TILE_EFFECTS.csv"
    csv_rows = []
    for row in all_rows:
        csv_rows.append({k: row[k] for k in
                         ("sample_id", "model", "module", "layer", "layer_quartile", "module_type",
                          "row_tile", "k_tile", "old_score", "audit_candidate", "audit_candidate_score",
                          "old_predicted_beneficial", "candidate_predicted_beneficial", "inclusion_probability",
                          "sampling_weight", "actual_ce", "actual_ce_se", "actual_kl", "actual_kl_se",
                          "actual_layer_reconstruction")})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    decision = {"schema_version": "1.0", "status": "DECIDED_AFTER_P51_DEVELOPMENT",
                "protocol_sha256": PROTOCOL, "G6": g6, "analysis_path": str(analysis_path),
                "analysis_sha256": runtime.sha256_file(analysis_path),
                "method_selector_policy": selection["selected_policy"],
                "replacement_selector_accepted": g6["passed"]}
    decision_path = CR / "provenance/P51_G6_DECISION.json"
    write_new(decision_path, decision)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P51-fidelity-analysis", "model_revision": decision["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None},
        "policies": [], "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(decision_path)],
            "summary": {"G6": g6, "tiles": len(all_rows)},
            "uncertainty": {"bootstrap_replicates": B, "exact_binomial_intervals": True},
            "attempted_endpoints": ["P51_TILE_FIDELITY", "G6"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "G6": g6["passed"],
                      "audit_selector": audit_selector}, sort_keys=True))


if __name__ == "__main__":
    main()
