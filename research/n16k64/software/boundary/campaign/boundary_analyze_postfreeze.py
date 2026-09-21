"""Frozen paired natural-cluster analysis for boundary and corruption outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as ss

from campaign import mapio, runtime
from campaign.boundary_common import DOMAINS, MECHANISM_ROOT, MECHANISM_RUN, MODELS, SEED_ROOT, atomic_json, sha256_file


BOOTSTRAPS = 10_000
P_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)

REPORT_IDENTITY_FIELDS = (
    "model", "model_class", "module_manifest_sha256", "length", "domains", "teacher",
    "evaluation_stage", "attention_backend", "attn_implementation", "spec", "windows",
)
WINDOW_IDENTITY_FIELDS = {
    "c4": ("repo", "revision", "rows_in_file", "seed", "windows", "window_tokens",
           "parent_window_tokens", "token_sha256", "documents", "unique_documents"),
    "wiki": ("repo", "revision", "text_sha256", "windows", "window_tokens", "token_sha256",
             "articles", "window_articles", "total_tokens", "used_tokens", "omitted_tail_tokens"),
}


def seed_for(*parts: object) -> int:
    return int(hashlib.sha256(":".join([str(SEED_ROOT), *map(str, parts)]).encode()).hexdigest()[:16], 16)


def cluster_labels(meta: dict, domain: str) -> list[str]:
    if domain == "wiki":
        return [f'a{row["first_article"]}' for row in meta["window_articles"]]
    documents = meta["documents"]
    per = int(meta["windows"]) // len(documents)
    return [row["document_sha256"] for row in documents for _ in range(per)]


def load_run(path: Path) -> tuple[dict, dict[str, dict]]:
    ppl = path / "ppl"
    report = json.loads((ppl / "ppl_report.json").read_text())
    return report, {domain: json.loads((ppl / f"windows_{domain}.json").read_text()) for domain in DOMAINS}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_evaluation_identity(model: str, old_report: dict, new_report: dict,
                               old_meta: dict[str, dict], new_meta: dict[str, dict]) -> dict:
    """Verify the scientific evaluation identity without hashing provenance-only metadata.

    The original gate compared evaluator ``evaluation_manifest_sha256`` values.  Qwen's
    mechanism report includes outcome-blind screening provenance fields that the boundary
    evaluator intentionally omits, so that container hash can differ even when every frozen
    token window and natural-cluster assignment is identical.  This post-freeze correction
    checks all scientific identity fields directly and retains both report hashes for audit.
    """
    report_drift = [field for field in REPORT_IDENTITY_FIELDS
                    if old_report.get(field) != new_report.get(field)]
    if report_drift:
        raise ValueError(f"{model}: scientific report identity drift: {report_drift}")
    domains = {}
    for domain in DOMAINS:
        fields = WINDOW_IDENTITY_FIELDS[domain]
        drift = [field for field in fields if old_meta[domain].get(field) != new_meta[domain].get(field)]
        if drift:
            raise ValueError(f"{model}/{domain}: frozen evaluation-window identity drift: {drift}")
        old_labels = cluster_labels(old_meta[domain], domain)
        new_labels = cluster_labels(new_meta[domain], domain)
        if old_labels != new_labels:
            raise ValueError(f"{model}/{domain}: natural clusters drift")
        identity = {field: old_meta[domain].get(field) for field in fields}
        domains[domain] = {
            "identity_sha256": canonical_sha256(identity),
            "token_window_count": len(old_meta[domain]["token_sha256"]),
            "natural_cluster_count": len(dict.fromkeys(old_labels)),
            "fields_verified": list(fields),
        }
    return {
        "report_fields_verified": list(REPORT_IDENTITY_FIELDS),
        "old_evaluation_manifest_sha256": old_report["evaluation_manifest_sha256"],
        "new_evaluation_manifest_sha256": new_report["evaluation_manifest_sha256"],
        "report_manifest_hashes_equal": (
            old_report["evaluation_manifest_sha256"] == new_report["evaluation_manifest_sha256"]
        ),
        "domains": domains,
    }


def combine_cluster_data(model: str, new_run: Path, mechanism_root: Path) -> dict[str, dict]:
    old_run = mechanism_root / "runs" / MECHANISM_RUN[model]
    old_report, old_meta = load_run(old_run)
    new_report, new_meta = load_run(new_run)
    if new_report.get("status") != "complete":
        raise ValueError(f"{model}: new PPL report incomplete")
    identity_audit = verify_evaluation_identity(model, old_report, new_report, old_meta, new_meta)
    out = {}
    for domain in DOMAINS:
        labels = cluster_labels(old_meta[domain], domain)
        unique = list(dict.fromkeys(labels))
        index = {value: i for i, value in enumerate(unique)}
        baseline_rows = old_report["evaluation"]["four_over_six"][domain]["windows"]
        tokens = np.zeros(len(unique), np.float64)
        for row, label in zip(baseline_rows, labels):
            tokens[index[label]] += float(row["tokens"])
        policy_rows = {"four_over_six": baseline_rows,
                       "corruption_p000_anchor": old_report["evaluation"]["full"][domain]["windows"]}
        policy_rows.update({policy: value[domain]["windows"] for policy, value in new_report["evaluation"].items()})
        baseline_window_tokens = [int(row["tokens"]) for row in baseline_rows]
        nll = {}
        for policy, rows in policy_rows.items():
            if len(rows) != len(labels) or [int(row["tokens"]) for row in rows] != baseline_window_tokens:
                raise ValueError(f"{model}/{domain}/{policy}: pairing drift")
            values = np.zeros(len(unique), np.float64)
            for row, label in zip(rows, labels):
                values[index[label]] += float(row["nll_sum"])
            nll[policy] = values
        absolute_ppl = {"four_over_six": float(old_report["evaluation"]["four_over_six"][domain]["ppl"]),
                        "corruption_p000_anchor": float(old_report["evaluation"]["full"][domain]["ppl"])}
        absolute_ppl.update({policy: float(value[domain]["ppl"]) for policy, value in new_report["evaluation"].items()})
        out[domain] = {"cluster_ids": np.asarray(unique), "tokens": tokens, "nll": nll,
                       "absolute_ppl": absolute_ppl,
                       "evaluation_manifest_sha256": old_report["evaluation_manifest_sha256"],
                       "new_evaluation_manifest_sha256": new_report["evaluation_manifest_sha256"],
                       "report_manifest_hashes_equal": identity_audit["report_manifest_hashes_equal"],
                       "scientific_identity_sha256": identity_audit["domains"][domain]["identity_sha256"],
                       "identity_fields_verified": identity_audit["domains"][domain]["fields_verified"],
                       "report_identity_fields_verified": identity_audit["report_fields_verified"],
                       "old_run": str(old_run), "new_run": str(new_run),
                       "old_report_sha256": sha256_file(old_run / "ppl/ppl_report.json"),
                       "new_report_sha256": sha256_file(new_run / "ppl/ppl_report.json"),
                       "windows": len(labels), "clusters": len(unique)}
    return out


def bootstrap_delta(delta: np.ndarray, tokens: np.ndarray, label: str) -> tuple[dict, np.ndarray]:
    estimate = float(delta.sum() / tokens.sum())
    centered = delta - estimate * tokens
    rng = np.random.default_rng(seed_for(label))
    idx = rng.integers(0, len(tokens), size=(BOOTSTRAPS, len(tokens)))
    noise = centered[idx].sum(axis=1) / tokens[idx].sum(axis=1)
    samples = estimate + noise
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(estimate)).sum())) / (BOOTSTRAPS + 1)
    rates = delta / tokens
    result = {"estimate": estimate, "ci95": [float(lo), float(hi)],
              "p_two_sided_plus_one": float(p), "bootstrap_replicates": BOOTSTRAPS,
              "clusters": len(tokens), "tokens": int(tokens.sum()),
              "relative_ppl_change": float(math.expm1(estimate)),
              "cluster_regression_probability": float(np.mean(rates > 0)),
              "cluster_quantiles": {f"q{q}": float(np.quantile(rates, q / 100)) for q in (90, 95, 99)},
              "cluster_worst": float(np.max(rates))}
    return result, samples


def joint_effect_bootstrap(delta: np.ndarray, tokens: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    estimates = delta.sum(axis=1) / tokens.sum()
    centered = delta - estimates[:, None] * tokens[None, :]
    rng = np.random.default_rng(seed_for(label))
    samples = np.empty((BOOTSTRAPS, delta.shape[0]), np.float64)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        idx = rng.integers(0, len(tokens), size=(size, len(tokens)))
        denom = tokens[idx].sum(axis=1)
        for row in range(delta.shape[0]):
            samples[start:start + size, row] = estimates[row] + centered[row][idx].sum(axis=1) / denom
    return estimates, samples


def fixed_effect_slope(x: np.ndarray, y: np.ndarray, group: list[int]) -> float:
    levels = sorted(set(group))
    columns = [np.ones(len(x)), x]
    for level in levels[1:]:
        columns.append(np.asarray([value == level for value in group], np.float64))
    return float(np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][1])


def scalar_result(observed: float, samples: np.ndarray, bounded: tuple[float, float] | None = None) -> dict:
    finite = samples[np.isfinite(samples)]
    noise = finite - finite.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    if bounded:
        lo, hi = max(bounded[0], lo), min(bounded[1], hi)
    p = (1 + int((np.abs(noise) >= abs(observed)).sum())) / (len(finite) + 1)
    return {"estimate": float(observed), "ci95": [float(lo), float(hi)],
            "p_two_sided_plus_one": float(p), "bootstrap_replicates": len(finite)}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided_plus_one"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_two_sided_plus_one"]))
        rows[index]["holm_adjusted_p"] = float(running)
        rows[index]["holm_reject_0p05"] = bool(running < 0.05)


def concordance(x: np.ndarray, y: np.ndarray) -> float:
    score = count = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            product = (x[i] - x[j]) * (y[i] - y[j])
            score += 1.0 if product < 0 else 0.5 if product == 0 else 0.0
            count += 1
    return score / count


def save_paired_array(root: Path, model: str, domain: str, panel: dict, map_hashes: dict[str, str]) -> dict:
    policies = list(panel["nll"])
    path = root / "arrays" / f"{model}_{domain}_boundary_corruption_paired_cluster_nll.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, policies=np.asarray(policies), cluster_ids=panel["cluster_ids"],
             cluster_tokens=panel["tokens"],
             cluster_nll_sum=np.stack([panel["nll"][policy] for policy in policies]),
             policy_map_sha256=np.asarray([map_hashes.get(policy, "non-map-or-reused") for policy in policies]),
             evaluation_manifest_sha256=np.asarray(panel["evaluation_manifest_sha256"]),
             new_evaluation_manifest_sha256=np.asarray(panel["new_evaluation_manifest_sha256"]),
             scientific_identity_sha256=np.asarray(panel["scientific_identity_sha256"]))
    return {"path": str(path), "sha256": sha256_file(path), "policies": len(policies),
            "clusters": len(panel["tokens"]), "tokens": int(panel["tokens"].sum())}


def analyze_boundary(model: str, domain: str, panel: dict, definitions: dict, promotion: dict,
                     family_rows: dict[str, list[dict]], pooled: list[dict]) -> dict:
    tokens, nll = panel["tokens"], panel["nll"]
    observations = []
    delta_rows = []
    partitions = []
    band_results = []
    by_partition = []
    baseline = nll["four_over_six"]
    for partition in definitions["partitions"]:
        local_x, local_delta, local_rows = [], [], []
        for row in partition["bands"]:
            delta = nll[row["policy"]] - baseline
            result, _ = bootstrap_delta(delta, tokens, f"A:{model}:{domain}:{row['policy']}")
            band = {"partition": partition["partition"], "side": row["side"], "band": row["band"],
                    "policy": row["policy"], "tiles": row["score_summary"]["tiles"],
                    "mean_kappa": row["score_summary"]["mean_kappa"],
                    "kappa_range": [row["score_summary"]["min_kappa"], row["score_summary"]["max_kappa"]],
                    "ce_bottleneck_share": row["score_summary"]["ce_bottleneck_share"],
                    "absolute_ppl": panel["absolute_ppl"][row["policy"]],
                    "delta_vs_four_over_six": result}
            band_results.append(band)
            observations.append(row["score_summary"]["mean_kappa"])
            delta_rows.append(delta)
            partitions.append(partition["partition"])
            local_x.append(row["score_summary"]["mean_kappa"])
            local_delta.append(delta)
            local_rows.append(band)
        by_partition.append((partition["partition"], np.asarray(local_x), np.stack(local_delta), local_rows))
    x = np.asarray(observations, np.float64)
    delta_matrix = np.stack(delta_rows)
    y, y_boot = joint_effect_bootstrap(delta_matrix, tokens, f"A:{model}:{domain}:joint")
    beta = fixed_effect_slope(x, y, partitions)
    beta_boot = np.asarray([fixed_effect_slope(x, row, partitions) for row in y_boot])
    slope = scalar_result(beta, beta_boot)
    slope.update({"model": model, "domain": domain,
                  "family": "A_validation_primary" if model == "mistral7b" else "A_development_primary",
                  "endpoint": "beta_kappa", "expected_direction": "negative",
                  "power_status": promotion["boundary_trend_slope_end_to_end"]})
    family_rows[slope["family"]].append(slope)

    selected_rows = [i for i, row in enumerate(band_results) if row["side"] == "selected"]
    rejected_rows = [i for i, row in enumerate(band_results) if row["side"] == "rejected"]
    aggregate_delta = delta_matrix[selected_rows].mean(axis=0) - delta_matrix[rejected_rows].mean(axis=0)
    selected_rejected, _ = bootstrap_delta(aggregate_delta, tokens, f"A:{model}:{domain}:selected-rejected")
    selected_rejected.update({"model": model, "domain": domain, "family": slope["family"],
                              "endpoint": "selected_minus_rejected", "expected_direction": "negative",
                              "power_status": promotion["selected_minus_rejected_aggregate"]})
    family_rows[slope["family"]].append(selected_rejected)

    weak = [i for i, row in enumerate(band_results) if row["side"] == "selected" and row["band"] == definitions["bands_per_side"]]
    near = [i for i, row in enumerate(band_results) if row["side"] == "rejected" and row["band"] == 1]
    boundary_delta = delta_matrix[weak].mean(axis=0) - delta_matrix[near].mean(axis=0)
    weak_near, _ = bootstrap_delta(boundary_delta, tokens, f"A:{model}:{domain}:weak-near")
    weak_near.update({"model": model, "domain": domain, "family": slope["family"],
                      "endpoint": "weakest_selected_minus_nearest_rejected", "expected_direction": "negative",
                      "power_status": promotion["weakest_selected_minus_nearest_rejected"]})
    family_rows[slope["family"]].append(weak_near)

    sensitivity = []
    for partition, local_x, local_delta, local_rows in by_partition:
        local_y, local_boot = joint_effect_bootstrap(local_delta, tokens, f"A:{model}:{domain}:part{partition}")
        metrics = {}
        for name, function in (("spearman", lambda a, b: ss.spearmanr(a, b).statistic),
                               ("kendall_tau", lambda a, b: ss.kendalltau(a, b).statistic),
                               ("pairwise_concordance", concordance)):
            observed = float(function(local_x, local_y))
            samples = np.asarray([function(local_x, row) for row in local_boot])
            metrics[name] = scalar_result(observed, samples, bounded=(-1, 1) if name != "pairwise_concordance" else (0, 1))
        sensitivity.append({"partition": partition, "metrics": metrics,
                            "ordered_kappa": local_x.tolist(), "observed_delta_nll": local_y.tolist()})

    rng = np.random.default_rng(seed_for("A", model, domain, "ordered-label-permutation"))
    permuted = np.empty(BOOTSTRAPS, np.float64)
    x_parts = [x[np.asarray(partitions) == partition] for partition in sorted(set(partitions))]
    for iteration in range(BOOTSTRAPS):
        shuffled = np.concatenate([rng.permutation(values) for values in x_parts])
        permuted[iteration] = fixed_effect_slope(shuffled, y, partitions)
    permutation_p = (1 + int((permuted <= beta).sum())) / (BOOTSTRAPS + 1)
    permutation = {"observed_beta_kappa": beta, "permutations": BOOTSTRAPS,
                   "p_one_sided_plus_one_expected_negative": permutation_p,
                   "null_quantiles": [float(v) for v in np.quantile(permuted, [0.025, 0.5, 0.975])]}
    pooled.append({"model": model, "domain": domain, "x": x, "y": y, "y_boot": y_boot,
                   "groups": partitions})
    return {"bands": band_results, "primary": {"beta_kappa": slope,
                                                 "selected_minus_rejected": selected_rejected,
                                                 "weakest_selected_minus_nearest_rejected": weak_near},
            "construction_sensitivity": sensitivity, "ordered_label_permutation": permutation,
            "common_support": {"coverage": definitions.get("selected_common_support_coverage"),
                               "tiles_per_band": definitions["tiles_per_global_band"],
                               "partitions": definitions["partitions_count"]}}


def analyze_corruption(model: str, domain: str, panel: dict, definitions: dict, promotion: dict,
                       family_rows: dict[str, list[dict]], pooled: list[dict]) -> dict:
    tokens, nll = panel["tokens"], panel["nll"]
    anchor = nll["corruption_p000_anchor"]
    delta_rows, p_values, pools, level_rows = [], [], [], []
    by_policy = {row["policy"]: row for row in definitions["near_boundary_maps"]}
    for pool in range(1, 5):
        delta_rows.append(np.zeros_like(tokens))
        p_values.append(0.0)
        pools.append(pool)
        level_rows.append({"pool": pool, "p": 0.0, "policy": "corruption_p000_anchor",
                           "absolute_ppl": panel["absolute_ppl"]["corruption_p000_anchor"]})
        for p in P_LEVELS[1:]:
            policy = f"corruption_near_pool{pool:02d}_p{round(100 * p):03d}"
            delta = nll[policy] - anchor
            result, _ = bootstrap_delta(delta, tokens, f"B:{model}:{domain}:{policy}")
            definition = by_policy[policy]
            level_rows.append({"pool": pool, "p": p, "policy": policy,
                               "achieved_global_p": definition["achieved_global_p"],
                               "removed_tiles": definition["removed_tiles"],
                               "absolute_ppl": panel["absolute_ppl"][policy], "delta_vs_p0": result})
            delta_rows.append(delta)
            p_values.append(p)
            pools.append(pool)
    delta_matrix = np.stack(delta_rows)
    y, y_boot = joint_effect_bootstrap(delta_matrix, tokens, f"B:{model}:{domain}:joint")
    p_array = np.asarray(p_values, np.float64)
    beta = fixed_effect_slope(p_array, y, pools)
    beta_boot = np.asarray([fixed_effect_slope(p_array, row, pools) for row in y_boot])
    slope = scalar_result(beta, beta_boot)
    family = "B_validation_primary" if model == "mistral7b" else "B_development_primary"
    slope.update({"model": model, "domain": domain, "family": family, "endpoint": "beta_p",
                  "expected_direction": "positive", "power_status": promotion["corruption_slope_end_to_end"]})
    family_rows[family].append(slope)
    p100_idx = [i for i, (p, pool) in enumerate(zip(p_values, pools)) if p == 1.0]
    p050_idx = [i for i, (p, pool) in enumerate(zip(p_values, pools)) if p == 0.5]
    p100_delta = delta_matrix[p100_idx].mean(axis=0)
    p050_delta = delta_matrix[p050_idx].mean(axis=0)
    p100, _ = bootstrap_delta(p100_delta, tokens, f"B:{model}:{domain}:p100")
    p100.update({"model": model, "domain": domain, "family": family, "endpoint": "p100_minus_p0",
                 "expected_direction": "positive", "power_status": promotion["p100_minus_p0"]})
    family_rows[family].append(p100)
    p050, _ = bootstrap_delta(p050_delta, tokens, f"B:{model}:{domain}:p050")
    p050.update({"model": model, "domain": domain, "family": family, "endpoint": "p050_near_minus_p0",
                 "expected_direction": "positive", "power_status": promotion["p050_minus_p0"]})
    family_rows[family].append(p050)

    random_deltas, random_rows = [], []
    for replicate in range(1, 5):
        policy = f"corruption_random_pool{replicate:02d}_p050"
        delta = nll[policy] - anchor
        result, _ = bootstrap_delta(delta, tokens, f"B:{model}:{domain}:{policy}")
        random_deltas.append(delta)
        random_rows.append({"replicate": replicate, "policy": policy,
                            "absolute_ppl": panel["absolute_ppl"][policy], "delta_vs_p0": result})
    random_mean = np.stack(random_deltas).mean(axis=0)
    near_random, _ = bootstrap_delta(p050_delta - random_mean, tokens, f"B:{model}:{domain}:near-random")

    estimates_by_pool = {pool: [0.0] for pool in range(1, 5)}
    for row in level_rows:
        if row["p"]:
            estimates_by_pool[row["pool"]].append(row["delta_vs_p0"]["estimate"])
    monotonic = {str(pool): {"nondecreasing_steps": int(np.sum(np.diff(values) >= 0)),
                              "total_steps": 5, "fully_monotone": bool(np.all(np.diff(values) >= 0)),
                              "estimates": values} for pool, values in estimates_by_pool.items()}
    dispersion = {}
    for p in P_LEVELS[1:]:
        values = [row["delta_vs_p0"]["estimate"] for row in level_rows if row["p"] == p]
        dispersion[f"p{round(100 * p):03d}"] = {"mean": float(np.mean(values)),
                                                  "sd_across_pools": float(np.std(values, ddof=1)),
                                                  "min": float(min(values)), "max": float(max(values))}
    pooled.append({"model": model, "domain": domain, "x": p_array, "y": y, "y_boot": y_boot, "groups": pools})
    return {"levels": level_rows, "matched_random_p050": random_rows,
            "primary": {"beta_p": slope, "p100_minus_p0": p100, "p050_near_minus_p0": p050},
            "p050_near_minus_matched_random": near_random, "monotonicity_by_pool": monotonic,
            "replicate_dispersion": dispersion,
            "aggregate_cluster_tails": {"p100": {key: p100[key] for key in (
                "cluster_regression_probability", "cluster_quantiles", "cluster_worst")},
                "p050_near": {key: p050[key] for key in (
                    "cluster_regression_probability", "cluster_quantiles", "cluster_worst")}}}


def standardized(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd else np.zeros_like(values)


def pooled_slope(panels: list[dict], included: set[str], label: str) -> dict:
    use = [panel for panel in panels if panel["model"] in included]
    x_parts, y_parts, models, corpora, groups = [], [], [], [], []
    for panel in use:
        x_parts.append(standardized(panel["x"]))
        y_parts.append(standardized(panel["y"]))
        models.extend([panel["model"]] * len(panel["x"]))
        corpora.extend([panel["domain"]] * len(panel["x"]))
        groups.extend([f"{panel['model']}:{panel['domain']}:{value}" for value in panel["groups"]])
    x, y = np.concatenate(x_parts), np.concatenate(y_parts)
    categories = [f"{m}:{c}:{g}" for m, c, g in zip(models, corpora, groups)]
    observed = fixed_effect_slope(x, y, categories)
    boot = np.empty(BOOTSTRAPS, np.float64)
    for iteration in range(BOOTSTRAPS):
        ys = np.concatenate([standardized(panel["y_boot"][iteration]) for panel in use])
        boot[iteration] = fixed_effect_slope(x, ys, categories)
    result = scalar_result(observed, boot)
    result.update({"label": label, "models": sorted(included), "panels": len(use),
                   "standardized_within_model_corpus": True,
                   "fixed_effects": ["model", "corpus", "partition_or_pool"]})
    return result


def boundary_classification(results: dict, pooled: dict) -> dict:
    dev = [results[model][domain]["primary"]["beta_kappa"] for model in ("llama8b", "qwen4b") for domain in DOMAINS]
    selected = [results[model][domain]["primary"]["selected_minus_rejected"] for model in ("llama8b", "qwen4b") for domain in DOMAINS]
    mistral = [results["mistral7b"][domain]["primary"]["beta_kappa"] for domain in DOMAINS]
    conditions = {"all_four_development_slopes_negative": all(row["estimate"] < 0 for row in dev),
                  "three_development_slopes_holm_significant": sum(row.get("holm_reject_0p05", False) for row in dev) >= 3,
                  "selected_minus_rejected_negative_all_development": all(row["estimate"] < 0 for row in selected),
                  "pooled_all_negative_ci": pooled["all_models"]["ci95"][1] < 0,
                  "pooled_leave_qwen_negative_ci": pooled["leave_qwen4b_out"]["ci95"][1] < 0,
                  "mistral_both_negative": all(row["estimate"] < 0 for row in mistral)}
    gate = all(conditions.values())
    statuses = [row["power_status"] for row in dev + selected + mistral]
    power_limited = "descriptive" in statuses
    classification = "strong_support" if gate and not power_limited else (
        "power_limited_support" if gate else "partial_support" if sum(conditions.values()) >= 4 else "not_supported")
    return {"classification": classification, "frozen_gate_passed": gate, "conditions": conditions,
            "power_qualification": "downgraded because at least one required panel was predeclared descriptive" if power_limited else "not downgraded",
            "claim_boundary": "aggregate composition-matched group ranking only; no individual-tile causality or calibration"}


def corruption_classification(results: dict, pooled: dict) -> dict:
    dev = [results[model][domain]["primary"]["beta_p"] for model in ("llama8b", "qwen4b") for domain in DOMAINS]
    p100 = [results[model][domain]["primary"]["p100_minus_p0"] for model in ("llama8b", "qwen4b") for domain in DOMAINS]
    mistral = [results["mistral7b"][domain]["primary"]["beta_p"] for domain in DOMAINS]
    conditions = {"all_four_development_slopes_positive": all(row["estimate"] > 0 for row in dev),
                  "three_development_slopes_holm_significant": sum(row.get("holm_reject_0p05", False) for row in dev) >= 3,
                  "p100_worse_all_development": all(row["estimate"] > 0 for row in p100),
                  "pooled_all_positive_ci": pooled["all_models"]["ci95"][0] > 0,
                  "pooled_leave_qwen_positive_ci": pooled["leave_qwen4b_out"]["ci95"][0] > 0,
                  "mistral_both_positive": all(row["estimate"] > 0 for row in mistral)}
    gate = all(conditions.values())
    statuses = [row["power_status"] for row in dev + p100 + mistral]
    power_limited = "descriptive" in statuses
    classification = "strong_support" if gate and not power_limited else (
        "power_limited_support" if gate else "partial_support" if sum(conditions.values()) >= 4 else "not_supported")
    return {"classification": classification, "frozen_gate_passed": gate, "conditions": conditions,
            "power_qualification": "downgraded because at least one required panel was predeclared descriptive" if power_limited else "not downgraded",
            "claim_boundary": "corruption p is imposed stress, not a selector error-rate estimate"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--map-run", required=True)
    parser.add_argument("--llama-run", required=True)
    parser.add_argument("--qwen-run", required=True)
    parser.add_argument("--mistral-run", required=True)
    parser.add_argument("--mechanism-root", default=str(MECHANISM_ROOT))
    args = parser.parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise SystemExit("protocol hash mismatch")
    root = Path(os.environ["CAMPAIGN_ROOT"])
    bands = json.loads((root / "BAND_DEFINITIONS.json").read_text())
    corruptions = json.loads((root / "CORRUPTION_POOL_DEFINITIONS.json").read_text())
    promotion = json.loads((root / "ENDPOINT_PROMOTION.json").read_text())
    entries = json.loads((Path(args.map_run) / "derived_maps/map_manifest.json").read_text())
    map_hashes = {model: {} for model in MODELS}
    for entry in entries:
        header, _, digest = mapio.read_map(entry["path"], entry["sha256"])
        if header["policy"]["name"] != entry["policy"] or digest != entry["sha256"]:
            raise ValueError(f"map verification failed {entry['path']}")
        map_hashes[entry["model"]][entry["policy"]] = entry["sha256"]
    map_hashes = {model: {**values, "four_over_six": "non-map-baseline"} for model, values in map_hashes.items()}
    run_paths = {"llama8b": Path(args.llama_run), "qwen4b": Path(args.qwen_run),
                 "mistral7b": Path(args.mistral_run)}
    combined = {model: combine_cluster_data(model, run_paths[model], Path(args.mechanism_root)) for model in MODELS}
    family_rows = {name: [] for name in ("A_development_primary", "A_validation_primary",
                                         "B_development_primary", "B_validation_primary")}
    boundary_results, corruption_results = {}, {}
    boundary_pooled, corruption_pooled = [], []
    arrays, provenance = {}, {}
    for model in MODELS:
        boundary_results[model], corruption_results[model], arrays[model], provenance[model] = {}, {}, {}, {}
        for domain in DOMAINS:
            panel = combined[model][domain]
            arrays[model][domain] = save_paired_array(root, model, domain, panel, map_hashes[model])
            pstatus = promotion["models"][model][domain]
            boundary_results[model][domain] = analyze_boundary(model, domain, panel, bands["models"][model],
                                                                pstatus, family_rows, boundary_pooled)
            corruption_results[model][domain] = analyze_corruption(model, domain, panel, corruptions["models"][model],
                                                                    pstatus, family_rows, corruption_pooled)
            provenance[model][domain] = {key: panel[key] for key in (
                "evaluation_manifest_sha256", "new_evaluation_manifest_sha256",
                "report_manifest_hashes_equal", "scientific_identity_sha256",
                "identity_fields_verified", "report_identity_fields_verified", "old_run", "new_run",
                "old_report_sha256", "new_report_sha256", "windows", "clusters")}
    for rows in family_rows.values():
        holm(rows)
    boundary_pool = {"all_models": pooled_slope(boundary_pooled, set(MODELS), "boundary_all_models"),
                     "leave_qwen4b_out": pooled_slope(boundary_pooled, {"llama8b", "mistral7b"},
                                                       "boundary_leave_qwen4b_out")}
    corruption_pool = {"all_models": pooled_slope(corruption_pooled, set(MODELS), "corruption_all_models"),
                       "leave_qwen4b_out": pooled_slope(corruption_pooled, {"llama8b", "mistral7b"},
                                                         "corruption_leave_qwen4b_out")}
    classifications = {"boundary": boundary_classification(boundary_results, boundary_pool),
                       "corruption": corruption_classification(corruption_results, corruption_pool)}
    boundary_payload = {"schema": "mixfp4-boundary-results/v1", "protocol_sha256": args.protocol_sha256,
                        "bootstrap_replicates": BOOTSTRAPS, "models": boundary_results,
                        "pooled_standardized": boundary_pool, "classification": classifications["boundary"],
                        "holm_families": {key: value for key, value in family_rows.items() if key.startswith("A_")}}
    corruption_payload = {"schema": "mixfp4-corruption-results/v1", "protocol_sha256": args.protocol_sha256,
                          "bootstrap_replicates": BOOTSTRAPS, "models": corruption_results,
                          "pooled_standardized": corruption_pool, "classification": classifications["corruption"],
                          "holm_families": {key: value for key, value in family_rows.items() if key.startswith("B_")}}
    atomic_json(root / "BOUNDARY_RESULTS.json", boundary_payload)
    atomic_json(root / "CORRUPTION_RESULTS.json", corruption_payload)
    atomic_json(root / "CAMPAIGN_RESULTS.json", {
        "schema": "mixfp4-boundary-corruption-results/v1", "protocol_sha256": args.protocol_sha256,
        "bootstrap_replicates": BOOTSTRAPS, "classifications": classifications,
        "paired_arrays": arrays, "provenance": provenance,
        "claims_prohibited": ["reliable individual-tile causal signs or magnitudes",
                              "calibrated individual-tile finite effects", "a true wrong-tile percentage",
                              "k=3 as a simultaneous confidence guarantee", "universal CE/KL/conjunction optimality",
                              "generalization to all LLMs", "native FP4/E0M3 Tensor Core execution",
                              "latency, throughput, speedup, runtime overhead, area, power, or Blackwell performance"]})
    boundary_csv = []
    for model in MODELS:
        for domain in DOMAINS:
            for row in boundary_results[model][domain]["bands"]:
                result = row["delta_vs_four_over_six"]
                boundary_csv.append({"model": model, "corpus": domain, "partition": row["partition"],
                                     "side": row["side"], "band": row["band"], "policy": row["policy"],
                                     "tiles": row["tiles"], "mean_kappa": row["mean_kappa"],
                                     "absolute_ppl": row["absolute_ppl"], "delta_nll": result["estimate"],
                                     "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1],
                                     "relative_ppl_change": result["relative_ppl_change"]})
    write_csv(root / "BOUNDARY_RESULTS.csv", boundary_csv)
    corruption_csv = []
    for model in MODELS:
        for domain in DOMAINS:
            for row in corruption_results[model][domain]["levels"]:
                result = row.get("delta_vs_p0", {"estimate": 0.0, "ci95": [0.0, 0.0], "relative_ppl_change": 0.0})
                corruption_csv.append({"model": model, "corpus": domain, "kind": "near_boundary",
                                       "pool": row["pool"], "p": row["p"], "policy": row["policy"],
                                       "absolute_ppl": row["absolute_ppl"], "delta_nll": result["estimate"],
                                       "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1],
                                       "relative_ppl_change": result["relative_ppl_change"]})
            for row in corruption_results[model][domain]["matched_random_p050"]:
                result = row["delta_vs_p0"]
                corruption_csv.append({"model": model, "corpus": domain, "kind": "matched_random",
                                       "pool": row["replicate"], "p": 0.5, "policy": row["policy"],
                                       "absolute_ppl": row["absolute_ppl"], "delta_nll": result["estimate"],
                                       "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1],
                                       "relative_ppl_change": result["relative_ppl_change"]})
    write_csv(root / "CORRUPTION_RESULTS.csv", corruption_csv)
    launch = json.loads((runtime.run_dir / "launch_record.json").read_text())
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "boundary-corruption", "protocol_freeze_sha256": args.protocol_sha256,
        "source": {"model_id": None, "model_revision": None, "tokenizer_revision": None,
                   "model_class": None, "module_manifest_sha256": None,
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
        "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(root / name) for name in (
            "BOUNDARY_RESULTS.json", "BOUNDARY_RESULTS.csv", "CORRUPTION_RESULTS.json",
            "CORRUPTION_RESULTS.csv", "CAMPAIGN_RESULTS.json")], "summary": classifications,
            "uncertainty": {"bootstrap_replicates": BOOTSTRAPS},
            "attempted_endpoints": list(family_rows), "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"classifications": classifications,
                      "family_counts": {key: len(value) for key, value in family_rows.items()},
                      "paired_arrays": sum(len(value) for value in arrays.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
