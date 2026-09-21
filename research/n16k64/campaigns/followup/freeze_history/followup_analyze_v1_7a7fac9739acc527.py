"""Frozen paired natural-cluster analysis for the N16K64 follow-up.

This module combines newly evaluated policies with the hash-verified baseline/full
evidence from the completed mechanism campaign.  All inferential families, seeds,
contrasts, gates, and sign conventions are fixed in FOLLOWUP_PROTOCOL.json before
GPU outcomes are produced.
"""
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


B = 10_000
BOOT_SEED = 20260917
MODELS = ("llama8b", "qwen4b", "mistral7b")
DOMAINS = ("c4", "wiki")
MECHANISM_RUN = {
    "llama8b": "V22_full_llama8b_attempt1",
    "qwen4b": "V21_full_qwen4b_attempt2",
    "mistral7b": "V30_mistral_one_shot_attempt1",
}
DOSE_GROUP = [f"dose_group_only_bin{i:02d}" for i in range(1, 9)]
DOSE_MINUS = [f"dose_full_minus_bin{i:02d}" for i in range(1, 9)]
OBJECTIVE_POLICY = {
    "four_over_six": "four_over_six",
    "ce_natural": "objective_ce_natural",
    "kl_natural": "objective_kl_natural",
    "conjunction": "full",
    "ce_matched_k": "objective_ce_matched_k",
    "kl_matched_k": "objective_kl_matched_k",
    "random_matched_k": "objective_random_matched_k",
}
OBJECTIVE_MAP_POLICY = {
    "ce_natural": "objective_ce_natural",
    "kl_natural": "objective_kl_natural",
    "conjunction": "objective_conjunction",
    "ce_matched_k": "objective_ce_matched_k",
    "kl_matched_k": "objective_kl_matched_k",
    "random_matched_k": "objective_random_matched_k",
}
OBJECTIVE_CONTRASTS = (
    ("ce_matched_k_minus_conjunction", "ce_matched_k", "conjunction"),
    ("kl_matched_k_minus_conjunction", "kl_matched_k", "conjunction"),
    ("ce_matched_k_minus_kl_matched_k", "ce_matched_k", "kl_matched_k"),
    ("ce_matched_k_minus_random", "ce_matched_k", "random_matched_k"),
    ("kl_matched_k_minus_random", "kl_matched_k", "random_matched_k"),
    ("conjunction_minus_random", "conjunction", "random_matched_k"),
)


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def cluster_labels(meta: dict, domain: str) -> list[str]:
    if domain == "wiki":
        return [f'a{row["first_article"]}' for row in meta["window_articles"]]
    if domain == "c4":
        documents = meta["documents"]
        per = int(meta["windows"]) // len(documents)
        return [d["document_sha256"] for d in documents for _ in range(per)]
    raise ValueError(domain)


def load_run(path: Path | str) -> tuple[dict, dict[str, dict]]:
    ppl = Path(path) / "ppl"
    report = json.loads((ppl / "ppl_report.json").read_text())
    metas = {domain: json.loads((ppl / f"windows_{domain}.json").read_text()) for domain in DOMAINS}
    return report, metas


def combine_cluster_data(model: str, new_run: Path, mechanism_root: Path) -> dict[str, dict]:
    old_run = mechanism_root / "runs" / MECHANISM_RUN[model]
    old_report, old_meta = load_run(old_run)
    new_report, new_meta = load_run(new_run)
    if old_report["evaluation_manifest_sha256"] != new_report["evaluation_manifest_sha256"]:
        raise ValueError(f"{model}: evaluation manifest drift")
    out = {}
    for domain in DOMAINS:
        if old_meta[domain]["token_sha256"] != new_meta[domain]["token_sha256"]:
            raise ValueError(f"{model}/{domain}: frozen window token drift")
        old_labels, new_labels = cluster_labels(old_meta[domain], domain), cluster_labels(new_meta[domain], domain)
        if old_labels != new_labels:
            raise ValueError(f"{model}/{domain}: natural-cluster assignment drift")
        unique = list(dict.fromkeys(old_labels))
        index = {label: i for i, label in enumerate(unique)}
        base_rows = old_report["evaluation"]["four_over_six"][domain]["windows"]
        tokens = np.zeros(len(unique), np.float64)
        for row, label in zip(base_rows, old_labels):
            tokens[index[label]] += float(row["tokens"])
        policy_rows = {
            "four_over_six": base_rows,
            "full": old_report["evaluation"]["full"][domain]["windows"],
        }
        for policy in new_report["evaluation"]:
            policy_rows[policy] = new_report["evaluation"][policy][domain]["windows"]
        nll = {}
        absolute_ppl = {
            "four_over_six": float(old_report["evaluation"]["four_over_six"][domain]["ppl"]),
            "full": float(old_report["evaluation"]["full"][domain]["ppl"]),
        }
        absolute_ppl.update({p: float(new_report["evaluation"][p][domain]["ppl"]) for p in new_report["evaluation"]})
        baseline_window_tokens = [int(row["tokens"]) for row in base_rows]
        for policy, rows in policy_rows.items():
            if len(rows) != len(old_labels) or [int(row["tokens"]) for row in rows] != baseline_window_tokens:
                raise ValueError(f"{model}/{domain}/{policy}: paired window/token drift")
            values = np.zeros(len(unique), np.float64)
            for row, label in zip(rows, old_labels):
                values[index[label]] += float(row["nll_sum"])
            nll[policy] = values
        out[domain] = {
            "cluster_ids": np.asarray(unique), "tokens": tokens, "nll": nll, "absolute_ppl": absolute_ppl,
            "evaluation_manifest_sha256": old_report["evaluation_manifest_sha256"],
            "old_run": str(old_run), "new_run": str(new_run),
            "old_report_sha256": sha(old_run / "ppl/ppl_report.json"),
            "new_report_sha256": sha(new_run / "ppl/ppl_report.json"),
            "windows": len(old_labels), "clusters": len(unique),
        }
    return out


def bootstrap_delta(delta_nll: np.ndarray, tokens: np.ndarray, label: str) -> dict:
    estimate = float(delta_nll.sum() / tokens.sum())
    centered_cluster = delta_nll - estimate * tokens
    rng = np.random.default_rng(seed_for(BOOT_SEED, label))
    idx = rng.integers(0, len(tokens), size=(B, len(tokens)))
    noise = centered_cluster[idx].sum(axis=1) / tokens[idx].sum(axis=1)
    boot = estimate + noise
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(estimate)).sum())) / (B + 1)
    rates = delta_nll / tokens
    return {
        "estimate": estimate, "ci95": [float(lo), float(hi)], "p_two_sided_plus_one": float(p),
        "bootstrap_replicates": B, "clusters": int(len(tokens)), "tokens": int(tokens.sum()),
        "relative_ppl_change": float(math.expm1(estimate)),
        "cluster_regression_probability": float(np.mean(rates > 0)),
        "cluster_quantiles": {f"q{q}": float(np.quantile(rates, q / 100)) for q in (90, 95, 99)},
        "cluster_worst": float(np.max(rates)),
    }


def bootstrap_scalar_pair(rate_a: np.ndarray, rate_b: np.ndarray, functional, label: str) -> dict:
    observed = float(functional(rate_a) - functional(rate_b))
    rng = np.random.default_rng(seed_for(BOOT_SEED, label))
    idx = rng.integers(0, len(rate_a), size=(B, len(rate_a)))
    sampled = np.empty(B, np.float64)
    for start in range(0, B, 250):
        batch = idx[start:start + 250]
        sampled[start:start + len(batch)] = [functional(rate_a[x]) - functional(rate_b[x]) for x in batch]
    noise = sampled - sampled.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(observed)).sum())) / (B + 1)
    return {"estimate": observed, "ci95": [float(lo), float(hi)], "p_two_sided_plus_one": float(p),
            "bootstrap_replicates": B}


def holm(rows: list[dict]) -> None:
    if not rows:
        return
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided_plus_one"])
    running, count = 0.0, len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * rows[index]["p_two_sided_plus_one"]))
        rows[index]["holm_adjusted_p"] = float(running)
        rows[index]["holm_reject_0p05"] = bool(running < 0.05)


def weighted_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    sw = np.sqrt(weights)
    beta = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
    fitted = design @ beta
    mean = np.average(y, weights=weights)
    total = float(np.sum(weights * (y - mean) ** 2))
    residual = float(np.sum(weights * (y - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(beta[1]), float(beta[0]), float(r2)


def scalar_bootstrap_result(observed: float, samples: np.ndarray, signed: bool = True) -> dict:
    finite = samples[np.isfinite(samples)]
    if not np.isfinite(observed) or len(finite) != len(samples):
        return {"estimate": float(observed), "ci95": [None, None], "p_two_sided_plus_one": None,
                "bootstrap_replicates": int(len(samples)), "finite_replicates": int(len(finite))}
    noise = finite - finite.mean()
    lo, hi = np.percentile(observed + noise, [2.5, 97.5])
    result = {"estimate": float(observed), "ci95": [float(lo), float(hi)],
              "bootstrap_replicates": int(len(finite)), "finite_replicates": int(len(finite))}
    if signed:
        result["p_two_sided_plus_one"] = float((1 + int((np.abs(noise) >= abs(observed)).sum())) / (len(finite) + 1))
    return result


def safe_corr(kind: str, x: np.ndarray, y: np.ndarray) -> float:
    if kind == "pearson":
        return float(ss.pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(ss.spearmanr(x, y).statistic)
    if kind == "kendall":
        return float(ss.kendalltau(x, y).statistic)
    raise ValueError(kind)


def dose_panel_statistics(delta_matrix: np.ndarray, tokens: np.ndarray, x_ce: np.ndarray,
                          x_margin: np.ndarray, label: str) -> tuple[dict, np.ndarray]:
    observed_y = delta_matrix.sum(axis=1) / tokens.sum()
    rng = np.random.default_rng(seed_for(BOOT_SEED, label, "joint-eight-bin"))
    y_boot = np.empty((B, 8), np.float64)
    for start in range(0, B, 250):
        size = min(250, B - start)
        idx = rng.integers(0, len(tokens), size=(size, len(tokens)))
        denominator = tokens[idx].sum(axis=1)
        for bin_index in range(8):
            y_boot[start:start + size, bin_index] = delta_matrix[bin_index][idx].sum(axis=1) / denominator
    variance = np.var(y_boot, axis=0, ddof=1)
    weights = 1.0 / np.maximum(variance, 1e-18)
    metrics, samples = {}, {}
    for prefix, x in (("ce", x_ce), ("margin", x_margin)):
        for kind in ("pearson", "spearman", "kendall"):
            obs = safe_corr(kind, x, observed_y)
            boot = np.asarray([safe_corr(kind, x, row) for row in y_boot])
            metrics[f"{prefix}_{kind}"] = scalar_bootstrap_result(obs, boot, signed=True)
            samples[f"{prefix}_{kind}"] = boot
        slope, intercept, r2 = weighted_fit(x, observed_y, weights)
        fits = np.asarray([weighted_fit(x, row, weights) for row in y_boot])
        metrics[f"{prefix}_slope"] = scalar_bootstrap_result(slope, fits[:, 0], signed=True)
        metrics[f"{prefix}_intercept"] = scalar_bootstrap_result(intercept, fits[:, 1], signed=True)
        metrics[f"{prefix}_r_squared"] = scalar_bootstrap_result(r2, fits[:, 2], signed=False)
        samples[f"{prefix}_slope"] = fits[:, 0]
    adjacent = np.diff(observed_y)
    metrics["ordered_monotonicity"] = {
        "expected": "observed marginal delta NLL nondecreasing from strongest bin01 to weakest bin08",
        "adjacent_nonnegative": int(np.sum(adjacent >= 0)), "adjacent_total": 7,
        "strictly_monotone": bool(np.all(adjacent >= 0)), "adjacent_differences": adjacent.tolist(),
    }
    metrics["inverse_variance_weights"] = weights.tolist()
    return {"observed_marginal": observed_y.tolist(), "metrics": metrics}, y_boot


def fixed_effect_slope(x: np.ndarray, y: np.ndarray, model: list[str], corpus: list[str]) -> float:
    model_levels = sorted(set(model))
    corpus_levels = sorted(set(corpus))
    cols = [x, np.ones(len(x))]
    for level in model_levels[1:]:
        cols.append(np.asarray([v == level for v in model], np.float64))
    for level in corpus_levels[1:]:
        cols.append(np.asarray([v == level for v in corpus], np.float64))
    return float(np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][0])


def standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)


def pooled_dose(panels: list[dict], predictor: str, include_models: set[str], label: str) -> dict:
    selected = [p for p in panels if p["model"] in include_models]
    x_parts, y_parts, models, corpora = [], [], [], []
    for panel in selected:
        x = panel["x_ce"] if predictor == "ce" else panel["x_margin"]
        x_parts.append(standardize(x))
        y_parts.append(standardize(panel["y"]))
        models.extend([panel["model"]] * 8)
        corpora.extend([panel["domain"]] * 8)
    x_all, y_all = np.concatenate(x_parts), np.concatenate(y_parts)
    observed = fixed_effect_slope(x_all, y_all, models, corpora)
    boot = np.empty(B, np.float64)
    for b in range(B):
        ys = np.concatenate([standardize(panel["y_boot"][b]) for panel in selected])
        boot[b] = fixed_effect_slope(x_all, ys, models, corpora)
    result = scalar_bootstrap_result(observed, boot, signed=True)
    result.update({"predictor": predictor, "standardized_within_model_corpus": True,
                   "fixed_effects": ["model", "corpus"], "models": sorted(include_models),
                   "panels": len(selected), "bin_observations": len(x_all), "label": label})
    return result


def save_paired_array(root: Path, model: str, domain: str, panel: dict, policy_map_hashes: dict[str, str]) -> dict:
    policies = list(panel["nll"])
    path = root / "arrays" / f"{model}_{domain}_paired_cluster_nll.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, policies=np.asarray(policies), cluster_ids=panel["cluster_ids"],
             cluster_tokens=panel["tokens"], cluster_nll_sum=np.stack([panel["nll"][p] for p in policies]),
             policy_map_sha256=np.asarray([policy_map_hashes.get(p, "non-map-or-reused") for p in policies]),
             evaluation_manifest_sha256=np.asarray(panel["evaluation_manifest_sha256"]))
    return {"path": str(path), "sha256": sha(path), "policies": len(policies),
            "clusters": int(len(panel["tokens"])), "tokens": int(panel["tokens"].sum())}


def analyze_dose(model: str, domain: str, panel: dict, bin_def: dict, family_rows: dict,
                 pooled_panels: list[dict]) -> dict:
    nll, tokens = panel["nll"], panel["tokens"]
    definitions = bin_def["models"][model]["definitions"]
    x_ce = np.asarray([row["score_summary"]["sum_ce_mean"] for row in definitions], np.float64)
    x_margin = np.asarray([row["score_summary"]["mean_combined_election_margin"] for row in definitions], np.float64)
    delta_matrix = np.stack([nll["full"] - nll[policy] for policy in DOSE_MINUS])
    group_matrix = np.stack([nll[policy] - nll["four_over_six"] for policy in DOSE_GROUP])
    bins = []
    for i in range(8):
        marginal = bootstrap_delta(delta_matrix[i], tokens, f"{model}:{domain}:A:bin{i + 1}:marginal")
        group = bootstrap_delta(group_matrix[i], tokens, f"{model}:{domain}:A:bin{i + 1}:group")
        bins.append({"bin": i + 1, "sum_ce_mean": float(x_ce[i]), "mean_combined_election_margin": float(x_margin[i]),
                     "tiles": definitions[i]["score_summary"]["tiles"],
                     "full_context_marginal": marginal, "group_only_vs_baseline": group,
                     "absolute_ppl": {"group_only": panel["absolute_ppl"][DOSE_GROUP[i]],
                                      "full_minus_bin": panel["absolute_ppl"][DOSE_MINUS[i]],
                                      "full": panel["absolute_ppl"]["full"],
                                      "four_over_six": panel["absolute_ppl"]["four_over_six"]}})
    statistics, y_boot = dose_panel_statistics(delta_matrix, tokens, x_ce, x_margin, f"{model}:{domain}:A")
    family = "A_validation" if model == "mistral7b" else "A_development"
    for endpoint in ("ce_pearson", "ce_spearman", "ce_kendall", "ce_slope",
                     "margin_pearson", "margin_spearman", "margin_kendall", "margin_slope"):
        row = statistics["metrics"][endpoint]
        row.update({"model": model, "domain": domain, "family": family, "endpoint": endpoint})
        family_rows[family].append(row)
    pooled_panels.append({"model": model, "domain": domain, "x_ce": x_ce, "x_margin": x_margin,
                          "y": np.asarray(statistics["observed_marginal"]), "y_boot": y_boot})
    return {"bins": bins, "statistics": statistics,
            "remainder_accounting": {k: bin_def["models"][model][k] for k in
                                     ("selected_tiles_before_remainder", "retained_union_tiles", "excluded_remainder_count")}}


def analyze_objective(model: str, domain: str, panel: dict, manifest: dict, selection: dict,
                      family_rows: dict) -> dict:
    nll, tokens = panel["nll"], panel["tokens"]
    policies = {}
    for label, evaluated in OBJECTIVE_POLICY.items():
        if label == "four_over_six":
            selected, map_info = 0, None
        else:
            map_policy = OBJECTIVE_MAP_POLICY[label]
            entry = manifest[(model, map_policy)]
            selected, map_info = entry["selected_tiles"], {"path": entry["path"], "sha256": entry["sha256"],
                                                            "mask_payload_sha256": entry["mask_payload_sha256"]}
        if label == "four_over_six":
            delta = bootstrap_delta(np.zeros_like(tokens), tokens, f"{model}:{domain}:B:baseline")
        else:
            delta = bootstrap_delta(nll[evaluated] - nll["four_over_six"], tokens, f"{model}:{domain}:B:{label}:baseline")
        policies[label] = {"evaluated_policy": evaluated, "selected_tiles": selected, "map": map_info,
                           "absolute_ppl": panel["absolute_ppl"][evaluated],
                           "delta_vs_four_over_six": delta,
                           "interpretation": "natural threshold; unequal budgets" if label in ("ce_natural", "kl_natural") else
                                             "matched per-stratum budget" if "matched" in label or label == "conjunction" else "baseline"}
    contrasts = {}
    family = "B_validation" if model == "mistral7b" else "B_development"
    for endpoint, left, right in OBJECTIVE_CONTRASTS:
        left_policy, right_policy = OBJECTIVE_POLICY[left], OBJECTIVE_POLICY[right]
        result = bootstrap_delta(nll[left_policy] - nll[right_policy], tokens, f"{model}:{domain}:B:{endpoint}")
        result.update({"model": model, "domain": domain, "family": family, "endpoint": endpoint,
                       "left": left, "right": right})
        contrasts[endpoint] = result
        family_rows[family].append(result)
    detail = selection["models"][model]["objective"]
    return {"policies": policies, "matched_budget_contrasts": contrasts,
            "pairwise_overlap": detail["pairwise_overlap"], "strata": detail["strata"],
            "selected_counts": detail["selected_counts"]}


def analyze_mistral_veto(domain: str, panel: dict, family_rows: dict) -> dict:
    nll, tokens = panel["nll"], panel["tokens"]
    actual_policy = "veto_full_plus_ce_vetoed_kl_approved"
    random_policy = "veto_full_plus_ce_vetoed_kl_matched_random"
    actual = nll[actual_policy] - nll["full"]
    random = nll[random_policy] - nll["full"]
    actual_full = bootstrap_delta(actual, tokens, f"mistral7b:{domain}:C:actual-full")
    random_full = bootstrap_delta(random, tokens, f"mistral7b:{domain}:C:random-full")
    actual_random = bootstrap_delta(actual - random, tokens, f"mistral7b:{domain}:C:actual-random")
    rate_a, rate_r = actual / tokens, random / tokens
    tails = {
        "q90_actual_minus_random": bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .90), f"mistral7b:{domain}:C:q90"),
        "q95_actual_minus_random": bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .95), f"mistral7b:{domain}:C:q95"),
        "q99_actual_minus_random": bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .99), f"mistral7b:{domain}:C:q99"),
        "regression_probability_actual_minus_random": bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.mean(x > 0), f"mistral7b:{domain}:C:prob"),
        "worst_actual_minus_random": bootstrap_scalar_pair(rate_a, rate_r, np.max, f"mistral7b:{domain}:C:worst"),
    }
    for endpoint, result in (("mean_actual_minus_full", actual_full), ("mean_actual_minus_random", actual_random), *tails.items()):
        result.update({"model": "mistral7b", "domain": domain, "family": "C_mistral", "endpoint": endpoint})
        family_rows["C_mistral"].append(result)
    return {"absolute_ppl": {"conjunction": panel["absolute_ppl"]["full"],
                              "actual_addback": panel["absolute_ppl"][actual_policy],
                              "matched_random_addback": panel["absolute_ppl"][random_policy]},
            "actual_vs_conjunction": actual_full, "matched_random_vs_conjunction": random_full,
            "actual_vs_matched_random": actual_random, "tails": tails}


def classify_dose(results: dict) -> dict:
    out = {}
    for predictor, expected in (("ce", "positive"), ("margin", "negative")):
        rows = [results[m][d]["statistics"]["metrics"][f"{predictor}_spearman"]
                for m in ("llama8b", "qwen4b") for d in DOMAINS]
        signs = [r["estimate"] > 0 if expected == "positive" else r["estimate"] < 0 for r in rows]
        excludes = [r["ci95"][0] > 0 if expected == "positive" else r["ci95"][1] < 0 for r in rows]
        passed = len(rows) == 4 and all(signs) and sum(excludes) >= 3
        out[predictor] = {"expected_direction": expected, "all_four_expected_direction": all(signs),
                          "ci_excluding_zero_in_expected_direction": sum(excludes), "passed": passed}
    overall = all(v["passed"] for v in out.values())
    return {"classification": "supported" if overall else "partially_supported" if any(v["passed"] for v in out.values()) else "not_supported",
            "gate": "for each predictor, all four development Spearman estimates have expected direction and at least three 95% CIs exclude zero",
            "predictor_gates": out, "passed": overall,
            "boundary": "group-level ranking evidence only; never individual-tile calibration or causality"}


def classify_objective(results: dict) -> dict:
    endpoints = defaultdict(list)
    for model in ("llama8b", "qwen4b"):
        for domain in DOMAINS:
            for name, row in results[model][domain]["matched_budget_contrasts"].items():
                endpoints[name].append(row)
    conjunction = {}
    for single in ("ce", "kl"):
        name = f"{single}_matched_k_minus_conjunction"
        rows = endpoints[name]
        passed = all(r["estimate"] > 0 for r in rows) and sum(r.get("holm_reject_0p05", False) for r in rows) >= 3
        conjunction[f"conjunction_better_than_{single}_matched_k"] = {
            "positive_in_all_four_development_endpoints": all(r["estimate"] > 0 for r in rows),
            "holm_significant_endpoints": sum(r.get("holm_reject_0p05", False) for r in rows), "passed": passed}
    info = {}
    for selector in ("ce_matched_k", "kl_matched_k", "conjunction"):
        name = f"{selector}_minus_random" if selector != "conjunction" else "conjunction_minus_random"
        rows = endpoints[name]
        info[selector] = {"negative_in_all_four_development_endpoints": all(r["estimate"] < 0 for r in rows),
                          "holm_significant_endpoints": sum(r.get("holm_reject_0p05", False) for r in rows),
                          "passed": all(r["estimate"] < 0 for r in rows) and sum(r.get("holm_reject_0p05", False) for r in rows) >= 3}
    conj_pass = all(v["passed"] for v in conjunction.values())
    return {"classification": "conjunction_supported_over_both_single_objectives" if conj_pass else "no_universal_conjunction_superiority",
            "gate": "each single-objective matched-K minus conjunction contrast is positive in all four development endpoints and Holm-significant in at least three",
            "conjunction_gates": conjunction, "information_vs_random_gates": info, "passed": conj_pass,
            "boundary": "natural-threshold maps are descriptive because selected counts differ"}


def classify_veto(veto: dict) -> dict:
    rows_full = [veto[d]["actual_vs_conjunction"] for d in DOMAINS]
    rows_random = [veto[d]["actual_vs_matched_random"] for d in DOMAINS]
    passed = (all(r["estimate"] > 0 for r in rows_full + rows_random) and
              all(r.get("holm_reject_0p05", False) for r in rows_full + rows_random))
    return {"classification": "mistral_veto_symmetry_supported" if passed else "mistral_veto_symmetry_not_supported",
            "gate": "actual add-back is worse than conjunction and matched random in both corpora, with all four mean contrasts Holm-significant in the frozen C-Mistral family",
            "passed": passed,
            "boundary": "completes the one-shot Mistral matrix; does not revise the completed mechanism development gate"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--map-run", required=True)
    ap.add_argument("--llama-run", required=True)
    ap.add_argument("--qwen-run", required=True)
    ap.add_argument("--mistral-run", required=True)
    ap.add_argument("--mechanism-root", required=True)
    args = ap.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise SystemExit("protocol hash mismatch")
    root = Path(os.environ["CAMPAIGN_ROOT"])
    mechanism = Path(args.mechanism_root)
    map_root = Path(args.map_run) / "derived_maps"
    manifest_entries = json.loads((map_root / "map_manifest.json").read_text())
    manifest = {(e["model"], e["policy"]): e for e in manifest_entries}
    bin_def = json.loads((map_root / "bin_definitions.json").read_text())
    selection = json.loads((map_root / "selection_definitions.json").read_text())
    for entry in manifest_entries:
        header, _, digest = mapio.read_map(entry["path"], entry["sha256"])
        if digest != entry["sha256"] or header["totals"]["selected_tiles"] != entry["selected_tiles"]:
            raise ValueError(f"map verification failed: {entry['path']}")

    run_paths = {"llama8b": Path(args.llama_run), "qwen4b": Path(args.qwen_run), "mistral7b": Path(args.mistral_run)}
    combined = {model: combine_cluster_data(model, run_paths[model], mechanism) for model in MODELS}
    policy_map_hashes = {e["policy"]: e["sha256"] for e in manifest_entries}
    policy_map_hashes["full"] = "reused-frozen-mechanism-full"
    policy_map_hashes["four_over_six"] = "non-map-baseline"
    paired_arrays, family_rows = {}, {name: [] for name in ("A_development", "A_validation", "B_development", "B_validation", "C_mistral")}
    dose_results, objective_results, pooled_panels = {}, {}, []
    for model in MODELS:
        dose_results[model], objective_results[model], paired_arrays[model] = {}, {}, {}
        for domain in DOMAINS:
            panel = combined[model][domain]
            paired_arrays[model][domain] = save_paired_array(root, model, domain, panel, policy_map_hashes)
            dose_results[model][domain] = analyze_dose(model, domain, panel, bin_def, family_rows, pooled_panels)
            objective_results[model][domain] = analyze_objective(model, domain, panel, manifest, selection, family_rows)

    mistral_veto = {domain: analyze_mistral_veto(domain, combined["mistral7b"][domain], family_rows) for domain in DOMAINS}
    for rows in family_rows.values():
        holm(rows)

    pooled = {
        "ce_all_models": pooled_dose(pooled_panels, "ce", set(MODELS), "all_models"),
        "ce_leave_qwen4b_out": pooled_dose(pooled_panels, "ce", {"llama8b", "mistral7b"}, "leave_qwen4b_out"),
        "margin_all_models": pooled_dose(pooled_panels, "margin", set(MODELS), "all_models"),
        "margin_leave_qwen4b_out": pooled_dose(pooled_panels, "margin", {"llama8b", "mistral7b"}, "leave_qwen4b_out"),
    }
    classifications = {
        "dose_response": classify_dose(dose_results),
        "objective_ablation": classify_objective(objective_results),
        "mistral_veto_completion": classify_veto(mistral_veto),
    }
    provenance = {
        model: {domain: {key: combined[model][domain][key] for key in
                         ("evaluation_manifest_sha256", "old_run", "new_run", "old_report_sha256", "new_report_sha256", "windows", "clusters")}
                for domain in DOMAINS} for model in MODELS
    }
    dose_payload = {"schema": "mixfp4-followup-dose-response/v1", "protocol_sha256": args.protocol_sha256,
                    "bootstrap_replicates": B, "models": dose_results, "pooled_standardized": pooled,
                    "classification": classifications["dose_response"], "family_rows": {k: v for k, v in family_rows.items() if k.startswith("A_")}}
    objective_payload = {"schema": "mixfp4-followup-objective-ablation/v1", "protocol_sha256": args.protocol_sha256,
                         "bootstrap_replicates": B, "models": objective_results,
                         "classification": classifications["objective_ablation"], "family_rows": {k: v for k, v in family_rows.items() if k.startswith("B_")}}
    veto_payload = {"schema": "mixfp4-followup-mistral-veto-completion/v1", "protocol_sha256": args.protocol_sha256,
                    "bootstrap_replicates": B, "class": "ce_vetoed_kl_approved", "corpora": mistral_veto,
                    "matching_definition": selection["models"]["mistral7b"]["veto"],
                    "classification": classifications["mistral_veto_completion"], "family_rows": family_rows["C_mistral"]}
    runtime.atomic_json(root / "DOSE_RESPONSE_RESULTS.json", dose_payload)
    runtime.atomic_json(root / "OBJECTIVE_ABLATION_RESULTS.json", objective_payload)
    runtime.atomic_json(root / "MISTRAL_VETO_COMPLETION.json", veto_payload)
    runtime.atomic_json(root / "FOLLOWUP_RESULTS.json", {
        "schema": "mixfp4-n16k64-followup-results/v1", "protocol_sha256": args.protocol_sha256,
        "bootstrap_replicates": B, "paired_arrays": paired_arrays, "provenance": provenance,
        "classifications": classifications,
        "claims_prohibited": ["reliable individual-tile causal signs or magnitudes", "k=3 simultaneous per-tile confidence guarantee",
                              "universal CE or KL superiority", "native FP4/E0M3 Tensor Core execution",
                              "latency or speedup", "runtime overhead", "area", "power", "Blackwell performance from fake quantization"],
    })

    dose_csv = []
    for model in MODELS:
        for domain in DOMAINS:
            for row in dose_results[model][domain]["bins"]:
                for endpoint in ("full_context_marginal", "group_only_vs_baseline"):
                    result = row[endpoint]
                    dose_csv.append({"model": model, "domain": domain, "bin": row["bin"], "endpoint": endpoint,
                                     "tiles": row["tiles"], "sum_ce_mean": row["sum_ce_mean"],
                                     "mean_combined_election_margin": row["mean_combined_election_margin"],
                                     "estimate": result["estimate"], "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1],
                                     "relative_ppl_change": result["relative_ppl_change"], "p_two_sided_plus_one": result["p_two_sided_plus_one"]})
    write_csv(root / "DOSE_RESPONSE_RESULTS.csv", dose_csv)
    objective_csv = []
    for model in MODELS:
        for domain in DOMAINS:
            out = objective_results[model][domain]
            for label, row in out["policies"].items():
                delta = row["delta_vs_four_over_six"]
                objective_csv.append({"model": model, "domain": domain, "row_type": "policy", "name": label,
                                      "selected_tiles": row["selected_tiles"], "absolute_ppl": row["absolute_ppl"],
                                      "estimate": delta["estimate"], "ci95_low": delta["ci95"][0], "ci95_high": delta["ci95"][1],
                                      "relative_ppl_change": delta["relative_ppl_change"], "p_two_sided_plus_one": delta["p_two_sided_plus_one"]})
            for name, row in out["matched_budget_contrasts"].items():
                objective_csv.append({"model": model, "domain": domain, "row_type": "contrast", "name": name,
                                      "selected_tiles": "", "absolute_ppl": "", "estimate": row["estimate"],
                                      "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1],
                                      "relative_ppl_change": row["relative_ppl_change"],
                                      "p_two_sided_plus_one": row["p_two_sided_plus_one"],
                                      "holm_adjusted_p": row["holm_adjusted_p"]})
    write_csv(root / "OBJECTIVE_ABLATION_RESULTS.csv", objective_csv)

    launch = json.loads((runtime.run_dir / "launch_record.json").read_text())
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "aligned-followup", "protocol_freeze_sha256": args.protocol_sha256,
        "source": {"model_id": None, "model_revision": None, "tokenizer_revision": None, "model_class": None,
                   "module_manifest_sha256": None, "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
        "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(root / name) for name in
                                     ("DOSE_RESPONSE_RESULTS.json", "DOSE_RESPONSE_RESULTS.csv",
                                      "OBJECTIVE_ABLATION_RESULTS.json", "OBJECTIVE_ABLATION_RESULTS.csv",
                                      "MISTRAL_VETO_COMPLETION.json", "FOLLOWUP_RESULTS.json")],
                    "summary": classifications, "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": list(family_rows), "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps({"classifications": classifications,
                      "family_counts": {k: len(v) for k, v in family_rows.items()},
                      "paired_arrays": sum(len(v) for v in paired_arrays.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
