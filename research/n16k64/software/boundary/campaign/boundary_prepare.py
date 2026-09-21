"""Outcome-blind coverage, corruption-pool, and paired-power gates.

This module must run before any new boundary/corruption PPL evaluation.  It reads
only frozen score moments/maps and completed-campaign paired arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from campaign import runtime
from campaign.boundary_common import (
    CANDIDATE_B,
    CORRUPTION_LEVELS,
    CORRUPTION_POOLS,
    DOMAINS,
    FOLLOWUP_ROOT,
    K_PRIMARY,
    MODELS,
    PARTITIONS_BY_B,
    SEED_ROOT,
    atomic_json,
    canonical_sha,
    keyed_digest,
    load_model_scores,
    module_stratum,
    round_half_up,
    score_summary,
    sha256_file,
    stratified_priority,
    tile_set_sha,
)


POWER_REPLICATES = 10_000
SESOI = (0.0010, 0.0025)
P_VALUES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def json_number(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    return "+inf" if value > 0 else "-inf"


def ordered_indices(mask: torch.Tensor, kappa: torch.Tensor, selected: bool) -> list[int]:
    flat = mask.flatten()
    candidates = torch.nonzero(flat if selected else ~flat, as_tuple=False).flatten()
    if not candidates.numel():
        return []
    order = torch.argsort(kappa[candidates], descending=True, stable=True)
    return [int(v) for v in candidates[order].tolist()]


def rotating_exclusion_ranks(k: int, b: int, partition: int) -> list[int]:
    remainder = k - b * (k // b)
    if remainder == 0:
        return []
    p_count = PARTITIONS_BY_B[b]
    offset = (partition + 0.5) / p_count
    ranks = [int(math.floor(((j + offset) * k) / remainder)) % k for j in range(remainder)]
    if len(set(ranks)) != remainder:
        raise AssertionError(f"nonunique rotating remainder ranks: K={k}, B={b}, partition={partition}")
    return sorted(ranks)


def list_overlap(left: dict[str, list[int]], right: dict[str, list[int]]) -> dict:
    intersection = union = 0
    for name in set(left) | set(right):
        a, b = set(left.get(name, [])), set(right.get(name, []))
        intersection += len(a & b)
        union += len(a | b)
    return {"intersection_tiles": intersection, "union_tiles": union,
            "jaccard": intersection / union if union else 1.0}


def candidate_coverage(model: str, model_data: dict, b: int) -> dict:
    selected_total = 0
    retained_total = 0
    per_band = 0
    rejected_shortages = []
    strata = []
    for name, selected_mask in model_data["reconstructed"].items():
        k = int(selected_mask.sum())
        rejected = selected_mask.numel() - k
        quota = k // b
        retained = b * quota
        selected_total += k
        retained_total += retained
        per_band += quota
        layer, projection, family = module_stratum(name)
        row = {"module": name, "layer": layer, "projection": projection, "module_family": family,
               "selected_tiles": k, "rejected_tiles": rejected, "quota_per_band": quota,
               "retained_selected_tiles": retained, "selected_remainder": k - retained,
               "required_rejected_tiles": retained, "rejected_available": rejected,
               "rejected_support_passed": rejected >= retained}
        strata.append(row)
        if rejected < retained:
            rejected_shortages.append(row)
    coverage = retained_total / selected_total
    criteria = {
        "selected_common_support_coverage_ge_0p75": coverage >= 0.75,
        "global_tiles_per_band_ge_100": per_band >= 100,
        "identical_selected_rejected_module_quotas": True,
        "rejected_support_sufficient": not rejected_shortages,
        "anchor_checks_passed": model_data["anchor_mismatches"] == 0 and model_data["upper_score_mismatches"] == 0,
        "rotating_remainder_rule_defined": True,
    }
    return {"model": model, "bands_per_side": b, "partitions": PARTITIONS_BY_B[b],
            "selected_tiles": selected_total, "retained_selected_tiles": retained_total,
            "selected_common_support_coverage": coverage, "tiles_per_global_band": per_band,
            "excluded_selected_tiles": selected_total - retained_total,
            "eligible_strata": sum(row["quota_per_band"] > 0 for row in strata),
            "total_strata": len(strata), "rejected_shortages": rejected_shortages,
            "criteria": criteria, "passed": all(criteria.values()), "strata": strata}


def boundary_definitions(model: str, model_data: dict, b: int) -> dict:
    p_count = PARTITIONS_BY_B[b]
    reference = model_data["reconstructed"]
    stats = model_data["stats"]
    ordered_selected = {name: ordered_indices(mask, stats[name]["kappa"], True) for name, mask in reference.items()}
    ordered_rejected = {name: ordered_indices(mask, stats[name]["kappa"], False) for name, mask in reference.items()}
    partitions = []
    for partition in range(p_count):
        selected_bands = [defaultdict(list) for _ in range(b)]
        rejected_bands = [defaultdict(list) for _ in range(b)]
        exclusions = []
        for name in reference:
            selected = ordered_selected[name]
            quota = len(selected) // b
            excluded_ranks = rotating_exclusion_ranks(len(selected), b, partition)
            excluded_set = set(excluded_ranks)
            retained = [index for rank, index in enumerate(selected) if rank not in excluded_set]
            if len(retained) != quota * b:
                raise AssertionError(f"{model}/{name}: retained count mismatch")
            for rank in excluded_ranks:
                index = selected[rank]
                width = int(reference[name].shape[1])
                exclusions.append({"module": name, "rank_strong_to_weak": rank,
                                   "flat_tile_index": index, "tile_row": index // width,
                                   "tile_col": index % width,
                                   "kappa": json_number(stats[name]["kappa"][index])})
            near_rejected = ordered_rejected[name][:quota * b]
            if len(near_rejected) != quota * b:
                raise AssertionError(f"{model}/{name}: rejected common support shortage")
            for band in range(b):
                selected_bands[band][name] = retained[band * quota:(band + 1) * quota]
                rejected_bands[band][name] = near_rejected[band * quota:(band + 1) * quota]
        band_rows = []
        for side, groups in (("selected", selected_bands), ("rejected", rejected_bands)):
            for band, lists in enumerate(groups, 1):
                compact = {name: values for name, values in lists.items() if values}
                policy = f"boundary_part{partition + 1:02d}_{side}_band{band:02d}"
                band_rows.append({"side": side, "band": band, "policy": policy,
                                  "order": ("selected band01 strongest; band%02d weakest" % b) if side == "selected" else
                                           ("rejected band01 nearest below k=3; band%02d progressively lower" % b),
                                  "tiles": compact, "tile_set_sha256": tile_set_sha(compact),
                                  "score_summary": score_summary(compact, stats),
                                  "per_module_counts": {name: len(lists[name]) for name in reference}})
        partitions.append({"partition": partition + 1, "remainder_offset_fraction": (partition + 0.5) / p_count,
                           "remainder_rule": "even score-quantile positions with deterministic rotating partition offset",
                           "excluded_selected_remainders": exclusions, "bands": band_rows})

    overlaps = []
    for left in range(p_count):
        for right in range(left + 1, p_count):
            left_by = {(r["side"], r["band"]): r for r in partitions[left]["bands"]}
            right_by = {(r["side"], r["band"]): r for r in partitions[right]["bands"]}
            for key in sorted(left_by):
                overlaps.append({"left_partition": left + 1, "right_partition": right + 1,
                                 "side": key[0], "band": key[1],
                                 **list_overlap(left_by[key]["tiles"], right_by[key]["tiles"])})

    global_per_band = sum(len(v) for v in partitions[0]["bands"][0]["tiles"].values())
    for partition in partitions:
        rows = {(row["side"], row["band"]): row for row in partition["bands"]}
        for band in range(1, b + 1):
            selected = rows[("selected", band)]
            rejected = rows[("rejected", band)]
            if selected["score_summary"]["tiles"] != global_per_band or rejected["score_summary"]["tiles"] != global_per_band:
                raise AssertionError(f"{model}: global boundary band count mismatch")
            if selected["per_module_counts"] != rejected["per_module_counts"]:
                raise AssertionError(f"{model}: selected/rejected module quota mismatch")
    return {"model": model, "bands_per_side": b, "partitions_count": p_count,
            "tiles_per_global_band": global_per_band,
            "ranking": "kappa=min(-mean_CE/SE_CE,-mean_KL/SE_KL), descending within exact module",
            "partitions": partitions, "partition_overlap_jaccard": overlaps,
            "invariants": {"equal_per_module_quota_within_partition": True,
                           "selected_rejected_composition_identical": True,
                           "selected_rejected_disjoint": True,
                           "remainders_spread_across_score_quantiles": True}}


def rng_seed(*parts: object) -> int:
    return int(hashlib.sha256(":".join([str(SEED_ROOT), *map(str, parts)]).encode()).hexdigest()[:16], 16)


def corruption_definitions(model: str, model_data: dict) -> dict:
    reference, stats = model_data["reconstructed"], model_data["stats"]
    reservoir_by_module: dict[str, list[int]] = {}
    pool_by_module: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(CORRUPTION_POOLS)]
    removal_priority: dict[str, list[int]] = {}
    addition_priority: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(CORRUPTION_POOLS)]
    shortages = []
    eligible_selected = 0
    selected_total = 0
    strata = []
    for name, selected_mask in reference.items():
        selected = ordered_indices(selected_mask, stats[name]["kappa"], True)
        rejected = ordered_indices(selected_mask, stats[name]["kappa"], False)
        k = len(selected)
        selected_total += k
        layer, projection, family = module_stratum(name)
        if len(rejected) < CORRUPTION_POOLS * k:
            shortages.append({"module": name, "selected_tiles": k, "rejected_tiles": len(rejected),
                              "required_rejected_tiles": CORRUPTION_POOLS * k})
            continue
        eligible_selected += k
        reservoir = rejected[:CORRUPTION_POOLS * k]
        reservoir_by_module[name] = reservoir
        removal_priority[name] = stratified_priority(selected, stats[name]["kappa"], f"{model}:{name}:selected-removal")
        pool_means = []
        for pool in range(CORRUPTION_POOLS):
            values = reservoir[pool::CORRUPTION_POOLS]
            if len(values) != k:
                raise AssertionError(f"{model}/{name}: pool size mismatch")
            pool_by_module[pool][name] = values
            addition_priority[pool][name] = stratified_priority(values, stats[name]["kappa"],
                                                                 f"{model}:{name}:pool{pool + 1}:addition")
            pool_means.append(float(stats[name]["kappa"][torch.tensor(values)].mean()) if values else None)
        strata.append({"module": name, "layer": layer, "projection": projection, "module_family": family,
                       "selected_tiles": k, "rejected_tiles": len(rejected), "reservoir_tiles": len(reservoir),
                       "pool_tiles": [len(pool_by_module[p][name]) for p in range(CORRUPTION_POOLS)],
                       "pool_mean_kappa": pool_means})

    coverage = eligible_selected / selected_total
    pool_rows = []
    pool_sets = []
    for pool in range(CORRUPTION_POOLS):
        compact = {name: list(values) for name, values in pool_by_module[pool].items() if values}
        pool_sets.append(compact)
        pool_rows.append({"pool": pool + 1, "tiles": compact, "tile_set_sha256": tile_set_sha(compact),
                          "score_summary": score_summary(compact, stats),
                          "addition_priority": {name: list(values) for name, values in addition_priority[pool].items() if values}})
    pool_overlap = []
    for left in range(CORRUPTION_POOLS):
        for right in range(left + 1, CORRUPTION_POOLS):
            pool_overlap.append({"left_pool": left + 1, "right_pool": right + 1,
                                 **list_overlap(pool_sets[left], pool_sets[right])})
    pool_hashes = [row["tile_set_sha256"] for row in pool_rows]
    distinct_pool_sets = len(set(pool_hashes)) == CORRUPTION_POOLS
    disjoint_pool_sets = all(row["intersection_tiles"] == 0 for row in pool_overlap)

    maps = []
    planned_p1 = []
    for pool in range(CORRUPTION_POOLS):
        for p in CORRUPTION_LEVELS:
            removed, added = {}, {}
            for name in reference:
                k = int(reference[name].sum())
                target = round_half_up(p * k)
                if target:
                    removed[name] = removal_priority[name][:target]
                    added[name] = addition_priority[pool][name][:target]
                if len(removed.get(name, [])) != len(added.get(name, [])):
                    raise AssertionError(f"{model}/{name}: corruption quota mismatch")
            policy = f"corruption_near_pool{pool + 1:02d}_p{round_half_up(100 * p):03d}"
            row = {"policy": policy, "kind": "near_boundary", "pool": pool + 1, "target_p": p,
                   "removed": removed, "added": added,
                   "removed_tile_set_sha256": tile_set_sha(removed),
                   "added_tile_set_sha256": tile_set_sha(added),
                   "removed_tiles": sum(map(len, removed.values())),
                   "added_tiles": sum(map(len, added.values())),
                   "achieved_global_p": sum(map(len, removed.values())) / selected_total,
                   "per_module_target_rule": "floor(p*K_s+0.5); nested prefixes of fixed stratified priority"}
            maps.append(row)
            if p == 1.0:
                planned_p1.append({"pool": pool + 1, "policy": policy,
                                   "selected_set_sha256": tile_set_sha(added),
                                   "selected_tiles": row["added_tiles"]})

    random_controls = []
    reservoir_sets = {name: set(values) for name, values in reservoir_by_module.items()}
    for replicate in range(CORRUPTION_POOLS):
        removed, added = {}, {}
        overlap = 0
        for name, selected_mask in reference.items():
            k = int(selected_mask.sum())
            target = round_half_up(0.50 * k)
            if not target:
                continue
            candidates = torch.nonzero(~selected_mask.flatten(), as_tuple=False).flatten().numpy()
            generator = np.random.default_rng(rng_seed(model, name, "matched-random", replicate + 1))
            draw = sorted(int(v) for v in generator.choice(candidates, size=target, replace=False).tolist())
            removed[name] = removal_priority[name][:target]
            added[name] = draw
            overlap += len(set(draw) & reservoir_sets[name])
        policy = f"corruption_random_pool{replicate + 1:02d}_p050"
        random_controls.append({"policy": policy, "kind": "matched_random", "replicate": replicate + 1,
                                "target_p": 0.5, "removed": removed, "added": added,
                                "removed_tile_set_sha256": tile_set_sha(removed),
                                "added_tile_set_sha256": tile_set_sha(added),
                                "removed_tiles": sum(map(len, removed.values())),
                                "added_tiles": sum(map(len, added.values())),
                                "overlap_with_near_boundary_reservoir": overlap,
                                "rng": "numpy PCG64, SHA-256-derived seed per model/module/replicate"})

    p1_hashes = [row["selected_set_sha256"] for row in planned_p1]
    random_hashes = [row["added_tile_set_sha256"] for row in random_controls]
    invariants = {
        "corruption_eligible_coverage_ge_0p95": coverage >= 0.95,
        "reservoir_exactly_4K_per_eligible_stratum": not shortages,
        "four_pool_tile_sets_distinct": distinct_pool_sets,
        "four_pool_tile_sets_disjoint": disjoint_pool_sets,
        "four_p1_selected_sets_distinct": len(set(p1_hashes)) == CORRUPTION_POOLS,
        "four_random_control_sets_distinct": len(set(random_hashes)) == CORRUPTION_POOLS,
        "exact_module_quota_preserved": True,
        "partial_levels_nested": True,
        "selected_removal_score_stratified": True,
    }
    return {"model": model, "selected_tiles": selected_total, "eligible_selected_tiles": eligible_selected,
            "corruption_eligible_coverage": coverage, "shortages": shortages,
            "reservoir": {name: values for name, values in reservoir_by_module.items() if values},
            "reservoir_tile_set_sha256": tile_set_sha(reservoir_by_module),
            "selected_removal_priority": {name: values for name, values in removal_priority.items() if values},
            "strata": strata, "pools": pool_rows, "pool_pairwise_overlap": pool_overlap,
            "near_boundary_maps": maps, "matched_random_controls": random_controls,
            "planned_p1_maps": planned_p1, "invariants": invariants, "passed": all(invariants.values())}


def weighted_effect(nll_a: np.ndarray, nll_b: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, float]:
    delta = np.asarray(nll_a - nll_b, np.float64)
    estimate = float(delta.sum() / tokens.sum())
    return delta - estimate * tokens, estimate


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def joint_slope_noise(centered: np.ndarray, tokens: np.ndarray, x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(POWER_REPLICATES, np.float64)
    clusters = len(tokens)
    for start in range(0, POWER_REPLICATES, 250):
        size = min(250, POWER_REPLICATES - start)
        idx = rng.integers(0, clusters, size=(size, clusters))
        denominator = tokens[idx].sum(axis=1)
        y = np.empty((size, centered.shape[0]), np.float64)
        for row in range(centered.shape[0]):
            y[:, row] = centered[row][idx].sum(axis=1) / denominator
        for offset in range(size):
            out[start + offset] = fit_slope(x, y[offset])
    return out


def paired_mean_noise(centered: np.ndarray, tokens: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tokens), size=(POWER_REPLICATES, len(tokens)))
    return centered[idx].sum(axis=1) / tokens[idx].sum(axis=1)


def power_from_noise(noise: np.ndarray, alpha: float) -> dict:
    noise = np.asarray(noise, np.float64)
    noise = noise - noise.mean()
    critical = float(np.quantile(noise, 1 - alpha))
    powers = {f"delta_nll_{value:.4f}": float(np.mean(noise + value > critical)) for value in SESOI}
    mde80 = critical - float(np.quantile(noise, 0.20))
    mde90 = critical - float(np.quantile(noise, 0.10))
    status = "inferential" if powers["delta_nll_0.0010"] >= 0.80 else (
        "limited_inference" if powers["delta_nll_0.0025"] >= 0.80 else "descriptive")
    return {"simulations": POWER_REPLICATES, "holm_conservative_alpha": alpha,
            "null_critical_value": critical, "null_noise_sd": float(noise.std(ddof=1)),
            "power": powers, "mde_80pct": float(mde80), "mde_90pct": float(mde90),
            "status": status}


def power_panel(model: str, domain: str) -> dict:
    path = FOLLOWUP_ROOT / f"arrays/{model}_{domain}_paired_cluster_nll.npz"
    with np.load(path) as payload:
        policies = [str(v) for v in payload["policies"]]
        pos = {name: i for i, name in enumerate(policies)}
        nll = np.asarray(payload["cluster_nll_sum"], np.float64)
        tokens = np.asarray(payload["cluster_tokens"], np.float64)
        cluster_ids = [str(v) for v in payload["cluster_ids"]]
    family_size = 6 if model == "mistral7b" else 12
    alpha = 0.05 / family_size
    baseline = nll[pos["four_over_six"]]
    full = nll[pos["full"]]

    group = np.stack([nll[pos[f"dose_group_only_bin{i:02d}"]] for i in range(1, 9)])
    group_centered = np.empty_like(group)
    for i in range(8):
        group_centered[i], _ = weighted_effect(group[i], baseline, tokens)
    boundary_slope_noise = -joint_slope_noise(group_centered, tokens, np.linspace(0, 1, 8),
                                               rng_seed("power", model, domain, "boundary-slope"))

    selrej_centered, _ = weighted_effect(group[0], group[-1], tokens)
    # Expected selected-minus-rejected is negative, so reverse to positive support magnitude.
    selected_rejected_noise = -paired_mean_noise(selrej_centered, tokens,
                                                  rng_seed("power", model, domain, "selected-rejected"))
    weak_centered, _ = weighted_effect(group[-1], group[-2], tokens)
    weak_near_noise = -paired_mean_noise(weak_centered, tokens,
                                         rng_seed("power", model, domain, "weak-near"))

    pseudo_names = ["full", "dose_full_minus_bin01", "dose_full_minus_bin02",
                    "dose_full_minus_bin03", "dose_full_minus_bin04", "objective_random_matched_k"]
    pseudo = np.stack([nll[pos[name]] for name in pseudo_names])
    pseudo_centered = np.empty_like(pseudo)
    for i, row in enumerate(pseudo):
        pseudo_centered[i], _ = weighted_effect(row, full, tokens)
    corruption_slope_noise = joint_slope_noise(pseudo_centered, tokens, np.asarray(P_VALUES),
                                                rng_seed("power", model, domain, "corruption-slope"))

    candidates = [f"dose_full_minus_bin{i:02d}" for i in range(1, 9)] + [
        "objective_ce_matched_k", "objective_kl_matched_k", "objective_random_matched_k"]
    candidate_noise = []
    for name in candidates:
        centered, _ = weighted_effect(nll[pos[name]], full, tokens)
        noise = paired_mean_noise(centered, tokens, rng_seed("power", model, domain, "candidate", name))
        candidate_noise.append((float(noise.std(ddof=1)), name, noise))
    _, conservative_name, conservative_noise = max(candidate_noise, key=lambda row: row[0])

    endpoints = {
        "boundary_trend_slope_end_to_end": power_from_noise(boundary_slope_noise, alpha),
        "selected_minus_rejected_aggregate": power_from_noise(selected_rejected_noise, alpha),
        "weakest_selected_minus_nearest_rejected": power_from_noise(weak_near_noise, alpha),
        "corruption_slope_end_to_end": power_from_noise(corruption_slope_noise, alpha),
        "p050_minus_p0": power_from_noise(conservative_noise, alpha),
        "p100_minus_p0": power_from_noise(conservative_noise, alpha),
    }
    endpoints["p050_minus_p0"]["variance_template_policy"] = conservative_name
    endpoints["p100_minus_p0"]["variance_template_policy"] = conservative_name
    return {"model": model, "corpus": domain, "source_array": str(path), "source_sha256": sha256_file(path),
            "clusters": len(tokens), "cluster_ids_sha256": canonical_sha(cluster_ids),
            "tokens": int(tokens.sum()), "holm_family_size": family_size, "endpoints": endpoints,
            "template_note": "Existing paired cluster arrays supply centered covariance/noise only; observed prior effects are removed before injecting each SESOI."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)

    model_data = {model: load_model_scores(model) for model in MODELS}
    critical_models = {}
    for model, data in model_data.items():
        selected = sum(int(mask.sum()) for mask in data["reconstructed"].values())
        total = sum(mask.numel() for mask in data["reconstructed"].values())
        critical_models[model] = {
            "primary_map": str(data["map_path"]), "primary_map_sha256": data["map_sha256"],
            "moments": str(data["moments_path"]), "moments_sha256": data["moments_sha256"],
            "selected_tiles": selected, "total_tiles": total, "modules": len(data["reconstructed"]),
            "kappa_gt_3_anchor_tile_mismatches": data["anchor_mismatches"],
            "upper_score_vs_kappa_tile_mismatches": data["upper_score_mismatches"],
            "zero_se_tile_count": len(data["zero_se_tiles"]), "zero_se_tiles": data["zero_se_tiles"],
            "zero_se_rule": {"SE=0,mean<0": "+inf", "SE=0,mean>=0": "-inf"},
            "passed": data["anchor_mismatches"] == 0 and data["upper_score_mismatches"] == 0,
        }
    critical = {"schema": "mixfp4-boundary-critical-k-summary/v1", "k": K_PRIMARY,
                "definition": "min(-mean_CE/SE_CE,-mean_KL/SE_KL)", "models": critical_models,
                "passed": all(row["passed"] for row in critical_models.values()), "outcome_blind": True}
    atomic_json(root / "CRITICAL_K_SUMMARY.json", critical)

    candidates = {str(b): {model: candidate_coverage(model, model_data[model], b) for model in MODELS} for b in CANDIDATE_B}
    selected_b = next((b for b in CANDIDATE_B if all(candidates[str(b)][m]["passed"] for m in MODELS)), None)
    coverage_passed = selected_b is not None and critical["passed"]
    coverage = {"schema": "mixfp4-boundary-coverage-gate/v1", "outcome_blind": True,
                "candidate_order": list(CANDIDATE_B), "selection_rule": "first/largest shared B passing every hard gate",
                "selected_B": selected_b, "partitions": PARTITIONS_BY_B.get(selected_b) if selected_b else None,
                "candidates": candidates, "passed": coverage_passed,
                "failures": [] if coverage_passed else ["no shared B passed or critical-k anchor failed"]}
    atomic_json(root / "COVERAGE_GATE.json", coverage)
    if not coverage_passed:
        raise SystemExit("coverage gate failed; GPU work prohibited")

    bands = {model: boundary_definitions(model, model_data[model], selected_b) for model in MODELS}
    band_payload = {"schema": "mixfp4-boundary-band-definitions/v1", "outcome_blind": True,
                    "selected_B": selected_b, "partitions": PARTITIONS_BY_B[selected_b], "models": bands}
    atomic_json(root / "BAND_DEFINITIONS.json", band_payload)

    corruptions = {model: corruption_definitions(model, model_data[model]) for model in MODELS}
    corruption_payload = {"schema": "mixfp4-corruption-pool-definitions/v1", "outcome_blind": True,
                          "p_levels": [0.0, *CORRUPTION_LEVELS], "pools": CORRUPTION_POOLS,
                          "models": corruptions}
    atomic_json(root / "CORRUPTION_POOL_DEFINITIONS.json", corruption_payload)
    uniqueness = {
        "schema": "mixfp4-corruption-uniqueness-gate/v1", "outcome_blind": True,
        "minimum_eligible_selected_coverage": 0.95,
        "models": {model: {"selected_tiles": row["selected_tiles"],
                            "eligible_selected_tiles": row["eligible_selected_tiles"],
                            "coverage": row["corruption_eligible_coverage"], "shortages": row["shortages"],
                            "pool_tile_set_sha256": [p["tile_set_sha256"] for p in row["pools"]],
                            "p1_selected_set_sha256": [p["selected_set_sha256"] for p in row["planned_p1_maps"]],
                            "invariants": row["invariants"], "passed": row["passed"]}
                   for model, row in corruptions.items()},
        "passed": all(row["passed"] for row in corruptions.values()),
    }
    uniqueness["failures"] = [] if uniqueness["passed"] else ["one or more model pool/coverage invariants failed"]
    atomic_json(root / "CORRUPTION_UNIQUENESS_GATE.json", uniqueness)
    if not uniqueness["passed"]:
        raise SystemExit("corruption uniqueness gate failed; GPU work prohibited")

    power_models = {model: {domain: power_panel(model, domain) for domain in DOMAINS} for model in MODELS}
    large_ok = {}
    for model in MODELS:
        large_ok[model] = any(
            power_models[model][domain]["endpoints"][endpoint]["power"]["delta_nll_0.0025"] >= 0.80
            for domain in DOMAINS
            for endpoint in ("selected_minus_rejected_aggregate", "p100_minus_p0")
        )
    hard_stop = not any(large_ok.values())
    power = {"schema": "mixfp4-boundary-corruption-power-analysis/v1", "outcome_blind": True,
             "simulation_replicates": POWER_REPLICATES, "seed_root": SEED_ROOT,
             "sesoi_delta_nll": list(SESOI), "holm_method": "conservative first-step alpha=0.05/family_size",
             "models": power_models, "large_primary_power_at_0p0025_by_model": large_ok,
             "hard_stop_triggered": hard_stop, "passed": not hard_stop,
             "interpretation": "Power status controls claim strength only; all valid required arms remain mandatory."}
    atomic_json(root / "POWER_ANALYSIS.json", power)
    promotion = {"schema": "mixfp4-boundary-corruption-endpoint-promotion/v1", "outcome_blind": True,
                 "rules": {"inferential": ">=80% power at delta-NLL 0.0010",
                           "limited_inference": "<80% at 0.0010 and >=80% at 0.0025",
                           "descriptive": "<80% at 0.0025"},
                 "models": {model: {domain: {name: row["status"] for name, row in power_models[model][domain]["endpoints"].items()}
                                     for domain in DOMAINS} for model in MODELS},
                 "arms_selected_by_power": False, "passed": not hard_stop}
    atomic_json(root / "ENDPOINT_PROMOTION.json", promotion)
    if hard_stop:
        raise SystemExit("paired power/MDE gate failed; GPU work prohibited")

    launch_path = Path(os.environ.get("CAMPAIGN_RUN_DIR", "")) / "launch_record.json"
    if launch_path.is_file():
        launch = json.loads(launch_path.read_text())
        runtime.atomic_json(Path(os.environ["CAMPAIGN_RUN_DIR"]) / "job_result.json", {
            "protocol_id": "boundary-corruption-pre-freeze", "protocol_freeze_sha256": "pre-freeze",
            "source": {"model_id": None, "model_revision": None, "tokenizer_revision": None,
                       "model_class": None, "module_manifest_sha256": None,
                       "source_manifest_sha256": launch["source_manifest_sha256"]},
            "environment": runtime.environment(),
            "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                     "token_hashes": {}, "overlap_audit": None}, "policies": [],
            "results": {"raw_outputs": [str(root / name) for name in (
                "CRITICAL_K_SUMMARY.json", "COVERAGE_GATE.json", "BAND_DEFINITIONS.json",
                "CORRUPTION_POOL_DEFINITIONS.json", "CORRUPTION_UNIQUENESS_GATE.json",
                "POWER_ANALYSIS.json", "ENDPOINT_PROMOTION.json")],
                "summary": {"selected_B": selected_b, "partitions": PARTITIONS_BY_B[selected_b],
                            "coverage_passed": True, "corruption_uniqueness_passed": True,
                            "power_gate_passed": True}, "uncertainty": {"power_simulations": POWER_REPLICATES},
                "attempted_endpoints": ["coverage", "corruption_uniqueness", "paired_power"],
                "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"selected_B": selected_b, "partitions": PARTITIONS_BY_B[selected_b],
                      "coverage": {m: candidates[str(selected_b)][m]["selected_common_support_coverage"] for m in MODELS},
                      "tiles_per_band": {m: candidates[str(selected_b)][m]["tiles_per_global_band"] for m in MODELS},
                      "corruption_coverage": {m: corruptions[m]["corruption_eligible_coverage"] for m in MODELS},
                      "power_passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
