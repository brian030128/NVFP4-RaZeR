"""Outcome-blind map construction for the frozen N16K64 follow-up.

All selections are deterministic functions of the verified seed0 direct-N16 score
moments and the frozen k=3 conjunction map.  No evaluation output is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from campaign import mapio, runtime


_workspace = Path(os.environ.get("MIXFP4_WORKSPACE_ROOT", Path.cwd()))
PRIMARY = Path(os.environ.get(
    "MIXFP4_PRIMARY_CAMPAIGN",
    _workspace / "research_runs/mixfp4_n16k64_full_validation_20260911T065444Z",
))
MECHANISM = Path(os.environ.get(
    "MIXFP4_MECHANISM_CAMPAIGN",
    _workspace / "research_runs/mixfp4_mechanism_24h_20260916T205501Z",
))
MODEL_RUN = {
    "llama8b": ("V30_calib_llama8b_seed0_attempt1", "llama8b"),
    "qwen4b": ("V30_calib_qwen4b_seed0_attempt1", "qwen4b"),
    "mistral7b": ("V61_calib_mistral7b_seed0_attempt2", "mistral7b"),
}
K = 3.0
DOSE_BINS = 8
VETO_BINS = 5
VETO_FRACTION = 0.25
RANDOM_SEED = 20260917


def file_sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module_stratum(name: str) -> tuple[int, str, str]:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        raise ValueError(f"cannot parse layer: {name}")
    projection = name.rsplit(".", 1)[-1]
    family = "attention" if ".self_attn." in name else "mlp" if ".mlp." in name else None
    if family is None:
        raise ValueError(f"out-of-scope module: {name}")
    return int(match.group(1)), projection, family


def mean_se(st: dict, objective: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(st["n"])
    total = st[f"{objective}_sum"].double()
    sq = st[f"{objective}_sq"].double()
    mean = total / n
    variance = ((sq - total * total / n) / (n - 1)).clamp_min(0)
    return mean, (variance / n).sqrt()


def empty_like(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(mask) for name, mask in reference.items()}


def mask_union(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: a[name] | b[name] for name in a}


def keyed_order(indices: list[int], label: str) -> list[int]:
    return sorted((int(i) for i in indices), key=lambda i: hashlib.sha256(f"{label}:{i}".encode()).digest())


def score_summary(mask: dict[str, torch.Tensor], stats: dict[str, dict[str, torch.Tensor]]) -> dict:
    values = {key: [] for key in ("ce_mean", "kl_mean", "uce", "ukl", "combined_margin")}
    for name, grid in mask.items():
        chosen = grid.flatten()
        if not chosen.any():
            continue
        for key in values:
            values[key].append(stats[name][key][chosen])

    def total(key: str) -> float:
        return float(torch.cat(values[key]).sum()) if values[key] else 0.0

    def average(key: str) -> float | None:
        return float(torch.cat(values[key]).mean()) if values[key] else None

    return {
        "tiles": sum(int(grid.sum()) for grid in mask.values()),
        "sum_ce_mean": total("ce_mean"),
        "sum_kl_mean": total("kl_mean"),
        "mean_ce_mean": average("ce_mean"),
        "mean_kl_mean": average("kl_mean"),
        "mean_u_ce": average("uce"),
        "mean_u_kl": average("ukl"),
        "mean_combined_election_margin": average("combined_margin"),
    }


def overlap_table(maps: dict[str, dict[str, torch.Tensor]]) -> list[dict]:
    names = list(maps)
    out = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intersection = sum(int((maps[left][n] & maps[right][n]).sum()) for n in maps[left])
            union = sum(int((maps[left][n] | maps[right][n]).sum()) for n in maps[left])
            out.append({"left": left, "right": right, "intersection_tiles": intersection,
                        "union_tiles": union, "jaccard": intersection / union if union else 1.0})
    return out


def dose_maps(model: str, full: dict[str, torch.Tensor], stats: dict[str, dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    bins = [empty_like(full) for _ in range(DOSE_BINS)]
    strata, exclusions = [], []
    for name in full:
        selected = torch.nonzero(full[name].flatten(), as_tuple=False).flatten()
        margin = stats[name]["combined_margin"][selected]
        order = torch.argsort(margin, descending=True, stable=True)
        ordered = selected[order]
        per_bin = int(ordered.numel()) // DOSE_BINS
        retained_n = per_bin * DOSE_BINS
        retained = ordered[:retained_n]
        remainder = ordered[retained_n:]
        layer, projection, family = module_stratum(name)
        row = {
            "module": name, "layer": layer, "projection": projection, "module_family": family,
            "selected_tiles": int(selected.numel()), "retained_tiles": retained_n,
            "excluded_remainder_tiles": int(remainder.numel()), "tiles_per_bin": per_bin,
            "remainder_rule": "stable strongest-to-weakest ordering; retain first 8*floor(n/8); exclude weakest remainder",
        }
        strata.append(row)
        width = int(full[name].shape[1])
        for index in remainder.tolist():
            i = int(index)
            exclusions.append({
                "module": name, "layer": layer, "projection": projection,
                "flat_tile_index": i, "tile_row": i // width, "tile_col": i % width,
                "combined_election_margin": float(stats[name]["combined_margin"][i]),
                "ce_mean": float(stats[name]["ce_mean"][i]), "kl_mean": float(stats[name]["kl_mean"][i]),
                "reason": "per-stratum remainder excluded to make eight equal ordered bins",
            })
        for b in range(DOSE_BINS):
            idx = retained[b * per_bin:(b + 1) * per_bin]
            if idx.numel():
                bins[b][name].view(-1)[idx] = True

    for name in full:
        counts = [int(mask[name].sum()) for mask in bins]
        if len(set(counts)) != 1:
            raise AssertionError(f"{model}: unequal per-stratum dose counts for {name}: {counts}")
        for i in range(DOSE_BINS):
            for j in range(i + 1, DOSE_BINS):
                if (bins[i][name] & bins[j][name]).any():
                    raise AssertionError(f"{model}: dose bins overlap in {name}")
    retained_union = empty_like(full)
    for name in full:
        for b in bins:
            retained_union[name] |= b[name]
        if (retained_union[name] & ~full[name]).any():
            raise AssertionError(f"{model}: dose union leaves conjunction in {name}")
    full_count = sum(int(x.sum()) for x in full.values())
    retained_count = sum(int(x.sum()) for x in retained_union.values())
    if full_count - retained_count != len(exclusions):
        raise AssertionError(f"{model}: dose remainder accounting mismatch")

    maps = {}
    definitions = []
    for b, mask in enumerate(bins, 1):
        label = f"bin{b:02d}"
        group = f"dose_group_only_{label}"
        minus = f"dose_full_minus_{label}"
        maps[group] = mask
        maps[minus] = {name: full[name] & ~mask[name] for name in full}
        definitions.append({
            "bin": b, "order": "1=strongest, 8=weakest among retained tiles",
            "group_only_policy": group, "full_minus_policy": minus,
            "score_summary": score_summary(mask, stats),
            "per_stratum_counts": {name: int(mask[name].sum()) for name in full},
        })
    detail = {
        "model": model, "bins": DOSE_BINS, "ranking_variable": "-max(U_CE,U_KL)",
        "stable_tie_break": "ascending flat tile index within module",
        "strata": strata, "excluded_remainder_tiles": exclusions,
        "selected_tiles_before_remainder": full_count, "retained_union_tiles": retained_count,
        "excluded_remainder_count": len(exclusions), "definitions": definitions,
        "invariants": {"equal_count_in_every_stratum": True, "bins_disjoint": True,
                       "retained_union_plus_remainders_equals_conjunction": True},
    }
    return maps, detail


def objective_maps(model: str, full: dict[str, torch.Tensor], stats: dict[str, dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    ce_natural, kl_natural = empty_like(full), empty_like(full)
    ce_matched, kl_matched, random_matched = empty_like(full), empty_like(full), empty_like(full)
    strata = []
    for name in full:
        uce, ukl = stats[name]["uce"], stats[name]["ukl"]
        ce_pass = uce < 0
        kl_pass = ukl < 0
        union_pass = ce_pass | kl_pass
        ce_natural[name] = ce_pass.reshape(full[name].shape)
        kl_natural[name] = kl_pass.reshape(full[name].shape)
        quota = int(full[name].sum())
        ce_idx = torch.nonzero(ce_pass, as_tuple=False).flatten()
        kl_idx = torch.nonzero(kl_pass, as_tuple=False).flatten()
        union_idx = torch.nonzero(union_pass, as_tuple=False).flatten()
        if min(ce_idx.numel(), kl_idx.numel(), union_idx.numel()) < quota:
            raise AssertionError(f"{model}: matched quota infeasible in {name}")
        ce_order = ce_idx[torch.argsort(uce[ce_idx], descending=False, stable=True)][:quota]
        kl_order = kl_idx[torch.argsort(ukl[kl_idx], descending=False, stable=True)][:quota]
        random_order = keyed_order(union_idx.tolist(), f"objective-random:{RANDOM_SEED}:{model}:{name}")[:quota]
        if quota:
            ce_matched[name].view(-1)[ce_order] = True
            kl_matched[name].view(-1)[kl_order] = True
            random_matched[name].view(-1)[torch.tensor(random_order, dtype=torch.long)] = True
        layer, projection, family = module_stratum(name)
        strata.append({"module": name, "layer": layer, "projection": projection, "module_family": family,
                       "conjunction_quota": quota, "ce_pass_support": int(ce_idx.numel()),
                       "kl_pass_support": int(kl_idx.numel()), "union_pass_support": int(union_idx.numel()),
                       "support_reduction": 0})
    for name in full:
        quota = int(full[name].sum())
        if not all(int(m[name].sum()) == quota for m in (ce_matched, kl_matched, random_matched)):
            raise AssertionError(f"{model}: objective matched-K mismatch in {name}")
    maps = {
        "objective_ce_natural": ce_natural,
        "objective_kl_natural": kl_natural,
        "objective_conjunction": {name: full[name].clone() for name in full},
        "objective_ce_matched_k": ce_matched,
        "objective_kl_matched_k": kl_matched,
        "objective_random_matched_k": random_matched,
    }
    detail = {
        "model": model, "k": K, "random_seed": RANDOM_SEED,
        "natural_threshold_interpretation": "each rule uses its own selected count; not budget matched",
        "matched_k_interpretation": "each module quota equals the frozen conjunction count",
        "strata": strata,
        "selected_counts": {policy: sum(int(x.sum()) for x in mask.values()) for policy, mask in maps.items()},
        "score_summaries": {policy: score_summary(mask, stats) for policy, mask in maps.items()},
        "pairwise_overlap": overlap_table(maps),
        "invariants": {"matched_k_per_stratum_exact": True, "support_reductions": 0,
                       "conjunction_reproduces_primary_anchor": True},
    }
    return maps, detail


def mistral_veto_maps(full: dict[str, torch.Tensor], stats: dict[str, dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    model = "mistral7b"
    locs, margins = [], []
    for name in full:
        cls = (stats[name]["ukl"] < 0) & (stats[name]["uce"] >= 0)
        idx = torch.nonzero(cls, as_tuple=False).flatten()
        vals = -stats[name]["ukl"][idx]
        for i, value in zip(idx.tolist(), vals.tolist()):
            locs.append((name, int(i)))
            margins.append(float(value))
    values = np.asarray(margins, np.float64)
    edges = np.quantile(values, np.linspace(0, 1, VETO_BINS + 1), method="linear")
    bins = np.searchsorted(edges[1:-1], values, side="right")
    target = min(int(math.floor(sum(int(x.sum()) for x in full.values()) * VETO_FRACTION)), len(locs))
    ordered = sorted(range(len(locs)), key=lambda j: (-margins[j], locs[j][0], locs[j][1]))[:target]
    proposed = set(ordered)
    all_by_cell, proposed_by_cell = {}, {}
    for j, ((name, _), bin_id) in enumerate(zip(locs, bins.tolist())):
        cell = (name, int(bin_id))
        all_by_cell.setdefault(cell, []).append(j)
        if j in proposed:
            proposed_by_cell.setdefault(cell, []).append(j)
    actual, random = empty_like(full), empty_like(full)
    cells, reductions = [], []
    for cell in sorted(all_by_cell):
        support = all_by_cell[cell]
        prop = sorted(proposed_by_cell.get(cell, []), key=lambda j: (-margins[j], locs[j][1]))
        keep_n = min(len(prop), len(support) // 2)
        kept = prop[:keep_n]
        kept_set = set(kept)
        remaining = [j for j in support if j not in kept_set]
        controls = keyed_order(remaining, f"veto-random:{RANDOM_SEED}:{model}:ce_vetoed_kl_approved:{cell[0]}:bin{cell[1]}")[:keep_n]
        for j in kept:
            name, index = locs[j]
            actual[name].view(-1)[index] = True
        for j in controls:
            name, index = locs[j]
            random[name].view(-1)[index] = True
        row = {"module": cell[0], "approving_kl_margin_bin": cell[1], "class_support": len(support),
               "proposed_actual": len(prop), "kept_actual": keep_n, "matched_random": len(controls)}
        cells.append(row)
        if keep_n < len(prop):
            reductions.append({**row, "reason": "disjoint exact-cell common support"})
    for name in full:
        if int(actual[name].sum()) != int(random[name].sum()):
            raise AssertionError(f"Mistral veto module mismatch: {name}")
        if (actual[name] & random[name]).any():
            raise AssertionError(f"Mistral veto actual/random overlap: {name}")
    maps = {
        "veto_full_plus_ce_vetoed_kl_approved": mask_union(full, actual),
        "veto_full_plus_ce_vetoed_kl_matched_random": mask_union(full, random),
    }

    # This class was frozen but not evaluated in the completed mechanism campaign.
    prior_names = {
        "veto_full_plus_ce_vetoed_kl_approved": "full_plus_ce_vetoed_kl_approved",
        "veto_full_plus_ce_vetoed_kl_matched_random": "full_plus_ce_vetoed_matched_random",
    }
    prior = {}
    for policy, old_policy in prior_names.items():
        old_path = MECHANISM / "runs/V10_derive_maps_attempt2/derived_maps/mistral7b" / f"mistral7b_{old_policy}.mixfp4map"
        old_header, old_masks, old_sha = mapio.read_map(old_path)
        mismatch = sum(int((maps[policy][name] != old_masks[name]).sum()) for name in full)
        if mismatch:
            raise AssertionError(f"Mistral frozen veto reproduction differs at {mismatch} tiles for {policy}")
        prior[policy] = {"path": str(old_path), "sha256": old_sha, "mask_tile_mismatches": mismatch,
                         "selected_tiles": old_header["totals"]["selected_tiles"]}
    detail = {
        "model": model, "class": "ce_vetoed_kl_approved", "approving_objective": "kl",
        "class_tiles": len(locs), "target_before_common_support": target,
        "actual_addback_tiles": sum(int(x.sum()) for x in actual.values()),
        "matched_random_addback_tiles": sum(int(x.sum()) for x in random.values()),
        "approving_margin_bins": VETO_BINS, "global_margin_bin_edges": edges.tolist(),
        "matching_cells": cells, "support_reductions": reductions,
        "prior_frozen_map_reproduction": prior,
        "invariants": {"module_and_margin_bin_counts_exact": True, "actual_random_disjoint": True,
                       "prior_frozen_masks_reproduced_exactly": True},
    }
    return maps, detail


def write_maps(model: str, maps: dict, stats: dict, base_header: dict, shapes: dict,
               protocol: dict, protocol_sha: str, source_manifest: str, out_dir: Path) -> list[dict]:
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for policy, masks in maps.items():
        family = "dose_response" if policy.startswith("dose_") else "objective_ablation" if policy.startswith("objective_") else "mistral_veto_completion"
        header = mapio.build_header(
            protocol_id=protocol["protocol_id"],
            policy={"name": policy, "family": family, "frozen_k": 3, "seed": RANDOM_SEED},
            model=base_header["model"], type_block=(16, 64), masks=masks, weight_shapes=shapes,
            source_manifest_sha256=source_manifest,
            calibration_manifest_sha256=base_header["calibration_manifest_sha256"],
        )
        path = model_dir / f"{model}_{policy}.mixfp4map"
        digest, written = mapio.write_map(path, header, masks, provenance={
            "protocol_sha256": protocol_sha,
            "source_primary_map_sha256": base_header["_source_sha256"],
            "source_moments_sha256": base_header["_moments_sha256"],
            "anchor_tile_mismatches": 0,
            "definition": protocol["map_definitions"][family],
        })
        entries.append({
            "model": model, "policy": policy, "family": family, "path": written,
            "sha256": digest, "mask_payload_sha256": mapio.payload_sha256(masks),
            "type_block": [16, 64], "selected_tiles": header["totals"]["selected_tiles"],
            "total_tiles": header["totals"]["total_tiles"], "score_summary": score_summary(masks, stats),
        })
    return entries


def derive_model(model: str, protocol: dict, protocol_sha: str, source_manifest: str, out_dir: Path) -> tuple[list[dict], dict, dict, dict]:
    run_name, stem = MODEL_RUN[model]
    source_run = PRIMARY / "runs" / run_name
    moments_path = source_run / "calibration/moments/moments_full.pt"
    base_path = source_run / f"maps/{stem}_seed0_n16_k3.mixfp4map"
    moments = torch.load(moments_path, map_location="cpu", weights_only=True, mmap=True)
    base_header, base_masks, base_sha = mapio.read_map(base_path)
    names = moments["names"]
    if names != list(base_masks):
        raise ValueError(f"{model}: moment/map order mismatch")
    stats, reconstructed = {}, {}
    for name in names:
        ce_mean, ce_se = mean_se(moments["n16"][name], "ce")
        kl_mean, kl_se = mean_se(moments["n16"][name], "kl")
        uce, ukl = ce_mean + K * ce_se, kl_mean + K * kl_se
        reconstructed[name] = ((uce < 0) & (ukl < 0)).reshape(base_masks[name].shape)
        stats[name] = {"ce_mean": ce_mean, "ce_se": ce_se, "kl_mean": kl_mean, "kl_se": kl_se,
                       "uce": uce, "ukl": ukl, "combined_margin": -torch.maximum(uce, ukl)}
    mismatch = sum(int((reconstructed[name] != base_masks[name]).sum()) for name in names)
    if mismatch:
        raise AssertionError(f"{model}: k=3 anchor differs at {mismatch} tiles")
    base_header["_source_sha256"] = base_sha
    base_header["_moments_sha256"] = file_sha(moments_path)
    shapes = {name: tuple(moments["shapes"][name]) for name in names}
    dose, dose_detail = dose_maps(model, reconstructed, stats)
    objective, objective_detail = objective_maps(model, reconstructed, stats)
    all_maps = {**dose, **objective}
    veto_detail = None
    if model == "mistral7b":
        veto, veto_detail = mistral_veto_maps(reconstructed, stats)
        all_maps.update(veto)
    entries = write_maps(model, all_maps, stats, base_header, shapes, protocol, protocol_sha, source_manifest, out_dir)
    anchor = {"primary_map": str(base_path), "primary_map_sha256": base_sha,
              "moments": str(moments_path), "moments_sha256": base_header["_moments_sha256"],
              "tile_mismatches": mismatch, "selected_tiles": sum(int(x.sum()) for x in reconstructed.values()),
              "total_tiles": sum(x.numel() for x in reconstructed.values()), "passed": mismatch == 0}
    del moments
    return entries, anchor, dose_detail, {"objective": objective_detail, "veto": veto_detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--models", default="llama8b,qwen4b,mistral7b")
    args = ap.parse_args()
    protocol_path = Path(args.protocol)
    protocol_sha = file_sha(protocol_path)
    if protocol_sha != args.protocol_sha256:
        raise SystemExit("follow-up protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text())
    launch = json.loads((runtime.run_dir / "launch_record.json").read_text())
    out = runtime.out_dir("derived_maps")
    entries, anchors, bins, definitions = [], {}, {}, {}
    for model in args.models.split(","):
        model_entries, anchor, dose, rest = derive_model(model, protocol, protocol_sha, launch["source_manifest_sha256"], out)
        entries.extend(model_entries)
        anchors[model], bins[model], definitions[model] = anchor, dose, rest
    runtime.atomic_json(out / "map_manifest.json", entries)
    runtime.atomic_json(out / "bin_definitions.json", {
        "schema": "mixfp4-followup-bin-definitions/v1", "protocol_sha256": protocol_sha,
        "models": bins, "outcome_blind": True,
    })
    runtime.atomic_json(out / "selection_definitions.json", {
        "schema": "mixfp4-followup-selection-definitions/v1", "protocol_sha256": protocol_sha,
        "anchors": anchors, "models": definitions, "outcome_blind": True,
    })
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": protocol["protocol_id"], "protocol_freeze_sha256": protocol_sha,
        "source": {"model_id": None, "model_revision": None, "tokenizer_revision": None, "model_class": None,
                   "module_manifest_sha256": None, "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None},
        "policies": [{"name": e["policy"], "weight_format": "FourOverSix/E0M3 tile mix",
                      "activation_format": "four_over_six_rows", "scale_block": 16,
                      "type_block": e["type_block"], "map_path": e["path"], "map_sha256": e["sha256"],
                      "selected_tiles": e["selected_tiles"], "total_tiles": e["total_tiles"],
                      "map_reloaded_for_evaluation": False} for e in entries],
        "results": {"raw_outputs": [str(p) for p in sorted(out.rglob("*")) if p.is_file()],
                    "summary": {"models": list(anchors), "maps": len(entries),
                                "anchor_reproductions_passed": all(a["passed"] for a in anchors.values()),
                                "mistral_veto_prior_mask_reproduction_passed": definitions["mistral7b"]["veto"]["invariants"]["prior_frozen_masks_reproduced_exactly"]},
                    "uncertainty": {}, "attempted_endpoints": ["map_derivation"], "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps({"maps": len(entries), "anchors": anchors,
                      "dose_bins": {m: len(d["definitions"]) for m, d in bins.items()},
                      "mistral_veto_addback_tiles": definitions["mistral7b"]["veto"]["actual_addback_tiles"]}, sort_keys=True))


if __name__ == "__main__":
    main()
