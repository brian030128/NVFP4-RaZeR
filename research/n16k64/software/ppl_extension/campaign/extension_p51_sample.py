"""Freeze the three 60-tile P51 manifests before opening exact effects."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from campaign import mapio as MIO
from campaign import runtime
from campaign import tiles as T
from campaign.extension_maps import moment_objects, moments
from campaign.extension_p50_maps import CANDIDATES, CR, MODELS, PROTOCOL, latest_complete, load


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def allocate_strata(sizes, total):
    """One per nonempty stratum, then capped proportional largest remainder."""
    sizes = {tuple(key): int(value) for key, value in sizes.items() if int(value) > 0}
    if total < len(sizes) or total > sum(sizes.values()):
        raise ValueError("sample total incompatible with nonempty strata/population")
    quota = {key: 1 for key in sizes}
    remaining = total - len(sizes)
    while remaining:
        active = [key for key in sorted(sizes) if quota[key] < sizes[key]]
        if not active:
            raise RuntimeError("quota redistribution exhausted population")
        denom = sum(sizes[key] for key in active)
        ideal = {key: remaining * sizes[key] / denom for key in active}
        floors = {key: min(sizes[key] - quota[key], int(math.floor(ideal[key]))) for key in active}
        added = sum(floors.values())
        for key, value in floors.items():
            quota[key] += value
        remaining -= added
        if remaining == 0:
            break
        eligible = [key for key in active if quota[key] < sizes[key]]
        if not eligible:
            continue
        # One largest-remainder pass; a new proportional round follows if caps
        # prevent consuming the entire remaining quota.
        for key in sorted(eligible, key=lambda x: (-(ideal[x] - math.floor(ideal[x])), x)):
            if not remaining:
                break
            quota[key] += 1
            remaining -= 1
    if sum(quota.values()) != total or any(quota[k] > sizes[k] for k in quota):
        raise RuntimeError("invalid allocated quota")
    return quota


def micro_counts(model, sizes, quota):
    """Round-robin the stratum quota over hash-ordered nonempty microcells."""
    ordered = sorted(sizes, key=lambda key: (
        hashlib.sha256(("P51-MICRO:" + model + ":" + canonical(list(key))).encode()).digest(), key))
    out = {key: 0 for key in ordered}
    left = int(quota)
    while left:
        progress = False
        for key in ordered:
            if out[key] < sizes[key]:
                out[key] += 1
                left -= 1
                progress = True
                if not left:
                    break
        if not progress:
            raise RuntimeError("microcell quota exceeds its population")
    return out


def stable_quintiles(values):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite score in quintile assignment")
    order = torch.argsort(values, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(values.numel(), dtype=order.dtype)
    return torch.clamp((rank * 5) // values.numel(), max=4).to(torch.uint8)


def flatten(names, group):
    return torch.cat([group[name].reshape(-1).double() for name in sorted(names)])


def unflatten(names, shapes, values):
    out, offset = {}, 0
    for name in sorted(names):
        count = (shapes[name][0] // 16) * (shapes[name][1] // 64)
        out[name] = values[offset:offset + count]
        offset += count
    if offset != values.numel():
        raise RuntimeError("score unflatten length mismatch")
    return out


def old_score(model, names, shapes, p41):
    aggregation = p41["aggregation"]
    if aggregation == "seed0_k2_fallback":
        blob, path, _ = moments(model, "seed0")
        objs = moment_objects(blob, "n16")
        group = {name: T.upper_bound(objs[name], 2.0, "ce_kl") for name in names}
        return group, {name: group[name] < 0 for name in names}, {
            "kind": "seed0_k2_fallback", "path": str(path), "sha256": runtime.sha256_file(path)}
    run = latest_complete(f"P41_maps_{model}")
    path = run / "derived_maps/aggregation_scores.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    lookup = {"seed0": blob["per_draw_u"][0],
              "pooled_per_sequence_moments": blob["pooled_u"],
              "draw_mean": blob["draw_mean_u"], "draw_median": blob["draw_median_u"]}
    if aggregation != "consensus_3_of_5":
        group = lookup[aggregation]
        return group, {name: group[name].reshape(-1) < 0 for name in names}, {
            "kind": aggregation, "path": str(path), "sha256": runtime.sha256_file(path)}
    votes, med, avg = blob["votes"], blob["draw_median_u"], blob["draw_mean_u"]
    vf, mf, af = flatten(names, votes), flatten(names, med), flatten(names, avg)
    idx = torch.arange(vf.numel())
    idx = idx[torch.argsort(af[idx], stable=True)]
    idx = idx[torch.argsort(mf[idx], stable=True)]
    idx = idx[torch.argsort(-vf[idx], stable=True)]
    rank = torch.empty_like(idx); rank[idx] = torch.arange(idx.numel(), dtype=idx.dtype)
    group = unflatten(names, shapes, rank.double())
    signs = {name: votes[name].reshape(-1) >= 3 for name in names}
    return group, signs, {"kind": "consensus_3_of_5_composite_rank",
                           "path": str(path), "sha256": runtime.sha256_file(path)}


def layer_index(name):
    return int(name.split("layers.", 1)[1].split(".", 1)[0]) if "layers." in name else -1


def rel_quintile(old_q, candidate_q):
    return 0 if old_q < candidate_q else (1 if old_q == candidate_q else 2)


def build_model(model, selection, sampling_spec_sha):
    stats_run = latest_complete(f"P50_selector_stats_{model}")
    stats_path = stats_run / "selector_stats/selector_statistics.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    names = list(stats["names"]); shapes = {name: tuple(stats["shapes"][name]) for name in names}
    if stats.get("protocol_sha256") != PROTOCOL:
        raise RuntimeError("selector statistics protocol mismatch")
    audit_policy = selection["P51_audit_policy"]
    candidate = audit_policy.removeprefix("p50_")
    if candidate not in CANDIDATES:
        raise RuntimeError("P51 audit policy is not a replacement candidate")
    candidate_group = stats["scores"][candidate]
    old_group, old_sign_group, old_source = old_score(model, names, shapes, load(CR / "provenance/P41_AGGREGATION_SELECTION.json"))
    old_flat = flatten(names, old_group); candidate_flat = flatten(names, candidate_group)
    oq = stable_quintiles(old_flat); cq = stable_quintiles(candidate_flat)
    old_by_name = unflatten(names, shapes, old_flat)
    cand_by_name = unflatten(names, shapes, candidate_flat)
    oq_by_name = unflatten(names, shapes, oq)
    cq_by_name = unflatten(names, shapes, cq)

    old_map_row = selection["P51_audit_maps"][model]  # candidate map is frozen here
    selected_map_row = selection["selected_maps"][model]
    p41_map_row = load(CR / "provenance/P41_AGGREGATION_SELECTION.json")["selected_maps"][model]
    _, candidate_masks, _ = MIO.read_map(old_map_row["map_path"], old_map_row["map_sha256"])
    _, winner_masks, _ = MIO.read_map(selected_map_row["map_path"], selected_map_row["map_sha256"])
    _, old_masks, _ = MIO.read_map(p41_map_row["map_path"], p41_map_row["map_sha256"])
    if list(old_masks) != names or list(candidate_masks) != names or list(winner_masks) != names:
        raise RuntimeError("P51 map/stat module order mismatch")

    nlayers = max(layer_index(name) for name in names) + 1
    if nlayers <= 0:
        raise RuntimeError("cannot derive layer quartiles")
    collapsed_sizes = defaultdict(int)
    micro_sizes = defaultdict(lambda: defaultdict(int))
    module_arrays = {}
    for name in names:
        count = old_by_name[name].numel()
        osign = old_sign_group[name].reshape(-1).bool().numpy()
        csign = (cand_by_name[name] < 0).numpy()
        oldq = oq_by_name[name].numpy().astype(np.int8)
        candq = cq_by_name[name].numpy().astype(np.int8)
        lq = min(3, 4 * layer_index(name) // nlayers)
        mtype = name.split(".")[-1]
        for i in range(count):
            s = (lq, int(osign[i]), int(csign[i]), rel_quintile(int(oldq[i]), int(candq[i])))
            u = (mtype, int(oldq[i]), int(candq[i]))
            collapsed_sizes[s] += 1; micro_sizes[s][u] += 1
        # Retain only compact arrays.  The categorical keys are regenerated in
        # the hash-selection pass instead of persisting millions of Python tuples.
        module_arrays[name] = (osign, csign, oldq, candq, lq, mtype)
    quota = allocate_strata(collapsed_sizes, 60)
    selected_per_micro = {}
    for stratum in sorted(quota):
        counts = micro_counts(model, micro_sizes[stratum], quota[stratum])
        for micro, value in counts.items():
            selected_per_micro[(stratum, micro)] = value

    heaps = {key: [] for key, value in selected_per_micro.items() if value}
    for name in sorted(names):
        osign, csign, oldq, candq, lq, mtype = module_arrays[name]
        gk = shapes[name][1] // 64
        for index in range(oldq.size):
            stratum = (lq, int(osign[index]), int(csign[index]),
                       rel_quintile(int(oldq[index]), int(candq[index])))
            microcell = (mtype, int(oldq[index]), int(candq[index]))
            key = (stratum, microcell); keep = selected_per_micro.get(key, 0)
            if not keep:
                continue
            row, kcol = divmod(index, gk)
            digest = hashlib.sha256(f"P51:{model}:{name}:{row}:{kcol}".encode()).digest()
            value = int.from_bytes(digest, "big")
            item = (-value, name, row, kcol, digest.hex())
            heap = heaps[key]
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    rows = []
    all_scores = {key: stats["scores"][key] for key in CANDIDATES}
    for (stratum, microcell), heap in heaps.items():
        population = micro_sizes[stratum][microcell]
        picked = len(heap)
        if picked != selected_per_micro[(stratum, microcell)]:
            raise RuntimeError("P51 heap/sample count mismatch")
        for _, name, row, kcol, digest in sorted(heap, key=lambda x: (-x[0], x[1], x[2], x[3])):
            gk = shapes[name][1] // 64; index = row * gk + kcol
            osign, csign, oldq, candq, _, _ = module_arrays[name]
            score_columns = {key: float(all_scores[key][name].reshape(-1)[index]) for key in CANDIDATES}
            rows.append({
                "sample_id": f"{model}:{name}:{row}:{kcol}", "model": model, "module": name,
                "layer": layer_index(name), "layer_quartile": stratum[0], "module_type": microcell[0],
                "row_tile": row, "k_tile": kcol, "flat_tile_in_module": index,
                "tile_shape": [16, 64], "selection_sha256": digest,
                "collapsed_stratum": list(stratum), "old_score_quintile": int(oldq[index]),
                "candidate_score_quintile": int(candq[index]),
                "old_predicted_beneficial": bool(osign[index]),
                "candidate_predicted_beneficial": bool(csign[index]),
                "old_score": float(old_by_name[name][index]), "audit_candidate": candidate,
                "audit_candidate_score": float(cand_by_name[name][index]), "candidate_scores": score_columns,
                "old_map_selected": bool(old_masks[name].reshape(-1)[index]),
                "audit_candidate_map_selected": bool(candidate_masks[name].reshape(-1)[index]),
                "method_winner_map_selected": bool(winner_masks[name].reshape(-1)[index]),
                "microcell_population": population, "microcell_selected": picked,
                "inclusion_probability": picked / population, "sampling_weight": population / picked,
            })
    rows.sort(key=lambda row: (row["module"], row["row_tile"], row["k_tile"]))
    if len(rows) != 60 or len({row["sample_id"] for row in rows}) != 60:
        raise RuntimeError(f"P51 sample size/uniqueness mismatch: {model}")
    token_hashes = stats["sequence_token_sha256"]
    audit_indices = list(range(8, 16)) + list(range(72, 80))
    manifest = {
        "schema_version": "1.0", "status": "FROZEN_BEFORE_EXACT_EFFECTS", "model": model,
        "protocol_sha256": PROTOCOL, "sampling_spec_sha256": sampling_spec_sha,
        "selector_statistics": {"run_id": stats_run.name, "path": str(stats_path),
                                "sha256": runtime.sha256_file(stats_path)},
        "old_score_source": old_source, "audit_candidate": candidate,
        "audit_indices": audit_indices, "audit_token_sha256": [token_hashes[i] for i in audit_indices],
        "map_inputs": {"old": p41_map_row, "audit_candidate": old_map_row,
                       "method_winner": selected_map_row},
        "eligible_tiles": int(old_flat.numel()), "sample_tiles": len(rows),
        "collapsed_stratum_population": {canonical(list(k)): v for k, v in sorted(collapsed_sizes.items())},
        "collapsed_stratum_quota": {canonical(list(k)): v for k, v in sorted(quota.items())},
        "microcell_population": {canonical([list(s), list(m)]): v for s in sorted(micro_sizes)
                                 for m, v in sorted(micro_sizes[s].items())},
        "microcell_selected": {canonical([list(s), list(m)]): v for (s, m), v in sorted(selected_per_micro.items())},
        "tiles": rows,
    }
    return manifest


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value); path.chmod(0o444)


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    selection_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    sampling_path = CR / "provenance/P51_SAMPLING_OPERATIONALIZATION.json"
    selection = load(selection_path); sampling_sha = runtime.sha256_file(sampling_path)
    rows = []
    for model in MODELS:
        manifest = build_model(model, selection, sampling_sha)
        path = CR / "plans" / f"P51_tiles_{model}.json"
        write_new(path, manifest)
        rows.append({"model": model, "path": str(path), "sha256": runtime.sha256_file(path),
                     "sample_tiles": len(manifest["tiles"]), "eligible_tiles": manifest["eligible_tiles"],
                     "audit_candidate": manifest["audit_candidate"]})
    freeze = {"schema_version": "1.0", "status": "LOCKED_BEFORE_FIRST_P51_EXACT_EFFECT",
              "protocol_sha256": PROTOCOL, "sampling_spec_sha256": sampling_sha,
              "P50_selection_sha256": runtime.sha256_file(selection_path), "manifests": rows,
              "total_sample_tiles": sum(row["sample_tiles"] for row in rows)}
    freeze_path = CR / "provenance/P51_SAMPLE_FREEZE.json"
    write_new(freeze_path, freeze)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P51-sample-freeze", "model_revision": runtime.sha256_file(freeze_path),
                   "tokenizer_revision": "multiple", "model_class": "sampling",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None},
        "policies": [], "results": {"raw_outputs": [str(freeze_path)] + [row["path"] for row in rows],
            "summary": {"models": len(rows), "sample_tiles": sum(row["sample_tiles"] for row in rows)},
            "uncertainty": {"sampling_weights_stored": True},
            "attempted_endpoints": ["P51_SAMPLE_FREEZE"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "freeze": str(freeze_path), "manifests": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
