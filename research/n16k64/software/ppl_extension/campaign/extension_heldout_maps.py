"""Apply the frozen development winner to the two reserved held-out families.

Only calibration artifacts are consumed here.  No WikiText, C4, accuracy, GSM8K,
or PG19 outcome is read.  Five held-out draws are validated as document-disjoint,
the P40/P41 threshold-and-budget rule is replayed, and an eligible P50 selector is
ranked at that exact budget before the P60 scale constant is attached to plans.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

from campaign import mapio as MIO
from campaign import runtime
from campaign import tiles as T
from campaign.extension_maps import Writer, _fixed_consensus, select_global
from campaign.extension_p50_maps import CANDIDATES


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("granite8b", "falcon3_10b")
DRAWS = ("seed0", "draw1", "draw2", "draw3", "draw4")


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, validation = (load(run / name) for name in
                                           ("launch_record.json", "job_status.json",
                                            "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and validation.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def document_hashes(manifest):
    return {row["document_sha256"] for domain in ("math", "code")
            for row in manifest[domain]["documents"]}


def moment_score(blob, names, shapes, k):
    return {name: T.upper_bound(T.Moments.from_state(blob["n16"][name]), k, "ce_kl")
            .reshape(shapes[name][0] // 16, shapes[name][1] // 64)
            for name in names}


def aggregate_scores(blobs, names, shapes, k, aggregation):
    per_draw = [moment_score(blob, names, shapes, k) for blob in blobs]
    if aggregation == "seed0":
        return per_draw[0], None
    stack = {name: torch.stack([row[name].reshape(-1).double() for row in per_draw])
             for name in names}
    if aggregation == "draw_mean":
        return {name: stack[name].mean(0).reshape(per_draw[0][name].shape)
                for name in names}, None
    if aggregation == "draw_median":
        return {name: stack[name].median(0).values.reshape(per_draw[0][name].shape)
                for name in names}, None
    if aggregation == "consensus_3_of_5":
        median = {name: stack[name].median(0).values.reshape(per_draw[0][name].shape)
                  for name in names}
        mean = {name: stack[name].mean(0).reshape(per_draw[0][name].shape)
                for name in names}
        votes = {name: (stack[name] < 0).sum(0).reshape(per_draw[0][name].shape)
                 for name in names}
        return median, (votes, median, mean, 3)
    if aggregation == "pooled_per_sequence_moments":
        pooled = {}
        for name in names:
            obj = T.Moments.from_state(blobs[0]["n16"][name])
            for blob in blobs[1:]:
                obj.add(T.Moments.from_state(blob["n16"][name]))
            pooled[name] = T.upper_bound(obj, k, "ce_kl").reshape(per_draw[0][name].shape)
        return pooled, None
    raise ValueError(aggregation)


def masks_for_rule(blobs, names, shapes, p41, seed0_count):
    if p41["aggregation"] == "seed0_k2_fallback":
        scores = moment_score(blobs[0], names, shapes, 2.0)
        masks = OrderedDict((name, scores[name] < 0) for name in names)
        return scores, masks, "natural", 2.0
    k = float(p41["global_k"])
    scores, consensus = aggregate_scores(blobs, names, shapes, k, p41["aggregation"])
    if p41["form"] == "fixed":
        masks = (select_global(names, scores, seed0_count) if consensus is None else
                 _fixed_consensus(names, *consensus[:3], seed0_count))
    else:
        if consensus is None:
            masks = OrderedDict((name, scores[name] < 0) for name in names)
        else:
            votes, _, _, threshold = consensus
            masks = OrderedDict((name, votes[name] >= threshold) for name in names)
    return scores, masks, p41["form"], k


def selector_key(selection):
    policy = selection["selected_policy"]
    if selection.get("fallback_to_old"):
        return None
    key = policy.removeprefix("p50_")
    if key not in CANDIDATES:
        raise RuntimeError(f"unknown frozen P50 selector: {policy}")
    return key


def map_entry(name, row, multiplier=1.0):
    header, _, digest = MIO.read_map(row["path"], row["sha256"])
    return {"name": name, "kind": "map", "map_path": row["path"],
            "map_sha256": digest, "map_policy": row["policy"],
            "protocol_id": header["protocol_id"], "type_block": row["type_block"],
            "expected_total_tiles": row["total_tiles"],
            "weight_scale_multiplier": float(multiplier)}


def derive(model):
    if model not in MODELS:
        raise ValueError(model)
    p40_path = CR / "provenance/P40_GLOBAL_K_SELECTION.json"
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    p50_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    p60_path = CR / "provenance/P60_SELECTION.json"
    op_path = CR / "provenance/P70_P75_WINNER_HELDOUT_OPERATIONALIZATION.json"
    p40, p41, p50, p60 = (load(path) for path in (p40_path, p41_path, p50_path, p60_path))
    if float(p40["global_k"]) != float(p41["global_k"]):
        raise RuntimeError("P40/P41 global-k identity mismatch")

    blobs, inputs, docs, names, shapes, model_class, module_sha = [], [], set(), None, None, None, None
    seed_maps = None
    manifest_set = {"schema_version": "1.0", "model": model, "draws": []}
    for draw in DRAWS:
        run = latest_complete(f"P70_calib_{model}_{draw}")
        report_path = run / "calibration/calibration_report.json"
        manifest_path = run / "calibration/calibration_manifest.json"
        moments_path = run / "calibration/moments/moments_full.pt"
        map_manifest_path = run / "calibration/map_manifest.json"
        report, manifest = load(report_path), load(manifest_path)
        if (report.get("status") != "complete" or report.get("model") != model
                or report.get("draw") != draw or report.get("freeze_sha256") != PROTOCOL):
            raise RuntimeError(f"heldout calibration identity failed: {model}/{draw}")
        current_docs = document_hashes(manifest)
        if len(current_docs) != 128 or current_docs & docs:
            raise RuntimeError(f"heldout draw overlap/cardinality failed: {model}/{draw}")
        docs |= current_docs
        blob = torch.load(moments_path, map_location="cpu", weights_only=False)
        if names is None:
            names = list(blob["names"]); shapes = {n: tuple(blob["shapes"][n]) for n in names}
            model_class = report["model_class"]; module_sha = report["module_manifest_sha256"]
            seed_maps = {row["policy"]: row for row in load(map_manifest_path)}
        elif (list(blob["names"]) != names or any(tuple(blob["shapes"][n]) != shapes[n] for n in names)
              or report["module_manifest_sha256"] != module_sha):
            raise RuntimeError(f"heldout draw module geometry differs: {model}/{draw}")
        blobs.append(blob)
        row = {"draw": draw, "run_id": run.name,
               "report_sha256": runtime.sha256_file(report_path),
               "manifest_file_sha256": runtime.sha256_file(manifest_path),
               "manifest_canonical_sha256": report["calibration_manifest_sha256"],
               "moments_sha256": runtime.sha256_file(moments_path),
               "map_manifest_sha256": runtime.sha256_file(map_manifest_path)}
        inputs.append(row); manifest_set["draws"].append(row)
    if len(docs) != 640:
        raise RuntimeError("heldout five-draw document audit failed")
    manifest_set["unique_documents"] = len(docs)
    manifest_set_path = runtime.out_dir("heldout_maps") / "calibration_manifest_set.json"
    runtime.atomic_json(manifest_set_path, manifest_set)

    seed_n16_header, seed_n16_masks, _ = MIO.read_map(seed_maps["n16_k2"]["path"], seed_maps["n16_k2"]["sha256"])
    seed0_count = int(seed_n16_header["totals"]["selected_tiles"])
    scores, base_masks, form, k = masks_for_rule(blobs, names, shapes, p41, seed0_count)
    budget = sum(int(base_masks[name].sum()) for name in names)
    key = selector_key(p50)
    stats_input = None
    if key is None:
        winner_masks = base_masks
    else:
        stats_run = latest_complete(f"P70_selector_stats_{model}")
        stats_path = stats_run / "selector_stats/selector_statistics.pt"
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        if (stats.get("model") != model or stats.get("protocol_sha256") != PROTOCOL
                or stats.get("names") != names or stats.get("module_manifest_sha256") != module_sha):
            raise RuntimeError("heldout selector-statistics identity mismatch")
        candidate = stats["scores"][key]
        if any(int(candidate[n].numel()) != int(base_masks[n].numel()) for n in names):
            raise RuntimeError("heldout selector-statistics geometry mismatch")
        winner_masks = select_global(names, candidate, budget)
        stats_input = {"run_id": stats_run.name, "path": str(stats_path),
                       "sha256": runtime.sha256_file(stats_path), "selector": key}

    writer = Writer(model, names, shapes, runtime.sha256_file(manifest_set_path),
                    model_class, runtime.sha256_file(op_path))
    base = writer.write("p70_first_order_threshold_budget", base_masks,
                        {"selector": "first_order_ce_kl", "aggregation": p41["aggregation"],
                         "form": form, "k": k, "heldout": True},
                        "frozen P40/P41 threshold, aggregation and budget replayed on heldout calibration")
    winner = writer.write("p70_frozen_winner", winner_masks,
                          {"selector": key or "first_order_ce_kl",
                           "aggregation": p41["aggregation"], "form": form, "k": k,
                           "target_tiles": budget, "weight_scale_multiplier": float(p60["selected_multiplier"]),
                           "activation": "four_over_six_rows", "heldout": True},
                          "frozen selector at exact threshold/budget count; no heldout quality consumed")
    report = {"schema_version": "1.0", "mode": "P70_HELDOUT_WINNER_MAP",
              "model": model, "status": "running", "protocol_sha256": PROTOCOL,
              "operationalization_sha256": runtime.sha256_file(op_path),
              "P40_selection_sha256": runtime.sha256_file(p40_path),
              "P41_selection_sha256": runtime.sha256_file(p41_path),
              "P50_selection_sha256": runtime.sha256_file(p50_path),
              "P60_selection_sha256": runtime.sha256_file(p60_path),
              "calibration_inputs": inputs, "unique_documents": len(docs),
              "calibration_manifest_set": {"path": str(manifest_set_path),
                                             "sha256": runtime.sha256_file(manifest_set_path)},
              "module_manifest_sha256": module_sha,
              "global_k": k, "aggregation": p41["aggregation"], "form": form,
              "seed0_n16_k2_tiles": seed0_count, "winner_selected_tiles": budget,
              "selector_statistics": stats_input, "selected_selector": key or "first_order_ce_kl",
              "weight_scale_multiplier": float(p60["selected_multiplier"]),
              "activation_kind": "four_over_six_rows",
              "base_map_policy": base["policy"], "winner_map_policy": winner["policy"]}
    writer.finish(report)

    # Add evaluation plans only after both exact maps and all fixed baseline
    # semantics have been resolved.  N16-k3 is retained as an extra anchor.
    plan = [
        {"name": "four_over_six", "kind": "four_over_six"},
        {"name": "nvfp4", "kind": "nvfp4"},
        {"name": "all_e0m3", "kind": "all_e0m3"},
        map_entry("n8_k2", seed_maps["n8_k2"]),
        map_entry("n16_k2", seed_maps["n16_k2"]),
        map_entry("n16_k3", seed_maps["n16_k3"]),
        map_entry("frozen_winner", winner, float(p60["selected_multiplier"])),
        {"name": "razer_wonly_shared_act", "kind": "razer_wonly_shared_act"},
        {"name": "razer_native_rows", "kind": "razer_native_rows"},
        {"name": "nover6_wonly_shared_act", "kind": "nover6_wonly_shared_act"},
        {"name": "nover6_native_rows", "kind": "nover6_native_rows"},
    ]
    plan_path = CR / "plans" / f"P71_heldout_ppl_{model}.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    runtime.atomic_json(plan_path, plan); plan_path.chmod(0o444)
    result_path = runtime.run_dir / "job_result.json"
    result = load(result_path)
    result["results"]["raw_outputs"].extend([str(manifest_set_path), str(plan_path)])
    result["results"]["summary"].update(
        winner_selected_tiles=budget, selector=key or "first_order_ce_kl",
        global_k=k, aggregation=p41["aggregation"], form=form,
        weight_scale_multiplier=float(p60["selected_multiplier"]),
        P71_plan_sha256=runtime.sha256_file(plan_path))
    result["results"]["attempted_endpoints"] = ["P70_HELDOUT_MAP_DERIVATION"]
    runtime.atomic_json(result_path, result)
    print(json.dumps({"status": "complete", "model": model, "winner_map": winner,
                      "P71_plan": str(plan_path)}, sort_keys=True))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=MODELS, required=True); a = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    derive(a.model)


if __name__ == "__main__":
    main()
