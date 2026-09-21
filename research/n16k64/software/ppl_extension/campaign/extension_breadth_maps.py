"""Calibration-only exact-map construction for conditional breadth stages P21/P42/P52.

Every transferable choice is read from an immutable development decision.  This
module never reads a breadth WikiText/C4 result while deriving a map.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

from campaign import mapio as MIO
from campaign import runtime
from campaign import tiles as T
from campaign.extension_heldout_maps import aggregate_scores, masks_for_rule
from campaign.extension_maps import Writer, select_global, select_per_module
from campaign.extension_p50_maps import CANDIDATES


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("qwen27b", "phi4", "olmo2_13b")
DRAWS = ("seed0", "draw1", "draw2", "draw3", "draw4")
PARENT = {"qwen27b": "V30_calib_qwen27b_seed0_attempt1",
          "phi4": "V61_calib_phi4_seed0_attempt1",
          "olmo2_13b": "V61_calib_olmo2_13b_seed0_attempt2"}
OP = CR / "provenance/P21_P42_P52_BREADTH_OPERATIONALIZATION.json"


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    found = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, valid = (load(run / name) for name in
                                     ("launch_record.json", "job_status.json",
                                      "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and valid.get("valid")):
            found.append((attempt, run))
    if not found:
        raise FileNotFoundError(prefix)
    return max(found)[1]


def calibration_run(model, draw):
    return (PR / "runs" / PARENT[model] if draw == "seed0"
            else latest_complete(f"P30_calib_{model}_{draw}"))


def calibration_blob(model, draw):
    run = calibration_run(model, draw)
    report_path = run / "calibration/calibration_report.json"
    manifest_path = run / "calibration/calibration_manifest.json"
    moments_path = run / "calibration/moments/moments_full.pt"
    report, manifest = load(report_path), load(manifest_path)
    if (report.get("status") != "complete" or report.get("model") != model
            or report.get("draw") != draw):
        raise RuntimeError(f"unaccepted breadth calibration: {model}/{draw}")
    digest = runtime.sha256_file(moments_path)
    if report.get("moment_files", {}).get("moments_full.pt") != digest:
        raise RuntimeError(f"breadth moments digest mismatch: {model}/{draw}")
    return torch.load(moments_path, map_location="cpu", weights_only=False), {
        "draw": draw, "run_id": run.name,
        "calibration_report_sha256": runtime.sha256_file(report_path),
        "calibration_manifest_file_sha256": runtime.sha256_file(manifest_path),
        "calibration_manifest_canonical_sha256": report["calibration_manifest_sha256"],
        "moments_path": str(moments_path), "moments_sha256": digest,
        "module_manifest_sha256": report["module_manifest_sha256"],
        "model_class": report["model_class"],
        "document_sha256": [row["document_sha256"]
                            for domain in ("math", "code")
                            for row in manifest[domain]["documents"]],
    }, report


def parent_map(model, policy):
    path = PR / "runs" / PARENT[model] / "calibration/map_manifest.json"
    row = next(row for row in load(path) if row["policy"] == policy)
    header, masks, digest = MIO.read_map(row["path"], row["sha256"])
    return row, header, masks, digest


def map_entry(name, row):
    path = row.get("map_path", row.get("path"))
    digest = row.get("map_sha256", row.get("sha256"))
    header, _, got = MIO.read_map(path, digest)
    return {"name": name, "kind": "map", "map_path": path,
            "map_sha256": got,
            "map_policy": row.get("map_policy", row.get("policy")),
            "protocol_id": header["protocol_id"],
            "type_block": row["type_block"],
            "expected_total_tiles": row.get("expected_total_tiles",
                                             row.get("total_tiles"))}


def selector_artifact(model):
    candidates = []
    for prefix in (f"P21_selector_stats_{model}", f"P42_selector_stats_{model}",
                   f"P52_selector_stats_{model}"):
        try:
            run = latest_complete(prefix)
        except FileNotFoundError:
            continue
        path = run / "selector_stats/selector_statistics.pt"
        report_path = run / "selector_stats/selector_stats_report.json"
        report = load(report_path)
        if (report.get("status") == "complete"
                and report.get("statistic_artifact", {}).get("sha256")
                == runtime.sha256_file(path)):
            candidates.append((run.stat().st_mtime_ns, run, path, report_path, report))
    if not candidates:
        raise FileNotFoundError(f"no accepted breadth selector statistics for {model}")
    _, run, path, report_path, report = max(candidates)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return artifact, {"run_id": run.name, "path": str(path),
                      "sha256": runtime.sha256_file(path),
                      "report_sha256": runtime.sha256_file(report_path)}


def seed_context(model):
    blob, info, report = calibration_blob(model, "seed0")
    names = list(blob["names"])
    shapes = {name: tuple(blob["shapes"][name]) for name in names}
    n16 = OrderedDict((name, T.Moments.from_state(blob["n16"][name]))
                      for name in names)
    n8 = OrderedDict((name, T.Moments.from_state(blob["n8"][name]))
                     for name in names)
    return blob, info, report, names, shapes, n16, n8


def strong_scores(model, names, shapes, selector=None):
    selection = load(CR / "provenance/P20_STRONGEST_CONTROL_SELECTION.json")
    selector = selector or selection["selector"]
    seed = PR / "runs" / PARENT[model]
    weight_path = seed / "calibration/moments/weight_tile_stats.pt"
    seed_report = load(seed / "calibration/calibration_report.json")
    if seed_report["moment_files"].get("weight_tile_stats.pt") != runtime.sha256_file(weight_path):
        raise RuntimeError("breadth weight statistics digest mismatch")
    weight = torch.load(weight_path, map_location="cpu", weights_only=False)
    stats_input = None
    if selector == "activation_weighted":
        stats, stats_input = selector_artifact(model)
        if (stats.get("model") != model or stats.get("protocol_sha256") != PROTOCOL
                or list(stats.get("names", [])) != names):
            raise RuntimeError("breadth activation statistic identity mismatch")
        scores, descending = stats["scores"]["activation_weighted_error"], False
    else:
        key = {"weight_mse": "mse_gain16", "magnitude": "l2_16",
               "change_norm": "dnorm16"}.get(selector)
        if key is None:
            raise ValueError(f"unsupported frozen P20 selector: {selector}")
        scores, descending = weight[key], True
    if list(scores) != names:
        raise RuntimeError("breadth strong-score module order mismatch")
    for name in names:
        expected = (shapes[name][0] // 16) * (shapes[name][1] // 64)
        if scores[name].numel() != expected or not torch.isfinite(scores[name]).all():
            raise RuntimeError(f"invalid breadth strong score: {model}/{name}")
    return scores, descending, selection, {
        "weight_statistics_path": str(weight_path),
        "weight_statistics_sha256": runtime.sha256_file(weight_path),
        "selector_statistics": stats_input,
    }


def write_plan(path, plan):
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, plan)
    path.chmod(0o444)


def append_result(plan_path, extra, attempted):
    path = runtime.run_dir / "job_result.json"
    result = load(path)
    result["results"]["raw_outputs"].append(str(plan_path))
    result["results"]["raw_outputs"].extend(extra)
    result["results"]["summary"].update(
        plan_path=str(plan_path), plan_sha256=runtime.sha256_file(plan_path))
    result["results"]["attempted_endpoints"] = [attempted]
    runtime.atomic_json(path, result)


def p21(model):
    _, seed_info, report, names, shapes, _, n8 = seed_context(model)
    n16_row, _, natural, natural_sha = parent_map(model, "n16_k2")
    counts = {name: int(natural[name].sum()) for name in names}
    target = sum(counts.values())
    scores, descending, selected, score_inputs = strong_scores(model, names, shapes)
    scope = selected["scope"]
    strong = (select_global(names, scores, target, descending)
              if scope == "global" else select_per_module(names, scores, counts, descending))
    u8 = {name: T.upper_bound(n8[name], 2.0, "ce_kl").reshape(
        shapes[name][0] // 8, shapes[name][1] // 64) for name in names}
    n8_masks = (select_global(names, u8, 2 * target)
                if scope == "global" else select_per_module(
                    names, u8, {name: 2 * counts[name] for name in names}))
    writer = Writer(model, names, shapes, seed_info["calibration_manifest_canonical_sha256"],
                    report["model_class"], runtime.sha256_file(OP))
    matched = writer.write("p21_n8_u2_matched", n8_masks,
                           {"selector": "n8_u2", "scope": scope,
                            "target_selected_weights": target * 1024},
                           "frozen P20 scope; ascending seed0 N8 U2; exact weight budget",
                           tb=T.N8)
    control = writer.write("p21_frozen_strongest_control", strong,
                           {"selector": selected["selector"], "scope": scope,
                            "target_selected_weights": target * 1024},
                           "frozen P20 strongest deterministic selector and scope")
    writer.finish({"schema_version": "1.0", "mode": "P21", "model": model,
                   "status": "running", "protocol_sha256": PROTOCOL,
                   "operationalization_sha256": runtime.sha256_file(OP),
                   "P20_selection_sha256": runtime.sha256_file(
                       CR / "provenance/P20_STRONGEST_CONTROL_SELECTION.json"),
                   "seed_calibration": seed_info,
                   "natural_n16_k2_map_sha256": natural_sha,
                   "target_selected_tiles": target,
                   "target_selected_weights": target * 1024,
                   "scope": scope, "selector": selected["selector"],
                   "score_inputs": score_inputs,
                   "module_manifest_sha256": report["module_manifest_sha256"]})
    plan = [map_entry("n16_k2_reference", n16_row),
            map_entry("p21_n8_u2_matched", matched),
            map_entry("p21_frozen_strongest_control", control)]
    plan_path = CR / "plans" / f"P21_density_breadth_{model}.json"
    write_plan(plan_path, plan)
    append_result(plan_path, [], "P21_K2_DENSITY_BREADTH_MAPS")


def five_draw_context(model):
    blobs, inputs, names, shapes, report0, module_sha = [], [], None, None, None, None
    documents = set()
    for draw in DRAWS:
        blob, info, report = calibration_blob(model, draw)
        current_documents = set(info.pop("document_sha256"))
        if len(current_documents) != 128 or current_documents & documents:
            raise RuntimeError(f"breadth draw document overlap/cardinality differs: {model}/{draw}")
        documents |= current_documents
        if names is None:
            names = list(blob["names"])
            shapes = {name: tuple(blob["shapes"][name]) for name in names}
            report0, module_sha = report, report["module_manifest_sha256"]
        elif (list(blob["names"]) != names
              or any(tuple(blob["shapes"][name]) != shapes[name] for name in names)
              or report["module_manifest_sha256"] != module_sha):
            raise RuntimeError(f"breadth draw geometry differs: {model}/{draw}")
        blobs.append(blob); inputs.append(info)
    if len(documents) != 640:
        raise RuntimeError(f"breadth five-draw document total differs: {model}")
    manifest_set = {"schema_version": "1.0", "model": model,
                    "draw_order": list(DRAWS), "unique_documents": len(documents),
                    "inputs": inputs}
    path = runtime.out_dir("breadth_maps") / "calibration_manifest_set.json"
    runtime.atomic_json(path, manifest_set)
    return blobs, inputs, names, shapes, report0, module_sha, path


def p42(model):
    blobs, inputs, names, shapes, report, module_sha, manifest_set = five_draw_context(model)
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    p41 = load(p41_path)
    n16_row, _, seed_masks, seed_sha = parent_map(model, "n16_k2")
    seed_count = sum(int(seed_masks[name].sum()) for name in names)
    _, aggregate_masks, form, k = masks_for_rule(blobs, names, shapes, p41, seed_count)
    target_counts = {name: int(aggregate_masks[name].sum()) for name in names}
    target = sum(target_counts.values())
    scores, descending, selected, score_inputs = strong_scores(model, names, shapes)
    strong = (select_global(names, scores, target, descending)
              if selected["scope"] == "global" else
              select_per_module(names, scores, target_counts, descending))
    writer = Writer(model, names, shapes, runtime.sha256_file(manifest_set),
                    report["model_class"], runtime.sha256_file(OP))
    agg = writer.write("p42_frozen_aggregation", aggregate_masks,
                       {"selector": "first_order_ce_kl", "global_k": k,
                        "aggregation": p41["aggregation"], "form": form},
                       "exact P40/P41 aggregation and form replay on breadth five draws")
    control = writer.write("p42_strong_control_budget_matched", strong,
                           {"selector": selected["selector"],
                            "scope": selected["scope"], "target_tiles": target},
                           "frozen P20 strong control matched to P42 candidate budget")
    writer.finish({"schema_version": "1.0", "mode": "P42", "model": model,
                   "status": "running", "protocol_sha256": PROTOCOL,
                   "operationalization_sha256": runtime.sha256_file(OP),
                   "P41_selection_sha256": runtime.sha256_file(p41_path),
                   "calibration_inputs": inputs,
                   "calibration_manifest_set": {"path": str(manifest_set),
                                                  "sha256": runtime.sha256_file(manifest_set)},
                   "seed0_n16_k2_map_sha256": seed_sha,
                   "seed0_selected_tiles": seed_count,
                   "candidate_selected_tiles": target,
                   "candidate_selected_weights": target * 1024,
                   "global_k": k, "aggregation": p41["aggregation"], "form": form,
                   "strong_selector": selected["selector"],
                   "strong_scope": selected["scope"], "score_inputs": score_inputs,
                   "module_manifest_sha256": module_sha})
    plan = [map_entry("seed0_n16_k2_reference", n16_row),
            map_entry("p42_frozen_aggregation", agg),
            map_entry("p42_strong_control_budget_matched", control)]
    plan_path = CR / "plans" / f"P42_aggregation_breadth_{model}.json"
    write_plan(plan_path, plan)
    append_result(plan_path, [str(manifest_set)], "P42_DRAW_AGG_BREADTH_MAPS")


def p52(model):
    blobs, inputs, names, shapes, report, module_sha, manifest_set = five_draw_context(model)
    p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
    p50_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    p20_path = CR / "provenance/P20_STRONGEST_CONTROL_SELECTION.json"
    p41, p50, p20 = load(p41_path), load(p50_path), load(p20_path)
    _, _, seed_masks, seed_sha = parent_map(model, "n16_k2")
    seed_count = sum(int(seed_masks[name].sum()) for name in names)
    _, old_masks, form, k = masks_for_rule(blobs, names, shapes, p41, seed_count)
    target = sum(int(old_masks[name].sum()) for name in names)
    replacement = p50["selected_policy"].removeprefix("p50_")
    if p50.get("fallback_to_old") or replacement not in CANDIDATES:
        raise RuntimeError("P52 requires a frozen non-old P50 selector")
    stats, stats_input = selector_artifact(model)
    if (stats.get("model") != model or stats.get("protocol_sha256") != PROTOCOL
            or list(stats.get("names", [])) != names
            or stats.get("module_manifest_sha256") != module_sha):
        raise RuntimeError("P52 selector-statistics identity mismatch")
    candidate_scores = stats["scores"][replacement]
    winner_masks = select_global(names, candidate_scores, target)
    strong, descending, _, score_inputs = strong_scores(
        model, names, shapes, selector=p20["selector"])
    strong_masks = select_global(names, strong, target, descending)
    writer = Writer(model, names, shapes, runtime.sha256_file(manifest_set),
                    report["model_class"], runtime.sha256_file(OP))
    old = writer.write("p52_old_first_order", old_masks,
                       {"selector": "first_order_ce_kl", "global_k": k,
                        "aggregation": p41["aggregation"], "form": form},
                       "P41-frozen old first-order aggregation replay on breadth draws")
    heuristic = writer.write("p52_strongest_p20_selector_global_rerank", strong_masks,
                             {"selector": p20["selector"], "scope": "global",
                              "target_tiles": target},
                             "P20-frozen strongest heuristic reranked globally at exact P52 budget")
    winner = writer.write("p52_frozen_replacement", winner_masks,
                          {"selector": replacement, "scope": "global",
                           "target_tiles": target},
                          "P50/P51-frozen replacement selector at exact P52 budget")
    writer.finish({"schema_version": "1.0", "mode": "P52", "model": model,
                   "status": "running", "protocol_sha256": PROTOCOL,
                   "operationalization_sha256": runtime.sha256_file(OP),
                   "P41_selection_sha256": runtime.sha256_file(p41_path),
                   "P50_selection_sha256": runtime.sha256_file(p50_path),
                   "P20_selection_sha256": runtime.sha256_file(p20_path),
                   "calibration_inputs": inputs,
                   "calibration_manifest_set": {"path": str(manifest_set),
                                                  "sha256": runtime.sha256_file(manifest_set)},
                   "seed0_n16_k2_map_sha256": seed_sha,
                   "target_selected_tiles": target,
                   "target_selected_weights": target * 1024,
                   "global_k": k, "aggregation": p41["aggregation"], "form": form,
                   "replacement_selector": replacement,
                   "selector_statistics": stats_input,
                   "strong_score_inputs": score_inputs,
                   "module_manifest_sha256": module_sha})
    plan = [map_entry("p52_old_first_order", old),
            map_entry("p52_strongest_p20_selector_global_rerank", heuristic),
            map_entry("p52_frozen_replacement", winner)]
    plan_path = CR / "plans" / f"P52_selector_breadth_{model}.json"
    write_plan(plan_path, plan)
    append_result(plan_path, [str(manifest_set)], "P52_SELECTOR_BREADTH_MAPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("p21", "p42", "p52"), required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    args = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    {"p21": p21, "p42": p42, "p52": p52}[args.stage](args.model)


if __name__ == "__main__":
    main()
