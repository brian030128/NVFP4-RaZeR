"""Build immutable P60 and P61 single-axis plans from the frozen selector maps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_p50_maps import CR, MODELS, PROTOCOL, load


SCALE = (("s1p0", 1.0), ("s0p95", 0.95), ("s1p05", 1.05))
ACTIVATIONS = (
    ("baseline_absmax", "four_over_six_rows"),
    ("mse_grid", "four_over_six_rows_mse_grid"),
    ("percentile_999", "four_over_six_rows_percentile_999"),
    ("percentile_9999", "four_over_six_rows_percentile_9999"),
    ("percentile_100_anchor", "four_over_six_rows_percentile_100"),
)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value); path.chmod(0o444)


def map_entry(name, row, **overrides):
    header, _, digest = MIO.read_map(row["map_path"], row["map_sha256"])
    out = {"name": name, "kind": "map", "map_policy": row["map_policy"],
           "map_path": row["map_path"], "map_sha256": digest,
           "type_block": row["type_block"], "protocol_id": header["protocol_id"],
           "expected_total_tiles": row["expected_total_tiles"]}
    out.update(overrides)
    return out


def build(stage, model, winner):
    row = winner["selected_maps"][model]
    plan = []
    if stage == "p60":
        for label, multiplier in SCALE:
            plan.append({"name": f"p60_fos_{label}", "kind": "four_over_six",
                         "weight_scale_multiplier": multiplier})
        for label, multiplier in SCALE:
            plan.append(map_entry(f"p60_winner_{label}", row, weight_scale_multiplier=multiplier))
    else:
        for prefix, base in (("fos", {"kind": "four_over_six"}), ("winner", None)):
            for label, activation in ACTIVATIONS:
                name = f"p61_{prefix}_{label}"
                if base is None:
                    plan.append(map_entry(name, row, activation_kind=activation))
                else:
                    plan.append({"name": name, **base, "activation_kind": activation})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("p60", "p61"))
    args = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    g6_path = CR / "provenance/P51_G6_DECISION.json"
    winner_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    operational_path = CR / "provenance/P60_P61_ANALYSIS_OPERATIONALIZATION.json"
    for path in (g6_path, winner_path, operational_path):
        if not path.exists():
            raise FileNotFoundError(path)
    winner = load(winner_path)
    rows = []
    for model in MODELS:
        plan = build(args.stage, model, winner)
        path = CR / "plans" / f"{args.stage.upper()}_{'weight_scale' if args.stage == 'p60' else 'activation'}_{model}.json"
        write_new(path, plan)
        rows.append({"model": model, "path": str(path), "sha256": runtime.sha256_file(path),
                     "entries": len(plan), "policy_order": [row["name"] for row in plan],
                     "winner_map_sha256": winner["selected_maps"][model]["map_sha256"]})
    manifest = {"schema_version": "1.0", "status": f"LOCKED_BEFORE_{args.stage.upper()}_QUALITY",
                "stage": args.stage.upper(), "protocol_sha256": PROTOCOL,
                "quantizer_spec_sha256": runtime.sha256_file(CR / "provenance/P60_P62_QUANTIZER_SPEC.json"),
                "operationalization_sha256": runtime.sha256_file(operational_path),
                "selector_selection_sha256": runtime.sha256_file(winner_path),
                "G6_decision_sha256": runtime.sha256_file(g6_path), "plans": rows}
    manifest_path = CR / "provenance" / f"{args.stage.upper()}_PLAN_MANIFEST.json"
    write_new(manifest_path, manifest)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{args.stage}-plan-freeze", "model_revision": runtime.sha256_file(manifest_path),
                   "tokenizer_revision": "multiple", "model_class": "plan_builder",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None},
        "policies": [], "results": {"raw_outputs": [str(manifest_path)] + [row["path"] for row in rows],
            "summary": {"stage": args.stage, "plans": len(rows), "entries": sum(row["entries"] for row in rows)},
            "uncertainty": {}, "attempted_endpoints": [f"{args.stage.upper()}_PLAN_FREEZE"],
            "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "stage": args.stage, "plans": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
