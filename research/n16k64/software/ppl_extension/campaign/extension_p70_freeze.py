"""Create the immutable global-winner freeze before held-out quality is opened."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import mapio as MIO
from campaign import runtime
from campaign.extension_heldout_maps import MODELS, latest_complete, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"


def one_map(model):
    run = latest_complete(f"P70_maps_{model}")
    report_path = run / "derived_maps/derivation_report.json"
    manifest_path = run / "derived_maps/map_manifest.json"
    report, manifest = load(report_path), load(manifest_path)
    if report.get("status") != "complete" or report.get("model") != model:
        raise RuntimeError(f"unaccepted heldout map derivation: {model}")
    rows = [row for row in manifest if row["policy"] == "p70_frozen_winner"]
    if len(rows) != 1:
        raise RuntimeError(f"missing unique heldout winner map: {model}")
    row = rows[0]
    header, _, digest = MIO.read_map(row["path"], row["sha256"])
    if digest != row["sha256"] or header["type_block"] != [16, 64]:
        raise RuntimeError(f"heldout winner map failed exact reload: {model}")
    plan_path = CR / "plans" / f"P71_heldout_ppl_{model}.json"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    return {"model": model, "map_run_id": run.name,
            "map_derivation_report": str(report_path),
            "map_derivation_report_sha256": runtime.sha256_file(report_path),
            "map_manifest": str(manifest_path),
            "map_manifest_sha256": runtime.sha256_file(manifest_path),
            "winner_map_path": row["path"], "winner_map_sha256": digest,
            "selected_tiles": row["selected_tiles"],
            "selected_weights": row["selected_weights"],
            "module_manifest_sha256": report["module_manifest_sha256"],
            "calibration_manifest_set": report["calibration_manifest_set"],
            "P71_plan_path": str(plan_path),
            "P71_plan_sha256": runtime.sha256_file(plan_path),
            "plan_policy_order": [x["name"] for x in load(plan_path)]}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    paths = {
        "P40": CR / "provenance/P40_GLOBAL_K_SELECTION.json",
        "P41": CR / "provenance/P41_AGGREGATION_SELECTION.json",
        "P50": CR / "provenance/P50_SELECTOR_SELECTION.json",
        "P51": CR / "provenance/P51_G6_DECISION.json",
        "P60": CR / "provenance/P60_SELECTION.json",
        "P61": CR / "provenance/P61_SELECTION.json",
        "P63_gate": CR / "provenance/P63_COMBINATION_GATE.json",
        "operationalization": CR / "provenance/P70_P75_WINNER_HELDOUT_OPERATIONALIZATION.json",
        "reservation": CR / "provenance/HELDOUT_PANEL_RESERVATION.json",
    }
    values = {key: load(path) for key, path in paths.items()}
    p40, p41, p50, p60 = values["P40"], values["P41"], values["P50"], values["P60"]
    selector = ("first_order_ce_kl" if p50["fallback_to_old"] else
                p50["selected_policy"].removeprefix("p50_"))
    heldout = [one_map(model) for model in MODELS]
    freeze = {
        "schema_version": "1.0", "status": "HASH_LOCKED_BEFORE_HELDOUT_QUALITY",
        "protocol_sha256": PROTOCOL,
        "relationship_to_parent": "post-hoc robustness extension; heldout endpoints below are prospective after this freeze",
        "global_configuration": {
            "tile_shape": [16, 64], "scale_block": 16,
            "weight_formats": ["E2M1", "E0M3"], "element_bits": 4,
            "global_k": float(p40["global_k"]),
            "aggregation": p41["aggregation"], "density_form": p41["form"],
            "selector": selector, "selector_fallback_to_old": bool(p50["fallback_to_old"]),
            "weight_scale_multiplier": float(p60["selected_multiplier"]),
            "activation_kind": "four_over_six_rows",
            "quantized_scope": "all text-model torch.nn.Linear weights except output head; embeddings, norms and KV cache excluded",
            "fake_quant_path": "BF16-dequantized software W4A4 matmul",
            "P61_excluded": values["P61"]["selected_label"] != "baseline_absmax",
            "P61_exclusion_reason": "Any selected P61 candidate adds disclosed unfused inference-time work and is G7-incompatible.",
            "calibration_size_merge": "none; P31/P32 are separately reported robustness evidence"
        },
        "heldout_models": heldout,
        "comparators": ["four_over_six", "nvfp4", "all_e0m3", "n8_k2", "n16_k2",
                        "n16_k3", "razer_wonly_shared_act", "razer_native_rows",
                        "nover6_wonly_shared_act", "nover6_native_rows"],
        "analysis": {"P71_family": "four winner-minus-FourOverSix endpoints",
                     "bootstrap_replicates": 10000, "correction": "Holm",
                     "material_regression": 0.004987541511038968},
        "source_and_decision_inputs": {
            key: {"path": str(path), "sha256": runtime.sha256_file(path)}
            for key, path in paths.items()},
        "candidate_changes_after_freeze": "forbidden",
        "native_or_performance_claims": "forbidden",
    }
    path = CR / "freeze/P70_GLOBAL_WINNER.json"
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, freeze); path.chmod(0o444)
    sidecar = CR / "freeze/P70_GLOBAL_WINNER.sha256"
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(runtime.sha256_file(path) + "  P70_GLOBAL_WINNER.json\n"); sidecar.chmod(0o444)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P70-global-winner-freeze",
                   "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "multiple", "model_class": "winner_freeze",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per heldout model",
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": {"passed": True, "models": list(MODELS),
                                   "documents_per_model": 640}},
        "policies": [],
        "results": {"raw_outputs": [str(path), str(sidecar)]
                    + [row["P71_plan_path"] for row in heldout],
                    "summary": freeze["global_configuration"], "uncertainty": {},
                    "attempted_endpoints": ["P70_WINNER_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "winner_sha256": runtime.sha256_file(path),
                      "models": list(MODELS)}, sort_keys=True))


if __name__ == "__main__":
    main()
