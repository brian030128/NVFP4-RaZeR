"""Materialize all predeclared boundary/corruption maps from frozen tile lists."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from campaign import mapio, runtime
from campaign.boundary_common import (
    MODELS,
    atomic_json,
    load_model_scores,
    mask_from_lists,
    score_summary,
    sha256_file,
    tile_set_sha,
)


def apply_corruption(reference: dict[str, torch.Tensor], removed: dict[str, list[int]],
                     added: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    out = {name: mask.clone() for name, mask in reference.items()}
    for name in reference:
        rem = removed.get(name, [])
        add = added.get(name, [])
        if len(rem) != len(add):
            raise AssertionError(f"{name}: remove/add quota mismatch")
        flat_ref, flat_out = reference[name].flatten(), out[name].flatten()
        if rem:
            r = torch.tensor(rem, dtype=torch.long)
            a = torch.tensor(add, dtype=torch.long)
            if not bool(flat_ref[r].all()):
                raise AssertionError(f"{name}: attempted to remove rejected tile")
            if bool(flat_ref[a].any()):
                raise AssertionError(f"{name}: attempted to add selected tile")
            if len(set(rem) & set(add)):
                raise AssertionError(f"{name}: removal/addition overlap")
            flat_out[r] = False
            flat_out[a] = True
        if int(out[name].sum()) != int(reference[name].sum()):
            raise AssertionError(f"{name}: corruption changed module quota")
    return out


def write_one(*, model: str, policy: str, family: str, masks: dict[str, torch.Tensor], data: dict,
              protocol: dict, protocol_sha: str, source_manifest: str, output: Path,
              definition: dict, definition_sha: str) -> dict:
    header = mapio.build_header(
        protocol_id=protocol["protocol_id"],
        policy={"name": policy, "family": family, "frozen_k": 3,
                "seed_root": protocol["statistics"]["seed_root"]},
        model=data["header"]["model"], type_block=(16, 64), masks=masks,
        weight_shapes=data["shapes"], source_manifest_sha256=source_manifest,
        calibration_manifest_sha256=data["header"]["calibration_manifest_sha256"],
    )
    path = output / model / f"{model}_{policy}.mixfp4map"
    digest, written = mapio.write_map(path, header, masks, provenance={
        "protocol_sha256": protocol_sha,
        "definition_artifact_sha256": definition_sha,
        "source_primary_map_sha256": data["map_sha256"],
        "source_moments_sha256": data["moments_sha256"],
        "anchor_tile_mismatches": data["anchor_mismatches"],
        "definition": definition,
    })
    lists = {name: [int(v) for v in torch.nonzero(mask.flatten(), as_tuple=False).flatten().tolist()]
             for name, mask in masks.items() if bool(mask.any())}
    return {"model": model, "policy": policy, "family": family, "path": written,
            "sha256": digest, "mask_payload_sha256": mapio.payload_sha256(masks),
            "tile_set_sha256": tile_set_sha(lists), "selected_tiles": header["totals"]["selected_tiles"],
            "total_tiles": header["totals"]["total_tiles"], "type_block": [16, 64],
            "score_summary": score_summary(lists, data["stats"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--bands", required=True)
    parser.add_argument("--corruption", required=True)
    args = parser.parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise SystemExit("protocol digest mismatch")
    protocol = json.loads(Path(args.protocol).read_text())
    bands = json.loads(Path(args.bands).read_text())
    corruption = json.loads(Path(args.corruption).read_text())
    if sha256_file(args.bands) != protocol["frozen_inputs"]["band_definitions_sha256"]:
        raise SystemExit("band definitions changed after protocol freeze")
    if sha256_file(args.corruption) != protocol["frozen_inputs"]["corruption_pool_definitions_sha256"]:
        raise SystemExit("corruption definitions changed after protocol freeze")
    launch = json.loads((runtime.run_dir / "launch_record.json").read_text())
    output = runtime.out_dir("derived_maps")
    entries = []
    anchors = {}
    p1_checks = {}
    for model in MODELS:
        data = load_model_scores(model)
        reference = data["reconstructed"]
        if data["anchor_mismatches"] or data["upper_score_mismatches"]:
            raise AssertionError(f"{model}: critical-k anchor mismatch")
        band_rows = []
        for partition in bands["models"][model]["partitions"]:
            band_rows.extend(partition["bands"])
        for definition in band_rows:
            masks = mask_from_lists(reference, definition["tiles"])
            entries.append(write_one(
                model=model, policy=definition["policy"], family="critical_k_boundary",
                masks=masks, data=data, protocol=protocol, protocol_sha=args.protocol_sha256,
                source_manifest=launch["source_manifest_sha256"], output=output,
                definition={"partition": next(p["partition"] for p in bands["models"][model]["partitions"]
                                               if definition in p["bands"]),
                            "side": definition["side"], "band": definition["band"],
                            "tile_set_sha256": definition["tile_set_sha256"]},
                definition_sha=sha256_file(args.bands)))

        anchor_policy = "corruption_p000_anchor"
        anchor_entry = write_one(
            model=model, policy=anchor_policy, family="full_map_corruption", masks=reference,
            data=data, protocol=protocol, protocol_sha=args.protocol_sha256,
            source_manifest=launch["source_manifest_sha256"], output=output,
            definition={"p": 0.0, "source": "exact reconstructed stored conjunction anchor"},
            definition_sha=sha256_file(args.corruption))
        entries.append(anchor_entry)
        anchors[model] = {"source_map": str(data["map_path"]), "source_map_sha256": data["map_sha256"],
                          "source_payload_sha256": mapio.payload_sha256(data["anchor"]),
                          "campaign_anchor_map": anchor_entry["path"],
                          "campaign_anchor_sha256": anchor_entry["sha256"],
                          "campaign_anchor_payload_sha256": anchor_entry["mask_payload_sha256"],
                          "tile_mismatches": data["anchor_mismatches"],
                          "payload_equal": anchor_entry["mask_payload_sha256"] == mapio.payload_sha256(data["anchor"])}

        cdef = corruption["models"][model]
        p1 = []
        for definition in cdef["near_boundary_maps"]:
            masks = apply_corruption(reference, definition["removed"], definition["added"])
            entry = write_one(
                model=model, policy=definition["policy"], family="full_map_corruption",
                masks=masks, data=data, protocol=protocol, protocol_sha=args.protocol_sha256,
                source_manifest=launch["source_manifest_sha256"], output=output,
                definition={key: definition[key] for key in (
                    "kind", "pool", "target_p", "removed_tile_set_sha256", "added_tile_set_sha256",
                    "removed_tiles", "added_tiles", "achieved_global_p")},
                definition_sha=sha256_file(args.corruption))
            entries.append(entry)
            if definition["target_p"] == 1.0:
                p1.append({"pool": definition["pool"], "policy": definition["policy"],
                           "map_sha256": entry["sha256"], "tile_set_sha256": entry["tile_set_sha256"],
                           "mask_payload_sha256": entry["mask_payload_sha256"]})
        for definition in cdef["matched_random_controls"]:
            masks = apply_corruption(reference, definition["removed"], definition["added"])
            entries.append(write_one(
                model=model, policy=definition["policy"], family="full_map_corruption_random_control",
                masks=masks, data=data, protocol=protocol, protocol_sha=args.protocol_sha256,
                source_manifest=launch["source_manifest_sha256"], output=output,
                definition={key: definition[key] for key in (
                    "kind", "replicate", "target_p", "removed_tile_set_sha256", "added_tile_set_sha256",
                    "removed_tiles", "added_tiles", "overlap_with_near_boundary_reservoir")},
                definition_sha=sha256_file(args.corruption)))
        if len({row["map_sha256"] for row in p1}) != 4 or len({row["tile_set_sha256"] for row in p1}) != 4:
            raise AssertionError(f"{model}: p=1 maps are not genuinely distinct")
        p1_checks[model] = p1
        del data

    if len(entries) != 171:
        raise AssertionError(f"expected 171 maps, got {len(entries)}")
    atomic_json(output / "map_manifest.json", entries)
    atomic_json(output / "anchor_checks.json", anchors)
    atomic_json(output / "p1_distinctness_checks.json", {
        "schema": "mixfp4-corruption-p1-map-distinctness/v1", "models": p1_checks,
        "passed": all(len({r["map_sha256"] for r in rows}) == 4 for rows in p1_checks.values())})
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": protocol["protocol_id"], "protocol_freeze_sha256": args.protocol_sha256,
        "source": {"model_id": None, "model_revision": None, "tokenizer_revision": None,
                   "model_class": None, "module_manifest_sha256": None,
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
        "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None},
        "policies": [{"name": e["policy"], "weight_format": "FourOverSix/E0M3 tile mix",
                      "activation_format": "four_over_six_rows", "scale_block": 16,
                      "type_block": e["type_block"], "map_path": e["path"], "map_sha256": e["sha256"],
                      "selected_tiles": e["selected_tiles"], "total_tiles": e["total_tiles"],
                      "map_reloaded_for_evaluation": False} for e in entries],
        "results": {"raw_outputs": [str(path) for path in sorted(output.rglob("*")) if path.is_file()],
                    "summary": {"maps": len(entries), "maps_per_model": 57,
                                "anchor_checks_passed": all(v["payload_equal"] for v in anchors.values()),
                                "p1_distinctness_passed": True}, "uncertainty": {},
                    "attempted_endpoints": ["map_derivation"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"maps": len(entries), "maps_per_model": 57,
                      "p1_hashes": {m: [r["map_sha256"] for r in rows] for m, rows in p1_checks.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()

