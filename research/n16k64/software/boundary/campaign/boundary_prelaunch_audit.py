"""Independent audit of every frozen map, plan, and pre-GPU gate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from campaign import mapio
from campaign.boundary_common import MODELS, atomic_json, load_model_scores, sha256_file


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bands", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    failures: list[str] = []
    require(sha256_file(args.protocol) == args.protocol_sha256, "protocol digest mismatch", failures)
    protocol = json.loads(Path(args.protocol).read_text())
    bands = json.loads(Path(args.bands).read_text())
    corruption = json.loads(Path(args.corruption).read_text())
    entries = json.loads(Path(args.manifest).read_text())
    plans = json.loads(Path(args.plans).read_text())
    require(len(entries) == 171, f"expected 171 maps, got {len(entries)}", failures)
    require(len({(e["model"], e["policy"]) for e in entries}) == len(entries), "duplicate map key", failures)
    require(len({e["path"] for e in entries}) == len(entries), "duplicate map path", failures)
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    loaded: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(dict)
    sidecars = 0
    for entry in entries:
        by_model[entry["model"]][entry["policy"]] = entry
        path = Path(entry["path"])
        require(path.is_file(), f"missing map {path}", failures)
        if not path.is_file():
            continue
        header, masks, digest = mapio.read_map(path, entry["sha256"])
        loaded[entry["model"]][entry["policy"]] = masks
        require(digest == entry["sha256"], f"map digest mismatch {path}", failures)
        require(mapio.payload_sha256(masks) == entry["mask_payload_sha256"],
                f"payload mismatch {entry['model']}/{entry['policy']}", failures)
        require(header["protocol_id"] == protocol["protocol_id"], f"protocol mismatch {path}", failures)
        require(header["policy"]["name"] == entry["policy"], f"policy mismatch {path}", failures)
        side = path.with_suffix(path.suffix + ".provenance.json")
        require(side.is_file(), f"missing map provenance {side}", failures)
        if side.is_file():
            payload = json.loads(side.read_text())
            require(payload["map_sha256"] == digest, f"provenance digest mismatch {side}", failures)
            sidecars += 1

    model_checks = {}
    for model in MODELS:
        data = load_model_scores(model)
        anchor = data["reconstructed"]
        campaign_anchor = loaded[model]["corruption_p000_anchor"]
        mismatch = sum(int((anchor[name] != campaign_anchor[name]).sum()) for name in anchor)
        require(mismatch == 0, f"{model}: p0 anchor mismatch", failures)
        b_rows = bands["models"][model]["partitions"]
        boundary_maps = 0
        for partition in b_rows:
            rows = {(row["side"], row["band"]): row for row in partition["bands"]}
            for side in ("selected", "rejected"):
                union = {name: torch.zeros_like(mask) for name, mask in anchor.items()}
                for band in range(1, bands["selected_B"] + 1):
                    row = rows[(side, band)]
                    masks = loaded[model][row["policy"]]
                    boundary_maps += 1
                    for name in anchor:
                        require(int(masks[name].sum()) == int(row["per_module_counts"][name]),
                                f"{model}/{row['policy']}/{name}: band quota mismatch", failures)
                        require(not bool((union[name] & masks[name]).any()),
                                f"{model}/{partition['partition']}/{side}: band overlap", failures)
                        union[name] |= masks[name]
                if side == "selected":
                    for name in anchor:
                        require(not bool((union[name] & ~anchor[name]).any()),
                                f"{model}/{partition['partition']}: selected band outside anchor", failures)
                else:
                    for name in anchor:
                        require(not bool((union[name] & anchor[name]).any()),
                                f"{model}/{partition['partition']}: rejected band overlaps anchor", failures)
            for band in range(1, bands["selected_B"] + 1):
                selected = loaded[model][rows[("selected", band)]["policy"]]
                rejected = loaded[model][rows[("rejected", band)]["policy"]]
                for name in anchor:
                    require(int(selected[name].sum()) == int(rejected[name].sum()),
                            f"{model}/{partition['partition']}/{band}/{name}: side quota mismatch", failures)
        cdef = corruption["models"][model]
        p1_hashes = []
        for row in cdef["near_boundary_maps"] + cdef["matched_random_controls"]:
            masks = loaded[model][row["policy"]]
            for name in anchor:
                require(int(masks[name].sum()) == int(anchor[name].sum()),
                        f"{model}/{row['policy']}/{name}: corruption quota mismatch", failures)
            if row.get("target_p") == 1.0:
                p1_hashes.append(by_model[model][row["policy"]]["sha256"])
        require(len(p1_hashes) == 4 and len(set(p1_hashes)) == 4, f"{model}: p1 map uniqueness failed", failures)
        for pool in range(1, 5):
            prior = None
            for p in (10, 25, 50, 75, 100):
                policy = f"corruption_near_pool{pool:02d}_p{p:03d}"
                current = loaded[model][policy]
                changed = {name: current[name] != anchor[name] for name in anchor}
                if prior is not None:
                    for name in anchor:
                        require(not bool((prior[name] & ~changed[name]).any()),
                                f"{model}/pool{pool}: nonnested p={p}", failures)
                prior = changed
        model_checks[model] = {"maps": len(by_model[model]), "boundary_maps": boundary_maps,
                               "anchor_tile_mismatches": mismatch, "p1_unique_hashes": p1_hashes,
                               "coverage": json.loads((root / "COVERAGE_GATE.json").read_text())["candidates"][str(bands["selected_B"])][model]["selected_common_support_coverage"],
                               "tiles_per_band": bands["models"][model]["tiles_per_global_band"]}

    for record in plans["plans"]:
        plan_path = Path(record["path"])
        require(sha256_file(plan_path) == record["sha256"], f"plan digest mismatch {record['model']}", failures)
        plan = json.loads(plan_path.read_text())
        require(len(plan) == 56, f"{record['model']}: plan count {len(plan)}", failures)
        require("corruption_p000_anchor" not in {row["name"] for row in plan},
                f"{record['model']}: p0 should be reused, not rerun", failures)
        for item in plan:
            entry = by_model[record["model"]][item["name"]]
            require(item["map_sha256"] == entry["sha256"],
                    f"{record['model']}/{item['name']}: plan/map digest mismatch", failures)

    code_dir = Path(__file__).resolve().parent
    for name, expected in protocol["code_hashes_at_freeze"].items():
        path = code_dir / name
        if name.startswith("test_"):
            path = code_dir.parent / "tests" / name
        require(path.is_file(), f"missing frozen code {name}", failures)
        if path.is_file():
            require(sha256_file(path) == expected, f"frozen code changed: {name}", failures)

    audit = {"schema": "mixfp4-boundary-corruption-prelaunch-audit/v1", "outcome_blind": True,
             "passed": not failures, "failures": failures, "protocol_sha256": args.protocol_sha256,
             "maps_verified": len(entries), "map_provenance_sidecars_verified": sidecars,
             "plans_verified": len(plans["plans"]), "models": model_checks,
             "gates": {name: {"sha256": sha256_file(root / name),
                               "passed": json.loads((root / name).read_text()).get("passed")}
                       for name in ("INPUT_GATE.json", "COVERAGE_GATE.json", "CORRUPTION_UNIQUENESS_GATE.json",
                                    "POWER_ANALYSIS.json", "ENDPOINT_PROMOTION.json")}}
    atomic_json(args.out, audit)

    freeze_paths = [Path(args.protocol), root / "INPUT_PROVENANCE.json", root / "INPUT_GATE.json",
                    root / "CRITICAL_K_SUMMARY.json", root / "COVERAGE_GATE.json",
                    root / "CORRUPTION_UNIQUENESS_GATE.json", root / "POWER_ANALYSIS.json",
                    root / "ENDPOINT_PROMOTION.json", Path(args.bands), Path(args.corruption),
                    Path(args.manifest), Path(args.plans), root / "RUN_MATRIX.csv", Path(args.out)]
    for entry in entries:
        freeze_paths.extend([Path(entry["path"]), Path(entry["path"]).with_suffix(".mixfp4map.provenance.json")])
    for record in plans["plans"]:
        freeze_paths.append(Path(record["path"]))
    unique = sorted(set(path.resolve() for path in freeze_paths), key=str)
    lines = []
    for path in unique:
        require(path.is_file(), f"freeze artifact missing {path}", failures)
        if path.is_file():
            try:
                relative = path.relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = str(path)
            lines.append(f"{sha256_file(path)}  {relative}\n")
    if failures:
        audit["passed"] = False
        audit["failures"] = failures
        atomic_json(args.out, audit)
        raise SystemExit(2)
    Path(args.freeze_manifest).write_text("".join(lines))
    print(json.dumps({"passed": True, "maps": len(entries), "freeze_entries": len(lines),
                      "freeze_manifest_sha256": sha256_file(args.freeze_manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
