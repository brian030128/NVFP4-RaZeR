"""Independent outcome-blind audit of frozen follow-up maps and GPU plans."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from campaign.mapio import payload_sha256, read_map


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    bins_path = Path(args.bins)
    selections_path = Path(args.selections)
    plans_path = Path(args.plans)
    protocol_path = Path(args.protocol)
    failures: list[str] = []
    checks: list[dict] = []

    require(sha(protocol_path) == args.protocol_sha256, "protocol digest differs from frozen digest", failures)
    protocol = json.loads(protocol_path.read_text())
    bins = json.loads(bins_path.read_text())
    selections = json.loads(selections_path.read_text())
    entries = json.loads(manifest_path.read_text())
    plan_manifest = json.loads(plans_path.read_text())
    require(bins["protocol_sha256"] == args.protocol_sha256, "bin protocol digest mismatch", failures)
    require(plan_manifest["protocol_sha256"] == args.protocol_sha256, "plan protocol digest mismatch", failures)
    require(len(entries) == 68, f"expected 68 maps, got {len(entries)}", failures)

    code_dir = Path(__file__).resolve().parent
    for name, expected in protocol["code_hashes_at_freeze"].items():
        candidate = code_dir / name
        if name.startswith("test_"):
            candidate = code_dir.parent / "tests" / name
        require(candidate.is_file(), f"frozen code missing: {candidate}", failures)
        if candidate.is_file():
            require(sha(candidate) == expected, f"frozen code hash mismatch: {name}", failures)

    keys = [(e["model"], e["policy"]) for e in entries]
    require(len(keys) == len(set(keys)), "duplicate model/policy map entries", failures)
    require(len({e["path"] for e in entries}) == len(entries), "duplicate map paths", failures)
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for entry in entries:
        by_model[entry["model"]][entry["policy"]] = entry
    require(set(by_model) == {"llama8b", "qwen4b", "mistral7b"}, "unexpected model set", failures)

    verified_maps = 0
    verified_sidecars = 0
    model_summaries = {}
    for model in ("llama8b", "qwen4b", "mistral7b"):
        loaded: dict[str, tuple[dict, dict[str, torch.Tensor]]] = {}
        for policy, entry in by_model[model].items():
            path = Path(entry["path"])
            require(path.is_file(), f"missing map: {path}", failures)
            if not path.is_file():
                continue
            header, masks, digest = read_map(path, entry["sha256"])
            verified_maps += 1
            require(header["totals"]["selected_tiles"] == entry["selected_tiles"],
                    f"selected total mismatch: {model}/{policy}", failures)
            require(header["totals"]["total_tiles"] == entry["total_tiles"],
                    f"total tile mismatch: {model}/{policy}", failures)
            require(payload_sha256(masks) == entry["mask_payload_sha256"],
                    f"payload digest mismatch: {model}/{policy}", failures)
            require(header["policy"]["name"] == policy, f"header policy mismatch: {model}/{policy}", failures)
            require(header["protocol_id"] == "aligned-followup", f"protocol mismatch: {model}/{policy}", failures)
            side = path.with_suffix(path.suffix + ".provenance.json")
            require(side.is_file(), f"missing provenance: {side}", failures)
            if side.is_file():
                side_data = json.loads(side.read_text())
                require(side_data["map_sha256"] == digest, f"sidecar map digest mismatch: {model}/{policy}", failures)
                require(side_data["mask_payload_sha256"] == entry["mask_payload_sha256"],
                        f"sidecar payload digest mismatch: {model}/{policy}", failures)
                verified_sidecars += 1
            loaded[policy] = (header, masks)

        conjunction = loaded["objective_conjunction"][1]
        bin_masks = []
        for idx in range(1, 9):
            group_name = f"dose_group_only_bin{idx:02d}"
            minus_name = f"dose_full_minus_bin{idx:02d}"
            group = loaded[group_name][1]
            minus = loaded[minus_name][1]
            expected_counts = bins["models"][model]["definitions"][idx - 1]["per_stratum_counts"]
            for module in conjunction:
                require(int(group[module].sum()) == int(expected_counts[module]),
                        f"bin count mismatch: {model}/{group_name}/{module}", failures)
                require(not bool((group[module] & ~conjunction[module]).any()),
                        f"bin outside conjunction: {model}/{group_name}/{module}", failures)
                require(torch.equal(minus[module], conjunction[module] & ~group[module]),
                        f"full-minus identity failed: {model}/{minus_name}/{module}", failures)
            bin_masks.append(group)
        for module in conjunction:
            union = torch.zeros_like(conjunction[module])
            for group in bin_masks:
                require(not bool((union & group[module]).any()), f"dose bins overlap: {model}/{module}", failures)
                union |= group[module]
            remainder = conjunction[module] & ~union
            declared_remainder = sum(1 for tile in bins["models"][model]["excluded_remainder_tiles"]
                                     if tile["module"] == module)
            require(int(remainder.sum()) == declared_remainder,
                    f"remainder count mismatch: {model}/{module}", failures)
        bdef = bins["models"][model]
        require(sum(e["selected_tiles"] for p, e in by_model[model].items()
                    if p.startswith("dose_group_only_")) == bdef["retained_union_tiles"],
                f"retained union total mismatch: {model}", failures)
        require(bdef["retained_union_tiles"] + bdef["excluded_remainder_count"] ==
                bdef["selected_tiles_before_remainder"], f"dose accounting mismatch: {model}", failures)
        margins = [d["score_summary"]["mean_combined_election_margin"] for d in bdef["definitions"]]
        require(all(a >= b for a, b in zip(margins, margins[1:])), f"dose margin order failed: {model}", failures)

        ce_nat = loaded["objective_ce_natural"][1]
        kl_nat = loaded["objective_kl_natural"][1]
        ce_match = loaded["objective_ce_matched_k"][1]
        kl_match = loaded["objective_kl_matched_k"][1]
        random_match = loaded["objective_random_matched_k"][1]
        for module in conjunction:
            quota = int(conjunction[module].sum())
            require(torch.equal(conjunction[module], ce_nat[module] & kl_nat[module]),
                    f"natural conjunction identity failed: {model}/{module}", failures)
            require(not bool((ce_match[module] & ~ce_nat[module]).any()),
                    f"CE matched map outside CE pass support: {model}/{module}", failures)
            require(not bool((kl_match[module] & ~kl_nat[module]).any()),
                    f"KL matched map outside KL pass support: {model}/{module}", failures)
            require(not bool((random_match[module] & ~(ce_nat[module] | kl_nat[module])).any()),
                    f"random map outside union pass support: {model}/{module}", failures)
            require(int(ce_match[module].sum()) == quota, f"CE matched quota mismatch: {model}/{module}", failures)
            require(int(kl_match[module].sum()) == quota, f"KL matched quota mismatch: {model}/{module}", failures)
            require(int(random_match[module].sum()) == quota, f"random matched quota mismatch: {model}/{module}", failures)

        veto_summary = None
        if model == "mistral7b":
            actual = loaded["veto_full_plus_ce_vetoed_kl_approved"][1]
            random = loaded["veto_full_plus_ce_vetoed_kl_matched_random"][1]
            actual_total = random_total = 0
            for module in conjunction:
                require(not bool((conjunction[module] & ~actual[module]).any()), f"actual omits conjunction: {module}", failures)
                require(not bool((conjunction[module] & ~random[module]).any()), f"random omits conjunction: {module}", failures)
                actual_add = actual[module] & ~conjunction[module]
                random_add = random[module] & ~conjunction[module]
                require(not bool((actual_add & random_add).any()), f"Mistral add-backs overlap: {module}", failures)
                require(int(actual_add.sum()) == int(random_add.sum()), f"Mistral module quota mismatch: {module}", failures)
                actual_total += int(actual_add.sum())
                random_total += int(random_add.sum())
            veto = selections["models"][model]["veto"]
            require(actual_total == veto["actual_addback_tiles"], "Mistral actual add-back total mismatch", failures)
            require(random_total == veto["matched_random_addback_tiles"], "Mistral random add-back total mismatch", failures)
            require(veto["invariants"]["module_and_margin_bin_counts_exact"], "Mistral margin-bin match not declared exact", failures)
            require(veto["invariants"]["prior_frozen_masks_reproduced_exactly"], "Mistral prior mask reproduction failed", failures)
            veto_summary = {"actual_addback_tiles": actual_total, "matched_random_addback_tiles": random_total,
                            "matching_cells": len(veto["matching_cells"]),
                            "support_reductions": len(veto["support_reductions"])}

        model_summaries[model] = {
            "maps": len(loaded),
            "conjunction_tiles": int(sum(int(mask.sum()) for mask in conjunction.values())),
            "dose_tiles_per_bin": [by_model[model][f"dose_group_only_bin{i:02d}"]["selected_tiles"] for i in range(1, 9)],
            "excluded_remainder_tiles": bdef["excluded_remainder_count"],
            "objective_selected_counts": selections["models"][model]["objective"]["selected_counts"],
            "mistral_veto": veto_summary,
        }

    plan_counts = {}
    for record in plan_manifest["plans"]:
        plan = json.loads(Path(record["path"]).read_text())
        require(sha(Path(record["path"])) == record["sha256"], f"plan digest mismatch: {record['model']}", failures)
        require(len(plan) == record["policies"], f"plan policy count mismatch: {record['model']}", failures)
        expected_names = set(record["names"])
        require({p["name"] for p in plan} == expected_names, f"plan policy names mismatch: {record['model']}", failures)
        require("objective_conjunction" not in expected_names, f"plan redundantly evaluates conjunction: {record['model']}", failures)
        for item in plan:
            entry = by_model[record["model"]][item["name"]]
            require(item["map_sha256"] == entry["sha256"],
                    f"plan/manifest digest mismatch: {record['model']}/{item['name']}", failures)
        plan_counts[record["model"]] = len(plan)

    audit = {
        "schema": "mixfp4-followup-prelaunch-audit/v1",
        "outcome_blind": True,
        "passed": not failures,
        "failures": failures,
        "protocol_sha256": args.protocol_sha256,
        "inputs": {
            "map_manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
            "bin_definitions": {"path": str(bins_path), "sha256": sha(bins_path)},
            "selection_definitions": {"path": str(selections_path), "sha256": sha(selections_path)},
            "plan_manifest": {"path": str(plans_path), "sha256": sha(plans_path)},
        },
        "checks": {
            "maps_verified": verified_maps,
            "provenance_sidecars_verified": verified_sidecars,
            "unique_map_keys": len(set(keys)),
            "frozen_code_hashes_verified": len(protocol["code_hashes_at_freeze"]),
            "plan_policy_counts": plan_counts,
            "dose_composition_and_full_minus_identities": not any("dose" in x or "bin" in x or "remainder" in x for x in failures),
            "objective_support_and_matched_quota_identities": not any("matched" in x or "conjunction identity" in x for x in failures),
            "mistral_addback_module_and_declared_margin_bin_match": not any("Mistral" in x for x in failures),
        },
        "models": model_summaries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"path": str(out), "sha256": sha(out), "passed": audit["passed"],
                      "failures": len(failures)}, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
