"""Build immutable GPU plans and the expanded required-arm matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from campaign.boundary_common import DOMAINS, MODELS, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-matrix", required=True)
    args = parser.parse_args()
    entries = json.loads(Path(args.manifest).read_text())
    by_model = {model: [] for model in MODELS}
    for entry in entries:
        by_model[entry["model"]].append(entry)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    matrix_rows = []
    for model in MODELS:
        evaluate = [entry for entry in by_model[model] if entry["policy"] != "corruption_p000_anchor"]
        evaluate.sort(key=lambda entry: entry["policy"])
        if len(evaluate) != 56:
            raise AssertionError(f"{model}: expected 56 new policies, got {len(evaluate)}")
        plan = [{"name": entry["policy"], "kind": "map", "map_path": entry["path"],
                 "map_sha256": entry["sha256"], "map_policy": entry["policy"],
                 "type_block": entry["type_block"], "expected_total_tiles": entry["total_tiles"],
                 "protocol_id": "boundary-corruption"} for entry in evaluate]
        path = output / f"{model}_boundary_corruption_full.json"
        atomic_json(path, plan)
        records.append({"model": model, "role": "validation" if model == "mistral7b" else "development",
                        "path": str(path), "sha256": sha256_file(path), "policies": len(plan),
                        "names": [row["name"] for row in plan],
                        "reused_not_evaluated": ["four_over_six", "corruption_p000_anchor/full_conjunction"]})
        for domain in DOMAINS:
            matrix_rows.extend({"experiment": "boundary" if item["name"].startswith("boundary_") else "corruption",
                                "model": model, "corpus": domain, "policy": item["name"],
                                "required": True, "evaluation": "new_gpu_full",
                                "map_sha256": item["map_sha256"]} for item in plan)
            matrix_rows.extend([
                {"experiment": "anchor", "model": model, "corpus": domain, "policy": "four_over_six",
                 "required": True, "evaluation": "verified_reuse", "map_sha256": "non-map-baseline"},
                {"experiment": "anchor", "model": model, "corpus": domain, "policy": "corruption_p000_anchor",
                 "required": True, "evaluation": "verified_reuse", "map_sha256":
                 next(e["sha256"] for e in by_model[model] if e["policy"] == "corruption_p000_anchor")},
            ])
    manifest = {"schema": "mixfp4-boundary-corruption-plan-manifest/v1",
                "protocol_sha256": args.protocol_sha256, "plans": records,
                "new_policies_per_model": 56, "required_gpu_policy_evaluations": 168}
    atomic_json(output / "PLAN_MANIFEST.json", manifest)
    with open(args.run_matrix, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(json.dumps({"plans": len(records), "policies_per_model": 56,
                      "matrix_rows": len(matrix_rows), "plan_manifest_sha256": sha256_file(output / "PLAN_MANIFEST.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()

