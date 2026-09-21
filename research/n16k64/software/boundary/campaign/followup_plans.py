"""Materialize immutable GPU evaluation plans from the follow-up map manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOSE = [f"dose_group_only_bin{i:02d}" for i in range(1, 9)] + [f"dose_full_minus_bin{i:02d}" for i in range(1, 9)]
OBJECTIVE = ["objective_ce_natural", "objective_kl_natural", "objective_ce_matched_k",
             "objective_kl_matched_k", "objective_random_matched_k"]
MISTRAL_VETO = ["veto_full_plus_ce_vetoed_kl_approved", "veto_full_plus_ce_vetoed_kl_matched_random"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    args = ap.parse_args()
    entries = json.loads(Path(args.manifest).read_text())
    by_key = {(entry["model"], entry["policy"]): entry for entry in entries}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for model in ("llama8b", "qwen4b", "mistral7b"):
        names = DOSE + OBJECTIVE + (MISTRAL_VETO if model == "mistral7b" else [])
        plan = []
        for name in names:
            entry = by_key[(model, name)]
            plan.append({
                "name": name, "kind": "map", "map_path": entry["path"], "map_sha256": entry["sha256"],
                "map_policy": name, "type_block": entry["type_block"],
                "expected_total_tiles": entry["total_tiles"], "protocol_id": "aligned-followup",
            })
        path = out / f"{model}_followup_full.json"
        path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        records.append({"model": model, "role": "validation" if model == "mistral7b" else "development",
                        "path": str(path), "sha256": sha(path), "policies": len(plan), "names": names,
                        "baseline_full_reused_not_evaluated": True})
    manifest = out / "PLAN_MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "mixfp4-followup-plan-manifest/v1",
                                    "protocol_sha256": args.protocol_sha256, "plans": records},
                                   indent=1, sort_keys=True) + "\n")
    print(json.dumps({"path": str(manifest), "sha256": sha(manifest),
                      "policy_counts": {r["model"]: r["policies"] for r in records}}, sort_keys=True))


if __name__ == "__main__":
    main()
