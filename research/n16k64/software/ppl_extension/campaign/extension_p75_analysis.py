"""Analyze exact-map A6000/Ada software-quality portability anchors."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign.extension_heldout_maps import latest_complete, load
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("llama8b", "granite8b")
DEVICES = ("a6000", "ada")


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    freeze_path = CR / "freeze/P75_PORTABILITY_MAPS.json"
    freeze = load(freeze_path)
    reports, runs, checks = {}, {}, {}
    for model in MODELS:
        expected_map = next(row["map_sha256"] for row in freeze["models"] if row["model"] == model)
        for device in DEVICES:
            run = latest_complete(f"P75_{device}_{model}")
            report = load(run / "ppl/ppl_report.json")
            install = next(row for row in report["installs"] if row["name"] == "frozen_winner")
            launch = load(run / "launch_record.json")
            if (report.get("status") != "complete"
                    or install["map_sha256"] != expected_map or len(launch["leased_uuids"]) != 1):
                raise RuntimeError(f"unaccepted P75 anchor: {device}/{model}")
            validate_policy_contract(report, ("four_over_six", "frozen_winner"),
                                     f"P75 {device}/{model}")
            requested = launch["requested"]["model"]
            if requested != device:
                raise RuntimeError(f"P75 device assignment mismatch: {device}/{model}")
            reports[(model, device)], runs[(model, device)] = report, run
        checks[model] = {}
        for domain in ("wiki", "c4"):
            a, at = arrays(reports[(model, "a6000")], "four_over_six", domain)
            w, wt = arrays(reports[(model, "a6000")], "frozen_winner", domain)
            aa, aat = arrays(reports[(model, "ada")], "four_over_six", domain)
            wa, wat = arrays(reports[(model, "ada")], "frozen_winner", domain)
            if not (np.array_equal(at, wt) and np.array_equal(at, aat) and np.array_equal(at, wat)):
                raise RuntimeError(f"P75 token pairing failed: {model}/{domain}")
            e0 = float((w - a).sum() / at.sum()); e1 = float((wa - aa).sum() / at.sum())
            ppls = {"a6000_four_over_six": reports[(model, "a6000")]["evaluation"]["four_over_six"][domain]["ppl"],
                    "a6000_winner": reports[(model, "a6000")]["evaluation"]["frozen_winner"][domain]["ppl"],
                    "ada_four_over_six": reports[(model, "ada")]["evaluation"]["four_over_six"][domain]["ppl"],
                    "ada_winner": reports[(model, "ada")]["evaluation"]["frozen_winner"][domain]["ppl"]}
            rel_fos = abs(ppls["ada_four_over_six"] / ppls["a6000_four_over_six"] - 1)
            rel_win = abs(ppls["ada_winner"] / ppls["a6000_winner"] - 1)
            effect_tol = max(0.25 * abs(e0), 0.002)
            row = {"aggregate_ppl": ppls, "relative_difference": {"four_over_six": rel_fos,
                                                                     "frozen_winner": rel_win},
                   "dlogppl_effect": {"a6000": e0, "ada": e1},
                   "effect_sign_equal": (e0 == 0 and e1 == 0) or (e0 < 0) == (e1 < 0),
                   "effect_absolute_difference": abs(e1 - e0),
                   "effect_tolerance": effect_tol}
            row["passed"] = (rel_fos <= 0.005 and rel_win <= 0.005
                             and row["effect_sign_equal"]
                             and row["effect_absolute_difference"] <= effect_tol)
            checks[model][domain] = row
    passed = all(row["passed"] for model in checks.values() for row in model.values())
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "portability_map_freeze_sha256": runtime.sha256_file(freeze_path),
                "run_inputs": {f"{model}_{device}": {"run_id": runs[(model, device)].name,
                    "report_sha256": runtime.sha256_file(runs[(model, device)] / "ppl/ppl_report.json")}
                    for model in MODELS for device in DEVICES},
                "checks": checks, "passed": passed,
                "per_window_identity_claim": "not made",
                "scope": "software fake-quant quality portability only",
                "native_performance_claim": "forbidden"}
    out = runtime.out_dir("analysis_p75")
    path = out / "P75_PORTABILITY_ANALYSIS.json"; runtime.atomic_json(path, analysis)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P75-portability-analysis",
                   "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": "per frozen map",
                 "evaluation_manifest_sha256": "per model exact 16+16 windows",
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path)], "summary": {"passed": passed},
                    "uncertainty": {"tolerance_only": True},
                    "attempted_endpoints": ["P75_PORTABILITY"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
