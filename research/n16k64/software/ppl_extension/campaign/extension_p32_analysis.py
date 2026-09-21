"""Analyze an authorized P32 256+256 calibration-size extension against P31."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.extension_p31_analysis import arrays, identity, latest_complete, load


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b")


def contrast(new, prior, policy, meta, model, domain):
    a, ta = arrays(new, policy, domain)
    b, tb = arrays(prior, policy, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P32 pairing failed for {model}/{domain}/{policy}")
    label = f"P32:{model}:{domain}:{policy}:256minus128"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, policy=policy,
               comparator_calibration="nested_128_plus_128",
               method_calibration="nested_256_plus_256", stream_label=label,
               raw_ppl_method=new["evaluation"][policy][domain]["ppl"],
               raw_ppl_comparator=prior["evaluation"][policy][domain]["ppl"])
    return row


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    gate_path = CR / "provenance/P31_G5_DECISION.json"
    gate = load(gate_path)
    if not gate.get("continue_to_p32"):
        raise RuntimeError("P32 analysis cannot run when G5 did not authorize P32")
    out = runtime.out_dir("analysis_p32")
    rows, raw, inputs = [], [], {}
    for model in MODELS:
        run = latest_complete(f"P32_quality_{model}")
        prior_run = latest_complete(f"P31_quality_{model}")
        new = load(run / "ppl/ppl_report.json")
        prior = load(prior_run / "ppl/ppl_report.json")
        if new.get("status") != "complete" or not new.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P32 PPL report: {model}")
        ident = identity(run, new, prior_run, prior)
        if not ident["passed"]:
            raise RuntimeError(f"P32/P31 evaluation identity failed: {model}")
        inputs[model] = {"run_id": run.name,
                         "report_sha256": runtime.sha256_file(run / "ppl/ppl_report.json"),
                         "P31_run_id": prior_run.name,
                         "P31_report_sha256": runtime.sha256_file(prior_run / "ppl/ppl_report.json"),
                         "pairing_identity": ident}
        for domain in ("wiki", "c4"):
            meta = load(run / f"ppl/windows_{domain}.json")
            for policy in ("n8_k2", "n16_k2"):
                rows.append(contrast(new, prior, policy, meta, model, domain))
            for calibration, report, rid in (("p32", new, run.name),
                                              ("p31", prior, prior_run.name)):
                for policy in ("four_over_six", "n8_k2", "n16_k2"):
                    raw.append({"model": model, "corpus": domain,
                                "calibration": calibration, "policy": policy,
                                "raw_ppl": report["evaluation"][policy][domain]["ppl"],
                                "mean_nll": report["evaluation"][policy][domain]["mean_nll"],
                                "run_id": rid})
    for policy in ("n8_k2", "n16_k2"):
        family = [row for row in rows if row["policy"] == policy]
        adjusted, rejected = S.holm([row["p_two_sided"] for row in family])
        for row, p, reject in zip(family, adjusted, rejected):
            row["p_holm"] = p
            row["holm_reject_0_05"] = reject
    n16 = [row for row in rows if row["policy"] == "n16_k2"]
    summary = {"per_model_cross_corpus_mean": {
                    model: float(np.mean([row["estimate"] for row in n16 if row["model"] == model]))
                    for model in MODELS},
               "median_six_cells": float(np.median([row["estimate"] for row in n16])),
               "worst_cell": float(max(row["estimate"] for row in n16))}
    analysis = {"schema_version": "1.0", "status": "complete",
                "protocol_sha256": PROTOCOL,
                "G5_decision_sha256": runtime.sha256_file(gate_path),
                "incremental_spec_sha256": runtime.sha256_file(
                    CR / "provenance/P31_P32_INCREMENTAL_SPEC.json"),
                "bootstrap_replicates": B, "run_inputs": inputs,
                "contrasts": {p: [row for row in rows if row["policy"] == p]
                              for p in ("n16_k2", "n8_k2")},
                "summary": summary, "development_only": True}
    analysis_path = out / "P32_CALIBRATION_SIZE_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P32_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P32-development-analysis",
                   "model_revision": runtime.sha256_file(analysis_path),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path)],
                    "summary": summary, "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": ["P32_CAL_SIZE_256"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
