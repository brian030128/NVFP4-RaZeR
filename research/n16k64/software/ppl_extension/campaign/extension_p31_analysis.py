"""P31 nested 128+128 analysis and immutable G5 continuation decision."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b")
PARENT_RUNS = {model: f"V40_ppl_ksweep_{model}_attempt1" for model in MODELS}


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            lr, st, vr = (load(run / name) for name in
                          ("launch_record.json", "job_status.json", "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if lr.get("status") == st.get("status") == "complete" and vr.get("valid"):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def identity(new_run, new, parent_run, parent):
    result = {}
    for domain in ("wiki", "c4"):
        an, tn = arrays(new, "four_over_six", domain)
        ao, to = arrays(parent, "four_over_six", domain)
        result[domain] = {
            "window_metadata_exact": load(new_run / f"ppl/windows_{domain}.json") == load(parent_run / f"ppl/windows_{domain}.json"),
            "token_counts_exact": bool(np.array_equal(tn, to)),
            "four_over_six_nll_sums_exact": bool(np.array_equal(an, ao)),
            "aggregate_exact": new["evaluation"]["four_over_six"][domain]["mean_nll"] ==
                               parent["evaluation"]["four_over_six"][domain]["mean_nll"],
        }
        result[domain]["passed"] = all(result[domain].values())
    return {"domains": result, "passed": all(row["passed"] for row in result.values())}


def contrast(new, parent, policy, meta, model, domain):
    a, ta = arrays(new, policy, domain)
    b, tb = arrays(parent, policy, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P31 pairing failed for {model}/{domain}/{policy}")
    label = f"P31:{model}:{domain}:{policy}:128minus64"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, policy=policy, comparator_calibration="seed0_64_plus_64",
               method_calibration="nested_128_plus_128", stream_label=label,
               raw_ppl_method=new["evaluation"][policy][domain]["ppl"],
               raw_ppl_comparator=parent["evaluation"][policy][domain]["ppl"])
    return row


def g5_decision(n16_rows):
    means = {model: float(np.mean([row["estimate"] for row in n16_rows if row["model"] == model]))
             for model in MODELS}
    qualifying = [model for model, value in means.items() if value <= -0.0005]
    safety = all(row["estimate"] <= 0.001 for row in n16_rows)
    return {"continue_to_p32": len(qualifying) >= 2 and safety,
            "per_model_cross_corpus_mean": means, "models_at_or_below_minus_0p0005": qualifying,
            "all_six_cells_at_or_below_plus_0p001": safety,
            "meaningful_gain_threshold": -0.0005, "safety_threshold": 0.001}


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    out = runtime.out_dir("analysis_p31")
    rows, raw, inputs = [], [], {}
    for model in MODELS:
        run = latest_complete(f"P31_quality_{model}")
        parent_run = PR / "runs" / PARENT_RUNS[model]
        new, parent = load(run / "ppl/ppl_report.json"), load(parent_run / "ppl/ppl_report.json")
        if new.get("status") != "complete" or not new.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P31 PPL report: {model}")
        ident = identity(run, new, parent_run, parent)
        if not ident["passed"]:
            raise RuntimeError(f"P31 parent pairing identity failed: {model}")
        inputs[model] = {"run_id": run.name, "report_sha256": runtime.sha256_file(run / "ppl/ppl_report.json"),
                         "parent_run_id": parent_run.name,
                         "parent_report_sha256": runtime.sha256_file(parent_run / "ppl/ppl_report.json"),
                         "pairing_identity": ident}
        for domain in ("wiki", "c4"):
            meta = load(run / f"ppl/windows_{domain}.json")
            for policy in ("n8_k2", "n16_k2"):
                rows.append(contrast(new, parent, policy, meta, model, domain))
            for source, report in (("p31", new), ("seed0", parent)):
                for policy in (("four_over_six", "n8_k2", "n16_k2") if source == "p31" else ("n8_k2", "n16_k2")):
                    raw.append({"model": model, "corpus": domain, "calibration": source,
                                "policy": policy, "raw_ppl": report["evaluation"][policy][domain]["ppl"],
                                "mean_nll": report["evaluation"][policy][domain]["mean_nll"],
                                "run_id": run.name if source == "p31" else parent_run.name})
    n16 = [row for row in rows if row["policy"] == "n16_k2"]
    n8 = [row for row in rows if row["policy"] == "n8_k2"]
    for family in (n16, n8):
        adjusted, rejected = S.holm([row["p_two_sided"] for row in family])
        for row, p, reject in zip(family, adjusted, rejected):
            row["p_holm"] = p; row["holm_reject_0_05"] = reject
    decision = g5_decision(n16)
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "incremental_spec_sha256": runtime.sha256_file(CR / "provenance/P31_P32_INCREMENTAL_SPEC.json"),
                "bootstrap_replicates": B, "run_inputs": inputs, "contrasts": {"n16_k2": n16, "n8_k2": n8},
                "G5": decision}
    analysis_path = out / "P31_CALIBRATION_SIZE_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P31_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    gate = {"schema_version": "1.0", "status": "FROZEN_AFTER_COMPLETE_P31",
            "protocol_sha256": PROTOCOL, **decision,
            "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path),
            "P32_terminal_status": "authorized_pending" if decision["continue_to_p32"] else "stopped_by_gate"}
    gate_path = CR / "provenance/P31_G5_DECISION.json"
    write_new(gate_path, gate)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P31-development-analysis", "model_revision": gate["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(gate_path)],
                    "summary": {"G5_continue": decision["continue_to_p32"],
                                "per_model_means": decision["per_model_cross_corpus_mean"]},
                    "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": ["P31_CAL_SIZE_128", "G5"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "G5_continue": decision["continue_to_p32"]}, sort_keys=True))


if __name__ == "__main__":
    main()
