"""P50 fixed-budget selector analysis and immutable pre-P51 selection freeze."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b")
DOMAINS = ("wiki", "c4")
OLD = "p50_old_first_order_ce_kl"
STRONG = "p50_strongest_p20_selector_global_rerank"
CANDIDATES = (
    "p50_activation_weighted_error",
    "p50_layer_output_reconstruction",
    "p50_diagonal_hessian_gptq_proxy",
)


def validate_policy_contract(report, expected):
    """Validate membership plus the two sequence-bearing policy records.

    ``runtime.atomic_json`` writes mapping keys in canonical sorted order, so the
    order of the ``evaluation`` object is deliberately not a plan-order record.
    The plan and install arrays retain the frozen evaluation order.
    """
    expected = tuple(expected)
    evaluation = tuple(report["evaluation"])
    if len(evaluation) != len(expected) or set(evaluation) != set(expected):
        raise RuntimeError("P50 evaluation policy membership drift")
    for field in ("plan", "installs"):
        observed = tuple(row["name"] for row in report[field])
        if observed != expected:
            raise RuntimeError(f"P50 {field} policy order drift")


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


def contrast(report, method, comparator, meta, model, domain):
    a, ta = arrays(report, method, domain)
    b, tb = arrays(report, comparator, domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"P50 pairing failed: {model}/{domain}/{method}/{comparator}")
    label = f"P50:{model}:{domain}:{method}:{comparator}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, corpus=domain, method=method, comparator=comparator,
               stream_label=label,
               raw_ppl_method=report["evaluation"][method][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
    return row


def select_candidate(summaries, require_eligible=True, tolerance=0.0001):
    pool = [row for row in summaries if row["eligible"]] if require_eligible else list(summaries)
    if not pool:
        return None
    best = min(row["median_six_cells"] for row in pool)
    stage = [row for row in pool if row["median_six_cells"] <= best + tolerance]
    worst = min(row["worst_cell"] for row in stage)
    stage = [row for row in stage if row["worst_cell"] == worst]
    order = {name: i for i, name in enumerate(CANDIDATES)}
    return min(stage, key=lambda row: order[row["policy_name"]])


def select_audit_candidate(summaries):
    selected = select_candidate(summaries, require_eligible=True)
    return selected if selected is not None else select_candidate(summaries, require_eligible=False)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    path.chmod(0o444)


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    gate_path = CR / "provenance/P50_G6_OPERATIONALIZATION.json"
    out = runtime.out_dir("analysis_p50")
    reports, runs, raw = {}, {}, []
    budgets = {}
    for model in MODELS:
        run = latest_complete(f"P50_quality_{model}")
        report_path = run / "ppl/ppl_report.json"
        report = load(report_path)
        if report.get("status") != "complete" or not report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted P50 quality report: {model}")
        try:
            validate_policy_contract(report, (OLD, STRONG, *CANDIDATES))
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}: {model}") from exc
        installs = {row["name"]: row for row in report["installs"]}
        selected = {name: int(installs[name]["selected_tiles"]) * 1024 for name in report["evaluation"]}
        if len(set(selected.values())) != 1:
            raise RuntimeError(f"P50 selected-weight identity failed: {model}")
        budgets[model] = {"selected_weights": next(iter(selected.values())), "per_policy": selected}
        reports[model], runs[model] = report, run
        for policy, domains in report["evaluation"].items():
            for domain in DOMAINS:
                raw.append({"model": model, "corpus": domain, "policy": policy,
                            "raw_ppl": domains[domain]["ppl"], "mean_nll": domains[domain]["mean_nll"],
                            "run_id": run.name})

    candidate_summaries = []
    all_contrasts = []
    for candidate in CANDIDATES:
        old_cells, strong_cells = [], []
        for model in MODELS:
            report, run = reports[model], runs[model]
            for domain in DOMAINS:
                meta = load(run / f"ppl/windows_{domain}.json")
                old_cells.append(contrast(report, candidate, OLD, meta, model, domain))
                strong_cells.append(contrast(report, candidate, STRONG, meta, model, domain))
        values = [row["estimate"] for row in old_cells]
        summary = {"policy_name": candidate,
                   "eligible": float(np.median(values)) <= -0.0005 and max(values) <= 0.001,
                   "median_six_cells": float(np.median(values)), "worst_cell": float(max(values)),
                   "candidate_minus_old": old_cells, "candidate_minus_strong_control": strong_cells}
        candidate_summaries.append(summary)
        all_contrasts.extend(old_cells + strong_cells)
    old_vs_strong = []
    for model in MODELS:
        report, run = reports[model], runs[model]
        for domain in DOMAINS:
            old_vs_strong.append(contrast(report, OLD, STRONG,
                                          load(run / f"ppl/windows_{domain}.json"), model, domain))
    all_contrasts.extend(old_vs_strong)

    selected = select_candidate(candidate_summaries, require_eligible=True)
    fallback = selected is None
    audit = select_audit_candidate(candidate_summaries)
    selected_policy = OLD if fallback else selected["policy_name"]
    selected_maps, audit_maps = {}, {}
    for model in MODELS:
        by_name = {row["name"]: row for row in reports[model]["plan"]}
        selected_maps[model] = {key: by_name[selected_policy][key] for key in
                                ("map_path", "map_sha256", "map_policy", "type_block",
                                 "protocol_id", "expected_total_tiles")}
        audit_maps[model] = {key: by_name[audit["policy_name"]][key] for key in
                             ("map_path", "map_sha256", "map_policy", "type_block",
                              "protocol_id", "expected_total_tiles")}
    analysis = {
        "schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
        "gate_spec_sha256": runtime.sha256_file(gate_path), "bootstrap_replicates": B,
        "development_only": True, "selected_weight_identity": budgets,
        "run_inputs": {model: {"run_id": runs[model].name,
                               "report_sha256": runtime.sha256_file(runs[model] / "ppl/ppl_report.json")}
                       for model in MODELS},
        "candidate_summaries": candidate_summaries, "old_minus_strong_control": old_vs_strong,
        "full_map_replacement_found": not fallback,
        "selected_replacement": selected, "P51_audit_candidate": audit,
        "post_selection_inference": "Development selection only; no naive confirmatory winner CI.",
    }
    analysis_path = out / "P50_SELECTOR_ANALYSIS.json"
    runtime.atomic_json(analysis_path, analysis)
    csv_path = out / "P50_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    contrast_path = out / "P50_PAIRED_CONTRASTS.json"
    runtime.atomic_json(contrast_path, all_contrasts)
    freeze = {
        "schema_version": "1.0", "status": "FROZEN_AFTER_P50_FULL_MAPS_BEFORE_P51_SAMPLING_OR_EFFECTS",
        "protocol_sha256": PROTOCOL, "gate_spec_sha256": runtime.sha256_file(gate_path),
        "fallback_to_old": fallback, "selected_policy": selected_policy,
        "selected_replacement": None if fallback else selected["policy_name"],
        "P51_audit_policy": audit["policy_name"],
        "selected_maps": selected_maps, "P51_audit_maps": audit_maps,
        "analysis_path": str(analysis_path), "analysis_sha256": runtime.sha256_file(analysis_path),
        "selection_rule": "frozen eligibility; median; within-0.0001 worst; frozen candidate order",
    }
    freeze_path = CR / "provenance/P50_SELECTOR_SELECTION.json"
    write_new(freeze_path, freeze)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P50-development-analysis", "model_revision": freeze["analysis_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(analysis_path), str(csv_path), str(contrast_path), str(freeze_path)],
                    "summary": {"full_map_replacement_found": not fallback,
                                "selected_policy": selected_policy,
                                "P51_audit_policy": audit["policy_name"]},
                    "uncertainty": {"bootstrap_replicates": B, "selection_adjusted": False},
                    "attempted_endpoints": ["P50_SELECTOR_DEV", "P50_FULL_MAP_FREEZE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "selected_policy": selected_policy,
                      "audit_policy": audit["policy_name"], "fallback": fallback}, sort_keys=True))


if __name__ == "__main__":
    main()
