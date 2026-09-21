"""Paired P72 accuracy, P73 GSM8K, and P74 PG19 analyses."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.extension_heldout_maps import latest_complete, load
from campaign.report_contracts import validate_policy_contract


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
TASKS = ("arc_easy", "arc_challenge", "hellaswag", "openbookqa", "boolq",
         "winogrande", "piqa", "mmlu")
METRIC = {"arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm",
          "openbookqa": "acc_norm", "boolq": "acc", "winogrande": "acc",
          "piqa": "acc_norm", "mmlu": "acc"}


def task_rows(samples, task):
    if task == "mmlu":
        rows = {}
        for name, entry in sorted(samples.items()):
            if name.startswith("mmlu_"):
                for key, value in S.load_samples(entry["path"], "acc").items():
                    rows[(name, key)] = value
        if not rows:
            raise RuntimeError("MMLU subject samples are absent")
        return rows
    return S.load_samples(samples[task]["path"], METRIC[task])


def gsm_rows(path, filter_name):
    rows = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("filter") == filter_name:
                key = (row.get("doc_id"), row.get("doc_hash"))
                rows[key] = float(row["exact_match"])
    if not rows:
        raise RuntimeError(f"GSM8K filter absent: {filter_name}")
    return rows


def serializable_accuracy(row):
    return {key: value for key, value in row.items() if key != "boot"}


def compare_accuracy(report, method, comparator, model):
    sa = report["results"][method]["samples"]
    sb = report["results"][comparator]["samples"]
    per, full = {}, {}
    for task in TASKS:
        label = f"P72:{model}:{task}:{method}:{comparator}"
        row = S.paired_accuracy(task_rows(sa, task), task_rows(sb, task), B=B,
                                seed=20260914, stream_label=label)
        full[task] = row; per[task] = serializable_accuracy(row)
        per[task].update(stream_label=label, model=model, task=task,
                         method=method, comparator=comparator)
    macro = S.macro_accuracy(full)
    macro.update(model=model, task="macro", method=method,
                 comparator=comparator,
                 stream_label=f"P72:{model}:macro:{method}:{comparator}")
    pvals = [per[task]["mcnemar_p"] for task in TASKS]
    adjusted, rejected = S.holm(pvals)
    for task, p, reject in zip(TASKS, adjusted, rejected):
        per[task]["mcnemar_p_holm"] = p
        per[task]["holm_reject_0_05"] = reject
    return {"tasks": per, "macro": macro,
            "negative_Holm_rejections": [task for task in TASKS
                                           if per[task]["diff"] < 0 and per[task]["holm_reject_0_05"]]}


def analyze_p72():
    gate = load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")
    models = gate["eligible_heldout_models"]
    contrasts, inputs, raw = {}, {}, []
    for model in models:
        run = latest_complete(f"P72_accuracy_{model}")
        report = load(run / "lmeval/lmeval_report.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"unaccepted P72 report: {model}")
        validate_policy_contract(
            report,
            ("frozen_winner", "four_over_six", "n8_k2", "n16_k3",
             "strongest_fair_baseline"),
            f"P72 {model}", mapping_field="results")
        inputs[model] = {"run_id": run.name,
                         "report_sha256": runtime.sha256_file(run / "lmeval/lmeval_report.json")}
        contrasts[model] = {}
        for comparator in ("four_over_six", "n8_k2", "n16_k3", "strongest_fair_baseline"):
            contrasts[model][comparator] = compare_accuracy(report, "frozen_winner", comparator, model)
        for policy, result in report["results"].items():
            for task, sample in result["samples"].items():
                raw.append({"model": model, "policy": policy, "task": task,
                            "sample_rows": sample["rows"], "sample_file_sha256": sample["file_sha256"],
                            "sample_content_sha256": sample["content_sha256"], "run_id": run.name})
    model_gate = {}
    for model in models:
        primary = contrasts[model]["four_over_six"]
        model_gate[model] = {"macro_diff": primary["macro"]["diff"],
                             "macro_ci95": primary["macro"]["ci95"],
                             "lower_ci_above_minus_0p005": primary["macro"]["ci95"][0] > -0.005,
                             "negative_Holm_rejections": primary["negative_Holm_rejections"]}
        model_gate[model]["passed"] = (model_gate[model]["lower_ci_above_minus_0p005"]
                                        and not model_gate[model]["negative_Holm_rejections"])
    passed = bool(models) and all(row["passed"] for row in model_gate.values())
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "stage": "P72_ACCURACY", "bootstrap_replicates": B,
                "run_inputs": inputs, "contrasts": contrasts,
                "G8_accuracy_noninferiority": {"passed": passed, "sesoi": -0.005,
                                                "models": model_gate},
                "rng": "independent stable task-name SeedSequence streams",
                "correction": "Holm within each model's frozen eight-task family"}
    return write_analysis("p72", analysis, raw, "P72_ACCURACY")


def analyze_p73():
    gate = load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")
    models = gate["gsm8k_models"]
    results, inputs, raw = {}, {}, []
    for model in models:
        run = latest_complete(f"P73_gsm8k_{model}")
        report = load(run / "lmeval/lmeval_report.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"unaccepted P73 report: {model}")
        validate_policy_contract(
            report, ("frozen_winner", "four_over_six", "strongest_fair_baseline"),
            f"P73 {model}", mapping_field="results")
        inputs[model] = {"run_id": run.name,
                         "report_sha256": runtime.sha256_file(run / "lmeval/lmeval_report.json")}
        results[model] = {}
        for comparator in ("four_over_six", "strongest_fair_baseline"):
            results[model][comparator] = {}
            for filt in ("flexible-extract", "strict-match"):
                a = gsm_rows(report["results"]["frozen_winner"]["samples"]["gsm8k"]["path"], filt)
                b = gsm_rows(report["results"][comparator]["samples"]["gsm8k"]["path"], filt)
                label = f"P73:{model}:gsm8k:{filt}:frozen_winner:{comparator}"
                row = S.paired_accuracy(a, b, B=B, seed=20260914, stream_label=label)
                results[model][comparator][filt] = serializable_accuracy(row)
                results[model][comparator][filt].update(
                    stream_label=label, model=model, task="gsm8k", filter=filt,
                    method="frozen_winner", comparator=comparator)
        for policy, result in report["results"].items():
            sample = result["samples"]["gsm8k"]
            raw.append({"model": model, "policy": policy, "task": "gsm8k",
                        "sample_rows": sample["rows"], "sample_file_sha256": sample["file_sha256"],
                        "sample_content_sha256": sample["content_sha256"], "run_id": run.name})
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "stage": "P73_GSM8K", "bootstrap_replicates": B,
                "run_inputs": inputs, "paired_results": results,
                "decoding": {"do_sample": False, "until": ["Question:", "</s>", "<|im_end|>"],
                             "max_gen_toks": 256, "num_fewshot": 5}}
    return write_analysis("p73", analysis, raw, "P73_GSM8K")


def ppl_arrays(report, policy, domain):
    rows = report["evaluation"][policy][domain]["windows"]
    return (np.asarray([row["nll_sum"] for row in rows], dtype=np.float64),
            np.asarray([row["tokens"] for row in rows], dtype=np.float64))


def analyze_p74():
    gate = load(CR / "provenance/P71_DOWNSTREAM_GATE_RESOLUTION.json")
    models = ["llama8b", "mistral7b", *gate["eligible_heldout_models"]]
    results, inputs, raw = {}, {}, []
    for model in models:
        run = latest_complete(f"P74_pg19_{model}")
        report = load(run / "ppl/ppl_report.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"unaccepted P74 report: {model}")
        validate_policy_contract(report,
                                 ("frozen_winner", "four_over_six", "n8_k2", "n16_k3"),
                                 f"P74 {model}")
        inputs[model] = {"run_id": run.name,
                         "report_sha256": runtime.sha256_file(run / "ppl/ppl_report.json")}
        results[model] = {}
        for domain in ("pg19_4k", "pg19_8k"):
            meta = load(run / f"ppl/windows_{domain}.json")
            if meta.get("source_books") != 20 or len({row["book"] for row in meta["window_meta"]}) != 20:
                raise RuntimeError(f"P74 does not contain 20 source books: {model}/{domain}")
            results[model][domain] = {}
            for comparator in ("four_over_six", "n8_k2", "n16_k3"):
                a, ta = ppl_arrays(report, "frozen_winner", domain)
                b, tb = ppl_arrays(report, comparator, domain)
                if not np.array_equal(ta, tb):
                    raise RuntimeError(f"P74 pairing failed: {model}/{domain}/{comparator}")
                label = f"P74:{model}:{domain}:frozen_winner:{comparator}"
                row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                                       seed=S.stable_seed_sequence(20260914, label))
                row.update(stream_label=label, model=model, context=domain,
                           method="frozen_winner", comparator=comparator,
                           raw_ppl_method=report["evaluation"]["frozen_winner"][domain]["ppl"],
                           raw_ppl_comparator=report["evaluation"][comparator][domain]["ppl"])
                results[model][domain][comparator] = row
            for policy in report["evaluation"]:
                ev = report["evaluation"][policy][domain]
                raw.append({"model": model, "context": domain, "policy": policy,
                            "raw_ppl": ev["ppl"], "mean_nll": ev["mean_nll"],
                            "source_books": 20, "run_id": run.name})
    analysis = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                "stage": "P74_PG19", "bootstrap_replicates": B,
                "run_inputs": inputs, "paired_results": results,
                "inference_unit": "source book", "source_books_per_model_context": 20,
                "exploratory_point_estimate_only": False}
    return write_analysis("p74", analysis, raw, "P74_PG19")


def write_analysis(stage, analysis, raw, endpoint):
    out = runtime.out_dir(f"analysis_{stage}")
    path = out / f"{endpoint}_ANALYSIS.json"; runtime.atomic_json(path, analysis)
    csv_path = out / f"{endpoint}_RAW.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    extra = []
    if stage == "p72":
        decision = {"schema_version": "1.0", "status": "FROZEN_AFTER_COMPLETE_P72",
                    "protocol_sha256": PROTOCOL, **analysis["G8_accuracy_noninferiority"],
                    "analysis_path": str(path), "analysis_sha256": runtime.sha256_file(path)}
        decision_path = CR / "provenance/P72_G8_ACCURACY_DECISION.json"
        if decision_path.exists():
            raise FileExistsError(decision_path)
        runtime.atomic_json(decision_path, decision); decision_path.chmod(0o444); extra.append(decision_path)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": endpoint + "-analysis", "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "multiple",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": "per-model",
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path), str(csv_path)] + [str(x) for x in extra],
                    "summary": {"stage": endpoint, "models": len(analysis["run_inputs"])},
                    "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": [endpoint], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "stage": endpoint,
                      "models": len(analysis["run_inputs"])}, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True,
                                                     choices=("p72", "p73", "p74")); a = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    {"p72": analyze_p72, "p73": analyze_p73, "p74": analyze_p74}[a.stage]()


if __name__ == "__main__":
    main()
