"""Frozen P10/P11 breadth analysis for the PPL-improvement extension.

The module never treats a same-named corpus as paired by convention.  It first proves
that the extension run reproduced the complete parent window metadata and every
FourOverSix window token count/NLL sum.  Only then may the parent k3/system arms be
used as paired comparators for the new k2 run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MATERIAL = math.log(1.005)
B = 10_000
MODELS = {
    "qwen27b": {
        "p10": "P10_k2_breadth_qwen27b",
        "p11": "P11_baseline_completion_qwen27b",
        "parent": "V31_ppl_primary_qwen27b_attempt1",
        "family": "Qwen",
    },
    "phi4": {
        "p10": "P10_k2_breadth_phi4",
        "p11": "P11_baseline_completion_phi4",
        "parent": "V62_ppl_primary_phi4_attempt3",
        "family": "Phi",
    },
    "olmo2_13b": {
        "p10": "P10_k2_breadth_olmo2_13b",
        "p11": "P11_baseline_completion_olmo2_13b",
        "parent": "V62_ppl_primary_olmo2_13b_attempt3",
        "family": "OLMo",
    },
}
SYSTEM_ARMS = [
    "nvfp4", "four_over_six", "all_e0m3", "razer_wonly_shared_act",
    "razer_native_rows", "nover6_wonly_shared_act", "nover6_native_rows",
]
TABLE_ARMS = ["bf16", "nvfp4", "four_over_six", "all_e0m3", "n8_k3", "n16_k3",
              "n8_k2", "n16_k2", "razer_wonly_shared_act", "razer_native_rows",
              "nover6_wonly_shared_act", "nover6_native_rows"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def latest_complete(prefix: str) -> Path:
    found = []
    for d in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(d.name.rsplit("_attempt", 1)[1])
            launch = load(d / "launch_record.json")
            status = load(d / "job_status.json")
            valid = load(d / "run_record_validation.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if launch.get("status") == "complete" and status.get("status") == "complete" and valid.get("valid"):
            found.append((attempt, d))
    if not found:
        raise FileNotFoundError(f"no accepted complete attempt for {prefix}")
    return max(found)[1]


def report(run: Path) -> dict:
    p = run / "ppl" / "ppl_report.json"
    x = load(p)
    if x.get("status") != "complete" or not x.get("reinstall_check", {}).get("identical"):
        raise RuntimeError(f"incomplete/non-reinstallable PPL report: {run.name}")
    return x


def window_rows(rep: dict, policy: str, domain: str):
    e = rep["evaluation"][policy][domain]
    return (np.asarray([r["nll_sum"] for r in e["windows"]], dtype=np.float64),
            np.asarray([r["tokens"] for r in e["windows"]], dtype=np.float64))


def identity(new_run: Path, new: dict, parent_run: Path, parent: dict, policy="four_over_six") -> dict:
    domains = {}
    for dom in ("wiki", "c4"):
        new_meta = load(new_run / "ppl" / f"windows_{dom}.json")
        old_meta = load(parent_run / "ppl" / f"windows_{dom}.json")
        an, tn = window_rows(new, policy, dom)
        ao, to = window_rows(parent, policy, dom)
        domains[dom] = {
            "window_metadata_equal": new_meta == old_meta,
            "window_metadata_sha256": sha(new_run / "ppl" / f"windows_{dom}.json"),
            "parent_window_metadata_sha256": sha(parent_run / "ppl" / f"windows_{dom}.json"),
            "window_count_equal": len(an) == len(ao),
            "token_counts_exact": bool(np.array_equal(tn, to)),
            "four_over_six_nll_sums_exact": bool(np.array_equal(an, ao)),
            "four_over_six_aggregate_exact": new["evaluation"][policy][dom]["mean_nll"] == parent["evaluation"][policy][dom]["mean_nll"],
        }
        domains[dom]["passed"] = all(v for k, v in domains[dom].items() if k.endswith("equal") or k.endswith("exact"))
    return {"domains": domains, "passed": all(v["passed"] for v in domains.values())}


def contrast(a_rep: dict, a_policy: str, b_rep: dict, b_policy: str, meta: dict,
             model: str, domain: str, family: str) -> dict:
    a, ta = window_rows(a_rep, a_policy, domain)
    b, tb = window_rows(b_rep, b_policy, domain)
    if not np.array_equal(ta, tb) or len(a) != len(b):
        raise RuntimeError(f"unpaired windows for {model} {domain} {a_policy}/{b_policy}")
    clusters = S.clusters_for(meta, domain)
    label = f"P10:{family}:{model}:{domain}:{a_policy}:{b_policy}"
    out = S.paired_dlogppl(a, b, ta, clusters, B=B, seed=S.stable_seed_sequence(20260914, label))
    out.update(model=model, family=family, corpus=domain, method=a_policy, comparator=b_policy,
               stream_label=label,
               raw_ppl_method=a_rep["evaluation"][a_policy][domain]["ppl"],
               raw_ppl_comparator=b_rep["evaluation"][b_policy][domain]["ppl"],
               mean_nll_method=a_rep["evaluation"][a_policy][domain]["mean_nll"],
               mean_nll_comparator=b_rep["evaluation"][b_policy][domain]["mean_nll"])
    return out


def adjust(rows: list[dict]) -> None:
    adjusted, rejected = S.holm([r["p_two_sided"] for r in rows])
    for row, p, reject in zip(rows, adjusted, rejected):
        row["p_holm"] = p
        row["holm_reject_0_05"] = reject


def missing(reason: str) -> dict:
    return {"status": "not_run", "reason": reason}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-p11", action="store_true")
    args = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    out = runtime.out_dir("analysis_breadth")
    analysis = {
        "schema_version": "1.0", "status": "running", "protocol_sha256": PROTOCOL,
        "analysis_spec_sha256": runtime.sha256_file(CR / "provenance/P10_P11_ANALYSIS_SPEC.json"),
        "B": B, "models": {}, "families": {}, "raw_table": [],
        "post_hoc_label": "k2 is a post-hoc robustness extension selected after the parent sweep",
    }
    families = {"breadth_quality": [], "threshold_improvement": [], "natural_tile_shape": []}
    raw_rows = []
    for model, spec in MODELS.items():
        p10_run = latest_complete(spec["p10"])
        p10 = report(p10_run)
        parent_run = PR / "runs" / spec["parent"]
        parent = report(parent_run)
        ident = identity(p10_run, p10, parent_run, parent)
        if not ident["passed"]:
            raise RuntimeError(f"parent pairing identity failed for {model}")
        p11_run = None
        p11 = None
        p11_identity = None
        try:
            p11_run = latest_complete(spec["p11"])
            p11 = report(p11_run)
            p11_identity = identity(p11_run, p11, parent_run, parent)
            if not p11_identity["passed"]:
                raise RuntimeError(f"P11 pairing identity failed for {model}")
        except FileNotFoundError:
            if not args.allow_missing_p11:
                raise
        m = {"p10_run": p10_run.name, "p10_report_sha256": sha(p10_run / "ppl/ppl_report.json"),
             "parent_run": parent_run.name, "parent_report_sha256": sha(parent_run / "ppl/ppl_report.json"),
             "parent_pairing_identity": ident,
             "p11_run": p11_run.name if p11_run else None, "p11_pairing_identity": p11_identity,
             "contrasts": {}}
        for dom in ("wiki", "c4"):
            meta = load(p10_run / "ppl" / f"windows_{dom}.json")
            rows = {
                "n16_k2_vs_four_over_six": contrast(p10, "n16_k2", p10, "four_over_six", meta, model, dom, spec["family"]),
                "n16_k2_vs_n16_k3": contrast(p10, "n16_k2", parent, "n16_k3", meta, model, dom, spec["family"]),
                "n16_k2_vs_n8_k2": contrast(p10, "n16_k2", p10, "n8_k2", meta, model, dom, spec["family"]),
            }
            m["contrasts"][dom] = rows
            families["breadth_quality"].append(rows["n16_k2_vs_four_over_six"])
            families["threshold_improvement"].append(rows["n16_k2_vs_n16_k3"])
            families["natural_tile_shape"].append(rows["n16_k2_vs_n8_k2"])
            sources = {**{k: (p10, p10_run.name) for k in ("n8_k2", "n16_k2")},
                       **{k: (parent, parent_run.name) for k in ("bf16", "nvfp4", "four_over_six", "all_e0m3", "n8_k3", "n16_k3")}}
            if p11:
                sources.update({k: (p11, p11_run.name) for k in p11["evaluation"] if k != "four_over_six"})
            for arm in TABLE_ARMS:
                if arm in sources and arm in sources[arm][0]["evaluation"]:
                    rr, rid = sources[arm]
                    ev = rr["evaluation"][arm][dom]
                    raw_rows.append({"model": model, "family": spec["family"], "corpus": dom, "method": arm,
                                     "status": "complete", "raw_ppl": ev["ppl"], "mean_nll": ev["mean_nll"],
                                     "run_id": rid})
                else:
                    z = missing("P11 baseline attempt unavailable" if arm in SYSTEM_ARMS else "not present in frozen plan")
                    raw_rows.append({"model": model, "family": spec["family"], "corpus": dom, "method": arm,
                                     **z, "raw_ppl": None, "mean_nll": None, "run_id": None})
        analysis["models"][model] = m
    for name, rows in families.items():
        adjust(rows)
        est = [r["estimate"] for r in rows]
        per_model = {m: float(np.mean([r["estimate"] for r in rows if r["model"] == m])) for m in MODELS}
        analysis["families"][name] = {
            "rows": rows, "median_six_cells": float(np.median(est)), "worst_cell": float(np.max(est)),
            "per_model_cross_corpus_mean": per_model,
            "non_qwen_median_four_cells": float(np.median([r["estimate"] for r in rows if r["model"] != "qwen27b"])),
        }
    bq = analysis["families"]["breadth_quality"]
    beneficial_models = [m for m in MODELS if all(r["estimate"] < 0 for r in bq["rows"] if r["model"] == m)]
    g1 = len(beneficial_models) >= 2 and all(r["estimate"] < MATERIAL for r in bq["rows"])
    ti = analysis["families"]["threshold_improvement"]
    g2 = ti["median_six_cells"] < 0 and all(v < MATERIAL for v in ti["per_model_cross_corpus_mean"].values())
    analysis["gates"] = {
        "G1": {"passed": g1, "beneficial_both_corpora_models": beneficial_models,
               "no_material_regression": all(r["estimate"] < MATERIAL for r in bq["rows"]),
               "material_threshold": MATERIAL,
               "non_qwen_all_beneficial": all(r["estimate"] < 0 for r in bq["rows"] if r["model"] != "qwen27b"),
               "claim_if_pass": "broadly robust post-hoc improvement versus FourOverSix only"},
        "G2": {"passed": g2, "median_six_cells": ti["median_six_cells"],
               "per_model_cross_corpus_mean": ti["per_model_cross_corpus_mean"],
               "practically_equivalent": abs(ti["median_six_cells"]) < MATERIAL,
               "material_threshold": MATERIAL},
        "G4_natural_precursor": {"status": "descriptive_only_until_P20_matched_density",
                                  "median_six_cells": analysis["families"]["natural_tile_shape"]["median_six_cells"],
                                  "worst_cell": analysis["families"]["natural_tile_shape"]["worst_cell"]},
    }
    analysis["raw_table"] = raw_rows
    analysis["status"] = "complete" if all(x["p11_run"] for x in analysis["models"].values()) else "partial_missing_P11"
    path = out / "P10_P11_BREADTH_ANALYSIS.json"
    runtime.atomic_json(path, analysis)
    csv_path = out / "P10_P11_RAW_PPL.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader(); w.writerows(raw_rows)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "breadth-analysis", "model_revision": analysis["analysis_spec_sha256"],
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path), str(csv_path)],
                    "summary": {"G1": g1, "G2": g2, "status": analysis["status"]},
                    "uncertainty": {"bootstrap_replicates": B},
                    "attempted_endpoints": ["P10_K2_BREADTH", "P11_BASELINE_COMPLETION", "G1", "G2"],
                    "missing_endpoints": [] if analysis["status"] == "complete" else ["P11_BASELINE_COMPLETION"]},
        "logs": [], "failures": []})
    print(json.dumps({"status": analysis["status"], "G1": g1, "G2": g2, "path": str(path)}, sort_keys=True), flush=True)
    if analysis["status"] != "complete" and not args.allow_missing_p11:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
