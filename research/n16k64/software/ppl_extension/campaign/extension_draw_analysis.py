"""Five-draw k2 robustness analysis with separate map and evaluation uncertainty."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import mapio as MIO
from campaign import runtime
from campaign import stats as S


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
B = 10_000
MODELS = ("llama8b", "qwen4b", "mistral7b", "qwen27b", "phi4", "olmo2_13b")
DEV_SEED = {m: PR / "runs" / f"V40_ppl_ksweep_{m}_attempt1" for m in MODELS[:3]}


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


def cluster_components(a, b, tokens, clusters):
    k = int(clusters.max()) + 1
    return (np.bincount(clusters, weights=a - b, minlength=k),
            np.bincount(clusters, weights=tokens, minlength=k))


def cluster_bootstrap(dc, nc, count, rng):
    idx = rng.integers(0, len(dc), size=(count, len(dc)))
    return dc[idx].sum(1) / nc[idx].sum(1)


def two_level_interval(components, B=B, seed=20260914):
    """Resample draw and document/article clusters at their respective levels."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    children = iter(ss.spawn(1 + len(components) * len(components) + len(components)))
    choice_rng = np.random.default_rng(next(children))
    choices = choice_rng.integers(0, len(components), size=(B, len(components)))
    boot = np.zeros(B, dtype=np.float64)
    for position in range(len(components)):
        values = np.empty(B, dtype=np.float64)
        for draw, (dc, nc) in enumerate(components):
            mask = choices[:, position] == draw
            rng = np.random.default_rng(next(children))
            values[mask] = cluster_bootstrap(dc, nc, int(mask.sum()), rng)
        boot += values / len(components)
    points = np.asarray([dc.sum() / nc.sum() for dc, nc in components])
    within_variances = []
    for dc, nc in components:
        rng = np.random.default_rng(next(children))
        within_variances.append(float(cluster_bootstrap(dc, nc, B, rng).var(ddof=1)))
    return {"estimate_draw_mean": float(points.mean()),
            "hierarchical_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "hierarchical_bootstrap_replicates": B,
            "between_draw_variance": float(points.var(ddof=1)),
            "mean_within_evaluation_bootstrap_variance": float(np.mean(within_variances)),
            "variance_components_are_descriptive": True}


def effect(report, policy, domain, meta, model, draw, resolution):
    a, ta = arrays(report, policy, domain)
    b, tb = arrays(report, "four_over_six", domain)
    if not np.array_equal(ta, tb):
        raise RuntimeError(f"unpaired draw windows: {model}/{draw}/{resolution}/{domain}")
    label = f"P30:{model}:{draw}:{resolution}:{domain}"
    row = S.paired_dlogppl(a, b, ta, S.clusters_for(meta, domain), B=B,
                           seed=S.stable_seed_sequence(20260914, label))
    row.update(model=model, draw=draw, resolution=resolution, corpus=domain,
               method=policy, comparator="four_over_six", stream_label=label,
               raw_ppl_method=report["evaluation"][policy][domain]["ppl"],
               raw_ppl_comparator=report["evaluation"]["four_over_six"][domain]["ppl"],
               mean_nll_method=report["evaluation"][policy][domain]["mean_nll"],
               mean_nll_comparator=report["evaluation"]["four_over_six"][domain]["mean_nll"])
    return row, cluster_components(a, b, ta, S.clusters_for(meta, domain))


def summarize_draws(rows, components, stream_label):
    values = np.asarray([row["estimate"] for row in rows], dtype=np.float64)
    sd = float(values.std(ddof=1))
    half = 2.7764451051977987 * sd / math.sqrt(5)
    out = {"model": rows[0]["model"], "resolution": rows[0]["resolution"],
           "corpus": rows[0]["corpus"],
           "draw_order": [row["draw"] for row in rows], "mean": float(values.mean()),
           "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
           "range": float(values.max() - values.min()), "worst_draw": rows[int(values.argmax())]["draw"],
           "between_draw_sd": sd, "small_sample_t95_for_draw_mean": [float(values.mean() - half),
                                                                        float(values.mean() + half)],
           "beneficial_sign_fraction": float((values < 0).mean()),
           "note": "Five maps are not treated as independent evaluation corpora; per-draw CIs remain separate."}
    out["two_level"] = two_level_interval(
        components, B=B, seed=S.stable_seed_sequence(20260914, stream_label))
    out["two_level"].update(
        model=rows[0]["model"], resolution=rows[0]["resolution"],
        corpus=rows[0]["corpus"],
        stream_label=stream_label,
        rng="NumPy PCG64 with a stable named SeedSequence",
        master_seed=20260914,
    )
    return out


def plan_entry(report, name):
    rows = [row for row in report["plan"] if row["name"] == name]
    if len(rows) != 1 or not rows[0].get("map_path"):
        raise RuntimeError(f"missing unique map plan entry {name}")
    return rows[0]


def map_stability(entries, type_block):
    maps, hashes, counts = {}, {}, {}
    for draw, entry in entries.items():
        header, masks, digest = MIO.read_map(entry["map_path"], entry["map_sha256"])
        if header["type_block"] != list(type_block):
            raise RuntimeError("draw map type-block mismatch")
        maps[draw], hashes[draw] = masks, digest
        counts[draw] = header["totals"]["selected_weights"]
    pairs = []
    for a, b in itertools.combinations(entries, 2):
        inter = sum(int((maps[a][name] & maps[b][name]).sum()) for name in maps[a])
        union = sum(int((maps[a][name] | maps[b][name]).sum()) for name in maps[a])
        pairs.append({"draw_a": a, "draw_b": b, "intersection_tiles": inter,
                      "union_tiles": union, "jaccard": inter / union if union else 1.0})
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    return {"map_sha256": hashes, "selected_weights": counts, "pairwise_jaccard": pairs,
            "jaccard_median": float(np.median([row["jaccard"] for row in pairs])),
            "jaccard_min": float(min(row["jaccard"] for row in pairs)),
            "selected_weight_mean": float(vals.mean()), "selected_weight_sd": float(vals.std(ddof=1)),
            "selected_weight_min": int(vals.min()), "selected_weight_max": int(vals.max())}


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    out = runtime.out_dir("analysis_draws")
    analysis = {"schema_version": "1.0", "status": "running", "protocol_sha256": PROTOCOL,
                "bootstrap_replicates": B, "models": {},
                "claim_label": "post-hoc k2 calibration-draw robustness extension"}
    raw = []
    for model in MODELS:
        seed_run = DEV_SEED[model] if model in DEV_SEED else latest_complete(f"P10_k2_breadth_{model}")
        draw_run = latest_complete(f"P30_draw_quality_{model}")
        seed_report, draw_report = load(seed_run / "ppl/ppl_report.json"), load(draw_run / "ppl/ppl_report.json")
        if draw_report.get("status") != "complete" or not draw_report.get("reinstall_check", {}).get("identical"):
            raise RuntimeError(f"unaccepted draw report: {model}")
        identity = {}
        for domain in ("wiki", "c4"):
            _, st = arrays(seed_report, "four_over_six", domain)
            _, dt = arrays(draw_report, "four_over_six", domain)
            sa, _ = arrays(seed_report, "four_over_six", domain)
            da, _ = arrays(draw_report, "four_over_six", domain)
            identity[domain] = {
                "window_metadata_exact": load(seed_run / f"ppl/windows_{domain}.json") ==
                                         load(draw_run / f"ppl/windows_{domain}.json"),
                "token_counts_exact": bool(np.array_equal(st, dt)),
                "four_over_six_repeat_nll_exact": bool(np.array_equal(sa, da)),
                "repeat_delta_mean_nll": float(da.sum() / dt.sum() - sa.sum() / st.sum())}
            if not identity[domain]["window_metadata_exact"] or not identity[domain]["token_counts_exact"]:
                raise RuntimeError(f"draw evaluation identity failed: {model}/{domain}")
        model_out = {"seed_run_id": seed_run.name, "draw_run_id": draw_run.name,
                     "seed_report_sha256": runtime.sha256_file(seed_run / "ppl/ppl_report.json"),
                     "draw_report_sha256": runtime.sha256_file(draw_run / "ppl/ppl_report.json"),
                     "evaluation_identity": identity, "quality": {}, "map_stability": {}}
        for resolution, tb in (("n8", (8, 64)), ("n16", (16, 64))):
            map_entries = {"seed0": plan_entry(seed_report, f"{resolution}_k2")}
            for index in range(1, 5):
                map_entries[f"draw{index}"] = plan_entry(draw_report, f"{resolution}_k2_draw{index}")
            model_out["map_stability"][resolution] = map_stability(map_entries, tb)
            for domain in ("wiki", "c4"):
                rows, components = [], []
                meta = load(seed_run / f"ppl/windows_{domain}.json")
                row, comp = effect(seed_report, f"{resolution}_k2", domain, meta, model, "seed0", resolution)
                rows.append(row); components.append(comp)
                for index in range(1, 5):
                    draw = f"draw{index}"
                    meta = load(draw_run / f"ppl/windows_{domain}.json")
                    row, comp = effect(draw_report, f"{resolution}_k2_{draw}", domain, meta,
                                       model, draw, resolution)
                    rows.append(row); components.append(comp)
                stream_label = f"P30:hierarchical:{model}:{resolution}:{domain}"
                model_out["quality"][f"{resolution}_{domain}"] = {
                    "per_draw": rows,
                    "across_draw": summarize_draws(rows, components, stream_label),
                }
                for row in rows:
                    raw.extend([{"model": model, "resolution": resolution, "corpus": domain,
                                 "draw": row["draw"], "policy": row["method"],
                                 "raw_ppl": row["raw_ppl_method"], "mean_nll": row["mean_nll_method"]},
                                {"model": model, "resolution": resolution, "corpus": domain,
                                 "draw": row["draw"], "policy": "four_over_six",
                                 "raw_ppl": row["raw_ppl_comparator"], "mean_nll": row["mean_nll_comparator"]}])
        analysis["models"][model] = model_out
    analysis["status"] = "complete"
    path = out / "P30_FIVE_DRAW_ANALYSIS.json"
    runtime.atomic_json(path, analysis)
    csv_path = out / "P30_RAW_PPL.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P30-five-draw-analysis", "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "multiple", "model_class": "analysis",
                   "module_manifest_sha256": "not_applicable", "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(), "data": {"calibration_manifest_sha256": None,
            "evaluation_manifest_sha256": None, "token_hashes": {}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path), str(csv_path)],
                    "summary": {"models": len(MODELS), "draws_per_model": 5},
                    "uncertainty": {"within_evaluation_bootstrap": B, "between_draw_small_n": 5,
                                    "hierarchical_bootstrap": B},
                    "attempted_endpoints": ["P30_K2_DRAWS"], "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "models": len(MODELS)}, sort_keys=True))


if __name__ == "__main__":
    main()
