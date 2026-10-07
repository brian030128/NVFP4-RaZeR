"""Paper tables from existing records only (CPU): results/paper_eval/cpu_tables/.

1. Table main-ppl: 6 paper models x {BF16, NVFP4, FourOverSix, FlipQuant 8x64 / 16x64 / 256x64}, WikiText-2 and C4.
   - BF16 / NVFP4 / FourOverSix: main-ppl 44f8cea (results/main_ppl/main_ppl.json names each record);
   - FlipQuant: the release maps' evaluation (/home/dev/flipquant_release/<model>/ppl/), 5 epochs, 256 windows, top-1000;
   - the convention check: the release evaluation's own BF16 and FourOverSix per-window NLLs must equal main-ppl's
     bit for bit (same windows, convention and numerics), so the FlipQuant rows and the main-ppl rows are one table;
   - loss recovered: the share of NVFP4's log-PPL loss vs BF16 a row removes, summed over the 12 model-corpus pairs;
   - a dagger where FlipQuant is not significantly better than FourOverSix (paired per-window ΔNLL, |Δ| > 2 SE, Δ < 0).
2. Tile granularity: 8x64 vs 16x64 (paired per window), the share of the 16x64 gain over FourOverSix that 256x64
   keeps, and the E0M3 tile share per unit.
3. Calibration cost of the 5-epoch release runs of the 6 models (time, training share, 256x64 vs 8x64).
The map files' sha256 (the release file and the uploaded Hugging Face copy, same tiles) are listed too.

    python experiments/paper_eval/tables_cpu.py
"""
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_eval" / "cpu_tables"
REL = Path("/home/dev/flipquant_release")
FQREL = Path("/home/dev/n16k64_campaign/fqrel")
HF_STAGE = Path("/home/dev/flipquant_hf/stage.json")
MAIN_PPL_COMMIT = "44f8cea"
MODELS = [("qwen3-1.7b", "qwen3_1p7b", "Qwen3-1.7B"), ("qwen3-8b", "qwen3_8b", "Qwen3-8B"),
          ("mistral-7b", "mistral7b_ins", "Mistral-7B-Instruct-v0.3"),
          ("nemotron-nano-9b-v2", "nemotron9b", "Nemotron-Nano-9B-v2"), ("phi4-14b", "phi4", "Phi-4"),
          ("qwen3.8-27b", "qwen27b", "Qwen3.8-27B")]
UNITS = ("8x64", "16x64", "256x64")
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))
ROWS = ["bf16", "nvfp4", "fo6"] + [f"fq_{u}" for u in UNITS]
DAG = "$^\\dag$"
NAMES = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "fq_8x64": "FlipQuant 8x64",
         "fq_16x64": "FlipQuant 16x64", "fq_256x64": "FlipQuant 256x64"}


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True,
                                     text=True).stdout)


def nll_of(path):
    r = json.loads(Path(path).read_text())
    if "results" in r:                                            # flipquant evaluation.ppl
        return {c: r["results"][c]["nll"] for c, _ in CORPORA}
    ev = next(iter(r["evaluations"].values()))["evaluation"]    # run_ppl_deploy (main-ppl, paper step 03)
    return {c: ev[c]["nll"] for c, _ in CORPORA}


def paired(x, y):
    assert len(x) == len(y)
    d = [a - b for a, b in zip(x, y)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta=m, se=se, two_se=2 * se, n=n, significant=abs(m) > 2 * se)


def mean(x):
    return sum(x) / len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mp = git_json(MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    hf = json.loads(HF_STAGE.read_text())["repos"] if HF_STAGE.exists() else {}
    hf_by_model = {d["model"]: d for d in hf.values()}
    rec = dict(sources={}, convention={}, nll_mean={}, ppl={}, vs_fo6={}, fq8_vs_fq16={}, e0m3={}, maps={}, cost={})
    nll = {}
    for key, mkey, title in MODELS:
        src = {r: mp[mkey]["sources"][r] for r in ("bf16", "nvfp4", "fo6")}
        src.update({f"fq_{u}": str(REL / key / "ppl" / f"flipquant_{u}.json") for u in UNITS})
        rec["sources"][key] = src
        nll[key] = {r: nll_of(p) for r, p in src.items()}
        # the convention check: the release evaluation's own BF16 / FourOverSix rows against main-ppl's
        own = {r: nll_of(REL / key / "ppl" / f"{r}.json") for r in ("bf16", "fo6")}
        rec["convention"][key] = {r: {c: own[r][c] == nll[key][r][c] for c, _ in CORPORA} for r in own}
        assert all(all(v.values()) for v in rec["convention"][key].values()), (key, rec["convention"][key])
        for r in ROWS:
            assert all(len(nll[key][r][c]) == len(nll[key]["bf16"][c]) for c, _ in CORPORA)
        rec["nll_mean"][key] = {r: {c: mean(nll[key][r][c]) for c, _ in CORPORA} for r in ROWS}
        rec["ppl"][key] = {r: {c: math.exp(v) for c, v in d.items()} for r, d in rec["nll_mean"][key].items()}
        rec["vs_fo6"][key] = {r: {c: paired(nll[key][r][c], nll[key]["fo6"][c]) for c, _ in CORPORA}
                              for r in ("nvfp4", *[f"fq_{u}" for u in UNITS])}
        rec["fq8_vs_fq16"][key] = {c: paired(nll[key]["fq_8x64"][c], nll[key]["fq_16x64"][c]) for c, _ in CORPORA}
        # maps: E0M3 share (the release run records) and the two sha256
        for u in UNITS:
            run = json.loads((REL / key / "records" / u / "run.json").read_text())
            rec["e0m3"].setdefault(key, {})[u] = dict(e0m3=run["e0m3_tiles"], tiles=run["tiles"],
                                                       share=run["e0m3_tiles"] / run["tiles"])
            h = hf_by_model.get(key, {}).get("maps", {}).get(u, {})
            rec["maps"].setdefault(key, {})[u] = dict(release_sha256=run["map_sha256"],
                                                      hf_sha256=h.get("sha256_uploaded"),
                                                      hf_tiles_bit_identical=h.get("tiles_bit_identical"))
    # loss recovered over the 12 model-corpus pairs
    pairs = [(k, c) for k, _, _ in MODELS for c, _ in CORPORA]
    lost = sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k]["bf16"][c] for k, c in pairs)
    rec["loss_recovered"] = {r: sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k][r][c] for k, c in pairs) / lost
                             for r in ROWS}
    # context: Qwen3-1.7B's FlipQuant rows are below BF16 on WikiText-2, so that pair weighs heavily in the sum
    p10 = [(k, c) for k, c in pairs if k != "qwen3-1.7b"]
    lost10 = sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k]["bf16"][c] for k, c in p10)
    rec["loss_recovered_without_qwen3_1p7b"] = {
        r: sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k][r][c] for k, c in p10) / lost10 for r in ROWS}
    rec["loss_recovered_per_model"] = {
        k: {r: sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k][r][c] for c, _ in CORPORA) /
            sum(rec["nll_mean"][k]["nvfp4"][c] - rec["nll_mean"][k]["bf16"][c] for c, _ in CORPORA) for r in ROWS}
        for k, _, _ in MODELS}
    # tile granularity
    d = {u: [rec["vs_fo6"][k][f"fq_{u}"][c]["delta"] for k, c in pairs] for u in UNITS}
    rec["granularity"] = dict(
        mean_dnll_vs_fo6={u: mean(d[u]) for u in UNITS},
        fq8_minus_fq16=dict(mean_over_pairs=mean([rec["fq8_vs_fq16"][k][c]["delta"] for k, c in pairs]),
                            significant_better_8x64=sum(rec["fq8_vs_fq16"][k][c]["significant"] and
                                                        rec["fq8_vs_fq16"][k][c]["delta"] < 0 for k, c in pairs),
                            significant_better_16x64=sum(rec["fq8_vs_fq16"][k][c]["significant"] and
                                                         rec["fq8_vs_fq16"][k][c]["delta"] > 0 for k, c in pairs),
                            pairs=len(pairs)),
        share_256_of_16=sum(d["256x64"]) / sum(d["16x64"]),
        share_256_of_16_per_model={k: sum(rec["vs_fo6"][k]["fq_256x64"][c]["delta"] for c, _ in CORPORA) /
                                   sum(rec["vs_fo6"][k]["fq_16x64"][c]["delta"] for c, _ in CORPORA) for k, _, _ in MODELS},
        e0m3_pooled={u: sum(rec["e0m3"][k][u]["e0m3"] for k, _, _ in MODELS) / sum(rec["e0m3"][k][u]["tiles"]
                                                                                    for k, _, _ in MODELS) for u in UNITS},
        e0m3_range={u: (min(rec["e0m3"][k][u]["share"] for k, _, _ in MODELS),
                        max(rec["e0m3"][k][u]["share"] for k, _, _ in MODELS)) for u in UNITS})
    # calibration cost (the release records, via the release report's own reader)
    sys.path.insert(0, str(FQREL))
    import report as RP  # noqa: E402
    for key, _, _ in MODELS:
        for u in UNITS:
            m = RP.unit_metrics(key, u)
            rec["cost"].setdefault(key, {})[u] = dict(
                trainer_total_s=m["trainer_total"], end_to_end_s=m.get("end_to_end"), setup_s=m["setup"],
                training_s=m["training"], epoch_mean_s=m["epoch_mean"], training_share=m["training"] / m["trainer_total"],
                gpu_peak_allocated_gib=m["gpu_alloc"], trainer_peak_rss_gib=m["ru_maxrss"], source=m["source"])
    c = rec["cost"]
    rec["cost_summary"] = dict(
        trainer_total_min={u: (min(c[k][u]["trainer_total_s"] for k in c) / 60, max(c[k][u]["trainer_total_s"] for k in c) / 60)
                           for u in UNITS},
        training_share={u: (min(c[k][u]["training_share"] for k in c), max(c[k][u]["training_share"] for k in c)) for u in UNITS},
        training_share_pooled=sum(c[k][u]["training_s"] for k in c for u in UNITS) /
        sum(c[k][u]["trainer_total_s"] for k in c for u in UNITS),
        time_256_vs_8={k: c[k]["256x64"]["trainer_total_s"] / c[k]["8x64"]["trainer_total_s"] - 1 for k in c},
        gpu_256_vs_8={k: c[k]["256x64"]["gpu_peak_allocated_gib"] / c[k]["8x64"]["gpu_peak_allocated_gib"] - 1 for k in c},
        total_6_models_3_units_h=sum(c[k][u]["trainer_total_s"] for k in c for u in UNITS) / 3600)
    (OUT / "tables.json").write_text(json.dumps(rec, indent=1) + "\n")
    write_md(rec)
    write_tex(rec)


def dag(rec, key, r, c):
    if not r.startswith("fq_"):
        return ""
    p = rec["vs_fo6"][key][r][c]
    return "" if (p["significant"] and p["delta"] < 0) else "†"


def write_md(rec):
    L = ["# Paper tables from existing records (CPU)", "",
         "Generated by `experiments/paper_eval/tables_cpu.py`; data in `tables.json`.", "",
         "## 1. Table main-ppl (PPL, WikiText-2 / C4)", "",
         "BF16, NVFP4 and FourOverSix: main-ppl 44f8cea. FlipQuant: the 5-epoch release maps (256 windows, top-1000), "
         "native sm_120, the paper convention. The release evaluation's own BF16 and FourOverSix per-window NLLs equal "
         "main-ppl's bit for bit on every model and corpus, so all rows share windows, convention and numerics. "
         "† = FlipQuant not significantly better than FourOverSix (paired per-window ΔNLL, 2 SE).", ""]
    head = ["row"] + [t for _, _, t in MODELS] + ["loss recovered"]
    L += ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for r in ROWS:
        cells = [f"{rec['ppl'][k][r]['wiki']:.2f}{dag(rec, k, r, 'wiki')} / {rec['ppl'][k][r]['c4']:.2f}{dag(rec, k, r, 'c4')}"
                 for k, _, _ in MODELS]
        L.append(f"| {NAMES[r]} | " + " | ".join(cells) + f" | {100 * rec['loss_recovered'][r]:.1f} % |")
    L += ["", "Loss recovered without Qwen3-1.7B (10 pairs; its FlipQuant rows are below BF16 on WikiText-2, which weighs "
          "heavily in the 12-pair sum): " + ", ".join(f"{NAMES[r]} {100 * rec['loss_recovered_without_qwen3_1p7b'][r]:.1f} %"
                                                      for r in ROWS if r not in ("bf16", "nvfp4")) + ". Per model (both "
          "corpora): " + "; ".join(f"{t} " + "/".join(f"{100 * rec['loss_recovered_per_model'][k][r]:.0f}"
                                                         for r in ("fo6", "fq_8x64", "fq_16x64", "fq_256x64"))
                                   for k, _, t in MODELS) + " % (FourOverSix / FlipQuant 8x64 / 16x64 / 256x64).", ""]
    L += ["", "Paired ΔNLL against FourOverSix (nats/token, ± 2 SE; * = |Δ| > 2 SE), WikiText-2 / C4:", "",
          "| row | " + " | ".join(t for _, _, t in MODELS) + " |", "|---" + "|---" * len(MODELS) + "|"]
    for r in ("nvfp4", *[f"fq_{u}" for u in UNITS]):
        L.append(f"| {NAMES[r]} | " + " | ".join(
            " / ".join(f"{rec['vs_fo6'][k][r][c]['delta']:+.4f} ± {rec['vs_fo6'][k][r][c]['two_se']:.4f}"
                       f"{'*' if rec['vs_fo6'][k][r][c]['significant'] else ''}" for c, _ in CORPORA)
            for k, _, _ in MODELS) + " |")
    g = rec["granularity"]
    L += ["", "## 2. Tile granularity", "",
          f"- Mean ΔNLL vs FourOverSix over the 12 model–corpus pairs: 8x64 {g['mean_dnll_vs_fo6']['8x64']:+.4f}, "
          f"16x64 {g['mean_dnll_vs_fo6']['16x64']:+.4f}, 256x64 {g['mean_dnll_vs_fo6']['256x64']:+.4f} nats/token.",
          f"- 8x64 − 16x64 (paired per window): mean {g['fq8_minus_fq16']['mean_over_pairs']:+.4f} nats/token; 8x64 "
          f"significantly better in {g['fq8_minus_fq16']['significant_better_8x64']} pairs, 16x64 in "
          f"{g['fq8_minus_fq16']['significant_better_16x64']}, of {g['fq8_minus_fq16']['pairs']}.",
          f"- 256x64 keeps {100 * g['share_256_of_16']:.1f} % of the 16x64 gain over FourOverSix (ΔNLL summed over the 12 "
          "pairs); per model: " + ", ".join(f"{t} {100 * g['share_256_of_16_per_model'][k]:.0f} %" for k, _, t in MODELS) + ".",
          "", "| model | " + " | ".join(f"8x64 − 16x64, {n}" for _, n in CORPORA) + " |", "|---|---:|---:|"]
    for k, _, t in MODELS:
        L.append(f"| {t} | " + " | ".join(f"{rec['fq8_vs_fq16'][k][c]['delta']:+.4f} ± {rec['fq8_vs_fq16'][k][c]['two_se']:.4f}"
                                          f"{'*' if rec['fq8_vs_fq16'][k][c]['significant'] else ''}" for c, _ in CORPORA) + " |")
    L += ["", "E0M3 tile share per unit:", "", "| model | " + " | ".join(UNITS) + " |", "|---|---:|---:|---:|"]
    for k, _, t in MODELS:
        L.append(f"| {t} | " + " | ".join(f"{100 * rec['e0m3'][k][u]['share']:.2f} %" for u in UNITS) + " |")
    L.append("| pooled (all 6 models) | " + " | ".join(f"{100 * g['e0m3_pooled'][u]:.2f} %" for u in UNITS) + " |")
    s = rec["cost_summary"]
    L += ["", "## 3. Calibration cost (5-epoch release runs, one RTX PRO 6000)", "",
          "| model | unit | trainer total | train_map end-to-end | setup | training | training share | GPU peak allocated |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, _, t in MODELS:
        for u in UNITS:
            x = rec["cost"][k][u]
            e2e = f"{x['end_to_end_s'] / 60:.1f} min" if x["end_to_end_s"] else "—"
            L.append(f"| {t} | {u} | {x['trainer_total_s'] / 60:.1f} min | {e2e} | {x['setup_s']:.0f} s | "
                     f"{x['training_s'] / 60:.1f} min | {100 * x['training_share']:.0f} % | {x['gpu_peak_allocated_gib']:.1f} GiB |")
    L += ["", f"- Trainer total per unit: " + "; ".join(f"{u} {a:.1f}–{b:.1f} min" for u, (a, b) in s["trainer_total_min"].items())
          + f"; all 18 runs {s['total_6_models_3_units_h']:.2f} h.",
          f"- Training share of the trainer total: {100 * s['training_share_pooled']:.0f} % pooled; per unit "
          + "; ".join(f"{u} {100 * a:.0f}–{100 * b:.0f} %" for u, (a, b) in s["training_share"].items()) + ".",
          "- 256x64 vs 8x64, trainer total: " + ", ".join(f"{t} {100 * s['time_256_vs_8'][k]:+.1f} %" for k, _, t in MODELS)
          + "; GPU peak: " + ", ".join(f"{t} {100 * s['gpu_256_vs_8'][k]:+.1f} %" for k, _, t in MODELS) + ".",
          "", "## Sources", ""]
    for k, _, t in MODELS:
        L.append(f"- {t}: " + "; ".join(f"{r} `{p}`" for r, p in rec["sources"][k].items()))
    L += ["", "## Map files (release file → Hugging Face copy; same tiles)", "",
          "| model | unit | release sha256 | HF sha256 |", "|---|---|---|---|"]
    for k, _, t in MODELS:
        for u in UNITS:
            m = rec["maps"][k][u]
            L.append(f"| {t} | {u} | `{m['release_sha256']}` | `{m['hf_sha256']}` |")
    (OUT / "TABLES.md").write_text("\n".join(L) + "\n")


def write_tex(rec):
    L = ["% generated by experiments/paper_eval/tables_cpu.py; PPL WikiText-2 / C4; \\dag: not significantly better than FourOverSix",
         "\\begin{tabular}{l" + "c" * len(MODELS) + "r}", "\\toprule",
         "Method & " + " & ".join(t for _, _, t in MODELS) + " & Loss rec. \\\\", "\\midrule"]
    for r in ROWS:
        cells = []
        for k, _, _ in MODELS:
            w, c4 = rec["ppl"][k][r]["wiki"], rec["ppl"][k][r]["c4"]
            dw = DAG if dag(rec, k, r, "wiki") else ""
            dc = DAG if dag(rec, k, r, "c4") else ""
            cells.append(f"{w:.2f}{dw} / {c4:.2f}{dc}")
        L.append(f"{NAMES[r]} & " + " & ".join(cells) + f" & {100 * rec['loss_recovered'][r]:.1f}\\% \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_main_ppl.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
