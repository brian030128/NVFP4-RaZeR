"""The final table main-ppl from per-window NLL (parts F, I, J, K): results/paper_eval/final/{FINAL.md, final.json,
table_main_ppl_final.tex}.

Sources per model (all on the same WikiText-2 / C4 windows; the paired statistics need the same windows):
- the four non-hybrid models (n16k64): BF16 / NVFP4 / FourOverSix from main-ppl 44f8cea's records, FlipQuant from the
  release records (as tables_cpu.py);
- Nemotron-Nano-9B-v2 and Qwen3.8-27B (n16k64-fast, amendment 3): part F's records for those four policies;
- IF4 and MixFP4 (Zou): part I's records; GPTQ‡: part J's; FOCUS: part K's (Qwen3.8-27B: TBD, it does not fit one GPU).
A row with a missing record is TBD in that cell; the loss recovered is then over the pairs it has (marked).

Statistics:
- loss recovered over the 12 model-corpus pairs: sum(NVFP4 - row) / sum(NVFP4 - BF16), mean NLL per pair;
- daggers: FlipQuant (each unit) NOT significantly better than FourOverSix (paired ΔNLL > -2 SE), per cell; and
  FlipQuant vs NVFP4 significance per cell;
- tile granularity: |8x64 - 16x64| paired ΔNLL per pair (max, mean, significance); 256x64's share of the 16x64 gain
  over FourOverSix, overall and per model.

    python tables_final.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables_cpu as T  # noqa: E402

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = T.ROOT / "results" / "paper_eval" / "final"
HYBRID = ("nemotron-nano-9b-v2", "qwen3.8-27b")
F_NAME = {"bf16": "bf16", "nvfp4": "nvfp4", "fo6": "fo6", "fq_8x64": "fq-8x64", "fq_16x64": "fq-16x64",
          "fq_256x64": "fq-256x64"}
EXTRA = {"if4": ("mainrows", "if4.json"), "zou": ("mainrows", "zou.json"), "gptq": ("gptqdd", "ppl.json"),
         "focus": ("focus", "ppl.json")}
ROWS = ["bf16", "nvfp4", "fo6", "if4", "zou", "gptq", "focus", "fq_8x64", "fq_16x64", "fq_256x64"]
NAMES = dict(T.NAMES, if4="IF4", zou="MixFP4 (Zou)", gptq="GPTQ‡", focus="FOCUS")
DDAG = "$^\\ddagger$"


def finished(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if "results" in r and "resources" not in r:
        return None
    return path


def sources(mp, key, mkey):
    src = {}
    if key in HYBRID:
        for r, f in F_NAME.items():
            src[r] = finished(RUN / "pplfast" / key / f"{f}.json")
    else:
        src.update({r: mp[mkey]["sources"][r] for r in ("bf16", "nvfp4", "fo6")})
        src.update({f"fq_{u}": str(T.REL / key / "ppl" / f"flipquant_{u}.json") for u in T.UNITS})
    for r, (part, name) in EXTRA.items():
        src[r] = finished(RUN / part / key / name)
    return {r: str(p) for r, p in src.items() if p}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mp = T.git_json(T.MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    rec = dict(sources={}, ppl={}, nll_mean={}, vs_fo6={}, vs_nvfp4={}, fq8_vs_fq16={}, missing=[])
    nll = {}
    for key, mkey, title in T.MODELS:
        src = sources(mp, key, mkey)
        rec["sources"][key] = src
        nll[key] = {r: T.nll_of(p) for r, p in src.items()}
        n0 = {c: len(nll[key]["bf16"][c]) for c, _ in T.CORPORA}
        for r, v in nll[key].items():
            assert all(len(v[c]) == n0[c] for c, _ in T.CORPORA), (key, r)
        rec["missing"] += [f"{key} {r}" for r in ROWS if r not in nll[key]]
        rec["nll_mean"][key] = {r: {c: T.mean(v[c]) for c, _ in T.CORPORA} for r, v in nll[key].items()}
        rec["ppl"][key] = {r: {c: math.exp(m) for c, m in d.items()} for r, d in rec["nll_mean"][key].items()}
        rec["vs_fo6"][key] = {r: {c: T.paired(nll[key][r][c], nll[key]["fo6"][c]) for c, _ in T.CORPORA}
                              for r in nll[key] if r not in ("fo6",)}
        rec["vs_nvfp4"][key] = {r: {c: T.paired(nll[key][r][c], nll[key]["nvfp4"][c]) for c, _ in T.CORPORA}
                                for r in nll[key] if r.startswith("fq_")}
        rec["fq8_vs_fq16"][key] = {c: T.paired(nll[key]["fq_8x64"][c], nll[key]["fq_16x64"][c]) for c, _ in T.CORPORA}
    pairs = [(k, c) for k, _, _ in T.MODELS for c, _ in T.CORPORA]
    nm = rec["nll_mean"]
    rec["loss_recovered"] = {}
    for r in ROWS:
        have = [(k, c) for k, c in pairs if r in nm[k]]
        if not have:
            continue
        lost = sum(nm[k]["nvfp4"][c] - nm[k]["bf16"][c] for k, c in have)
        rec["loss_recovered"][r] = dict(value=sum(nm[k]["nvfp4"][c] - nm[k][r][c] for k, c in have) / lost,
                                        pairs=len(have))
    # every row on the pairs FOCUS has (Qwen3.8-27B's FOCUS cell is TBD), for a like-for-like comparison
    fpairs = [(k, c) for k, c in pairs if "focus" in nm[k]]
    if fpairs:
        lost_f = sum(nm[k]["nvfp4"][c] - nm[k]["bf16"][c] for k, c in fpairs)
        rec["loss_recovered_focus_pairs"] = {r: sum(nm[k]["nvfp4"][c] - nm[k][r][c] for k, c in fpairs) / lost_f
                                             for r in ROWS if all(r in nm[k] for k, _ in fpairs)}
        rec["loss_recovered_focus_pairs_n"] = len(fpairs)
    # FOCUS against FlipQuant, and GPTQ-double-dagger against NVFP4 / FourOverSix, per cell (paired)
    rec["focus_vs_fq"], rec["gptq_vs"] = {}, {}
    for k, _, _ in T.MODELS:
        if "focus" in nll[k]:
            rec["focus_vs_fq"][k] = {u: {c: T.paired(nll[k]["focus"][c], nll[k][f"fq_{u}"][c]) for c, _ in T.CORPORA}
                                     for u in T.UNITS}
        if "gptq" in nll[k]:
            rec["gptq_vs"][k] = {ref: {c: T.paired(nll[k]["gptq"][c], nll[k][ref][c]) for c, _ in T.CORPORA}
                                 for ref in ("nvfp4", "fo6")}
    # daggers and FlipQuant vs NVFP4
    rec["daggers"] = {k: {u: {c: not (rec["vs_fo6"][k][f"fq_{u}"][c]["delta"] < 0 and
                                      rec["vs_fo6"][k][f"fq_{u}"][c]["significant"]) for c, _ in T.CORPORA}
                          for u in T.UNITS} for k, _, _ in T.MODELS}
    rec["fq_vs_nvfp4_significantly_better"] = {
        k: {u: {c: rec["vs_nvfp4"][k][f"fq_{u}"][c]["delta"] < 0 and rec["vs_nvfp4"][k][f"fq_{u}"][c]["significant"]
                for c, _ in T.CORPORA} for u in T.UNITS} for k, _, _ in T.MODELS}
    # tile granularity
    g8 = [rec["fq8_vs_fq16"][k][c] for k, c in pairs]
    d16 = {k: sum(rec["vs_fo6"][k]["fq_16x64"][c]["delta"] for c, _ in T.CORPORA) for k, _, _ in T.MODELS}
    d256 = {k: sum(rec["vs_fo6"][k]["fq_256x64"][c]["delta"] for c, _ in T.CORPORA) for k, _, _ in T.MODELS}
    rec["granularity"] = dict(
        abs_8x64_minus_16x64=dict(max=max(abs(p["delta"]) for p in g8), mean=T.mean([abs(p["delta"]) for p in g8]),
                                  mean_signed=T.mean([p["delta"] for p in g8]),
                                  significant=sum(p["significant"] for p in g8), pairs=len(g8),
                                  argmax=max(((k, c) for k, c in pairs),
                                             key=lambda kc: abs(rec["fq8_vs_fq16"][kc[0]][kc[1]]["delta"]))),
        share_256_of_16_overall=sum(d256.values()) / sum(d16.values()),
        share_256_of_16_per_model={k: d256[k] / d16[k] for k in d16})
    # markdown
    L = ["# Final table main-ppl (per-window NLL; parts F, I, J, K)", "",
         "PPL (WikiText-2 / C4). Non-hybrid models in n16k64 (main-ppl 44f8cea + the release FlipQuant records); "
         "Nemotron-Nano-9B-v2 and Qwen3.8-27B in n16k64-fast (part F). † = FlipQuant not significantly better than "
         "FourOverSix (paired ΔNLL > -2 SE).", "",
         "| row | " + " | ".join(t for _, _, t in T.MODELS) + " | loss recovered (12 pairs) |",
         "|---|" + "---:|" * (len(T.MODELS) + 1)]
    for r in ROWS:
        cells = []
        for k, _, _ in T.MODELS:
            if r not in rec["ppl"][k]:
                cells.append("TBD")
                continue
            dag = ""
            if r.startswith("fq_"):
                dag = "".join("†" if rec["daggers"][k][r[3:]][c] else "" for c, _ in T.CORPORA)
            cells.append(" / ".join(f"{rec['ppl'][k][r][c]:.4f}" for c, _ in T.CORPORA) + dag)
        lr = rec["loss_recovered"].get(r)
        lrs = "—" if lr is None else f"{100 * lr['value']:.1f} %" + ("" if lr["pairs"] == 12 else f" ({lr['pairs']} pairs)")
        L.append(f"| {NAMES[r]} | " + " | ".join(cells) + f" | {lrs} |")
    gr = rec["granularity"]
    L += ["", f"Tile granularity: |8x64 − 16x64| paired ΔNLL: max {gr['abs_8x64_minus_16x64']['max']:.4f} "
          f"({gr['abs_8x64_minus_16x64']['argmax']}), mean {gr['abs_8x64_minus_16x64']['mean']:.4f}, signed mean "
          f"{gr['abs_8x64_minus_16x64']['mean_signed']:+.4f}; significant in {gr['abs_8x64_minus_16x64']['significant']} "
          f"of {gr['abs_8x64_minus_16x64']['pairs']} pairs. 256x64 keeps {100 * gr['share_256_of_16_overall']:.1f} % of "
          "the 16x64 gain over FourOverSix overall; per model " + ", ".join(
              f"{k} {100 * v:.0f} %" for k, v in gr["share_256_of_16_per_model"].items()) + ".",
          "", "FlipQuant vs NVFP4: significantly better in " + str(sum(
              v for k in rec["fq_vs_nvfp4_significantly_better"].values() for u in k.values() for v in u.values()))
          + " of 36 cells; daggers (vs FourOverSix): " + str(sum(
              v for k in rec["daggers"].values() for u in k.values() for v in u.values())) + " of 36 cells."]
    if rec.get("loss_recovered_focus_pairs"):
        L += ["", f"Loss recovered over the {rec['loss_recovered_focus_pairs_n']} pairs FOCUS has (without Qwen3.8-27B): "
              + "; ".join(f"{NAMES[r]} {100 * v:.1f} %" for r, v in rec["loss_recovered_focus_pairs"].items()
                          if r not in ("bf16", "nvfp4")) + "."]
    if rec["focus_vs_fq"]:
        L += ["", "FOCUS − FlipQuant, paired ΔNLL x 1e-3 ± 2 SE (negative: FOCUS better; * beyond 2 SE), WikiText-2 / C4:", "",
              "| model | vs 8x64 | vs 16x64 | vs 256x64 |", "|---|---:|---:|---:|"]
        for k, d in rec["focus_vs_fq"].items():
            L.append(f"| {k} | " + " | ".join(" / ".join(
                f"{1e3 * d[u][c]['delta']:+.2f} ± {1e3 * d[u][c]['two_se']:.2f}{'*' if d[u][c]['significant'] else ''}"
                for c, _ in T.CORPORA) for u in T.UNITS) + " |")
    if rec["gptq_vs"]:
        L += ["", "GPTQ‡ − NVFP4 and GPTQ‡ − FourOverSix, paired ΔNLL x 1e-3 ± 2 SE, WikiText-2 / C4:", "",
              "| model | vs NVFP4 | vs FourOverSix |", "|---|---:|---:|"]
        for k, d in rec["gptq_vs"].items():
            L.append(f"| {k} | " + " | ".join(" / ".join(
                f"{1e3 * d[r][c]['delta']:+.2f} ± {1e3 * d[r][c]['two_se']:.2f}{'*' if d[r][c]['significant'] else ''}"
                for c, _ in T.CORPORA) for r in ("nvfp4", "fo6")) + " |")
    if rec["missing"]:
        L += ["", "Missing (TBD): " + ", ".join(rec["missing"]) + "."]
    # LaTeX: one row per method, WikiText-2 / C4 PPL per model, the loss recovered; daggers on FlipQuant cells
    tex = ["% table main-ppl (final): WikiText-2 / C4 PPL; loss recovered over the 12 model-corpus pairs (FOCUS: the 10 "
           "it has, Qwen3.8-27B TBD); \\dag: FlipQuant not significantly better than FourOverSix",
           "Method & " + " & ".join(t for _, _, t in T.MODELS) + " & Loss rec. \\\\"]
    for r in ROWS:
        cells = []
        for k, _, _ in T.MODELS:
            if r not in rec["ppl"][k]:
                cells.append("TBD")
                continue
            dag = "$^\\dag$" if r.startswith("fq_") and any(rec["daggers"][k][r[3:]][c] for c, _ in T.CORPORA) else ""
            cells.append(" / ".join(f"{rec['ppl'][k][r][c]:.2f}" for c, _ in T.CORPORA) + dag)
        lr = rec["loss_recovered"].get(r)
        lrs = "--" if lr is None else f"{100 * lr['value']:.1f}\\%" + ("" if lr["pairs"] == 12 else "$^{*}$")
        label = NAMES[r].replace("‡", DDAG)
        tex.append(f"{label} & " + " & ".join(cells) + f" & {lrs} \\\\")
    (OUT / "table_main_ppl_final.tex").write_text("\n".join(tex) + "\n")
    (OUT / "final.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "FINAL.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
