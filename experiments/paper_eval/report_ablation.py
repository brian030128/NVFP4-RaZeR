"""G (tab:ablation) report: results/paper_eval/ablation/{ABLATION.md, ablation.json, table_ablation_wiki.tex,
table_ablation_c4.tex}.

Rows per model: Random (seeds 0 / 1 / 2, NLL averaged per window), Activation-weighted, One-shot gradient, FlipQuant
(the release map, part F's record); columns 8x64 / 16x64 / 256x64. A cell is the paired ΔNLL against FourOverSix
(part F's record; check G0) on the same windows, x 1e-3 nats/token, ± 2 SE; every cell's PPL is listed too.

    python report_ablation.py
"""
import json
import math
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "ablation"
MODELS = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
UNITS = ("8x64", "16x64", "256x64")
ROWS = [("random", "Random"), ("act", "Activation-weighted"), ("oneshot", "One-shot gradient"), ("flipquant", "FlipQuant")]
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))


def load(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if "resources" in r else None


def nll(model, unit, row):
    """{corpus: [per-window NLL]} of a cell (Random: the per-window mean of the three seeds), or None."""
    if row == "flipquant":
        recs = [load(RUN / "pplfast" / model / f"fq-{unit}.json")]
    elif row == "random":
        recs = [load(RUN / "ablation" / model / unit / f"ppl_random{s}.json") for s in (0, 1, 2)]
    else:
        recs = [load(RUN / "ablation" / model / unit / f"ppl_{row}.json")]
    if any(r is None for r in recs):
        return None
    out = {}
    for c, _ in CORPORA:
        lists = [r["results"][c]["nll"] for r in recs]
        assert len({len(v) for v in lists}) == 1
        out[c] = [sum(x) / len(x) for x in zip(*lists)]
    return out


def paired(x, ref):
    assert len(x) == len(ref)
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta_milli=1e3 * m, two_se_milli=2e3 * se, n=n, significant=abs(m) > 2 * se)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(cells={}, ppl={}, maps={})
    md = ["# tab:ablation: how the map is chosen (Nemotron-Nano-9B-v2, Qwen3.8-27B)", "",
          "Native (build_V, --kernel-set auto, paper convention, n16k64-fast), flipquant paper-sm120-runs. Every "
          "baseline flips exactly the release map's k_l tiles per layer. Cells: paired ΔNLL vs FourOverSix, x 1e-3 "
          "nats/token, ± 2 SE (* = beyond 2 SE); Random = the per-window mean of seeds 0 / 1 / 2. FourOverSix and "
          "FlipQuant are part F's records (check G0: paper-sm120-runs reproduces them bit for bit).", ""]
    tex = {}
    for c, cname in CORPORA:
        md += [f"## {cname}", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
        lines = [f"% tab:ablation ({cname}): paired dNLL vs FourOverSix, x1e-3 nats/token, +-2SE; native sm120 "
                 "(build_V, kernel-set auto), per-token FourOverSix activations"]
        for model, title in MODELS:
            fo6 = load(RUN / "pplfast" / model / "fo6.json")
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\")
            for row, label in ROWS:
                cells, tcells = [], []
                for u in UNITS:
                    x = nll(model, u, row)
                    if x is None or fo6 is None:
                        cells.append("TBD"); tcells.append("TBD")
                        continue
                    p = paired(x[c], fo6["results"][c]["nll"])
                    rec["cells"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = p
                    rec["ppl"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = math.exp(
                        sum(x[c]) / len(x[c]))
                    cells.append(f"{p['delta_milli']:+.2f} ± {p['two_se_milli']:.2f}{'*' if p['significant'] else ''}")
                    tcells.append(f"${p['delta_milli']:.1f}$\\,{{\\scriptsize$\\pm${p['two_se_milli']:.1f}}}")
                md.append(f"| {title} | {label} | " + " | ".join(cells) + " |")
                lines.append(f"{label} & " + " & ".join(tcells) + " \\\\")
        md.append("")
        tex[c] = "\n".join(lines) + "\n"
        (OUT / f"table_ablation_{c}.tex").write_text(tex[c])
    md += ["## PPL per cell (WikiText-2 / C4)", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
    for model, title in MODELS:
        fo6 = load(RUN / "pplfast" / model / "fo6.json")
        if fo6:
            md.append(f"| {title} | FourOverSix | " + " | ".join(
                " / ".join(f"{math.exp(sum(fo6['results'][c]['nll']) / len(fo6['results'][c]['nll'])):.4f}"
                           for c, _ in CORPORA) for _ in UNITS) + " |")
        for row, label in ROWS:
            pp = rec["ppl"].get(model, {}).get(row, {})
            md.append(f"| {title} | {label} | " + " | ".join(
                " / ".join(f"{pp[u][c]:.4f}" for c, _ in CORPORA) if u in pp else "TBD" for u in UNITS) + " |")
    # FlipQuant against each baseline directly (paired on the same windows)
    md += ["", "## FlipQuant vs each baseline (paired ΔNLL FlipQuant − baseline, x 1e-3, ± 2 SE; negative = FlipQuant "
           "better)", "", "| model | unit | vs Random | vs Activation-weighted | vs One-shot (WikiText-2; C4) |",
           "|---|---|---:|---:|---:|"]
    for model, title in MODELS:
        for u in UNITS:
            fq = nll(model, u, "flipquant")
            if fq is None:
                continue
            cells = []
            for row in ("random", "act", "oneshot"):
                b = nll(model, u, row)
                if b is None:
                    cells.append("TBD")
                    continue
                ps = [paired(fq[c], b[c]) for c, _ in CORPORA]
                rec.setdefault("fq_vs", {}).setdefault(model, {}).setdefault(u, {})[row] = dict(zip(("wiki", "c4"), ps))
                cells.append("; ".join(f"{q['delta_milli']:+.2f} ± {q['two_se_milli']:.2f}{'*' if q['significant'] else ''}"
                                       for q in ps))
            md.append(f"| {title} | {u} | " + " | ".join(cells) + " |")
    # the maps' own records: k_l matching, positives / negatives, the one-shot run's settings
    for model, _ in MODELS:
        for u in UNITS:
            d = RUN / "ablation" / model / u
            for m in ("act", "oneshot"):
                r = load(d / f"{m}.json") or (json.loads((d / f"{m}.json").read_text()) if (d / f"{m}.json").exists() else None)
                if not r:
                    continue
                info = dict(status=r.get("status"))
                if m == "act":
                    info.update(windows=(r.get("data") or {}).get("fit_windows"),
                                layers_short_of_positive=r.get("layers_short_of_positive"))
                else:
                    meta = r.get("meta", {})
                    info.update(steps=(meta.get("oneshot") or {}).get("steps"),
                                layers_short_of_negative=r.get("layers_short_of_negative"),
                                settings={k: (meta.get("settings") or {}).get(k) for k in ("fit_windows", "teacher_topk",
                                                                                          "epochs", "lr", "init_logit")})
                rec["maps"].setdefault(model, {}).setdefault(u, {})[m] = info
    md += ["", "Maps: " + json.dumps(rec["maps"])[:2000], ""]
    (OUT / "ablation.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "ABLATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
