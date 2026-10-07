"""A: Table latency (PROTOCOL.md) from the prefill records: results/paper_eval/latency/{LATENCY.md, latency.json,
table_latency.tex}.

Per (model, shape, policy): the median over rounds of each process's CUDA-graph median (ms), and the eager one.
Ratios are paired within rounds (round r's X / round r's reference), reported as the median [min, max] over rounds.

    python report_latency.py [ROOT]          # ROOT: latency (default) or smoke
"""
import json
import statistics
import sys
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval"
MODELS = [("qwen3-1.7b", "Qwen3-1.7B"), ("qwen3-8b", "Qwen3-8B"), ("mistral-7b", "Mistral-7B-Instruct-v0.3"),
          ("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("phi4-14b", "Phi-4"), ("qwen3.8-27b", "Qwen3.8-27B")]
POLS = ["bf16", "nvfp4", "fo6", "fo6-wB", "fq-8x64", "fq-16x64", "fq-256x64"]
NAMES = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "fo6-wB": "FourOverSix (wB)", "fq-8x64": "FlipQuant 8x64",
         "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64"}
COMPARE = [("fq-16x64", "fo6"), ("fq-8x64", "fo6"), ("fq-256x64", "fo6"), ("fq-8x64", "fo6-wB"), ("nvfp4", "fo6"),
           ("fo6-wB", "fo6"), ("bf16", "fo6")]


def load(root):
    data = {}
    for m, _ in MODELS:
        for p in POLS:
            for f in sorted((RUN / root / m / p).glob("round*.json")) if (RUN / root / m / p).exists() else []:
                r = json.loads(f.read_text())
                if r.get("status") == "complete":
                    data.setdefault(m, {}).setdefault(p, {})[int(f.stem[5:])] = r
    return data


def main(root="latency"):
    data = load(root)
    out = OUT / root
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(ms={}, eager_ms={}, ratios={}, rounds={}, capture={}, weights_gib={})
    shapes = None
    for m, _ in MODELS:
        if m not in data:
            continue
        for p, rr in data[m].items():
            shapes = shapes or list(next(iter(rr.values()))["eager"])
            mode = "graph" if all("ms" in r["graph"].get(s, {}) for r in rr.values() for s in shapes) else "eager"
            rec["capture"].setdefault(m, {})[p] = mode
            rec["ms"].setdefault(m, {})[p] = {s: statistics.median(r[mode][s]["ms"] for r in rr.values()) for s in shapes}
            rec["eager_ms"].setdefault(m, {})[p] = {s: statistics.median(r["eager"][s]["ms"] for r in rr.values())
                                                    for s in shapes}
            rec["rounds"].setdefault(m, {})[p] = sorted(rr)
            rec["weights_gib"].setdefault(m, {})[p] = statistics.median(r["weights_gib"] for r in rr.values())
        for a, b in COMPARE:
            if a in data[m] and b in data[m]:
                common = sorted(set(data[m][a]) & set(data[m][b]))
                mode = "graph" if rec["capture"][m][a] == rec["capture"][m][b] == "graph" else "eager"
                rows = {}
                for s in shapes:
                    v = [100 * (data[m][a][r][mode][s]["ms"] / data[m][b][r][mode][s]["ms"] - 1) for r in common]
                    rows[s] = dict(median=statistics.median(v), min=min(v), max=max(v), rounds=len(v))
                rec["ratios"].setdefault(m, {})[f"{a} vs {b}"] = dict(mode=mode, shapes=rows,
                                                                      median_over_shapes=statistics.median(
                                                                          x["median"] for x in rows.values()))
    (out / "latency.json").write_text(json.dumps(rec, indent=1) + "\n")
    L = [f"# Table latency: CUDA-graph prefill on the RTX PRO 6000 ({root})", "",
         "Medians over rounds of each process's median (ms); ratios paired within rounds, median [min, max] over rounds. "
         "Kernels: build_V. Harness: flipquant evaluation.latency prefill (PROTOCOL.md).", ""]
    for m, title in MODELS:
        if m not in rec["ms"]:
            continue
        L += [f"## {title}", "", "| shape | " + " | ".join(NAMES[p] for p in POLS if p in rec["ms"][m]) + " |",
              "|---" + "|---:" * len([p for p in POLS if p in rec["ms"][m]]) + "|"]
        for s in shapes:
            L.append(f"| {s} | " + " | ".join(f"{rec['ms'][m][p][s]:.2f}" for p in POLS if p in rec["ms"][m]) + " |")
        L += ["", "| shape | " + " | ".join(f"{NAMES[a]} vs {NAMES[b]}" for a, b in COMPARE if f"{a} vs {b}" in rec["ratios"].get(m, {}))
              + " |", "|---" + "|---:" * len(rec["ratios"].get(m, {})) + "|"]
        for s in shapes:
            L.append(f"| {s} | " + " | ".join(
                f"{x['shapes'][s]['median']:+.1f} % [{x['shapes'][s]['min']:+.1f}, {x['shapes'][s]['max']:+.1f}]"
                for k, x in rec["ratios"].get(m, {}).items()) + " |")
        modes = {p: v for p, v in rec["capture"][m].items() if v != "graph"}
        if modes:
            L += ["", f"Eager timings (no CUDA graph) for: {', '.join(modes)}."]
        L.append("")
    L += ["## Summary: the median over the 8 shapes of the paired ratio", "",
          "| model | " + " | ".join(f"{NAMES[a]} vs {NAMES[b]}" for a, b in COMPARE) + " |", "|---" + "|---:" * len(COMPARE) + "|"]
    for m, title in MODELS:
        if m in rec["ratios"]:
            L.append(f"| {title} | " + " | ".join(
                f"{rec['ratios'][m][f'{a} vs {b}']['median_over_shapes']:+.1f} %" if f"{a} vs {b}" in rec["ratios"][m]
                else "—" for a, b in COMPARE) + " |")
    (out / "LATENCY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-10:]))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["latency"]))
