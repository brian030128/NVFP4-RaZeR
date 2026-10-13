"""Part W (PROTOCOL.md): results/paper_eval/latency_llama/{LATENCY.md, latency.json}, A's format, from the records of
run_latency_llama.py, plus the Mistral session-drift check against A (58fe644), the combined 1x2048 table of the six
paper models, and the six-model summaries.

    python report_latency_llama.py
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_latency as RL  # noqa: E402

A_COMMIT = "58fe644"
ROOT = "latency_llama"
NEW = [("llama3.2-1b-instruct", "Llama-3.2-1B-Instruct"), ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct")]
DRIFT = ("mistral-7b", "Mistral-7B-Instruct-v0.3 (session-drift check: fo6 and fq-16x64 only)")
SIX = [("llama3.2-1b-instruct", "Llama-3.2-1B-Instruct", "W"), ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct", "W"),
       ("mistral-7b", "Mistral-7B-Instruct-v0.3", "A"), ("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2", "A"),
       ("phi4-14b", "Phi-4", "A"), ("qwen3.8-27b", "Qwen3.8-27B", "A")]
ROWS = ["bf16", "nvfp4", "fo6", "fq-16x64", "fq-8x64", "fq-256x64"]
SHAPE = "1x2048"


def a_record():
    root = Path(__file__).resolve().parents[2]
    return json.loads(subprocess.run(["git", "-C", str(root), "show", f"{A_COMMIT}:results/paper_eval/latency/latency.json"],
                                     capture_output=True, text=True, check=True).stdout)


def main():
    RL.MODELS[:] = [*NEW, DRIFT]
    RL.main(ROOT)                                        # A's tables and paired ratios for the new records
    out = RL.OUT / ROOT
    rec = json.loads((out / "latency.json").read_text())
    md = (out / "LATENCY.md").read_text().rstrip("\n").split("\n")
    a = a_record()
    w_data = RL.load(ROOT)
    md[0] = "# Table latency, part W: CUDA-graph prefill on the RTX PRO 6000 (Llama Instruct models; Mistral drift check)"
    md.insert(2, f"PROTOCOL.md part W: A's protocol (58fe644) unchanged except the model list; flipquant paper-sm120-runs "
                 "6f5ad27 (its timed path is A's); records under RUN/latency_llama.")
    md.insert(3, "")
    L = []
    # ---- the drift check
    m = "mistral-7b"
    drift = {}
    L += ["", "## Session-drift check: Mistral-7B-Instruct-v0.3 against A (58fe644)", "",
          "Median over rounds (graph ms), part W against A, and the difference; A's Mistral numbers are not replaced.", "",
          "| shape | fo6 W | fo6 A | Δ | fq-16x64 W | fq-16x64 A | Δ | fq-16x64 vs fo6, W | the same, A |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    shapes = list(rec["ms"][m]["fo6"])
    for s in shapes:
        cells, drift[s] = [], {}
        for p in ("fo6", "fq-16x64"):
            w_ms, a_ms = rec["ms"][m][p][s], a["ms"][m][p][s]
            drift[s][p] = dict(w=w_ms, a=a_ms, pct=100 * (w_ms / a_ms - 1))
            cells += [f"{w_ms:.2f}", f"{a_ms:.2f}", f"{100 * (w_ms / a_ms - 1):+.2f} %"]
        rw = rec["ratios"][m]["fq-16x64 vs fo6"]["shapes"][s]
        ra = a["ratios"][m]["fq-16x64 vs fo6"]["shapes"][s]
        drift[s]["ratio"] = dict(w=rw, a=ra)
        cells += [f"{rw['median']:+.1f} % [{rw['min']:+.1f}, {rw['max']:+.1f}]",
                  f"{ra['median']:+.1f} % [{ra['min']:+.1f}, {ra['max']:+.1f}]"]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    allp = [drift[s][p]["pct"] for s in shapes for p in ("fo6", "fq-16x64")]
    L += ["", f"Cells against A: median {statistics.median(allp):+.2f} %, range [{min(allp):+.2f}, {max(allp):+.2f}] % "
          f"(16 cells); the fq-16x64 vs fo6 ratio's median over shapes: W "
          f"{rec['ratios'][m]['fq-16x64 vs fo6']['median_over_shapes']:+.2f} %, A "
          f"{a['ratios'][m]['fq-16x64 vs fo6']['median_over_shapes']:+.2f} %."]
    rec["drift_check"] = dict(model=m, a_commit=A_COMMIT, cells=drift,
                              summary=dict(median_pct=statistics.median(allp), min_pct=min(allp), max_pct=max(allp)))
    # ---- the combined 1x2048 table of the six paper models
    src = {mm: (rec if who == "W" else a) for mm, _, who in SIX}
    raw = {mm: {p: src[mm]["ms"][mm][p][SHAPE] for p in ROWS} for mm, _, _ in SIX}
    rec["combined_1x2048"] = dict(models=[mm for mm, _, _ in SIX], source={mm: who for mm, _, who in SIX}, ms=raw)
    L += ["", f"## Combined {SHAPE} table: the six paper models (A's records for Mistral, Nemotron, Phi-4 and Qwen3.8-27B; "
          "part W's for the two Llama models)", "",
          "Raw medians over rounds (graph ms):", "",
          "| policy | " + " | ".join(t for _, t, _ in SIX) + " |", "|---" + "|---:" * len(SIX) + "|"]
    for p in ROWS:
        L.append(f"| {RL.NAMES[p]}{' (extra)' if p == 'fq-256x64' else ''} | "
                 + " | ".join(f"{raw[mm][p]:.4f}" for mm, _, _ in SIX) + " |")
    L += ["", "LaTeX, each cell rounded once to one decimal from the raw value above:", "", "```latex"]
    for p in ROWS:
        L.append(f"{RL.NAMES[p]} & " + " & ".join(f"{raw[mm][p]:.1f}" for mm, _, _ in SIX) + r" \\")
    L += ["```"]
    # ---- six-model summaries
    ratios = {mm: src[mm]["ratios"][mm] for mm, _, _ in SIX}
    summ = {}
    L += ["", "## Six-model summaries", ""]
    for key in ("fq-16x64 vs fo6", "fq-8x64 vs fo6"):
        per = {mm: [ratios[mm][key]["shapes"][s]["median"] for s in ratios[mm][key]["shapes"]] for mm, _, _ in SIX}
        allv = [v for vv in per.values() for v in vv]
        med = {mm: ratios[mm][key]["median_over_shapes"] for mm, _, _ in SIX}
        at = {mm: ratios[mm][key]["shapes"][SHAPE]["median"] for mm, _, _ in SIX}
        a_p, b_p = key.split(" vs ")
        rom = {mm: 100 * (raw[mm][a_p] / raw[mm][b_p] - 1) for mm, _, _ in SIX}
        summ[key] = dict(range_all=[min(allv), max(allv)], per_model_median=med, at_1x2048_paired=at,
                         at_1x2048_ratio_of_medians=rom)
        L += [f"**{RL.NAMES[a_p]} vs {RL.NAMES[b_p]}** (paired within rounds; the per-shape value is the median over rounds):",
              f"- Over all 8 shapes and six models: [{min(allv):+.2f}, {max(allv):+.2f}] %.",
              "- Per-model median over the 8 shapes: " + "; ".join(f"{t} {med[mm]:+.2f} %" for mm, t, _ in SIX) + ".",
              f"- At {SHAPE}, the paired ratio: [{min(at.values()):+.2f}, {max(at.values()):+.2f}] % ("
              + "; ".join(f"{t} {at[mm]:+.2f}" for mm, t, _ in SIX) + ").",
              f"- At {SHAPE}, the ratio of the medians: [{min(rom.values()):+.2f}, {max(rom.values()):+.2f}] % ("
              + "; ".join(f"{t} {rom[mm]:+.2f}" for mm, t, _ in SIX) + ").", ""]
    sp = {mm: raw[mm]["bf16"] / raw[mm]["fq-16x64"] for mm, _, _ in SIX}
    summ["bf16_over_fq16_1x2048"] = sp
    L += [f"**BF16 / FlipQuant 16x64 at {SHAPE}** (ratio of the medians): [{min(sp.values()):.2f}, {max(sp.values()):.2f}] x ("
          + "; ".join(f"{t} {sp[mm]:.2f}x" for mm, t, _ in SIX) + ")."]
    rec["six_model"] = summ
    (out / "latency.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "LATENCY.md").write_text("\n".join(md + L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
