"""P: the harness parity verdict (PROTOCOL.md): results/paper_eval/parity/{PARITY.md, parity.json}.

    python report_parity.py
"""
import json
import statistics
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval/parity")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "parity"
SHAPES = ["1x128", "1x256", "1x512", "1x1024", "1x2048", "1x4096", "1x8192", "4x2048"]
POLS = ["bf16", "fo6", "fq-16x64"]


def ms(h, pol, r, mode="graph"):
    rec = json.loads((RUN / h / pol / f"round{r}.json").read_text())
    return {s: rec[mode][s]["ms"] for s in SHAPES}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rounds = (1, 2, 3)
    t = {h: {p: {r: ms(h, p, r) for r in rounds} for p in POLS} for h in ("razer", "flipquant")}
    med = {h: {p: {s: statistics.median(t[h][p][r][s] for r in rounds) for s in SHAPES} for p in POLS} for h in t}
    ratio = {p: {s: 100 * (med["flipquant"][p][s] / med["razer"][p][s] - 1) for s in SHAPES} for p in POLS}
    over = {h: {s: 100 * (statistics.median(t[h]["fq-16x64"][r][s] / t[h]["fo6"][r][s] for r in rounds) - 1)
                for s in SHAPES} for h in t}
    gap = {s: over["flipquant"][s] - over["razer"][s] for s in SHAPES}
    crit = {p: dict(median=statistics.median(ratio[p].values()), worst=max(abs(v) for v in ratio[p].values()),
                    pass_=abs(statistics.median(ratio[p].values())) <= 1 and max(abs(v) for v in ratio[p].values()) <= 3)
            for p in POLS}
    crit["overhead"] = dict(median_gap_pt=statistics.median(gap.values()), pass_=abs(statistics.median(gap.values())) <= 0.5)
    verdict = all(c["pass_"] for c in crit.values())
    rec = dict(median_ms=med, flipquant_over_razer_pct=ratio, fq16_over_fo6_pct=over, overhead_gap_pt=gap, criteria=crit,
               verdict="PASS" if verdict else "FAIL")
    (OUT / "parity.json").write_text(json.dumps(rec, indent=1) + "\n")
    L = ["# P: prefill harness parity on Phi-4 (flipquant evaluation.latency vs NVFP4-RaZeR bench_prefill.py)", "",
         f"**Verdict: {rec['verdict']}** (PROTOCOL.md: per policy, flipquant / RaZeR within ±1 % at the median over the 8 "
         "shapes and ±3 % at every shape; the FlipQuant 16x64 / FourOverSix ratio within 0.5 points at the median).", "",
         "CUDA-graph ms, the median over 3 rounds of each process's median; both harnesses on build_V, the same paper "
         "16x64 map.", "",
         "| shape | " + " | ".join(f"{p} RaZeR | {p} flipquant | Δ %" for p in POLS) + " | FQ/FO6 RaZeR | FQ/FO6 flipquant | gap pt |",
         "|---" + "|---:" * (3 * len(POLS) + 3) + "|"]
    for s in SHAPES:
        cells = []
        for p in POLS:
            cells += [f"{med['razer'][p][s]:.2f}", f"{med['flipquant'][p][s]:.2f}", f"{ratio[p][s]:+.2f}"]
        L.append(f"| {s} | " + " | ".join(cells) + f" | {over['razer'][s]:+.2f} % | {over['flipquant'][s]:+.2f} % | "
                 f"{gap[s]:+.2f} |")
    L += ["", "| criterion | value | pass |", "|---|---:|---|"]
    for p in POLS:
        L.append(f"| {p}: median Δ over shapes / worst |Δ| | {crit[p]['median']:+.2f} % / {crit[p]['worst']:.2f} % | "
                 f"{crit[p]['pass_']} |")
    L.append(f"| FQ/FO6 gap, median over shapes | {crit['overhead']['median_gap_pt']:+.2f} pt | {crit['overhead']['pass_']} |")
    (OUT / "PARITY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
