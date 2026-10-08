"""H (tab:ptq) and I (IF4 / MixFP4 (Zou), simulated) reports.

    python report_ptq.py ptq        # results/paper_eval/ptq/{PTQ.md, ptq.json, table_ptq.tex}
    python report_ptq.py mainrows   # results/paper_eval/mainrows/{MAINROWS.md, mainrows.json}

tab:ptq: Nemotron-Nano-9B-v2 and Qwen3.8-27B at 16x64, rows {RTN, GPTQ, Hadamard} x {NVFP4 (NVFP4 activations),
FourOverSix, FlipQuant 16x64}: WikiText-2 / C4 PPL, and the paired ΔNLL vs RTN FourOverSix (x 1e-3, ± 2 SE). RTN = part
F's records. Check H1: the NVFP4 GPTQ codes equal under NVFP4 / FourOverSix activations.

mainrows: IF4 and MixFP4 (Zou), fake, 1x16, the methods' own rules, all six models; the spot check against main-ppl
44f8cea's records for the four non-hybrid models (per-window NLL and the installed weight sha256, bit for bit).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables_cpu as T  # noqa: E402

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = T.ROOT / "results" / "paper_eval"
HYB = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
FMTS = [("nvfp4", "NVFP4"), ("fo6", "FourOverSix"), ("fq-16x64", "FlipQuant (16$\\times$64)")]


def load(p):
    try:
        r = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
    return r if "resources" in r else None


def ppl_of(r, c):
    v = r["results"][c]["nll"]
    return math.exp(sum(v) / len(v))


def ptq():
    out = OUT / "ptq"
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(rows={}, check_h1={})
    md = ["# tab:ptq: FlipQuant with PTQ methods (16x64; PPL only)", "",
          "Native (build_V, --kernel-set auto), n16k64-fast, flipquant paper-sm120-runs. RTN: part F's records. GPTQ: "
          "fixed RTN grid, block 128, damp 0.01, no act-order, the release 256 x 512 fit set, BF16 propagation (the "
          "user's decision). Hadamard: block 16, applied in PyTorch before the native kernel (not fused); FlipQuant's "
          "map retrained in the rotated basis with the release settings. NVFP4 rows: NVFP4 activations (per-token "
          "scales); the others: per-token FourOverSix. Cells: WikiText-2 / C4 PPL; ΔNLL vs RTN FourOverSix x 1e-3 ± 2 SE "
          "(* beyond 2 SE).", "", "| model | method | format | WikiText-2 | C4 | ΔNLL wiki | ΔNLL C4 |",
          "|---|---|---|---:|---:|---:|---:|"]
    tex = ["% tab:ptq: WikiText-2 / C4 PPL at 16x64 (native sm120); GPTQ with BF16 propagation; Hadamard block 16 "
           "(rotation in PyTorch before the kernel, not fused)"]
    for model, title in HYB:
        ref = load(RUN / "pplfast" / model / "fo6.json")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for method in ("rtn", "gptq", "hadamard"):
            for fmt, label in FMTS:
                path = (RUN / "pplfast" / model / f"{fmt}.json" if method == "rtn" else
                        RUN / "ptq" / model / f"{method}_{fmt}.json")
                r = load(path)
                if r is None:
                    md.append(f"| {title} | {method} | {label} | TBD | TBD | | |")
                    tex.append(f"{method.upper() if method != 'hadamard' else 'Hadamard'} & {label} & TBD & TBD \\\\")
                    continue
                cells = {c: ppl_of(r, c) for c, _ in T.CORPORA}
                d = {c: T.paired(r["results"][c]["nll"], ref["results"][c]["nll"]) for c, _ in T.CORPORA} if ref else {}
                rec["rows"].setdefault(model, {}).setdefault(method, {})[fmt] = dict(
                    ppl=cells, vs_rtn_fo6=d, source=str(path), ptq=r.get("ptq"))
                ds = [f"{1e3 * d[c]['delta']:+.2f} ± {1e3 * d[c]['two_se']:.2f}{'*' if d[c]['significant'] else ''}"
                      for c, _ in T.CORPORA] if d else ["", ""]
                md.append(f"| {title} | {method} | {label} | {cells['wiki']:.4f} | {cells['c4']:.4f} | " + " | ".join(ds) + " |")
                mname = {"rtn": "RTN", "gptq": "GPTQ", "hadamard": "Hadamard"}[method]
                tex.append(f"{mname} & {label} & {cells['wiki']:.2f} & {cells['c4']:.2f} \\\\")
        a, b = (load(RUN / "ptq" / model / f"gptq_{r}.json") for r in ("nvfp4", "nvfp4-fo6"))
        if a and b:
            sa, sb = (x.get("ptq", {}).get("codes_sha256") for x in (a, b))
            rec["check_h1"][model] = dict(nvfp4_act=sa, fo6_act=sb, equal=sa == sb and sa is not None)
    md += ["", "Check H1 (NVFP4 GPTQ codes under NVFP4 / FourOverSix activations): " + json.dumps(rec["check_h1"]), ""]
    (out / "ptq.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (out / "PTQ.md").write_text("\n".join(md) + "\n")
    (out / "table_ptq.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


def mainrows():
    out = OUT / "mainrows"
    out.mkdir(parents=True, exist_ok=True)
    mp = T.git_json(T.MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    rec = dict(rows={}, spot_check={})
    md = ["# IF4 and MixFP4 (Zou), simulated, the methods' own rules (1x16, own E2M1, per-token activation scales)", "",
          "flipquant paper-sm120-runs `evaluation.ppl --mode fake --weight if4 / zou_mixfp4 --act-method own --unit "
          "1x16`; the hybrid models in n16k64-fast. Spot check (non-hybrid models): the main-ppl 44f8cea records (the "
          "NVFP4-RaZeR implementation), per-window NLL and installed weight sha256, bit for bit.", "",
          "| model | IF4 WikiText-2 / C4 | MixFP4 (Zou) WikiText-2 / C4 | spot check |", "|---|---:|---:|---|"]
    for key, mkey, title in T.MODELS:
        cells, checks = [], []
        for row in ("if4", "zou"):
            r = load(RUN / "mainrows" / key / f"{row}.json")
            if r is None:
                cells.append("TBD")
                continue
            ppl = {c: ppl_of(r, c) for c, _ in T.CORPORA}
            rec["rows"].setdefault(key, {})[row] = dict(ppl=ppl, installed_weight_sha256=(r.get("adaptive") or {}).get(
                "installed_weight_sha256"))
            cells.append(" / ".join(f"{ppl[c]:.4f}" for c, _ in T.CORPORA))
            if key not in ("nemotron-nano-9b-v2", "qwen3.8-27b"):
                ref_path = mp[mkey]["sources"].get(row)
                ref = json.loads(Path(ref_path).read_text())
                ev = next(iter(ref["evaluations"].values()))
                same_nll = all(r["results"][c]["nll"] == ev["evaluation"][c]["nll"] for c, _ in T.CORPORA)
                ws_new = rec["rows"][key][row]["installed_weight_sha256"]
                ws_ref = ev.get("installed_weight_sha256")
                rec["spot_check"].setdefault(key, {})[row] = dict(nll_bit_identical=same_nll, weight_sha_new=ws_new,
                                                                  weight_sha_ref=ws_ref,
                                                                  weight_sha_equal=ws_new == ws_ref if ws_new and ws_ref else None,
                                                                  reference=ref_path)
                checks.append(f"{row}: NLL {'=' if same_nll else '≠'}, weights "
                              f"{'=' if ws_new == ws_ref else '≠' if ws_new and ws_ref else '?'}")
        hybrid = key in ("nemotron-nano-9b-v2", "qwen3.8-27b")
        note = "n/a (hybrid: n16k64-fast)" if hybrid else "; ".join(checks) or "pending"
        md.append(f"| {title} | " + " | ".join(cells) + f" | {note} |")
    (out / "mainrows.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "MAINROWS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"ptq": ptq, "mainrows": mainrows}[sys.argv[1]]()
