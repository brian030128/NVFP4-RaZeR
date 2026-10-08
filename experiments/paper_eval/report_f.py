"""F (amendment 3): the hybrid models' main-table PPL re-measured in n16k64-fast, against the fallback-env numbers the
CPU tables use (BF16 / NVFP4 / FourOverSix: main-ppl 44f8cea's records; FlipQuant: the release records). The maps are
as calibrated under the fallback kernels. -> results/paper_eval/pplfast/{PPLFAST.md, pplfast.json}

Per (model, policy, corpus): PPL in each env, and the paired ΔNLL fast − fallback on the same windows (± 2 SE); then the
table's comparisons in each env: ΔNLL vs FourOverSix (± 2 SE) and the loss recovered over these two models' 4 pairs.

    python report_f.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables_cpu as T  # noqa: E402

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = T.ROOT / "results" / "paper_eval" / "pplfast"
HYBRID = [m for m in T.MODELS if m[0] in ("nemotron-nano-9b-v2", "qwen3.8-27b")]
POL = {"bf16": "bf16", "nvfp4": "nvfp4", "fo6": "fo6", "fq_8x64": "fq-8x64", "fq_16x64": "fq-16x64",
       "fq_256x64": "fq-256x64"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mp = T.git_json(T.MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    rec = dict(sources={}, ppl={}, fast_vs_fallback={}, vs_fo6={}, loss_recovered={}, missing=[])
    nll = {}
    for key, mkey, title in HYBRID:
        src = {r: mp[mkey]["sources"][r] for r in ("bf16", "nvfp4", "fo6")}
        src.update({f"fq_{u}": str(T.REL / key / "ppl" / f"flipquant_{u}.json") for u in T.UNITS})
        fast = {r: RUN / "pplfast" / key / f"{POL[r]}.json" for r in T.ROWS}
        rec["sources"][key] = dict(fallback=src, fast={r: str(p) for r, p in fast.items()})
        for r in T.ROWS:
            if not fast[r].exists() or "resources" not in json.loads(fast[r].read_text()):
                rec["missing"].append(f"{key} {r}")
                continue
            nll.setdefault(key, {})[r] = dict(fallback=T.nll_of(src[r]), fast=T.nll_of(fast[r]))
    L = ["# F: the hybrid models' main-table PPL in n16k64-fast vs the fallback env", "",
         "flipquant `evaluation.ppl --paper-convention` (NVFP4: `--act-scope row`, per-token NVFP4 scales, amendment "
         "11), the same windows; fallback = the records of the CPU tables (main-ppl 44f8cea for BF16 / NVFP4 / "
         "FourOverSix, the release records for FlipQuant); fast = n16k64-fast (amendments 3 and 5). The maps are as "
         "calibrated (under the fallback kernels). ✗ = |ΔNLL| beyond 2 SE.", "",
         "| model | policy | WikiText-2 fallback → fast | ΔNLL fast − fallback | C4 fallback → fast | ΔNLL fast − fallback |",
         "|---|---|---:|---:|---:|---:|"]
    for key, _, title in HYBRID:
        for r in T.ROWS:
            d = nll.get(key, {}).get(r)
            if not d:
                continue
            cells = []
            for c, _ in T.CORPORA:
                a, b = d["fallback"][c], d["fast"][c]
                pa, pb = math.exp(T.mean(a)), math.exp(T.mean(b))
                pr = T.paired(b, a)
                rec["ppl"].setdefault(key, {}).setdefault(r, {})[c] = dict(fallback=pa, fast=pb)
                rec["fast_vs_fallback"].setdefault(key, {}).setdefault(r, {})[c] = pr
                cells += [f"{pa:.4f} → {pb:.4f}", f"{pr['delta']:+.5f} ± {pr['two_se']:.5f}{' ✗' if pr['significant'] else ''}"]
            L.append(f"| {title} | {T.NAMES[r]} | " + " | ".join(cells) + " |")
    cells = [p for k in rec["fast_vs_fallback"].values() for r in k.values() for p in r.values()]
    if cells:
        rec["within_2se"] = sum(not p["significant"] for p in cells)
        L += ["", f"**{rec['within_2se']} of {len(cells)} fast − fallback differences are within 2 SE.**"]
    L += ["", "The table's comparisons in each env (ΔNLL vs FourOverSix, ± 2 SE; * = beyond 2 SE):", "",
          "| model | policy | WikiText-2 fallback | fast | C4 fallback | fast |", "|---|---|---:|---:|---:|---:|"]
    for key, _, title in HYBRID:
        if not all(r in nll.get(key, {}) for r in T.ROWS):
            continue
        for r in ("nvfp4",) + tuple(f"fq_{u}" for u in T.UNITS):
            cells = []
            for c, _ in T.CORPORA:
                for env in ("fallback", "fast"):
                    p = T.paired(nll[key][r][env][c], nll[key]["fo6"][env][c])
                    rec["vs_fo6"].setdefault(key, {}).setdefault(r, {}).setdefault(c, {})[env] = p
                    cells.append(f"{p['delta']:+.4f} ± {p['two_se']:.4f}{'*' if p['significant'] else ''}")
            L.append(f"| {title} | {T.NAMES[r]} | " + " | ".join(cells) + " |")
    done = [k for k, _, _ in HYBRID if all(r in nll.get(k, {}) for r in T.ROWS)]
    if done:
        pairs = [(k, c) for k in done for c, _ in T.CORPORA]
        for env in ("fallback", "fast"):
            m = {k: {r: {c: T.mean(nll[k][r][env][c]) for c, _ in T.CORPORA} for r in T.ROWS} for k in done}
            lost = sum(m[k]["nvfp4"][c] - m[k]["bf16"][c] for k, c in pairs)
            rec["loss_recovered"][env] = {r: sum(m[k]["nvfp4"][c] - m[k][r][c] for k, c in pairs) / lost
                                          for r in T.ROWS if r != "bf16"}
        lr = rec["loss_recovered"]
        L += ["", f"Loss recovered over these {len(pairs)} model–corpus pairs (Σ(NVFP4 − policy) / Σ(NVFP4 − BF16) in "
              "NLL), fallback → fast: " + "; ".join(f"{T.NAMES[r]} {100 * lr['fallback'][r]:.1f} % → "
                                                   f"{100 * lr['fast'][r]:.1f} %" for r in T.ROWS if r != "bf16") + "."]
    if rec["missing"]:
        L += ["", "Missing fast-env records: " + ", ".join(rec["missing"]) + "."]
    (OUT / "pplfast.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "PPLFAST.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
