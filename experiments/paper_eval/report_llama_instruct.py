"""Part P report (PROTOCOL.md part P): results/paper_eval/llama_instruct/{REPORT.md, llama_instruct.json, records/}.
Run after tables_final.py (whose final.json holds every row's per-cell statistics, the daggers and the mean capped NLL
recovery). CPU only.

    python report_llama_instruct.py
"""
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_llama_instruct as P  # noqa: E402
import tables_cpu as T  # noqa: E402
import tables_final as F  # noqa: E402

OUT = T.ROOT / "results" / "paper_eval" / "llama_instruct"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = json.loads((F.OUT / "final.json").read_text())
    rec = dict(models={}, capped_recovery=final.get("capped_recovery"), capped_pairs=final.get("capped_pairs"),
               capped_flags=final.get("capped_flags"))
    md = ["# Part P: Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct in tab:main-ppl", "",
          "PROTOCOL.md part P. Every row with the existing main-table settings, n16k64, native sm120 (build_V, "
          "`--kernel-set auto`) except IF4 / MixFP4 (Zou), simulated as part I. The final table is "
          "`results/paper_eval/final/` (tables_final.py). ΔNLL x 1e-3 ± 2 SE, * beyond 2 SE.", ""]
    for model, key, title in F.LLAMA:
        d = P.ROOT / model
        if not (d / "ppl" / "bf16.json").exists():
            md += [f"## {title}", "", "Not run (the HF license of meta-llama/Llama-3.2-1B-Instruct is not accepted for "
                   "the edgeai-lab account; the model stays stopped until it is).", ""]
            rec["models"][model] = dict(status="not run (gated)")
            continue
        m = rec["models"][model] = dict(maps={}, ppl=final["ppl"].get(model), vs_fo6=final["vs_fo6"].get(model),
                                        vs_nvfp4=final["vs_nvfp4"].get(model), daggers=final["daggers"].get(model))
        md += [f"## {title}", "", "| unit | wall (s) | trainer total (s) | device peak (MiB) | GPU allocated peak (GiB) | "
               "E0M3 share | fit extension |", "|---|---:|---:|---:|---:|---:|---|"]
        for unit in P.UNITS:
            r = json.loads((d / "records" / unit / "run.json").read_text())
            sys.path.insert(0, str(P.S.FQ2))
            from flipquant import maps as M
            tiles, _, _ = M.load(d / "maps" / f"flipquant_{unit}.pt")
            e0, n = sum(int(t.sum()) for t in tiles.values()), sum(t.numel() for t in tiles.values())
            m["maps"][unit] = dict(seconds=r["seconds"], device_peak_mib=r["device_peak_mib"], e0m3_tiles=e0, tiles=n,
                                   e0m3_share=e0 / n, trainer=r["trainer"], extension_ok=r["extension"]["ok"],
                                   map_sha256=r["map_sha256"], settings=r["settings"])
            md.append(f"| {unit} | {r['seconds']:.0f} | {r['trainer']['total_seconds']:.0f} | {r['device_peak_mib']:,} | "
                      f"{(r['trainer']['gpu_peak_allocated_gib'] or [0])[0]:.1f} | {e0:,} / {n:,} ({100 * e0 / n:.3f} %) | "
                      f"{'all checks pass' if r['extension']['ok'] else 'FAILED'} |")
            dst = OUT / "records" / model / unit
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("run.json", "reproduction.json", "trainer_report.json"):
                shutil.copyfile(d / "records" / unit / f, dst / f)
        w = json.loads((d / "windows.json").read_text()) if (d / "windows.json").exists() else {}
        m["windows"] = w
        if w:
            md += ["", "Calibration windows (256 x 512): the 128 of the paper rule (`builder_seed0` with this model's "
                   "tokenizer; not taken from an archived record) and the 128 of the extension rule (seed 20260930)."
                   + (f" Equal to base Llama-3.1-8B's release windows, token for token: fit {w.get('fit_equals_base')}, "
                      f"extension {w.get('extension_equals_base')}." if "fit_equals_base" in w else "")]
            shutil.copyfile(d / "windows.json", OUT / f"windows_{model}.json")
        md += ["", "| row | WikiText-2 | C4 | ΔNLL vs FourOverSix (wiki / C4) | FlipQuant vs NVFP4 (wiki / C4) |",
               "|---|---:|---:|---:|---:|"]
        for r in F.ROWS:
            if r not in (m["ppl"] or {}):
                md.append(f"| {F.NAMES[r]} | TBD | TBD | | |")
                continue
            p = m["ppl"][r]
            v = m["vs_fo6"].get(r)
            vs = " / ".join(f"{1e3 * v[c]['delta']:+.2f} ± {1e3 * v[c]['two_se']:.2f}{'*' if v[c]['significant'] else ''}"
                            for c, _ in T.CORPORA) if v else ""
            n4 = m["vs_nvfp4"].get(r)
            ns = " / ".join(f"{1e3 * n4[c]['delta']:+.2f} ± {1e3 * n4[c]['two_se']:.2f}{'*' if n4[c]['significant'] else ''}"
                            for c, _ in T.CORPORA) if n4 else ""
            dag = "†" if r.startswith("fq_") and any(m["daggers"][r[3:]][c] for c, _ in T.CORPORA) else ""
            md.append(f"| {F.NAMES[r]}{dag} | {p['wiki']:.4f} | {p['c4']:.4f} | {vs} | {ns} |")
        # the largest single-window differences vs FourOverSix per row (a cell driven by one window shows up here)
        src = final["sources"][model]
        fo6 = T.nll_of(src["fo6"])
        out = {}
        for r in F.ROWS:
            if r in src and r != "fo6":
                x = T.nll_of(src[r])
                for c, _ in T.CORPORA:
                    diffs = [a - b for a, b in zip(x[c], fo6[c])]
                    i = max(range(len(diffs)), key=lambda j: abs(diffs[j]))
                    out.setdefault(r, {})[c] = dict(window=i, delta_nats=diffs[i],
                                                    mean_without=(sum(diffs) - diffs[i]) / (len(diffs) - 1))
        m["largest_window_vs_fo6"] = out
        md += ["", "Largest single-window NLL difference vs FourOverSix (nats; window index), and the mean ΔNLL x 1e-3 "
               "without that window, C4: " + "; ".join(
                   f"{F.NAMES[r]} {v['c4']['delta_nats']:+.2f} (w{v['c4']['window']}; without it "
                   f"{1e3 * v['c4']['mean_without']:+.2f})" for r, v in out.items()) + ".", ""]
    cr = final.get("capped_recovery") or {}
    md += ["## Mean capped NLL recovery", "",
           f"Over the {len(final.get('capped_pairs') or [])} common pairs (the models with FOCUS; the coordinator's "
           "definition, per pair 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - NLL_BF16)), no floor, equal weight). "
           "Flagged pairs (NLL_NVFP4 - NLL_BF16 <= 0): " + (", ".join(final.get("capped_flags") or []) or "none") + ".", "",
           "| row | mean capped NLL recovery |", "|---|---:|"]
    md += [f"| {F.NAMES[r]} | {cr[r]['value']:.1f} % ({cr[r]['pairs']} pairs) |" for r in F.ROWS if r in cr]
    md += ["", "Per-pair values: `results/paper_eval/final/FINAL.md`. Updated table: "
           "`results/paper_eval/final/table_main_ppl_final.tex`."]
    (OUT / "llama_instruct.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
