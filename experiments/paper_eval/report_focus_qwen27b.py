"""Part Q report (PROTOCOL.md part Q): results/paper_eval/focus_qwen27b/{REPORT.md, focus_qwen27b.json, records/}.
Run after `run_focus_qwen27b.py ppl` and tables_final.py (whose final.json holds the paired statistics and the mean
capped NLL recovery). CPU only.

    python report_focus_qwen27b.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_focus_qwen27b as Q  # noqa: E402
import tables_cpu as T  # noqa: E402
import tables_final as F  # noqa: E402

OUT = T.ROOT / "results" / "paper_eval" / "focus_qwen27b"


def pm(p):
    return f"{1e3 * p['delta']:+.2f} ± {1e3 * p['two_se']:.2f}{'*' if p['significant'] else ''}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    k = Q.MODEL
    prov = json.loads((Q.WORK / "provenance.json").read_text())
    dl = json.loads((Q.WORK / "download.json").read_text())
    r = json.loads(Q.OUT.read_text())
    final = json.loads((F.OUT / "final.json").read_text())
    src = final["sources"][k]
    assert Path(src["focus"]) == Q.OUT, src.get("focus")
    nll = {row: T.nll_of(p) for row, p in src.items()}
    ppl = final["ppl"][k]
    vs = dict(fo6=final["vs_fo6"][k]["focus"],
              nvfp4={c: T.paired(nll["focus"][c], nll["nvfp4"][c]) for c, _ in T.CORPORA},
              bf16={c: T.paired(nll["focus"][c], nll["bf16"][c]) for c, _ in T.CORPORA},
              **{f"fq_{u}": final["focus_vs_fq"][k][u] for u in T.UNITS})
    plausible = {c: ppl["bf16"][c] < ppl["focus"][c] < ppl["nvfp4"][c] for c, _ in T.CORPORA}
    largest = {}
    for c, _ in T.CORPORA:
        d = [a - b for a, b in zip(nll["focus"][c], nll["fo6"][c])]
        i = max(range(len(d)), key=lambda j: abs(d[j]))
        largest[c] = dict(window=i, delta_nats=d[i], mean_without=(sum(d) - d[i]) / (len(d) - 1))
    # the mean capped NLL recovery: the new common pairs, and the previous ones (without this model) for reference
    per = final["capped_per_pair"]
    prev = {row: [v for p, v in vals.items() if not p.startswith(k + "/") and v is not None] for row, vals in per.items()}
    prev = {row: sum(v) / len(v) for row, v in prev.items() if v}
    rec = dict(provenance=prov, download=dl, evaluation=dict(
        record=str(Q.OUT), seconds=r["resources"]["seconds"], gpus=r["resources"]["gpus"],
        gpu_peak_gib=r["resources"]["gpu_peak_gib"], source=r["source"], focus=r["focus"], native=r["native"],
        convention=r["convention"], ppl={c: r["results"][c]["ppl"] for c, _ in T.CORPORA},
        windows={c: r["results"][c]["windows"] for c, _ in T.CORPORA}),
        ppl=ppl, focus_vs=vs, plausible=plausible, largest_window_vs_fo6=largest,
        capped_recovery=final["capped_recovery"], capped_pairs=final["capped_pairs"], capped_flags=final["capped_flags"],
        capped_per_pair=per, capped_recovery_previous_pairs=prev)
    a, h = prov["windows"], prov["hyperparameters"]
    md = ["# Part Q: FOCUS on Qwen3.8-27B (the state trained on the cluster, evaluated as part K)", "",
          "PROTOCOL.md part Q. ΔNLL x 1e-3 ± 2 SE (paired over the same windows), * beyond 2 SE; negative = FOCUS better.",
          "", "## Provenance of the state", "",
          f"- Repository `{prov['repo']}` @ `{prov['revision']}`, file `{dl['file']}`: {dl['bytes']:,} bytes, sha256 "
          f"`{dl['sha256']}` ({'equal to' if dl['sha256_equals_hub_lfs'] else 'DIFFERENT from'} the Hub's LFS sha256; "
          f"downloaded into the HF cache on /home in {dl['seconds']:.0f} s).",
          f"- Windows: the state's {a['state']} window token sha256 are {'identical, in order, to' if a['identical_in_order'] else 'NOT equal to'} "
          f"the {a['release']} release fit windows for {k} (the record's 128 fit windows, math then code, and the release "
          f"16x64 trainer report's 128 extension windows); its fit_extension.json records "
          f"{'equal' if a['extension_records_identical'] else 'DIFFER from'} that report's, settings "
          f"{'equal' if a['extension_settings_identical'] else 'DIFFERENT'}.",
          "- Hyperparameters equal to part K's (`run_sm120.FOCUS_COMMON`): "
          + ", ".join(f"{n} {v['state']}" + ("" if v["equal"] else " (DIFFERENT)") for n, v in h.items() if v["state"] != "")
          + ".",
          f"- Recorded deviation: {prov['deviations']['note']} (job {prov['deviations']['job']}, "
          f"{', '.join(prov['deviations']['gpus'] or [])}).",
          f"- Training: status {prov['status']}, {prov['steps']} steps, {prov['training_seconds']:.0f} s; KL-top "
          f"{prov['kl_top']['first']:.4f} -> {prov['kl_top']['last']:.4f}.", "",
          "## Evaluation", "",
          f"Part K's command: `evaluation.ppl --weight focus --focus <state> --focus-deploy --act fourover6 --act-scope row "
          f"--paper-convention`, native (build_V, `--kernel-set auto`), n16k64-fast (transformers "
          f"{r['source']['transformers']}, torch {r['source']['torch']}), revision `{r['source']['revision'][:8]}`. "
          f"One GPU ({', '.join(r['resources']['gpus'])}): {r['resources']['seconds']:.0f} s, torch peak "
          f"{max(r['resources']['gpu_peak_gib']):.1f} GiB; kernel `{r['native']['kernel']}`, "
          f"{r['native']['modules']} modules ({r['quantized_modules']} quantized).", "",
          "| row | WikiText-2 | C4 |", "|---|---:|---:|"]
    md += [f"| {F.NAMES[row]} | {ppl[row]['wiki']:.4f} | {ppl[row]['c4']:.4f} |" for row in F.ROWS if row in ppl]
    md += ["", "Plausibility (BF16 < FOCUS < NVFP4): " + ", ".join(f"{c} {v}" for c, v in plausible.items()) + ".", "",
           "## FOCUS against the other rows (paired ΔNLL x 1e-3 ± 2 SE)", "",
           "| FOCUS − | WikiText-2 | C4 |", "|---|---:|---:|"]
    md += [f"| {F.NAMES[row]} | " + " | ".join(pm(v[c]) for c, _ in T.CORPORA) + " |" for row, v in vs.items()]
    md += ["", "Largest single-window NLL difference vs FourOverSix (nats; window) and the mean ΔNLL x 1e-3 without it: "
           + "; ".join(f"{c} {v['delta_nats']:+.3f} (w{v['window']}; without it {1e3 * v['mean_without']:+.2f})"
                       for c, v in largest.items()) + ".", "",
           "## Mean capped NLL recovery", "",
           f"Over the {len(final['capped_pairs'])} common pairs (every model with all methods: "
           + ", ".join(final["capped_pairs"]) + "); per pair 100 * min(1, (NLL_NVFP4 - NLL_row) / (NLL_NVFP4 - "
           "NLL_BF16)), no floor, equal weight. Pairs with a non-positive gap (NLL_NVFP4 - NLL_BF16 <= 0): "
           + (", ".join(final["capped_flags"]) or "none") + ".", "",
           f"| row | {len(final['capped_pairs'])} pairs | the previous pairs (without {k}) |", "|---|---:|---:|"]
    cr = final["capped_recovery"]
    md += [f"| {F.NAMES[row]} | {cr[row]['value']:.1f} % | {prev[row]:.1f} % |" for row in F.ROWS if row in cr]
    md += ["", "Per pair:", "", "| row | " + " | ".join(final["capped_pairs"]) + " |",
           "|---|" + "---:|" * len(final["capped_pairs"])]
    md += [f"| {F.NAMES[row]} | " + " | ".join("—" if per[row].get(p) is None else f"{per[row][p]:.1f}"
                                               for p in final["capped_pairs"]) + " |" for row in F.ROWS if row in per]
    md += ["", "The updated table: `results/paper_eval/final/table_main_ppl_final.tex` (FINAL.md, final.json). "
           "Records: `records/` (the provenance check, the download, the PPL record, and the state's report.json / "
           "fit_extension.json from the Hub)."]
    dst = OUT / "records"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("provenance.json", "download.json", "report.json", "fit_extension.json"):
        shutil.copyfile(Q.WORK / f, dst / f)
    shutil.copyfile(Q.OUT, dst / "ppl.json")
    (OUT / "focus_qwen27b.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
