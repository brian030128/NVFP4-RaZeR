"""Part X amendment 2 "X2" (PROTOCOL.md): the Qwen3 zero-shot report in results/paper_eval/zeroshot6_qwen3/ (REPORT.md,
zeroshot6_qwen3.json). Part X's tables, paired tests and old-vs-new rule (report_zeroshot6's functions) for Qwen3-1.7B
and Qwen3-8B.

    python report_zs6_qwen3.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_zeroshot6 as R  # noqa: E402
import run_gpu as G  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402
import run_zs6_qwen3 as X2  # noqa: E402

OUT = G.RZ / "results" / "paper_eval" / "zeroshot6_qwen3"
MODELS = X2.MODELS
TITLES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B"}
TASKS = R.TASK_NAMES
NAMES = R.NAMES


def dagger(row):
    return "†" if row in ("if4", "zou") else ""


def x2_interval(qt):
    """X2's wall-clock interval (the first START to the last END of its 20 runs) and part Y's runs around it."""
    runs = {k: v for k, v in qt.items() if any(k == f"zs6_panel_{m}_{r}" for m in MODELS for r in Z.ROWS)}
    starts = sorted(v["start"] for v in runs.values() if "start" in v)
    ends = sorted(v["end"] for v in runs.values() if "end" in v)
    return dict(first_start=starts[0] if starts else None, last_end=ends[-1] if ends else None, runs=len(runs))


def main():
    data, rec = {}, dict(models={}, checks={}, protocol="PROTOCOL.md part X amendment 2 (X2)")
    for m in MODELS:
        for row in Z.ROWS:
            r = R.load(Z.ZS / "panel" / m / f"{row}.json")
            if r is not None and r.get("status") == "complete":
                data.setdefault(m, {})[row] = r
    R.OLD_MODELS.clear()
    R.OLD_MODELS.update({"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B"})
    old = R.old_panel()
    qt = R.queue_times()
    pre = R.load(X2.X2 / "preflight.json") or {}
    md = ["# Zero-shot accuracy: six tasks, Qwen3-1.7B and Qwen3-8B, ten rows (part X amendment 2, X2)", "",
          "Part X's protocol, unchanged, for two more models (PROTOCOL.md part X amendment 2):",
          "- **Harness:** flipquant origin/main `3db0072`, `evaluation.downstream`, in the same fresh detached worktree, "
          "unmodified.",
          "- **Settings:** lm-eval 0.4.11, 0-shot, batch 16, per-token activations. Datasets are pinned, LAMBADA at "
          "900124bf.",
          "- **Execution:** native rows on the RTX PRO 6000 (build_V, `--kernel-set auto`). IF4 and MixFP4 (Zou) are "
          "simulated with their own rules (†), as in tab:main-ppl.",
          "- **Checks:** part X's check_record on every run. Every input was re-hashed before its run against the "
          "registered records.",
          "- Every cell is a new run. Part X's report for the other six models is `results/paper_eval/zeroshot6/`.", "",
          "**Columns.** Accuracy in %: acc_norm for PIQA, HellaSwag, ARC-e and ARC-c; acc for WinoGrande and LAMBADA. "
          "Avg. is the unweighted mean of the six.",
          f"- **Paired columns:** they pool every question of the six tasks ({sum(Z.SIZES.values()):,} per row). Each "
          "gives the difference in points, the row-only / reference-only correct counts and the exact McNemar p.",
          "- `*` = p < 0.05 (acc_panel_tables.py's formula).", "",
          "| model | method | " + " | ".join(n for _, n in TASKS) + " | Avg. | vs FourOverSix | vs BF16 |",
          "|---|---|" + "---:|" * (len(TASKS) + 1) + "---|---|"]
    for m in MODELS:
        rows = data.get(m, {})
        mrec = rec["models"][m] = {}
        for row in Z.ROWS:
            r = rows.get(row)
            if r is None:
                md.append(f"| {TITLES[m]} | {NAMES[row]} | " + " | ".join("TBD" for _ in TASKS) + " | TBD | | |")
                continue
            accs = {t: R.acc(r, t) for t, _ in TASKS}
            avg = sum(accs.values()) / len(accs)
            cells, cmp = [], {}
            for ref in ("fo6", "bf16"):
                if row == ref or ref not in rows:
                    cells.append("—" if row == ref else "")
                    continue
                pr = R.paired(r, rows[ref])
                cmp[ref] = pr
                cells.append(f"{pr['diff']:+.2f}{'*' if pr['p'] < 0.05 else ''} ({pr['row_only']}/{pr['ref_only']}, "
                             f"p={pr['p']:.2g})")
            mrec[row] = dict(acc=accs, avg=avg, paired=cmp,
                             lm_eval={t: dict(r["tasks"][t]["metrics"]) for t, _ in TASKS},
                             timing=dict(load_seconds=r.get("load_seconds"), seconds=r.get("seconds"),
                                         tasks={t: r["tasks"][t]["seconds"] for t, _ in TASKS},
                                         queue=qt.get(f"zs6_panel_{m}_{row}")),
                             sample_digest={t: r["tasks"][t]["sample_digest"] for t, _ in TASKS},
                             peak_gib=max(r["tasks"][t]["peak_gpu_allocated_gib"] for t, _ in TASKS),
                             tokenization=r.get("tokenization"))
            md.append(f"| {TITLES[m]} | {NAMES[row]}{dagger(row)} | " + " | ".join(f"{accs[t]:.2f}" for t, _ in TASKS)
                      + f" | {avg:.2f} | " + " | ".join(cells) + " |")
    complete = [m for m in MODELS if len(rec["models"][m]) == len(Z.ROWS)]
    S = ["## Summary", "",
         "Avg. of the six tasks (%) per model, with the pooled paired difference from FourOverSix in points (* = exact "
         "McNemar p < 0.05; the full table has the counts and p). Mean = the unweighted mean of the two models' Avg.", "",
         "| method | " + " | ".join(TITLES[m] for m in MODELS) + " | mean |", "|---|" + "---:|" * (len(MODELS) + 1)]
    rec["summary"] = {}
    for row in Z.ROWS:
        cells = []
        for m in MODELS:
            v = rec["models"][m].get(row)
            if v is None:
                cells.append("TBD")
                continue
            pr = v["paired"].get("fo6")
            cells.append(f"{v['avg']:.2f}" + (f" ({pr['diff']:+.2f}{'*' if pr['p'] < 0.05 else ''})" if pr else ""))
        mean = sum(rec["models"][m][row]["avg"] for m in complete) / len(complete) if complete else None
        rec["summary"][row] = dict(mean_avg=mean)
        S.append(f"| {NAMES[row]}{dagger(row)} | " + " | ".join(cells) + (f" | {mean:.2f} |" if mean is not None else " | |"))
    at = next(i for i, line in enumerate(md) if line.startswith("| model | method | PIQA"))
    md[at:at] = S + ["", "## Full table", ""]
    # sample digests: within each model (the driver's check) and against part X's models
    px = R.load(R.OUT / "zeroshot6.json") or {}
    ref = {t: {r["sample_digest"][t] for m in (px.get("models") or {}).values() for r in m.values()} for t, _ in TASKS}
    dig = {t: {rec["models"][m][r]["sample_digest"][t] for m in rec["models"] for r in rec["models"][m]} for t, _ in TASKS}
    rec["checks"]["sample_digest_equal_within_and_across_x2"] = {t: len(v) == 1 for t, v in dig.items()}
    rec["checks"]["sample_digest_equal_to_part_x"] = {t: dig[t] == ref[t] for t, _ in TASKS}
    md += ["", "**Sample digests.** Every row of a model saw the same samples: equal sample_digest per task, checked by "
           "the driver.",
           "- Across the two models: " + ", ".join(f"{n} {'equal' if rec['checks']['sample_digest_equal_within_and_across_x2'][t] else 'differ'}"
                                                   for t, n in TASKS) + ".",
           "- Against part X's six models: " + ", ".join(f"{n} {'equal' if rec['checks']['sample_digest_equal_to_part_x'][t] else 'differ'}"
                                                          for t, n in TASKS) + "."]
    md += ["", "## LaTeX", "", "```latex"]
    for m in MODELS:
        md.append(f"\\midrule\\multicolumn{{8}}{{l}}{{\\textit{{{TITLES[m]}}}}} \\\\")
        for row in Z.ROWS:
            v = rec["models"][m].get(row)
            if v is not None:
                md.append(f"{R.LATEX[row]} & " + " & ".join(f"{v['acc'][t]:.2f}" for t, _ in TASKS)
                          + f" & {v['avg']:.2f} \\\\")
    md += ["```", "", "The columns: method, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, LAMBADA, Avg. Values are rounded "
           "once, from the per-example records."]
    # old vs new (part X's rule)
    md += ["", "## Old vs new: the co-author's acc_panel (flipquant main docs/results/acc_panel/acc_panel.md)", "",
           "- **Old:** fake quant on H200 (scripts/run_main_acc.py, run_small_acc.py). Only the table's rounded "
           "percentages exist; there are no per-example records.",
           "- **New:** this run, with native rows on the RTX PRO 6000.",
           "- Δ = new − old in points. 2 SE = 2 sqrt(p1 q1 / n + p2 q2 / n) (binomial, unpaired). ✗ = |Δ| > 2 SE.", "",
           "| model | method | " + " | ".join(n for _, n in TASKS) + " | Avg. Δ |", "|---|---|" + "---|" * (len(TASKS) + 1)]
    rec["old_vs_new"], flagged = {}, 0
    for m, rows in old.items():
        for row, o in rows.items():
            v = rec["models"].get(m, {}).get(row)
            if v is None:
                continue
            cells, entry = [], {}
            for t, _ in TASKS:
                n = Z.SIZES[t]
                p1, p2 = v["acc"][t] / 100, o[t] / 100
                se2 = 2 * 100 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
                d = v["acc"][t] - o[t]
                entry[t] = dict(new=v["acc"][t], old=o[t], diff=d, two_se=se2, beyond=abs(d) > se2)
                flagged += abs(d) > se2
                cells.append(f"{v['acc'][t]:.2f} / {o[t]:.2f} ({d:+.2f}{' ✗' if abs(d) > se2 else ''})")
            oavg = sum(o.values()) / len(o)
            entry["avg"] = dict(new=v["avg"], old=oavg, diff=v["avg"] - oavg)
            rec["old_vs_new"].setdefault(m, {})[row] = entry
            md.append(f"| {TITLES[m]} | {NAMES[row]} | " + " | ".join(cells) + f" | {v['avg'] - oavg:+.2f} |")
    n_cells = sum(len(r) * len(TASKS) for r in rec["old_vs_new"].values())
    rec["old_vs_new_summary"] = dict(cells=n_cells, beyond_2se=flagged)
    md += ["", f"Cells beyond 2 SE: {flagged} of {n_cells}. Each cell reads new / old (Δ).",
           "- Old maps: run_small_acc.py / run_main_acc.py. The release maps here are the edgeai-lab uploads, with tiles "
           "bit-identical (release-maps 0d6e56b).",
           "- Old FOCUS states: the edgeai-lab Qwen3 FOCUS states are this part's files (results/paper_eval/focus_hf, "
           "sha256 equal).",
           "- Known difference: the execution (fake on H200 vs native on the RTX PRO 6000)."]
    # inputs, checks, the hold
    iv = x2_interval(qt)
    rec["interval"] = iv
    rec["preflight"] = dict(ok=pre.get("ok"), utc=pre.get("utc"), flipquant=pre.get("flipquant"))
    md += ["", "## Inputs and checks", "",
           f"- **Preflight** (CPU, {pre.get('utc')}): every input equals its registered record, and flipquant is clean "
           f"at 3db0072. Result: {'passed' if pre.get('ok') else 'NOT passed'}.",
           "  - FOCUS states: 5fa877acad86... (1.7B), e4689e64c66f... (8B).",
           "  - GPTQ‡ codes files: bb50de2383f7... and 335a69b142c7...; their code digests equal the main-table records.",
           "  - Release maps: the release summary's map_sha256.",
           "  - IF4 / Zou: the main-table rows' installed-weight sha256.",
           "  - Each input was hashed again before its run.",
           "- **Every run passed part X's check_record:**",
           "  - complete, with every example (21,848 per row), 0-shot, batch 16, the main-table revision;",
           "  - lm-eval 0.4.11, and every dataset load pinned;",
           "  - equal sample digests within the model;",
           "  - native coverage and the kernel families from build_V;",
           "  - the GPTQ‡ codes_sha256 equal to the main-table record, and the IF4 / Zou installed-weight sha256 equal "
           "to the main-table rows';",
           "  - the FOCUS state and map paths.",
           f"- **Interval:** X2's runs, the first START to the last END: {iv['first_start']} to {iv['last_end']} "
           f"({iv['runs']} runs). See Timings and part Y amendment 2 for where it falls among part Y's rows."]
    # timings
    md += ["", "## Timings", "",
           "Per run: model load, each task's wall time and the whole process, in seconds; plus the queue START (UTC). "
           "Simulated rows are marked †.", "",
           "| model | method | load | " + " | ".join(n for _, n in TASKS) + " | total | start |",
           "|---|---|---:|" + "---:|" * (len(TASKS) + 1) + "---|"]
    total = 0.0
    for m in MODELS:
        for row in Z.ROWS:
            v = rec["models"][m].get(row)
            if v is None:
                continue
            tm = v["timing"]
            total += tm["seconds"] or 0
            md.append(f"| {TITLES[m]} | {NAMES[row]}{dagger(row)} | {tm['load_seconds']:.0f} | "
                      + " | ".join(f"{tm['tasks'][t]:.0f}" for t, _ in TASKS) + f" | {tm['seconds']:.0f} | "
                      f"{(tm['queue'] or {}).get('start', '')} |")
    rec["total_process_hours"] = total / 3600
    md += ["", f"Sum of the process times: {total / 3600:.2f} h. Wall time, from the first START to the last END: "
           f"{iv['first_start']} to {iv['last_end']}."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zeroshot6_qwen3.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
