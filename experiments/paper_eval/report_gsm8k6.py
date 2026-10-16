"""Part Y (PROTOCOL.md): the step 0 report (results/paper_eval/gsm8k6/STEP0.md, step0.json, and the gate
RUN/gsm8k6/step0/step0_ok.json that the panel requires) and the final report (REPORT.md, gsm8k6.json).

    python report_gsm8k6.py step0
    python report_gsm8k6.py panel
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_gsm8k6 as Y  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

OUT = G.RZ / "results" / "paper_eval" / "gsm8k6"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def git(*a, repo):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()


def step0():
    rec, ok = {}, {}
    # (a) the generation path, f2a56f1 (part N) vs 4bbe0b1
    diff = git("diff", "--stat", "f2a56f1", Y.FQ_COMMIT, "--", "evaluation/downstream.py", "evaluation/lm_eval_datasets.py",
               repo=Y.FQ)
    rec["a"] = dict(files=["evaluation/downstream.py", "evaluation/lm_eval_datasets.py"], diff_stat=diff,
                    changed_since_part_n=git("diff", "--stat", "f2a56f1", Y.FQ_COMMIT, "--", ".", ":(exclude)results",
                                             ":(exclude)docs", ":(exclude)tests", repo=Y.FQ).splitlines()[-1])
    ok["a"] = diff == ""
    # (b) default paths, 4bbe0b1 vs main 3db0072: the code diff and the ten-row numeric parity
    rec["b_code"] = git("diff", "--stat", Y.FQ_COMMIT, Y.MAIN_COMMIT, "--", "flipquant", "models", "evaluation", "kernels",
                        repo=Y.FQ).splitlines()
    par = {}
    for row in Y.ROWS:
        a = load(Y.ROOT / "step0" / "parity" / "fq4bbe0b1" / f"{row}.json")
        b = load(Y.ROOT / "step0" / "parity" / "main3db0072" / f"{row}.json")
        if a is None or b is None:
            par[row] = dict(ok=False, missing=True)
            continue
        c = dict(nll_equal={k: a["results"][k]["nll"] == b["results"][k]["nll"] for k in a["results"]},
                 windows={k: len(v["nll"]) for k, v in a["results"].items()})
        if row == "gptq":
            c["codes_sha256"] = [(a.get("ptq") or {}).get("codes_sha256"), (b.get("ptq") or {}).get("codes_sha256")]
        if row in ("if4", "zou"):
            c["installed_weight_sha256"] = [Z.installed_sha(a), Z.installed_sha(b)]
        c["ok"] = all(c["nll_equal"].values()) and all(v[0] == v[1] and v[0] for k, v in c.items()
                                                       if k in ("codes_sha256", "installed_weight_sha256"))
        par[row] = c
    rec["b_parity"] = dict(model="llama3.2-1b-instruct", limit=2, rows=par)
    ok["b"] = len(par) == len(Y.ROWS) and all(v.get("ok") for v in par.values())
    # (c') Nemotron FO6 / FQ16 vs part N's samples
    ident = load(Y.ROOT / "step0" / "identity_part_n.json") or {}
    rec["c"] = ident
    ok["c"] = bool(ident) and all(v.get("ok") for v in ident.values())
    rec["ok"] = ok
    rec["all_ok"] = all(ok.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (Y.ROOT / "step0" / "step0_ok.json").write_text(json.dumps(dict(ok=rec["all_ok"], checks=ok), indent=1) + "\n")
    yes = lambda v: "yes" if v else "**NO**"  # noqa: E731
    L = ["# Part Y, step 0: the harness (flipquant paper-sm120-runs 4bbe0b1)", "",
         "PROTOCOL.md part Y. Overall: " + ("**all checks passed**" if rec["all_ok"] else "**FAILED**") + ".", "",
         "## (a) The generation path: f2a56f1 (part N's harness) vs 4bbe0b1", "",
         f"`evaluation/downstream.py` and `evaluation/lm_eval_datasets.py`: "
         f"{'byte-identical' if ok['a'] else 'DIFFER: ' + diff}. Everything changed since f2a56f1 (code, outside results / "
         f"docs / tests): {rec['a']['changed_since_part_n']} -- calibration, the opt-in `--fo6-tensor-qmax` and "
         "`--gptq-candidates`, and the registry; checked numerically by (b) and (c').", "",
         "## (b) Default paths: 4bbe0b1 vs main 3db0072", "",
         "Code that differs (evaluation-relevant directories):", "", "```", *rec["b_code"], "```", "",
         "Numeric parity, Llama-3.2-1B-Instruct, `evaluation.ppl --limit 2` (2 WikiText-2 + 2 C4 windows) on both "
         "commits, the same session and env:", "",
         "| row | per-window NLL bitwise (wiki, c4) | other |", "|---|---|---|"]
    for row, c in par.items():
        if c.get("missing"):
            L.append(f"| {row} | missing | |")
            continue
        other = ""
        if "codes_sha256" in c:
            other = f"codes_sha256 {str(c['codes_sha256'][0])[:12]} / {str(c['codes_sha256'][1])[:12]}"
        if "installed_weight_sha256" in c:
            other = f"installed weights {str(c['installed_weight_sha256'][0])[:12]} / {str(c['installed_weight_sha256'][1])[:12]}"
        L.append(f"| {row} | {yes(c['nll_equal'].get('wiki'))}, {yes(c['nll_equal'].get('c4'))} | {other} |")
    L += ["", "## (c') Nemotron-Nano-9B-v2 FourOverSix / FlipQuant 16x64 vs part N's samples", "",
          "All 1,319 problems with part N's exact command, env and batch 64 (these runs are the panel's cells). Per "
          "(problem, filter): the generation (`resps`) and the extracted answer (`filtered_resps`).", "",
          "| row | samples | generations identical | answers identical | first difference |", "|---|---:|---:|---:|---|"]
    for row, v in ident.items():
        fd = v.get("first_difference")
        L.append(f"| {row} | {v['samples']} | {v['generations_identical']} | {v['answers_identical']} | "
                 f"{'none' if not fd else 'doc ' + str(fd['doc_id']) + ' / ' + str(fd['filter'])} |")
    L += ["", "Checks: " + ", ".join(f"({k}) {yes(v)}" for k, v in ok.items()) + "."]
    (OUT / "STEP0.md").write_text("\n".join(L) + "\n")
    print(json.dumps(dict(all_ok=rec["all_ok"], **ok)))


# ---------------------------------------------------------------------------------------------- the panel

TITLES = {"llama3.2-1b-instruct": "Llama-3.2-1B-Instruct", "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
          "mistral-7b": "Mistral-7B-Instruct-v0.3", "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2", "phi4-14b": "Phi-4",
          "qwen3.8-27b": "Qwen3.8-27B"}
ORDER = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b", "qwen3.8-27b"]
NAMES = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "focus": "FOCUS", "gptq": "GPTQ‡", "if4": "IF4†",
         "zou": "MixFP4 (Zou)†", "fq-8x64": "FlipQuant 8x64", "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64"}
LATEX = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "focus": "FOCUS", "gptq": "GPTQ$^{\\ddagger}$",
         "if4": "IF4$^{\\dagger}$", "zou": "MixFP4$^{\\dagger}$", "fq-8x64": "\\method{} $8{\\times}64$",
         "fq-16x64": "\\method{} $16{\\times}64$", "fq-256x64": "\\method{} $256{\\times}64$"}
FILTERS = ("strict_match", "flexible_extract")
# part N / O0 records for the old-vs-new comparison
OLD = {("nemotron-nano-9b-v2", "bf16"): G.RUN / "ptq_round3" / "bf16" / "nemotron-nano-9b-v2" / "bf16.json",
       ("qwen3.8-27b", "bf16"): G.RUN / "ptq_round3" / "bf16" / "qwen3.8-27b" / "bf16.json",
       **{(m, r): Y.PART_N / m / f"rtn_{r}.json" for m in ("nemotron-nano-9b-v2", "qwen3.8-27b")
          for r in ("nvfp4", "fo6", "fq-16x64")}}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lh = n * math.log(0.5)
    return min(1.0, 2 * sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + lh)
                            for i in range(min(b, c) + 1)))


def examples(r):
    return r["tasks"][Y.TASK]["examples"]


def correct(e, f="strict_match"):
    return e[f]["exact_match"] == 1.0


def paired(a, b, f="strict_match"):
    """a vs b on the same problems: (difference in points, a-only correct, b-only correct, exact p)."""
    ea, eb = examples(a), examples(b)
    assert set(ea) == set(eb)
    c = sum(correct(ea[k], f) and not correct(eb[k], f) for k in ea)
    d = sum(correct(eb[k], f) and not correct(ea[k], f) for k in ea)
    return dict(diff=100 * (c - d) / len(ea), row_only=c, ref_only=d, p=mcnemar(d, c), n=len(ea))


def acc(r, f="strict_match"):
    ex = examples(r)
    p = sum(correct(e, f) for e in ex.values()) / len(ex)
    return 100 * p, 200 * math.sqrt(p * (1 - p) / len(ex))


#: gsm8k_llama's scorer (gsm8k-cot.yaml, exact_match): these regexes removed, then lowercased, before comparing
SCORER_IGNORE = (",", "\\$", "(?s).*#### ", "\\.$")


def scored(answer):
    """An extracted answer as lm-eval's exact_match compares it (regexes_to_ignore, ignore_case)."""
    for rx in SCORER_IGNORE:
        answer = re.sub(rx, "", answer)
    return answer.lower()


NO_ANSWER = "[invalid]"             # lm-eval's regex filter output when it extracts nothing


def agree(a, ref):
    """Per problem, against ref (the coordinator's definition, 2026-10-16):
    - an answer agrees when both sides extracted one and they are equal after the scorer's normalization;
    - a problem where neither side extracted an answer is not agreement (it is counted on its own: *_none_both);
    - one side without an answer is a disagreement.
    Also: whether the generation is byte-identical. Raw strings would count "20." and "20" as different, although both
    score alike."""
    ea, eb = examples(a), examples(ref)
    out = {}
    for k in ea:
        e = dict(text=ea[k]["raw"] == eb[k]["raw"])
        for f, name in (("strict_match", "strict"), ("flexible_extract", "flexible")):
            x, y = ea[k][f]["response"], eb[k][f]["response"]
            e[name] = x != NO_ANSWER and y != NO_ANSWER and scored(x) == scored(y)
            e[f"{name}_none_both"] = x == NO_ANSWER and y == NO_ANSWER
        out[k] = e
    return out


def no_answer(r, f="strict_match"):
    """How many generations have no extracted answer under filter f."""
    return sum(e[f]["response"] == NO_ANSWER for e in examples(r).values())


def fidelity(a, ref):
    g = agree(a, ref)
    n = len(g)
    ea, eb = examples(a), examples(ref)
    flips = sum(correct(ea[k]) != correct(eb[k]) for k in ea)
    return dict(n=n, strict=100 * sum(v["strict"] for v in g.values()) / n,
                flexible=100 * sum(v["flexible"] for v in g.values()) / n, flips=flips,
                strict_none_both=sum(v["strict_none_both"] for v in g.values()),
                flexible_none_both=sum(v["flexible_none_both"] for v in g.values()),
                strict_no_answer_row=no_answer(a), strict_no_answer_ref=no_answer(ref),
                row_only=sum(correct(ea[k]) and not correct(eb[k]) for k in ea),
                ref_only=sum(correct(eb[k]) and not correct(ea[k]) for k in ea),
                identical=100 * sum(v["text"] for v in g.values()) / n)


def lengths(r):
    gen = [e["generated"] for e in examples(r).values()]
    toks = [g["tokens"] for g in gen]
    return dict(mean=sum(toks) / len(toks), max=max(toks), budget_hits=sum(not g["eos"] for g in gen))


def label(m, r):
    """The row's name, with § on the rows amendment 3 ran with the chat template's date pinned and ¤ on the rows
    amendment 4 ran with expandable segments."""
    return NAMES[r] + ("§" if (m, r) in Y.DATE_PIN else "") + ("¤" if (m, r) in getattr(Y, "ALLOC", {}) else "")


def bmark(*records):
    """◊ when the paired records ran at different batch sizes (different batch compositions)."""
    return "◊" if len({x["batch_size"] for x in records if x is not None}) > 1 else ""


def x2_interval():
    """Part X amendment 2's runs inside part Y (amendment 2): the first START and the last END of its 20 runs."""
    starts, ends = [], []
    for line in (G.RUN / "queue.log").read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[2].startswith("zs6_panel_qwen3-"):
            (starts if p[1] == "START" else ends if p[1] == "END" else []).append(p[0])
    return (min(starts), max(ends)) if starts and ends else (None, None)


def queue_times():
    t = {}
    for line in (G.RUN / "queue.log").read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] in ("START", "END") and p[2].startswith("gsm8k6_"):
            t.setdefault(p[2], {})[p[1].lower()] = p[0]
    return t


def panel():
    data = {m: {r: load(Y.ROOT / m / f"{r}.json") for r in Y.ROWS} for m in ORDER}
    data = {m: {r: v for r, v in rows.items() if v and v.get("status") == "complete"} for m, rows in data.items()}
    rec = dict(models={}, fidelity={}, fq16_vs={}, old_vs_new={}, timings={})
    qt = queue_times()
    md = ["# GSM8K: six paper models, ten rows (part Y)", "",
          "lm-eval 0.4.11 `gsm8k_llama` as shipped (8-shot CoT, chat template with few-shot as multi-turn, greedy, "
          "max_gen_toks 1024), all 1,319 problems, thinking off (Qwen3.8-27B `enable_thinking=False`; Nemotron "
          "`/no_think`; the others have no thinking mode). Part N's protocol, command and env, on flipquant "
          "paper-sm120-runs 4bbe0b1 (step 0: STEP0.md). Native rows on the RTX PRO 6000 (build_V, `--kernel-set "
          "auto`); IF4 and MixFP4 (Zou) simulated (†). Batch 64; Qwen3.8-27B BF16 at 48 (O0's memory rule). Every "
          "cell is a new run; the Nemotron FourOverSix / FlipQuant 16x64 cells are step 0 (c')'s runs.", "",
          "GSM8K generations depend on the batch composition (part M's pad check), so all rows of a model run the full "
          "set at one batch size and see the same length-sorted batches -- except Qwen3.8-27B BF16 (batch 48): its "
          "paired comparisons with the batch-64 rows (vs BF16, fidelity) compare different batch compositions and are "
          "marked ◊; so is any other pairing of two runs at different batch sizes (amendment 4's fallback).", "",
          "Accuracy in % ± 2 SE (binomial). Paired columns: strict-match difference in points (row-only correct / "
          "reference-only correct, exact McNemar p), * = p < 0.05.", ""]
    for m in ORDER:
        rows = data[m]
        md += [f"## {TITLES[m]}", "", "| method | strict-match | flexible-extract | vs FourOverSix | vs BF16 |",
               "|---|---:|---:|---|---|"]
        mrec = rec["models"][m] = {}
        for r in Y.ROWS:
            v = rows.get(r)
            if v is None:
                md.append(f"| {label(m, r)} | TBD | TBD | | |")
                continue
            s, se = acc(v)
            f, fe = acc(v, "flexible_extract")
            cells, cmp = [], {}
            for ref in ("fo6", "bf16"):
                if r == ref or ref not in rows:
                    cells.append("—" if r == ref else "")
                    continue
                pr = paired(v, rows[ref])
                cmp[ref] = pr
                mark = bmark(v, rows[ref])
                cells.append(f"{pr['diff']:+.2f}{'*' if pr['p'] < 0.05 else ''}{mark} ({pr['row_only']}/{pr['ref_only']}, "
                             f"p={pr['p']:.2g})")
            ln = lengths(v)
            mrec[r] = dict(strict=s, strict_2se=se, flexible=f, flexible_2se=fe, paired=cmp, lengths=ln,
                           batch=v["batch_size"], seconds=v.get("seconds"), load_seconds=v.get("load_seconds"),
                           task_seconds=v["tasks"][Y.TASK]["seconds"], queue=qt.get(f"gsm8k6_{m}_{r}"))
            md.append(f"| {label(m, r)} | {s:.2f} ± {se:.2f} | {f:.2f} ± {fe:.2f} | " + " | ".join(cells) + " |")
        md.append("")
    # summary
    complete = [m for m in ORDER if len(data[m]) == len(Y.ROWS)]
    md += ["## Summary: strict-match per model", "", "Mean = the unweighted mean over the models with all ten rows ("
           + str(len(complete)) + ").", "",
           "| method | " + " | ".join(TITLES[m] for m in ORDER) + " | Mean |", "|---|" + "---:|" * (len(ORDER) + 1)]
    rec["summary"] = {}
    for r in Y.ROWS:
        vals = [rec["models"][m].get(r, {}).get("strict") for m in ORDER]
        mean = sum(rec["models"][m][r]["strict"] for m in complete) / len(complete) if complete else None
        rec["summary"][r] = dict(mean=mean, per_model=dict(zip(ORDER, vals)))
        md.append(f"| {NAMES[r]} | " + " | ".join("TBD" if v is None else f"{v:.2f}" for v in vals)
                  + (f" | {mean:.2f} |" if mean is not None else " | |"))
    # fidelity to BF16
    md += ["", "## Fidelity to BF16", "",
           "Per problem against the same model's BF16 run: the extracted answer equal (strict, flexible) after the "
           "scorer's own normalization (lm-eval's exact_match: `,`, `$`, `#### ` and a trailing `.` removed, case "
           "ignored), correctness flips (row-only + BF16-only correct, strict), and byte-identical generations. Pooled = over every problem of "
           "the models shown. ◊ = the two runs had different batch sizes (Qwen3.8-27B BF16 ran at batch 48), so their "
           "batch compositions differ.", "",
           "| method | " + " | ".join(TITLES[m] for m in ORDER) + " | pooled |", "|---|" + "---|" * (len(ORDER) + 1)]
    for r in Y.ROWS:
        if r == "bf16":
            continue
        cells, pool = [], dict(n=0, strict=0.0, flexible=0.0, flips=0, identical=0.0)
        for m in ORDER:
            v, ref = data[m].get(r), data[m].get("bf16")
            if v is None or ref is None:
                cells.append("TBD")
                continue
            fd = fidelity(v, ref)
            rec["fidelity"].setdefault(r, {})[m] = fd
            for k in ("strict", "flexible", "identical"):
                pool[k] += fd[k] * fd["n"] / 100
            pool["flips"] += fd["flips"]
            pool["n"] += fd["n"]
            cells.append(f"{fd['strict']:.1f} / {fd['flexible']:.1f}; {fd['flips']} ({fd['row_only']}+{fd['ref_only']}); "
                         f"{fd['identical']:.1f}{bmark(v, ref)}")
        if pool["n"]:
            rec["fidelity"].setdefault(r, {})["pooled"] = dict(n=pool["n"], strict=100 * pool["strict"] / pool["n"],
                                                               flexible=100 * pool["flexible"] / pool["n"],
                                                               flips=pool["flips"], identical=100 * pool["identical"] / pool["n"])
            pc = rec["fidelity"][r]["pooled"]
            cells.append(f"{pc['strict']:.1f} / {pc['flexible']:.1f}; {pc['flips']}; {pc['identical']:.1f}")
        else:
            cells.append("")
        md.append(f"| {NAMES[r]} | " + " | ".join(cells) + " |")
    md += ["", "Each cell: answer agreement strict / flexible (%); correctness flips (row-only + BF16-only); byte-identical "
           "generations (%).",
           "",
           "**Agreement.** Both sides extracted an answer, and the two are equal after the scorer's normalization. A "
           "problem where neither side extracted an answer (`[invalid]`) is not agreement; it is counted below. One side "
           "without an answer is a disagreement.", "",
           "### Generations without a strict-match answer", "",
           "Per row, the problems whose generation has no strict-match answer (`[invalid]`: no \"final answer is <number>\"). "
           "In brackets: the problems where the row and BF16 both have none.", "",
           "| method | " + " | ".join(TITLES[m] for m in ORDER) + " | total |", "|---|" + "---:|" * (len(ORDER) + 1)]
    rec["no_answer"] = {}
    for r in Y.ROWS:
        cells, tot = [], 0
        for m in ORDER:
            v, ref = data[m].get(r), data[m].get("bf16")
            if v is None:
                cells.append("TBD")
                continue
            n_none = no_answer(v)
            both = None if r == "bf16" or ref is None else sum(x["strict_none_both"] for x in agree(v, ref).values())
            rec["no_answer"].setdefault(r, {})[m] = dict(strict=n_none, strict_both_with_bf16=both)
            tot += n_none
            cells.append(f"{n_none}" + ("" if both is None else f" ({both})"))
        md.append(f"| {NAMES[r]} | " + " | ".join(cells) + f" | {tot} |")
    # FlipQuant 16x64 vs the others on agreement with BF16
    md += ["", "### FlipQuant 16x64 vs FOCUS, IF4, Zou and FourOverSix: per-problem agreement with BF16 (exact McNemar)", "",
           "Strict-extracted answer equal to BF16's: FlipQuant-only agreeing / other-only agreeing, p. Pooled over the "
           "models shown.", "",
           "| vs | " + " | ".join(TITLES[m] for m in ORDER) + " | pooled |", "|---|" + "---|" * (len(ORDER) + 1)]
    for other in ("focus", "if4", "zou", "fo6"):
        cells, pb, pc_ = [], 0, 0
        for m in ORDER:
            a, o, ref = (data[m].get(x) for x in ("fq-16x64", other, "bf16"))
            if a is None or o is None or ref is None:
                cells.append("TBD")
                continue
            ga, go = agree(a, ref), agree(o, ref)
            b = sum(ga[k]["strict"] and not go[k]["strict"] for k in ga)
            c = sum(go[k]["strict"] and not ga[k]["strict"] for k in ga)
            pb, pc_ = pb + b, pc_ + c
            rec["fq16_vs"].setdefault(other, {})[m] = dict(fq_only=b, other_only=c, p=mcnemar(b, c))
            cells.append(f"{b} / {c}, p={mcnemar(b, c):.2g}{bmark(a, o, ref)}")
        rec["fq16_vs"].setdefault(other, {})["pooled"] = dict(fq_only=pb, other_only=pc_, p=mcnemar(pb, pc_))
        cells.append(f"{pb} / {pc_}, p={mcnemar(pb, pc_):.2g}")
        md.append(f"| {NAMES[other]} | " + " | ".join(cells) + " |")
    # old vs new
    md += ["", "## Old vs new: part N (RTN rows) and O0 (BF16)", "",
           "Strict-match accuracy new / old, the paired difference (new-only / old-only correct, exact McNemar p) and "
           "the share of problems whose generation is byte-identical. Old: part N (flipquant paper-sm120-runs f2a56f1) and "
           "O0 (BF16, part O), the same command, env and batch (Qwen3.8-27B BF16 48 in both).", "",
           "| model | method | new | old | paired | byte-identical |", "|---|---|---:|---:|---|---:|"]
    for (m, r), p in OLD.items():
        new, old = data[m].get(r), load(p)
        if new is None or old is None:
            md.append(f"| {TITLES[m]} | {NAMES[r]} | TBD | | | |")
            continue
        pr = paired(new, old)
        same = 100 * sum(v["text"] for v in agree(new, old).values()) / len(examples(new))
        rec["old_vs_new"][f"{m}/{r}"] = dict(new=acc(new)[0], old=acc(old)[0], paired=pr, identical=same, old_record=str(p))
        md.append(f"| {TITLES[m]} | {NAMES[r]} | {acc(new)[0]:.2f} | {acc(old)[0]:.2f} | {pr['diff']:+.2f} "
                  f"({pr['row_only']}/{pr['ref_only']}, p={pr['p']:.2g}) | {same:.1f} % |")
    # deviations and notes
    x2a, x2b = x2_interval()
    b48 = {r: rec["models"]["qwen3.8-27b"].get(r, {}).get("batch") for r in Y.ROWS}
    cost = {}
    for m in ORDER:
        fo6 = rec["models"][m].get("fo6")
        for r in ("if4", "zou"):
            v = rec["models"][m].get(r)
            if v and fo6:
                cost.setdefault(m, {})[r] = v["seconds"] / fo6["seconds"]
    rec["notes"] = dict(x2_interval=[x2a, x2b], qwen27b_batches=b48, fake_over_fo6=cost,
                        date_pinned_rows=[list(k) for k in sorted(Y.DATE_PIN)], date_pin=sorted(set(Y.DATE_PIN.values())))
    old_tmp = Y.ROOT / "llama3.2-1b-instruct" / "if4_date-oct16.json"
    md += ["", "## Deviations and notes", "",
           f"1. **Batch.** Qwen3.8-27B BF16 ran at batch {b48.get('bf16')}: its record's batch_size and its logged "
           "command both say so. Every other run used 64. Qwen3.8-27B BF16's paired comparisons are marked ◊.",
           f"2. **X2 inside part Y (amendments 1 and 2).** Part X amendment 2's 20 zero-shot runs ran from {x2a} to {x2b}, "
           "between Llama-3.1-8B-Instruct's NVFP4 and FO6 rows.",
           "   - X2's hold took effect at that boundary, not at the registered one (after part Y's native rows).",
           "   - Part Y's driver waited for an idle GPU, by its own rule. Nothing was stopped, and every job ran alone on "
           "the GPU.",
           "   - No part Y result depends on the position.",
           "3. **Date pin (amendment 3), the rows marked §.** Llama-3.2-1B-Instruct's chat template writes today's date into "
           "its system header.",
           "   - The model's native rows ran on 2026-10-15. Its IF4 and MixFP4 rows ran with the date pinned to "
           "\"15 Oct 2026\": gsm8k6_date.py pins transformers' strftime_now and nothing else.",
           "   - Both passed the registered check: sample_digest equal to BF16's, so all 2,638 prompts are "
           "byte-identical.",
           "   - The first, unpinned IF4 run (2026-10-16 01:59–02:50 UTC, \"16 Oct 2026\") stopped the panel on that "
           f"check. Its record is kept as `{old_tmp.name.replace('.json', '.*')}` and is not a panel cell.",
           "   - No other model's template writes a date. Llama-3.1-8B-Instruct's header is a fixed \"26 Jul 2024\".",
           "4. **Old vs new and dates.** The old records compared (part N / O0: Nemotron, Qwen3.8-27B) have no date in their "
           "prompts. No older Llama-3.2-1B-Instruct GSM8K record exists, because parts N / O and tab:ptq cover Nemotron "
           "and Qwen3.8-27B only, so no \"Today Date\" line is compared.",
           "5. **Allocator and batch (amendment 4 and its addendum), the rows marked ¤.** Qwen3.8-27B's IF4 and MixFP4 "
           "rows ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Under the default allocator, IF4 had run out "
           "of memory at batch 64 from fragmentation (if4_oom-b64-default-alloc.log).",
           "   - Before the full rows, a --limit 64 pre-check showed the allocator changes no output: "
           + (f"{a4['generations_identical']} / {a4['samples']} samples byte-identical."
              if (a4 := load(Y.ROOT / "amend4" / "check.json")) else "pending.")
           + " Every other row ran under the default allocator, as registered.",
           "   - The batch each ran at: " + "; ".join(
               f"{NAMES[r]} {rec['models']['qwen3.8-27b'][r]['batch']}" for r in ("if4", "zou")
               if r in rec["models"].get("qwen3.8-27b", {})) + ".",
           "   - A row below 64 is marked ◊ against batch-64 rows. A row at 48 shares the batch composition of Qwen3.8-27B "
           "BF16 (48), so that pairing carries no ◊.",
           "6. **Simulated rows' cost** (process seconds, against the model's native FourOverSix row): "
           + "; ".join(f"{TITLES[m]} IF4 {c.get('if4', float('nan')):.1f}x, MixFP4 {c.get('zou', float('nan')):.1f}x"
                       for m, c in cost.items()) + "."]
    # timings and lengths
    md += ["", "## Timings and generated lengths", "",
           "Per run: model load and the task's wall time (s), mean / max generated tokens (to the first end-of-sequence "
           "token), and the problems that used the whole 1,024-token budget.", "",
           "| model | method | batch | load | task | total | mean tokens | max | budget hits | start (UTC) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    total = 0.0
    for m in ORDER:
        for r in Y.ROWS:
            v = rec["models"][m].get(r)
            if v is None:
                continue
            total += v["seconds"] or 0
            md.append(f"| {TITLES[m]} | {label(m, r)} | {v['batch']} | {v['load_seconds']:.0f} | {v['task_seconds']:.0f} | "
                      f"{v['seconds']:.0f} | {v['lengths']['mean']:.0f} | {v['lengths']['max']} | "
                      f"{v['lengths']['budget_hits']} | {(v['queue'] or {}).get('start', '')} |")
    rec["total_process_hours"] = total / 3600
    md += ["", f"Sum of the process times: {total / 3600:.1f} h."]
    # LaTeX
    md += ["", "## LaTeX (strict-match, %)", "", "```latex",
           "Method & " + " & ".join(TITLES[m] for m in ORDER) + " & Mean \\\\", "\\midrule"]
    for r in Y.ROWS:
        s = rec["summary"][r]
        vals = [s["per_model"][m] for m in ORDER]
        md.append(f"{LATEX[r]} & " + " & ".join("--" if v is None else f"{v:.2f}" for v in vals)
                  + (f" & {s['mean']:.2f} \\\\" if s["mean"] is not None else " & -- \\\\"))
    md += ["```"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gsm8k6.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    {"step0": step0, "panel": panel}[sys.argv[1]]()
