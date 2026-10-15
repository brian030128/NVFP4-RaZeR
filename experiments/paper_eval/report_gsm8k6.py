"""Part Y (PROTOCOL.md): the step 0 report (results/paper_eval/gsm8k6/STEP0.md, step0.json, and the gate
RUN/gsm8k6/step0/step0_ok.json that the panel requires) and the final report (REPORT.md, gsm8k6.json).

    python report_gsm8k6.py step0
    python report_gsm8k6.py panel
"""
import json
import math
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


if __name__ == "__main__":
    {"step0": step0}[sys.argv[1]]()
