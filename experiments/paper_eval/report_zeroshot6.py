"""Part X (PROTOCOL.md): the step 0 report (STEP0.md, step0.json, and RUN/zeroshot6/step0/step0_ok.json, which the
panel requires) and the final report (REPORT.md, zeroshot6.json) in results/paper_eval/zeroshot6/.

    python report_zeroshot6.py step0
    python report_zeroshot6.py panel
"""
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

OUT = G.RZ / "results" / "paper_eval" / "zeroshot6"
ZS = Z.ZS
PARITY_POLICIES = ("bf16", "fo6", "fq-16x64")


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def git_show(repo, rev, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True,
                          check=True).stdout


# ---------------------------------------------------------------------------------------------- step 0 (a)

def functions(src):
    """{name: ast.dump without docstrings} of a module's top-level functions."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                                     and isinstance(node.body[0].value.value, str)) else node.body
            node.body = body
            out[node.name] = ast.dump(node, annotate_fields=False)
    return out


def literal(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "dict" and not v.args:
                return {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            return ast.literal_eval(v)
    return None


def settings_diff():
    rz = (Z.RZ_MAIN / "run_lmeval_deploy.py").read_text()
    fq = (Z.FQ / "evaluation" / "downstream.py").read_text()
    a, b = functions(rz), functions(fq)
    same_fn = {n: a.get(n) == b.get(n) for n in ("primary", "mc_record", "sample_hashes", "summarize", "summarize_group")}
    rz_pins = (Z.RZ_MAIN / "lm_eval_datasets.py").read_bytes()
    fq_pins = (Z.FQ / "evaluation" / "lm_eval_datasets.py").read_bytes()
    prim_rz = dict(literal(rz, "PRIMARY") or {}, mmlu="acc")
    prim_fq = {k: v for k, v in (literal(fq, "PRIMARY") or {}).items()}
    return dict(functions_identical=same_fn,
                lm_eval_datasets_sha256=dict(razer=hashlib.sha256(rz_pins).hexdigest(),
                                             flipquant=hashlib.sha256(fq_pins).hexdigest()),
                flipquant_extra_pins=literal(fq, "EXTRA_PINS"), primary_razer=prim_rz, primary_flipquant=prim_fq,
                primary_differences={k: (prim_rz.get(k), prim_fq.get(k)) for k in set(prim_rz) | set(prim_fq)
                                     if prim_rz.get(k) != prim_fq.get(k)},
                razer_commit=subprocess.run(["git", "-C", Z.RZ_MAIN, "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
                razer_dirty=bool(subprocess.run(["git", "-C", Z.RZ_MAIN, "status", "--porcelain", "--untracked-files=no"],
                                                capture_output=True, text=True).stdout.strip()),
                flipquant_commit=subprocess.run(["git", "-C", Z.FQ, "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip())


DIFF_ROWS = [
    ("lm-eval", "0.4.11, the env's install (a compat import only below 0.4.10)", "0.4.11, the env's install", "same"),
    ("HFLM", "`HFLM(pretrained=model, tokenizer=tok, batch_size=B, backend='causal')`; `add_bos_token=False` only with "
     "--no-bos, which 04_downstream.py does not pass", "`HFLM(pretrained=model, tokenizer=tok, batch_size=batch, "
     "backend=\"causal\")`", "same (default BOS: 0.4.11 keeps the tokenizer's special tokens; both record it)"),
    ("batch", "--batch-size, 16; 04_downstream.py: 16, Qwen3.8-27B 8 (paper_common.LMEVAL_BATCH)",
     "16, Qwen3.8-27B 8 (BATCH)", "same"),
    ("simple_evaluate", "limit, log_samples=True, bootstrap_iters=100000, random_seed=0, numpy/torch/fewshot seeds 1234, "
     "num_fewshot", "the same arguments, plus task_manager=TaskManager(include_path=evaluation/lm_eval_tasks)",
     "differs: the include path adds only mmlu_pro_loglik (no standard task is redefined); no effect on these tasks"),
    ("dataset pins", "lm_eval_datasets.py (PINS)", "the same module, vendored unchanged, plus EXTRA_PINS (MMLU-Pro, "
     "EleutherAI/lambada_openai @ 900124bf)", "differs: flipquant pins LAMBADA, NVFP4-RaZeR does not know it"),
    ("few-shot", "--num-fewshot (04_downstream.py: mmlu=5 only); the rest lm-eval's defaults", "FEWSHOT = {mmlu: 5}; "
     "the rest lm-eval's defaults", "same (the six tasks are 0-shot)"),
    ("primary metric", "PRIMARY: acc_norm (ARC, HellaSwag, OpenBookQA, PIQA), acc (BoolQ, WinoGrande, MMLU), "
     "exact_match (GSM8K); no LAMBADA", "the same, plus lambada_openai = acc", "differs: LAMBADA has no primary metric in "
     "run_lmeval_deploy.py (primary() raises), so LAMBADA cannot run there"),
    ("records", "examples, loglikelihoods, sample_digest, prompt_hash, metrics, timings, native_gemm, coverage",
     "the same fields and functions (ast-identical, below)", "same"),
    ("activation scope", "convention (c): per-token scales (fake: four_over_six_rows; native: NativeLinear)",
     "--act-scope row (the default here; the sm120 path accepts only row)", "same"),
    ("TF32", "off (matmul and cuDNN)", "off (matmul and cuDNN)", "same"),
    ("model loading", "run_ppl_deploy.load_for_evaluation: AutoModelForCausalLM from the calibration record's snapshot "
     "and revision, BF16, attn eager then set_attn_implementation('sdpa'), device cuda, eval; every quantized matrix's "
     "sha256 checked against the record (Qwen3.8-27B: Qwen3_5ForConditionalGeneration, sdpa)",
     "models.loader.load: the registry's hf_id at --revision, BF16, attn sdpa, device cuda, eval; no weight hash check",
     "differs in route (the attention is SDPA on both; the parity shows equal outputs on the shared model)"),
    ("tokenizer", "AutoTokenizer.from_pretrained(snapshot path, revision)", "AutoTokenizer.from_pretrained(hf_id, "
     "revision)", "same files"),
    ("revisions", "the calibration record's (data roots); base Mistral-7B-v0.3, base Llama-3.1-8B, Phi-4, Qwen3.8-27B "
     "and Qwen3-4B only (sm120/eval/common.MODELS)", "registry pins or --revision (part X passes the main-table "
     "revisions for all six models)", "differs: NVFP4-RaZeR has none of the Instruct models of the panel"),
    ("policies", "exported artifacts (native:<artifact>[:<kernel>]) or fake (c); SM120_BUILD_DIR",
     "models.cli flags; weights packed in memory by NVFP4-RaZeR's packer (models/sm120.py); --kernel-build-dir",
     "differs in route; the same build_V libraries"),
    ("processes", "several policies per process (BF16 / fake first, then native), one HFLM per policy",
     "one policy per process", "differs; the parity runs both ways"),
]


def step0():
    rec = dict(settings=settings_diff())
    ok = {}
    # (b) parity
    rz = load(ZS / "step0" / Z.PARITY["dir"] / "razer" / "report.json") or {}
    par, libs = {}, G.manifests()
    for pol in PARITY_POLICIES:
        fq = load(ZS / "step0" / Z.PARITY["dir"] / "flipquant" / f"{pol}.json") or {}
        ev = (rz.get("evaluations") or {}).get(pol) or {}
        for t in Z.PARITY_TASKS.split(","):
            a, b = (ev.get("tasks") or {}).get(t), (fq.get("tasks") or {}).get(t)
            if not a or not b:
                par[f"{pol}/{t}"] = dict(ok=False, missing=True)
                continue
            lla = [x for v in a["loglikelihoods"].values() for x in v["ll"]]
            c = dict(examples=len(b["examples"]), choices=len(lla), digest=a["sample_digest"] == b["sample_digest"],
                     correctness=a["examples"] == b["examples"], loglikelihoods=a["loglikelihoods"] == b["loglikelihoods"],
                     metrics=a["metrics"] == b["metrics"], acc=b["metrics"].get(Z.primary_metric(t) + ",none"))
            c["ok"] = all(c[k] for k in ("digest", "correctness", "loglikelihoods", "metrics"))
            par[f"{pol}/{t}"] = c
        if pol != "bf16":
            lr = Z.libraries_ok((ev.get("install") or {}).get("kernel_set"), libs)
            lf = Z.libraries_ok([((fq.get("policy") or {}).get("native") or {}).get("kernel_set"),
                                 (fq.get("coverage") or {}).get("kernel_set")], libs)
            par[f"{pol}/libraries"] = dict(razer=lr[1], flipquant=lf[1], ok=lr[0] and lf[0],
                                           razer_kernel=(ev.get("install") or {}).get("kernel"),
                                           flipquant_kernel=((fq.get("policy") or {}).get("native") or {}).get("kernel"))
    rec["parity"] = dict(razer_status=rz.get("status"), tasks=Z.PARITY_TASKS, limit=24, cells=par,
                         tokenization=dict(razer=rz.get("tokenization"),
                                           flipquant=(load(ZS / "step0" / Z.PARITY["dir"] / "flipquant" / "bf16.json") or {}).get(
                                               "tokenization")))
    ok["parity"] = rz.get("status") == "complete" and all(v.get("ok") for v in par.values())
    # (c) IF4 / Zou layers
    ident = load(ZS / "step0" / "identity_if4_zou.json") or {}
    rec["identity_if4_zou"] = ident
    ok["identity_if4_zou"] = bool(ident.get("ok"))
    # (c) ppl checks
    ppl = {}
    for model in Z.MODELS:
        for row in ("if4", "zou", "focus", "gptq"):
            p = ZS / "step0" / "ppl" / model / f"{row}.json"
            if row in ("if4", "zou") and model != "mistral-7b":
                continue
            r = load(p)
            ref_path = {"if4": Z.mainrow_record(model, "if4"), "zou": Z.mainrow_record(model, "zou"),
                        "focus": Z.focus_ppl_record(model), "gptq": Z.gptq_record(model)}[row]
            ref = load(ref_path)
            if r is None or ref is None:
                ppl[f"{model}/{row}"] = dict(ok=False, missing=str(p if r is None else ref_path))
                continue
            c = dict(record=str(ref_path), windows={k: len(v["nll"]) for k, v in r["results"].items()},
                     nll_equal={k: v["nll"] == ref["results"][k]["nll"][:len(v["nll"])] for k, v in r["results"].items()},
                     nll={k: v["nll"] for k, v in r["results"].items()},
                     seconds=(r.get("resources") or {}).get("seconds"))
            if row in ("if4", "zou"):
                c["installed_weight_sha256"] = [Z.installed_sha(r), Z.installed_sha(ref)]
                c["ok"] = all(c["nll_equal"].values()) and c["installed_weight_sha256"][0] == c["installed_weight_sha256"][1]
            elif row == "gptq":
                got, want = (r.get("ptq") or {}), (ref.get("ptq") or {})
                gargs = {k: (r["args"].get(k), ref["args"].get(k)) for k in ref["args"]
                         if k.startswith("gptq_") and k != "gptq_cache" and r["args"].get(k) != ref["args"].get(k)}
                # an opt-in option of paper-sm120-runs that main lacks, unused (None) in the record, is not a difference
                absent = {k: v for k, v in gargs.items() if v[0] is None and v[1] == "None" and k not in r["args"]}
                gargs = {k: v for k, v in gargs.items() if k not in absent}
                c.update(codes_sha256=[got.get("codes_sha256"), want.get("codes_sha256")], loaded=got.get("loaded"),
                         data_sha256=[got.get("data_sha256"), want.get("data_sha256")], gptq_args_differing=gargs,
                         gptq_args_absent_in_main_unused_in_record=sorted(absent))
                c["ok"] = (all(c["nll_equal"].values()) and got.get("codes_sha256") == want.get("codes_sha256")
                           and got.get("data_sha256") == want.get("data_sha256") and not gargs)
            else:
                c["state"] = (r.get("focus") or {}).get("state")
                c["ok"] = all(c["nll_equal"].values())
            ppl[f"{model}/{row}"] = c
    rec["ppl_checks"] = ppl
    ok["ppl_checks"] = len(ppl) == 14 and all(v.get("ok") for v in ppl.values())
    # (c) codes files, FOCUS states, the Qwen3.8-27B download
    codes = load(ZS / "step0" / "codes.json") or {}
    rec["codes_files"] = codes
    ok["codes_files"] = bool(codes.get("ok"))
    codes_sha = {m: v["file_sha256"] for m, v in (codes.get("files") or {}).items()}
    for m in Z.LLAMA:
        f = Z.codes_file(m)
        codes_sha[m] = Z.sha256_file(f) if f.exists() else None
    rec["codes_file_sha256"] = codes_sha
    ok["llama_codes_files"] = all(codes_sha.get(m) for m in Z.LLAMA)
    states = {}
    for m in Z.MODELS:
        p = Z.focus_state(m)
        got = Z.sha256_file(p)
        states[m] = dict(path=str(p), bytes=p.stat().st_size, sha256=got, record=Z.FOCUS_SHA[m],
                         ok=Z.FOCUS_SHA[m] is None or got == Z.FOCUS_SHA[m])
    rec["focus_states"] = states
    ok["focus_states"] = all(v["ok"] for v in states.values())
    dl = load(ZS / "step0" / "focus_qwen27b_download.json") or {}
    rec["qwen27b_download"] = dl
    ok["qwen27b_download"] = bool(dl.get("equal_to_part_q") and dl.get("sha256_equals_hub_lfs"))
    rec["ok"] = ok
    fn = rec["settings"]["functions_identical"]
    # summarize_group (group tasks such as MMLU only; none of the six) differs by one inlined local variable
    rec["all_ok"] = all(ok.values()) and all(fn[k] for k in ("primary", "mc_record", "sample_hashes", "summarize")) and \
        len(set(rec["settings"]["lm_eval_datasets_sha256"].values())) == 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "STEP0.md").write_text(step0_md(rec) + "\n")
    gate = dict(ok=rec["all_ok"], codes_file_sha256=codes_sha, focus_state_sha256={m: v["sha256"] for m, v in states.items()},
                checks=ok)
    (ZS / "step0" / "step0_ok.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps(dict(all_ok=rec["all_ok"], **ok), indent=1))


def yes(v):
    return "yes" if v else "**NO**"


def step0_md(rec):
    s = rec["settings"]
    L = ["# Part X, step 0: code identity (flipquant main vs NVFP4-RaZeR)", "",
         f"flipquant origin/main `{s['flipquant_commit'][:7]}` (fresh detached worktree, unmodified) against NVFP4-RaZeR "
         f"`{s['razer_commit'][:7]}` (kernel-opt, the clean main checkout{', DIRTY' if s['razer_dirty'] else ''}); "
         "PROTOCOL.md part X. Overall: " + ("**all checks passed**" if rec["all_ok"] else "**FAILED**") + ".", "",
         "## (a) Settings: evaluation/downstream.py vs run_lmeval_deploy.py + experiments/paper/04_downstream.py", "",
         "| item | NVFP4-RaZeR | flipquant main | verdict |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in DIFF_ROWS]
    fn = s["functions_identical"]
    L += ["", "Checked from the sources:",
          f"- The record functions `primary`, `mc_record`, `sample_hashes`, `summarize` and `summarize_group` are "
          f"AST-identical apart from docstrings: " + ", ".join(f"{k} {yes(v)}" for k, v in fn.items()) + "."
          + (" `summarize_group` differs only in form: NVFP4-RaZeR binds `groups = res.get('groups', {})` and then "
             "reads `groups.get(task)`, flipquant inlines it as `res.get(\"groups\", {}).get(task)`; it serves group "
             "tasks (MMLU) only, none of the six." if not fn.get("summarize_group") else ""),
          f"- `lm_eval_datasets.py` sha256: NVFP4-RaZeR {s['lm_eval_datasets_sha256']['razer'][:16]}..., flipquant "
          f"{s['lm_eval_datasets_sha256']['flipquant'][:16]}... (equal: "
          f"{yes(len(set(s['lm_eval_datasets_sha256'].values())) == 1)}).",
          f"- flipquant's extra pins: " + ", ".join(f"{k} @ {v[:10]}" for k, v in (s["flipquant_extra_pins"] or {}).items())
          + ". The panel checks that every LAMBADA load used 900124bf.",
          "- Primary-metric differences (NVFP4-RaZeR, flipquant): " + (", ".join(
              f"{k}: {v}" for k, v in sorted(s["primary_differences"].items())) or "none") + ".",
          "- **LAMBADA**: run_lmeval_deploy.py has no primary metric for lambada_openai, so it cannot evaluate LAMBADA; "
          "only flipquant's dataset pin is checked, as the protocol says.", ""]
    p = rec["parity"]
    L += ["## (b) Same-session harness parity: Phi-4, --limit 24, batch 16", "",
          "NVFP4-RaZeR `run_lmeval_deploy.py --model phi4` (one process: bf16, then fo6 = native phi4_fo6 / auto_stock, "
          "then fq-16x64 = native phi4_tc_16x64 / auto; SM120_BUILD_DIR=build_V) against flipquant `evaluation.downstream "
          "--model phi4-14b --revision 2db69c1c` (one process per policy; build_V, --kernel-set auto; the paper 16x64 map "
          "the artifact was exported from). NVFP4-RaZeR's harness knows none of the panel's Instruct models (deviation "
          "3), and base Mistral-7B-v0.3's weights are no longer cached, so the parity runs on Phi-4 (amendment 1).", "",
          "| policy / task | examples | choices | sample digest | correctness | per-choice LL (bitwise) | metrics | primary |",
          "|---|---:|---:|---|---|---|---|---:|"]
    for k, c in p["cells"].items():
        if k.endswith("/libraries"):
            continue
        if c.get("missing"):
            L.append(f"| {k} | missing | | | | | | |")
            continue
        L.append(f"| {k} | {c['examples']} | {c['choices']} | {yes(c['digest'])} | {yes(c['correctness'])} | "
                 f"{yes(c['loglikelihoods'])} | {yes(c['metrics'])} | {c['acc']:.4f} |")
    for k, c in p["cells"].items():
        if k.endswith("/libraries"):
            L.append(f"- {k.split('/')[0]}: kernels NVFP4-RaZeR {c['razer_kernel']}, flipquant {c['flipquant_kernel']}; "
                     f"every library from build_V: {yes(c['ok'])} ({len(c['razer'])} / {len(c['flipquant'])} libraries).")
    tk = p.get("tokenization") or {}
    L += [f"- Tokenization (both record it): NVFP4-RaZeR {json.dumps(tk.get('razer'))}; flipquant "
          f"{json.dumps(tk.get('flipquant'))}.", ""]
    i = rec["identity_if4_zou"]
    L += ["## (c) IF4, Zou, FOCUS deploy and GPTQ‡ against the paths behind tab:main-ppl", "",
          "### IF4 / Zou, layer by layer (Mistral-7B-Instruct-v0.3, GPU)", "",
          f"Reference: NVFP4-RaZeR 44f8cea `quantize/adaptive_formats.py` (sha256 "
          f"{(i.get('reference') or {}).get('sha256', '?')[:16]}...), the code of main-ppl's IF4 / Zou records; flipquant "
          f"main `flipquant/if4.py`, `flipquant/zou_mixfp4.py`.", "",
          "| rule | weights bitwise | installed-weight sha256: reference / flipquant / main-ppl record | activations bitwise "
          "(one WikiText-2 window) |", "|---|---|---|---|"]
    for rule, v in (i.get("rules") or {}).items():
        w, a = v["weights"], v["activations"]
        L.append(f"| {rule} | {w['modules'] - len(w['unequal'])}/{w['modules']} | {w['sha256_reference'][:12]} / "
                 f"{w['sha256_flipquant'][:12]} / {str(w['sha256_main_ppl_record'])[:12]} | "
                 f"{a['calls'] - len(a['unequal'])}/{a['calls']} calls ({a['rows']} rows) |")
    L += ["", "### evaluation.ppl spot checks against the main-table records (flipquant main, per-window NLL bitwise)", "",
          "| model / row | windows (wiki, c4) | NLL equal | other | record |", "|---|---|---|---|---|"]
    for k, c in rec["ppl_checks"].items():
        if c.get("missing"):
            L.append(f"| {k} | missing: {c['missing']} | **NO** | | |")
            continue
        other = ""
        if "codes_sha256" in c:
            other = (f"codes_sha256 {c['codes_sha256'][0][:12]} vs {c['codes_sha256'][1][:12]} "
                     f"({'loaded' if c['loaded'] else 'computed by main'}); data_sha256 equal "
                     f"{yes(c['data_sha256'][0] == c['data_sha256'][1])}"
                     + (f"; args differ {c['gptq_args_differing']}" if c["gptq_args_differing"] else "")
                     + (f"; {', '.join(c['gptq_args_absent_in_main_unused_in_record'])} absent in main (opt-in, unused "
                        "in the record)" if c.get("gptq_args_absent_in_main_unused_in_record") else ""))
        elif "installed_weight_sha256" in c:
            other = f"installed weights {c['installed_weight_sha256'][0][:12]} vs {c['installed_weight_sha256'][1][:12]}"
        L.append(f"| {k} | {c['windows'].get('wiki')}, {c['windows'].get('c4')} | "
                 f"{yes(all(c['nll_equal'].values()))} | {other} | `{c['record'].replace(str(G.RUN), 'RUN')}` |")
    L += ["", "### GPTQ‡ codes files (CPU) and FOCUS states", "", "| model | codes file | code digest = record | key = record |",
          "|---|---|---|---|"]
    for m, v in ((rec["codes_files"] or {}).get("files") or {}).items():
        L.append(f"| {m} | `{v['file'].replace(str(G.RUN), 'RUN')}` ({v['bytes']:,} B) | {yes(v['codes_sha256'] == v['record_codes_sha256'])} "
                 f"({v['codes_sha256'][:12]}) | {yes(not v['key_differences'])} |")
    for m in Z.LLAMA:
        L.append(f"| {m} | `{str(Z.codes_file(m)).replace(str(G.RUN), 'RUN')}` (regenerated in step 0 by main's GPTQ) | "
                 f"see the ppl check | |")
    L += ["", "| model | FOCUS state | sha256 | recorded sha256 | equal |", "|---|---|---|---|---|"]
    for m, v in rec["focus_states"].items():
        L.append(f"| {m} | `{v['path'].replace(str(G.RUN), 'RUN')}` | {v['sha256'][:16]}... | "
                 f"{v['record'][:16] if v['record'] else 'none (recorded here)'} | {yes(v['ok']) if v['record'] else 'n/a'} |")
    d = rec["qwen27b_download"]
    L += ["", f"Qwen3.8-27B FOCUS state downloaded again: {d.get('bytes', 0):,} bytes, sha256 {str(d.get('sha256'))[:16]}..., "
          f"equal to part Q's record {yes(d.get('equal_to_part_q'))}, equal to the Hub's LFS sha256 "
          f"{yes(d.get('sha256_equals_hub_lfs'))} ({(d.get('seconds') or 0) / 60:.0f} min).", "",
          "Checks: " + ", ".join(f"{k} {yes(v)}" for k, v in rec["ok"].items()) + "."]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- the panel

TITLES = {"llama3.2-1b-instruct": "Llama-3.2-1B-Instruct", "mistral-7b": "Mistral-7B-Instruct-v0.3",
          "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct", "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2",
          "phi4-14b": "Phi-4", "qwen3.8-27b": "Qwen3.8-27B"}
NAMES = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "focus": "FOCUS", "gptq": "GPTQ‡", "if4": "IF4",
         "zou": "MixFP4 (Zou)", "fq-8x64": "FlipQuant 8x64", "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64"}
LATEX = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "focus": "FOCUS", "gptq": "GPTQ$^{\\ddagger}$",
         "if4": "IF4$^{\\dagger}$", "zou": "MixFP4$^{\\dagger}$", "fq-8x64": "\\method{} $8{\\times}64$",
         "fq-16x64": "\\method{} $16{\\times}64$", "fq-256x64": "\\method{} $256{\\times}64$"}
TASK_NAMES = [("piqa", "PIQA"), ("hellaswag", "HellaSwag"), ("winogrande", "WinoGrande"), ("arc_easy", "ARC-e"),
              ("arc_challenge", "ARC-c"), ("lambada_openai", "LAMBADA")]
OLD_MODELS = {"mistral-7b": "Mistral-7B-Instruct-v0.3", "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2", "phi4-14b": "phi-4",
              "qwen3.8-27b": "Qwen3.8-27B"}
OLD_ROWS = {"BF16": "bf16", "FourOverSix": "fo6", "FOCUS": "focus", "FlipQuant 8x64": "fq-8x64",
            "FlipQuant 16x64": "fq-16x64", "FlipQuant 256x64": "fq-256x64"}


def mcnemar(b: int, c: int) -> float:
    """scripts/acc_panel_tables.py (flipquant main), unchanged: the exact two-sided McNemar p."""
    n = b + c
    if n == 0:
        return 1.0
    lh = n * math.log(0.5)
    return min(1.0, 2 * sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + lh)
                            for i in range(min(b, c) + 1)))


def acc(r, t):
    ex = r["tasks"][t]["examples"]
    return 100 * sum(ex.values()) / len(ex)


def paired(a, b):
    """Pooled over every question of the six tasks: (difference in points, row-only correct, ref-only correct, p)."""
    ka = {f"{t}:{k}": v for t, _ in TASK_NAMES for k, v in a["tasks"][t]["examples"].items()}
    kb = {f"{t}:{k}": v for t, _ in TASK_NAMES for k, v in b["tasks"][t]["examples"].items()}
    assert set(ka) == set(kb)
    c = sum(ka[k] > kb[k] for k in ka)          # a correct only
    bb = sum(kb[k] > ka[k] for k in ka)         # reference correct only
    return dict(diff=100 * (c - bb) / len(ka), row_only=c, ref_only=bb, p=mcnemar(bb, c), n=len(ka))


def old_panel():
    text = (Z.FQ / "docs" / "results" / "acc_panel" / "acc_panel.md").read_text()
    part = text.split("## Zero-shot", 1)[1]
    out = {}
    for line in part.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] in ("model", "---"):
            continue
        model = next((k for k, v in OLD_MODELS.items() if v == cells[0]), None)
        row = OLD_ROWS.get(cells[1])
        if model and row:
            out.setdefault(model, {})[row] = {t: float(cells[2 + i]) for i, (t, _) in enumerate(TASK_NAMES)}
    return out


def queue_times():
    times = {}
    for line in (G.RUN / "queue.log").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("START", "END") and parts[2].startswith("zs6_"):
            times.setdefault(parts[2], {})[parts[1].lower()] = parts[0]
    return times


def panel():
    data, rec = {}, dict(models={}, checks={})
    for m in Z.MODELS:
        for row in Z.ROWS:
            r = load(ZS / "panel" / m / f"{row}.json")
            if r is not None and r.get("status") == "complete":
                data.setdefault(m, {})[row] = r
    xc = {row: load(ZS / "crosscheck" / "mistral-7b" / f"{row}_fake.json") for row in ("fo6", "fq-16x64")}
    old = old_panel()
    qt = queue_times()
    md = ["# Zero-shot accuracy: six tasks, six models, ten rows (part X)", "",
          "flipquant origin/main `3db0072` `evaluation.downstream` (fresh detached worktree, unmodified; the "
          "llama3.1-8b-instruct registry entry added at run time by `zs6_fq.py`, deviation 1), lm-eval 0.4.11, 0-shot, "
          "the paper's settings (batch 16, Qwen3.8-27B 8; per-token activations; datasets pinned, LAMBADA at 900124bf). "
          "Native rows on the RTX PRO 6000 (build_V, `--kernel-set auto`); IF4 and MixFP4 (Zou) simulated with their own "
          "rules (†), as in tab:main-ppl. Every cell is a new run. Step 0 (code identity against NVFP4-RaZeR): "
          "STEP0.md. PROTOCOL.md part X.", "",
          "Accuracy in %: acc_norm for PIQA, HellaSwag, ARC-e and ARC-c; acc for WinoGrande and LAMBADA. Avg. = the "
          "unweighted mean of the six. The paired columns pool every question of the six tasks (20,848 per row): "
          "difference in points (row-only correct / reference-only correct, exact McNemar p), * = p < 0.05 "
          "(acc_panel_tables.py's formula).", "",
          "| model | method | " + " | ".join(n for _, n in TASK_NAMES) + " | Avg. | vs FourOverSix | vs BF16 |",
          "|---|---|" + "---:|" * (len(TASK_NAMES) + 1) + "---|---|"]
    for m in Z.MODELS:
        rows = data.get(m, {})
        mrec = rec["models"][m] = {}
        for row in Z.ROWS:
            r = rows.get(row)
            if r is None:
                md.append(f"| {TITLES[m]} | {NAMES[row]} | " + " | ".join("TBD" for _ in TASK_NAMES) + " | TBD | | |")
                continue
            accs = {t: acc(r, t) for t, _ in TASK_NAMES}
            avg = sum(accs.values()) / len(accs)
            cells, cmp = [], {}
            for ref in ("fo6", "bf16"):
                if row == ref or ref not in rows:
                    cells.append("—" if row == ref else "")
                    continue
                pr = paired(r, rows[ref])
                cmp[ref] = pr
                cells.append(f"{pr['diff']:+.2f}{'*' if pr['p'] < 0.05 else ''} ({pr['row_only']}/{pr['ref_only']}, "
                             f"p={pr['p']:.2g})")
            mrec[row] = dict(acc=accs, avg=avg, paired=cmp,
                             lm_eval={t: {k: v for k, v in r["tasks"][t]["metrics"].items()} for t, _ in TASK_NAMES},
                             timing=dict(load_seconds=r.get("load_seconds"), seconds=r.get("seconds"),
                                         tasks={t: r["tasks"][t]["seconds"] for t, _ in TASK_NAMES},
                                         model_seconds={t: r["tasks"][t]["model_timing"].get("loglikelihood_seconds")
                                                        for t, _ in TASK_NAMES},
                                         queue=qt.get(f"zs6_panel_{m}_{row}")),
                             sample_digest={t: r["tasks"][t]["sample_digest"] for t, _ in TASK_NAMES},
                             peak_gib=max(r["tasks"][t]["peak_gpu_allocated_gib"] for t, _ in TASK_NAMES),
                             tokenization=r.get("tokenization"))
            md.append(f"| {TITLES[m]} | {NAMES[row]}{'†' if row in ('if4', 'zou') else ''} | "
                      + " | ".join(f"{accs[t]:.2f}" for t, _ in TASK_NAMES) + f" | {avg:.2f} | " + " | ".join(cells) + " |")
    # sample digests across models
    dig = {t: {m: rec["models"][m][r]["sample_digest"][t] for m in rec["models"] for r in rec["models"][m]}
           for t, _ in TASK_NAMES}
    rec["checks"]["sample_digest_equal_across_models"] = {t: len(set(v.values())) == 1 for t, v in dig.items()}
    md += ["", "Every row of a model saw the same samples (equal sample_digest per task; the driver checks it). Across "
           "models: " + ", ".join(f"{n} {'equal' if rec['checks']['sample_digest_equal_across_models'][t] else 'differ'}"
                                  for t, n in TASK_NAMES) + "."]
    # LaTeX
    md += ["", "## LaTeX", "", "```latex"]
    for m in Z.MODELS:
        md.append(f"\\midrule\\multicolumn{{9}}{{l}}{{\\textit{{{TITLES[m]}}}}} \\\\")
        for row in Z.ROWS:
            v = rec["models"][m].get(row)
            if v is None:
                continue
            md.append(f"{LATEX[row]} & " + " & ".join(f"{v['acc'][t]:.2f}" for t, _ in TASK_NAMES)
                      + f" & {v['avg']:.2f} \\\\")
    md += ["```", "", "The columns: method, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, LAMBADA, Avg.; values rounded once "
           "from the per-example records."]
    # old vs new
    md += ["", "## Old vs new: the co-author's acc_panel (flipquant main docs/results/acc_panel/acc_panel.md)", "",
           "Old: fake quant on H200 (scripts/run_main_acc.py, run_small_acc.py), only the table's rounded percentages "
           "(no per-example records here). New: this run (native rows on the RTX PRO 6000). Δ = new − old in points; "
           "2 SE = 2 sqrt(p1 q1 / n + p2 q2 / n) (binomial, unpaired); ✗ = |Δ| > 2 SE. The old Llama-3.1-8B row is the base "
           "model and is not compared.", "",
           "| model | method | " + " | ".join(n for _, n in TASK_NAMES) + " | Avg. Δ |", "|---|---|" + "---|" * (len(TASK_NAMES) + 1)]
    rec["old_vs_new"] = {}
    flagged = 0
    for m, rows in old.items():
        for row, o in rows.items():
            v = rec["models"].get(m, {}).get(row)
            if v is None:
                continue
            cells, entry = [], {}
            for t, _ in TASK_NAMES:
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
    n_cells = sum(len(r) * len(TASK_NAMES) for r in rec["old_vs_new"].values())
    md += ["", f"Cells beyond 2 SE: {flagged} of {n_cells}. Each cell reads new / old (Δ)."]
    rec["old_vs_new_summary"] = dict(cells=n_cells, beyond_2se=flagged)
    # native / fake cross-check
    md += ["", "## Native vs simulated (fake) cross-check: Mistral-7B-Instruct-v0.3", "",
           "The same flags with `--mode fake`. Per task: accuracy native / fake, the paired difference (native-only / "
           "fake-only correct, exact McNemar p), the share of questions with the same correctness, and the per-choice "
           "log-likelihood difference (mean |Δ| and max |Δ|, nats).", "",
           "| row | task | native | fake | Δ (points) | native-only / fake-only, p | same correctness | mean / max |ΔLL| |",
           "|---|---|---:|---:|---:|---|---:|---|"]
    rec["crosscheck"] = {}
    for row, f in xc.items():
        nat = data.get("mistral-7b", {}).get(row)
        if f is None or nat is None or f.get("status") != "complete":
            md.append(f"| {NAMES[row]} | pending | | | | | | |")
            continue
        for t, tn in TASK_NAMES:
            a, b = nat["tasks"][t], f["tasks"][t]
            assert a["sample_digest"] == b["sample_digest"]
            ea, eb = a["examples"], b["examples"]
            c = sum(ea[k] > eb[k] for k in ea)
            bb = sum(eb[k] > ea[k] for k in ea)
            same = sum(ea[k] == eb[k] for k in ea) / len(ea)
            dl = [abs(x - y) for k in a["loglikelihoods"] for x, y in zip(a["loglikelihoods"][k]["ll"], b["loglikelihoods"][k]["ll"])]
            e = dict(native=acc(nat, t), fake=acc(f, t), native_only=c, fake_only=bb, p=mcnemar(bb, c), same=same,
                     mean_abs_dll=sum(dl) / len(dl), max_abs_dll=max(dl))
            rec["crosscheck"].setdefault(row, {})[t] = e
            md.append(f"| {NAMES[row]} | {tn} | {e['native']:.2f} | {e['fake']:.2f} | {e['native'] - e['fake']:+.2f} | "
                      f"{c} / {bb}, p={e['p']:.2g} | {100 * same:.2f} % | {e['mean_abs_dll']:.4f} / {e['max_abs_dll']:.3f} |")
        pr = paired(nat, f)
        rec["crosscheck"][row]["pooled"] = pr
        md.append(f"| {NAMES[row]} | pooled | {sum(acc(nat, t) for t, _ in TASK_NAMES) / 6:.2f} | "
                  f"{sum(acc(f, t) for t, _ in TASK_NAMES) / 6:.2f} | {pr['diff']:+.2f} | {pr['row_only']} / {pr['ref_only']}, "
                  f"p={pr['p']:.2g} | | |")
        rec["crosscheck"][row]["fake_seconds"] = f.get("seconds")
    # timings
    md += ["", "## Timings", "", "Per run: model load, each task's wall time, and the whole process (seconds); queue START "
           "(UTC). Simulated rows are marked †.", "",
           "| model | method | load | " + " | ".join(n for _, n in TASK_NAMES) + " | total | start |",
           "|---|---|---:|" + "---:|" * (len(TASK_NAMES) + 1) + "---|"]
    total = 0.0
    for m in Z.MODELS:
        for row in Z.ROWS:
            v = rec["models"][m].get(row)
            if v is None:
                continue
            tm = v["timing"]
            total += tm["seconds"] or 0
            md.append(f"| {TITLES[m]} | {NAMES[row]}{'†' if row in ('if4', 'zou') else ''} | {tm['load_seconds']:.0f} | "
                      + " | ".join(f"{tm['tasks'][t]:.0f}" for t, _ in TASK_NAMES) + f" | {tm['seconds']:.0f} | "
                      f"{(tm['queue'] or {}).get('start', '')} |")
    for row, f in xc.items():
        if f and f.get("status") == "complete":
            total += f.get("seconds") or 0
            md.append(f"| Mistral-7B-Instruct-v0.3 | {NAMES[row]} (fake, cross-check) | {f['load_seconds']:.0f} | "
                      + " | ".join(f"{f['tasks'][t]['seconds']:.0f}" for t, _ in TASK_NAMES) + f" | {f['seconds']:.0f} | "
                      f"{(qt.get(f'zs6_xcheck_mistral-7b_{row}') or {}).get('start', '')} |")
    md += ["", f"Sum of the process times: {total / 3600:.1f} h."]
    rec["total_process_hours"] = total / 3600
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zeroshot6.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:80]))


if __name__ == "__main__":
    {"step0": step0, "panel": panel}[sys.argv[1]]()
