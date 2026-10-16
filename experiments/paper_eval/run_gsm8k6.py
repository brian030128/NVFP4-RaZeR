"""Part Y (PROTOCOL.md): GSM8K (lm-eval 0.4.11 gsm8k_llama, part N's protocol) for the six paper models x ten rows on
flipquant paper-sm120-runs 4bbe0b1 (worktree sm120runs/wt), with part X's verified artifacts.

    python run_gsm8k6.py step0      # (b) ten-row numeric parity 4bbe0b1 vs main 3db0072 on Llama-3.2-1B-Instruct;
                                    # (c') Nemotron FO6 / FQ16 full runs (the panel's cells) vs part N's samples
    python run_gsm8k6.py panel      # native rows for every model (small first), then IF4 / Zou

Records: RUN/gsm8k6/<model>/<row>.json (+ .log, + .gsm8k_llama.jsonl.gz); step 0 under RUN/gsm8k6/step0/. Part N's
runner (run_gsm8k.run_ts: START / END in queue.log, a time-stamped log) and env (run_gpu.env()); one job at a time.
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gpu as G  # noqa: E402
import run_gsm8k as M  # noqa: E402
import run_sm120 as S  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

FQ = S.FQ2                                                          # flipquant paper-sm120-runs worktree
FQ_COMMIT = "4bbe0b1005ef15d618057da55733aa52612da3af"
MAIN_WT, MAIN_COMMIT = Z.FQ, Z.FQ_COMMIT                            # flipquant main 3db0072 (part X's worktree)
ROOT = G.RUN / "gsm8k6"
PART_N = G.RUN / "ptq_gsm8k_lmeval"
TASK = "gsm8k_llama"
MODELS = list(Z.MODELS)                                             # small first, Qwen3.8-27B last
NATIVE_ROWS = ["bf16", "nvfp4", "fo6", "focus", "gptq", "fq-8x64", "fq-16x64", "fq-256x64"]
FAKE_ROWS = ["if4", "zou"]
ROWS = NATIVE_ROWS[:5] + FAKE_ROWS + NATIVE_ROWS[5:]                # the report's order
THINK_OFF = {"qwen3.8-27b": ["--enable-thinking", "false"], "nemotron-nano-9b-v2": ["--system-instruction", "/no_think"]}
GSM8K_PIN = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
N_DOCS = 1319


def batch(model, row):
    return 48 if (model, row) == ("qwen3.8-27b", "bf16") else 64       # O0's memory rule for Qwen3.8-27B BF16


def quant(model, row):
    """Part N's quantization flags (run_gsm8k.quant) for part X's rows."""
    q = [*G.NATIVE, "--kernel-set", "auto"]
    if row == "bf16":
        return ["--mode", "bf16"]
    if row == "nvfp4":
        return q + ["--weight", "nvfp4", "--act", "nvfp4", "--act-scope", "row"]
    if row == "fo6":
        return q + ["--weight", "fourover6", "--act", "fourover6", "--act-scope", "row"]
    if row.startswith("fq-"):
        return q + ["--weight", "mixfp4", "--act", "fourover6", "--map", Z.map_path(model, row[3:]), "--act-scope", "row"]
    if row == "focus":
        return q + ["--weight", "focus", "--focus", Z.focus_state(model), "--focus-deploy", "--act", "fourover6",
                    "--act-scope", "row"]
    if row == "gptq":
        return q + ["--weight", "nvfp4", "--act", "fourover6", "--any-act", "--act-scope", "row", *Z.gptq_flags(model)]
    if row in ("if4", "zou"):
        return ["--mode", "fake", "--weight", "if4" if row == "if4" else "zou_mixfp4", "--act-method", "own", "--unit", "1x16"]
    raise KeyError(row)


#: amendment 3: Llama-3.2-1B-Instruct's chat template writes today's date into its system header; its fake rows render
#: the date its native rows saw (gsm8k6_date.py pins strftime_now and nothing else)
DATE_PIN = {("llama3.2-1b-instruct", "if4"): "2026-10-15", ("llama3.2-1b-instruct", "zou"): "2026-10-15"}


def cmd(model, row, out):
    """Part N's command (run_gsm8k_lmeval.cmd); amendment 3's two rows go through gsm8k6_date.py."""
    launch = ["-u", HERE / "gsm8k6_date.py", DATE_PIN[(model, row)]] if (model, row) in DATE_PIN else ["-u", "-m"]
    return [G.py(model), *launch, "evaluation.downstream", "--model", model, "--revision", Z.REV[model],
            *quant(model, row), "--tasks", TASK, "--apply-chat-template", *THINK_OFF.get(model, []),
            "--batch-size", batch(model, row), "--gen-lengths", "--samples-out", Path(out).with_suffix(""), "--out", out]


def must(rc, what):
    if rc not in (None, 0):
        G.stop(f"gsm8k6 {what}: rc={rc}")


def checkout(wt, commit):
    head = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", wt, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    if head != commit or dirty:
        G.stop(f"gsm8k6: flipquant {wt} is at {head[:7]} (dirty: {bool(dirty)}), not a clean {commit[:7]}")


def samples(path):
    """Part N's / this part's lm-eval samples: {(doc_id, filter): (resps, filtered_resps)}."""
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            s = json.loads(line)
            out[(int(s["doc_id"]), s["filter"])] = (s["resps"], s["filtered_resps"])
    return out


def think_tags(path):
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            s = json.loads(line)
            n += sum(("<think>" in t or "</think>" in t) for r in s["resps"] for t in (r if isinstance(r, list) else [r]))
    return n


# ---------------------------------------------------------------------------------------------- step 0

def step0_parity():
    """(b): every row's default path, 4bbe0b1 vs main 3db0072, evaluation.ppl --limit 2 on Llama-3.2-1B-Instruct."""
    model = "llama3.2-1b-instruct"
    for tag, wt, commit in (("fq4bbe0b1", FQ, FQ_COMMIT), ("main3db0072", MAIN_WT, MAIN_COMMIT)):
        checkout(wt, commit)
        for row in ROWS:
            out = ROOT / "step0" / "parity" / tag / f"{row}.json"
            c = [G.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", Z.REV[model], *quant(model, row),
                 "--limit", 2, "--out", out]
            must(G.run(f"gsm8k6_parity_{tag}_{row}", c, out, wt), f"parity {tag} {row}")


def step0_identity():
    """(c'): Nemotron FO6 / FQ16, the panel's cells, against part N's samples, generation by generation."""
    model = "nemotron-nano-9b-v2"
    checkout(FQ, FQ_COMMIT)
    res = {}
    for row, old in (("fo6", "rtn_fo6"), ("fq-16x64", "rtn_fq-16x64")):
        run_row(model, row, check=False)
        a = samples(ROOT / model / f"{row}.{TASK}.jsonl.gz")
        b = samples(PART_N / model / f"{old}.{TASK}.jsonl.gz")
        keys = sorted(set(a) | set(b))
        same_gen = [k for k in keys if k in a and k in b and a[k][0] == b[k][0]]
        same_ans = [k for k in keys if k in a and k in b and a[k][1] == b[k][1]]
        diff = [k for k in keys if not (k in a and k in b and a[k] == b[k])]
        res[row] = dict(samples=len(keys), generations_identical=len(same_gen), answers_identical=len(same_ans),
                        first_difference=None if not diff else dict(doc_id=diff[0][0], filter=diff[0][1],
                                                                     new=a.get(diff[0]), part_n=b.get(diff[0])),
                        part_n=str(PART_N / model / f"{old}.json"))
        res[row]["ok"] = not diff and len(keys) == 2 * N_DOCS
    (ROOT / "step0").mkdir(parents=True, exist_ok=True)
    (ROOT / "step0" / "identity_part_n.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for row, v in res.items():
        G.log(f"gsm8k6 identity {model} {row}: generations {v['generations_identical']}/{v['samples']}, answers "
              f"{v['answers_identical']}/{v['samples']}")
    if not all(v["ok"] for v in res.values()):
        G.stop("gsm8k6 step 0 (c'): the Nemotron FO6 / FQ16 generations differ from part N's (identity_part_n.json)")


# ---------------------------------------------------------------------------------------------- the panel

def check_record(model, row, path, libs, digests):
    r = json.loads(Path(path).read_text())
    if r.get("status") != "complete":
        return ["not complete"]
    bad, pol = [], r.get("policy") or {}
    if (pol.get("source") or {}).get("revision") != Z.REV[model]:
        bad.append(f"revision {(pol.get('source') or {}).get('revision')}")
    if r["versions"]["lm_eval"] != "0.4.11":
        bad.append(f"lm_eval {r['versions']['lm_eval']}")
    if r["batch_size"] != batch(model, row):
        bad.append(f"batch {r['batch_size']}")
    co = r.get("chat_options") or {}
    want_sys = "/no_think" if model == "nemotron-nano-9b-v2" else None
    want_think = "false" if model == "qwen3.8-27b" else None
    if not (co.get("apply_chat_template") and co.get("fewshot_as_multiturn")) or co.get("system_instruction") != want_sys \
            or co.get("enable_thinking") != want_think:
        bad.append(f"chat options {co}")
    if model == "qwen3.8-27b" and (r.get("chat_template_args") or {}).get("enable_thinking") is not False:
        bad.append(f"chat_template_args {r.get('chat_template_args')}")
    loads = r.get("dataset_loads") or []
    if not loads or any((x["loaded"], x["revision"]) != GSM8K_PIN for x in loads):
        bad.append(f"dataset loads {loads[:2]}")
    t = (r.get("tasks") or {}).get(TASK)
    if t is None:
        return bad + ["gsm8k_llama missing"]
    ex = t.get("examples") or {}
    if len(ex) != N_DOCS or t.get("n_shot") != 8:
        bad.append(f"{len(ex)} problems, n_shot {t.get('n_shot')}")
    incomplete = [k for k, e in ex.items() if not all(f in e for f in ("strict_match", "flexible_extract", "generated"))
                  or (e.get("generated") or {}).get("budget") != 1024]
    if incomplete:
        bad.append(f"examples without both filters or gen lengths: {incomplete[:3]}")
    if digests.setdefault("digest", t.get("sample_digest")) != t.get("sample_digest"):
        bad.append("sample_digest differs from the model's first row")
    spath = Path(path).with_suffix(f".{TASK}.jsonl.gz")
    if not spath.exists():
        bad.append("samples file missing")
    elif think_tags(spath):
        bad.append(f"think tags in {think_tags(spath)} outputs")
    if row in Z.FAMILY:
        nat = pol.get("native") or {}
        if nat.get("kernel") != Z.FAMILY[row] or (nat.get("kernel_set") or {}).get("family") != Z.FAMILY[row]:
            bad.append(f"kernel family {nat.get('kernel')} != {Z.FAMILY[row]}")
        if Path(nat.get("build_dir") or "/nonexistent").resolve() != G.BUILD_V.resolve():
            bad.append(f"build_dir {nat.get('build_dir')}")
        ok, found = Z.libraries_ok([nat.get("kernel_set"), (r.get("coverage") or {}).get("kernel_set")], libs)
        if not ok:
            bad.append(f"libraries not from build_V: {found[:3]}")
        cov = r.get("coverage") or {}
        if not cov or cov.get("native_called") != cov.get("native") or cov.get("remaining_bf16_linears_called"):
            bad.append("coverage")
    if row == "gptq":
        want = json.loads(Z.gptq_record(model).read_text())["ptq"]["codes_sha256"]
        if (pol.get("ptq") or {}).get("codes_sha256") != want:
            bad.append(f"codes_sha256 {(pol.get('ptq') or {}).get('codes_sha256')}")
    if row in ("if4", "zou"):
        want = Z.installed_sha(json.loads(Z.mainrow_record(model, row).read_text()))
        if not want or Z.installed_sha(r) != want:
            bad.append(f"installed weight sha256 {Z.installed_sha(r)} != {want}")
    if row == "focus" and (pol.get("focus") or {}).get("state") != str(Z.focus_state(model)):
        bad.append("FOCUS state path")
    if row.startswith("fq-") and (pol.get("map") or {}).get("path") != str(Z.map_path(model, row[3:])):
        bad.append("map path")
    return bad


_LIBS, _DIGESTS = None, {}


def run_row(model, row, check=True):
    global _LIBS
    out = ROOT / model / f"{row}.json"
    if not G.done(out):
        checkout(FQ, FQ_COMMIT)
        bad = Z.verify_inputs(model, row)
        if bad:
            G.stop(f"gsm8k6 {model} {row}: {bad}")
    must(M.run_ts(f"gsm8k6_{model}_{row}", cmd(model, row, out), out, FQ), f"{model} {row}")
    if check:
        _LIBS = _LIBS or G.manifests()
        bad = check_record(model, row, out, _LIBS, _DIGESTS.setdefault(model, {}))
        if bad:
            G.stop(f"gsm8k6 {model} {row}: {bad}")


def panel():
    if not (ROOT / "step0" / "step0_ok.json").exists() or not json.loads((ROOT / "step0" / "step0_ok.json").read_text()).get("ok"):
        G.stop("gsm8k6: step 0 has not passed")
    for rows in (NATIVE_ROWS, FAKE_ROWS):
        for model in MODELS:
            for row in rows:
                run_row(model, row)
            G.log(f"gsm8k6 {model}: rows {','.join(rows)} done")
    G.log("DONE gsm8k6 panel")


if __name__ == "__main__":
    what = sys.argv[1]
    if not (G.RUN / "env" / "fast_env_verified.json").exists():
        G.stop("the hybrid models run only in n16k64-fast, after its verification (amendment 3)")
    if what == "step0":
        G.log("gsm8k6 step0")
        step0_parity()
        step0_identity()
        G.log("DONE gsm8k6 step0 runs")
    elif what == "panel":
        G.log("gsm8k6 panel")
        panel()
    else:
        raise SystemExit(f"unknown part {what}")
