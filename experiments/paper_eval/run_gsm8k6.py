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
import time
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


#: amendment 4: Qwen3.8-27B's fake rows ran out of memory at batch 64 under the default allocator (fragmentation, 13:08
#: UTC 2026-10-16); these two rows run with expandable segments, every other row as registered
ALLOC = {("qwen3.8-27b", "if4"): "expandable_segments:True", ("qwen3.8-27b", "zou"): "expandable_segments:True"}
#: amendment 4's addendum (the user's approval): these rows fall back to batch 48, then 32, each only after a CUDA OOM
#: at the batch before; an OOM at 32 stops the run
FALLBACK = (64, 48, 32)


def oom_log(model, row, b):
    """The log of a run of an ALLOC row that ran out of memory at batch b (kept, renamed)."""
    return ROOT / model / f"{row}_oom-b{b}-expandable.log"


def expected_batches(model, row):
    """The batch sizes a complete record may carry: the registered one; for the ALLOC rows also each fallback batch
    whose predecessor's OOM log exists."""
    if (model, row) not in ALLOC:
        return {batch(model, row)}
    ok = {FALLBACK[0]}
    for prev, nxt in zip(FALLBACK, FALLBACK[1:]):
        if not oom_log(model, row, prev).exists():
            break
        ok.add(nxt)
    return ok


def run_ts_env(name, cmd_, out, cwd, extra):
    """run_gsm8k.run_ts with extra environment variables, which it writes to the job's log (amendment 4)."""
    if G.done(out):
        return None
    G.idle()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    G.log(f"START {name}")
    t0 = time.time()
    with open(out.with_suffix(".log"), "a") as f:
        f.write(f"{time.time():.3f} ENV " + " ".join(f"{k}={v}" for k, v in extra.items()) + "\n")
        f.write(f"{time.time():.3f} CMD " + " ".join(map(str, cmd_)) + "\n")
        f.flush()
        p = subprocess.Popen(list(map(str, cmd_)), cwd=cwd, env=G.env(extra), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1)
        for line in p.stdout:
            f.write(f"{time.time():.3f} {line}")
            f.flush()
        rc = p.wait()
    G.log(f"END {name} rc={rc} {time.time() - t0:.0f}s")
    return rc


def cmd(model, row, out, batch_size=None):
    """Part N's command (run_gsm8k_lmeval.cmd); amendment 3's two rows go through gsm8k6_date.py; amendment 4's
    fallback passes its batch."""
    launch = ["-u", HERE / "gsm8k6_date.py", DATE_PIN[(model, row)]] if (model, row) in DATE_PIN else ["-u", "-m"]
    return [G.py(model), *launch, "evaluation.downstream", "--model", model, "--revision", Z.REV[model],
            *quant(model, row), "--tasks", TASK, "--apply-chat-template", *THINK_OFF.get(model, []),
            "--batch-size", batch_size or batch(model, row), "--gen-lengths", "--samples-out", Path(out).with_suffix(""),
            "--out", out]


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
    if r["batch_size"] not in expected_batches(model, row):
        bad.append(f"batch {r['batch_size']} (allowed {sorted(expected_batches(model, row))})")
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


def alloc_row(model, row, out):
    """Amendment 4 (+ addendum): expandable segments at batch 64, then 48, then 32. Each fallback happens only after a
    CUDA out-of-memory error at the batch before; that run's log is kept as oom_log(). Any other failure returns its rc
    (the caller stops)."""
    extra = {"PYTORCH_CUDA_ALLOC_CONF": ALLOC[(model, row)]}
    out, rc = Path(out), None
    for i, b in enumerate(FALLBACK):
        if G.done(out):
            return rc
        if i and not oom_log(model, row, FALLBACK[i - 1]).exists():
            continue                                  # reached only by a complete earlier batch (done above)
        if oom_log(model, row, b).exists():
            continue                                  # this batch already ran out of memory (a restart)
        rc = run_ts_env(f"gsm8k6_{model}_{row}", cmd(model, row, out, b), out, FQ, extra)
        if rc == 0:
            return rc
        log = out.with_suffix(".log")
        text = log.read_text(errors="replace") if log.exists() else ""
        if "torch.OutOfMemoryError" not in text and "CUDA out of memory" not in text:
            return rc
        log.rename(oom_log(model, row, b))
        for f in (out, out.with_suffix(f".{TASK}.jsonl.gz")):
            if f.exists():
                f.rename(f.with_name(f.name.replace(f"{row}.", f"{row}_oom-b{b}-expandable.", 1)))
        nxt = FALLBACK[i + 1] if i + 1 < len(FALLBACK) else None
        G.log(f"gsm8k6 {model} {row}: CUDA out of memory at batch {b} (amendment 4 addendum); "
              + (f"rerunning at batch {nxt}" if nxt else "no fallback left"))
    if not G.done(out):
        G.stop(f"gsm8k6 {model} {row}: CUDA out of memory at batch {FALLBACK[-1]} (amendment 4 addendum: stop and report)")
    return 0


def run_row(model, row, check=True):
    global _LIBS
    out = ROOT / model / f"{row}.json"
    if not G.done(out):
        checkout(FQ, FQ_COMMIT)
        bad = Z.verify_inputs(model, row)
        if bad:
            G.stop(f"gsm8k6 {model} {row}: {bad}")
    if (model, row) in ALLOC:
        rc = alloc_row(model, row, out)
    else:
        rc = M.run_ts(f"gsm8k6_{model}_{row}", cmd(model, row, out), out, FQ)
    must(rc, f"{model} {row}")
    if check:
        _LIBS = _LIBS or G.manifests()
        bad = check_record(model, row, out, _LIBS, _DIGESTS.setdefault(model, {}))
        if bad:
            G.stop(f"gsm8k6 {model} {row}: {bad}")


def amend4_check():
    """Amendment 4's pre-check: Qwen3.8-27B IF4 with --limit 64 (one batch of 64), under the default allocator and
    under expandable segments. Every generation and extracted answer must be byte-identical."""
    model, row, d = "qwen3.8-27b", "if4", ROOT / "amend4"
    checkout(FQ, FQ_COMMIT)
    outs = {}
    for tag, extra in (("default", None), ("expandable", {"PYTORCH_CUDA_ALLOC_CONF": ALLOC[(model, row)]})):
        out = d / f"{model}_{row}_limit64_{tag}.json"
        c = cmd(model, row, out) + ["--limit", 64]
        name = f"gsm8k6_amend4_check_{tag}"
        must(run_ts_env(name, c, out, FQ, extra) if extra else M.run_ts(name, c, out, FQ), f"amendment 4 check {tag}")
        outs[tag] = out
    a, b = (samples(outs[t].with_suffix(f".{TASK}.jsonl.gz")) for t in ("default", "expandable"))
    keys = sorted(set(a) | set(b))
    gen = sum(k in a and k in b and a[k][0] == b[k][0] for k in keys)
    ans = sum(k in a and k in b and a[k][1] == b[k][1] for k in keys)
    res = dict(samples=len(keys), generations_identical=gen, answers_identical=ans,
               ok=len(keys) == 2 * 64 and gen == ans == len(keys), records={t: str(o) for t, o in outs.items()})
    (d / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    G.log(f"gsm8k6 amendment 4 pre-check: generations {gen}/{len(keys)}, answers {ans}/{len(keys)} byte-identical")
    if not res["ok"]:
        G.stop("gsm8k6 amendment 4 pre-check: the allocator changed the output (RUN/gsm8k6/amend4/check.json)")


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
    elif what == "amend4_check":
        G.log("gsm8k6 amendment 4 pre-check")
        amend4_check()
    elif what == "panel":
        G.log("gsm8k6 panel")
        panel()
    else:
        raise SystemExit(f"unknown part {what}")
