"""Part AA (PROTOCOL.md): calibration size, 1,024 / 2,048 fit windows for FlipQuant 16x64 and FOCUS, on flipquant
calib-size f912a30 (worktree calib_size/wt).

    python run_calib_size.py windows     # CPU: the 1,024 / 2,048 sets (calibration.fit_windows), the nesting checks
    python run_calib_size.py step0       # (b) calibration identity, (c) evaluation identity, (c2) PPL references,
                                         # (d) FOCUS reproducibility
    python run_calib_size.py probes      # 2,048-window memory probes: FlipQuant Qwen3.8-27B, FOCUS Phi-4
    python run_calib_size.py pilot       # Llama-3.2-1B-Instruct and Phi-4 at 1,024: FQ16 + FOCUS, zero-shot / GSM8K / PPL
    python run_calib_size.py panel --size=1024 [--models=a,b]    # amendment 1: the full panel at one size, ORDER

The audit is calib_size_audit.py (CPU). Records under RUN/calib_size/; START / END in RUN/queue.log; one GPU job at a
time on an idle GPU (run_gpu). A finished record is not run again; a failed job or check stops the part (STOP).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gpu as G  # noqa: E402
import run_gsm8k as M  # noqa: E402
import run_gsm8k6 as Y  # noqa: E402
import run_llama_instruct as LI  # noqa: E402
import run_sm120 as S  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

FQ3 = Path("/home/dev/n16k64_campaign/calib_size/wt")              # flipquant calib-size
#: the registered commit: f912a30 for step 0, the probes and the pilot; 898d00e (+ train_focus --offload) for the panel,
#: after calib-size's fast-forward (amendment 2)
CODE = "898d00e"
VAULT = Path("/vault/flipquant_paper_eval/calib_size")             # amendment 2: FOCUS states after the pilot
ROOT = G.RUN / "calib_size"
WIN = ROOT / "windows"
DATA = S.DATA_ROOT                                                  # fqrel/tmopt_data, the release runs' records
KEYS = {"llama3.2-1b-instruct": "llama1b_ins", "phi4-14b": "phi4", "qwen3.8-27b": "qwen27b", "mistral-7b": "mistral7b_ins",
        "llama3.1-8b-instruct": "llama8b_ins", "nemotron-nano-9b-v2": "nemotron9b", "qwen3-1.7b": "qwen3_1p7b",
        "qwen3-8b": "qwen3_8b"}
PILOT = ["llama3.2-1b-instruct", "phi4-14b"]
#: amendment 1: the pilot, then the coordinator's order; 1,024 for all, then 2,048 in the same order
ORDER = PILOT + ["mistral-7b", "llama3.1-8b-instruct", "nemotron-nano-9b-v2", "qwen3.8-27b", "qwen3-1.7b", "qwen3-8b"]
SIZES = {m: (1024, 2048) for m in ORDER}
QWEN3 = ("qwen3-1.7b", "qwen3-8b")
HOST_TEACHER = {"qwen3.8-27b"}               # amendment 1: its top-K teacher cannot stay on the GPU at either size
FOCUS_PENDING = set()                        # amendment 2: Qwen3.8-27B's FOCUS runs locally with the offload
FOCUS_EXTRA = {"qwen3.8-27b": ["--offload"]}  # amendment 2: part U's host-offloaded AdamW (ported, 898d00e)
FOCUS_MICRO = {"qwen3.8-27b": 1}              # batch1_only (as focus_qwen27b and part U); every other model 8
THINK = dict(Y.THINK_OFF, **{m: ["--enable-thinking", "false"] for m in QWEN3})
QWEN3_GSM_ROWS = ["bf16", "nvfp4", "fo6", "gptq", "fq-16x64", "focus"]   # amendment 1: native GSM8K baselines
FLOOR = 10e9                                 # /home's floor (bytes)
TRAIN = ["--unit", "16x64", "--epochs", 5, "--teacher-topk", 1000]
DATE = "2026-10-15"                                                 # part Y amendment 3: Llama-1B's GSM8K prompts
LLAMA1B = "llama3.2-1b-instruct"
E0M3_MAX = 0.25                                                     # amendment 1: a larger E0M3 share stops the run


def stop(msg):
    G.stop(f"calib_size: {msg}")


def must(rc, what):
    if rc not in (None, 0):
        stop(f"{what}: rc={rc}")


def checkout():
    """calib_size/wt clean at the registered commit."""
    def git(*a):
        return subprocess.run(["git", "-C", str(FQ3), *a], capture_output=True, text=True).stdout.strip()
    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if not head.startswith(CODE) or dirty:
        stop(f"flipquant {FQ3} at {head[:7]} (dirty: {bool(dirty)}), not a clean {CODE}")
    return head


def release_report(model):
    """The release (part P for Llama-1B) 16x64 trainer report: the 256 set's extension records."""
    return Z.calib_extension(model) if model in Z.MODELS else S.trainer_report(model)


def record(model):
    return DATA / KEYS[model] / "calibration" / "report.json"


def windows_file(model, n):
    return WIN / f"{model}_{n}.json"


def token_list_sha(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def cpu_run(name, cmd, out, cwd, extra=None):
    """run_gpu.run without the idle-GPU wait (CPU jobs)."""
    if Path(out).exists():
        return None
    (G.RUN / "logs").mkdir(parents=True, exist_ok=True)
    G.log(f"START {name} (CPU)")
    t0 = time.time()
    with open(G.RUN / "logs" / f"{name}.log", "w") as f:
        f.write(" ".join(map(str, cmd)) + "\n")
        f.flush()
        rc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=G.env(extra), stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    G.log(f"END {name} rc={rc} {time.time() - t0:.0f}s")
    return rc


# ---------------------------------------------------------------------------------------------- windows

def ext_records(path):
    return json.loads(Path(path).read_text())["fit_extension"]


def by_source(records):
    return {s: [r for r in records if r["source"] == s] for s in ("math", "code")}


def nesting(model):
    """The registered checks: 256 (release) ⊂ 1,024 ⊂ 2,048, per source and in order; equal skip counts."""
    rel = ext_records(release_report(model))
    out, prev, prev_n = dict(model=model, release=str(release_report(model))), rel, 256
    for n in SIZES[model]:
        w = json.loads(windows_file(model, n).read_text())
        ext = w["fit_extension"]
        a, b = by_source(prev["records"]), by_source(ext["records"])
        k = prev_n // 2 - 64
        checks = dict(per_source=all(len(b[s]) == n // 2 - 64 for s in b),
                      prefix=all(b[s][:k] == a[s][:k] and len(a[s]) == k for s in b),
                      skipped_equal=ext["skipped_documents"] == rel["skipped_documents"],
                      base_equal=w["token_sha256"][:128] == json.loads(windows_file(model, SIZES[model][0]).read_text())
                      ["token_sha256"][:128],
                      distinct=len(set(w["token_sha256"])) == n)
        out[str(n)] = dict(checks=checks, ok=all(checks.values()), skipped=ext["skipped_documents"],
                           token_list_sha256=token_list_sha(w["token_sha256"]), contains=prev_n)
        prev, prev_n = ext, n
    out["ok"] = all(out[str(n)]["ok"] for n in SIZES[model])
    return out


def windows():
    WIN.mkdir(parents=True, exist_ok=True)
    checkout()
    for model, sizes in SIZES.items():
        for n in sizes:
            out = windows_file(model, n)
            must(cpu_run(f"cs_windows_{model}_{n}", [G.PY, "-m", "calibration.fit_windows", "--model", model,
                                                     "--fit-windows", n, "--data-root", DATA, "--out", out], out, FQ3),
                 f"windows {model} {n}")
    res = {m: nesting(m) for m in SIZES}
    (WIN / "nesting.json").write_text(json.dumps(res, indent=1) + "\n")
    bad = [m for m, r in res.items() if not r["ok"]]
    if bad:
        stop(f"windows: nesting / skip checks failed for {bad} (windows/nesting.json)")
    G.log("calib_size windows: drawn; 256 ⊂ 1,024 ⊂ 2,048 for " + ", ".join(res))


# ---------------------------------------------------------------------------------------------- jobs

def train_map_cmd(model, n, out, host=False, probe=None):
    cmd = [G.PY, "-m", "calibration.train_map", "--model", model, *TRAIN, "--fit-windows", n, "--data-root", DATA]
    if host:
        cmd.append("--teacher-host")
    if probe:
        cmd += ["--probe-steps", probe]
    return cmd + ["--out", out]


def focus_cmd(model, extension, micro, out, probe=None, extra=()):
    cmd = [G.py(model), "-m", "calibration.train_focus", "--model", model, "--revision", Z.REV[model], "--data",
           "release", "--calib-record", record(model), "--calib-extension", extension, *S.FOCUS_COMMON, "--micro", micro,
           *extra]
    if probe:
        cmd += ["--probe-steps", probe]
    return cmd + ["--out", out]


def measured(name, cmd, done_file, fresh_dir=True):
    """run_llama_instruct.measured (the device-peak sampler) from calib-size; returns (rc, seconds, peak MiB)."""
    checkout()
    return LI.measured(name, cmd, done_file, cwd=FQ3, fresh_dir=fresh_dir)


def write_meta(path, **kw):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(kw, indent=1, default=str) + "\n")


def calibrate_fq(model, n, out, host=False):
    """FlipQuant 16x64 at n windows; out/job.json records the command, the sampler and the cost."""
    meta = out.parent / f"{out.name}.job.json"
    if (out / "reproduction.json").exists() and meta.exists():
        return json.loads(meta.read_text())
    if out.exists():
        stop(f"{out} exists without a record; inspect it before rerunning")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = train_map_cmd(model, n, out, host=host)
    rc, seconds, peak = measured(f"cs_fq16_{model}_{n}{'_host' if host else ''}", cmd, out / "reproduction.json")
    must(rc, f"train_map {model} {n}")
    rep = json.loads((out / "reproduction.json").read_text())
    report = json.loads((out / "run" / "report.json").read_text())
    res = report["resources"]
    rec = dict(model=model, windows=n, teacher_host=host, command=" ".join(map(str, cmd[1:])), checkout=checkout(),
               rc=rc, job_seconds=seconds, device_peak_mib=peak, run_map_sha256=rep["run_map_sha256"],
               map_file_sha256=rep["map_file_sha256"], e0m3_share=rep["e0m3_share"],
               train_map_wall_seconds=rep["wall_seconds"], settings=rep["settings"],
               trainer=dict(total_seconds=res.get("total_seconds"), setup_seconds=report.get("setup_seconds"),
                            training_seconds=report.get("training_seconds"),
                            gpu_peak_allocated_gib=res.get("gpu_peak_allocated_gib"),
                            gpu_peak_reserved_gib=res.get("gpu_peak_reserved_gib"),
                            cpu_peak_rss_gib=res.get("cpu_peak_rss_gib"),
                            phases={p["name"]: dict(seconds=p.get("seconds"), host_peak_rss=p.get("host_peak_rss"),
                                                    gpu_peak_allocated=p.get("gpu_peak_allocated"))
                                    for p in (res.get("phases") or {}).get("phases", [])}),
               teacher_storage=report.get("teacher_storage"))
    write_meta(meta, **rec)
    return rec


def focus_state_ok(state):
    return (state / "report.json").exists() and json.loads((state / "report.json").read_text()).get("status") == "complete"


def calibrate_focus(model, extension, d, tag, meta_dir=None, micro=8, extra=()):
    """FOCUS with FOCUS_COMMON at micro-batch 8, the micro-batch of every local 256-window FOCUS run (Qwen3.8-27B: 1,
    with the offload). Amendment 1 (the coordinator's gate): an OOM is not worked around (no halving); it stops the run.
    The state goes to d/<tag> (amendment 2: /vault); the job record to meta_dir/<tag>.job.json (default d), with a copy of
    the state's report.json. Returns the job record."""
    meta_dir = meta_dir or d
    meta = meta_dir / f"{tag}.job.json"
    state = d / tag
    if focus_state_ok(state) and meta.exists():
        return json.loads(meta.read_text())
    d.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if state.exists():
        stop(f"{state} exists from an earlier attempt; inspect it")
    cmd = focus_cmd(model, extension, micro, state, extra=extra)
    name = f"cs_focus_{model}_{d.name}_micro{micro}"     # the 2026-10-20 naming fix: the size (or control) in the name
    rc, seconds, peak = measured(name, cmd, state / "report.json")
    if rc != 0:
        log = (G.RUN / "logs" / f"{name}.log").read_text(errors="replace")
        oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
        stop(f"focus {model} {tag}: {'CUDA out of memory at micro 8 (amendment 1: stop, no halving)' if oom else f'rc={rc}'}")
    r = json.loads((state / "report.json").read_text())
    rec = dict(model=model, tag=tag, micro=micro, command=" ".join(map(str, cmd[1:])), checkout=checkout(), rc=rc,
               job_seconds=seconds, device_peak_mib=peak, steps=r.get("steps"), windows=r["data"]["windows"],
               training_seconds=r.get("training_seconds"), calibration_seconds=r.get("calibration_seconds"),
               gpu_peak_gib=(r.get("resources") or {}).get("gpu_peak_gib"),
               cpu_peak_rss_gib=(r.get("resources") or {}).get("cpu_peak_rss_gib"), offload=r.get("offload"),
               state=str(state / "focus.pt"), state_bytes=(state / "focus.pt").stat().st_size,
               state_sha256=file_sha256(state / "focus.pt"), token_sha256=r["data"]["records"])
    if meta_dir != d:
        (meta_dir / "report.json").write_text((state / "report.json").read_text())
    write_meta(meta, **rec)
    return rec


def file_sha256(path):
    """A fresh read of the whole file (no cache): amendment 2's write record and read-back check."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- evaluation rows

def zs_cmd(model, quant, out):
    """Part X's command; every model runs python -m (calib-size's registry has llama3.1-8b-instruct, amendment 1)."""
    return [G.py(model), "-m", "evaluation.downstream", "--model", model, "--revision", Z.REV[model], *quant,
            "--tasks", Z.TASKS, "--out", out]


def gsm_cmd(model, quant, out):
    launch = ["-u", HERE / "gsm8k6_date.py", DATE] if model == LLAMA1B else ["-u", "-m"]
    return [G.py(model), *launch, "evaluation.downstream", "--model", model, "--revision", Z.REV[model], *quant,
            "--tasks", Y.TASK, "--apply-chat-template", *THINK.get(model, []), "--batch-size", Y.batch(model, "x"),
            "--gen-lengths", "--samples-out", Path(out).with_suffix(""), "--out", out]


def ppl_cmd(model, quant, out):
    return [G.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", Z.REV[model], *quant,
            "--paper-convention", "--out", out]


def quant_fq(map_path, ppl=False):
    q = [*G.NATIVE, "--kernel-set", "auto", "--weight", "mixfp4", "--act", "fourover6", "--map", map_path]
    return q if ppl else q + ["--act-scope", "row"]


def quant_focus(state):
    return [*G.NATIVE, "--kernel-set", "auto", "--weight", "focus", "--focus", state, "--focus-deploy", "--act",
            "fourover6", "--act-scope", "row"]


def quant_fo6(ppl=False):
    q = [*G.NATIVE, "--kernel-set", "auto", "--weight", "fourover6", "--act", "fourover6"]
    return q if ppl else q + ["--act-scope", "row"]


def quant_zs(kind, artifact):
    """Part X's row_args (run_zeroshot6.row_args) with this row's map or state."""
    if kind == "fq":
        return Z.NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", artifact]
    if kind == "focus":
        return Z.NATIVE + ["--weight", "focus", "--focus", artifact, "--focus-deploy", "--act", "fourover6",
                           "--act-scope", "row"]
    return Z.NATIVE + ["--weight", "fourover6", "--act", "fourover6"]


def check_zs(model, kind, path, artifact):
    """Part X's checks (run_zeroshot6.check_record) with this row's artifact; the digests must equal part X's rows'."""
    digests = {t: json.loads((Z.ZS / "panel" / model / "fo6.json").read_text())["tasks"][t]["sample_digest"]
               for t in Z.SIZES}
    row = {"fq": "fq-16x64", "focus": "focus", "fo6": "fo6"}[kind]
    bad = [b for b in Z.check_record(model, row, path, G.manifests(), dict(digests))
           if b not in ("map path", "FOCUS state path")]
    pol = json.loads(Path(path).read_text()).get("policy") or {}
    got = (pol.get("map") or {}).get("path") if kind == "fq" else (pol.get("focus") or {}).get("state") if kind == "focus" \
        else None
    if kind != "fo6" and got != str(artifact):
        bad.append(f"artifact path {got} != {artifact}")
    return bad


def gsm_digest(model):
    """The model's GSM8K sample digest: part Y's FO6 record, or (Qwen3) this part's BF16 baseline."""
    ref = Y.ROOT / model / "fo6.json"
    if not ref.exists():
        ref = ROOT / model / "eval" / "gsm8k" / "bf16.json"
    return json.loads(ref.read_text())["tasks"][Y.TASK]["sample_digest"]


def y_checks(model, row, path, digest):
    """run_gsm8k6.check_record; for Qwen3 (amendment 1) its chat-option check is replaced by thinking off through
    --enable-thinking false (chat options and the template's args), as Qwen3.8-27B's."""
    bad = Y.check_record(model, row, path, G.manifests(), {"digest": digest})
    if model in QWEN3:
        bad = [b for b in bad if not b.startswith(("chat options", "chat_template_args"))]
        r = json.loads(Path(path).read_text())
        co = r.get("chat_options") or {}
        if not (co.get("apply_chat_template") and co.get("fewshot_as_multiturn")) or co.get("system_instruction") \
                or co.get("enable_thinking") != "false" \
                or (r.get("chat_template_args") or {}).get("enable_thinking") is not False:
            bad.append(f"Qwen3 chat options {co} / {r.get('chat_template_args')}")
    return bad


def check_gsm(model, kind, path, artifact):
    """Part Y's checks (run_gsm8k6.check_record) with this row's artifact; the digest must equal the model's GSM8K
    records' (part Y's; Qwen3: this part's BF16 baseline)."""
    row = {"fq": "fq-16x64", "focus": "focus", "fo6": "fo6"}[kind]
    bad = [b for b in y_checks(model, row, path, gsm_digest(model)) if b not in ("map path", "FOCUS state path")]
    pol = json.loads(Path(path).read_text()).get("policy") or {}
    got = (pol.get("map") or {}).get("path") if kind == "fq" else (pol.get("focus") or {}).get("state") if kind == "focus" \
        else None
    if kind != "fo6" and got != str(artifact):
        bad.append(f"artifact path {got} != {artifact}")
    return bad


def check_ppl(path):
    r = json.loads(Path(path).read_text())
    bad = [] if "resources" in r else ["not complete"]
    for c in ("wiki", "c4"):
        if c not in (r.get("results") or {}):
            bad.append(f"{c} missing")
    return bad


def eval_row(model, name, kind, artifact, which=("zs", "gsm", "ppl"), expect_sha=None):
    """One row: zero-shot (part X), GSM8K (part Y), PPL (paper convention), each from calib-size. expect_sha
    (amendment 2): the FOCUS state is re-read and its sha256 checked before each evaluation that has not run yet."""
    d = ROOT / model / "eval"

    def verify(out):
        if expect_sha and not G.done(out):
            got = file_sha256(artifact)
            if got != expect_sha:
                stop(f"{model} {name}: the state's sha256 {got[:12]} != the recorded {expect_sha[:12]} ({artifact})")
            G.log(f"calib_size {model} {name}: state sha256 verified before {Path(out).parent.name}")
    if "zs" in which:
        out = d / "zs" / f"{name}.json"
        q = quant_zs(kind, artifact)
        verify(out)
        checkout()
        must(G.run(f"cs_zs_{model}_{name}", zs_cmd(model, q, out), out, FQ3, Z.ENV), f"zero-shot {model} {name}")
        bad = check_zs(model, kind, out, artifact)
        if bad:
            stop(f"zero-shot {model} {name}: {bad}")
    if "gsm" in which:
        out = d / "gsm8k" / f"{name}.json"
        q = quant_fq(artifact) if kind == "fq" else quant_focus(artifact) if kind == "focus" else quant_fo6()
        verify(out)
        if not G.done(out):
            checkout()
            out.parent.mkdir(parents=True, exist_ok=True)
        must(M.run_ts(f"cs_gsm8k_{model}_{name}", gsm_cmd(model, q, out), out, FQ3), f"gsm8k {model} {name}")
        bad = check_gsm(model, kind, out, artifact)
        if bad:
            stop(f"gsm8k {model} {name}: {bad}")
    if "ppl" in which:
        out = d / "ppl" / f"{name}.json"
        q = quant_fq(artifact, ppl=True) if kind == "fq" else quant_focus(artifact) if kind == "focus" \
            else quant_fo6(ppl=True)
        verify(out)
        checkout()
        must(G.run(f"cs_ppl_{model}_{name}", ppl_cmd(model, q, out), out, FQ3), f"ppl {model} {name}")
        bad = check_ppl(out)
        if bad:
            stop(f"ppl {model} {name}: {bad}")


# ---------------------------------------------------------------------------------------------- step 0

def same_tiles(a, b):
    sys.path.insert(0, str(FQ3))
    import torch
    from flipquant import maps as FM
    ta, ua, _ = FM.load(a)
    tb, ub, _ = FM.load(b)
    return ua == ub and set(ta) == set(tb) and all(torch.equal(ta[n], tb[n]) for n in ta)


def step0_code():
    head = checkout()
    files = subprocess.run(["git", "-C", FQ3, "diff", "--name-only", "caf69be", head], capture_output=True,
                           text=True).stdout.split()
    want = ["calibration/fit_windows.py", "calibration/tmopt_launch.py", "calibration/train_focus.py",
            "calibration/train_map.py", "tests/test_calib_size.py"]
    ev = subprocess.run(["git", "-C", FQ3, "diff", "--quiet", "caf69be", head, "--", "evaluation", "flipquant", "models",
                         "kernels"]).returncode
    return dict(head=head, files=files, ok=sorted(files) == want and ev == 0)


def step0_calibration():
    """(b): Llama-1B FQ16 at the release settings, GPU teacher and --teacher-host, vs part P's map."""
    model, res = LLAMA1B, {}
    ref = json.loads((G.RUN / "llama_instruct" / model / "records" / "16x64" / "run.json").read_text())
    for host in (False, True):
        out = ROOT / "step0" / f"fq16_256_{'host' if host else 'gpu'}"
        rec = calibrate_fq(model, 256, out, host=host)
        res["host" if host else "gpu"] = dict(
            run_map_sha256=rec["run_map_sha256"], equal_sha=rec["run_map_sha256"] == ref["run_map_sha256"],
            tiles_equal=same_tiles(out / "map.pt", Z.map_path(model, "16x64")),
            teacher_devices=(rec.get("teacher_storage") or {}).get("devices"),
            gpu_peak_allocated_gib=rec["trainer"]["gpu_peak_allocated_gib"], device_peak_mib=rec["device_peak_mib"],
            seconds=rec["job_seconds"])
    res["part_p_run_map_sha256"] = ref["run_map_sha256"]
    res["ok"] = all(v["equal_sha"] and v["tiles_equal"] for k, v in res.items() if k in ("gpu", "host")) \
        and res["host"]["teacher_devices"] == ["cpu"] and res["gpu"]["teacher_devices"] == ["cuda:0"]
    return res


def samples(path):
    return Y.samples(path)


def zs_identity(new, old):
    a, b = json.loads(Path(new).read_text()), json.loads(Path(old).read_text())
    out = {}
    for t in Z.SIZES:
        x, y = a["tasks"][t], b["tasks"][t]
        out[t] = dict(sample_digest=x["sample_digest"] == y["sample_digest"], examples=x["examples"] == y["examples"],
                      loglikelihoods=x["loglikelihoods"] == y["loglikelihoods"], metrics=x["metrics"] == y["metrics"])
    return dict(tasks=out, ok=all(all(v.values()) for v in out.values()))


def gsm_identity(new, old):
    a, b = samples(Path(new).with_suffix(f".{Y.TASK}.jsonl.gz")), samples(Path(old).with_suffix(f".{Y.TASK}.jsonl.gz"))
    keys = sorted(set(a) | set(b))
    diff = [k for k in keys if not (k in a and k in b and a[k] == b[k])]
    return dict(samples=len(keys), identical=len(keys) - len(diff), ok=not diff and len(keys) == 2 * Y.N_DOCS,
                first_difference=None if not diff else dict(doc_id=diff[0][0], filter=diff[0][1]))


def nll(path):
    """Per-window NLL of a flipquant evaluation.ppl record or a main-ppl run_ppl_deploy record (tables_cpu.nll_of)."""
    import tables_cpu as T
    return T.nll_of(path)


def ppl_identity(new, old):
    a, b = nll(new), nll(old)
    return dict(equal={c: a[c] == b[c] for c in a}, windows={c: len(a[c]) for c in a},
                max_abs_diff={c: max(abs(x - y) for x, y in zip(a[c], b[c])) if len(a[c]) == len(b[c]) else None
                              for c in a}, ok=all(a[c] == b[c] for c in a))


def ppl_reference(model, kind):
    """The record the final table uses (tables_final.sources) for this model's FO6 / FQ16 / FOCUS PPL."""
    import tables_cpu as T
    import tables_final as TF
    mp = T.git_json(T.MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    mkey = next(k for key, k, _ in TF.FINAL_MODELS if key == model)
    src = TF.sources(mp, model, mkey)
    return Path(src[{"fo6": "fo6", "fq": "fq_16x64", "focus": "focus"}[kind]])


def step0_focus():
    """(d): Llama-1B FOCUS at 256, part P's command, vs part P's state, tensor by tensor."""
    import torch
    model = LLAMA1B
    rec = calibrate_focus(model, LI.trainer_report(model), ROOT / "step0" / "focus_256", "state")
    new = torch.load(ROOT / "step0" / "focus_256" / "state" / "focus.pt", map_location="cpu", weights_only=False)
    old = torch.load(Z.focus_state(model), map_location="cpu", weights_only=False)
    names = sorted(old["layers"])
    same = set(new["layers"]) == set(old["layers"])
    diff = []
    for n in names:
        for k, v in old["layers"][n].items():
            w = new["layers"].get(n, {}).get(k, "missing")
            same_v = (v is None and w is None) or (isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor)
                                                   and v.dtype == w.dtype and torch.equal(v, w))
            if not same_v:
                diff.append(f"{n}.{k}")
    return dict(job=rec, modules=len(names), same_modules=same, differing_tensors=len(diff), first=diff[:5],
                identical=same and not diff, state_sha256=rec["state_sha256"],
                part_p_state_sha256=Z.sha256_file(Z.focus_state(model)))


def step0():
    (ROOT / "step0").mkdir(parents=True, exist_ok=True)
    out = ROOT / "step0" / "step0.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    G.log("calib_size step 0")
    res["a_code"] = step0_code()
    if not res["a_code"]["ok"]:
        stop(f"step 0 (a): {res['a_code']}")
    res["b_calibration"] = step0_calibration()
    out.write_text(json.dumps(res, indent=1) + "\n")
    if not res["b_calibration"]["ok"]:
        stop("step 0 (b): the calib-size maps differ from part P's (step0/step0.json)")
    # (c) evaluation identity: Llama-1B FQ16 with part P's map
    model, m = LLAMA1B, Z.map_path(LLAMA1B, "16x64")
    eval_row(model, "fq16_256", "fq", m)
    d = ROOT / model / "eval"
    res["c_identity"] = dict(
        zeroshot=zs_identity(d / "zs" / "fq16_256.json", Z.ZS / "panel" / model / "fq-16x64.json"),
        gsm8k=gsm_identity(d / "gsm8k" / "fq16_256.json", Y.ROOT / model / "fq-16x64.json"),
        ppl=ppl_identity(d / "ppl" / "fq16_256.json", G.RUN / "llama_instruct" / model / "ppl" / "fq-16x64.json"))
    res["c_identity"]["ok"] = all(v["ok"] for v in res["c_identity"].values())
    out.write_text(json.dumps(res, indent=1) + "\n")
    if not res["c_identity"]["ok"]:
        stop("step 0 (c): calib-size's evaluation differs from parts X / Y / P (step0/step0.json)")
    # (c2) PPL references on calib-size, both pilot models
    res["c2_ppl_references"] = {}
    for model in PILOT:
        refs = (("fo6", "fo6", None), ("fq16_256", "fq", Z.map_path(model, "16x64")),
                ("focus256", "focus", Z.focus_state(model)))
        res["c2_ppl_references"][model] = {}
        for name, kind, art in refs:
            eval_row(model, name, kind, art, which=("ppl",))
            ref = ppl_reference(model, kind)
            res["c2_ppl_references"][model][name] = dict(
                record=str(ref), identity=ppl_identity(ROOT / model / "eval" / "ppl" / f"{name}.json", ref))
    out.write_text(json.dumps(res, indent=1) + "\n")
    # (d) FOCUS reproducibility
    res["d_focus"] = step0_focus()
    out.write_text(json.dumps(res, indent=1) + "\n")
    if not res["d_focus"]["identical"]:
        G.log("calib_size step 0 (d): the retrained Llama-1B FOCUS-256 state differs from part P's; evaluating it as "
              "the run-to-run reference")
        eval_row(LLAMA1B, "focus256_rerun", "focus", ROOT / "step0" / "focus_256" / "state" / "focus.pt")
    res["ok"] = res["a_code"]["ok"] and res["b_calibration"]["ok"] and res["c_identity"]["ok"]
    out.write_text(json.dumps(res, indent=1) + "\n")
    (ROOT / "step0" / "step0_ok.json").write_text(json.dumps(dict(ok=res["ok"]), indent=1) + "\n")
    G.log(f"calib_size step 0 done: ok={res['ok']}")


# ---------------------------------------------------------------------------------------------- probes

def probes():
    if not json.loads((ROOT / "step0" / "step0_ok.json").read_text()).get("ok"):
        stop("probes: step 0 has not passed")
    d = ROOT / "probes"
    d.mkdir(parents=True, exist_ok=True)
    res = {}
    # FlipQuant Qwen3.8-27B 16x64, 2,048 windows, teacher on the host
    model, out = "qwen3.8-27b", d / "qwen3.8-27b_fq16_2048"
    meta = d / "qwen3.8-27b_fq16_2048.job.json"
    if not meta.exists():
        if out.exists():
            stop(f"{out} exists without a record")
        cmd = train_map_cmd(model, 2048, out, host=True, probe=4)
        rc, seconds, peak = measured("cs_probe_fq16_qwen3.8-27b_2048", cmd, out / "probe.json")
        write_meta(meta, command=" ".join(map(str, cmd[1:])), rc=rc, job_seconds=seconds, device_peak_mib=peak)
    res[model] = dict(json.loads(meta.read_text()), probe=json.loads((out / "probe.json").read_text())
                      if (out / "probe.json").exists() else None)
    # FOCUS Phi-4, 2,048 windows (the pre-drawn set)
    model, out = "phi4-14b", d / "phi4-14b_focus_2048"
    meta = d / "phi4-14b_focus_2048.job.json"
    if not meta.exists():
        if out.exists():
            stop(f"{out} exists without a record")
        cmd = focus_cmd(model, windows_file(model, 2048), 8, out, probe=2)
        rc, seconds, peak = measured("cs_probe_focus_phi4-14b_2048", cmd, out / "report.json")
        write_meta(meta, command=" ".join(map(str, cmd[1:])), rc=rc, job_seconds=seconds, device_peak_mib=peak)
    res[model] = dict(json.loads(meta.read_text()), report=json.loads((out / "report.json").read_text())
                      if (out / "report.json").exists() else None)
    (d / "probes.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    G.log("calib_size probes done")


# ---------------------------------------------------------------------------------------------- pilot

def pilot():
    if not json.loads((ROOT / "step0" / "step0_ok.json").read_text()).get("ok"):
        stop("pilot: step 0 has not passed")
    audit = WIN / "audit.json"
    if not audit.exists() or not json.loads(audit.read_text()).get("pilot_ok"):
        stop("pilot: the 13-gram audit has not passed for the 1,024 sets (windows/audit.json)")
    n = 1024
    for model in PILOT:
        d = ROOT / model
        fq = calibrate_fq(model, n, d / f"fq16_{n}")
        if fq["e0m3_share"] > E0M3_MAX:
            stop(f"pilot {model}: E0M3 share {fq['e0m3_share']:.3f} above {E0M3_MAX} (amendment 1: implausible)")
        run = d / f"fq16_{n}" / "run" / "report.json"
        pre = json.loads(windows_file(model, n).read_text())
        got = json.loads(run.read_text())["fit_extension"]
        if got["records"] != pre["fit_extension"]["records"] or got["skipped_documents"] != pre["fit_extension"][
                "skipped_documents"]:
            stop(f"pilot {model}: the FlipQuant run's fit extension differs from windows/{model}_{n}.json")
        foc = calibrate_focus(model, run, d / f"focus_{n}", "state")
        if foc["token_sha256"] != pre["token_sha256"] or foc["windows"] != n or foc["steps"] != n // 32:
            stop(f"pilot {model}: FOCUS's windows differ from the FlipQuant run's ({foc['windows']} windows, "
                 f"{foc['steps']} steps)")
        G.log(f"calib_size pilot {model}: FQ16 E0M3 share {fq['e0m3_share']:.4f}; FOCUS {foc['steps']} steps, "
              f"micro {foc['micro']}; windows token-list sha256 {token_list_sha(pre['token_sha256'])[:12]}")
        eval_row(model, f"fq16_{n}", "fq", d / f"fq16_{n}" / "map.pt")
        eval_row(model, f"focus_{n}", "focus", d / f"focus_{n}" / "state" / "focus.pt")
    G.log("DONE calib_size pilot")


# ---------------------------------------------------------------------------------------------- amendment 1: the panel

def merge_json(path, key, value):
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = value
    path.write_text(json.dumps(d, indent=1, default=str) + "\n")


def windows_for(model):
    """The model's 1,024 / 2,048 sets (CPU) and the nesting checks (windows/nesting.json, merged)."""
    checkout()
    for n in SIZES[model]:
        out = windows_file(model, n)
        must(cpu_run(f"cs_windows_{model}_{n}", [G.PY, "-m", "calibration.fit_windows", "--model", model,
                                                 "--fit-windows", n, "--data-root", DATA, "--out", out], out, FQ3),
             f"windows {model} {n}")
    res = nesting(model)
    merge_json(WIN / "nesting.json", model, res)
    if not res["ok"]:
        stop(f"windows {model}: nesting / skip checks failed (windows/nesting.json)")


def audit_for(model):
    """The 13-gram audit and its positive control for the model (CPU), unless recorded; returns the model's entry."""
    a, c = WIN / "audit.json", WIN / "audit_control.json"
    have = lambda p: p.exists() and model in (json.loads(p.read_text()).get("models") or {})  # noqa: E731
    for script, path in (("calib_size_audit.py", a), ("calib_size_audit_control.py", c)):
        if not have(path):
            G.log(f"START cs_{script[:-3]}_{model} (CPU)")
            t0 = time.time()
            with open(G.RUN / "logs" / f"cs_{script[:-3]}_{model}.log", "w") as f:
                rc = subprocess.run([G.PY, HERE / script, f"--models={model}"], cwd=HERE,
                                    env=G.env({"CUDA_VISIBLE_DEVICES": ""}), stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL).returncode
            G.log(f"END cs_{script[:-3]}_{model} rc={rc} {time.time() - t0:.0f}s")
            must(rc, f"{script} {model}")
    ctl = json.loads(c.read_text())["models"][model]
    if ctl["plain_windows_with_hits"] or any(d != ctl["picked"] for d in ctl["spliced_detected"].values()):
        stop(f"audit control {model}: {ctl}")
    return json.loads(a.read_text())["models"][model]


def identity_for(model):
    """Amendment 1: the model's FlipQuant 16x64 (256) zero-shot row on calib-size, bit-identical to part X / X2's."""
    rec = ROOT / model / "identity.json"
    if rec.exists() and json.loads(rec.read_text())["ok"]:
        return
    eval_row(model, "fq16_256", "fq", Z.map_path(model, "16x64"), which=("zs",))
    res = zs_identity(ROOT / model / "eval" / "zs" / "fq16_256.json", Z.ZS / "panel" / model / "fq-16x64.json")
    rec.write_text(json.dumps(res, indent=1) + "\n")
    if not res["ok"]:
        stop(f"identity {model}: calib-size's zero-shot differs from part X's record ({rec})")
    G.log(f"calib_size identity {model}: zero-shot FQ16-256 bit-identical to part X / X2")


def ppl_refs_for(model):
    """(c2) for the model: FO6, FQ16-256, FOCUS-256 PPL on calib-size vs the final table's records (reported)."""
    rec = ROOT / model / "ppl_refs.json"
    if rec.exists():
        return
    out = {}
    for name, kind, art in (("fo6", "fo6", None), ("fq16_256", "fq", Z.map_path(model, "16x64")),
                            ("focus256", "focus", Z.focus_state(model))):
        eval_row(model, name, kind, art, which=("ppl",))
        ref = ppl_reference(model, kind)
        out[name] = dict(record=str(ref), identity=ppl_identity(ROOT / model / "eval" / "ppl" / f"{name}.json", ref))
    rec.write_text(json.dumps(out, indent=1) + "\n")
    G.log(f"calib_size ppl references {model}: " + ", ".join(f"{k} {v['identity']['ok']}" for k, v in out.items()))


def qwen3_gsm(model):
    """Amendment 1: the native GSM8K baselines of Qwen3-1.7B / Qwen3-8B (part Y's protocol, thinking off)."""
    import run_zs6_qwen3 as X2                     # X2's registered FOCUS-state and codes-file hashes
    d = ROOT / model / "eval" / "gsm8k"
    for row in QWEN3_GSM_ROWS:
        out = d / f"{row}.json"
        if not G.done(out):
            checkout()
            bad = X2.verify_inputs(model, row)
            if bad:
                stop(f"qwen3 gsm8k {model} {row}: {bad}")
            out.parent.mkdir(parents=True, exist_ok=True)
        q = Y.quant(model, row)
        cmd = [G.py(model), "-u", "-m", "evaluation.downstream", "--model", model, "--revision", Z.REV[model], *q,
               "--tasks", Y.TASK, "--apply-chat-template", *THINK[model], "--batch-size", 64, "--gen-lengths",
               "--samples-out", out.with_suffix(""), "--out", out]
        must(M.run_ts(f"cs_gsm8k_{model}_{row}", cmd, out, FQ3), f"qwen3 gsm8k {model} {row}")
        bad = y_checks(model, row, out, gsm_digest(model))
        if bad:
            stop(f"qwen3 gsm8k {model} {row}: {bad}")
    G.log(f"calib_size qwen3 gsm8k baselines {model}: {len(QWEN3_GSM_ROWS)} rows")


def disk_ok(model):
    """Amendments 1-2: /vault holds a state of the model's 256-window size plus 1 GB, and /home keeps its floor."""
    need = Path(Z.focus_state(model)).stat().st_size + 1e9
    vault, home = shutil.disk_usage(VAULT).free, shutil.disk_usage("/home").free
    return vault >= need and home >= FLOOR + 1e9, dict(vault_free=vault, home_free=home), need


def calibrate_size(model, n):
    d = ROOT / model
    fq = calibrate_fq(model, n, d / f"fq16_{n}", host=model in HOST_TEACHER)
    if fq["e0m3_share"] > E0M3_MAX:
        stop(f"{model} {n}: E0M3 share {fq['e0m3_share']:.3f} above {E0M3_MAX} (amendment 1: implausible)")
    run = d / f"fq16_{n}" / "run" / "report.json"
    pre = json.loads(windows_file(model, n).read_text())
    got = json.loads(run.read_text())["fit_extension"]
    if got["records"] != pre["fit_extension"]["records"] or got["skipped_documents"] != pre["fit_extension"][
            "skipped_documents"]:
        stop(f"{model} {n}: the FlipQuant run's fit extension differs from windows/{model}_{n}.json")
    eval_row(model, f"fq16_{n}", "fq", d / f"fq16_{n}" / "map.pt")
    if model in FOCUS_PENDING:
        G.log(f"NOTE calib_size {model} {n}: FOCUS pending the user's decision on its route; TBD")
        return
    meta_dir = d / f"focus_{n}"
    if (meta_dir / "state.job.json").exists():           # the pilot's (state in RUN) or an earlier panel run's
        foc = json.loads((meta_dir / "state.job.json").read_text())
    else:
        ok, free, need = disk_ok(model)
        if not ok:
            stop(f"PAUSE disk before FOCUS {model} {n}: {free}, the state needs about {need / 1e9:.1f} GB "
                 f"(/home floor {FLOOR / 1e9:.0f} GB)")
        if model in FOCUS_EXTRA:
            offload_test()
        foc = calibrate_focus(model, run, VAULT / model / f"focus_{n}", "state", meta_dir=meta_dir,
                              micro=FOCUS_MICRO.get(model, 8), extra=FOCUS_EXTRA.get(model, ()))
    if foc["token_sha256"] != pre["token_sha256"] or foc["windows"] != n or foc["steps"] != n // 32:
        stop(f"{model} {n}: FOCUS's windows differ from the FlipQuant run's ({foc['windows']} windows, "
             f"{foc['steps']} steps)")
    state = Path(foc.get("state") or meta_dir / "state" / "focus.pt")
    eval_row(model, f"focus_{n}", "focus", state, expect_sha=foc["state_sha256"])
    G.log(f"calib_size {model} {n}: FQ16 E0M3 share {fq['e0m3_share']:.4f}; FOCUS {foc['steps']} steps; rows done")


def offload_test():
    """Amendment 2: tests/test_focus_offload.py on the GPU, on calib-size, before the first offloaded FOCUS run."""
    out = ROOT / "offload_test.json"
    if out.exists() and json.loads(out.read_text())["ok"]:
        return
    head = checkout()
    G.idle()
    G.log("START cs_offload_test")
    t0 = time.time()
    r = subprocess.run([G.PY_FAST, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_focus_offload.py"],
                       cwd=FQ3, env=G.env(), capture_output=True, text=True)
    G.log(f"END cs_offload_test rc={r.returncode} {time.time() - t0:.0f}s")
    res = dict(flipquant=head, rc=r.returncode, output=(r.stdout + r.stderr)[-2000:],
               ok=r.returncode == 0 and "1 passed" in r.stdout)
    out.write_text(json.dumps(res, indent=1) + "\n")
    if not res["ok"]:
        stop(f"offload test: rc={r.returncode} ({out})")


def control_qwen27b():
    """Amendment 3: Qwen3.8-27B FOCUS-256 retrained locally exactly as its 1,024 run (FOCUS_COMMON, --offload --micro
    1, device_map cuda, the release 256 windows), the environment-matched reference; then its zero-shot, GSM8K and
    PPL (row focus256_local). The state goes to /vault with the sha256 checks of amendment 2."""
    model = "qwen3.8-27b"
    meta_dir = ROOT / model / "focus_256_local"
    if (meta_dir / "state.job.json").exists():
        foc = json.loads((meta_dir / "state.job.json").read_text())
    else:
        ok, free, need = disk_ok(model)
        if not ok:
            stop(f"PAUSE disk before the control: {free}, the state needs about {need / 1e9:.1f} GB")
        offload_test()
        foc = calibrate_focus(model, S.trainer_report(model), VAULT / model / "focus_256_local", "state",
                              meta_dir=meta_dir, micro=FOCUS_MICRO[model], extra=FOCUS_EXTRA[model])
    cluster = json.loads((G.RUN / "focus_qwen27b" / "report.json").read_text())["data"]["records"]
    if foc["windows"] != 256 or foc["steps"] != 8 or foc["token_sha256"] != cluster:
        stop(f"control {model}: {foc['windows']} windows, {foc['steps']} steps; windows equal to the cluster "
             f"state's: {foc['token_sha256'] == cluster}")
    eval_row(model, "focus256_local", "focus", Path(foc["state"]), expect_sha=foc["state_sha256"])
    G.log(f"calib_size control {model}: FOCUS-256 local (amendment 3) trained and evaluated")


def wait_checkout(timeout=6 * 3600):
    """Amendment 2: the panel waits until calib_size/wt is fast-forwarded to CODE after the pilot."""
    t0, noted = time.time(), False
    while True:
        r = subprocess.run(["git", "-C", str(FQ3), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(FQ3), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        if r.startswith(CODE) and not dirty:
            return r
        if not noted:
            G.log(f"NOTE calib_size panel: waiting for {FQ3} at {CODE} (now {r[:7]}; amendment 2's fast-forward)")
            noted = True
        if time.time() - t0 > timeout:
            stop(f"panel: {FQ3} not at {CODE} after {timeout / 3600:.0f} h")
        time.sleep(30)


def panel(n, models):
    if not json.loads((ROOT / "step0" / "step0_ok.json").read_text()).get("ok"):
        stop("panel: step 0 has not passed")
    wait_checkout()
    if n == 2048:                                    # amendment 3: the control runs before the 2,048 panel
        control_qwen27b()
    for model in models:
        windows_for(model)
        a = audit_for(model)
        if not a.get(f"ok_{n}", not any(h["membership"] == str(n) for h in a["hits"])):     # the first run's entries
            stop(f"audit {model}: a hit among the {n} set's new windows (windows/audit.json)")
        identity_for(model)
        ppl_refs_for(model)
        if model in QWEN3:
            qwen3_gsm(model)
        calibrate_size(model, n)
        G.log(f"calib_size panel {n}: {model} done")
    G.log(f"DONE calib_size panel {n}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "windows":
        windows()
    elif what == "step0":
        step0()
    elif what == "probes":
        probes()
    elif what == "pilot":
        pilot()
    elif what == "panel":
        size = int(next(a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--size=")))
        sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
        panel(size, [m for m in ORDER if not sel or m in sel[0]])
    else:
        raise SystemExit(f"unknown part {what}")
