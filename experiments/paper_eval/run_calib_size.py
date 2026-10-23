"""Part AA (PROTOCOL.md): calibration size, 1,024 / 2,048 fit windows for FlipQuant 16x64 and FOCUS, on flipquant
calib-size f912a30 (worktree calib_size/wt).

    python run_calib_size.py windows     # CPU: the 1,024 / 2,048 sets (calibration.fit_windows), the nesting checks
    python run_calib_size.py step0       # (b) calibration identity, (c) evaluation identity, (c2) PPL references,
                                         # (d) FOCUS reproducibility
    python run_calib_size.py probes      # 2,048-window memory probes: FlipQuant Qwen3.8-27B, FOCUS Phi-4
    python run_calib_size.py pilot       # Llama-3.2-1B-Instruct and Phi-4 at 1,024: FQ16 + FOCUS, zero-shot / GSM8K / PPL
    python run_calib_size.py panel --size=1024 [--models=a,b]    # amendment 1: the full panel at one size, ORDER
    python run_calib_size.py gptq_step0     # amendment 5: GPTQ‡ arguments; Llama-1B GPTQ‡-256 codes and rows vs P / X / Y
    python run_calib_size.py gptq_measure   # amendment 5's ETA gate: Llama-1B GPTQ‡-1k's calibration, then the ETA
    python run_calib_size.py gptq_panel     # amendment 5: GPTQ‡ at 1,024 for the eight models, then 2,048, ORDER

The audit is calib_size_audit.py (CPU). Records under RUN/calib_size/; START / END in RUN/queue.log; one GPU job at a
time on an idle GPU (run_gpu). A finished record is not run again; a failed job or check stops the part (STOP).
"""
import hashlib
import json
import math
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
#: after calib-size's fast-forward (amendment 2); amendment 5's GPTQ‡ commands set it to GPTQ_CODE
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
E0M3_MAX = 0.40                 # amendment 4 (was 0.25, amendment 1): a larger E0M3 share stops the run


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

def train_map_cmd(model, n, out, host=False, probe=None, unit="16x64"):
    """TRAIN at `unit` (amendment 6: 8x64 / 256x64 with the same release settings)."""
    cmd = [G.PY, "-m", "calibration.train_map", "--model", model, "--unit", unit, *TRAIN[2:], "--fit-windows", n,
           "--data-root", DATA]
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


UNIT_TAG = {"16x64": "16", "8x64": "8", "256x64": "256"}


def calibrate_fq(model, n, out, host=False, unit="16x64"):
    """FlipQuant at `unit` (16x64 unless amendment 6) and n windows; out/job.json records the command, the sampler and
    the cost."""
    meta = out.parent / f"{out.name}.job.json"
    if (out / "reproduction.json").exists() and meta.exists():
        return json.loads(meta.read_text())
    if out.exists():
        stop(f"{out} exists without a record; inspect it before rerunning")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = train_map_cmd(model, n, out, host=host, unit=unit)
    name = f"cs_fq{UNIT_TAG[unit]}_{model}_{n}{'_host' if host else ''}"
    rc, seconds, peak = measured(name, cmd, out / "reproduction.json")
    if rc != 0:
        log = (G.RUN / "logs" / f"{name}.log").read_text(errors="replace") if (G.RUN / "logs" / f"{name}.log").exists() \
            else ""
        if "OutOfMemoryError" in log or "CUDA out of memory" in log:
            stop(f"train_map {model} {unit} {n}: CUDA out of memory (stop; no setting is changed)")
    must(rc, f"train_map {model} {unit} {n}")
    rep = json.loads((out / "reproduction.json").read_text())
    report = json.loads((out / "run" / "report.json").read_text())
    res = report["resources"]
    rec = dict(model=model, unit=unit, windows=n, teacher_host=host, command=" ".join(map(str, cmd[1:])), checkout=checkout(),
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
    if kind in FQ_KINDS:
        return Z.NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", artifact]
    if kind == "focus":
        return Z.NATIVE + ["--weight", "focus", "--focus", artifact, "--focus-deploy", "--act", "fourover6",
                           "--act-scope", "row"]
    return Z.NATIVE + ["--weight", "fourover6", "--act", "fourover6"]


ROW = {"fq": "fq-16x64", "focus": "focus", "fo6": "fo6", "gptq": "gptq", "fq8": "fq-8x64", "fq256": "fq-256x64"}
FQ_KINDS = ("fq", "fq8", "fq256")          # amendment 6: the 8x64 / 256x64 maps (their kernel families via part X's rows)


def artifact_checks(model, kind, path, artifact, codes=None):
    """The record's policy names this row's artifact: the map, the FOCUS state, or (amendment 5) GPTQ‡'s codes file,
    read from the cache (loaded) with the calibration's codes digest. The last replaces parts X / Y's check of the
    256-window codes digest."""
    pol = json.loads(Path(path).read_text()).get("policy") or {}
    if kind == "gptq":
        p = pol.get("ptq") or {}
        bad = [] if p.get("loaded") is True else [f"codes not loaded from the cache ({p.get('loaded')})"]
        if p.get("cache") != str(artifact):
            bad.append(f"codes file {p.get('cache')} != {artifact}")
        if not codes or p.get("codes_sha256") != codes:
            bad.append(f"codes digest {str(p.get('codes_sha256'))[:12]} != the calibration's {str(codes)[:12]}")
        return bad
    got = (pol.get("map") or {}).get("path") if kind in FQ_KINDS else (pol.get("focus") or {}).get("state") \
        if kind == "focus" else None
    return [f"artifact path {got} != {artifact}"] if kind != "fo6" and got != str(artifact) else []


def ignored(kind, msg):
    """Parts X / Y's checks that name their own artifact: the map path, the FOCUS state path, the 256 codes digest."""
    return msg in ("map path", "FOCUS state path") or (kind == "gptq" and msg.startswith("codes_sha256 "))


def check_zs(model, kind, path, artifact, codes=None):
    """Part X's checks (run_zeroshot6.check_record) with this row's artifact; the digests must equal part X's rows'."""
    digests = {t: json.loads((Z.ZS / "panel" / model / "fo6.json").read_text())["tasks"][t]["sample_digest"]
               for t in Z.SIZES}
    bad = [b for b in Z.check_record(model, ROW[kind], path, G.manifests(), dict(digests)) if not ignored(kind, b)]
    return bad + artifact_checks(model, kind, path, artifact, codes)


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


def check_gsm(model, kind, path, artifact, codes=None):
    """Part Y's checks (run_gsm8k6.check_record) with this row's artifact; the digest must equal the model's GSM8K
    records' (part Y's; Qwen3: this part's BF16 baseline)."""
    bad = [b for b in y_checks(model, ROW[kind], path, gsm_digest(model)) if not ignored(kind, b)]
    return bad + artifact_checks(model, kind, path, artifact, codes)


def check_ppl(path):
    r = json.loads(Path(path).read_text())
    bad = [] if "resources" in r else ["not complete"]
    for c in ("wiki", "c4"):
        if c not in (r.get("results") or {}):
            bad.append(f"{c} missing")
    return bad


def eval_row(model, name, kind, artifact, which=("zs", "gsm", "ppl"), expect_sha=None, codes=None):
    """One row: zero-shot (part X), GSM8K (part Y), PPL (paper convention), each from calib-size. expect_sha
    (amendment 2): the FOCUS state is re-read and its sha256 checked before each evaluation that has not run yet.
    Amendment 5: kind "gptq" takes artifact = (windows, codes file) and the calibration's codes digest; its PPL is its
    calibration job (calibrate_gptq), not run here."""
    d = ROOT / model / "eval"
    path = artifact[1] if kind == "gptq" else artifact
    what = "codes file" if kind == "gptq" else "state"

    def verify(out):
        if expect_sha and not G.done(out):
            got = file_sha256(path)
            if got != expect_sha:
                stop(f"{model} {name}: the {what}'s sha256 {got[:12]} != the recorded {expect_sha[:12]} ({path})")
            G.log(f"calib_size {model} {name}: {what} sha256 verified before {Path(out).parent.name}")
    if "zs" in which:
        out = d / "zs" / f"{name}.json"
        q = gptq_zs_quant(model, *artifact) if kind == "gptq" else quant_zs(kind, artifact)
        verify(out)
        checkout()
        must(G.run(f"cs_zs_{model}_{name}", zs_cmd(model, q, out), out, FQ3, Z.ENV), f"zero-shot {model} {name}")
        bad = check_zs(model, kind, out, path, codes)
        if bad:
            stop(f"zero-shot {model} {name}: {bad}")
    if "gsm" in which:
        out = d / "gsm8k" / f"{name}.json"
        q = gptq_gsm_quant(model, *artifact) if kind == "gptq" else quant_fq(artifact) if kind in FQ_KINDS \
            else quant_focus(artifact) if kind == "focus" else quant_fo6()
        verify(out)
        if not G.done(out):
            checkout()
            out.parent.mkdir(parents=True, exist_ok=True)
        must(M.run_ts(f"cs_gsm8k_{model}_{name}", gsm_cmd(model, q, out), out, FQ3), f"gsm8k {model} {name}")
        bad = check_gsm(model, kind, out, path, codes)
        if bad:
            stop(f"gsm8k {model} {name}: {bad}")
    if "ppl" in which and kind == "gptq":
        stop(f"{model} {name}: GPTQ‡'s PPL is its calibration job (amendment 5), not an eval_row run")
    if "ppl" in which:
        out = d / "ppl" / f"{name}.json"
        q = quant_fq(artifact, ppl=True) if kind in FQ_KINDS else quant_focus(artifact) if kind == "focus" \
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


# ---------------------------------------------------------------------------------------------- amendment 4: theta

THETA_CODE = "9f19863"             # calib-size + tmopt_launch --save-theta (fast-forwarded after the 2,048 panel)


def theta_rerun(model=LLAMA1B, n=2048):
    """Amendment 4: the model's FQ16-n calibration re-run once with the passive --save-theta hook, from the exact
    command the panel ran (its train.log's first line; only --out and the hook differ). The rerun's map must equal the
    panel's (the trainer's map sha256); then, from theta: the share of E0M3 tiles with theta < 0.02 (one step of lr
    0.02 above the threshold), and the share of all tiles with |theta| < 0.02."""
    rec_path = ROOT / model / f"theta_{n}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    def git(*a):
        return subprocess.run(["git", "-C", str(FQ3), *a], capture_output=True, text=True).stdout.strip()
    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if not head.startswith(THETA_CODE) or dirty:
        stop(f"theta: {FQ3} at {head[:7]} (dirty {bool(dirty)}), not {THETA_CODE}")
    panel = ROOT / model / f"fq16_{n}"
    cmd = (panel / "train.log").read_text().splitlines()[0].split()
    out = ROOT / model / f"theta_{n}"
    if out.exists():
        stop(f"theta: {out} exists from an earlier attempt; inspect it")
    i = cmd.index("--out")
    cmd[i + 1] = str(out / "run")
    j = cmd.index("--")
    cmd[j:j] = ["--save-theta", str(out / "theta.pt")]
    if not cmd[1].endswith("calibration/tmopt_launch.py"):
        stop(f"theta: unexpected trainer command {cmd[:3]}")
    sys.path.insert(0, str(FQ3))
    from calibration import tmopt_common as RC
    out.mkdir(parents=True)
    G.idle()
    G.log(f"START cs_theta_{model}_{n}")
    t0 = time.time()
    with open(G.RUN / "logs" / f"cs_theta_{model}_{n}.log", "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=RC.ENGINE, env=RC.env(), stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    G.log(f"END cs_theta_{model}_{n} rc={rc} {time.time() - t0:.0f}s")
    must(rc, f"theta {model} {n}")
    import torch
    a = json.loads((panel / "run" / "report.json").read_text())["map_sha256"]
    b = json.loads((out / "run" / "report.json").read_text())["map_sha256"]
    theta = torch.load(out / "theta.pt", map_location="cpu", weights_only=True)
    hard = torch.load(panel / "run" / "map.pt", map_location="cpu", weights_only=True)
    same = a == b and all(torch.equal(theta[k] > 0, hard[k].bool()) for k in hard)
    t = torch.cat([v.flatten().float() for v in theta.values()])
    e = t[t > 0]
    res = dict(model=model, windows=n, flipquant=head, map_sha256_panel=a, map_sha256_rerun=b, reproduces=same,
               tiles=t.numel(), e0m3_tiles=e.numel(), e0m3_share=e.numel() / t.numel(),
               e0m3_within_one_step=float((e < 0.02).float().mean()),
               all_within_one_step=float((t.abs() < 0.02).float().mean()),
               e2m1_within_one_step=float(((t <= 0) & (t > -0.02)).float().sum() / (t <= 0).sum()),
               quantiles_e0m3=[float(x) for x in torch.quantile(e, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))],
               seconds=time.time() - t0)
    rec_path.write_text(json.dumps(res, indent=1) + "\n")
    if not same:
        stop(f"theta: the rerun's map differs from the panel's ({rec_path})")
    G.log(f"calib_size theta {model} {n}: map reproduced; E0M3 tiles with theta < 0.02: "
          f"{res['e0m3_within_one_step']:.1%}")
    return res



# ---------------------------------------------------------------------------------------------- amendment 5: GPTQ‡

#: calib-size 8dbe2e4: 9f19863's code (9f19863..8dbe2e4 adds results/calib_size and one .gitignore line)
GPTQ_CODE = "8dbe2e4"
GPTQ_FILE = "codes_nvfp4_act-fo6.pt"
GPTQ_Q = ["--weight", "nvfp4", "--act", "fourover6", "--any-act"]
GPTQ_STEP0 = ROOT / "step0" / "gptq_256"
GPTQ_MAX_GIB = 91.0            # a 2,048 calibration whose extrapolated peak allocation is above this does not start
#: Llama-1B GPTQ‡-256: part P's codes digest (its record; also part X step 0's and the HF GPTQ-BF16Calib upload's) and
#: the uploaded codes file's sha256 (results/paper_eval/hf_gptq_focus)
P_CODES = "952850d7d8cf4d3d61b0eee6672151a6a2736f3fa6d49264b8dda09417f15e2c"
HF_FILE = "5f65382fe016aec8736c520051a32ee165ee80ee6572281b946055b78c3a4722"
#: the cache key's fixed fields (flipquant.gptq.cache_key with propagate bf16: the calibration runs with act "none")
GPTQ_KEY = dict(weight="nvfp4", act="none", unit=None, mode="fake", map_sha256=None, rotate="none", damp=0.01, block=128,
                calib="release", rules="modelopt-0.47", hessian_input="bf16", propagate="bf16")


def gptq_flags(model, extension, cache):
    """run_sm120.gptq_flags (part J; part P's flags for the Llama-Instruct models) with this row's windows and codes."""
    return ["--ptq", "gptq", "--gptq-propagate", "bf16", "--gptq-hessian-input", "bf16", "--gptq-calib", "release",
            "--gptq-calib-record", record(model), "--gptq-calib-extension", extension, "--gptq-block", 128,
            "--gptq-damp", 0.01, "--gptq-batch", 1 if model == "qwen3.8-27b" else 8, "--gptq-cache", cache]


def gptq_ppl_cmd(model, extension, cache, out):
    """The main table's GPTQ‡ job (run_sm120.gptq, run_llama_instruct.gptq): evaluation.ppl, native build_V, the
    nvfp4-fo6 row's flags; GPTQ writes the codes to `cache`, then the row's PPL is measured."""
    return [G.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", Z.REV[model], *G.NATIVE,
            "--kernel-set", "auto", *S.GPTQ_ROWS["nvfp4-fo6"][0], *gptq_flags(model, extension, cache), "--out", out]


def gptq_zs_quant(model, extension, cache):
    """run_zeroshot6.row_args(model, "gptq") with this row's windows and codes."""
    return Z.NATIVE + GPTQ_Q + gptq_flags(model, extension, cache)


def gptq_gsm_quant(model, extension, cache):
    """run_gsm8k6.quant(model, "gptq") with this row's windows and codes."""
    return [*G.NATIVE, "--kernel-set", "auto", *GPTQ_Q, "--act-scope", "row", *gptq_flags(model, extension, cache)]


def strs(cmd):
    return [str(a) for a in cmd]


def gptq_args_check():
    """Step 0: this driver's GPTQ‡ arguments, with the main table's windows and codes files, against what the main
    table ran: the PPL job's command as logged (the first line of parts J / P's job logs; part P ran without a cache),
    and parts X / Y's row arguments."""
    res = {}
    for model in ORDER:
        ext, codes = Z.calib_extension(model), Z.codes_file(model)
        log = G.RUN / "logs" / (f"llama_gptq_{model}.log" if model in Z.LLAMA else f"gptq_{model}_nvfp4-fo6.log")
        ran = log.read_text(errors="replace").splitlines()[0].split()
        mine = strs(gptq_ppl_cmd(model, ext, codes, ran[ran.index("--out") + 1]))
        if model in Z.LLAMA:
            i = mine.index("--gptq-cache")
            del mine[i:i + 2]
        r = dict(log=log.name, ppl_command=mine == ran,
                 zeroshot_args=strs(gptq_zs_quant(model, ext, codes)) == strs(Z.row_args(model, "gptq")),
                 gsm8k_args=strs(gptq_gsm_quant(model, ext, codes)) == strs(Y.quant(model, "gptq")),
                 record=str(record(model)) == str(Z.calib_record(model)))
        r["ok"] = all(r[k] for k in ("ppl_command", "zeroshot_args", "gsm8k_args", "record"))
        if not r["ppl_command"]:
            r["difference"] = [(a, b) for a, b in zip(mine, ran) if a != b][:4] + [len(mine), len(ran)]
        res[model] = r
    res["ok"] = all(v["ok"] for v in res.values())
    return res


def gptq_paths(model, n):
    """(windows, codes file, PPL record, job record) of a GPTQ‡ row: step 0's at 256 (Llama-1B, part P's windows), the
    panel's at 1,024 / 2,048 (codes on /vault)."""
    if n == 256:
        assert model == LLAMA1B
        return (Z.calib_extension(model), GPTQ_STEP0 / GPTQ_FILE, ROOT / model / "eval" / "ppl" / "gptq256.json",
                ROOT / "step0" / "gptq_256.job.json")
    return (windows_file(model, n), VAULT / model / f"gptq_{n}" / GPTQ_FILE,
            ROOT / model / "eval" / "ppl" / f"gptq_{n}.json", ROOT / model / f"gptq_{n}.job.json")


def gptq_calibration_checks(model, n, rec, extension):
    """The calibration record: computed (not loaded), the main table's key at n windows, this row's windows file, the
    digests present, both PPLs finite."""
    bad = [] if rec["loaded"] is False else [f"loaded {rec['loaded']}"]
    bad += [f"key {f} {rec['key'].get(f)!r} != {v!r}" for f, v in dict(GPTQ_KEY, windows=n).items()
            if rec["key"].get(f) != v]
    cal = rec.get("calibration") or {}
    if cal.get("trainer_report") != str(extension) or cal.get("record") != str(record(model)) or cal.get("windows") != n:
        bad.append(f"calibration windows {cal}")
    if not rec.get("codes_sha256") or not rec["key"].get("data_sha256"):
        bad.append("codes / data digest missing")
    if any(v is None or not math.isfinite(v) for v in rec["ppl"].values()):
        bad.append(f"ppl {rec['ppl']}")
    return bad


def calibrate_gptq(model, n):
    """GPTQ‡ at n windows inside its PPL job (amendment 5). The job record: the command, the release sampler's device
    peak, the cost, GPTQ's key and codes digest, the PPL, and the codes file's sha256 from a fresh read after the job.
    An OOM stops the run (no setting is changed)."""
    ext, cache, out, meta = gptq_paths(model, n)
    if meta.exists():
        return json.loads(meta.read_text())
    if G.done(out) or Path(cache).exists():
        stop(f"gptq {model} {n}: {out if G.done(out) else cache} exists without a job record; inspect it")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = gptq_ppl_cmd(model, ext, cache, out)
    name = f"cs_gptq_{model}_{n}"
    rc, seconds, peak = measured(name, cmd, out, fresh_dir=False)
    if rc != 0:
        log = (G.RUN / "logs" / f"{name}.log").read_text(errors="replace")
        oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
        stop(f"gptq {model} {n}: {'CUDA out of memory (amendment 5: stop; no setting is changed)' if oom else f'rc={rc}'}")
    r = json.loads(out.read_text())
    p = r.get("ptq") or {}
    rec = dict(model=model, windows=n, command=" ".join(map(str, cmd[1:])), checkout=checkout(), rc=rc,
               job_seconds=seconds, device_peak_mib=peak, gptq_seconds=p.get("seconds"), loaded=p.get("loaded"),
               key={k: p.get(k) for k in (*GPTQ_KEY, "model", "windows", "data_sha256")},
               codes_sha256=p.get("codes_sha256"), mean_output_error_vs_rtn=p.get("mean_output_error_vs_rtn"),
               mean_codes_changed=p.get("mean_codes_changed"), calibration=r.get("gptq_calibration"),
               gpu_peak_gib=(r.get("resources") or {}).get("gpu_peak_gib"), process_seconds=(r.get("resources") or {})
               .get("seconds"), ppl={c: ((r.get("results") or {}).get(c) or {}).get("ppl") for c in ("wiki", "c4")},
               cache=str(cache), cache_bytes=Path(cache).stat().st_size, cache_sha256=file_sha256(cache))
    write_meta(meta, **rec)
    bad = gptq_calibration_checks(model, n, rec, ext) + check_ppl(out)
    if bad:
        stop(f"gptq {model} {n}: {bad} ({meta})")
    G.log(f"calib_size gptq {model} {n}: GPTQ {rec['gptq_seconds']:.0f}s, job {seconds:.0f}s, device peak {peak} MiB, "
          f"codes {rec['codes_sha256'][:12]}, file sha256 {rec['cache_sha256'][:12]}")
    return rec


def zs_average(r):
    return 100 * sum(r["tasks"][t]["metrics"][Z.PRIMARY[t] + ",none"] for t in Z.SIZES) / len(Z.SIZES)


def gsm_strict(r):
    return 100 * r["tasks"][Y.TASK]["metrics"]["exact_match,strict_match"]


def gptq_reference(model):
    """GPTQ‡-256's records: zero-shot (part X / X2), GSM8K (part Y; Qwen3: this part's native baseline), PPL (the main
    table's job). Step 0 shows calib-size reproduces them."""
    gsm = (ROOT / model / "eval" / "gsm8k" if model in QWEN3 else Y.ROOT / model) / "gptq.json"
    return (json.loads((Z.ZS / "panel" / model / "gptq.json").read_text()), json.loads(gsm.read_text()),
            json.loads(Z.gptq_record(model).read_text()))


def gptq_plausible(model, rec=None, zs=None, gsm=None):
    """Amendment 5's implausible results: GPTQ's calibration output error not below RTN's; a PPL more than 5 % above
    GPTQ‡-256's; a zero-shot average more than 3 points, or a GSM8K strict score more than 10 points, below it."""
    z0, g0, p0 = gptq_reference(model)
    bad = []
    if rec is not None:
        if not rec["mean_output_error_vs_rtn"] < 1:
            bad.append(f"calibration output error {rec['mean_output_error_vs_rtn']} x RTN")
        for c in ("wiki", "c4"):
            ref = p0["results"][c]["ppl"]
            if not (math.isfinite(rec["ppl"][c]) and rec["ppl"][c] <= 1.05 * ref):
                bad.append(f"{c} PPL {rec['ppl'][c]:.4f} vs GPTQ‡-256 {ref:.4f}")
    if zs is not None and zs_average(zs) < zs_average(z0) - 3:
        bad.append(f"zero-shot {zs_average(zs):.2f} vs GPTQ‡-256 {zs_average(z0):.2f}")
    if gsm is not None and gsm_strict(gsm) < gsm_strict(g0) - 10:
        bad.append(f"GSM8K {gsm_strict(gsm):.2f} vs GPTQ‡-256 {gsm_strict(g0):.2f}")
    return bad


def gptq_windows_ok(model, n):
    """The windows file against part AA's committed record: its token-list sha256 equals calib_size.json's (full) and
    WINDOWS.md's (prefix, in the row of this size), and the model's nesting checks pass."""
    got = token_list_sha(json.loads(windows_file(model, n).read_text())["token_sha256"])
    res = G.RZ / "results" / "paper_eval" / "calib_size"
    nest = json.loads((res / "calib_size.json").read_text())["windows"]["nesting"][model]
    rows = [ln for ln in (res / "WINDOWS.md").read_text().splitlines() if f"`{got[:16]}…`" in ln and f"| {n:,} |" in ln]
    bad = [] if nest[str(n)]["token_list_sha256"] == got else [f"token list {got[:12]} != calib_size.json's"]
    bad += [] if rows else ["token list not in WINDOWS.md at this size"]
    bad += [] if nest["ok"] and json.loads((WIN / "nesting.json").read_text())[model]["ok"] else ["nesting checks"]
    return bad


def gptq_disk_ok(model):
    """/vault holds a codes file of the model's 256-window size plus 1 GB; /home keeps its floor."""
    need = Path(Z.codes_file(model)).stat().st_size + 1e9
    vault, home = shutil.disk_usage(VAULT).free, shutil.disk_usage("/home").free
    return vault >= need and home >= FLOOR + 1e9, dict(vault_free=vault, home_free=home), need


def gptq_memory(model):
    """The 2,048 calibration's peak allocation (torch's, the records' gpu_peak_gib), extrapolated linearly in the window
    count from the main table's 256 job and this part's 1,024 job."""
    a256 = json.loads(Z.gptq_record(model).read_text())["resources"]["gpu_peak_gib"][0]
    a1k = json.loads((ROOT / model / "gptq_1024.job.json").read_text())["gpu_peak_gib"][0]
    return dict(gib_256=a256, gib_1024=a1k, predicted_2048=a1k + (a1k - a256) * (2048 - 1024) / (1024 - 256))


def gptq_calibrate_size(model, n):
    """The pre-run checks (windows, audit, disk; at 2,048 the memory gate), then the calibration and its gates."""
    ext, cache, out, meta = gptq_paths(model, n)
    if not meta.exists():
        bad = gptq_windows_ok(model, n)
        if bad:
            stop(f"gptq {model} {n}: windows {bad}")
        a = json.loads((WIN / "audit.json").read_text())["models"][model]
        if not a.get(f"ok_{n}", not any(h["membership"] == str(n) for h in a["hits"])):
            stop(f"gptq {model} {n}: the audit has a hit among the {n} set's windows")
        ok, free, need = gptq_disk_ok(model)
        if not ok:
            stop(f"PAUSE disk before gptq {model} {n}: {free}, the codes need about {need / 1e9:.1f} GB")
        if n == 2048:
            mem = gptq_memory(model)
            merge_json(ROOT / "gptq_memory.json", model, mem)
            G.log(f"calib_size gptq {model} 2048: predicted peak allocation {mem['predicted_2048']:.1f} GiB "
                  f"(256: {mem['gib_256']:.1f}, 1,024: {mem['gib_1024']:.1f})")
            if mem["predicted_2048"] > GPTQ_MAX_GIB:
                stop(f"gptq {model} 2048: predicted peak allocation {mem['predicted_2048']:.1f} GiB > {GPTQ_MAX_GIB} "
                     f"(amendment 5: stop before the calibration)")
    rec = calibrate_gptq(model, n)
    bad = gptq_plausible(model, rec=rec)
    if bad:
        stop(f"gptq {model} {n}: implausible {bad}")
    return rec


def gptq_size(model, n):
    rec = gptq_calibrate_size(model, n)
    ext, cache, _, _ = gptq_paths(model, n)
    eval_row(model, f"gptq_{n}", "gptq", (ext, cache), which=("zs", "gsm"), expect_sha=rec["cache_sha256"],
             codes=rec["codes_sha256"])
    d = ROOT / model / "eval"
    zs, gsm = (json.loads((d / k / f"gptq_{n}.json").read_text()) for k in ("zs", "gsm8k"))
    bad = gptq_plausible(model, zs=zs, gsm=gsm)
    if bad:
        stop(f"gptq {model} {n}: implausible {bad}")
    G.log(f"calib_size gptq {model} {n}: zero-shot {zs_average(zs):.2f}, GSM8K {gsm_strict(gsm):.2f}, wiki "
          f"{rec['ppl']['wiki']:.4f}, c4 {rec['ppl']['c4']:.4f}; row done")


def gptq_step0():
    """Amendment 5, step 0 (RUN/calib_size/step0/gptq_step0.json): the arguments; Llama-1B GPTQ‡-256 on calib-size with
    part P's command: codes equal to part P's (required), the codes file vs the uploaded one (reported), PPL / zero-shot
    / GSM8K identical to parts P / X / Y."""
    out = ROOT / "step0" / "gptq_step0.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    G.log("calib_size gptq step 0")
    res["args"] = gptq_args_check()
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if not res["args"]["ok"]:
        stop(f"gptq step 0: the arguments differ from the main table's ({out})")
    model = LLAMA1B
    GPTQ_STEP0.mkdir(parents=True, exist_ok=True)
    rec = calibrate_gptq(model, 256)
    res["calibration"] = dict(codes_sha256=rec["codes_sha256"], part_p_codes_sha256=P_CODES,
                              codes_equal=rec["codes_sha256"] == P_CODES, file_sha256=rec["cache_sha256"],
                              uploaded_file_sha256=HF_FILE, file_equal_uploaded=rec["cache_sha256"] == HF_FILE,
                              data_sha256=rec["key"]["data_sha256"], gptq_seconds=rec["gptq_seconds"],
                              job_seconds=rec["job_seconds"], device_peak_mib=rec["device_peak_mib"],
                              gpu_peak_gib=rec["gpu_peak_gib"])
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if not res["calibration"]["codes_equal"]:
        stop(f"gptq step 0: Llama-1B GPTQ‡-256's codes {rec['codes_sha256'][:12]} != part P's {P_CODES[:12]} ({out})")
    ext, cache, ppl_out, _ = gptq_paths(model, 256)
    eval_row(model, "gptq256", "gptq", (ext, cache), which=("zs", "gsm"), expect_sha=rec["cache_sha256"],
             codes=rec["codes_sha256"])
    d = ROOT / model / "eval"
    res["identity"] = dict(ppl=ppl_identity(ppl_out, Z.gptq_record(model)),
                           zeroshot=zs_identity(d / "zs" / "gptq256.json", Z.ZS / "panel" / model / "gptq.json"),
                           gsm8k=gsm_identity(d / "gsm8k" / "gptq256.json", Y.ROOT / model / "gptq.json"))
    res["identity"]["ok"] = all(v["ok"] for v in res["identity"].values())
    res["ok"] = res["args"]["ok"] and res["calibration"]["codes_equal"] and res["identity"]["ok"]
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if not res["ok"]:
        stop(f"gptq step 0: calib-size's GPTQ‡-256 evaluations differ from parts P / X / Y ({out})")
    G.log(f"calib_size gptq step 0 done: ok (codes {P_CODES[:12]}, file equal to the upload: "
          f"{res['calibration']['file_equal_uploaded']})")


def queue_seconds(prefix):
    """{job name: seconds} of the queue.log END lines whose name starts with prefix."""
    out = {}
    for line in (G.RUN / "queue.log").read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and p[1] == "END" and p[2].startswith(prefix) and p[4].endswith("s"):
            out[p[2]] = float(p[4][:-1])
    return out


def gptq_eta():
    """The ETA of the rest of amendment 5 (RUN/calib_size/gptq_eta.json), from Llama-1B's GPTQ seconds at 256 (step 0)
    and 1,024: GPTQ time linear in the window count, scaled to each model's main-table GPTQ seconds; each job's other
    time as its main-table job; the codes file written at 100 MB/s; the zero-shot and GSM8K times of part AA's FQ16 row
    of the same model and size, plus the codes file read twice per evaluation (sha256, load) at 85 MB/s."""
    s0 = json.loads((ROOT / "step0" / "gptq_256.job.json").read_text())
    k1 = json.loads((ROOT / LLAMA1B / "gptq_1024.job.json").read_text())
    per256, slope = s0["gptq_seconds"], (k1["gptq_seconds"] - s0["gptq_seconds"]) / (1024 - 256)
    jobs = dict(queue_seconds("gptq_"), **queue_seconds("llama_gptq_"))
    evals = dict(queue_seconds("cs_zs_"), **queue_seconds("cs_gsm8k_"))
    rows, total = {}, 0.0
    for n in (1024, 2048):
        for model in ORDER:
            done = k1["job_seconds"] if (model, n) == (LLAMA1B, 1024) else None
            r = json.loads(Z.gptq_record(model).read_text())
            g256 = r["ptq"]["seconds"]
            job256 = jobs.get(f"llama_gptq_{model}" if model in Z.LLAMA else f"gptq_{model}_nvfp4-fo6")
            size = Path(Z.codes_file(model)).stat().st_size
            gptq = g256 * (per256 + slope * (n - 256)) / per256
            calib = (job256 - g256) + gptq + size / 100e6
            ev = evals[f"cs_zs_{model}_fq16_{n}"] + evals[f"cs_gsm8k_{model}_fq16_{n}"] + 4 * size / 85e6
            rows[f"{model}_{n}"] = dict(calibration_s=done if done is not None else calib, evaluations_s=ev,
                                        gptq_s=gptq, measured=done is not None)
            total += (0 if done is not None else calib) + ev
    out = dict(llama1b_gptq_seconds={"256": per256, "1024": k1["gptq_seconds"]}, slope_s_per_window=slope,
               llama1b_1024=dict(job_seconds=k1["job_seconds"], device_peak_mib=k1["device_peak_mib"],
                                 gpu_peak_gib=k1["gpu_peak_gib"]),
               rows=rows, remaining_hours=total / 3600, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (ROOT / "gptq_eta.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def gptq_measure():
    """Amendment 5's ETA gate: Llama-1B GPTQ‡-1k's calibration (its PPL included), then the ETA; nothing else."""
    s0 = ROOT / "step0" / "gptq_step0.json"
    if not (s0.exists() and json.loads(s0.read_text()).get("ok")):
        stop("gptq measure: amendment 5's step 0 has not passed")
    gptq_calibrate_size(LLAMA1B, 1024)
    eta = gptq_eta()
    G.log(f"calib_size gptq measure: Llama-1B 1,024 GPTQ {eta['llama1b_gptq_seconds']['1024']:.0f}s (256: "
          f"{eta['llama1b_gptq_seconds']['256']:.0f}s); remaining ETA {eta['remaining_hours']:.1f} h")
    G.log("DONE calib_size gptq measure")


def gptq_panel(models=ORDER):
    s0 = ROOT / "step0" / "gptq_step0.json"
    if not (s0.exists() and json.loads(s0.read_text()).get("ok")):
        stop("gptq panel: amendment 5's step 0 has not passed")
    for n in (1024, 2048):
        for model in models:
            gptq_size(model, n)
            G.log(f"calib_size gptq panel {n}: {model} done")
        G.log(f"calib_size gptq panel {n} done")
    G.log("DONE calib_size gptq")


# ---------------------------------------------------------------------------------------------- amendment 6: 8x64 / 256x64

#: calib-size 394e683: 9f19863's code (9f19863..394e683 adds results/calib_size and one .gitignore line)
A6_CODE = "394e683"
A6_UNITS = {"8x64": "fq8", "256x64": "fq256"}          # unit: evaluation kind; rows fq8_<N> / fq256_<N>
A6_E0M3_MAX = 0.60                                    # the user's gate for these two tiles
A6_QWEN3_GSM = ["fq-8x64", "fq-256x64"]               # the 256-window GSM8K references the Qwen3 models lack


def a6_step0():
    """Amendment 6, step 0 (RUN/calib_size/step0/a6_step0.json), Llama-1B at the release 256 windows on calib-size:
    - each tile's map recalibrated with part P's settings: run_map_sha256 and tiles equal to part P's (required);
    - part P's map of each tile evaluated on calib-size: zero-shot, GSM8K (date pinned) and PPL identical to parts X / Y
      / P (required), as step 0 (c) for 16x64; then the existing records are the 256-window references."""
    model = LLAMA1B
    out = ROOT / "step0" / "a6_step0.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    G.log("calib_size a6 step 0")
    d = ROOT / model / "eval"
    for unit, kind in A6_UNITS.items():
        ref = json.loads((G.RUN / "llama_instruct" / model / "records" / unit / "run.json").read_text())
        o = ROOT / "step0" / f"{kind}_256"
        rec = calibrate_fq(model, 256, o, unit=unit)
        cal = dict(run_map_sha256=rec["run_map_sha256"], part_p_run_map_sha256=ref["run_map_sha256"],
                   equal_sha=rec["run_map_sha256"] == ref["run_map_sha256"],
                   tiles_equal=same_tiles(o / "map.pt", Z.map_path(model, unit)), e0m3_share=rec["e0m3_share"],
                   job_seconds=rec["job_seconds"], device_peak_mib=rec["device_peak_mib"])
        res[unit] = dict(calibration=cal)
        out.write_text(json.dumps(res, indent=1) + "\n")
        if not (cal["equal_sha"] and cal["tiles_equal"]):
            stop(f"a6 step 0: Llama-1B {unit} at 256 differs from part P's map ({out})")
        name = f"{kind}_256"
        eval_row(model, name, kind, Z.map_path(model, unit))
        ident = dict(zeroshot=zs_identity(d / "zs" / f"{name}.json", Z.ZS / "panel" / model / f"fq-{unit}.json"),
                     gsm8k=gsm_identity(d / "gsm8k" / f"{name}.json", Y.ROOT / model / f"fq-{unit}.json"),
                     ppl=ppl_identity(d / "ppl" / f"{name}.json", G.RUN / "llama_instruct" / model / "ppl" / f"fq-{unit}.json"))
        ident["ok"] = all(v["ok"] for v in ident.values())
        res[unit]["identity"] = ident
        out.write_text(json.dumps(res, indent=1) + "\n")
        if not ident["ok"]:
            stop(f"a6 step 0: Llama-1B {unit}'s evaluations on calib-size differ from parts X / Y / P ({out})")
    res["ok"] = all(res[u]["calibration"]["equal_sha"] and res[u]["calibration"]["tiles_equal"] and res[u]["identity"]["ok"]
                    for u in A6_UNITS)
    out.write_text(json.dumps(res, indent=1) + "\n")
    G.log(f"calib_size a6 step 0 done: ok={res['ok']}")
    if not res["ok"]:
        stop(f"a6 step 0 failed ({out})")


def a6_qwen3_gsm(model):
    """The Qwen3 models' native GSM8K rows of the 256-window 8x64 / 256x64 maps: amendment 1's Qwen3 protocol (part Y's,
    thinking off, batch 64) with part X2's input checks, into RUN/calib_size/<model>/eval/gsm8k/."""
    import run_zs6_qwen3 as X2
    d = ROOT / model / "eval" / "gsm8k"
    for row in A6_QWEN3_GSM:
        out = d / f"{row}.json"
        if not G.done(out):
            checkout()
            bad = X2.verify_inputs(model, row)
            if bad:
                stop(f"a6 qwen3 gsm8k {model} {row}: {bad}")
            out.parent.mkdir(parents=True, exist_ok=True)
        q = Y.quant(model, row)
        cmd = [G.py(model), "-u", "-m", "evaluation.downstream", "--model", model, "--revision", Z.REV[model], *q,
               "--tasks", Y.TASK, "--apply-chat-template", *THINK[model], "--batch-size", 64, "--gen-lengths",
               "--samples-out", out.with_suffix(""), "--out", out]
        must(M.run_ts(f"cs_gsm8k_{model}_{row}", cmd, out, FQ3), f"a6 qwen3 gsm8k {model} {row}")
        bad = y_checks(model, row, out, gsm_digest(model))
        if bad:
            stop(f"a6 qwen3 gsm8k {model} {row}: {bad}")
    G.log(f"calib_size a6 qwen3 gsm8k {model}: {len(A6_QWEN3_GSM)} rows")


def a6_size(model, unit, n=1024):
    """One tile at n windows: the windows checks, the calibration (Qwen3.8-27B with the host teacher), the E0M3 gate
    (60 %), the trainer's fit extension equal to the windows file, then zero-shot, GSM8K and PPL."""
    kind = A6_UNITS[unit]
    d = ROOT / model
    out = d / f"{kind}_{n}"
    if not (d / f"{kind}_{n}.job.json").exists():
        bad = gptq_windows_ok(model, n)
        if bad:
            stop(f"a6 {model} {unit}: windows {bad}")
        if shutil.disk_usage("/home").free < FLOOR + 2e9:
            stop(f"PAUSE disk before a6 {model} {unit}: /home below its floor + 2 GB")
    fq = calibrate_fq(model, n, out, host=model in HOST_TEACHER, unit=unit)
    if fq["e0m3_share"] > A6_E0M3_MAX:
        stop(f"a6 {model} {unit} {n}: E0M3 share {fq['e0m3_share']:.3f} above {A6_E0M3_MAX} (implausible)")
    pre = json.loads(windows_file(model, n).read_text())
    got = json.loads((out / "run" / "report.json").read_text())["fit_extension"]
    if got["records"] != pre["fit_extension"]["records"] or got["skipped_documents"] != pre["fit_extension"][
            "skipped_documents"]:
        stop(f"a6 {model} {unit} {n}: the run's fit extension differs from windows/{model}_{n}.json")
    eval_row(model, f"{kind}_{n}", kind, out / "map.pt")
    G.log(f"calib_size a6 {model} {unit} {n}: E0M3 share {fq['e0m3_share']:.4f}; row done")
    return fq


def a6_eta():
    """After the first model: its two tiles' calibration and evaluation seconds against its FQ16-1k's, the ratio applied
    to every other model's FQ16-1k (part AA), plus the Qwen3 GSM8K rows at part Y's FQ row times (RUN/calib_size/a6_eta.json)."""
    jobs = {}
    for line in (G.RUN / "queue.log").read_text().splitlines():
        q = line.split()
        if len(q) >= 5 and q[1] == "END" and q[2].startswith("cs_") and q[4].endswith("s"):
            jobs[q[2]] = float(q[4][:-1])

    def row_seconds(model, cal, name):
        return jobs.get(cal, 0) + sum(jobs.get(f"cs_{k}_{model}_{name}", 0) for k in ("zs", "gsm8k", "ppl"))
    m0 = ORDER[0]
    base0 = row_seconds(m0, f"cs_fq16_{m0}_1024", "fq16_1024")
    ratio = {u: row_seconds(m0, f"cs_fq{UNIT_TAG[u]}_{m0}_1024", f"{k}_1024") / base0 for u, k in A6_UNITS.items()}
    rows, total = {}, 0.0
    for model in ORDER[1:]:
        base = row_seconds(model, f"cs_fq16_{model}_1024" + ("_host" if model in HOST_TEACHER else ""), "fq16_1024")
        for u in A6_UNITS:
            rows[f"{model}_{u}"] = base * ratio[u]
            total += base * ratio[u]
        if model in QWEN3:
            g = sum(jobs.get(f"cs_gsm8k_{model}_fq16_1024", 0) for _ in A6_QWEN3_GSM)
            rows[f"{model}_qwen3_gsm8k"] = g
            total += g
    out = dict(first_model=m0, first_model_seconds={u: row_seconds(m0, f"cs_fq{UNIT_TAG[u]}_{m0}_1024", f"{k}_1024")
                                                    for u, k in A6_UNITS.items()},
               first_model_fq16_1024_seconds=base0, ratio=ratio, rows=rows, remaining_hours=total / 3600,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (ROOT / "a6_eta.json").write_text(json.dumps(out, indent=1) + "\n")
    G.log(f"calib_size a6 ETA after {m0}: ratio 8x64 {ratio['8x64']:.2f}, 256x64 {ratio['256x64']:.2f} of FQ16-1k; "
          f"remaining {total / 3600:.1f} h")
    return out


def a6_panel(models=ORDER):
    s0 = ROOT / "step0" / "a6_step0.json"
    if not (s0.exists() and json.loads(s0.read_text()).get("ok")):
        stop("a6 panel: amendment 6's step 0 has not passed")
    for i, model in enumerate(models):
        if model in QWEN3:
            a6_qwen3_gsm(model)
        for unit in A6_UNITS:
            a6_size(model, unit)
        G.log(f"calib_size a6 panel: {model} done")
        if model == ORDER[0] and not (ROOT / "a6_eta.json").exists():
            a6_eta()
    G.log("DONE calib_size a6")

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
    elif what == "theta":
        theta_rerun()
    elif what in ("a6_step0", "a6_panel"):
        CODE = A6_CODE                              # amendment 6: calib-size 394e683
        {"a6_step0": a6_step0, "a6_panel": a6_panel}[what]()
    elif what in ("gptq_step0", "gptq_measure", "gptq_panel"):
        CODE = GPTQ_CODE                            # amendment 5: calib-size 8dbe2e4
        {"gptq_step0": gptq_step0, "gptq_measure": gptq_measure, "gptq_panel": gptq_panel}[what]()
    elif what == "panel":
        size = int(next(a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--size=")))
        sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
        panel(size, [m for m in ORDER if not sel or m in sel[0]])
    else:
        raise SystemExit(f"unknown part {what}")
