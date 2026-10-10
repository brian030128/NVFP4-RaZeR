"""Part P (PROTOCOL.md): Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct in tab:main-ppl, with every method of the table,
after part O; the existing main-table rows' exact settings; n16k64; one GPU job at a time.

    python run_llama_instruct.py record --models M,..   # the calibration record and development records (CPU; downloads)
    python run_llama_instruct.py maps --models M,..     # 1. FlipQuant maps, 3 units, measured as the release runs
    python run_llama_instruct.py ppl --models M,..      # 2. native PPL: BF16, NVFP4, FourOverSix, FlipQuant x 3
    python run_llama_instruct.py mainrows --models M,.. # 3. IF4 and MixFP4 (Zou), simulated (part I)
    python run_llama_instruct.py gptq --models M,..     # 4. GPTQ-double-dagger (part J), no code file
    python run_llama_instruct.py focus --models M,..    # 5. FOCUS (part K)
    python run_llama_instruct.py windows --models M,..  # the 256 windows' token hashes (CPU; vs base Llama-3.1-8B)

Layout under RUN/llama_instruct/<model>/: runs/<model>_<unit>/ (train_map's output), maps/flipquant_<unit>.pt,
records/<unit>/{run.json, reproduction.json, trainer_report.json, train.log} (the release layout), ppl/<label>.json,
mainrows/{if4,zou}.json, gptqdd/ppl.json, focus/{state/, ppl.json}. The calibration data root is the release runs'
(fqrel/tmopt_data). --focus-root moves FOCUS states elsewhere (the disk decision).
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_sm120 as S  # noqa: E402

ROOT = G.RUN / "llama_instruct"
DATA = S.DATA_ROOT                                      # the release runs' calibration data root
UNITS = ("8x64", "16x64", "256x64")
TRAIN = ["--fit-windows", "256", "--epochs", "5", "--teacher-topk", "1000"]
KEYS = {"llama3.1-8b-instruct": "llama8b_ins", "llama3.2-1b-instruct": "llama1b_ins"}
REV = {"llama3.1-8b-instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
       "llama3.2-1b-instruct": "9213176726f574b556790deb65791e0c5aa438b6"}
# the vendored run_multiround.PUBLISHED: the published C4 evaluation documents the extension rule skips
PUBLISHED = {"llama8b_ins": "results/kse_paper/job_336566/llama8b/report.json"}
BASE_RELEASE = G.REL / "llama3.1-8b" / "records" / "16x64" / "trainer_report.json"   # base Llama-3.1-8B's 256 windows


class Sampler(threading.Thread):
    """The device's peak memory.used (MiB), sampled every 2 s while a job runs (fqrel/release.py)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.halt = 0, threading.Event()

    def run(self):
        while not self.halt.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()
            if out:
                self.peak = max(self.peak, int(out[0]))
            self.halt.wait(2)


def measured(name, cmd, out, cwd=S.FQ2):
    """G.run with the release runs' device-peak sampler; returns (rc, seconds, peak MiB) or None if done."""
    if G.done(out):
        return None
    sampler = Sampler()
    G.idle()
    sampler.start()
    t = time.time()
    rc = G.run(name, cmd, out, cwd)
    sampler.halt.set()
    sampler.join()
    return rc, time.time() - t, sampler.peak


def checkout():
    def git(*a):
        return subprocess.run(["git", "-C", str(S.FQ2), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), clean=git("status", "--porcelain", "--untracked-files=no") == "")


def record(models):
    for model in models:
        out = ROOT / model / "record.json"
        if G.done(out):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [G.PY, "-m", "calibration.tmopt_ext", "record", "--model", model, "--data-root", DATA]
        t = time.time()
        G.log(f"START llama_record_{model} (CPU; downloads the weights)")
        r = subprocess.run(list(map(str, cmd)), cwd=S.FQ2, env=G.env({"CUDA_VISIBLE_DEVICES": ""}), capture_output=True,
                           text=True)
        (ROOT / model / "record.log").write_text(r.stdout + r.stderr)
        G.log(f"END llama_record_{model} rc={r.returncode} {time.time() - t:.0f}s")
        if r.returncode:
            G.stop(f"llama record {model}: rc={r.returncode}")
        key = KEYS[model]
        rec = json.loads((DATA / key / "calibration" / "report.json").read_text())
        out.write_text(json.dumps(dict(status="complete", model=model, key=key, record=str(DATA / key / "calibration"),
                                       prepare_summary=json.loads((DATA / key / "prepare_summary.json").read_text()),
                                       matrices=len(rec["matrices"]), revision=rec["revision"],
                                       seconds=time.time() - t), indent=1) + "\n")


def check_extension(run_dir, key, first_records=None):
    """fqrel/release.py's check of one run's fit extension (128 windows, distinct, none a fit, development or published
    C4 document, nor a development window's tokens; identical to the model's first unit)."""
    rep = json.loads((run_dir / "reproduction.json").read_text())
    report = json.loads((run_dir / "run" / "report.json").read_text())
    ext = report["fit_extension"]
    prior = json.loads((DATA / key / "calibration" / "report.json").read_text())
    fit = {d["document_sha256"] for s in ("math", "code") for d in prior["fit"][s]["documents"]}
    dev_docs, dev_tokens = set(), set()
    for manifest in rep["development_records"]:
        records = json.loads(Path(manifest).read_text())["records"]
        assert len(records) == 64, manifest
        dev_docs |= {r["document_sha256"] for r in records}
        dev_tokens |= {r["token_sha256"] for r in records}
    c4 = set()
    if key in PUBLISHED:
        c4 = {d["document_sha256"] for d in json.loads((S.FQ2 / "calibration" / "tmopt" / PUBLISHED[key]).read_text())
              ["data"]["c4_paper"]["documents"]}
    records = ext["records"]
    docs, tokens = [r["document_sha256"] for r in records], [r["token_sha256"] for r in records]
    checks = dict(
        windows_128=ext["windows"] == 128 and len(records) == 128,
        math_64_code_64=[r["source"] for r in records].count("math") == 64 and [r["source"] for r in records].count("code") == 64,
        skip_fit_128=ext["skipped_documents"]["fit"] == len(fit) == 128,
        skip_development_192=ext["skipped_documents"]["development"] == len(dev_docs) == 192,
        skip_c4_published=ext["skipped_documents"]["c4_evaluation"] == len(c4),
        distinct_documents=len(set(docs)) == 128, distinct_tokens=len(set(tokens)) == 128,
        disjoint_from_fit=not set(docs) & fit, disjoint_from_development=not set(docs) & dev_docs,
        disjoint_from_c4=not set(docs) & c4, tokens_not_development=not set(tokens) & dev_tokens,
        development_disjoint_from_fit=not dev_docs & fit,
        same_as_first_unit=first_records is None or records == first_records)
    return checks, records, dict(skipped=ext["skipped_documents"], c4_published=len(c4))


def maps(models):
    for model in models:
        key, first = KEYS[model], None
        for unit in UNITS:
            d = ROOT / model
            out, target, recs = d / "runs" / f"{model}_{unit}", d / "maps" / f"flipquant_{unit}.pt", d / "records" / unit
            if (recs / "run.json").exists():
                first = first or check_extension(out, key)[1]
                continue
            if out.exists():
                G.stop(f"{out} exists without a record; inspect it before rerunning")
            cmd = [G.PY, "-m", "calibration.train_map", "--model", model, "--unit", unit, *TRAIN, "--data-root", DATA,
                   "--out", out]
            code = checkout()
            res = measured(f"llama_train_{model}_{unit}", cmd, out / "reproduction.json")
            rc, seconds, peak = res if res else (0, None, None)
            rec = dict(model=model, unit=unit, command=" ".join(map(str, cmd[1:])), checkout=code, rc=rc,
                       seconds=seconds, device_peak_mib=peak)
            if rc != 0:
                G.stop(f"llama train {model} {unit}: rc={rc}")
            checks, records, info = check_extension(out, key, first)
            first = first or records
            rec["extension"] = dict(checks=checks, **info, ok=all(checks.values()))
            rep = json.loads((out / "reproduction.json").read_text())
            report = json.loads((out / "run" / "report.json").read_text())
            res_ = report["resources"]
            rec.update(map_sha256=rep["map_file_sha256"], run_map_sha256=rep["run_map_sha256"], settings=rep["settings"],
                       e0m3_share=rep["e0m3_share"], train_map_wall_seconds=rep["wall_seconds"],
                       trainer=dict(total_seconds=res_.get("total_seconds"), setup_seconds=report.get("setup_seconds"),
                                    training_seconds=report.get("training_seconds"),
                                    gpu_peak_allocated_gib=res_.get("gpu_peak_allocated_gib"),
                                    gpu_peak_reserved_gib=res_.get("gpu_peak_reserved_gib"),
                                    cpu_peak_rss_gib=res_.get("cpu_peak_rss_gib"),
                                    phases={p["name"]: dict(seconds=p.get("seconds"), host_peak_rss=p.get("host_peak_rss"),
                                                            gpu_peak_allocated=p.get("gpu_peak_allocated"))
                                            for p in (res_.get("phases") or {}).get("phases", [])}),
                       teacher_storage=report.get("teacher_storage"))
            if not rec["extension"]["ok"]:
                G.stop(f"llama train {model} {unit}: the fit-extension check failed {checks}")
            recs.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / "reproduction.json", recs / "reproduction.json")
            shutil.copyfile(out / "run" / "report.json", recs / "trainer_report.json")
            shutil.copyfile(out / "train.log", recs / "train.log")
            shutil.copyfile(out / "map.pt", target)
            (recs / "run.json").write_text(json.dumps(rec, indent=1) + "\n")


def trainer_report(model):
    return ROOT / model / "records" / "16x64" / "trainer_report.json"


def ppl_cmd(model, quant, out):
    return [G.PY, "-m", "evaluation.ppl", "--model", model, "--revision", REV[model], *quant, "--out", out]


POLICIES = {"bf16": ["--mode", "bf16", "--paper-convention"],
            "nvfp4": [*G.NATIVE, "--kernel-set", "auto", "--weight", "nvfp4", "--act", "nvfp4", "--act-scope", "row"],
            "fo6": [*G.NATIVE, "--kernel-set", "auto", "--weight", "fourover6", "--act", "fourover6", "--paper-convention"],
            **{f"fq-{u}": [*G.NATIVE, "--kernel-set", "auto", "--weight", "mixfp4", "--act", "fourover6", "--map", f"@{u}",
                           "--paper-convention"] for u in UNITS}}


def ppl(models):
    for model in models:
        for pol, q in POLICIES.items():
            q = [str(ROOT / model / "maps" / f"flipquant_{a[1:]}.pt") if a.startswith("@") else a for a in q]
            out = ROOT / model / "ppl" / f"{pol}.json"
            res = measured(f"llama_ppl_{model}_{pol}", ppl_cmd(model, q, out), out)
            if res and res[0] != 0:
                G.stop(f"llama ppl {model} {pol}: rc={res[0]}")


def mainrows(models):
    for model in models:
        for weight, name in (("if4", "if4"), ("zou_mixfp4", "zou")):
            out = ROOT / model / "mainrows" / f"{name}.json"
            q = ["--mode", "fake", "--weight", weight, "--act-method", "own", "--unit", "1x16"]
            res = measured(f"llama_main_{model}_{name}", ppl_cmd(model, q, out), out)
            if res and res[0] != 0:
                G.stop(f"llama main {model} {name}: rc={res[0]}")


def gptq(models):
    """Part J's GPTQ-double-dagger (run_sm120.GPTQ_ROWS['nvfp4-fo6'], gptq_flags: BF16 propagation, the release fit set
    with the model's own 16x64 trainer report as the extension), without --gptq-cache (disk; codes_sha256 recorded)."""
    for model in models:
        out = ROOT / model / "gptqdd" / "ppl.json"
        q, _ = S.GPTQ_ROWS["nvfp4-fo6"]
        flags = ["--ptq", "gptq", "--gptq-propagate", "bf16", "--gptq-hessian-input", "bf16", "--gptq-calib", "release",
                 "--gptq-calib-record", DATA / KEYS[model] / "calibration" / "report.json",
                 "--gptq-calib-extension", trainer_report(model), "--gptq-block", 128, "--gptq-damp", 0.01,
                 "--gptq-batch", 8]
        res = measured(f"llama_gptq_{model}", ppl_cmd(model, [*G.NATIVE, "--kernel-set", "auto", *q, *flags], out), out)
        if res and res[0] != 0:
            G.stop(f"llama gptq {model}: rc={res[0]}")


def focus(models, focus_root):
    """Part K (run_sm120.focus): FOCUS_COMMON, the release fit set (record + the model's own 16x64 trainer report), micro
    8 halved on OOM; deployed as NVFP4 codes, FourOverSix per-token activations, native."""
    for model in models:
        state = Path(focus_root) / model / "state"
        rep_ = state / "report.json"
        if not G.done(rep_):
            for micro in (8, 4, 2, 1):
                if state.exists():
                    G.stop(f"llama focus {model}: {state} exists from an earlier attempt; nothing is deleted here")
                cmd = [G.PY, "-m", "calibration.train_focus", "--model", model, "--revision", REV[model], "--data",
                       "release", "--calib-record", DATA / KEYS[model] / "calibration" / "report.json",
                       "--calib-extension", trainer_report(model), *S.FOCUS_COMMON, "--micro", micro, "--out", state]
                state.parent.mkdir(parents=True, exist_ok=True)
                rc = S.run_fresh_dir(f"llama_focus_train_{model}_micro{micro}", cmd, rep_, S.FQ2)
                if rc in (None, 0):
                    break
                logf = (G.RUN / "logs" / f"llama_focus_train_{model}_micro{micro}.log").read_text(errors="replace")
                if "OutOfMemoryError" not in logf and "CUDA out of memory" not in logf:
                    G.stop(f"llama focus train {model}: rc={rc}")
                G.stop(f"llama focus {model}: OOM at micro {micro}; report (a rerun needs a fresh state directory)")
        out = ROOT / model / "focus" / "ppl.json"
        q = [*G.NATIVE, "--kernel-set", "auto", "--weight", "focus", "--focus", state / "focus.pt", "--focus-deploy",
             "--act", "fourover6", "--act-scope", "row", "--paper-convention"]
        res = measured(f"llama_focus_ppl_{model}", ppl_cmd(model, q, out), out)
        if res and res[0] != 0:
            G.stop(f"llama focus ppl {model}: rc={res[0]}")


def windows(models):
    """Every window's token sha256 (the 128 of the record, the 128 of the extension), and, for Llama-3.1-8B-Instruct,
    whether they equal base Llama-3.1-8B's release windows."""
    base = json.loads(BASE_RELEASE.read_text())
    base_prior = json.loads((DATA / "llama8b" / "calibration" / "report.json").read_text()) \
        if (DATA / "llama8b" / "calibration" / "report.json").exists() else None
    for model in models:
        key = KEYS[model]
        prior = json.loads((DATA / key / "calibration" / "report.json").read_text())
        rep = json.loads(trainer_report(model).read_text())
        fit = {s: prior["fit"][s]["token_sha256"] for s in ("math", "code")}
        ext = [r["token_sha256"] for r in rep["fit_extension"]["records"]]
        out = dict(model=model, key=key, fit_token_sha256=fit, extension_token_sha256=ext)
        if model == "llama3.1-8b-instruct":
            bext = [r["token_sha256"] for r in base["fit_extension"]["records"]]
            out.update(base_release=str(BASE_RELEASE), extension_equals_base=ext == bext,
                       fit_equals_base=(base_prior is not None and all(
                           fit[s] == base_prior["fit"][s]["token_sha256"] for s in ("math", "code"))))
        (ROOT / model / "windows.json").write_text(json.dumps(out, indent=1) + "\n")
        print(model, {k: v for k, v in out.items() if k.endswith("_base")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("record", "maps", "ppl", "mainrows", "gptq", "focus", "windows"))
    ap.add_argument("--models", default="llama3.1-8b-instruct")
    ap.add_argument("--focus-root", default=str(ROOT / "focus_states"))
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    assert set(models) <= set(KEYS), models
    G.log(f"llama_instruct (part P) {a.what} models {models}")
    {"record": record, "maps": maps, "ppl": ppl, "mainrows": mainrows, "gptq": gptq, "windows": windows}.get(
        a.what, lambda m: focus(m, a.focus_root))(models)
