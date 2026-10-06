"""Release maps: calibrate every model x unit with flipquant's calibration.train_map at the release settings, then
evaluate each map and the model's FourOverSix and BF16 references with evaluation.ppl (native, --paper-convention,
WikiText-2 and C4, all windows). One GPU job at a time, on an idle GPU.

    python release.py [--models a,b,...] [--units 8x64,...] [--no-ppl] [--ppl-only]

Resumable: a step whose release output already exists (and checks out) is skipped. A model stops at its first failed
step (training, the fit-extension check, or a PPL run); the others go on.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

W = Path("/home/dev/n16k64_campaign/fqopt/wt")              # the flipquant checkout (branch optimized-defaults)
PY = "/home/dev/.conda/envs/n16k64/bin/python"
BASE = Path("/home/dev/n16k64_campaign/fqrel")
DATA = BASE / "tmopt_data"                                  # calibration records, prepared by train_map
RUNS = BASE / "runs"                                        # train_map's output directories
LOGS = BASE / "logs"
REL = Path("/home/dev/flipquant_release")                  # the release directory (outside git)
MODELS = ["qwen3-1.7b", "mistral-7b", "mistral-7b-base", "llama3.1-8b", "qwen3-8b", "phi4-14b",
          "nemotron-nano-9b-v2", "qwen3.8-27b"]
UNITS = ["8x64", "16x64", "256x64"]
TRAIN = ["--fit-windows", "256", "--epochs", "5", "--teacher-topk", "1000"]
# the vendored run_multiround.PUBLISHED: the published C4 evaluation documents the extension rule skips
PUBLISHED = {"llama8b": "results/kse_paper/job_336566/llama8b/report.json",
             "qwen27b": "results/kse_paper/job_336969/qwen27b/report.json"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(BASE / "queue.log", "a") as f:
        f.write(line + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def env(offline):
    e = dict(os.environ)
    for k in ("SM120_BUILD_DIR", "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES"):
        e.pop(k, None)
    e.update(PYTHONDONTWRITEBYTECODE="1", HF_HOME="/home/dev/.cache/huggingface", HF_HUB_OFFLINE="1" if offline else "0")
    return e


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout
    return bool(out.strip())


class Sampler(threading.Thread):
    """The device's peak memory.used (MiB), sampled every 2 s while a job runs."""

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


def gpu_job(name, cmd, offline):
    """Run one GPU job from the checkout; returns (rc, seconds, device peak MiB)."""
    waited = time.time()
    while gpu_busy():
        if time.time() - waited > 7200:
            raise SystemExit(f"the GPU was busy for 2 h before {name}")
        time.sleep(30)
    log(f"START {name}")
    sampler = Sampler()
    sampler.start()
    t = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        f.write(" ".join(map(str, cmd)) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=W, env=env(offline), stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    seconds = time.time() - t
    sampler.halt.set()
    sampler.join()
    log(f"END {name} rc={rc} {seconds:.0f}s device_peak={sampler.peak}MiB")
    return rc, seconds, sampler.peak


def checkout():
    def git(*a):
        return subprocess.run(["git", "-C", str(W), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), clean=git("status", "--porcelain", "--untracked-files=no") == "")


def check_extension(run_dir, first_records=None):
    """The fit extension of one run: 128 windows (64 math, 64 code) of distinct documents and distinct tokens, none of
    them a paper fit, development or published C4 document, nor a development window's tokens; identical to the
    model's first unit."""
    rep = json.loads((run_dir / "reproduction.json").read_text())
    report = json.loads((run_dir / "run" / "report.json").read_text())
    ext, key = report["fit_extension"], rep["model_key"]
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
        c4 = {d["document_sha256"]
              for d in json.loads((W / "calibration" / "tmopt" / PUBLISHED[key]).read_text())["data"]["c4_paper"]["documents"]}
    records = ext["records"]
    docs = [r["document_sha256"] for r in records]
    tokens = [r["token_sha256"] for r in records]
    checks = dict(
        windows_128=ext["windows"] == 128 and len(records) == 128,
        math_64_code_64=[r["source"] for r in records].count("math") == 64 and [r["source"] for r in records].count("code") == 64,
        skip_fit_128=ext["skipped_documents"]["fit"] == len(fit) == 128,
        skip_development_192=ext["skipped_documents"]["development"] == len(dev_docs) == 192,
        skip_c4_published=ext["skipped_documents"]["c4_evaluation"] == len(c4),
        distinct_documents=len(set(docs)) == 128,
        distinct_tokens=len(set(tokens)) == 128,
        disjoint_from_fit=not set(docs) & fit,
        disjoint_from_development=not set(docs) & dev_docs,
        disjoint_from_c4=not set(docs) & c4,
        tokens_not_development=not set(tokens) & dev_tokens,
        development_disjoint_from_fit=not dev_docs & fit,
        same_as_first_unit=first_records is None or records == first_records)
    return checks, records, dict(skipped=ext["skipped_documents"], c4_published=len(c4))


def train(model, unit, state):
    name = f"train_{model}_{unit}"
    out, target = RUNS / f"{model}_{unit}", REL / model / f"flipquant_{unit}.pt"
    if target.exists() and (REL / model / "records" / unit / "run.json").exists():
        rec = json.loads((REL / model / "records" / unit / "run.json").read_text())
        checks, records, _ = check_extension(out, state.get("first_records"))      # resumed: the same checks again
        state.setdefault("first_records", records)
        rec["extension"]["checks"], rec["extension"]["ok"] = checks, all(checks.values())
        return rec
    if out.exists():
        raise SystemExit(f"{out} exists without a release record; inspect it before rerunning")
    code = checkout()
    cmd = [PY, "-m", "calibration.train_map", "--model", model, "--unit", unit, *TRAIN, "--data-root", DATA,
           "--out", out]
    rc, seconds, peak = gpu_job(name, cmd, offline=False)
    rec = dict(model=model, unit=unit, command=" ".join(map(str, cmd[1:])), checkout=code, rc=rc, seconds=seconds,
               device_peak_mib=peak, log=str(LOGS / f"{name}.log"))
    if rc != 0:
        rec["error"] = (LOGS / f"{name}.log").read_text()[-3000:]
        return rec
    checks, records, info = check_extension(out, state.get("first_records"))
    state.setdefault("first_records", records)
    rec["extension"] = dict(checks=checks, **info, ok=all(checks.values()))
    rep = json.loads((out / "reproduction.json").read_text())
    report = json.loads((out / "run" / "report.json").read_text())
    res = report["resources"]
    rec.update(map_sha256=rep["map_file_sha256"], run_map_sha256=rep["run_map_sha256"],
               settings=rep["settings"], non_default_settings=rep["non_default_settings"],
               e0m3_share=rep["e0m3_share"], train_map_wall_seconds=rep["wall_seconds"],
               trainer=dict(total_seconds=res.get("total_seconds"), setup_seconds=report.get("setup_seconds"),
                            training_seconds=report.get("training_seconds"),
                            gpu_peak_allocated_gib=res.get("gpu_peak_allocated_gib"),
                            gpu_peak_reserved_gib=res.get("gpu_peak_reserved_gib"),
                            cpu_peak_rss_gib=res.get("cpu_peak_rss_gib"),
                            phases={p["name"]: dict(seconds=p.get("seconds"), host_peak_rss=p.get("host_peak_rss"),
                                                    gpu_peak_allocated=p.get("gpu_peak_allocated"))
                                    for p in (res.get("phases") or {}).get("phases", [])}),
               teacher_storage=report.get("teacher_storage"))
    if not rec["extension"]["ok"]:
        return rec
    import torch
    sys.path.insert(0, str(W))
    from flipquant import maps as M
    tiles, u, meta = M.load(out / "map.pt")
    rec.update(e0m3_tiles=meta["e0m3_tiles"], tiles=meta["tiles"], modules=meta["modules"])
    dest = REL / model / "records" / unit
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "reproduction.json", dest / "reproduction.json")
    shutil.copyfile(out / "run" / "report.json", dest / "trainer_report.json")
    shutil.copyfile(out / "train.log", dest / "train.log")
    shutil.copyfile(out / "map.pt", target)
    assert sha(target) == rep["map_file_sha256"]
    (dest / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def ppl(model, label, extra):
    name = f"ppl_{model}_{label}"
    out = REL / model / "ppl" / f"{label}.json"
    if out.exists():
        return dict(label=label, out=str(out), skipped=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "evaluation.ppl", "--model", model, *extra, "--paper-convention", "--out", out]
    rc, seconds, peak = gpu_job(name, cmd, offline=False)
    return dict(label=label, out=str(out), rc=rc, seconds=seconds, device_peak_mib=peak, command=" ".join(map(str, cmd[1:])),
                checkout=checkout())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--units", default=",".join(UNITS))
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--ppl-only", action="store_true")
    args = ap.parse_args()
    for d in (RUNS, LOGS, REL):
        d.mkdir(parents=True, exist_ok=True)
    log(f"BEGIN models={args.models} units={args.units} checkout={checkout()}")
    for model in args.models.split(","):
        state, failed, summary = {}, None, dict(model=model, train={}, ppl={})
        if not args.ppl_only:
            for unit in args.units.split(","):
                rec = train(model, unit, state)
                summary["train"][unit] = {k: v for k, v in rec.items() if k != "error"}
                if rec.get("rc") != 0 and "rc" in rec:
                    failed = f"train {unit} rc={rec['rc']}"
                elif rec.get("extension") and not rec["extension"]["ok"]:
                    failed = f"fit extension {unit}: {[k for k, v in rec['extension']['checks'].items() if not v]}"
                if failed:
                    log(f"STOP {model}: {failed}")
                    break
                log(f"OK {model} {unit}: e0m3 {rec.get('e0m3_tiles')}/{rec.get('tiles')} map {rec.get('map_sha256', '')[:12]}"
                    f" extension ok")
        if not failed and not args.no_ppl:
            runs = [("bf16", ["--mode", "bf16"]), ("fo6", ["--mode", "native", "--weight", "fourover6"])]
            runs += [(f"flipquant_{u}", ["--mode", "native", "--weight", "mixfp4", "--map",
                                         str(REL / model / f"flipquant_{u}.pt")]) for u in args.units.split(",")]
            # the model card's simulated-quantization command, smoke-tested (2 windows per corpus)
            runs += [("smoke_fake_16x64", ["--mode", "fake", "--weight", "mixfp4", "--map",
                                           str(REL / model / "flipquant_16x64.pt"), "--limit", "2"])]
            for label, extra in runs:
                r = ppl(model, label, extra)
                summary["ppl"][label] = r
                if r.get("rc", 0) != 0:
                    failed = f"ppl {label} rc={r['rc']}"
                    log(f"STOP {model}: {failed}")
                    break
        summary["failed"] = failed
        (BASE / "state").mkdir(exist_ok=True)
        (BASE / "state" / f"{model}.json").write_text(json.dumps(summary, indent=1) + "\n")
    log("DONE")


if __name__ == "__main__":
    main()
