"""The GPU parts of the paper evaluation (PROTOCOL.md): one process per job, one job at a time on an idle GPU.

    python run_gpu.py parity      # P: harness parity on Phi-4 (NVFP4-RaZeR bench_prefill.py vs flipquant prefill)
    python run_gpu.py smoke       # S: every (model, policy) once, 1x128 (not a result)
    python run_gpu.py latency     # A: 6 models x 7 policies x 5 rounds x 8 shapes, CUDA graphs
    python run_gpu.py memory      # B: 6 models x 6 policies, mem_probe.py
    python run_gpu.py ownership   # C1: the 18 release artifacts installed with --ownership-check
    python run_gpu.py fakeppl     # C2: 5 models x 3 units, simulated (fake) PPL, paper convention

Each job writes its record (skipped when it exists and is complete), its log, and START/END lines to queue.log. A
registered check that fails stops the run (exit 1).
"""
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RZ = Path(__file__).resolve().parents[2]                  # NVFP4-RaZeR (branch paper-eval)
FQ = Path("/home/dev/n16k64_campaign/fqopt/wt")           # flipquant main = 120173a (read-only)
BUILD_V = Path("/home/dev/n16k64_campaign/kernel_opt/build_V")
REL = Path("/home/dev/flipquant_release")
RUN = Path("/home/dev/n16k64_campaign/paper_eval")
PAPER = Path("/home/dev/n16k64_campaign/paper")
PY = "/home/dev/.conda/envs/n16k64/bin/python"
PY_FAST = "/home/dev/.conda/envs/n16k64-fast/bin/python"   # amendment 3: the hybrid models' env
HYBRID = ("nemotron-nano-9b-v2", "qwen3.8-27b")
MODELS = ["qwen3-1.7b", "qwen3-8b", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b", "qwen3.8-27b"]
UNITS = ("8x64", "16x64", "256x64")
SHAPES = "1x128,1x256,1x512,1x1024,1x2048,1x4096,1x8192,4x2048"
SEED = 20260928                                            # step 05's: round r shuffles with SEED + r
NATIVE = ["--mode", "native", "--kernel-build-dir", str(BUILD_V)]
POLICIES = {"bf16": ["--mode", "bf16"],
            "nvfp4": NATIVE + ["--weight", "nvfp4", "--act", "nvfp4"],
            "fo6": NATIVE + ["--weight", "fourover6", "--act", "fourover6"],
            "fo6-wB": NATIVE + ["--weight", "fourover6", "--act", "fourover6", "--kernel-set", "auto_stock_wB"],
            **{f"fq-{u}": NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", f"@{u}"] for u in UNITS}}
FAMILY = {"nvfp4": "stock_ko", "fo6": "stock_ko", "fo6-wB": "stock_wB_ko", "fq-8x64": "mixed_wB_ko",
          "fq-16x64": "mixed_ko", "fq-256x64": "mixed256_ko"}
ENV_DROP = ("PYTHONPATH", "SM120_BUILD_DIR", "PYTORCH_CUDA_ALLOC_CONF")


def env(extra=None):
    e = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    e.update(PYTHONDONTWRITEBYTECODE="1", HF_HOME="/home/dev/.cache/huggingface")
    e.update(extra or {})
    return e


def log(msg):
    with open(RUN / "queue.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n")


def idle():
    while True:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.split()
        if not [p for p in q if p.strip()]:
            return
        time.sleep(15)


def done(path):
    try:
        return json.loads(Path(path).read_text()).get("status", "complete") == "complete"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def py(model):
    """The interpreter of a model's env (amendment 3): n16k64-fast for the hybrid models, n16k64 otherwise."""
    return PY_FAST if model in HYBRID else PY


def policy_args(model, pol):
    out = []
    for a in POLICIES[pol]:
        out.append(str(REL / model / f"flipquant_{a[1:]}.pt") if a.startswith("@") else a)
    return out


def run(name, cmd, out, cwd, extra_env=None):
    if done(out):
        return None
    idle()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    log(f"START {name}")
    t0 = time.time()
    with open(RUN / "logs" / f"{name}.log", "w") as f:
        f.write(" ".join(map(str, cmd)) + "\n")
        f.flush()
        rc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env(extra_env), stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    log(f"END {name} rc={rc} {time.time() - t0:.0f}s")
    return rc


def manifests():
    libs = set()
    for d in BUILD_V.iterdir():
        m = d / "manifest.json"
        if m.exists():
            libs.add(json.loads(m.read_text())["library_sha256"])
    return libs


def shas(obj):
    """Every sha256-looking value under a 'kernel_set' or 'kernel_sha256' record."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) and "sha" in k:
                out.append(v)
            else:
                out += shas(v)
    elif isinstance(obj, list):
        for v in obj:
            out += shas(v)
    return out


def check_fq_record(path, pol, libs, require_graph=True):
    """The registered checks of a flipquant prefill record: complete, every captured shape's graph logits equal
    eager's, coverage, the policy's kernel family from build_V (every library sha256 one of build_V's)."""
    r = json.loads(Path(path).read_text())
    bad = []
    if r.get("status") != "complete":
        bad.append("not complete")
    if require_graph and (r.get("capture_error") or not r.get("graph_equals_eager")):
        bad.append(f"graph: capture_error={r.get('capture_error')} graph_equals_eager={r.get('graph_equals_eager')}")
    if pol != "bf16":
        p = (r.get("policy") or {}).get("native") or {}             # the install report (models/sm120.install)
        if p.get("kernel") != FAMILY[pol] or (r.get("kernel_set") or {}).get("family") != FAMILY[pol]:
            bad.append(f"kernel family {p.get('kernel')} / {(r.get('kernel_set') or {}).get('family')} != {FAMILY[pol]}")
        if Path(p.get("build_dir") or "/nonexistent").resolve() != BUILD_V.resolve():
            bad.append(f"build_dir {p.get('build_dir')}")
        loaded = list(((r.get("kernel_set") or {}).get("kernels") or {}).values()) + \
            list(((p.get("kernel_set") or {}).get("kernels") or {}).values())
        if not loaded:
            bad.append("no loaded kernels recorded")
        foreign = [s for s in loaded if s not in libs]
        if foreign:
            bad.append(f"libraries not from build_V: {foreign[:3]}")
    return bad


def fq_prefill(model, pol, rnd, out, shapes=SHAPES, reps=7, no_graph=False):
    return [py(model), "-m", "evaluation.latency", "prefill", "--model", model, *policy_args(model, pol), "--shapes", shapes,
            "--reps", str(reps), "--label", pol, "--round", str(rnd), "--out", out, *(["--no-graph"] if no_graph else [])]


def stop(msg):
    log(f"STOP {msg}")
    sys.exit(f"STOP: {msg}")


def latency(stage, rounds, shapes, reps, root, no_graph_models=()):
    libs = manifests()
    for model in MODELS:
        pols = list(POLICIES)
        for r in range(1, rounds + 1):
            order = list(pols)
            random.Random(SEED + r).shuffle(order)
            log(f"{stage} {model} round {r} order {','.join(order)} (seed {SEED + r})")
            for pol in order:
                out = RUN / root / model / pol / f"round{r}.json"
                ng = model in no_graph_models
                rc = run(f"{stage}_{model}_{pol}_r{r}", fq_prefill(model, pol, r, out, shapes, reps, ng), out, FQ)
                if rc is None:
                    continue
                if rc != 0 or not out.exists():
                    stop(f"{stage} {model} {pol} r{r}: rc={rc}")
                bad = check_fq_record(out, pol, libs, require_graph=not ng)
                if bad:
                    stop(f"{stage} {model} {pol} r{r}: {bad}")


def parity():
    """P: Phi-4, both harnesses on build_V, the same paper 16x64 map; BF16, FourOverSix, FlipQuant 16x64."""
    libs = manifests()
    art = {"fo6": (PAPER / "artifacts" / "phi4_fo6", "auto_stock"), "fq-16x64": (PAPER / "artifacts" / "phi4_tc_16x64", "auto")}
    paper_map = PAPER / "maps" / "phi4_16x64" / "map.pt"
    for r in range(1, 4):
        harnesses = ["razer", "flipquant"] if r % 2 else ["flipquant", "razer"]
        pols = ["bf16", "fo6", "fq-16x64"]
        pols = pols[(r - 1) % 3:] + pols[:(r - 1) % 3]
        log(f"parity round {r} harnesses {','.join(harnesses)} policies {','.join(pols)}")
        for h in harnesses:
            for pol in pols:
                out = RUN / "parity" / h / pol / f"round{r}.json"
                if h == "razer":
                    cmd = [PY, RZ / "experiments" / "paper" / "bench_prefill.py", "--model", "phi4", "--label", pol,
                           "--round", str(r), "--shapes", SHAPES, "--reps", "7", "--out", out]
                    if pol != "bf16":
                        cmd += ["--artifact", art[pol][0], "--kernel", art[pol][1]]
                    rc = run(f"parity_razer_{pol}_r{r}", cmd, out, RZ, {"SM120_BUILD_DIR": str(BUILD_V)})
                else:
                    pa = policy_args("phi4-14b", pol) if pol != "fq-16x64" else \
                        NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", str(paper_map), "--unit", "16x64"]
                    cmd = [PY, "-m", "evaluation.latency", "prefill", "--model", "phi4-14b", *pa, "--shapes", SHAPES,
                           "--reps", "7", "--label", pol, "--round", str(r), "--out", out]
                    rc = run(f"parity_flipquant_{pol}_r{r}", cmd, out, FQ)
                    if rc == 0 and check_fq_record(out, pol, libs):
                        stop(f"parity flipquant {pol} r{r}: {check_fq_record(out, pol, libs)}")
                if rc not in (None, 0):
                    stop(f"parity {h} {pol} r{r}: rc={rc}")


def memory():
    for model in MODELS:
        for pol in ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]:
            out = RUN / "memory" / model / f"{pol}.json"
            cmd = [py(model), RZ / "experiments" / "paper_eval" / "mem_probe.py", "--model", model, *policy_args(model, pol),
                   "--label", pol, "--out", out]
            rc = run(f"memory_{model}_{pol}", cmd, out, FQ)
            if rc not in (None, 0):
                stop(f"memory {model} {pol}: rc={rc}")


def ownership():
    for model in MODELS:
        for u in UNITS:
            out = RUN / "ownership" / f"{model}_{u}.json"
            cmd = [py(model), "-m", "evaluation.ppl", "--model", model, *policy_args(model, f"fq-{u}"), "--paper-convention",
                   "--ownership-check", "--limit", "1", "--out", out]
            rc = run(f"ownership_{model}_{u}", cmd, out, FQ)
            if rc not in (None, 0):
                stop(f"ownership {model} {u}: rc={rc}")


def fakeppl():
    for model in [m for m in MODELS if m != "qwen3.8-27b"]:
        for u in UNITS:
            out = RUN / "fakeppl" / f"{model}_{u}.json"
            cmd = [py(model), "-m", "evaluation.ppl", "--model", model, "--mode", "fake", "--weight", "mixfp4", "--act",
                   "fourover6", "--map", REL / model / f"flipquant_{u}.pt", "--paper-convention", "--out", out]
            rc = run(f"fakeppl_{model}_{u}", cmd, out, FQ)
            if rc not in (None, 0):
                stop(f"fakeppl {model} {u}: rc={rc}")


if __name__ == "__main__":
    what = sys.argv[1]
    rest = sys.argv[2:]
    sel = [a.split("=", 1)[1] for a in rest if a.startswith("--models=")]
    if sel:                                    # amendment 3: a model subset, in the registered order
        keep = sel[0].split(",")
        assert set(keep) <= set(MODELS), keep
        MODELS[:] = [m for m in MODELS if m in keep]
    rest = [a for a in rest if not a.startswith("--models=")]
    no_graph = [m for m in (rest[0].split(",") if rest else []) if m]
    RUN.mkdir(parents=True, exist_ok=True)
    if what != "parity" and any(m in HYBRID for m in MODELS) and not (RUN / "env" / "fast_env_verified.json").exists():
        stop("the hybrid models run only in n16k64-fast, after its verification (amendment 3)")
    log(f"{what} models {','.join(MODELS)}")
    if what == "parity":
        parity()
    elif what == "smoke":
        latency("smoke", 1, "1x128", 3, "smoke", no_graph)
    elif what == "latency":
        latency("latency", 5, SHAPES, 7, "latency", no_graph)
    else:
        {"memory": memory, "ownership": ownership, "fakeppl": fakeppl}[what]()
    log(f"DONE {what}")
