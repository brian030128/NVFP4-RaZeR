"""Part X (PROTOCOL.md): six zero-shot tasks x six models x ten rows through flipquant main 3db0072, after step 0.

    python run_zeroshot6.py step0           # codes (CPU), parity, identity, ppl checks; then report_zeroshot6.py step0
    python run_zeroshot6.py panel [--models=a,b]   # the panel (small models first), the Mistral fake cross-check

One GPU job at a time on an idle GPU (run_gpu.run): records under RUN/zeroshot6/, logs RUN/logs/zs6_*.log, START / END
in RUN/queue.log. A finished record is not run again; a failed job or check stops the run (STOP in queue.log).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gpu as G  # noqa: E402
import run_sm120 as S  # noqa: E402

FQ = Path("/home/dev/n16k64_campaign/zeroshot6/wt")               # flipquant origin/main, detached, unmodified
FQ_COMMIT = "3db0072e891c8697c2c960ca73c6ccc73dd1d380"
RZ_MAIN = Path("/home/dev/NVFP4-RaZeR")                           # NVFP4-RaZeR kernel-opt (the harness reference)
RZ_COMMIT = "9580a2a"
ZS = G.RUN / "zeroshot6"
TASKS = "piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai"
SIZES = dict(piqa=1838, hellaswag=10042, winogrande=1267, arc_easy=2376, arc_challenge=1172, lambada_openai=5153)
LAMBADA_PIN = ("EleutherAI/lambada_openai", "900124bf3b8235c6daf21033af9948b3f07346c4")
MODELS = ["llama3.2-1b-instruct", "mistral-7b", "llama3.1-8b-instruct", "nemotron-nano-9b-v2", "phi4-14b", "qwen3.8-27b"]
LLAMA = {"llama3.2-1b-instruct": "llama1b_ins", "llama3.1-8b-instruct": "llama8b_ins"}
REV = dict(S.REV, **{"llama3.2-1b-instruct": "9213176726f574b556790deb65791e0c5aa438b6",
                     "llama3.1-8b-instruct": "0e9e39f249a16976918f6564b8830bc894c89659"})
ROWS = ["bf16", "nvfp4", "fo6", "focus", "gptq", "if4", "zou", "fq-8x64", "fq-16x64", "fq-256x64"]
FAMILY = {"nvfp4": "stock_ko", "fo6": "stock_ko", "focus": "stock_ko", "gptq": "stock_ko", "fq-8x64": "mixed_wB_ko",
          "fq-16x64": "mixed_ko", "fq-256x64": "mixed256_ko"}
NATIVE = ["--mode", "native", "--kernel-build-dir", str(G.BUILD_V), "--kernel-set", "auto"]
ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
FOCUS_SHA = {"mistral-7b": "7963825abb5a23c9d38082c76063d477ee249f5b652f0e3edb5393d1dea376aa",
             "nemotron-nano-9b-v2": "cb772e159dd8d6a5fb72f3ab63c14bb58a481db39bfa4aca48ca0600bd510c3f",
             "phi4-14b": "be66f52576587c0c71473e19c378bb71b2a48d89802ffa0d3e24c1c4690cd38f",
             "llama3.1-8b-instruct": "e311a189f43cf9161f837a855346683c1cd7225bb1984cd27eb796751c80c82a",
             "qwen3.8-27b": "baf02d9c36551191fe7bc216f6408346cc7d34471cc3e9b524eb60e3f81eeb9b",
             "llama3.2-1b-instruct": None}      # no earlier record: step 0 records it
PAPER = G.PAPER
PARITY_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande"
PRIMARY = dict(piqa="acc_norm", hellaswag="acc_norm", arc_easy="acc_norm", arc_challenge="acc_norm", winogrande="acc",
               lambada_openai="acc")


def primary_metric(task):
    return PRIMARY[task]


# ---------------------------------------------------------------------------------------------- inputs and records

def focus_state(model):
    if model == "qwen3.8-27b":
        return Path(json.loads((ZS / "step0" / "focus_qwen27b_download.json").read_text())["path"])
    if model in LLAMA:
        return G.RUN / "llama_instruct" / "focus_states" / model / "state" / "focus.pt"
    return G.RUN / "focus" / model / "state" / "focus.pt"


def map_path(model, unit):
    if model in LLAMA:
        return G.RUN / "llama_instruct" / model / "maps" / f"flipquant_{unit}.pt"
    return G.REL / model / f"flipquant_{unit}.pt"


def map_sha(model, unit):
    if model in LLAMA:
        rec = json.loads((G.RZ / "results" / "paper_eval" / "llama_instruct" / "llama_instruct.json").read_text())
        return rec["models"][model]["maps"][unit]["map_sha256"]
    for e in json.loads((G.REL / "summary.json").read_text()):
        if e["model"] == model:
            return e["runs"][unit]["map_sha256"]
    raise KeyError(model)


def gptq_record(model):
    if model in LLAMA:
        return G.RUN / "llama_instruct" / model / "gptqdd" / "ppl.json"
    return G.RUN / "gptqdd" / model / "ppl.json"


def codes_file(model):
    if model in LLAMA:
        return ZS / "gptq" / model / "codes_nvfp4_act-fo6.pt"
    return G.RUN / ("ptq" if model in G.HYBRID else "gptqdd") / model / "codes_nvfp4_act-fo6.pt"


def calib_record(model):
    return S.DATA_ROOT / LLAMA[model] / "calibration" / "report.json" if model in LLAMA else S.record(model)


def calib_extension(model):
    if model in LLAMA:
        return G.RUN / "llama_instruct" / model / "records" / "16x64" / "trainer_report.json"
    return S.trainer_report(model)


def gptq_flags(model):
    """Part J's flags (run_sm120.gptq_flags) with the model's record, extension and codes file."""
    return ["--ptq", "gptq", "--gptq-propagate", "bf16", "--gptq-hessian-input", "bf16", "--gptq-calib", "release",
            "--gptq-calib-record", calib_record(model), "--gptq-calib-extension", calib_extension(model),
            "--gptq-block", 128, "--gptq-damp", 0.01, "--gptq-batch", 1 if model == "qwen3.8-27b" else 8,
            "--gptq-cache", codes_file(model)]


def mainrow_record(model, row):
    base = G.RUN / "llama_instruct" / model / "mainrows" if model in LLAMA else G.RUN / "mainrows" / model
    return base / f"{row}.json"


def focus_ppl_record(model):
    return G.RUN / "llama_instruct" / model / "focus" / "ppl.json" if model in LLAMA else G.RUN / "focus" / model / "ppl.json"


def installed_sha(r):
    ad = (r.get("policy") or r).get("adaptive") or {}
    return (ad.get("method") or {}).get("installed_weight_sha256") or ad.get("installed_weight_sha256")


_HASHES = {}


def sha256_file(path):
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def step0_value(*keys):
    v = json.loads((ZS / "step0" / "step0_ok.json").read_text())
    for k in keys:
        v = v[k]
    return v


def verify_inputs(model, row):
    """The file a row reads, re-hashed before its run, against its record (PROTOCOL.md)."""
    if row == "focus":
        want = FOCUS_SHA[model] or step0_value("focus_state_sha256", model)
        got = sha256_file(focus_state(model))
        return [] if got == want else [f"FOCUS state sha256 {got[:12]} != {want[:12]}"]
    if row.startswith("fq-"):
        got, want = sha256_file(map_path(model, row[3:])), map_sha(model, row[3:])
        return [] if got == want else [f"map sha256 {got[:12]} != {want[:12]}"]
    if row == "gptq":
        want = step0_value("codes_file_sha256", model)
        got = sha256_file(codes_file(model))
        return [] if got == want else [f"codes file sha256 {got[:12]} != step 0's {want[:12]}"]
    return []


def launcher(model, module):
    """python -m <module>, or zs6_fq.py <module> for the model main's registry lacks (deviation 1)."""
    if model == "llama3.1-8b-instruct":
        return [G.py(model), HERE / "zs6_fq.py", module]
    return [G.py(model), "-m", module]


def row_args(model, row, fake=False):
    native = ["--mode", "fake"] if fake else NATIVE
    if row == "bf16":
        return ["--mode", "bf16"]
    if row == "nvfp4":
        return native + ["--weight", "nvfp4", "--act", "nvfp4"]
    if row == "fo6":
        return native + ["--weight", "fourover6", "--act", "fourover6"]
    if row == "focus":
        return native + ["--weight", "focus", "--focus", focus_state(model), "--focus-deploy", "--act", "fourover6",
                         "--act-scope", "row"]
    if row == "gptq":
        return native + ["--weight", "nvfp4", "--act", "fourover6", "--any-act", *gptq_flags(model)]
    if row in ("if4", "zou"):
        return ["--mode", "fake", "--weight", "if4" if row == "if4" else "zou_mixfp4", "--act-method", "own", "--unit", "1x16"]
    if row.startswith("fq-"):
        return native + ["--weight", "mixfp4", "--act", "fourover6", "--map", map_path(model, row[3:])]
    raise KeyError(row)


def must(rc, what):
    if rc not in (None, 0):
        G.stop(f"zeroshot6 {what}: rc={rc}")


def kernel_libs(obj):
    """The library sha256 values of kernel-set records: every `kernels` {name: sha256} map, and any sha-named field."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "kernels" and isinstance(v, dict):
                    out.extend(x for x in v.values() if isinstance(x, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return sorted(set(out) | set(G.shas(obj)))


def libraries_ok(obj, libs):
    found = kernel_libs(obj)
    return bool(found) and all(s in libs for s in found), found


# ---------------------------------------------------------------------------------------------- step 0

def step0_codes():
    out = ZS / "step0" / "codes.json"
    if not G.done(out):
        G.log("START zs6_step0_codes (CPU)")
        rc = subprocess.run([G.PY, HERE / "zs6_codes.py", "--out", out], cwd=HERE, env=G.env(),
                            stdout=open(G.RUN / "logs" / "zs6_step0_codes.log", "w"), stderr=subprocess.STDOUT).returncode
        G.log(f"END zs6_step0_codes rc={rc}")
        must(rc, "step 0 codes")


# amendment 1: the parity model is Phi-4 (base Mistral-7B-v0.3's weights are no longer cached)
PARITY = dict(dir="parity_phi4", razer_model="phi4", flipquant_model="phi4-14b", revision=S.REV["phi4-14b"],
              data_root="/home/dev/n16k64_campaign/multimodel/data", fo6=PAPER / "artifacts" / "phi4_fo6",
              fq=PAPER / "artifacts" / "phi4_tc_16x64", map=PAPER / "maps" / "phi4_16x64" / "map.pt")


def step0_parity():
    d = ZS / "step0" / PARITY["dir"]
    rz = [G.PY, RZ_MAIN / "run_lmeval_deploy.py", "--model", PARITY["razer_model"], "--data-root", PARITY["data_root"],
          "--evaluate", "bf16=bf16", "--evaluate", f"fo6=native:{PARITY['fo6']}:auto_stock",
          "--evaluate", f"fq-16x64=native:{PARITY['fq']}:auto",
          "--tasks", PARITY_TASKS, "--limit", 24, "--batch-size", 16, "--out", d / "razer"]
    must(G.run("zs6_parity_phi4_razer", rz, d / "razer" / "report.json", RZ_MAIN,
               dict(ENV, PYTHONPATH=f"{RZ_MAIN / 'sm120'}:{RZ_MAIN}", SM120_BUILD_DIR=str(G.BUILD_V))), "parity razer")
    pols = {"bf16": ["--mode", "bf16"], "fo6": NATIVE + ["--weight", "fourover6", "--act", "fourover6"],
            "fq-16x64": NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", PARITY["map"], "--unit", "16x64"]}
    for pol, q in pols.items():
        out = d / "flipquant" / f"{pol}.json"
        cmd = [G.PY, "-m", "evaluation.downstream", "--model", PARITY["flipquant_model"], "--revision", PARITY["revision"],
               *q, "--tasks", PARITY_TASKS, "--limit", 24, "--out", out]
        must(G.run(f"zs6_parity_phi4_flipquant_{pol}", cmd, out, FQ, ENV), f"parity flipquant {pol}")


def step0_identity():
    ref = ZS / "step0" / "ref" / "adaptive_formats_44f8cea.py"
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_bytes(subprocess.run(["git", "-C", RZ_MAIN, "show", "44f8cea:quantize/adaptive_formats.py"],
                                       capture_output=True, check=True).stdout)
    out = ZS / "step0" / "identity_if4_zou.json"
    if not out.exists():
        must(G.run("zs6_identity_if4_zou", [G.PY, HERE / "zs6_identity.py", "--ref", ref, "--out", out], out, FQ, ENV),
             "identity if4/zou")


def step0_ppl():
    """evaluation.ppl --limit checks against the main-table records: IF4 / Zou (Mistral, 4 windows), FOCUS and GPTQ (all
    six models, 2 windows). The Llama GPTQ runs compute and save the codes the panel loads."""
    jobs = [("mistral-7b", r, ["--mode", "fake", "--weight", w, "--act-method", "own", "--unit", "1x16", "--limit", 4])
            for r, w in (("if4", "if4"), ("zou", "zou_mixfp4"))]
    for model in MODELS:
        jobs.append((model, "focus", NATIVE + ["--weight", "focus", "--focus", focus_state(model), "--focus-deploy",
                                              "--act", "fourover6", "--act-scope", "row", "--paper-convention",
                                              "--limit", 2]))
        jobs.append((model, "gptq", NATIVE + ["--weight", "nvfp4", "--act", "fourover6", "--any-act", "--paper-convention",
                                             *gptq_flags(model), "--limit", 2]))
    for model, row, q in jobs:
        out = ZS / "step0" / "ppl" / model / f"{row}.json"
        cmd = launcher(model, "evaluation.ppl") + ["--model", model, "--revision", REV[model], *q, "--out", out]
        must(G.run(f"zs6_step0_ppl_{model}_{row}", cmd, out, FQ, ENV), f"step 0 ppl {model} {row}")


def step0():
    for d in ("step0", "gptq"):
        (ZS / d).mkdir(parents=True, exist_ok=True)
    step0_codes()
    step0_parity()
    step0_identity()
    dl = ZS / "step0" / "focus_qwen27b_download.json"      # written by zs6_download.py when it has finished
    for _ in range(240):
        if dl.exists():
            break
        time.sleep(30)
    else:
        G.stop("zeroshot6 step 0: the Qwen3.8-27B FOCUS download has not finished in 2 h (zs6_download.py)")
    if not json.loads(dl.read_text()).get("equal_to_part_q"):
        G.stop("zeroshot6 step 0: the downloaded Qwen3.8-27B FOCUS state differs from part Q's record")
    step0_ppl()
    G.log("DONE zeroshot6 step0 runs")


# ---------------------------------------------------------------------------------------------- the panel

def check_record(model, row, path, libs, digests, fake=False):
    r = json.loads(Path(path).read_text())
    bad = []
    if r.get("status") != "complete":
        return ["not complete"]
    pol = r.get("policy") or {}
    if (pol.get("source") or {}).get("revision") != REV[model]:
        bad.append(f"revision {(pol.get('source') or {}).get('revision')}")
    if r["versions"]["lm_eval"] != "0.4.11":
        bad.append(f"lm_eval {r['versions']['lm_eval']}")
    if r["batch_size"] != (8 if model == "qwen3.8-27b" else 16):
        bad.append(f"batch {r['batch_size']}")
    pins = r.get("dataset_pins") or {}
    if pins.get(LAMBADA_PIN[0]) != LAMBADA_PIN[1]:
        bad.append("lambada pin")
    loads = r.get("dataset_loads") or []
    off = [x for x in loads if x["loaded"] not in pins or x["revision"] != pins[x["loaded"]]]
    if off or not any(x["loaded"] == LAMBADA_PIN[0] and x["revision"] == LAMBADA_PIN[1] for x in loads):
        bad.append(f"dataset loads {off[:2]}")
    for t, n in SIZES.items():
        e = (r.get("tasks") or {}).get(t)
        if e is None:
            bad.append(f"{t} missing")
            continue
        if e["examples_evaluated"] != n or len(e["examples"]) != n:
            bad.append(f"{t}: {e['examples_evaluated']} examples")
        if e.get("n_shot") != 0:
            bad.append(f"{t}: n_shot {e.get('n_shot')}")
        if digests.setdefault(t, e.get("sample_digest")) != e.get("sample_digest"):
            bad.append(f"{t}: sample_digest differs from the model's first row")
    if row in FAMILY and not fake:
        nat = pol.get("native") or {}
        if nat.get("kernel") != FAMILY[row] or (nat.get("kernel_set") or {}).get("family") != FAMILY[row]:
            bad.append(f"kernel family {nat.get('kernel')} / {(nat.get('kernel_set') or {}).get('family')} != {FAMILY[row]}")
        if Path(nat.get("build_dir") or "/nonexistent").resolve() != G.BUILD_V.resolve():
            bad.append(f"build_dir {nat.get('build_dir')}")
        ok, found = libraries_ok([nat.get("kernel_set"), (r.get("coverage") or {}).get("kernel_set")], libs)
        if not ok:
            bad.append(f"libraries not from build_V: {found[:3]}")
        cov = r.get("coverage") or {}
        if not cov or cov.get("native_called") != cov.get("native") or cov.get("remaining_bf16_linears_called"):
            bad.append("coverage")
    if row == "gptq":
        want = json.loads(gptq_record(model).read_text())["ptq"]["codes_sha256"]
        if (pol.get("ptq") or {}).get("codes_sha256") != want:
            bad.append(f"codes_sha256 {(pol.get('ptq') or {}).get('codes_sha256')}")
    if row in ("if4", "zou"):
        want = installed_sha(json.loads(mainrow_record(model, row).read_text()))
        if not want or installed_sha(r) != want:
            bad.append(f"installed weight sha256 {installed_sha(r)} != {want}")
    if row == "focus" and (pol.get("focus") or {}).get("state") != str(focus_state(model)):
        bad.append("FOCUS state path")
    if row.startswith("fq-") and (pol.get("map") or {}).get("path") != str(map_path(model, row[3:])):
        bad.append("map path")
    return bad


def run_row(model, row, libs, digests, fake=False):
    out = ZS / ("crosscheck" if fake else "panel") / model / (f"{row}_fake.json" if fake else f"{row}.json")
    if not G.done(out):
        bad = verify_inputs(model, row)
        if bad:
            G.stop(f"zeroshot6 {model} {row}: {bad}")
    cmd = launcher(model, "evaluation.downstream") + ["--model", model, "--revision", REV[model],
                                                      *row_args(model, row, fake), "--tasks", TASKS, "--out", out]
    name = f"zs6_{'xcheck' if fake else 'panel'}_{model}_{row}"
    must(G.run(name, cmd, out, FQ, ENV), f"{model} {row}{' fake' if fake else ''}")
    bad = check_record(model, row, out, libs, digests, fake)
    if bad:
        G.stop(f"zeroshot6 {model} {row}{' fake' if fake else ''}: {bad}")


def panel(models):
    head = subprocess.run(["git", "-C", FQ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", FQ, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if head != FQ_COMMIT or dirty:
        G.stop(f"zeroshot6: flipquant {FQ} is at {head[:7]} (dirty: {bool(dirty)}), not a clean {FQ_COMMIT[:7]}")
    if not (ZS / "step0" / "step0_ok.json").exists() or not step0_value("ok"):
        G.stop("zeroshot6: step 0 has not passed")
    libs = G.manifests()
    for model in models:
        digests = {}
        for row in ROWS:
            run_row(model, row, libs, digests)
        if model == "mistral-7b":                     # the native / fake cross-check
            for row in ("fo6", "fq-16x64"):
                run_row(model, row, libs, digests, fake=True)
        G.log(f"zeroshot6 {model}: all rows done")
    G.log("DONE zeroshot6 panel")


if __name__ == "__main__":
    what = sys.argv[1]
    sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
    if not (G.RUN / "env" / "fast_env_verified.json").exists():
        G.stop("the hybrid models run only in n16k64-fast, after its verification (amendment 3)")
    if what == "step0":
        G.log("zeroshot6 step0")
        step0()
    elif what == "panel":
        models = [m for m in MODELS if not sel or m in sel[0]]
        G.log(f"zeroshot6 panel models {','.join(models)}")
        panel(models)
    else:
        raise SystemExit(f"unknown part {what}")
