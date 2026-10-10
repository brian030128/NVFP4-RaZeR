"""Part O (PROTOCOL.md): round 3 of tab:ptq, on Nemotron-Nano-9B-v2 and Qwen3.8-27B at 16x64.

    python run_ptq_round3.py bf16-pilot --batch B                # O0: BF16 on part N's pilot documents, batch B
    python run_ptq_round3.py bf16 --batch <model>:<b>,...         # O0: the BF16 GSM8K runs (all 1,319 problems)
    python run_ptq_round3.py gptq [--models M,..]                 # O2: GPTQ with activation quantization on, + PPL
    python run_ptq_round3.py e0m3 [--models M,..]                 # O3: the all-E0M3 map and its GPTQ codes
    python run_ptq_round3.py train --source rtn|gptq [--models M,..] [--env release|fast]   # O3: TM-OPT+TC, hooked
    python run_ptq_round3.py ppl3 [--models M,..]                 # O3: native PPL, the retrained map on GPTQ candidates
    python run_ptq_round3.py gsm8k [--models M,..]                # O4: GSM8K, the four new rows, batch 64

Every GPU job goes through run_gpu.run / run_gsm8k.run_ts: one at a time on an idle GPU, START / END in queue.log, a
finished record is not run again. GSM8K is part N's command (run_gsm8k_lmeval) with this part's quantization flags.
GPTQ code files go under --codes-root (default RUN/ptq_round3/codes); a GPTQ job starts only if the disk holds its file
plus a 10 GB margin.
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
import run_gsm8k as M  # noqa: E402
import run_gsm8k_lmeval as NL  # noqa: E402
import run_sm120 as S  # noqa: E402

MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
ROOT = G.RUN / "ptq_round3"
UNIT = "16x64"
NATIVE = [*G.NATIVE, "--kernel-set", "auto"]
FILE_GB = {"nemotron-nano-9b-v2": 5, "qwen3.8-27b": 16}   # one GPTQ code file, rounded up (part H: 4.3-4.8 / 13.7-15.2 GB)
MARGIN_GB = 10
# the trainer's interpreter: the release maps' env (their records: n16k64), or n16k64-fast
TRAIN_PY = {"release": G.PY, "fast": G.PY_FAST}
# O2's rows: part H's quantization flags (run_sm120.GPTQ_ROWS); PPL as part H (FourOverSix activations: the paper
# convention; NVFP4: per-token NVFP4 scales), GSM8K as part N (run_gsm8k.quant: --act-scope row)
ROWS = {"nvfp4": ["--weight", "nvfp4", "--act", "nvfp4"],
        "fo6": ["--weight", "fourover6", "--act", "fourover6"],
        "fq-16x64": ["--weight", "mixfp4", "--act", "fourover6", "--map", "@release"]}
PPL_CONV = {"nvfp4": ["--act-scope", "row"], "fo6": ["--paper-convention"], "fq-16x64": ["--paper-convention"]}


def release_map(model):
    return G.REL / model / f"flipquant_{UNIT}.pt"


def codes(root, model, row):
    """GPTQ code file of O2's row or O3's E0M3 candidate ("e0m3")."""
    return Path(root) / model / f"codes_{row}.pt"


def gptq_flags(model, cache=None, candidates=None):
    """O2 / O3: part H's GPTQ (fixed RTN grid, block 128, damp 0.01, no act-order, the release 256 x 512 fit set), with
    activation quantization on during calibration: flipquant's defaults, spelled out (the Hessian from the per-token
    quantized input; earlier layers run as they are evaluated, quantized weights and per-token activations)."""
    f = ["--ptq", "gptq", "--gptq-propagate", "quantized", "--gptq-hessian-input", "quantized", "--gptq-calib", "release",
         "--gptq-calib-record", S.record(model), "--gptq-calib-extension", S.trainer_report(model),
         "--gptq-block", 128, "--gptq-damp", 0.01, "--gptq-batch", 1 if model == "qwen3.8-27b" else 8]
    if cache is not None:
        f += ["--gptq-cache", cache]
    if candidates is not None:
        f += ["--gptq-candidates", *candidates]
    return f


def row_flags(model, row):
    return [str(release_map(model)) if a == "@release" else a for a in ROWS[row]]


def disk_ok(root, model):
    Path(root).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    if free < FILE_GB[model] + MARGIN_GB:
        G.stop(f"round3: {free:.0f} GB free under {root}, a {model} code file needs {FILE_GB[model]} GB + "
               f"{MARGIN_GB} GB margin")


# ----------------------------------------------------------------------------------------------------------------- O0
def gsm8k_cmd(model, quant, batch, out, doc_ids=None):
    """Part N's command (run_gsm8k_lmeval.cmd) with the quantization flags ``quant``."""
    c = [G.py(model), "-u", "-m", "evaluation.downstream", "--model", model, "--revision", S.REV[model], *quant,
         "--tasks", NL.TASK, "--apply-chat-template", *NL.THINK_OFF[model], "--batch-size", batch, "--gen-lengths",
         "--samples-out", Path(out).with_suffix(""), "--out", out]
    if doc_ids:
        c += ["--doc-ids", doc_ids]
    return c


class GpuMemory:
    """Device memory in use (nvidia-smi, MiB), sampled every second while a job runs; the maximum."""

    def __init__(self):
        self.peak, self.samples, self._stop = 0, 0, threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                used, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
                self.peak, self.total, self.samples = max(self.peak, used), total, self.samples + 1
            except (ValueError, IndexError):
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.t.join()


def bf16_pilot(batch):
    for model in MODELS:
        out = ROOT / "bf16" / "pilot" / model / f"pilot_b{batch}.json"
        with GpuMemory() as mem:
            rc = M.run_ts(f"round3_bf16_pilot_{model}_b{batch}",
                          gsm8k_cmd(model, ["--mode", "bf16"], batch, out, NL.ROOT / "pilot" / model / "doc_ids.json"), out)
        if rc is not None:
            out.with_suffix(".mem.json").write_text(json.dumps(dict(rc=rc, batch=batch, peak_mib=mem.peak,
                                                                    total_mib=getattr(mem, "total", None),
                                                                    samples=mem.samples), indent=1) + "\n")
            print(model, "batch", batch, "rc", rc, "nvidia-smi peak MiB", mem.peak, flush=True)


def bf16(batches):
    for model in MODELS:
        out = ROOT / "bf16" / model / "bf16.json"
        S.must(M.run_ts(f"round3_bf16_{model}", gsm8k_cmd(model, ["--mode", "bf16"], batches[model], out), out),
               f"round3 bf16 gsm8k {model}")


# ----------------------------------------------------------------------------------------------------------------- O2
def gptq(models, root):
    for model in models:
        for row in ROWS:
            out = ROOT / model / f"gptq_{row}.json"
            if G.done(out):
                continue
            disk_ok(root, model)
            q = [*NATIVE, *row_flags(model, row), *PPL_CONV[row], *gptq_flags(model, codes(root, model, row))]
            S.must(G.run(f"round3_gptq_{model}_{row}", S.ppl_cmd(model, q, out), out, S.FQ2), f"round3 gptq {model} {row}")


# ----------------------------------------------------------------------------------------------------------------- O3
def all_e0m3_map(model):
    """Every tile of the release 16x64 map's modules E0M3: the E0M3 candidate's grid for flipquant's mixfp4 GPTQ."""
    out = ROOT / model / f"all_e0m3_{UNIT}.pt"
    if out.exists():
        return out
    code = r"""
import sys, torch
from flipquant import maps as M
tiles, unit, meta = M.load(sys.argv[1])
M.save(sys.argv[2], {n: torch.ones_like(t, dtype=torch.bool) for n, t in tiles.items()}, unit,
       dict(method="all-E0M3 (part O, round 3: the E0M3 candidate's grid)", model=meta.get("model"),
            registry_key=meta.get("registry_key"), revision=meta.get("revision"), rotate="none",
            modules_from=sys.argv[1]))
back, unit2, _ = M.load(sys.argv[2])
assert unit2 == unit and set(back) == set(tiles) and all(bool(back[n].all()) and back[n].shape == tiles[n].shape for n in tiles)
print("ALL-E0M3", len(back), "modules", sum(t.numel() for t in back.values()), "tiles")
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([G.py(model), "-c", code, release_map(model), out], cwd=S.FQ2, capture_output=True, text=True,
                       env=G.env({"CUDA_VISIBLE_DEVICES": ""}))
    if r.returncode:
        G.stop(f"round3 all-E0M3 map {model}: {r.stderr[-2000:]}")
    print(r.stdout.strip(), flush=True)
    return out


def e0m3(models, root):
    for model in models:
        amap = all_e0m3_map(model)
        out = ROOT / model / "gptq_e0m3.json"
        if G.done(out):
            continue
        disk_ok(root, model)
        cmd = [G.py(model), "-m", "calibration.gptq_codes", "--model", model, "--revision", S.REV[model],
               "--weight", "mixfp4", "--act", "fourover6", "--act-scope", "row", "--map", amap,
               *gptq_flags(model, codes(root, model, "e0m3")), "--out", out]
        S.must(G.run(f"round3_e0m3_{model}", cmd, out, S.FQ2), f"round3 e0m3 codes {model}")


def trained_map(model, source):
    return ROOT / model / f"fq-{UNIT}_{source}cand.pt"


def train(models, root, source, env):
    for model in models:
        if source == "rtn":
            cand = ["rtn", "rtn"]
        else:
            cand = [codes(root, model, "fo6"), codes(root, model, "e0m3")]
            for p in cand:
                if not Path(p).exists():
                    G.stop(f"round3 train {model}: no candidate file {p}")
        out = trained_map(model, source)
        cmd = [TRAIN_PY[env], "-m", "calibration.tmopt_ext", "train", "--model", model, "--unit", UNIT,
               "--data-root", S.DATA_ROOT, "--candidates-e2m1", cand[0], "--candidates-e0m3", cand[1], "--out", out]
        S.must(G.run(f"round3_train_{model}_{source}_{env}", cmd, out.with_suffix(".json"), S.FQ2),
               f"round3 train {model} {source}")


def ppl3(models, root):
    for model in models:
        out = ROOT / model / f"gptqcand_fq-{UNIT}.json"
        cand = (codes(root, model, "fo6"), codes(root, model, "e0m3"))
        q = [*NATIVE, "--weight", "mixfp4", "--act", "fourover6", "--map", trained_map(model, "gptq"),
             "--paper-convention", *gptq_flags(model, candidates=cand)]
        S.must(G.run(f"round3_ppl3_{model}", S.ppl_cmd(model, q, out), out, S.FQ2), f"round3 ppl3 {model}")


# ----------------------------------------------------------------------------------------------------------------- O4
def gsm8k_rows(model, root):
    rows = {f"gptq_{r}": [*NATIVE, *row_flags(model, r), "--act-scope", "row", *gptq_flags(model, codes(root, model, r))]
            for r in ROWS}
    rows[f"gptqcand_fq-{UNIT}"] = [*NATIVE, "--weight", "mixfp4", "--act", "fourover6", "--map", trained_map(model, "gptq"),
                                   "--act-scope", "row",
                                   *gptq_flags(model, candidates=(codes(root, model, "fo6"), codes(root, model, "e0m3")))]
    return rows


def gsm8k(models, root):
    for model in models:
        for name, q in gsm8k_rows(model, root).items():
            out = ROOT / "gsm8k" / model / f"{name}.json"
            S.must(M.run_ts(f"round3_gsm8k_{model}_{name}", gsm8k_cmd(model, q, 64, out), out),
                   f"round3 gsm8k {model} {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("bf16-pilot", "bf16", "gptq", "e0m3", "train", "ppl3", "gsm8k"))
    ap.add_argument("--batch", default="64", help="bf16-pilot: the batch size; bf16: <model>:<batch>,...")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--codes-root", type=Path, default=ROOT / "codes")
    ap.add_argument("--source", choices=("rtn", "gptq"), default=None, help="train: the candidates")
    ap.add_argument("--env", choices=tuple(TRAIN_PY), default="release", help="train: the trainer's env")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    assert set(models) <= set(MODELS), models
    if a.what == "bf16-pilot":
        G.log(f"round3 bf16 pilot (O0) batch {a.batch}")
        bf16_pilot(int(a.batch))
    elif a.what == "bf16":
        bs = dict(x.split(":") for x in a.batch.split(","))
        assert set(bs) == set(MODELS), bs
        G.log(f"round3 bf16 gsm8k (O0) batches {bs}")
        bf16({m: int(b) for m, b in bs.items()})
    elif a.what == "gptq":
        G.log(f"round3 gptq (O2) models {models} codes {a.codes_root}")
        gptq(models, a.codes_root)
    elif a.what == "e0m3":
        G.log(f"round3 e0m3 candidate (O3) models {models} codes {a.codes_root}")
        e0m3(models, a.codes_root)
    elif a.what == "train":
        assert a.source, "--source rtn|gptq"
        G.log(f"round3 train (O3) source {a.source} env {a.env} models {models}")
        train(models, a.codes_root, a.source, a.env)
    elif a.what == "ppl3":
        G.log(f"round3 ppl3 (O3) models {models}")
        ppl3(models, a.codes_root)
    else:
        G.log(f"round3 gsm8k (O4) models {models}")
        gsm8k(models, a.codes_root)
