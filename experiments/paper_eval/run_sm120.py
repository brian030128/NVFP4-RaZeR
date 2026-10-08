"""The remaining SM120 paper tables (PROTOCOL.md, parts G-L) on flipquant branch paper-sm120-runs: the co-author's
map-ablation, ptq-combo, IF4, MIXFP4 and FOCUS branches merged onto main 120173a and adapted to the paper's final
settings (the release maps, the release 256 x 512 fit set, n16k64 / n16k64-fast, build_V with --kernel-set auto, the
paper convention). One job at a time on an idle GPU (run_gpu.run); records under RUN/<part>/, START/END in queue.log.

    python run_sm120.py ablation [--models=a,b]     # G: tab:ablation (Nemotron-Nano-9B-v2, Qwen3.8-27B x 3 units)
    python run_sm120.py hadamard [--models=a,b]     # H (part): tab:ptq's Hadamard rows (16x64)
    python run_sm120.py gptq [--models=a,b]         # H (part) + J: GPTQ rows (16x64) and GPTQ-double-dagger, BF16 propagation
    python run_sm120.py focus [--models=a,b]        # K: FOCUS (8 steps on the release fit set), deployed natively
    python run_sm120.py mainrows [--models=a,b]     # I: IF4 and MixFP4 (Zou), fake, the methods' own rules, 6 models

A job whose record exists and is complete is skipped; a failed job stops the part (exit 1, STOP in queue.log).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402

FQ2 = Path("/home/dev/n16k64_campaign/sm120runs/wt")              # flipquant paper-sm120-runs
DATA_ROOT = Path("/home/dev/n16k64_campaign/fqrel/tmopt_data")    # the release runs' calibration records
MKEY = {"qwen3-1.7b": "qwen3_1p7b", "qwen3-8b": "qwen3_8b", "mistral-7b": "mistral7b_ins",
        "nemotron-nano-9b-v2": "nemotron9b", "phi4-14b": "phi4", "qwen3.8-27b": "qwen27b"}
ABLATION_MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
ABLATION_MAPS = ("random0", "random1", "random2", "act", "oneshot")
# the release maps' revisions (meta["revision"]; the FOCUS runbook's table), passed explicitly to every new job
REV = {"qwen3-1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "qwen3-8b": "b968826d9c46dd6066d109eabc6255188de91218",
       "mistral-7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",
       "nemotron-nano-9b-v2": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
       "phi4-14b": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25", "qwen3.8-27b": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
ROT = ["--rotate", "hadamard", "--rotate-block", "16"]


def record(model):
    """The reference calibration record (the release maps' first 128 fit windows)."""
    return DATA_ROOT / MKEY[model] / "calibration" / "report.json"


def trainer_report(model):
    """A release run's trainer report: its fit_extension records the other 128 windows (the same for every unit)."""
    return G.REL / model / "records" / "16x64" / "trainer_report.json"


def native_ppl(model, map_path, out):
    return [G.py(model), "-m", "evaluation.ppl", "--model", model, *G.NATIVE, "--kernel-set", "auto",
            "--weight", "mixfp4", "--act", "fourover6", "--map", map_path, "--paper-convention", "--out", out]


def must(rc, what):
    if rc not in (None, 0):
        G.stop(f"{what}: rc={rc}")


def check_g0():
    """G0: paper-sm120-runs evaluates a release map as main did (part F): per-window NLL bit for bit."""
    model, u = "nemotron-nano-9b-v2", "16x64"
    out = G.RUN / "ablation" / "g0" / f"{model}_fq-{u}.json"
    must(G.run(f"abl_g0_{model}_fq-{u}", native_ppl(model, G.REL / model / f"flipquant_{u}.pt", out), out, FQ2),
         "ablation G0 evaluation")
    new, ref = (json.loads(Path(f).read_text())["results"] for f in (out, G.RUN / "pplfast" / model / f"fq-{u}.json"))
    bad = [c for c in ("wiki", "c4") if new[c]["nll"] != ref[c]["nll"]]
    if bad:
        G.stop(f"G0: paper-sm120-runs differs from part F on {bad} ({model} FlipQuant {u}); F's records are not reused")
    G.log(f"G0 ok: {model} FlipQuant {u} per-window NLL equal to part F's on wiki and c4")


def ablation(models):
    """G: per model and unit, the k_l-matched baseline maps (random seeds 0/1/2, activation-weighted on the release
    fit set, one-shot TM-OPT+TC gradient with the release settings), then each map's native PPL."""
    root = G.RUN / "ablation"
    check_g0()
    for model in models:
        for u in G.UNITS:
            match = G.REL / model / f"flipquant_{u}.pt"
            d = root / model / u
            for s in (0, 1, 2):
                out = d / f"random{s}.pt"
                must(G.run(f"abl_map_{model}_{u}_random{s}",
                           [G.py(model), "-m", "calibration.baseline_maps", "--method", "random", "--seed", s,
                            "--model", model, "--match", match, "--out", out], out.with_suffix(".json"), FQ2),
                     f"ablation map {model} {u} random{s}")
            out = d / "act.pt"
            must(G.run(f"abl_map_{model}_{u}_act",
                       [G.py(model), "-m", "calibration.baseline_maps", "--method", "act", "--model", model,
                        "--match", match, "--calib-record", record(model), "--calib-extension", trainer_report(model),
                        "--batch", 1 if model == "qwen3.8-27b" else 8, "--out", out], out.with_suffix(".json"), FQ2),
                 f"ablation map {model} {u} act")
            out = d / "oneshot.pt"
            must(G.run(f"abl_map_{model}_{u}_oneshot",
                       [G.py(model), "-m", "calibration.tmopt_ext", "oneshot", "--model", model, "--unit", u,
                        "--match", match, "--data-root", DATA_ROOT, "--out", out], out.with_suffix(".json"), FQ2),
                 f"ablation map {model} {u} oneshot")
            for m in ABLATION_MAPS:
                out = d / f"ppl_{m}.json"
                must(G.run(f"abl_ppl_{model}_{u}_{m}", native_ppl(model, d / f"{m}.pt", out), out, FQ2),
                     f"ablation ppl {model} {u} {m}")


def ppl_cmd(model, quant, out):
    """evaluation.ppl for ``quant`` (the policy's flags, convention included), the release revision."""
    return [G.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", REV[model], *quant, "--out", out]


def hadamard(models):
    """H, Hadamard rows (block 16), 16x64: NVFP4 (per-token NVFP4 scales), FourOverSix and FlipQuant with a map
    retrained in the rotated basis (calibration.tmopt_ext train --rotate, the release settings)."""
    root = G.RUN / "ptq"
    for model in models:
        d = root / model
        rmap = d / "fq-16x64_hadamard.pt"
        must(G.run(f"ptq_train_{model}_hadamard16",
                   [G.py(model), "-m", "calibration.tmopt_ext", "train", "--model", model, "--unit", "16x64", *ROT,
                    "--data-root", DATA_ROOT, "--out", rmap], rmap.with_suffix(".json"), FQ2),
             f"ptq {model} rotated map")
        native = [*G.NATIVE, "--kernel-set", "auto"]
        rows = {"nvfp4": native + ["--weight", "nvfp4", "--act", "nvfp4", "--act-scope", "row", *ROT],
                "fo6": native + ["--weight", "fourover6", "--act", "fourover6", "--paper-convention", *ROT],
                "fq-16x64": native + ["--weight", "mixfp4", "--act", "fourover6", "--map", rmap, "--paper-convention",
                                      *ROT]}
        for fmt, q in rows.items():
            out = d / f"hadamard_{fmt}.json"
            must(G.run(f"ptq_ppl_{model}_hadamard_{fmt}", ppl_cmd(model, q, out), out, FQ2), f"ptq {model} hadamard {fmt}")


def gptq_flags(model, cache):
    """GPTQ with the user's settings: BF16 propagation (every activation quantizer off during calibration), the
    unquantized Hessian input, the release 256 x 512 fit set, block 128, damp 0.01, no act-order."""
    return ["--ptq", "gptq", "--gptq-propagate", "bf16", "--gptq-hessian-input", "bf16", "--gptq-calib", "release",
            "--gptq-calib-record", record(model), "--gptq-calib-extension", trainer_report(model),
            "--gptq-block", 128, "--gptq-damp", 0.01, "--gptq-batch", 1 if model == "qwen3.8-27b" else 8,
            "--gptq-cache", cache]


GPTQ_ROWS = {   # name: (quantization flags, codes file name); nvfp4-fo6 is the main table's GPTQ-double-dagger
    "nvfp4": (["--weight", "nvfp4", "--act", "nvfp4", "--act-scope", "row"], "codes_nvfp4_act-nvfp4.pt"),
    "nvfp4-fo6": (["--weight", "nvfp4", "--act", "fourover6", "--any-act", "--paper-convention"],
                  "codes_nvfp4_act-fo6.pt"),
    "fo6": (["--weight", "fourover6", "--act", "fourover6", "--paper-convention"], "codes_fo6.pt"),
    "fq-16x64": (["--weight", "mixfp4", "--act", "fourover6", "--map", "@16x64", "--paper-convention"],
                 "codes_fq-16x64.pt")}


def codes_digest(path):
    """sha256 over a GPTQ codes file's code tensors (flipquant.gptq's saved format), module by module."""
    import hashlib
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for n in sorted(obj["codes"]):
        c = obj["codes"][n]
        h.update(n.encode())
        for k in ("q", "s", "g"):
            h.update(c[k].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(c["fmt"].numpy().tobytes() if torch.is_tensor(c["fmt"]) else str(c["fmt"]).encode())
    return h.hexdigest()


def gptq(models):
    """H's GPTQ rows (hybrids, 16x64) and J (GPTQ-double-dagger: NVFP4 codes with FourOverSix per-token activations,
    all six models). For the hybrids the NVFP4 codes are computed twice, inside an NVFP4-activation run and inside a
    FourOverSix-activation run (separate caches), and must be equal (the user's check of BF16 propagation)."""
    for model in models:
        hybrid = model in ABLATION_MODELS
        d = G.RUN / ("ptq" if hybrid else "gptqdd") / model
        rows = list(GPTQ_ROWS) if hybrid else ["nvfp4-fo6"]
        for row in rows:
            q, codes = GPTQ_ROWS[row]
            q = [*G.NATIVE, "--kernel-set", "auto"] + [G.REL / model / "flipquant_16x64.pt" if a == "@16x64" else a for a in q]
            out = d / f"gptq_{row}.json"
            must(G.run(f"gptq_{model}_{row}", ppl_cmd(model, [*q, *gptq_flags(model, d / codes)], out), out, FQ2),
                 f"gptq {model} {row}")
        if hybrid:
            a, b = (d / GPTQ_ROWS[r][1] for r in ("nvfp4", "nvfp4-fo6"))
            da, db = codes_digest(a), codes_digest(b)
            ra, rb = (json.loads((d / f"gptq_{r}.json").read_text()) for r in ("nvfp4", "nvfp4-fo6"))
            sa, sb = ((r.get("ptq") or {}).get("codes_sha256") for r in (ra, rb))
            if da != db or sa != sb:
                G.stop(f"gptq {model}: the NVFP4 codes differ between NVFP4 and FourOverSix activations "
                       f"(files {da[:12]} / {db[:12]}, reports {str(sa)[:12]} / {str(sb)[:12]})")
            G.log(f"GPTQ codes check ok: {model} NVFP4 codes equal under NVFP4 / FourOverSix activations ({da[:16]})")
            # J for the hybrid model: the FourOverSix-activation NVFP4 run is the GPTQ-double-dagger cell
            jd = G.RUN / "gptqdd" / model
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "ppl.json").write_text((d / "gptq_nvfp4-fo6.json").read_text())
        else:
            (d / "ppl.json").write_text((d / "gptq_nvfp4-fo6.json").read_text())


FOCUS_COMMON = ["--epochs", 1, "--batch", 32, "--lr-scale", "5e-3", "--lr-sub", "1e-3", "--topk", 1000, "--num-sub", 2,
                "--init-q", 6, "--act", "fourover6", "--act-scope", "row", "--seed", 42, "--ppl-datasets", ""]


def focus(models):
    """K: FOCUS with its own settings (FOCUS_PAPER_RUNBOOK.md) on the release 256 x 512 fit set (8 optimizer steps),
    micro-batch 8 halved on OOM (the global batch stays 32; recorded), deployed as NVFP4 codes on the sm120 path with
    FourOverSix per-token activations. Qwen3.8-27B is not run (FOCUS does not fit one 96 GB GPU; the user accepted)."""
    for model in models:
        if model == "qwen3.8-27b":
            G.log("NOTE focus qwen3.8-27b: not run (does not fit one GPU; TBD, accepted by the user)")
            continue
        d = G.RUN / "focus" / model
        state = d / "state"
        rep_ = state / "report.json"
        if not G.done(rep_):
            for micro in (8, 4, 2, 1):
                if state.exists():
                    import shutil
                    shutil.rmtree(state)
                cmd = [G.py(model), "-m", "calibration.train_focus", "--model", model, "--revision", REV[model],
                       "--data", "release", "--calib-record", record(model), "--calib-extension", trainer_report(model),
                       *FOCUS_COMMON, "--micro", micro, "--out", state]
                rc = G.run(f"focus_train_{model}_micro{micro}", cmd, rep_, FQ2)
                if rc in (None, 0):
                    break
                logf = (G.RUN / "logs" / f"focus_train_{model}_micro{micro}.log").read_text(errors="replace")
                if "OutOfMemoryError" not in logf and "CUDA out of memory" not in logf:
                    must(rc, f"focus train {model}")
                G.log(f"NOTE focus {model}: OOM at micro-batch {micro}; halving (global batch 32 unchanged)")
            else:
                G.stop(f"focus {model}: OOM even at micro-batch 1")
        out = d / "ppl.json"
        q = [*G.NATIVE, "--kernel-set", "auto", "--weight", "focus", "--focus", state / "focus.pt", "--focus-deploy",
             "--act", "fourover6", "--act-scope", "row", "--paper-convention"]
        must(G.run(f"focus_ppl_{model}", ppl_cmd(model, q, out), out, FQ2), f"focus ppl {model}")


def mainrows(models):
    """I: IF4 (Cook et al.) and MixFP4 (Zou et al.), fake, each method's own weight and activation rules at 1x16 with
    its own E2M1 and per-token activation scales (--act-method own)."""
    root = G.RUN / "mainrows"
    for model in models:
        for weight, name in (("if4", "if4"), ("zou_mixfp4", "zou")):
            out = root / model / f"{name}.json"
            q = ["--mode", "fake", "--weight", weight, "--act-method", "own", "--unit", "1x16"]
            rc = G.run(f"main_{model}_{name}", ppl_cmd(model, q, out), out, FQ2)
            if rc not in (None, 0):
                logf = (G.RUN / "logs" / f"main_{model}_{name}.log").read_text(errors="replace")
                if "OutOfMemoryError" in logf or "CUDA out of memory" in logf:
                    G.log(f"NOTE main {model} {name}: OOM (the cell stays empty; reported)")
                    continue
                must(rc, f"main {model} {name}")


if __name__ == "__main__":
    what = sys.argv[1]
    sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
    if not (G.RUN / "env" / "fast_env_verified.json").exists():
        G.stop("the hybrid models run only in n16k64-fast, after its verification (amendment 3)")
    G.RUN.mkdir(parents=True, exist_ok=True)
    if what == "ablation":
        models = [m for m in ABLATION_MODELS if not sel or m in sel[0]]
        G.log(f"ablation models {','.join(models)}")
        ablation(models)
    elif what == "hadamard":
        models = [m for m in ABLATION_MODELS if not sel or m in sel[0]]
        G.log(f"hadamard models {','.join(models)}")
        hadamard(models)
    elif what == "gptq":
        models = [m for m in G.MODELS if not sel or m in sel[0]]
        G.log(f"gptq models {','.join(models)}")
        gptq(models)
    elif what == "focus":
        models = [m for m in G.MODELS if not sel or m in sel[0]]
        G.log(f"focus models {','.join(models)}")
        focus(models)
    elif what == "mainrows":
        models = [m for m in G.MODELS if not sel or m in sel[0]]
        G.log(f"mainrows models {','.join(models)}")
        mainrows(models)
    else:
        raise SystemExit(f"unknown part {what}")
    G.log(f"DONE {what}")
