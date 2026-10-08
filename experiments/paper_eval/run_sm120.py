"""The remaining SM120 paper tables (PROTOCOL.md, parts G-L) on flipquant branch paper-sm120-runs: the co-author's
map-ablation, ptq-combo, IF4, MIXFP4 and FOCUS branches merged onto main 120173a and adapted to the paper's final
settings (the release maps, the release 256 x 512 fit set, n16k64 / n16k64-fast, build_V with --kernel-set auto, the
paper convention). One job at a time on an idle GPU (run_gpu.run); records under RUN/<part>/, START/END in queue.log.

    python run_sm120.py ablation [--models=a,b]     # G: tab:ablation (Nemotron-Nano-9B-v2, Qwen3.8-27B x 3 units)

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
    else:
        raise SystemExit(f"unknown part {what}")
    G.log(f"DONE {what}")
