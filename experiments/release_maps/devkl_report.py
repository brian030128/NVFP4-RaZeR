"""Dev-KL records: dev_kl.json, dev_kl.png and the tables of NOTE.md (written by hand around them), from the trainer's
report.json of each run (devkl_run.py), with the checks:
- N's map_epoch005.pt is the release map (the release run's run-map sha256, and tile by tile);
- O's final map.pt is the paper map (experiments/paper/maps.sha256.json).

    python devkl_report.py OUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

B = Path("/home/dev/n16k64_campaign/fqrel")
RUNS = B / "devkl"
REL = {"llama8b": "llama3.1-8b", "phi4": "phi4-14b"}
PAPER = json.loads(Path("/home/dev/n16k64_campaign/fqopt/wt/calibration/tmopt/experiments/paper/maps.sha256.json")
                   .read_text())["maps"]
STEPS = {"N": 32, "O": 16}               # steps per epoch: 256 or 128 windows, 8 per step


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tiles_equal(a, b):
    x, y = torch.load(a, map_location="cpu", weights_only=True), torch.load(b, map_location="cpu", weights_only=True)
    return set(x) == set(y) and all(x[n].dtype == y[n].dtype and torch.equal(x[n], y[n]) for n in x)


def records_identical(a, b):
    """Every tensor record of two torch.save files byte for byte. The archive prefix is the file's stem
    (map_epoch005/ vs map/), so the files' own sha256 differ even for the same map."""
    import zipfile
    za, zb = zipfile.ZipFile(a), zipfile.ZipFile(b)
    ra = {n.split("/", 1)[1]: n for n in za.namelist() if "/data/" in n}
    rb = {n.split("/", 1)[1]: n for n in zb.namelist() if "/data/" in n}
    return set(ra) == set(rb) and all(za.read(ra[k]) == zb.read(rb[k]) for k in ra)


def series(key, setting):
    d = RUNS / f"{key}_16x64_{setting}"
    r = json.loads((d / "report.json").read_text())
    points = [dict(step=0, epoch=0, dev_kl=r["initial_dev"]["kl"], dev_ce=r["initial_dev"]["ce"], train_kl=None,
                   e0m3_tiles=0, dev_seconds=None)]
    for e in r["epochs"]:
        if "dev_kl" in e:
            points.append(dict(step=(e["epoch"] + 1) * STEPS[setting], epoch=e["epoch"] + 1, dev_kl=e["dev_kl"],
                               dev_ce=e["dev_ce"], train_kl=e["train_kl"], e0m3_tiles=e["e0m3_units"],
                               dev_seconds=e["dev_seconds"]))
    res = r["resources"]
    return dict(points=points, final_map_sha256=r["map_sha256"], status=r["status"],
                teacher_storage=r.get("teacher_storage"), total_seconds=res["total_seconds"],
                setup_seconds=r["setup_seconds"], training_seconds=r["training_seconds"],
                gpu_peak_allocated_gib=res["gpu_peak_allocated_gib"][0], cpu_peak_rss_gib=res["cpu_peak_rss_gib"],
                initial_dev_seconds=next((p["seconds"] for p in res["phases"]["phases"] if p["name"] == "initial_dev_eval"), None),
                epoch_seconds=[e["epoch_seconds"] for e in r["epochs"]], development=r.get("development"),
                args={k: r["args"].get(k) for k in ("epochs", "eval_every", "teacher_topk", "fit_windows", "dev_backend",
                                                    "lr", "init_logit", "batch", "accum", "no_dev")})


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(models={})
    for key in ("llama8b", "phi4"):
        m = dict(N=series(key, "N"), O=series(key, "O"))
        release = json.loads((Path("/home/dev/flipquant_release") / REL[key] / "records" / "16x64" / "run.json").read_text())
        n5 = RUNS / f"{key}_16x64_N" / "map_epoch005.pt"
        rel_map = B / "runs" / f"{REL[key]}_16x64" / "run" / "map.pt"
        m["checks"] = dict(
            N_epoch5_sha256=sha(n5), release_run_map_sha256=release["run_map_sha256"],
            release_map_file_sha256_check=sha(rel_map) == release["run_map_sha256"],
            N_epoch5_is_release_map=tiles_equal(n5, rel_map) and records_identical(n5, rel_map),
            O_final_sha256=m["O"]["final_map_sha256"], paper_map_sha256=PAPER[f"{key}_16x64"]["map_sha256"],
            O_final_is_paper_map=m["O"]["final_map_sha256"] == PAPER[f"{key}_16x64"]["map_sha256"])
        rec["models"][key] = m
    rec["disjointness"] = json.loads((RUNS / "disjointness.json").read_text())
    (out / "dev_kl.json").write_text(json.dumps(rec, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("llama8b", "phi4")):
        for s, style in (("N", "-o"), ("O", "-s")):
            pts = rec["models"][key][s]["points"]
            ax.plot([p["step"] for p in pts], [p["dev_kl"] for p in pts], style, ms=3,
                    label=f"{s}: " + ("256 windows, top-1000, 10 epochs" if s == "N" else "128 windows, full vocab, 20 epochs"))
        ax.axvline(160, color="grey", lw=0.8, ls=":")
        ax.set_title({"llama8b": "Llama-3.1-8B 16x64", "phi4": "Phi-4 16x64"}[key])
        ax.set_xlabel("optimizer step (8 sequences each)")
        ax.set_ylabel("dev KL (192 windows, full vocabulary)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dev_kl.png", dpi=110)

    for key in ("llama8b", "phi4"):
        m = rec["models"][key]
        print(f"\n{key}: checks {m['checks']}")
        o = {p["step"]: p for p in m["O"]["points"]}

        def kl(x):
            return "—" if x is None else f"{x:.5f}"
        print("| step | N epoch | N dev KL | N train KL (top-1000) | N E0M3 | O epoch | O dev KL | O train KL | O E0M3 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in m["N"]["points"]:
            q = o.get(p["step"])
            right = f"{q['epoch']} | {kl(q['dev_kl'])} | {kl(q['train_kl'])} | {q['e0m3_tiles']:,}" if q else "| | |"
            print(f"| {p['step']} | {p['epoch']} | {kl(p['dev_kl'])} | {kl(p['train_kl'])} | {p['e0m3_tiles']:,} | {right} |")

if __name__ == "__main__":
    main(sys.argv[1])
