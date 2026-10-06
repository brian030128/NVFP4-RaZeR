"""Dev-KL records: dev_kl.json, paired.json, ppl.json, dev_kl.png and the tables of NOTE.md (written by hand around
them), from the trainer's report.json of each run (devkl_run.py) and the PPL reports, with the checks:
- N's map_epoch005.pt is the release map (tile by tile and every tensor record; the file names differ, so the files'
  sha256 cannot be compared);
- O's final map.pt is the paper-setting map: the paper map (experiments/paper/maps.sha256.json) for Llama and Phi-4,
  the committed maps/qwen3-1.7b/flipquant_16x64.pt for Qwen3-1.7B (its recorded run-map sha256, and tile by tile).

    python devkl_report.py OUT_DIR
"""
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import torch

B = Path("/home/dev/n16k64_campaign/fqrel")
W = Path("/home/dev/n16k64_campaign/fqopt/wt")
RUNS = B / "devkl"
MODELS = ("llama8b", "phi4", "qwen3_1p7b")
REL = {"llama8b": "llama3.1-8b", "phi4": "phi4-14b", "qwen3_1p7b": "qwen3-1.7b"}
TITLE = {"llama8b": "Llama-3.1-8B 16x64", "phi4": "Phi-4 16x64", "qwen3_1p7b": "Qwen3-1.7B 16x64"}
PAPER = json.loads((W / "calibration/tmopt/experiments/paper/maps.sha256.json").read_text())["maps"]
# the paper-setting PPL records (run_ppl_deploy, the same windows as the release evaluation)
PAPER_PPL = {"llama8b": Path("/home/dev/n16k64_campaign/paper/ppl/llama8b/ours-16x64/report.json"),
             "phi4": Path("/home/dev/n16k64_campaign/paper/ppl/phi4/ours-16x64/report.json"),
             "qwen3_1p7b": Path("/home/dev/n16k64_campaign/main_ppl/ppl/qwen3_1p7b/ours-16x64/report.json")}
STEPS = {"N": 32, "O": 16}               # steps per epoch: 256 or 128 windows, 8 per step
CORPORA = ("wiki", "c4")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bare(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["tiles"] if isinstance(obj, dict) and "tiles" in obj else obj


def tiles_equal(a, b):
    x, y = bare(a), bare(b)
    return set(x) == set(y) and all(x[n].dtype == y[n].dtype and torch.equal(x[n], y[n]) for n in x)


def records_identical(a, b):
    """Every tensor record of two torch.save files byte for byte (the archive prefix is the file's stem)."""
    za, zb = zipfile.ZipFile(a), zipfile.ZipFile(b)
    ra = {n.split("/", 1)[1]: n for n in za.namelist() if "/data/" in n}
    rb = {n.split("/", 1)[1]: n for n in zb.namelist() if "/data/" in n}
    return set(ra) == set(rb) and all(za.read(ra[k]) == zb.read(rb[k]) for k in ra)


def paired(x, y):
    assert len(x) == len(y), (len(x), len(y))
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta=m, two_se=2 * se, significant=abs(m) > 2 * se, n=n)


def run(key, setting):
    r = json.loads((RUNS / f"{key}_16x64_{setting}" / "report.json").read_text())
    values = {0: r["initial_dev"]["kl_values"]}
    points = [dict(step=0, epoch=0, dev_kl=r["initial_dev"]["kl"], dev_ce=r["initial_dev"]["ce"], train_kl=None,
                   e0m3_tiles=0, dev_seconds=None)]
    for e in r["epochs"]:
        if "dev_kl" in e:
            step = (e["epoch"] + 1) * STEPS[setting]
            values[step] = e["dev_kl_values"]
            points.append(dict(step=step, epoch=e["epoch"] + 1, dev_kl=e["dev_kl"], dev_ce=e["dev_ce"],
                               train_kl=e["train_kl"], e0m3_tiles=e["e0m3_units"], dev_seconds=e["dev_seconds"]))
    res = r["resources"]
    phases = {p["name"]: p["seconds"] for p in res["phases"]["phases"]}
    return dict(points=points, final_map_sha256=r["map_sha256"], status=r["status"], total_seconds=res["total_seconds"],
                setup_seconds=r["setup_seconds"], training_seconds=r["training_seconds"],
                epoch_seconds=[e["epoch_seconds"] for e in r["epochs"]],
                initial_dev_seconds=phases.get("initial_dev_eval"), final_dev_seconds=phases.get("final_dev_eval"),
                gpu_peak_allocated_gib=res["gpu_peak_allocated_gib"][0], cpu_peak_rss_gib=res["cpu_peak_rss_gib"],
                development=r.get("development"),
                args={k: r["args"].get(k) for k in ("epochs", "eval_every", "teacher_topk", "fit_windows", "dev_backend",
                                                    "lr", "init_logit", "batch", "accum", "no_dev")}), values


def ppl_nll(path):
    r = json.loads(Path(path).read_text())
    if "results" in r:                                            # flipquant evaluation.ppl
        return {c: r["results"][c]["nll"] for c in CORPORA}
    ev = next(iter(r["evaluations"].values()))["evaluation"]    # run_ppl_deploy
    return {c: ev[c]["nll"] for c in CORPORA}


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec, pairs, ppl = dict(models={}), {}, {}
    for key in MODELS:
        N, nv = run(key, "N")
        O, ov = run(key, "O")
        release = json.loads((Path("/home/dev/flipquant_release") / REL[key] / "records/16x64/run.json").read_text())
        n5, rel_map = RUNS / f"{key}_16x64_N/map_epoch005.pt", B / f"runs/{REL[key]}_16x64/run/map.pt"
        o_final = RUNS / f"{key}_16x64_O/map.pt"
        if f"{key}_16x64" in PAPER:
            paper_sha, paper_tiles = PAPER[f"{key}_16x64"]["map_sha256"], None
        else:                                                    # the committed paper-setting map
            committed = W / "maps" / REL[key] / "flipquant_16x64.pt"
            paper_sha = torch.load(committed, map_location="cpu", weights_only=False)["meta"]["source_map_sha256"]
            paper_tiles = tiles_equal(o_final, committed)
        checks = dict(N_epoch5_is_release_map=tiles_equal(n5, rel_map) and records_identical(n5, rel_map),
                      release_map_file_sha256_check=sha(rel_map) == release["run_map_sha256"],
                      O_final_sha256=O["final_map_sha256"], paper_setting_map_sha256=paper_sha,
                      O_final_tiles_equal_committed_map=paper_tiles,
                      O_final_is_paper_setting_map=O["final_map_sha256"] == paper_sha and paper_tiles in (None, True))
        rec["models"][key] = dict(N=N, O=O, checks=checks)
        nbest = min((s for s in nv if s > 0), key=lambda s: sum(nv[s]) / len(nv[s]))
        obest = min((s for s in ov if s > 0), key=lambda s: sum(ov[s]) / len(ov[s]))
        pairs[key] = {"release map (N ep5) - paper-setting map (O end)": paired(nv[160], ov[320]),
                      "N ep10 - paper-setting map": paired(nv[320], ov[320]),
                      "N ep10 - N ep5": paired(nv[320], nv[160]),
                      "release map - start": paired(nv[160], nv[0]),
                      "paper-setting map - start": paired(ov[320], ov[0]),
                      f"N ep10 - N lowest (step {nbest})": paired(nv[320], nv[nbest]),
                      f"O end - O lowest (step {obest})": paired(ov[320], ov[obest]),
                      f"N lowest (step {nbest}) - N ep5": paired(nv[nbest], nv[160]),
                      "O step 320 - O step 160": paired(ov[320], ov[160])}
        # PPL of the N maps against the release map, the paper-setting map and FO6 (the same windows)
        base = Path("/home/dev/flipquant_release") / REL[key] / "ppl"
        refs = {"FO6": base / "fo6.json", "release (N ep5)": base / "flipquant_16x64.json",
                "paper-setting": PAPER_PPL[key]}
        cand = {f"N ep{int(p.stem.split('_ep')[1]):d}": p for p in sorted((RUNS / "ppl").glob(f"{REL[key]}_N_ep*.json"))}
        nll = {k: ppl_nll(v) for k, v in {**refs, **cand}.items()}
        # the paper-setting PPL record's artifact names its source map: it must be O's final map, byte for byte
        art = next(iter(json.loads(PAPER_PPL[key].read_text())["evaluations"].values()))["artifact"]
        src = json.loads((Path(art) / "artifact.json").read_text())["note"]["record"]["map"]
        checks["paper_setting_ppl_source_map"] = src
        checks["paper_setting_ppl_source_map_is_O_final"] = sha(src) == O["final_map_sha256"]
        ppl[key] = dict(ppl={k: {c: math.exp(sum(v[c]) / len(v[c])) for c in CORPORA} for k, v in nll.items()},
                        paired={c_name: {r_name: {c: paired(nll[c_name][c], nll[r_name][c]) for c in CORPORA}
                                         for r_name in refs} for c_name in cand},
                        reference_pairs={f"{a} vs {b}": {c: paired(nll[a][c], nll[b][c]) for c in CORPORA}
                                         for a, b in (("release (N ep5)", "paper-setting"), ("release (N ep5)", "FO6"),
                                                      ("paper-setting", "FO6"))},
                        files={k: str(v) for k, v in {**refs, **cand}.items()})
    rec["disjointness"] = json.loads((RUNS / "disjointness.json").read_text())
    (out / "dev_kl.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "paired.json").write_text(json.dumps(pairs, indent=1) + "\n")
    (out / "ppl.json").write_text(json.dumps(ppl, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 4))
    for ax, key in zip(axes, MODELS):
        for s, style in (("N", "-o"), ("O", "-s")):
            pts = rec["models"][key][s]["points"]
            ax.plot([p["step"] for p in pts], [p["dev_kl"] for p in pts], style, ms=3,
                    label=f"{s}: " + ("256 windows, top-1000, 10 epochs" if s == "N" else "128 windows, full vocab, 20 epochs"))
        ax.axvline(160, color="grey", lw=0.8, ls=":")
        ax.set_title(TITLE[key])
        ax.set_xlabel("optimizer step (8 sequences each)")
        ax.set_ylabel("dev KL (192 windows, full vocabulary)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dev_kl.png", dpi=110)

    def kl(x):
        return "—" if x is None else f"{x:.5f}"
    for key in MODELS:
        m = rec["models"][key]
        print(f"\n### {TITLE[key]}\nchecks {m['checks']}")
        o = {p["step"]: p for p in m["O"]["points"]}
        print("| step | N epoch | N dev KL | N train KL | N E0M3 | O epoch | O dev KL | O train KL | O E0M3 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in m["N"]["points"]:
            q = o.get(p["step"])
            right = f"{q['epoch']} | {kl(q['dev_kl'])} | {kl(q['train_kl'])} | {q['e0m3_tiles']:,}" if q else "| | |"
            print(f"| {p['step']} | {p['epoch']} | {kl(p['dev_kl'])} | {kl(p['train_kl'])} | {p['e0m3_tiles']:,} | {right} |")
        for name, v in pairs[key].items():
            print(f"   {name:52s} {v['delta']:+.5f} ± {v['two_se']:.5f}{' *' if v['significant'] else ''}")
        print("   PPL:", {k: {c: round(x, 4) for c, x in v.items()} for k, v in ppl[key]["ppl"].items()})
        for c_name, refs in [*ppl[key]["paired"].items(), *((k, {"": v}) for k, v in ppl[key]["reference_pairs"].items())]:
            for r_name, v in refs.items():
                print(f"   {c_name}{' vs ' + r_name if r_name else ''}: " + "; ".join(
                    f"{c} {v[c]['delta']:+.4f} ± {v[c]['two_se']:.4f}{' *' if v[c]['significant'] else ''}" for c in CORPORA))


if __name__ == "__main__":
    main(sys.argv[1])
