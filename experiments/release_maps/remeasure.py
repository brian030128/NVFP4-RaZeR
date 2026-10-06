"""Re-measure finished release calibrations whose process-tree host RSS was not sampled when they ran. Each
(model, unit) runs again through topk-cal's measure.py (results/topk_cal/flipquant_parity/measure.py, copied byte for
byte), one at a time on an idle GPU, from the same checkout, data root and settings, into
remeasure/<model>_<unit>_remeasure/. The re-run must reproduce the release map: the same run-map sha256 (the trainer's
bare map.pt) and, tile by tile, the same flipquant map. The result goes next to the release records as
records/<unit>/remeasure.json.

    python remeasure.py --runs qwen3-1.7b:8x64,qwen3-1.7b:16x64,...
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from release import BASE, DATA, LOGS, PY, REL, TRAIN, W, checkout, env, gpu_busy, log, sha

OUT = BASE / "remeasure"


def trainer_metrics(report):
    res = report["resources"]
    phases = res["phases"]["phases"]
    return dict(total_seconds=res["total_seconds"], setup_seconds=report["setup_seconds"],
                training_seconds=report["training_seconds"],
                epoch_seconds=[e["epoch_seconds"] for e in report["epochs"]],
                phase_seconds={p["name"]: p["seconds"] for p in phases if p["name"] != "training"},
                gpu_peak_allocated_gib=res["gpu_peak_allocated_gib"][0], gpu_peak_reserved_gib=res["gpu_peak_reserved_gib"][0],
                cpu_peak_rss_gib=res["cpu_peak_rss_gib"], host_rss_after_model_load_gib=phases[0]["host_rss_end"],
                teacher_storage_bytes=report["teacher_storage"]["bytes"],
                batch=report["args"]["batch"], accum=report["args"]["accum"])


def remeasure(model, unit):
    name = f"{model}_{unit}_remeasure"
    out, mjson = OUT / name, OUT / f"{name}.measure.json"
    dest = REL / model / "records" / unit / "remeasure.json"
    if dest.exists():
        log(f"SKIP remeasure {model} {unit}: done")
        return json.loads(dest.read_text())
    if out.exists():
        raise SystemExit(f"{out} exists without a result; inspect it")
    while gpu_busy():
        import time
        time.sleep(30)
    code = checkout()
    cmd = [PY, str(BASE / "measure.py"), "--out", str(mjson), "--log", str(LOGS / f"{name}.log"), "--cwd", str(W), "--",
           PY, "-m", "calibration.train_map", "--model", model, "--unit", unit, *TRAIN, "--data-root", str(DATA),
           "--out", str(out)]
    log(f"START remeasure {model} {unit}")
    rc = subprocess.run(cmd, env=env(offline=False), stdin=subprocess.DEVNULL).returncode
    log(f"END remeasure {model} {unit} rc={rc}")
    rec = dict(model=model, unit=unit, checkout=code, rc=rc, command=" ".join(cmd[cmd.index("--") + 1:]))
    if rc != 0:
        rec["error"] = (LOGS / f"{name}.log").read_text()[-3000:]
        dest.write_text(json.dumps(rec, indent=1) + "\n")
        return rec
    import torch
    sys.path.insert(0, str(W))
    from flipquant import maps as M
    original = json.loads((REL / model / "records" / unit / "run.json").read_text())
    rep = json.loads((out / "reproduction.json").read_text())
    report = json.loads((out / "run" / "report.json").read_text())
    a, _, _ = M.load(out / "map.pt")
    b, _, _ = M.load(REL / model / f"flipquant_{unit}.pt")
    rec.update(measure=json.loads(mjson.read_text()), trainer=trainer_metrics(report),
               train_map_wall_seconds=rep["wall_seconds"], run_map_sha256=rep["run_map_sha256"],
               release_run_map_sha256=original["run_map_sha256"],
               same_run_map=rep["run_map_sha256"] == original["run_map_sha256"],
               same_tiles=set(a) == set(b) and all(torch.equal(a[n], b[n]) for n in a))
    rec["reproduces"] = rec["same_run_map"] and rec["same_tiles"]
    dest.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"{'OK' if rec['reproduces'] else 'MISMATCH'} remeasure {model} {unit}: run map {rep['run_map_sha256'][:12]} "
        f"tree {rec['measure']['peak_rss_gib']['tree']:.2f} GiB wall {rec['measure']['wall_seconds']:.0f} s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="model:unit,model:unit,...")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in args.runs.split(","):
        model, unit = spec.split(":")
        remeasure(model, unit)
    log("DONE remeasure")


if __name__ == "__main__":
    main()
