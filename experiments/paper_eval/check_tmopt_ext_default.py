"""Part AC (PROTOCOL.md, items 3 and 4): flipquant paper-1k-reruns' tmopt_ext, with its opt-in --fit-windows /
--teacher-host left at their defaults, builds the same trainer command as calib-size d940fbf's tmopt_ext, for every
command kind the items use (oneshot, train, train --rotate, train --candidates-*) and five models. CPU only:
tmopt_common.run is replaced, so nothing is trained.

    python check_tmopt_ext_default.py OUT.json        # run with flipquant paper-1k-reruns on sys.path (cwd)
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FQN = Path("/home/dev/n16k64_campaign/paper1k/wt")
OLD = "d940fbf"
MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b", "mistral-7b", "phi4-14b", "llama3.1-8b-instruct"]
KINDS = [("oneshot", "16x64", 1, ["--oneshot", "g.pt"]), ("train", "8x64", 5, []),
         ("train --rotate", "16x64", 5, ["--rotate", "hadamard", "--rotate-block", "16"]),
         ("train --candidates", "16x64", 5, ["--candidates-e2m1", "a", "--candidates-e0m3", "b"])]


def main(out):
    sys.path.insert(0, str(FQN))
    from calibration import tmopt_common as RC
    from calibration import tmopt_ext as NEW
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "tmopt_ext_old.py"
    src.write_text(subprocess.run(["git", "-C", str(FQN), "show", f"{OLD}:calibration/tmopt_ext.py"], capture_output=True,
                                  text=True, check=True).stdout)
    spec_ = importlib.util.spec_from_file_location("calibration.tmopt_ext_old", src)
    old = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(old)
    run, seen = tmp / "run", []

    def fake_run(cmd, log):
        seen.append([str(c) for c in cmd])
        (run / "run").mkdir(parents=True, exist_ok=True)
        (run / "run" / "report.json").write_text(json.dumps(dict(status="complete")))
        return 0

    RC.run = fake_run
    rec = tmp / "report.json"
    rec.write_text(json.dumps(dict(revision="r")))
    rows = []
    for model in MODELS:
        spec, key = RC.resolve(model)
        for kind, unit, epochs, hooks in KINDS:
            seen.clear()
            old.run_trainer(spec, key, unit, run, Path("/root"), rec, epochs, hooks)
            NEW.run_trainer(spec, key, unit, run, Path("/root"), rec, epochs, hooks)
            rows.append(dict(model=model, kind=kind, unit=unit, identical=seen[0] == seen[1],
                             fit_windows=seen[1][seen[1].index("--fit-windows") + 1]))
    head = subprocess.run(["git", "-C", str(FQN), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(new=head, old=OLD, rows=rows, all_identical=all(r["identical"] for r in rows))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"{sum(r['identical'] for r in rows)} of {len(rows)} commands identical; all identical: {res['all_identical']}")


if __name__ == "__main__":
    main(sys.argv[1])
