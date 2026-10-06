"""Development-set KL during calibration (records: NVFP4-RaZeR release-maps results/release_maps/dev_kl/).

The vendored trainer (calibration/tmopt/run_train_map.py at flipquant 120173a) runs directly, with the command
calibration.train_map would build (train_map.command), except that --no-dev is dropped so that the trainer's own
development evaluation runs (monitor only), and --eval-every is set per setting:

- N (the release setting): --fit-windows 256 --teacher-topk 1000, 10 epochs (32 steps each), --eval-every 1;
- O (the paper setting): 128 windows, the full-vocabulary teacher, 20 epochs (16 steps each), --eval-every 2.

Both evaluate at steps 0, 32, ..., 320. Same data root as the release runs.

    python devkl_run.py MODEL_KEY N|O
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev/n16k64_campaign/fqopt/wt")
from calibration import tmopt_common as RC  # noqa: E402
from calibration import train_map as TM  # noqa: E402

DATA = Path("/home/dev/n16k64_campaign/fqrel/tmopt_data")
# Llama-3.1-8B: the release data root holds only its development manifests (enough for the extension's skip set); the
# trainer's development evaluation reads the windows too. devkl_data/llama8b = the release root's calibration record
# + the three full development sets of the paper's data root (cost_comparison/data; their manifests are byte-identical
# to the vendored ones)
DATA_BY_KEY = {"llama8b": Path("/home/dev/n16k64_campaign/fqrel/devkl_data")}
OUT = Path("/home/dev/n16k64_campaign/fqrel/devkl")
RQ_BUILD = "/home/dev/NVFP4-RaZeR/repro_local/realquant/build"     # the native dev evaluator's library (libb8x64.so)


def command(key, setting):
    data = DATA_BY_KEY.get(key, DATA)
    deviation = RC.transformers_deviation(data / key / "calibration" / "report.json")
    run_dir = OUT / f"{key}_16x64_{setting}"
    if setting == "N":
        cmd = TM.command(key, "16x64", data, deviation, run_dir, epochs=10, teacher_topk=1000, fit_windows=256)
    else:
        cmd = TM.command(key, "16x64", data, deviation, run_dir)
    cmd = [str(c) for c in cmd]
    cmd.remove("--no-dev")
    cmd[cmd.index("--eval-every") + 1] = "1" if setting == "N" else "2"
    return cmd, run_dir


if __name__ == "__main__":
    key, setting = sys.argv[1], sys.argv[2]
    cmd, run_dir = command(key, setting)
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists")
    env = RC.env()
    env["RQ_BUILD_DIR"] = RQ_BUILD
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=RC.ENGINE, env=env).returncode)
