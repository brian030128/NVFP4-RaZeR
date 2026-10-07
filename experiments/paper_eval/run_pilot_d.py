"""D pilot (timing only; not a result): flipquant evaluation.accuracy on Nemotron-Nano-9B-v2 and Qwen3.8-27B, natively,
--recommended-decoding with thinking on, the registry's budgets, batch 16, seed 0.

- Lengths and speed (BF16 and FlipQuant 16x64): gsm8k, math500, ifeval --limit 16 in one process; aime --limit 1
  --samples 8 (aime24 + aime25: two batches of 8) in a second.
- Speed only (NVFP4, FourOverSix, FlipQuant 8x64, 256x64): gsm8k --limit 16 (one batch of 16).
Every log line is prefixed with the wall-clock time, so each batch's duration is in the log; the per-sample records
(tokens, truncation) are the harness's own .jsonl.

    python run_pilot_d.py
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402

MODELS_D = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
POLS = ["bf16", "fq-16x64", "nvfp4", "fo6", "fq-8x64", "fq-256x64"]
LENGTH = ("bf16", "fq-16x64")
ROOT = G.RUN / "d_pilot"


def run_ts(name, cmd, out):
    if G.done(out):
        return None
    G.idle()
    out.parent.mkdir(parents=True, exist_ok=True)
    (G.RUN / "logs").mkdir(parents=True, exist_ok=True)
    G.log(f"START {name}")
    t0 = time.time()
    with open(G.RUN / "logs" / f"{name}.log", "w") as f:
        f.write(f"{t0:.3f} " + " ".join(map(str, cmd)) + "\n")
        p = subprocess.Popen(list(map(str, cmd)), cwd=G.FQ, env=G.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, text=True, bufsize=1)
        for line in p.stdout:
            f.write(f"{time.time():.3f} {line}")
            f.flush()
        rc = p.wait()
    G.log(f"END {name} rc={rc} {time.time() - t0:.0f}s")
    return rc


def main():
    for model in MODELS_D:
        for pol in POLS:
            base = [G.PY, "-u", "-m", "evaluation.accuracy", "--model", model, *G.policy_args(model, pol),
                    "--recommended-decoding", "--batch", "16", "--seed", "0"]
            tasks = "gsm8k,math500,ifeval" if pol in LENGTH else "gsm8k"
            out = ROOT / model / pol / "main.json"
            rc = run_ts(f"dpilot_{model}_{pol}_main", base + ["--tasks", tasks, "--limit", "16", "--out", out], out)
            if rc not in (None, 0):
                G.stop(f"d pilot {model} {pol} main: rc={rc}")
            if pol in LENGTH:
                out = ROOT / model / pol / "aime.json"
                rc = run_ts(f"dpilot_{model}_{pol}_aime", base + ["--tasks", "aime", "--limit", "1", "--samples", "8",
                                                                   "--out", out], out)
                if rc not in (None, 0):
                    G.stop(f"d pilot {model} {pol} aime: rc={rc}")
    G.log("DONE d_pilot")


if __name__ == "__main__":
    main()
