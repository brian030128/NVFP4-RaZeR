"""D pilot (timing and lengths only; not a result): flipquant evaluation.accuracy, natively, on Nemotron-Nano-9B-v2 and
Qwen3.8-27B in n16k64-fast (amendment 3), --recommended-decoding with thinking on, seed 0. The user's design
(2026-10-07): every task capped at 32768 new tokens, to measure the lengths before the budgets are chosen.

- Lengths (BF16 and FlipQuant 16x64), batch 16, --max-new-tokens 32768:
  gsm8k, math500, ifeval --limit 16 (one process); aime --limit 2 --samples 8 (aime24 + aime25, 16 samples each).
- Batch scaling (all 6 policies): gsm8k --max-new-tokens 512 at batch 16 and 32 (64 prompts) and 64 (128 prompts, so
  that a second batch exists), a near-fixed-length workload (with thinking on nearly every sample reaches 512), giving
  the per-step decode time per batch size.
Every log line carries its wall-clock time, so each batch's duration is in the log; the per-sample tokens and truncation
are the harness's own .jsonl.

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
CAP = 32768
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
    if not (G.RUN / "env" / "fast_env_verified.json").exists():
        G.stop("the D pilot runs the hybrid models in n16k64-fast, after its verification")
    for model in MODELS_D:
        for pol in POLS:
            base = [G.py(model), "-u", "-m", "evaluation.accuracy", "--model", model, *G.policy_args(model, pol),
                    "--recommended-decoding", "--seed", "0"]
            if pol in LENGTH:
                jobs = [("main", ["--tasks", "gsm8k,math500,ifeval", "--limit", "16", "--batch", "16",
                                  "--max-new-tokens", str(CAP)]),
                        ("aime", ["--tasks", "aime", "--limit", "2", "--samples", "8", "--batch", "16",
                                  "--max-new-tokens", str(CAP)])]
            else:
                jobs = []
            # a batch's duration is the gap between two "generated" lines (the first batch's start is not logged),
            # so every batch size gets at least two batches: 64 prompts at 16 and 32, 128 at 64
            jobs += [(f"batch{b}", ["--tasks", "gsm8k", "--limit", str(max(64, 2 * b)), "--batch", str(b),
                                    "--max-new-tokens", "512"]) for b in (16, 32, 64)]
            for tag, extra in jobs:
                out = ROOT / model / pol / f"{tag}.json"
                rc = run_ts(f"dpilot_{model}_{pol}_{tag}", base + extra + ["--out", out], out)
                if rc not in (None, 0):
                    G.log(f"NOTE dpilot {model} {pol} {tag}: rc={rc} (recorded; the pilot continues)")
    G.log("DONE d_pilot")


if __name__ == "__main__":
    main()
