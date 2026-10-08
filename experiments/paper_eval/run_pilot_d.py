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
# amendment 7: the Qwen3.8-27B BF16 AIME length job ran out of memory at batch 16 (SDPA's repeat_kv at ~22.7k context,
# 79.7 GiB allocated + 11.6 GiB reserved but unallocated); it runs at batch 8 with the allocator's expandable segments
LENGTH_BATCH = {("qwen3.8-27b", "bf16", "aime"): 8}
JOB_ENV = {("qwen3.8-27b", "bf16", "aime"): {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}
# amendment 8: the user cancelled that rerun (its AIME lengths are priced from FlipQuant 16x64's instead)
CANCELLED = {("qwen3.8-27b", "bf16", "aime")}


def run_ts(name, cmd, out, extra_env=None):
    if G.done(out):
        return None
    G.idle()
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = G.RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    if log.exists():  # an earlier pass's log holds its batch timings: keep it as <name>.pass<k>.log
        k = 1
        while (logs / f"{name}.pass{k}.log").exists():
            k += 1
        log.rename(logs / f"{name}.pass{k}.log")
    G.log(f"START {name}")
    t0 = time.time()
    env = {**G.env(), **(extra_env or {})}
    with open(log, "w") as f:
        f.write(f"{t0:.3f} " + " ".join(map(str, cmd)) + (f" [env {extra_env}]" if extra_env else "") + "\n")
        p = subprocess.Popen(list(map(str, cmd)), cwd=G.FQ, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
                        ("aime", ["--tasks", "aime", "--limit", "2", "--samples", "8", "--batch",
                                  str(LENGTH_BATCH.get((model, pol, "aime"), 16)), "--max-new-tokens", str(CAP)])]
            else:
                jobs = []
            # a batch's duration is the gap between two "generated" lines (the first batch's start is not logged),
            # so every batch size gets at least two batches: 64 prompts at 16 and 32, 128 at 64
            jobs += [(f"batch{b}", ["--tasks", "gsm8k", "--limit", str(max(64, 2 * b)), "--batch", str(b),
                                    "--max-new-tokens", "512"]) for b in (16, 32, 64)]
            for tag, extra in jobs:
                if (model, pol, tag) in CANCELLED:
                    continue
                out = ROOT / model / pol / f"{tag}.json"
                rc = run_ts(f"dpilot_{model}_{pol}_{tag}", base + extra + ["--out", out], out, JOB_ENV.get((model, pol, tag)))
                if rc not in (None, 0):
                    G.log(f"NOTE dpilot {model} {pol} {tag}: rc={rc} (recorded; the pilot continues)")
    G.log("DONE d_pilot")


if __name__ == "__main__":
    main()
