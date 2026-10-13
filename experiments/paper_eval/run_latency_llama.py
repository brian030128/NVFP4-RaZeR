"""Part W (PROTOCOL.md): A's prefill latency for Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct, after a
session-drift check on Mistral-7B-Instruct-v0.3 (fo6 and fq-16x64). A's protocol and checks (run_gpu.latency), the
model list changed; flipquant paper-sm120-runs 6f5ad27 (worktree sm120runs/wt), whose registry has both Instruct
models pinned.

    python run_latency_llama.py

Records: RUN/latency_llama/<model>/<policy>/round<r>.json. One process per (model, policy, round), round r in the order
random.Random(20260928 + r).shuffle(the run's policy list); a finished record is not run again.
"""
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402

FQ = Path("/home/dev/n16k64_campaign/sm120runs/wt")              # flipquant paper-sm120-runs
FQ_COMMIT = "6f5ad27"
ROOT = "latency_llama"
MAPS = G.RUN / "llama_instruct"                                    # part P's maps: <model>/maps/flipquant_<unit>.pt
RUNS = [("mistral-7b", ["fo6", "fq-16x64"]),                       # the session-drift check, first
        ("llama3.2-1b-instruct", list(G.POLICIES)),
        ("llama3.1-8b-instruct", list(G.POLICIES))]


def policy_args(model, pol):
    out = []
    for a in G.POLICIES[pol]:
        if a.startswith("@"):
            u = a[1:]
            out.append(str(G.REL / model / f"flipquant_{u}.pt") if model == "mistral-7b"
                       else str(MAPS / model / "maps" / f"flipquant_{u}.pt"))
        else:
            out.append(a)
    return out


def checkout():
    def git(*a):
        return subprocess.run(["git", "-C", str(FQ), *a], capture_output=True, text=True).stdout.strip()
    head, clean = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no") == ""
    if not (head.startswith(FQ_COMMIT) and clean):
        G.stop(f"latency_llama: flipquant {FQ} is at {head[:7]} (clean: {clean}), not a clean {FQ_COMMIT}")
    return head


def main():
    checkout()
    libs = G.manifests()
    for model, pols in RUNS:
        for r in range(1, 6):
            order = list(pols)
            random.Random(G.SEED + r).shuffle(order)
            G.log(f"latency_llama {model} round {r} order {','.join(order)} (seed {G.SEED + r})")
            for pol in order:
                out = G.RUN / ROOT / model / pol / f"round{r}.json"
                cmd = [G.py(model), "-m", "evaluation.latency", "prefill", "--model", model, *policy_args(model, pol),
                       "--shapes", G.SHAPES, "--reps", "7", "--label", pol, "--round", str(r), "--out", out]
                rc = G.run(f"latency_llama_{model}_{pol}_r{r}", cmd, out, FQ)
                if rc is None:
                    continue
                if rc != 0 or not out.exists():
                    G.stop(f"latency_llama {model} {pol} r{r}: rc={rc}")
                bad = G.check_fq_record(out, pol, libs)
                if bad:
                    G.stop(f"latency_llama {model} {pol} r{r}: {bad}")
    G.log("DONE latency_llama")


if __name__ == "__main__":
    main()
