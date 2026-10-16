"""Part Z (PROTOCOL.md): B (memory, a30df07's protocol) and C1 (ownership check) for Llama-3.2-1B-Instruct and
Llama-3.1-8B-Instruct.

B's and C1's commands are run_gpu.memory and run_gpu.ownership, with part P's maps and part W's arrangement:
- flipquant paper-sm120-runs (worktree sm120runs/wt), because main 120173a's registry lacks both Instruct models;
- maps RUN/llama_instruct/<model>/maps/flipquant_<unit>.pt;
- build_V; run_gpu.env.

    python run_llama_bc.py

Records: RUN/memory/<model>/<policy>.json and RUN/ownership/<model>_<unit>.json, B's and C1's layout. One job at a time
on an idle GPU (run_gpu.run). A finished record is not run again; a failed job stops the run.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gpu as G  # noqa: E402
import run_latency_llama as W  # noqa: E402

FQ = W.FQ                                                   # flipquant paper-sm120-runs (sm120runs/wt)
CODE_COMMIT = "4bbe0b1005ef15d618057da55733aa52612da3af"     # its code (part Y's); later commits change docs / results only
MODELS = ["llama3.2-1b-instruct", "llama3.1-8b-instruct"]
POLICIES = ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]


def checkout():
    """sm120runs/wt clean, with code identical to 4bbe0b1 (anything later touches only docs/ and results/)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(FQ), *a], capture_output=True, text=True)
    head = git("rev-parse", "HEAD").stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    code = git("diff", "--name-only", CODE_COMMIT, "HEAD").stdout.split()
    other = [f for f in code if not f.startswith(("docs/", "results/"))]
    if dirty or other or git("merge-base", "--is-ancestor", CODE_COMMIT, "HEAD").returncode:
        G.stop(f"part Z: flipquant {FQ} at {head[:7]} (dirty: {bool(dirty)}; code changed since {CODE_COMMIT[:7]}: {other[:5]})")
    G.log(f"part Z: flipquant {FQ} at {head[:7]}, clean, code = {CODE_COMMIT[:7]}")
    return head


def memory():
    for model in MODELS:
        for pol in POLICIES:
            out = G.RUN / "memory" / model / f"{pol}.json"
            cmd = [G.py(model), G.RZ / "experiments" / "paper_eval" / "mem_probe.py", "--model", model,
                   *W.policy_args(model, pol), "--label", pol, "--out", out]
            rc = G.run(f"memory_{model}_{pol}", cmd, out, FQ, {"MEM_PROBE_FLIPQUANT": str(FQ)})
            if rc not in (None, 0):
                G.stop(f"part Z memory {model} {pol}: rc={rc}")


def ownership():
    for model in MODELS:
        for u in G.UNITS:
            out = G.RUN / "ownership" / f"{model}_{u}.json"
            cmd = [G.py(model), "-m", "evaluation.ppl", "--model", model, *W.policy_args(model, f"fq-{u}"),
                   "--paper-convention", "--ownership-check", "--limit", "1", "--out", out]
            rc = G.run(f"ownership_{model}_{u}", cmd, out, FQ)
            if rc not in (None, 0):
                G.stop(f"part Z ownership {model} {u}: rc={rc}")


if __name__ == "__main__":
    checkout()
    G.log("part Z: B (memory) and C1 (ownership) for the two Llama-Instruct models")
    memory()
    ownership()
    G.log("DONE part Z")
