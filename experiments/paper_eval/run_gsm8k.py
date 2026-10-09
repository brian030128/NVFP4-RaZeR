"""Part M (PROTOCOL.md): tab:ptq's GSM8K column. Greedy GSM8K (1,319 test problems), thinking off, 2048 new tokens at
most, on the 18 tab:ptq configurations (Nemotron-Nano-9B-v2 and Qwen3.8-27B x RTN / GPTQ / Hadamard x NVFP4 /
FourOverSix / FlipQuant 16x64), native 16x64 in n16k64-fast with build_V and --kernel-set auto, exactly the artifacts and
activations of part H's PPL runs. flipquant `evaluation.accuracy` (paper-sm120-runs), its GSM8K prompt and scorer.

    python run_gsm8k.py prompts                       # CPU: the rendered prompts (the thinking switch), pilot/<model>/
    python run_gsm8k.py pilot                         # M0: lengths, timing, batch-1 identity (RTN FourOverSix)
    python run_gsm8k.py check --check-batch=64        # amendment 14: the batch-64 timing check (the same 64 problems)
    python run_gsm8k.py run --batch=<model>:<b>,...   # M1: the 18 configurations, resumable (one job at a time)

Records: RUN/ptq_gsm8k/<model>/<method>_<fmt>.json (+ .jsonl per problem, as evaluation.accuracy writes them) and
.log (every stdout line with its time); START/END in queue.log.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_sm120 as S  # noqa: E402

MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
METHODS = ("rtn", "gptq", "hadamard")
FMTS = ("nvfp4", "fo6", "fq-16x64")
ROOT = G.RUN / "ptq_gsm8k"
CODES = {"nvfp4": "codes_nvfp4_act-nvfp4.pt", "fo6": "codes_fo6.pt", "fq-16x64": "codes_fq-16x64.pt"}
MAX_NEW = 2048
PILOT_LIMIT = {"nemotron-nano-9b-v2": (64, 64), "qwen3.8-27b": (64, 32)}     # (batch 16, batch 1) problems


def fq_map(model, method):
    return G.RUN / "ptq" / model / "fq-16x64_hadamard.pt" if method == "hadamard" else G.REL / model / "flipquant_16x64.pt"


def quant(model, method, fmt):
    """The flags of part H's PPL run of this configuration (part F's for RTN), without the PPL-only ones."""
    q = [*G.NATIVE, "--kernel-set", "auto"]
    q += {"nvfp4": ["--weight", "nvfp4", "--act", "nvfp4"],
          "fo6": ["--weight", "fourover6", "--act", "fourover6"],
          "fq-16x64": ["--weight", "mixfp4", "--act", "fourover6", "--map", fq_map(model, method)]}[fmt]
    q += ["--act-scope", "row"]
    if method == "gptq":
        q += S.gptq_flags(model, G.RUN / "ptq" / model / CODES[fmt])
    elif method == "hadamard":
        q += S.ROT
    return q


def acc_cmd(model, method, fmt, batch, out, limit=None):
    cmd = [G.py(model), "-u", "-m", "evaluation.accuracy", "--model", model, "--revision", S.REV[model],
           *quant(model, method, fmt), "--tasks", "gsm8k", "--no-think", "--max-new-tokens", MAX_NEW,
           "--batch", batch, "--seed", 0, "--out", out]
    return cmd + (["--limit", limit] if limit else [])


def run_ts(name, cmd, out, cwd=S.FQ2):
    """G.run with every stdout line time-stamped (the job's log is out.with_suffix('.log'))."""
    if G.done(out):
        return None
    G.idle()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    G.log(f"START {name}")
    t0 = time.time()
    with open(out.with_suffix(".log"), "a") as f:
        f.write(f"{time.time():.3f} CMD " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=G.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, text=True, bufsize=1)
        for line in p.stdout:
            f.write(f"{time.time():.3f} {line}")
            f.flush()
        rc = p.wait()
    G.log(f"END {name} rc={rc} {time.time() - t0:.0f}s")
    return rc


def prompts():
    """The rendered GSM8K prompt of problem 0 per model, thinking off, as evaluation.accuracy builds it (CPU only)."""
    code = r"""
import json, sys
from transformers import AutoTokenizer
from evaluation.accuracy import build_prompt, load_task
from models.registry import get
spec = get(sys.argv[1])
tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=sys.argv[2])
row = load_task("gsm8k")[0]
off, on = build_prompt(tok, spec, row, False), build_prompt(tok, spec, row, True)
print(json.dumps(dict(model=spec.key, thinking_registry=spec.thinking, prompt_off=off, prompt_on=on,
                      prompt_off_tokens=len(tok(off, add_special_tokens=False).input_ids),
                      pad_token=tok.pad_token, pad_token_id=tok.pad_token_id, eos_token=tok.eos_token)))
"""
    for model in MODELS:
        d = ROOT / "pilot" / model
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([G.py(model), "-c", code, model, S.REV[model]], cwd=S.FQ2, capture_output=True, text=True,
                           env=G.env({"CUDA_VISIBLE_DEVICES": ""}))
        if r.returncode:
            G.stop(f"gsm8k prompts {model}: {r.stderr[-2000:]}")
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        (d / "prompt.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        print(f"{model}: thinking-off prompt ends with {rec['prompt_off'][-40:]!r}; pad {rec['pad_token']!r}")


def pilot():
    """M0 (not a result): RTN FourOverSix, the first 64 problems at batch 16 and the first 64 / 32 at batch 1."""
    for model in MODELS:
        d = ROOT / "pilot" / model
        n16, n1 = PILOT_LIMIT[model]
        for batch, n in ((16, n16), (1, n1)):
            out = d / f"rtn_fo6_b{batch}.json"
            S.must(run_ts(f"gsm8k_pilot_{model}_b{batch}", acc_cmd(model, "rtn", "fo6", batch, out, limit=n), out),
                   f"gsm8k pilot {model} batch {batch}")


def check(batch):
    """Amendment 14: RTN FourOverSix, the pilot's first 64 problems at the chosen batch (timing, memory, agreement)."""
    for model in MODELS:
        out = ROOT / "pilot" / model / f"rtn_fo6_b{batch}.json"
        S.must(run_ts(f"gsm8k_check_{model}_b{batch}", acc_cmd(model, "rtn", "fo6", batch, out, limit=64), out),
               f"gsm8k batch-{batch} check {model}")


def run(batches):
    """M1: the 18 configurations at the batch registered per model after M0 (Nemotron first)."""
    for model in MODELS:
        b = batches[model]
        for method in METHODS:
            for fmt in FMTS:
                out = ROOT / model / f"{method}_{fmt}.json"
                S.must(run_ts(f"gsm8k_{model}_{method}_{fmt}", acc_cmd(model, method, fmt, b, out), out),
                       f"gsm8k {model} {method} {fmt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("prompts", "pilot", "check", "run"))
    ap.add_argument("--batch", default=None, help="<model>:<batch>,... (run)")
    ap.add_argument("--check-batch", type=int, default=64, help="check: the batch size")
    a = ap.parse_args()
    if a.what == "prompts":
        prompts()
    elif a.what == "pilot":
        G.log("gsm8k pilot (part M0)")
        pilot()
    elif a.what == "check":
        G.log(f"gsm8k batch-{a.check_batch} check (amendment 14)")
        check(a.check_batch)
    else:
        bs = dict(x.split(":") for x in a.batch.split(","))
        assert set(bs) == set(MODELS), bs
        G.log(f"gsm8k run (part M1) batches {bs}")
        run({m: int(b) for m, b in bs.items()})
