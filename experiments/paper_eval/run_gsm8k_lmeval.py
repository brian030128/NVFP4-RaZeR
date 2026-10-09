"""Part N (PROTOCOL.md): tab:ptq's GSM8K column with lm-eval's own method -- lm-eval 0.4.11's `gsm8k_llama` exactly as
shipped (8-shot CoT, first_n shots, apply_chat_template + fewshot_as_multiturn, greedy, max_gen_toks 1024,
strict_match / flexible_extract), thinking off through lm-eval's own switches, on the 18 tab:ptq configurations
(Nemotron-Nano-9B-v2 and Qwen3.8-27B x RTN / GPTQ / Hadamard x NVFP4 / FourOverSix / FlipQuant 16x64), native 16x64,
n16k64-fast, build_V, --kernel-set auto, with part H's artifacts (run_gsm8k.quant). flipquant `evaluation.downstream`
(paper-sm120-runs f2a56f1): HFLM on the natively quantized model, generate_until through model.generate.

    python run_gsm8k_lmeval.py docids                   # CPU: the pilot's documents per model (pilot/<model>/doc_ids.json)
    python run_gsm8k_lmeval.py smoke                    # N0: the thinking switch and the rendered prompt, 8 problems
    python run_gsm8k_lmeval.py pilot [--batch=64]       # N0: the 64 longest questions + 64 others at the batch size
    python run_gsm8k_lmeval.py run --batch=<model>:<b>,...   # N1: the 18 configurations (one job at a time)

Records: RUN/ptq_gsm8k_lmeval/<model>/<method>_<fmt>.json (+ .log, + .gsm8k_llama.jsonl.gz: lm-eval's samples).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_gsm8k as M  # noqa: E402
import run_sm120 as S  # noqa: E402

MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
ROOT = G.RUN / "ptq_gsm8k_lmeval"
TASK = "gsm8k_llama"
# thinking off through lm-eval's own switches: HFLM enable_thinking (chat_template_args) / lm-eval's system instruction
THINK_OFF = {"qwen3.8-27b": ["--enable-thinking", "false"], "nemotron-nano-9b-v2": ["--system-instruction", "/no_think"]}


def cmd(model, method, fmt, batch, out, limit=None, doc_ids=None):
    c = [G.py(model), "-u", "-m", "evaluation.downstream", "--model", model, "--revision", S.REV[model],
         *M.quant(model, method, fmt), "--tasks", TASK, "--apply-chat-template", *THINK_OFF[model],
         "--batch-size", batch, "--gen-lengths", "--samples-out", Path(out).with_suffix(""), "--out", out]
    if limit:
        c += ["--limit", limit]
    if doc_ids:
        c += ["--doc-ids", doc_ids]
    return c


def docids():
    """The pilot's documents per model: the 64 questions with the most tokens (the prompts differ only in the question,
    so these are the full run's longest batch, its memory worst case) and 64 others drawn with seed 0."""
    code = r"""
import json, sys, random
from datasets import load_dataset
from transformers import AutoTokenizer
from models.registry import get
spec = get(sys.argv[1])
tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=sys.argv[2])
ds = load_dataset("openai/gsm8k", "main", split="test", revision="740312add88f781978c0658806c59bc2815b9866")
n = [len(tok(q, add_special_tokens=False).input_ids) for q in ds["question"]]
order = sorted(range(len(n)), key=lambda i: (-n[i], i))
longest = order[:64]
rest = [i for i in range(len(n)) if i not in set(longest)]
others = sorted(random.Random(0).sample(rest, 64))
print(json.dumps(dict(docs=sorted(longest) + others, longest=sorted(longest), others=others,
                      longest_tokens=[n[i] for i in longest[:3]], median_tokens=sorted(n)[len(n) // 2])))
"""
    for model in MODELS:
        d = ROOT / "pilot" / model
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([G.py(model), "-c", code, model, S.REV[model]], cwd=S.FQ2, capture_output=True, text=True,
                           env=G.env({"CUDA_VISIBLE_DEVICES": ""}))
        if r.returncode:
            G.stop(f"gsm8k_lmeval docids {model}: {r.stderr[-2000:]}")
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        (d / "doc_ids_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        (d / "doc_ids.json").write_text(json.dumps({TASK: rec["docs"]}) + "\n")
        print(model, "longest question tokens", rec["longest_tokens"], "median", rec["median_tokens"])


def smoke():
    for model in MODELS:
        out = ROOT / "pilot" / model / "smoke_rtn_fo6.json"
        S.must(M.run_ts(f"gsm8k_lmeval_smoke_{model}", cmd(model, "rtn", "fo6", 8, out, limit=8), out),
               f"gsm8k_lmeval smoke {model}")


def pilot(batch):
    for model in MODELS:
        d = ROOT / "pilot" / model
        out = d / f"pilot_rtn_fo6_b{batch}.json"
        S.must(M.run_ts(f"gsm8k_lmeval_pilot_{model}_b{batch}", cmd(model, "rtn", "fo6", batch, out,
                                                                    doc_ids=d / "doc_ids.json"), out),
               f"gsm8k_lmeval pilot {model} batch {batch}")


def run(batches):
    for model in MODELS:
        for method in M.METHODS:
            for fmt in M.FMTS:
                out = ROOT / model / f"{method}_{fmt}.json"
                S.must(M.run_ts(f"gsm8k_lmeval_{model}_{method}_{fmt}", cmd(model, method, fmt, batches[model], out), out),
                       f"gsm8k_lmeval {model} {method} {fmt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("docids", "smoke", "pilot", "run"))
    ap.add_argument("--batch", default="64", help="pilot: the batch size; run: <model>:<batch>,...")
    a = ap.parse_args()
    if a.what == "docids":
        docids()
    elif a.what == "smoke":
        G.log("gsm8k_lmeval smoke (part N0)")
        smoke()
    elif a.what == "pilot":
        G.log(f"gsm8k_lmeval pilot (part N0) batch {a.batch}")
        pilot(int(a.batch))
    else:
        bs = dict(x.split(":") for x in a.batch.split(","))
        assert set(bs) == set(MODELS), bs
        G.log(f"gsm8k_lmeval run (part N1) batches {bs}")
        run({m: int(b) for m, b in bs.items()})
