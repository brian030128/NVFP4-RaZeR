"""M0 diagnostic (not a result): per problem of a batch-16 pilot run, its left-pad count in its batch and whether its
completion equals batch 1's. Rows with no padding that still differ show that the batch dependence is not a padding
leak. CPU only (tokenizer); run with the model's env from the flipquant worktree:

    cd /home/dev/n16k64_campaign/sm120runs/wt && <env python> .../gsm8k_padcheck.py
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, "/home/dev/n16k64_campaign/sm120runs/wt")
from evaluation.accuracy import build_prompt, load_task  # noqa: E402
from models.registry import get  # noqa: E402

RUN = Path("/home/dev/n16k64_campaign/paper_eval/ptq_gsm8k/pilot")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "ptq_gsm8k" / "pilot" / "padcheck.json"
REV = {"nemotron-nano-9b-v2": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
       "qwen3.8-27b": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
rows = {r["id"]: r for r in load_task("gsm8k")}
out = {}
for model, rev in REV.items():
    spec = get(model)
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=rev)
    b16 = [json.loads(l) for l in open(RUN / model / "rtn_fo6_b16.jsonl")]
    b1 = {json.loads(l)["id"]: json.loads(l) for l in open(RUN / model / "rtn_fo6_b1.jsonl")}
    per = []
    for k in range(0, len(b16), 16):
        chunk = b16[k:k + 16]
        lens = [len(tok(build_prompt(tok, spec, rows[r["id"]], False), add_special_tokens=False).input_ids) for r in chunk]
        per += [dict(id=r["id"], pad=max(lens) - n, identical=r["completion"] == b1[r["id"]]["completion"])
                for r, n in zip(chunk, lens) if r["id"] in b1]
    out[model] = dict(no_pad=dict(n=sum(p["pad"] == 0 for p in per), identical=sum(p["identical"] for p in per if p["pad"] == 0)),
                      padded=dict(n=sum(p["pad"] > 0 for p in per), identical=sum(p["identical"] for p in per if p["pad"] > 0)),
                      problems=per)
    print(model, out[model]["no_pad"], out[model]["padded"])
OUT.write_text(json.dumps(out, indent=1) + "\n")
