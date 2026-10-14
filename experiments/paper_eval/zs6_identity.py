"""Part X, step 0 (c), IF4 / Zou (PROTOCOL.md): flipquant main's flipquant/if4.py and zou_mixfp4.py against the
implementation behind main-ppl's IF4 / Zou records, NVFP4-RaZeR main-ppl 44f8cea quantize/adaptive_formats.py, on
Mistral-7B-Instruct-v0.3 (GPU, n16k64; run from the flipquant worktree).

- Weights: every quantized Linear, quantize(w, rule) vs quantize_weight(w, "1x16", "own"), bitwise; the installed-weight
  sha256 of both (name, then the installed BF16 bytes, in module order) against main-ppl's record.
- Activations: every quantized Linear's input on the first 2048-token WikiText-2 window of the BF16 model,
  quantize_rows(x, rule) vs quantize_act(x), bitwise (the forward itself stays BF16).

    cd <flipquant worktree> && python zs6_identity.py --ref <adaptive_formats.py of 44f8cea> --out <json>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path[0] = os.getcwd()          # the flipquant worktree

import torch  # noqa: E402

from flipquant import if4 as if4_  # noqa: E402
from flipquant import zou_mixfp4 as zou_  # noqa: E402
from flipquant.data import eval_windows  # noqa: E402
from models.loader import load, quantizable_linears  # noqa: E402

MODEL, REV = "mistral-7b", "c170c708c41dac9275d15a8fff4eca08d52bab71"
RECORDS = {r: Path(f"/home/dev/n16k64_campaign/main_ppl/ppl/mistral7b_ins/{r}/report.json") for r in ("if4", "zou")}
RULES = {"if4": if4_, "zou": zou_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    spec_ = importlib.util.spec_from_file_location("adaptive_formats_44f8cea", args.ref)
    AF = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(AF)
    t0 = time.time()
    model, tok, spec = load(MODEL, revision=REV)
    linears = quantizable_linears(model, spec)
    out = dict(model=MODEL, revision=model.flipquant_source["revision"], modules=len(linears),
               flipquant_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               reference=dict(path=str(args.ref), sha256=hashlib.sha256(args.ref.read_bytes()).hexdigest(),
                              source="NVFP4-RaZeR 44f8cea:quantize/adaptive_formats.py"), rules={})
    for rule, mod in RULES.items():
        h_ref, h_new, unequal = hashlib.sha256(), hashlib.sha256(), []
        for name, lin in linears.items():
            w = lin.weight.detach()
            q_ref = AF.quantize(w, rule, None)[0]
            q_new = mod.quantize_weight(w, "1x16", "own", False)[0]
            if q_ref.dtype != q_new.dtype or not torch.equal(q_ref, q_new):
                unequal.append(name)
            for h, q in ((h_ref, q_ref), (h_new, q_new)):
                h.update(name.encode())
                h.update(q.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            del q_ref, q_new
        rec = json.loads(RECORDS[rule].read_text())
        ev = next(iter(rec["evaluations"].values()))
        out["rules"][rule] = dict(weights=dict(modules=len(linears), unequal=unequal, sha256_reference=h_ref.hexdigest(),
                                               sha256_flipquant=h_new.hexdigest(),
                                               sha256_main_ppl_record=ev.get("installed_weight_sha256"),
                                               record=str(RECORDS[rule])))
        torch.cuda.empty_cache()
    # activations: every quantized Linear's input on one window, both rules, compared inside a pre-hook
    acts = {r: dict(calls=0, rows=0, unequal=[]) for r in RULES}

    def hook(mod, inp, _name=None):
        x = inp[0]
        for rule, m in RULES.items():
            a, b = AF.quantize_rows(x, rule), m.quantize_act(x)
            acts[rule]["calls"] += 1
            acts[rule]["rows"] += x.numel() // x.shape[-1]
            if a.dtype != b.dtype or not torch.equal(a, b):
                acts[rule]["unequal"].append(_name)
        return None
    handles = [lin.register_forward_pre_hook(lambda mod, inp, _n=n: hook(mod, inp, _n)) for n, lin in linears.items()]
    windows, meta = eval_windows(tok, "wiki", 2048)
    ids = windows[0].to(model.get_input_embeddings().weight.device)        # [1, 2048]
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    for rule in RULES:
        out["rules"][rule]["activations"] = dict(acts[rule], window="wiki window 0 (2048 tokens)")
        w = out["rules"][rule]["weights"]
        out["rules"][rule]["ok"] = (not w["unequal"] and w["sha256_reference"] == w["sha256_flipquant"]
                                    == w["sha256_main_ppl_record"] and acts[rule]["calls"] == len(linears)
                                    and not acts[rule]["unequal"])
    out["ok"] = all(r["ok"] for r in out["rules"].values())
    out["seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({r: dict(ok=v["ok"], weights_unequal=len(v["weights"]["unequal"]),
                              acts=f"{v['activations']['calls'] - len(v['activations']['unequal'])}/{v['activations']['calls']}")
                      for r, v in out["rules"].items()}))
    if not out["ok"]:
        sys.exit("IF4 / Zou identity check failed")


if __name__ == "__main__":
    main()
