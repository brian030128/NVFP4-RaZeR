"""
    Select the election rule per model by MEASURING the objective, not predicting it.

    WHY THIS IS DIFFERENT FROM SECTION 5
    ------------------------------------
    §5 tried to predict the right rule from aggregate weight/activation statistics, and every
    candidate lost to a fixed rule. But that was the wrong question. The election exists to minimize
    the layer output error

        E(rule) = sum_layers  tr(dW S dW^T)  /  tr(W S W^T),      S = E[x x^T]

    and `lambda` is a fudge factor compensating for the fact that the election optimizes only the
    DIAGONAL of S. We already collect calibration data. So instead of guessing which lambda repairs
    the surrogate, evaluate the real thing for each candidate rule and take the argmin.

    That costs one covariance per layer plus one matmul per (layer, rule) -- no perplexity, no
    backprop, no search. It is affordable at quantization time, which is exactly where the choice
    has to be made.

    THE EXPERIMENT THIS RUNS
    ------------------------
    The question is not "can we compute E(rule)" -- obviously we can. It is whether E(rule) RANKS
    the rules the way perplexity does. If it does, rule selection is solved and needs no predictor.
    If it does not, then the layer output error is itself not what perplexity rewards, which would
    be a much more interesting negative result than §5's, and would explain why every attempt to
    pick a rule from first principles has failed.

    Two guards against fooling ourselves:

      * The importance driving the ELECTION comes from calibration batches A; the covariance `S`
        used to SCORE it comes from held-out batches B. Without that split a rule is graded on the
        data it was fitted to, and the more aggressive rule wins by construction.
      * `tr(dW S dW^T)` is computed with the FULL S, not its diagonal. Using the diagonal here would
        grade each rule by the very surrogate whose inadequacy is the thing under test.

    Usage:
        python analyze_rule_selection.py --model_name qwen3-4b --out results/ruleselect/qwen3-4b.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.importance import collect_importance
from quantize.quantizer import parse_mix_4_6_dtype, quant_mix_4_6


RULES = [
    "mix_4_6_clipa1_e2m1",              # the NVFP4 floor -- must score exactly 0 improvement
    "mix_4_6_clipa1_hess_h1.5",
    "mix_4_6_clipa1_hess_h10",
    "mix_4_6_clipa1_hess_m1",
    "mix_4_6_clipa1_hess_impg16_h10",
]


def get_batches(tokenizer, n, seq_len, device, offset=0):
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    return [ids[:, (offset + i) * seq_len:(offset + i + 1) * seq_len].to(device) for i in range(n)]


@torch.no_grad()
def collect_cov(model, batches, targets):
    """E[x x^T] at the input of each target Linear, float64 to keep the small off-diagonal terms."""
    cov, cnt, handles = {}, {}, []

    def hook_for(name):
        def hook(_m, inp, _o):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            cov[name] = x.T @ x if name not in cov else cov[name] + x.T @ x
            cnt[name] = cnt.get(name, 0) + x.shape[0]
        return hook

    for n, m in model.named_modules():
        if n in targets:
            handles.append(m.register_forward_hook(hook_for(n)))
    for b in batches:
        model(b)
    for h in handles:
        h.remove()
    return {k: v / cnt[k] for k, v in cov.items()}


@torch.no_grad()
def quantize(name, w, imp, block=(8, 64)):
    (metric, elect, margin, use_imp, clip, cg, ag, perm, rot, rn, rg, ro, pv,
     ia, ie, ig) = parse_mix_4_6_dtype(name)
    return quant_mix_4_6(w, 4, 16, type_block=block, metric=metric, elect=elect, margin=margin,
                         clip=clip, clip_min_gain=cg, alpha_min_gain=ag, permute=perm,
                         importance=imp if use_imp else None,
                         imp_alpha=ia, imp_elect=ie, imp_gran=ig)


@torch.no_grad()
def weighted_error(dW, S):
    """tr(dW S dW^T) -- the exact layer output error, with the FULL covariance."""
    return float(((dW @ S) * dW).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer_stride", type=int, default=6)
    ap.add_argument("--projections", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--calib_batches", type=int, default=4)
    ap.add_argument("--eval_batches", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = json.load(open("model2path.json"))[args.model_name]
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    dev = next(model.parameters()).device

    projs = set(args.projections.split(","))
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.startswith("model.layers."):
            if int(name.split(".")[2]) % args.layer_stride == 0 and name.split(".")[-1] in projs:
                targets.append(name)
    print(f"{len(targets)} target layers", flush=True)

    # A: drives the election.  B: held out, grades it.
    print("collecting importance on batch set A ...", flush=True)
    imp_all = collect_importance(model, get_batches(tok, args.calib_batches, args.seq_len, dev))
    print("collecting covariance on held-out batch set B ...", flush=True)
    cov = collect_cov(model, get_batches(tok, args.eval_batches, args.seq_len, dev,
                                         offset=args.calib_batches), set(targets))

    mods = dict(model.named_modules())
    totals = {r: 0.0 for r in RULES}
    denom = 0.0
    per_layer = []

    for name in targets:
        if name not in cov or name not in imp_all:
            continue
        S = cov[name].to(torch.float32)
        w = mods[name].weight.data.to(torch.float32)
        imp = imp_all[name].to(torch.float32).to(w.device)
        base = weighted_error(w, S.to(w.device))
        denom += base
        row = {"layer": name, "base": base}
        for r in RULES:
            dW = w - quantize(r, w, imp).to(torch.float32)
            e = weighted_error(dW, S.to(w.device))
            totals[r] += e
            row[r] = e / base
        per_layer.append(row)
        best = min(RULES, key=lambda r: row[r])
        print(f"  {name:<38} " + " ".join(f"{row[r]:.5f}" for r in RULES)
              + f"   best={best.replace('mix_4_6_clipa1_','')}", flush=True)
        cov[name] = None
        del S

    print("\n=== relative layer output error, summed over sampled layers ===")
    ref = totals["mix_4_6_clipa1_e2m1"]
    ranking = sorted(RULES, key=lambda r: totals[r])
    for r in ranking:
        print(f"  {r.replace('mix_4_6_clipa1_',''):<20} {totals[r]/denom:.6f}   "
              f"vs e2m1 {100*(totals[r]-ref)/ref:+.3f}%")
    print(f"\nARGMIN = {ranking[0].replace('mix_4_6_clipa1_','')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"model": args.model_name,
               "totals": {r: totals[r] for r in RULES},
               "denom": denom,
               "ranking": [r.replace("mix_4_6_clipa1_", "") for r in ranking],
               "layers": per_layer}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
