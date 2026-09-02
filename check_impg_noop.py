"""
    Is the per-type-block importance a no-op on REAL weights?

    Theory and the synthetic test both say yes, bit for bit: a positive constant across a type block
    divides out of the alpha search and out of every election rule. But the perplexity runs disagree
    for two of three models -- `impg64` reproduced the unweighted run to every printed digit on
    Qwen3-4B, and drifted ~0.003 wikitext on Llama-3.1-8B and Qwen3-8B.

    The synthetic test uses 5e5 elements. A real model has ~7e9 per config, 1.3e4 times more, so a
    flip rate too rare to appear there can still flip thousands of blocks in a real run. The
    suspected mechanism is that `sum_j (c * d_j^2)` and `c * sum_j d_j^2` are different floats, so
    the alpha search's `err < best_err` can break a near-tie the other way.

    This settles it on the actual weight distributions: quantize real tensors both ways and count.
    Importance is synthetic on purpose -- the invariance claim does not depend on where the weights
    come from, and this keeps the check free of a calibration pass.
"""
import argparse, json, os

import torch
from transformers import AutoModelForCausalLM

from quantize.quantizer import parse_mix_4_6_dtype, quant_mix_4_6


def run(name, w, imp, tb):
    (metric, elect, margin, use_imp, clip, cg, ag, perm, rot, rn, rg, ro, pv,
     ia, ie, ig) = parse_mix_4_6_dtype(name)
    bm, bk = (int(x) for x in tb.split("x"))
    return quant_mix_4_6(w, 4, 16, type_block=(bm, bk), metric=metric, elect=elect, margin=margin,
                         clip=clip, clip_min_gain=cg, alpha_min_gain=ag, permute=perm,
                         importance=imp if use_imp else None,
                         imp_alpha=ia, imp_elect=ie, imp_gran=ig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="llama-3.1-8b-local")
    ap.add_argument("--max_layers", type=int, default=4)
    ap.add_argument("--type_block", default="8x64")
    ap.add_argument("--base", default="mix_4_6_clipheadx_m1")
    ap.add_argument("--gran", default="mix_4_6_clipheadx_hess_impg64_m1")
    args = ap.parse_args()

    path = json.load(open("model2path.json"))[args.model_name]
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    g = torch.Generator().manual_seed(0)

    tot_diff = tot_el = 0
    print(f"{'layer':>28} {'shape':>16} {'differ':>10} {'of':>12} {'%':>9} {'max|d|':>10}")
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        li = name.split(".")[2] if name.startswith("model.layers.") else None
        if li is None or int(li) >= args.max_layers:
            continue
        w = mod.weight.data.to(torch.float32)
        imp = torch.rand(w.shape[1], generator=g).pow(6) * 1000 + 1e-3
        a = run(args.base, w, imp, args.type_block)
        b = run(args.gran, w, imp, args.type_block)
        d = (a != b)
        n = int(d.sum())
        tot_diff += n
        tot_el += a.numel()
        print(f"{name.replace('model.layers.',''):>28} {str(tuple(w.shape)):>16} {n:>10} "
              f"{a.numel():>12} {100*n/a.numel():>8.5f}% "
              f"{(a.float()-b.float()).abs().max().item():>10.3e}")

    print(f"\nTOTAL {tot_diff} / {tot_el} elements differ ({100*tot_diff/max(tot_el,1):.6f}%)")
    if tot_diff == 0:
        print("=> per-type-block importance IS an exact no-op on real weights too;")
        print("   the perplexity drift must come from somewhere other than the quantizer.")
    else:
        print("=> per-type-block importance is NOT bit-exact on real weights.")
        print(f"   Extrapolated to a ~7e9-element model: ~{7e9*tot_diff/max(tot_el,1):.3g} elements.")
        print("   This is the noise floor for any impg comparison on these models.")


if __name__ == "__main__":
    main()
