"""
    W4A4 perplexity with a fixed, calibration-derived ACTIVATION channel permutation.

    The activation tag grid carries real per-channel structure where the weight one does not:

        col_share (channel effect)   weights 0.0037    activations 0.1687
        reorder lift over control    weights +0.003    activations +0.081 at 16x64

    because E0M3/E2M1 is decided by block peakedness and activation outliers live in FIXED channels,
    so the same channels make their blocks peaked for every token. Grouping them is therefore
    grouping like with like, which is what the whole study was looking for.

    Procedure:
      1. capture activations at each quantized site on a few calibration batches
      2. build the activation tag grid and search a 16-channel-chunk permutation for it
         (channel axis only -- token order is not permutable at inference)
      3. freeze one permutation per channel width and quantize with it

    Permutations are keyed by channel count, which distinguishes the residual, FF-intermediate and
    head-dim axes in a Llama block. See QuantConfig.a_perm for why that is a simulation shortcut.

        python run_actperm_ppl.py --model_name llama-3.1-8b-local --a_type_block 16x64
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize import QuantConfig, quant_weight                       # noqa: E402
from quantize.reorder import (                                       # noqa: E402
    expand_chunk_perm, scale_block_gain, search_permutation,
)


@torch.no_grad()
def derive_act_perms(model, calib, type_block, clip, rule, margin, max_tokens, seed, verbose=True):
    """ One fixed 16-chunk channel permutation per activation width, from calibration activations. """
    bm, bk = (int(v) for v in type_block.split("x"))
    captured = {}

    def hook(mod, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
        k = x.shape[-1]
        prev = captured.get(k)
        captured[k] = x if prev is None else torch.cat([prev, x], 0)[:max_tokens]

    handles = [m.register_forward_hook(hook) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "head" not in n]
    for ids in calib:
        model(ids.to(model.device))
    for h in handles:
        h.remove()

    perms = {}
    for width, x in sorted(captured.items()):
        gs = (x.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        gain = scale_block_gain(x / gs, 16, "mse", clip)
        res = search_permutation(gain, bm, bk, 16, rule, margin, rounds=8,
                                 swap_samples=40000, seed=seed, axes="cols")
        perms[width] = expand_chunk_perm(res["chunk_perm"], 16)
        if verbose:
            print(f"  width {width:>6}: tokens={x.shape[0]} "
                  f"identity={res['baseline_recovered']:.3f} -> search={res['recovered']:.3f}",
                  flush=True)
    return perms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--datasets", type=lambda s: s.split(","), default=["wikitext"])
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--w_dtype", type=str, default="mix_4_6_clipa1_h1.5")
    ap.add_argument("--a_dtype", type=str, default="mix_4_6_clipa1_h1.5")
    ap.add_argument("--w_type_block", type=str, default="8x64")
    ap.add_argument("--a_type_block", type=str, default="16x64")
    ap.add_argument("--clip", type=str, default="a1")
    ap.add_argument("--rule", type=str, default="harm")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--act_perm", action="store_true", help="Apply the derived permutation.")
    ap.add_argument("--calib_batches", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from utils import load_model_and_tokenizer

    qc = QuantConfig(w_bits=4, w_dtype=args.w_dtype, w_groupsize=16,
                     w_type_block=args.w_type_block,
                     a_bits=4, a_dtype=args.a_dtype, a_groupsize=16,
                     a_type_block=args.a_type_block)
    model, tok = load_model_and_tokenizer(args.model_name, qc, device_map="cuda")
    model.eval()

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(wt["text"][:20000]), return_tensors="pt").input_ids
    calib = [enc[:, i * args.seq_len:(i + 1) * args.seq_len] for i in range(args.calib_batches)]

    if args.act_perm:
        t0 = time.time()
        print("deriving activation channel permutations ...", flush=True)
        qc.a_perm = derive_act_perms(model, calib, args.a_type_block, args.clip,
                                     args.rule, args.margin, args.max_tokens, args.seed)
        print(f"  done in {time.time() - t0:.0f}s\n", flush=True)

    quant_weight(model, qc)

    out = {}
    for ds in args.datasets:
        if ds == "wikitext":
            te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            ids = tok("\n\n".join(te["text"]), return_tensors="pt").input_ids.to(model.device)
        else:
            raise ValueError(ds)
        n = ids.numel() // args.seq_len
        nlls, lf = [], torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for i in range(n):
                b = ids[:, i * args.seq_len:(i + 1) * args.seq_len]
                lg = model(b).logits
                loss = lf(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), b[:, 1:].reshape(-1))
                nlls.append(loss * (args.seq_len - 1))
        ppl = torch.exp(torch.stack(nlls).sum() / (n * (args.seq_len - 1)))
        out[ds] = float(ppl)
        print(f"{ds} perplexity: {ppl.item()}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tag = f"{args.w_dtype}@{args.w_type_block}__{args.a_dtype}@{args.a_type_block}" \
          f"{'__actperm' if args.act_perm else ''}"
    prev = json.load(open(args.output)) if os.path.isfile(args.output) else {}
    prev[tag] = out
    json.dump(prev, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}  [{tag}]")


if __name__ == "__main__":
    main()
