"""
    Does the ACTIVATION tag grid have the channel structure the weight tag grid lacked?

    Every negative result in this study came from the weight tag grid, whose per-channel component
    is ~0.4% of the variance -- so a permutation of the reduction axis had nothing to group.
    Activations are the opposite case: measured earlier in this repo, the per-channel share of
    log|X| variance is 0.165 (up to 0.79 on o_proj inputs), with diag(S) spreads to 7e11, because
    activation outliers sit in fixed channels. That is the premise SmoothQuant and AWQ rely on.

    The question is whether that structure carries into the E0M3/E2M1 PREFERENCE, which is what a
    type block votes on. Magnitude structure does not automatically imply preference structure --
    the weight case had col_share 0.057 for magnitude but 0.004 for the tag grid.

    Two constraints specific to the activation operand:

      * ROWS ARE TOKENS and are not permutable at inference, so only the CHANNEL axis is available.
        The searches here therefore run with axes="cols".
      * The A-operand MMA tile is m16 x k64, so the smallest realizable activation type block is
        16x64, not the 8x64 that applies to weights. 8x64 is reported as an upper bound.

    Reports, per layer: the tag grid's row/column/residual variance split, tile retention at several
    shapes, and the reordering lift over a cell-shuffle control -- the same discipline used on the
    weights, so the numbers are directly comparable.

        python analyze_act_typeblock.py --model_name llama-3.1-8b-local
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.reorder import (          # noqa: E402
    additive_shares, interaction_structure, scale_block_gain, search_permutation, shuffle_control,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--nsamples", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--clip", type=str, default="headx")
    ap.add_argument("--type_blocks", type=lambda s: s.split(","), default=["16x64", "8x64"])
    ap.add_argument("--rule", type=str, default="harm")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    here = os.path.dirname(os.path.abspath(__file__))
    path = json.load(open(os.path.join(here, "model2path.json")))[args.model_name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(data["text"][:20000]), return_tensors="pt").input_ids

    captured = {}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
            prev = captured.get(name)
            captured[name] = x if prev is None else torch.cat([prev, x], 0)[: args.max_tokens]
        return fn

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % args.layer_stride != 0:
            continue
        if args.projections and not any(name.endswith(p) for p in args.projections):
            continue
        handles.append(mod.register_forward_hook(hook(re.sub(r"^model\.", "", name))))

    with torch.no_grad():
        for i in range(args.nsamples):
            model(enc[:, i * args.seq_len:(i + 1) * args.seq_len].to(model.device))
    for h in handles:
        h.remove()
    del model
    print(f"\n[act typeblock] {len(captured)} layers x {args.max_tokens} tokens\n", flush=True)

    rows = []
    for name, x in captured.items():
        gs = (x.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        gain = scale_block_gain(x / gs, 16, "mse", args.clip)
        r, c, e = additive_shares(gain)
        gen = torch.Generator().manual_seed(args.seed)
        ist = interaction_structure(gain, generator=gen)
        ctrl_g = shuffle_control(gain, torch.Generator().manual_seed(args.seed + 1))
        ceil = float(gain.clamp(min=0).sum())
        pos = float((gain > 0).float().mean())

        print(f"  {name}  grid={tuple(gain.shape)}  E0M3 cells={pos:.3f}  "
              f"variance row={r:.3f} col={c:.3f} resid={e:.3f}  "
              f"col|corr|={ist['col_corr_abs']:.4f}  sv1={ist['sv1']:.4f}", flush=True)

        rec = dict(layer=name, tokens=int(x.shape[0]), channels=int(x.shape[1]),
                   pos_share=round(pos, 4), row_share=round(r, 4), col_share=round(c, 4),
                   resid_share=round(e, 4), col_corr_abs=round(ist["col_corr_abs"], 5),
                   sv1=round(ist["sv1"], 5))
        for tb in args.type_blocks:
            bm, bk = (int(v) for v in tb.split("x"))
            res = search_permutation(gain, bm, bk, 16, args.rule, args.margin,
                                     rounds=6, swap_samples=40000, seed=args.seed, axes="cols")
            cres = search_permutation(ctrl_g, bm, bk, 16, args.rule, args.margin,
                                      rounds=6, swap_samples=40000, seed=args.seed, axes="cols")
            lift = res["recovered"] - cres["recovered"]
            print(f"      {tb:>6}  identity={res['baseline_recovered']:.3f}  "
                  f"search={res['recovered']:.3f}  control={cres['recovered']:.3f}  "
                  f"lift/ctl={lift:+.4f}", flush=True)
            rec[f"id_{tb}"] = round(res["baseline_recovered"], 4)
            rec[f"se_{tb}"] = round(res["recovered"], 4)
            rec[f"ct_{tb}"] = round(cres["recovered"], 4)
            rec[f"lift_{tb}"] = round(lift, 4)
        rows.append(rec)

    n = len(rows)
    a = lambda k: sum(r[k] for r in rows) / n
    print(f"\n=== mean over {n} layers (ACTIVATIONS) ===")
    print(f"  E0M3-preferring cells {a('pos_share'):.3f}")
    print(f"  variance: row(token) {a('row_share'):.4f}  col(channel) {a('col_share'):.4f}  "
          f"resid {a('resid_share'):.4f}")
    print(f"  col profile |corr| {a('col_corr_abs'):.4f}   sv1 {a('sv1'):.4f}")
    print(f"  compare WEIGHT tag grid: col_share 0.0037, sv1 0.1334\n")
    for tb in args.type_blocks:
        print(f"  {tb:>6}  identity {a(f'id_{tb}'):.3f}  search {a(f'se_{tb}'):.3f}  "
              f"control {a(f'ct_{tb}'):.3f}  lift/ctl {a(f'lift_{tb}'):+.4f}")

    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
