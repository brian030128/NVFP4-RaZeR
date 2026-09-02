"""
    Reordering BEFORE a block-diagonal Hadamard rotation.

    QuaRot rotates the whole hidden dimension with one randomized Hadamard absorbed into the
    weights. Against that, a permutation is provably useless: a dense Hadamard already mixes every
    channel and `PH` is just another orthogonal matrix. Reordering only has leverage for a
    BLOCK-DIAGONAL rotation -- a `rotate_size`-wide chunked Hadamard, which is what this repo
    implements and what is cheap to apply on the fly.

    There, which columns share a chunk decides how much the rotation can dissolve:

        rotation collapses {big, small x 15} into 16 medium values -- block_max drops ~4x
        rotation does nothing to {big x 16}   -- every element already needs a coarse scale

    So rotation wants outliers SPREAD, one per chunk, which is the exact opposite of the magnitude
    sorting that `run_blockorder_sim.py` measured (and which lost). `spread_order` sorts and then
    deals columns round-robin so every chunk gets one column from each magnitude stratum.

    The falsifiable prediction: `spread` should beat `identity` at rotate_size 16, and the advantage
    should SHRINK as rotate_size grows, vanishing when a chunk spans the whole dimension -- because
    that limit is QuaRot, where the permutation is absorbed.

        python run_rotorder_sim.py --model_name llama-3.1-8b --layer_stride 8
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.blockorder import magnitude_order, rotation_split_error, spread_order  # noqa: E402


def model_tensors(model_name, layer_stride, projections, max_tensors):
    from transformers import AutoModelForCausalLM
    model2path = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "model2path.json")))
    model = AutoModelForCausalLM.from_pretrained(model2path[model_name],
                                                 torch_dtype=torch.bfloat16, device_map="cpu")
    tensors = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % layer_stride != 0:
            continue
        if projections and not any(name.endswith(p) for p in projections):
            continue
        tensors[f"{re.sub(r'^model.', '', name)} {tuple(module.weight.shape)}"] = \
            module.weight.data.clone().float()
        if len(tensors) >= max_tensors:
            break
    del model
    return tensors


def synthetic_tensors(seed=0):
    torch.manual_seed(seed)
    out = {}
    # 1% outlier COLUMNS -- the case rotation is for, and the case where spreading them one per
    # chunk should matter most
    w = torch.randn(1024, 1024)
    out_cols = torch.randperm(1024)[:10]
    w[:, out_cols] *= 25.0
    out["1% outlier columns [1024,1024]"] = w
    out["gaussian [1024,1024]"] = torch.randn(1024, 1024)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--max_tensors", type=int, default=1000)
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--clip", type=str, default="a1")
    ap.add_argument("--rotate_sizes", type=lambda s: [int(v) for v in s.split(",")],
                    default=[16, 64, 128])
    ap.add_argument("--min_gain", type=float, default=0.1,
                    help="rotmin<t>: rotate a chunk only if it beats no-rotation by this fraction. "
                         "CLAUDE.md measures rotmin0.1 as the best realizable config in the study.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    tensors = (synthetic_tensors(args.seed) if args.model_name is None
               else model_tensors(args.model_name, args.layer_stride, args.projections,
                                  args.max_tensors))
    tag = args.model_name or "synthetic"
    print(f"[rotorder] {len(tensors)} tensors from {tag}, clip={args.clip}, "
          f"rotmin={args.min_gain}\n", flush=True)

    rows = []
    for name, w in tensors.items():
        gs = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        ws = (w / gs).float()
        gen = torch.Generator().manual_seed(args.seed)
        K = ws.shape[1]

        print(f"  {name}", flush=True)
        for rs in args.rotate_sizes:
            if K % rs:
                continue
            orders = {
                "identity": None,
                "spread":   spread_order(ws, "rms", rs),
                "sorted":   magnitude_order(ws, "rms"),
                "random":   torch.randperm(K, generator=gen),
            }
            base = rotation_split_error(ws, None, args.groupsize, args.clip,
                                        rotate_size=rs, min_gain=args.min_gain)["norot"]
            line = f"    rot{rs:<4}"
            rec = dict(model=tag, tensor=name, rotate_size=rs)
            for label, cols in orders.items():
                r = rotation_split_error(ws, cols, args.groupsize, args.clip,
                                         rotate_size=rs, min_gain=args.min_gain)
                pct = lambda v: 100.0 * (base - v) / base
                line += (f"  {label}: norot {pct(r['norot']):+6.2f}% "
                         f"rotmin {pct(r['percol']):+6.2f}% ({100 * r['rotated_share']:.0f}% rot)")
                rec[f"{label}_norot_pct"] = round(pct(r["norot"]), 4)
                rec[f"{label}_rotmin_pct"] = round(pct(r["percol"]), 4)
                rec[f"{label}_allrot_pct"] = round(pct(r["allrot"]), 4)
                rec[f"{label}_rotated_share"] = round(r["rotated_share"], 4)
            print(line, flush=True)
            rows.append(rec)

        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)

    print("\n=== mean over tensors: E2M1 error reduction vs the unrotated identity order ===")
    print(f"{'rot':>6} {'identity+rotmin':>16} {'spread+rotmin':>14} {'sorted+rotmin':>14} "
          f"{'random+rotmin':>14} {'spread-identity':>16}")
    for rs in args.rotate_sizes:
        sel = [r for r in rows if r["rotate_size"] == rs]
        if not sel:
            continue
        a = lambda k: sum(r[k] for r in sel) / len(sel)
        print(f"{rs:>6} {a('identity_rotmin_pct'):15.3f}% {a('spread_rotmin_pct'):13.3f}% "
              f"{a('sorted_rotmin_pct'):13.3f}% {a('random_rotmin_pct'):13.3f}% "
              f"{a('spread_rotmin_pct') - a('identity_rotmin_pct'):15.3f}%")
    print("\nPrediction: the last column shrinks toward 0 as rot grows -- a wide Hadamard needs no "
          "help choosing what to mix.")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
