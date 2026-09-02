"""
    Does reordering COLUMNS to homogenize 16-element scale blocks reduce NVFP4 error?

    Unlike `run_reorder_sim.py`, this permutes individual columns, so it changes which elements
    share a scale block and therefore the block maxima -- the quantity NVFP4's error is actually set
    by. See `quantize/blockorder.py` for why that is the lever that matters.

    Reported per tensor, as a percentage reduction in the true quantizer error (best-alpha search,
    the same code path `quant_mix_4_6` uses), against three orders:

        identity  -- the current order
        sorted    -- columns sorted by magnitude
        refined   -- sorted, then Kernighan-Lin swaps on sum(block_max^2)
        RANDOM    -- the control. A random permutation has the same marginal column statistics and
                     no grouping, so any gain a random order shows is not from grouping.

    Plus the go/no-go statistic: the share of log|W| variance carried by a per-COLUMN factor. Sorting
    can only work if the rows agree about which columns are large.

        python run_blockorder_sim.py --model_name llama-3.1-8b --layer_stride 8
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.blockorder import (      # noqa: E402
    block_cost, column_profile_agreement, magnitude_order, refine_column_order,
)
from quantize.reorder import scale_block_gain      # noqa: E402


def model_tensors(model_name, layer_stride, projections, max_tensors):
    from transformers import AutoModelForCausalLM
    model2path = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "model2path.json")))
    model = AutoModelForCausalLM.from_pretrained(
        model2path[model_name], torch_dtype=torch.bfloat16, device_map="cpu")
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
    out["gaussian [1024,1024]"] = torch.randn(1024, 1024)
    # a shared per-column scale -- exactly the structure sorting exploits, so this must work
    w = torch.randn(1024, 1024) * torch.exp(torch.randn(1, 1024) * 1.5)
    out["column-scaled [1024,1024]"] = w
    # per-ELEMENT outliers, no column structure -- sorting must NOT help here
    w = torch.randn(1024, 1024)
    mask = torch.rand(1024, 1024) < 0.01
    w[mask] *= 25.0
    out["scattered outliers [1024,1024]"] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--max_tensors", type=int, default=1000)
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--clip", type=str, default="headx")
    ap.add_argument("--metric", type=str, default="mse")
    ap.add_argument("--stats", type=lambda s: s.split(","), default=["rms", "max"])
    ap.add_argument("--refine_rounds", type=int, default=6)
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
    print(f"[blockorder] {len(tensors)} tensors from {tag}, clip={args.clip}\n", flush=True)

    rows = []
    for name, w in tensors.items():
        t0 = time.time()
        gs = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        ws = (w / gs).float()
        K  = ws.shape[1]
        gen = torch.Generator().manual_seed(args.seed)

        r_sh, c_sh, e_sh = column_profile_agreement(ws, generator=gen)

        def err(cols):
            _, e2, e0 = scale_block_gain(ws if cols is None else ws[:, cols], args.groupsize,
                                         args.metric, args.clip, return_losses=True)
            return float(e2.to(torch.float64).sum()), float(
                torch.minimum(e2, e0).to(torch.float64).sum())

        base_e2, base_best = err(None)
        base_bc = block_cost(ws, args.groupsize)
        print(f"  {name}  log|W| variance: row={r_sh:.3f} col={c_sh:.3f} resid={e_sh:.3f}"
              f"   ({time.time() - t0:.1f}s)", flush=True)

        rand = torch.randperm(K, generator=gen)
        cand = {"random(control)": rand}
        for st in args.stats:
            cand[f"sorted-{st}"] = magnitude_order(ws, st)
        best_stat = args.stats[0]
        cand[f"refined-{best_stat}"] = refine_column_order(
            ws, magnitude_order(ws, best_stat), args.groupsize,
            rounds=args.refine_rounds, generator=gen)

        rec = dict(model=tag, tensor=name, row_share=round(r_sh, 4), col_share=round(c_sh, 4),
                   resid_share=round(e_sh, 4))
        for label, cols in cand.items():
            e2, bst = err(cols)
            bc = block_cost(ws, args.groupsize, cols)
            print(f"      {label:>18}   E2M1 err {100 * (base_e2 - e2) / base_e2:+7.3f}%   "
                  f"per-block-best {100 * (base_best - bst) / base_best:+7.3f}%   "
                  f"sum block_max^2 {100 * (base_bc - bc) / base_bc:+7.3f}%", flush=True)
            key = label.replace("(control)", "_ctl").replace("-", "_")
            rec[f"{key}_e2m1_pct"] = round(100 * (base_e2 - e2) / base_e2, 4)
            rec[f"{key}_best_pct"] = round(100 * (base_best - bst) / base_best, 4)
            rec[f"{key}_bc_pct"]   = round(100 * (base_bc - bc) / base_bc, 4)
        rows.append(rec)
        print(flush=True)

        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)

    print("=== mean over tensors (positive = error REDUCED) ===")
    keys = [k for k in rows[0] if k.endswith("_e2m1_pct")]
    print(f"{'order':>22} {'E2M1 err':>10} {'block_max^2':>12}")
    for k in keys:
        base = k[:-len("_e2m1_pct")]
        a = lambda kk: sum(r[kk] for r in rows) / len(rows)
        print(f"{base:>22} {a(k):9.3f}% {a(base + '_bc_pct'):11.3f}%")
    print(f"\n{'mean col_share of log|W| variance':>40}: "
          f"{sum(r['col_share'] for r in rows) / len(rows):.4f}")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
