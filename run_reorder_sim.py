"""
    How much of the 1x16 E0M3 gain can a row/column REORDERING give back to a coarse type block?

    For every weight matrix sampled, this builds the tag grid

        G[i, j] = loss_E2M1(scale block i,j) - loss_E0M3(scale block i,j)

    and runs the balanced co-clustering search in `quantize/reorder.py` for each requested type
    block and election rule. Three numbers per row of the output:

        identity   -- the realized gain in the current, unpermuted order (what `mix_4_6` gets today)
        search     -- the realized gain after reordering
        control    -- the SAME search run on a cell-shuffled copy of G

    all as a fraction of the 1x16 ceiling `sum relu(G)`. `control` is the point of the exercise: a
    balanced partition search over tens of thousands of tiles concentrates positive mass even in
    i.i.d. noise, so `search - control` is the part attributable to rows and columns genuinely
    sharing a data-type preference, and `search - identity` is the part that would show up in a
    perplexity run.

    Examples

        python run_reorder_sim.py                                   # synthetic tensors, quick
        python run_reorder_sim.py --model_name llama-2-7b --layer_stride 8
        python run_reorder_sim.py --model_name llama-3.1-8b --type_blocks 8x64,32x128 \
                                  --rules argmin,h1.5 --out results/reorder/llama31.csv
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

from quantize.reorder import (          # noqa: E402
    additive_shares, scale_block_gain, search_permutation, shuffle_control,
)


DEFAULT_TYPE_BLOCKS = ["8x64", "16x64", "32x64", "32x128"]
DEFAULT_RULES       = ["argmin", "h1.5"]


def parse_rule(spec: str):
    """ "argmin" | "h<lambda>" | "m<z>" | "v<t>" | "dominance" -> (rule, margin). """
    if spec in ("argmin", "dominance", "never", "always"):
        return spec, 0.0
    m = re.fullmatch(r"h([0-9.]+)", spec)
    if m:
        return "harm", float(m.group(1))
    m = re.fullmatch(r"m([0-9.]+)", spec)
    if m:
        return "margin", float(m.group(1))
    m = re.fullmatch(r"v([0-9.]+)", spec)
    if m:
        return "vote", float(m.group(1))
    raise ValueError(f'Unrecognized election rule "{spec}".')


def parse_type_block(spec: str):
    m = re.fullmatch(r"(\d+)x(\d+)", spec)
    assert m, f'Type block must look like "<M>x<K>", got "{spec}".'
    return int(m.group(1)), int(m.group(2))


# ----------------------------------------------------------------------------------------------
# Tensors to sweep
# ----------------------------------------------------------------------------------------------
def synthetic_tensors(seed: int = 0):
    torch.manual_seed(seed)
    out = {}
    out["gaussian [1024,1024]"] = torch.randn(1024, 1024)

    # A tensor whose E0M3 preference IS structured by row and by column: half the rows and half the
    # column chunks are heavy-tailed. If the search cannot find this, it is broken.
    w = torch.randn(1024, 1024)
    heavy_row = torch.zeros(1024, dtype=torch.bool); heavy_row[::2] = True
    heavy_col = torch.zeros(1024, dtype=torch.bool); heavy_col[:512] = True
    mask = heavy_row[:, None] & heavy_col[None, :]
    w[mask] = w[mask] * torch.distributions.Gamma(0.4, 0.4).sample(w[mask].shape).sqrt()
    out["planted heavy-tail [1024,1024]"] = w

    w = torch.randn(1024, 1024)
    w[:, ::64] *= 20.0
    out["column outliers [1024,1024]"] = w
    return out


def model_tensors(model_name: str, layer_stride: int, projections, max_tensors: int):
    from transformers import AutoModelForCausalLM

    model2path = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "model2path.json")))
    model = AutoModelForCausalLM.from_pretrained(
        model2path[model_name], torch_dtype=torch.bfloat16, device_map="cpu"
    )

    tensors = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % layer_stride != 0:
            continue
        if projections and not any(name.endswith(p) for p in projections):
            continue
        short = re.sub(r"^model\.", "", name)
        tensors[f"{short} {tuple(module.weight.shape)}"] = module.weight.data.clone().float()
        if len(tensors) >= max_tensors:
            break
    del model
    return tensors


# ----------------------------------------------------------------------------------------------
def sign_structure(gain):
    """
        How close the SIGN pattern of the tag grid is to a rank-1 checkerboard.

        A row-partition x column-partition product can only express "row group b and column group c
        agree", so the ceiling for the whole idea is set by how well sign(G) ~ a_i b_j fits. Reported
        as the fraction of |G| mass that the best rank-1 sign model gets right, found by two
        alternating sweeps from the row-marginal sign.
    """
    g = gain.to(torch.float32)
    w = g.abs()
    a = torch.sign(g.sum(dim=1)); a[a == 0] = 1.0
    for _ in range(8):
        b = torch.sign((a[:, None] * g).sum(dim=0)); b[b == 0] = 1.0
        a = torch.sign((b[None, :] * g).sum(dim=1)); a[a == 0] = 1.0
    agree = (torch.sign(g) == (a[:, None] * b[None, :])).to(torch.float32)
    return float((agree * w).sum() / w.sum().clamp(min=1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=None,
                    help="Model from model2path.json; synthetic tensors if unset.")
    ap.add_argument("--layer_stride", type=int, default=8,
                    help="Sample every Nth decoder layer.")
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--max_tensors", type=int, default=1000)
    ap.add_argument("--type_blocks", type=lambda s: s.split(","), default=DEFAULT_TYPE_BLOCKS)
    ap.add_argument("--rules", type=lambda s: s.split(","), default=DEFAULT_RULES)
    ap.add_argument("--clip", type=str, default="heade0",
                    help="Clip preset for the tag grid. `heade0` is the configuration CLAUDE.md "
                         "recommends when the E0M3 branch is used at all.")
    ap.add_argument("--metric", type=str, default="mse")
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--swap_samples", type=int, default=40000)
    ap.add_argument("--diagnostics_only", action="store_true",
                    help="Report the structure statistics only; skip the (expensive) searches. "
                         "This is the cheap way to answer whether reordering CAN help at all.")
    ap.add_argument("--no_control", action="store_true",
                    help="Skip the cell-shuffle control (halves the runtime).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="CSV output path.")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.model_name is None:
        tensors = synthetic_tensors(args.seed)
        tag = "synthetic"
    else:
        tensors = model_tensors(args.model_name, args.layer_stride, args.projections,
                                args.max_tensors)
        tag = args.model_name
    print(f"[reorder] {len(tensors)} tensors from {tag}, "
          f"clip={args.clip} metric={args.metric}", flush=True)

    def flush():
        if not args.out or not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    rows = []
    for name, w in tensors.items():
        t0 = time.time()
        w = w.float()
        # the same global scale `quant_mix_4_6` uses; a positive scalar, so it changes no sign
        gscale   = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        gain     = scale_block_gain(w / gscale, args.groupsize, args.metric, args.clip)
        ctrlgain = None if args.no_control else shuffle_control(
            gain, torch.Generator().manual_seed(args.seed + 1))

        pos_share = float((gain > 0).to(torch.float32).mean())
        rank1     = sign_structure(gain)
        r_sh, c_sh, e_sh = additive_shares(gain)
        print(f"\n  {name}  grid={tuple(gain.shape)}  E0M3-preferring cells={pos_share:.3f}  "
              f"rank1 sign fit={rank1:.3f}  variance: row={r_sh:.3f} col={c_sh:.3f} "
              f"resid={e_sh:.3f}  ({time.time() - t0:.1f}s)", flush=True)

        if args.diagnostics_only:
            rows.append(dict(model=tag, tensor=name, type_block="-", rule="-",
                             clip=args.clip, metric=args.metric,
                             pos_share=round(pos_share, 4), rank1_sign_fit=round(rank1, 4),
                             row_share=round(r_sh, 4), col_share=round(c_sh, 4),
                             resid_share=round(e_sh, 4)))
            flush()
            continue

        for tb in args.type_blocks:
            bm, bk = parse_type_block(tb)
            for rspec in args.rules:
                rule, margin = parse_rule(rspec)
                t1 = time.time()
                res = search_permutation(gain, bm, bk, args.groupsize, rule, margin,
                                         rounds=args.rounds, swap_samples=args.swap_samples,
                                         seed=args.seed)
                ctrl = float("nan")
                if ctrlgain is not None:
                    ctrl = search_permutation(ctrlgain, bm, bk, args.groupsize, rule, margin,
                                              rounds=args.rounds,
                                              swap_samples=args.swap_samples,
                                              seed=args.seed)["recovered"]
                rows.append(dict(
                    model=tag, tensor=name, type_block=tb, rule=rspec,
                    clip=args.clip, metric=args.metric,
                    pos_share=round(pos_share, 4), rank1_sign_fit=round(rank1, 4),
                    row_share=round(r_sh, 4), col_share=round(c_sh, 4),
                    resid_share=round(e_sh, 4),
                    identity=round(res["baseline_recovered"], 4),
                    search=round(res["recovered"], 4),
                    control=round(ctrl, 4),
                    lift_vs_identity=round(res["recovered"] - res["baseline_recovered"], 4),
                    lift_vs_control=round(res["recovered"] - ctrl, 4),
                    init=res["init"], seconds=round(time.time() - t1, 1),
                ))
                print(f"      {tb:>8} {rspec:>8}   identity={rows[-1]['identity']:.3f}  "
                      f"search={rows[-1]['search']:.3f}  control={rows[-1]['control']:.3f}  "
                      f"(+{rows[-1]['lift_vs_identity']:.3f} vs identity, "
                      f"+{rows[-1]['lift_vs_control']:.3f} vs control, "
                      f"init={res['init']}, {rows[-1]['seconds']:.0f}s)", flush=True)
        flush()      # incremental, so a wall-clock timeout still leaves usable results

    # ---- summary ----
    n = len(rows)
    print(f"\n=== structure of the tag grid, mean over {n} rows ===")
    for k in ("pos_share", "rank1_sign_fit", "row_share", "col_share", "resid_share"):
        print(f"{k:>16}: {sum(r[k] for r in rows) / n:.4f}")
    if args.diagnostics_only:
        flush()
        if args.out:
            print(f"\nwrote {args.out}")
        return

    # ---- mean over tensors, per (type block, rule) ----
    print("\n=== mean over tensors (fraction of the 1x16 ceiling) ===")
    print(f"{'type_block':>10} {'rule':>8} {'identity':>9} {'search':>8} {'control':>8} "
          f"{'lift/id':>8} {'lift/ctl':>9}")
    for tb in args.type_blocks:
        for rspec in args.rules:
            sel = [r for r in rows if r["type_block"] == tb and r["rule"] == rspec]
            if not sel:
                continue
            avg = lambda k: sum(r[k] for r in sel) / len(sel)
            print(f"{tb:>10} {rspec:>8} {avg('identity'):9.3f} {avg('search'):8.3f} "
                  f"{avg('control'):8.3f} {avg('lift_vs_identity'):8.3f} "
                  f"{avg('lift_vs_control'):9.3f}")

    flush()
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
