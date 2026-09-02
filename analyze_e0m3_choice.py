"""
    WHICH scale blocks want E0M3, how much of the tensor that is, and what changes when the
    selection loss is importance-weighted (`hess`) instead of plain weight MSE.

    Three questions:

      1. What fraction of blocks prefer E0M3 at 1x16, and what fraction of TILES elect it at 8x64 --
         under MSE and under the diagonal-Hessian criterion.
      2. Where do the two criteria disagree? `hess` weights each element by E[x_j^2] of the channel
         it multiplies, so a block containing one high-importance channel is scored almost entirely
         by that channel. The flip rate says how much of the election that changes.
      3. Does the choice track OUTLIERS? E2M1 is log-spaced -- fine near zero, coarse near the block
         max -- so it should suit peaked blocks with a large max/rms. E0M3 is uniform, so it should
         suit flat blocks. If that is the mechanism, gain should correlate negatively with max/rms,
         and the E0M3-preferring population should have visibly lower max/rms and kurtosis.

    Reports the block-level statistics split by which grid wins, plus rank correlations.

        python analyze_e0m3_choice.py --model_name llama-3.1-8b-local --layer_stride 8
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
    _labels_from_order, elect_mask, gain_features, scale_block_gain,
)


def spearman(a, b):
    """ Rank correlation, on a subsample to keep the sort cheap. """
    n = min(a.numel(), 200000)
    idx = torch.randperm(a.numel())[:n]
    ra = torch.argsort(torch.argsort(a.flatten()[idx])).to(torch.float64)
    rb = torch.argsort(torch.argsort(b.flatten()[idx])).to(torch.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-30))


def tile_elect_share(gain, block_m, chunks, rule, margin):
    """ Fraction of 8x64 tiles that elect E0M3, for a given tag grid. """
    M, N = gain.shape
    g = torch.nn.functional.pad(gain, (0, (-N) % chunks, 0, (-M) % block_m)).to(torch.float64)
    nrg, ncg = g.shape[0] // block_m, g.shape[1] // chunks
    feat = gain_features(g)
    A = torch.zeros(g.shape[0], ncg, feat.shape[-1], dtype=feat.dtype)
    A.index_add_(1, _labels_from_order(torch.arange(g.shape[1]), chunks, g.shape[1]), feat)
    phi = torch.zeros(nrg, ncg, feat.shape[-1], dtype=feat.dtype)
    phi.index_add_(0, _labels_from_order(torch.arange(g.shape[0]), block_m, g.shape[0]), A)
    return float(elect_mask(phi, rule, margin, block_m * chunks).to(torch.float32).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--clip", type=str, default="headx")
    ap.add_argument("--type_block", type=str, default="8x64")
    ap.add_argument("--rule", type=str, default="harm")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--calib_batches", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantize import collect_importance

    here = os.path.dirname(os.path.abspath(__file__))
    path = json.load(open(os.path.join(here, "model2path.json")))[args.model_name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(data["text"][:20000]), return_tensors="pt").input_ids
    calib = [enc[:, i * args.seq_len:(i + 1) * args.seq_len] for i in range(args.calib_batches)]
    imp_all = collect_importance(model, calib, device=model.device)
    print(f"collected importance for {len(imp_all)} layers\n", flush=True)

    bm, bk = (int(v) for v in args.type_block.split("x"))
    chunks = bk // args.groupsize

    print(f"{'layer':>30} {'E0M3@1x16':>10} {'hess':>7} | {'tiles@8x64':>10} {'hess':>7} | "
          f"{'flip%':>6} | {'maxrms E0M3':>12} {'E2M1':>7} | {'rho(g,maxrms)':>14}")
    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % args.layer_stride != 0:
            continue
        if args.projections and not any(name.endswith(p) for p in args.projections):
            continue
        short = re.sub(r"^model\.", "", name)
        imp = imp_all.get(name)
        if imp is None:
            continue

        w = mod.weight.data.float().cpu()
        gs = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        ws = w / gs
        impc = imp.float().cpu()

        g_mse, e2_mse, _ = scale_block_gain(ws, args.groupsize, "mse", args.clip,
                                            return_losses=True)
        g_hes = scale_block_gain(ws, args.groupsize, "mse", args.clip, importance=impc)

        # CALIBRATION-FREE candidates for the same job. Plain MSE lets a few high-energy blocks
        # carry a tile; each of these removes that bias in a different way, using weights only.
        g_sqnr = scale_block_gain(ws, args.groupsize, "sqnr", args.clip)
        benergy = ws.reshape(ws.shape[0], -1, args.groupsize).pow(2).sum(dim=-1)
        g_rel  = g_mse / e2_mse.clamp(min=1e-30)          # relative improvement per block
        g_norm = g_mse / benergy.clamp(min=1e-30)         # gain per unit block energy

        blocks = ws.reshape(ws.shape[0], -1, args.groupsize)
        bmax = blocks.abs().amax(dim=-1)
        brms = blocks.pow(2).mean(dim=-1).sqrt().clamp(min=1e-30)
        maxrms = bmax / brms                                     # peakedness / outlier-ness
        c = blocks - blocks.mean(dim=-1, keepdim=True)
        kurt = (c.pow(4).mean(dim=-1) / c.pow(2).mean(dim=-1).clamp(min=1e-30).pow(2))
        bimp = impc.reshape(1, -1, args.groupsize).mean(dim=-1).expand_as(bmax)

        p_mse = (g_mse > 0)
        p_hes = (g_hes > 0)
        flip = float((p_mse != p_hes).to(torch.float32).mean())

        rec = dict(
            model=args.model_name, layer=short,
            e0m3_share_mse=round(float(p_mse.to(torch.float32).mean()), 4),
            e0m3_share_hess=round(float(p_hes.to(torch.float32).mean()), 4),
            tile_elect_mse=round(tile_elect_share(g_mse, bm, chunks, args.rule, args.margin), 4),
            tile_elect_hess=round(tile_elect_share(g_hes, bm, chunks, args.rule, args.margin), 4),
            flip_share=round(flip, 4),
            maxrms_e0m3=round(float(maxrms[p_mse].mean()), 4),
            maxrms_e2m1=round(float(maxrms[~p_mse].mean()), 4),
            kurt_e0m3=round(float(kurt[p_mse].mean()), 4),
            kurt_e2m1=round(float(kurt[~p_mse].mean()), 4),
            imp_e0m3=round(float(bimp[p_hes].mean()), 6),
            imp_e2m1=round(float(bimp[~p_hes].mean()), 6),
            rho_gain_maxrms=round(spearman(g_mse, maxrms), 4),
            rho_gain_kurt=round(spearman(g_mse, kurt), 4),
            rho_hessgain_imp=round(spearman(g_hes, bimp), 4),
            # how well each CALIBRATION-FREE criterion reproduces the importance-weighted ranking
            rho_hess_mse=round(spearman(g_hes, g_mse), 4),
            rho_hess_sqnr=round(spearman(g_hes, g_sqnr), 4),
            rho_hess_rel=round(spearman(g_hes, g_rel), 4),
            rho_hess_norm=round(spearman(g_hes, g_norm), 4),
            # and how often each agrees with hess on the actual per-block VERDICT
            agree_mse=round(float(((g_mse > 0) == p_hes).to(torch.float32).mean()), 4),
            agree_sqnr=round(float(((g_sqnr > 0) == p_hes).to(torch.float32).mean()), 4),
            agree_rel=round(float(((g_rel > 0) == p_hes).to(torch.float32).mean()), 4),
            # tile election share under each calibration-free criterion, vs hess's 0.176
            tile_elect_sqnr=round(tile_elect_share(g_sqnr, bm, chunks, args.rule, args.margin), 4),
            tile_elect_rel=round(tile_elect_share(g_rel, bm, chunks, args.rule, args.margin), 4),
        )
        rows.append(rec)
        print(f"{short:>30} {rec['e0m3_share_mse']:10.3f} {rec['e0m3_share_hess']:7.3f} | "
              f"{rec['tile_elect_mse']:10.3f} {rec['tile_elect_hess']:7.3f} | "
              f"{100 * flip:5.1f}% | {rec['maxrms_e0m3']:12.3f} {rec['maxrms_e2m1']:7.3f} | "
              f"{rec['rho_gain_maxrms']:14.3f}", flush=True)

    n = len(rows)
    a = lambda k: sum(r[k] for r in rows) / n
    print(f"\n=== mean over {n} tensors ===")
    print(f"  blocks preferring E0M3 at 1x16:   MSE {a('e0m3_share_mse'):.3f}   "
          f"hess {a('e0m3_share_hess'):.3f}")
    print(f"  tiles electing E0M3 at {args.type_block}:      MSE {a('tile_elect_mse'):.3f}   "
          f"hess {a('tile_elect_hess'):.3f}")
    print(f"  blocks whose preferred grid FLIPS under hess: {100 * a('flip_share'):.1f}%")
    print(f"\n  block max/rms   E0M3-preferring {a('maxrms_e0m3'):.3f}   "
          f"E2M1-preferring {a('maxrms_e2m1'):.3f}")
    print(f"  block kurtosis  E0M3-preferring {a('kurt_e0m3'):.3f}   "
          f"E2M1-preferring {a('kurt_e2m1'):.3f}")
    print(f"  mean importance E0M3-elected    {a('imp_e0m3'):.6f}   "
          f"E2M1 {a('imp_e2m1'):.6f}")
    print(f"\n  Spearman rho(gain, max/rms)   {a('rho_gain_maxrms'):+.3f}   "
          f"(negative = outlier blocks prefer E2M1)")
    print(f"  Spearman rho(gain, kurtosis)  {a('rho_gain_kurt'):+.3f}")
    print(f"  Spearman rho(hess gain, block importance) {a('rho_hessgain_imp'):+.3f}")

    print(f"\n=== can a CALIBRATION-FREE criterion stand in for hess? ===")
    print(f"{'criterion':>12} {'rho vs hess':>12} {'verdict agree':>14} {'tiles elected':>14}")
    print(f"{'hess':>12} {1.0:12.3f} {1.0:14.3f} {a('tile_elect_hess'):14.3f}   <- target")
    print(f"{'mse':>12} {a('rho_hess_mse'):12.3f} {a('agree_mse'):14.3f} "
          f"{a('tile_elect_mse'):14.3f}")
    print(f"{'sqnr':>12} {a('rho_hess_sqnr'):12.3f} {a('agree_sqnr'):14.3f} "
          f"{a('tile_elect_sqnr'):14.3f}")
    print(f"{'relgain':>12} {a('rho_hess_rel'):12.3f} {a('agree_rel'):14.3f} "
          f"{a('tile_elect_rel'):14.3f}")
    print(f"{'gain/energy':>12} {a('rho_hess_norm'):12.3f} {'-':>14} {'-':>14}")
    print("\nA criterion that beats `mse` on rho AND lands near hess's tile-election share is a")
    print("calibration-free candidate; one that matches mse is just mse in different units.")

    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------
# Appended: can a CALIBRATION-FREE criterion reproduce what `hess` does?
#
# `hess` needs E[x_j^2]. CLAUDE.md already rejected the two obvious weight-only proxies for it
# (preceding RMSNorm gamma^2 flips sign on the MLP projections; column energy ||W_:,j||^2 has no
# consistent sign). So the route is not to predict importance, but to remove the bias that made
# plain MSE wrong: summed squared error lets a few HIGH-ENERGY blocks carry a whole tile, and those
# are exactly the peaked, outlier-bearing blocks that the analysis above shows prefer E2M1.
#
# Three calibration-free candidates, all already in the quantizer:
#   sqnr      -- per-block error normalized by that block's own signal energy
#   relgain   -- gain_b / loss_E2M1(b), the relative improvement
#   logenergy -- MSE gain divided by block energy, a cruder version of the same idea
#
# The question is which of them ranks blocks the way the importance-weighted gain does.
