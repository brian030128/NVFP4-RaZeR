"""
    WHICH TILES does `hess` decline, and can a weight-only rule reproduce that decision?

    The importance-weighted criterion improves perplexity by electing E0M3 on ~33% fewer 8x64 tiles
    (0.263 -> 0.176 on Llama-3.1-8B) and by electing on lower-importance channels. So the value is
    entirely in the tiles it DECLINES that plain MSE would have taken. This script isolates that
    set and asks what distinguishes it, using features that need no calibration data.

    Tiles are bucketed by the two decisions:

        both      -- MSE elects, hess elects            (agreement, keep)
        declined  -- MSE elects, hess does NOT          <- the tiles that matter
        added     -- hess elects, MSE does not
        neither   -- both decline

    and profiled on weight-only statistics:

        max_maxrms   the peakiest block in the tile (max over blocks of block_max / block_rms)
        mean_maxrms  average peakedness
        peaked_frac  fraction of the tile's blocks above max/rms 2.5
        energy_conc  largest block energy / total tile energy -- can one block carry the vote?
        gain_conc    largest |gain| / total |gain|            -- the same, in gain terms

    plus the calibration-only statistics (mean and max block importance, and importance
    concentration) so the weight-only features can be checked against what hess actually sees.

    Then it scores candidate CALIBRATION-FREE rules by how often they reproduce hess's decision,
    against the trivial baselines. A rule is only interesting if it beats "always agree with MSE".

        python analyze_hess_rules.py --model_name llama-3.1-8b-local
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.reorder import _labels_from_order, elect_mask, gain_features, scale_block_gain


def tile_stats(x, block_m, chunks):
    """ (M, N) block statistic -> (n_tile, block_m*chunks) with the tile's blocks on the last axis. """
    M, N = x.shape
    x = torch.nn.functional.pad(x, (0, (-N) % chunks, 0, (-M) % block_m))
    nrg, ncg = x.shape[0] // block_m, x.shape[1] // chunks
    return x.reshape(nrg, block_m, ncg, chunks).permute(0, 2, 1, 3).reshape(-1, block_m * chunks)


def elect(gain, block_m, chunks, rule, margin):
    t = tile_stats(gain, block_m, chunks)
    phi = gain_features(t).sum(dim=1)
    return elect_mask(phi, rule, margin, block_m * chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--clip", type=str, default="a1")
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

    bm, bk = (int(v) for v in args.type_block.split("x"))
    chunks = bk // args.groupsize
    gsz = args.groupsize

    buckets = {k: [] for k in ("both", "declined", "added", "neither")}
    rule_hits = {k: [0, 0] for k in ("mse", "sqnr", "relgain", "veto2.5", "veto3.0", "mse_h3")}
    rows = []

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % args.layer_stride != 0:
            continue
        if args.projections and not any(name.endswith(p) for p in args.projections):
            continue
        imp = imp_all.get(name)
        if imp is None:
            continue

        w = mod.weight.data.float().cpu()
        gs = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        ws = w / gs
        impc = imp.float().cpu()

        g_mse, e2, _ = scale_block_gain(ws, gsz, "mse", args.clip, return_losses=True)
        g_hes = scale_block_gain(ws, gsz, "mse", args.clip, importance=impc)
        g_sqn = scale_block_gain(ws, gsz, "sqnr", args.clip)
        g_rel = g_mse / e2.clamp(min=1e-30)

        blk = ws.reshape(ws.shape[0], -1, gsz)
        maxrms = blk.abs().amax(dim=-1) / blk.pow(2).mean(dim=-1).sqrt().clamp(min=1e-30)
        energy = blk.pow(2).sum(dim=-1)
        bimp = impc.reshape(1, -1, gsz).mean(dim=-1).expand_as(energy).contiguous()

        e_mse = elect(g_mse, bm, chunks, args.rule, args.margin)
        e_hes = elect(g_hes, bm, chunks, args.rule, args.margin)

        t_maxrms, t_energy = tile_stats(maxrms, bm, chunks), tile_stats(energy, bm, chunks)
        t_imp, t_gain = tile_stats(bimp, bm, chunks), tile_stats(g_mse, bm, chunks)

        feats = dict(
            max_maxrms=t_maxrms.amax(dim=1), mean_maxrms=t_maxrms.mean(dim=1),
            peaked_frac=(t_maxrms > 2.5).to(torch.float32).mean(dim=1),
            energy_conc=t_energy.amax(dim=1) / t_energy.sum(dim=1).clamp(min=1e-30),
            gain_conc=t_gain.abs().amax(dim=1) / t_gain.abs().sum(dim=1).clamp(min=1e-30),
            imp_mean=t_imp.mean(dim=1), imp_max=t_imp.amax(dim=1),
            imp_conc=t_imp.amax(dim=1) / t_imp.sum(dim=1).clamp(min=1e-30),
        )
        cat = {"both": e_mse & e_hes, "declined": e_mse & ~e_hes,
               "added": ~e_mse & e_hes, "neither": ~e_mse & ~e_hes}
        for k, mask in cat.items():
            if mask.any():
                buckets[k].append({f: float(v[mask].mean()) for f, v in feats.items()} |
                                  {"n": int(mask.sum())})

        # candidate calibration-free rules, scored on agreement with hess's decision
        cands = {
            "mse":     e_mse,
            "sqnr":    elect(g_sqn, bm, chunks, args.rule, args.margin),
            "relgain": elect(g_rel, bm, chunks, args.rule, args.margin),
            "veto2.5": e_mse & (feats["max_maxrms"] <= 2.5),
            "veto3.0": e_mse & (feats["max_maxrms"] <= 3.0),
            "mse_h3":  elect(g_mse, bm, chunks, "harm", 3.0),
        }
        for k, v in cands.items():
            rule_hits[k][0] += int((v == e_hes).sum())
            rule_hits[k][1] += int(e_hes.numel())

        rows.append(dict(layer=re.sub(r"^model\.", "", name),
                         elect_mse=round(float(e_mse.float().mean()), 4),
                         elect_hess=round(float(e_hes.float().mean()), 4),
                         declined=round(float((e_mse & ~e_hes).float().mean()), 4)))

    print(f"\n=== tile buckets, mean weight-only features ({len(rows)} tensors) ===")
    hdr = ["max_maxrms", "mean_maxrms", "peaked_frac", "energy_conc", "gain_conc",
           "imp_mean", "imp_max", "imp_conc"]
    print(f"{'bucket':>9} {'share':>7} " + " ".join(f"{h:>11}" for h in hdr))
    tot = sum(sum(b["n"] for b in v) for v in buckets.values())
    for k, v in buckets.items():
        if not v:
            continue
        n = sum(b["n"] for b in v)
        avg = {h: sum(b[h] * b["n"] for b in v) / n for h in hdr}
        print(f"{k:>9} {n / tot:7.3f} " + " ".join(f"{avg[h]:11.4f}" for h in hdr))

    print(f"\n=== calibration-free rules vs hess's decision ===")
    print(f"{'rule':>10} {'agreement':>10}")
    for k, (hit, n) in sorted(rule_hits.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"{k:>10} {hit / max(n, 1):10.4f}")
    print("\n`mse` is the do-nothing baseline: any rule must beat it to be worth anything.")

    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
