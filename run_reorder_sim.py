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
    _labels_from_order, additive_shares, election_stats, interaction_structure,
    scale_block_gain, search_permutation, shuffle_control,
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
    ap.add_argument("--clip", type=str, default="a1",
                    help="Clip preset for the tag grid. `a1` is the configuration CLAUDE.md "
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
    ap.add_argument("--axes", type=str, default="both", choices=["both", "rows", "cols"],
                    help="Which permutations the search may use. MUST match the quant_mix_4_6 "
                         "mode being compared against: cocl->both, coclcol->cols, coclrow->rows.")
    ap.add_argument("--search_rule", type=str, default="",
                    help="Objective the SEARCH maximizes, if different from the evaluation rule. "
                         "e.g. --search_rule purity --rules h1.5 searches for agreement inside "
                         "tiles and then scores the result under the deployed election.")
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
        gain, err_e2m1, _ = scale_block_gain(w / gscale, args.groupsize, args.metric, args.clip,
                                             return_losses=True)
        # The quantization error actually being paid, if every block used E2M1 (= plain NVFP4 under
        # this clip preset). This is what turns "fraction of the 1x16 ceiling" into "fraction of the
        # MSE", which is the number that says whether any of this matters.
        e2m1_total = float(err_e2m1.to(torch.float64).sum())
        ctrlgain = None if args.no_control else shuffle_control(
            gain, torch.Generator().manual_seed(args.seed + 1))

        pos_share = float((gain > 0).to(torch.float32).mean())
        rank1     = sign_structure(gain)
        r_sh, c_sh, e_sh = additive_shares(gain)
        ceil_pct  = 100.0 * float(gain.clamp(min=0).to(torch.float64).sum()) / e2m1_total
        print(f"\n  {name}  grid={tuple(gain.shape)}  E0M3-preferring cells={pos_share:.3f}  "
              f"rank1 sign fit={rank1:.3f}  variance: row={r_sh:.3f} col={c_sh:.3f} "
              f"resid={e_sh:.3f}  1x16 ceiling={ceil_pct:.2f}% of MSE  "
              f"({time.time() - t0:.1f}s)", flush=True)

        if args.diagnostics_only:
            # The additive decomposition above is blind to CO-CLUSTER structure: rows that fall
            # into groups with different PROFILES have a_i ~ 0 and land wholly in the residual,
            # yet are exactly what an 8x64 tile could exploit. So probe profile similarity too,
            # and probe it against a cell-shuffled copy -- these statistics mean nothing in
            # absolute terms, only as an excess over the structureless baseline.
            gen  = torch.Generator().manual_seed(args.seed + 2)
            real = interaction_structure(gain, generator=gen)
            ctrl = interaction_structure(
                shuffle_control(gain, torch.Generator().manual_seed(args.seed + 3)),
                generator=torch.Generator().manual_seed(args.seed + 2))
            print(f"      profile similarity   row |corr|={real['row_corr_abs']:.4f} "
                  f"(shuffled {ctrl['row_corr_abs']:.4f})   p99.9={real['row_corr_p999']:.4f} "
                  f"({ctrl['row_corr_p999']:.4f})", flush=True)
            print(f"      profile similarity   col |corr|={real['col_corr_abs']:.4f} "
                  f"(shuffled {ctrl['col_corr_abs']:.4f})   p99.9={real['col_corr_p999']:.4f} "
                  f"({ctrl['col_corr_p999']:.4f})", flush=True)
            print(f"      singular spectrum    sv1={real['sv1']:.4f} ({ctrl['sv1']:.4f})  "
                  f"top4={real['sv_top4']:.4f} ({ctrl['sv_top4']:.4f})  "
                  f"top16={real['sv_top16']:.4f} ({ctrl['sv_top16']:.4f})", flush=True)

            rec = dict(model=tag, tensor=name, type_block="-", rule="-",
                       clip=args.clip, metric=args.metric,
                       pos_share=round(pos_share, 4), rank1_sign_fit=round(rank1, 4),
                       row_share=round(r_sh, 4), col_share=round(c_sh, 4),
                       resid_share=round(e_sh, 4),
                       ceiling_pct_of_mse=round(ceil_pct, 4))
            for k, v in real.items():
                rec[k] = round(v, 6)
            for k, v in ctrl.items():
                rec[f"ctrl_{k}"] = round(v, 6)
            rows.append(rec)
            flush()
            continue

        for tb in args.type_blocks:
            bm, bk = parse_type_block(tb)
            for rspec in args.rules:
                rule, margin = parse_rule(rspec)
                # the search may optimize a DIFFERENT objective from the one it is scored under
                srule, smargin = ((args.search_rule, 0.0) if args.search_rule in
                                  ("purity", "puritycount") else
                                  (parse_rule(args.search_rule) if args.search_rule
                                   else (rule, margin)))
                t1 = time.time()
                res = search_permutation(gain, bm, bk, args.groupsize, srule, smargin,
                                         rounds=args.rounds, swap_samples=args.swap_samples,
                                         seed=args.seed, axes=args.axes)
                ctrl, ctrl_res = float("nan"), None
                if ctrlgain is not None:
                    ctrl_res = search_permutation(ctrlgain, bm, bk, args.groupsize,
                                                  srule, smargin,
                                                  rounds=args.rounds,
                                                  swap_samples=args.swap_samples,
                                                  seed=args.seed, axes=args.axes)
                    ctrl = ctrl_res["recovered"]
                # How the aggregate gain was banked: from homogeneous tiles, or by packing
                # losers in with winners until the sum clears the bar? Same aggregate, very
                # different perplexity behaviour -- see reorder.election_stats.
                chunks = bk // args.groupsize
                Mg, Ng = gain.shape
                pad    = torch.nn.functional.pad(
                    gain, (0, (-Ng) % chunks, 0, (-Mg) % bm)).to(torch.float64)
                nrg, ncg = pad.shape[0] // bm, pad.shape[1] // chunks
                ident_lab = (_labels_from_order(torch.arange(pad.shape[0]), bm, pad.shape[0]),
                             _labels_from_order(torch.arange(pad.shape[1]), chunks, pad.shape[1]))
                found_lab = (_labels_from_order(res["row_perm"], bm, pad.shape[0]),
                             _labels_from_order(res["chunk_perm"], chunks, pad.shape[1]))
                es = lambda lab: election_stats(pad, lab[0], lab[1], nrg, ncg, rule, margin,
                                                bm * chunks, e2m1_total)
                st_id, st_se = es(ident_lab), es(found_lab)
                # purity of the CONTROL's own best partition -- without it, "purity rose" is the
                # same uncontrolled comparison that made the realized-gain numbers misleading
                st_ct = None
                if ctrl_res is not None:
                    cpad = torch.nn.functional.pad(
                        ctrlgain, (0, (-Ng) % chunks, 0, (-Mg) % bm)).to(torch.float64)
                    ct_lab = (_labels_from_order(ctrl_res["row_perm"], bm, cpad.shape[0]),
                              _labels_from_order(ctrl_res["chunk_perm"], chunks, cpad.shape[1]))
                    st_ct = election_stats(cpad, ct_lab[0], ct_lab[1], nrg, ncg, rule, margin,
                                           bm * chunks, e2m1_total)
                # under a different search objective, res["recovered"] is in the SEARCH rule's
                # units; the deployed number is what election_stats reports under the eval rule
                # With --search_rule, res["score"]/["baseline"] are in the SEARCH objective's
                # units (e.g. total majority mass), not realized gain, so every derived column has
                # to be recomputed from election_stats under the EVALUATION rule -- including the
                # MSE-cut columns, which were left in purity units in an earlier run.
                ceil = res["ceiling"]
                if ceil > 0:
                    res["baseline"] = st_id["realized"]
                    res["score"]    = st_se["realized"]
                    res["baseline_recovered"] = st_id["realized"] / ceil
                    res["recovered"] = st_se["realized"] / ceil
                    if st_ct is not None:
                        ctrl = st_ct["realized"] / ceil
                _c = lambda k: (f"{st_ct[k]:.4f}" if st_ct else "n/a")
                print(f"      {'':>8} {'':>8}   PURITY count {st_id['purity_count']:.4f} -> "
                      f"{st_se['purity_count']:.4f} (shuffled {_c('purity_count')})   "
                      f"mass {st_id['purity_mass']:.4f} -> {st_se['purity_mass']:.4f} "
                      f"(shuffled {_c('purity_mass')})", flush=True)
                print(f"      {'':>8} {'':>8}   harmed blocks: identity="
                      f"{100 * st_id['harmed_share']:.2f}%  search="
                      f"{100 * st_se['harmed_share']:.2f}%   harm mass: "
                      f"{st_id['harm_pct_of_mse']:.3f}% -> {st_se['harm_pct_of_mse']:.3f}% of MSE"
                      f"   tiles electing E0M3: {100 * st_id['elected_tile_share']:.1f}% -> "
                      f"{100 * st_se['elected_tile_share']:.1f}%", flush=True)

                rows.append(dict(
                    model=tag, tensor=name, type_block=tb, rule=rspec,
                    purity_count_id=round(st_id["purity_count"], 5),
                    purity_count_se=round(st_se["purity_count"], 5),
                    purity_mass_id=round(st_id["purity_mass"], 5),
                    purity_mass_se=round(st_se["purity_mass"], 5),
                    purity_count_ct=round(st_ct["purity_count"], 5) if st_ct else float("nan"),
                    purity_mass_ct=round(st_ct["purity_mass"], 5) if st_ct else float("nan"),
                    harmed_id=round(100 * st_id["harmed_share"], 4),
                    harmed_se=round(100 * st_se["harmed_share"], 4),
                    harmmass_id=round(st_id["harm_pct_of_mse"], 4),
                    harmmass_se=round(st_se["harm_pct_of_mse"], 4),
                    elected_id=round(100 * st_id["elected_tile_share"], 4),
                    elected_se=round(100 * st_se["elected_tile_share"], 4),
                    clip=args.clip, metric=args.metric,
                    pos_share=round(pos_share, 4), rank1_sign_fit=round(rank1, 4),
                    row_share=round(r_sh, 4), col_share=round(c_sh, 4),
                    resid_share=round(e_sh, 4),
                    ceiling_pct_of_mse=round(ceil_pct, 4),
                    # the same three numbers as a PERCENTAGE OF THE QUANTIZATION MSE, which is what
                    # "how much does the total MSE go down" actually asks
                    identity_mse_cut=round(100.0 * res["baseline"] / e2m1_total, 4),
                    search_mse_cut=round(100.0 * res["score"] / e2m1_total, 4),
                    control_mse_cut=round(100.0 * ctrl * res["ceiling"] / e2m1_total, 4)
                    if e2m1_total else float("nan"),
                    identity=round(res["baseline_recovered"], 4),
                    search=round(res["recovered"], 4),
                    control=round(ctrl, 4),
                    lift_vs_identity=round(res["recovered"] - res["baseline_recovered"], 4),
                    lift_vs_control=round(res["recovered"] - ctrl, 4),
                    init=res["init"], seconds=round(time.time() - t1, 1),
                ))
                print(f"      {tb:>8} {rspec:>8}   MSE cut: identity="
                      f"{rows[-1]['identity_mse_cut']:.3f}%  search={rows[-1]['search_mse_cut']:.3f}%"
                      f"  control={rows[-1]['control_mse_cut']:.3f}%  "
                      f"(ceiling {ceil_pct:.3f}%)", flush=True)
                print(f"      {'':>8} {'':>8}   identity={rows[-1]['identity']:.3f}  "
                      f"search={rows[-1]['search']:.3f}  control={rows[-1]['control']:.3f}  "
                      f"(+{rows[-1]['lift_vs_identity']:.3f} vs identity, "
                      f"+{rows[-1]['lift_vs_control']:.3f} vs control, "
                      f"init={res['init']}, {rows[-1]['seconds']:.0f}s)", flush=True)
        flush()      # incremental, so a wall-clock timeout still leaves usable results

    # ---- summary ----
    n = len(rows)
    print(f"\n=== structure of the tag grid, mean over {n} rows ===")
    keys = ["pos_share", "rank1_sign_fit", "row_share", "col_share", "resid_share",
            "ceiling_pct_of_mse"]
    if args.diagnostics_only:
        keys += ["row_corr_abs", "ctrl_row_corr_abs", "row_corr_p999", "ctrl_row_corr_p999",
                 "col_corr_abs", "ctrl_col_corr_abs", "col_corr_p999", "ctrl_col_corr_p999",
                 "sv1", "ctrl_sv1", "sv_top4", "ctrl_sv_top4", "sv_top16", "ctrl_sv_top16"]
    for k in keys:
        if k in rows[0]:
            print(f"{k:>20}: {sum(r[k] for r in rows) / n:.4f}")
    if args.diagnostics_only:
        flush()
        if args.out:
            print(f"\nwrote {args.out}")
        return

    # ---- mean over tensors, per (type block, rule) ----
    print("\n=== mean over tensors (fraction of the 1x16 ceiling) ===")
    print(f"{'type_block':>10} {'rule':>8} {'identity':>9} {'search':>8} {'control':>8} "
          f"{'lift/id':>8} {'lift/ctl':>9} | {'MSEcut_id':>9} {'MSEcut_se':>9} {'MSEcut_ct':>9} "
          f"{'ceiling':>8}")
    for tb in args.type_blocks:
        for rspec in args.rules:
            sel = [r for r in rows if r["type_block"] == tb and r["rule"] == rspec]
            if not sel:
                continue
            avg = lambda k: sum(r[k] for r in sel) / len(sel)
            print(f"{tb:>10} {rspec:>8} {avg('identity'):9.3f} {avg('search'):8.3f} "
                  f"{avg('control'):8.3f} {avg('lift_vs_identity'):8.3f} "
                  f"{avg('lift_vs_control'):9.3f} | {avg('identity_mse_cut'):8.3f}% "
                  f"{avg('search_mse_cut'):8.3f}% {avg('control_mse_cut'):8.3f}% "
                  f"{avg('ceiling_pct_of_mse'):7.3f}%")
            print(f"{'':>10} {'':>8}   PURITY count {avg('purity_count_id'):.4f} -> "
                  f"{avg('purity_count_se'):.4f} (shuffled {avg('purity_count_ct'):.4f})   "
                  f"mass {avg('purity_mass_id'):.4f} -> {avg('purity_mass_se'):.4f} "
                  f"(shuffled {avg('purity_mass_ct'):.4f})")
            print(f"{'':>10} {'':>8}   harmed blocks {avg('harmed_id'):.2f}% -> "
                  f"{avg('harmed_se'):.2f}%   harm mass {avg('harmmass_id'):.3f}% -> "
                  f"{avg('harmmass_se'):.3f}% of MSE   tiles electing E0M3 "
                  f"{avg('elected_id'):.1f}% -> {avg('elected_se'):.1f}%")

    flush()
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
