#!/usr/bin/env python3
"""Kernel-opt analysis (CPU only): how much of the format dispatch could be skipped, and at which level.

    python experiments/kernel_opt/dispatch_ceiling.py --out JSON [--md MD]

For the FlipQuant (ours) TM-OPT+TC maps of the paper run (/home/dev/n16k64_campaign/paper/artifacts/<model>_tc_<unit>
.mixfp4map; Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B) at 16x64, 256x64 (executed as 16x64 granules by
n16k64_wA: every 256x64 tile is 16 identical 16-row granules; the map files store them that way) and 8x64 (n8k64_wB
family).

Granules and spans, as the kernels execute them (from their TiledMma; sm120/kernel/src/mixed_nvfp4_gemm.cu):
- weights on A (n16k64_wA, 4x2 warps): a granule is 16 weight rows x 64 K. A CTA's weight panel is 128 rows (8 granule
  rows); warp w_m (0..3) owns granule rows w_m and w_m + 4 of its panel (CuTe tiles the 64-row warp layout twice).
- weights on B (n8k64_wB, 1x8 warps, PERM_N = 128): a granule is 8 weight columns x 64 K. A CTA's weight panel is 128
  columns (16 granules); warp w owns granules g(w) and g(w) + 8, g(w) = 2 (w % 2) + (w // 2) % 2 + 4 (w // 4).
  The 64-column builds (n8k64_wB_m*, n8k64_wB_n64: 4 warps along 64 columns, PERM_N = 64): panel 8 granules, warp w owns
  g(w) and g(w) + 4, g(w) = 2 (w % 2) + (w // 2) % 2.
- A k_tile is 128 K = 2 granule columns. A warp's dispatch pattern per k_tile is its 2 granules x 2 k-blocks = 4 bits:
  0 = all E2M1, 15 = all E0M3.
Metrics per module, and aggregated over a forward as the plain share of modules and FLOP-weighted (weight = out x in,
the module's share of the GEMM work at any token count):
  zero_module     the module has no E0M3 tile (a whole-module skip could run a no-dispatch kernel)
  cta_panel       share of CTA weight panels with no E0M3 tile over the full K (a per-CTA skip)
  warp_span       share of warp spans with no E0M3 tile over the full K (a per-warp skip)
  pattern_0/15/other   shares of per-(warp, k_tile) dispatch patterns; taken_branches = mean popcount of the pattern,
                  the branches taken to reach its arm in today's tree (all-E2M1 falls through, results/kernel_opt/e0m3)
Each also: 'perm' = after the best output-channel permutation for that metric (granule rows / columns holding E0M3
clustered: panels and warp spans exactly optimal; the k_tile patterns by pairing rows with similar k_tile masks, a
heuristic, so an estimate), and 'random' = seeded Bernoulli tags at each module's own density (the same metrics).
A sensitivity variant pairs a warp's two granules as adjacent rows instead of the kernels' strided pairs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import mapio  # noqa: E402

ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
UNITS = ('16x64', '256x64', '8x64')


def layout(unit):
    """(granule rows per panel, list of warp granule pairs) for the kernel that executes this unit."""
    if unit in ('16x64', '256x64'):
        return {'wA128': (8, [(w, w + 4) for w in range(4)])}
    g = [2 * (w % 2) + (w // 2) % 2 + 4 * (w // 4) for w in range(8)]
    g64 = [2 * (w % 2) + (w // 2) % 2 for w in range(4)]
    return {'wB128': (16, [(x, x + 8) for x in g]), 'wB64': (8, [(x, x + 4) for x in g64])}


def adjacent_layout(unit):
    if unit in ('16x64', '256x64'):
        return {'wA128_adjacent': (8, [(2 * w, 2 * w + 1) for w in range(4)])}
    return {'wB128_adjacent': (16, [(2 * w, 2 * w + 1) for w in range(8)]), 'wB64_adjacent': (8, [(2 * w, 2 * w + 1) for w in range(4)])}


def pad_rows(mask, panel):
    r = (-mask.shape[0]) % panel
    return np.concatenate([mask, np.zeros((r, mask.shape[1]), bool)]) if r else mask


def metrics(mask, panel, pairs):
    """mask: bool [granule rows, K granules] -> dict of counts (numerators, denominators)."""
    m = pad_rows(mask, panel)
    rows_any = m.any(axis=1)
    npan = m.shape[0] // panel
    pan = rows_any.reshape(npan, panel)
    cta_clean = int((~pan.any(axis=1)).sum())
    warp_clean = sum(int((~(pan[:, a] | pan[:, b])).sum()) for a, b in pairs)
    kt = m.shape[1] // 2
    # per-(warp, k_tile) pattern: bits = granule a k-block 0, a k-block 1, b k-block 0, b k-block 1 (bit order does not
    # change popcount or the 0 / 15 classes)
    mk = m[:, :2 * kt].reshape(m.shape[0], kt, 2)
    p0 = p15 = pc = 0
    hist = np.zeros(16, np.int64)
    for a, b in pairs:
        ra = mk.reshape(npan, panel, kt, 2)[:, a]
        rb = mk.reshape(npan, panel, kt, 2)[:, b]
        pop = ra.sum(axis=2) + rb.sum(axis=2)
        p0 += int((pop == 0).sum())
        p15 += int((pop == 4).sum())
        pc += int(pop.sum())
        # the kernels' bit order (collective, pattern bit layout): granule flags of the warp's first atom group in the
        # low bits; here bit 0/1 = granule a at k-blocks 0/1, bit 2/3 = granule b (popcount and 0/15 do not depend on it)
        code = ra[..., 0].astype(np.int64) | (ra[..., 1].astype(np.int64) << 1) | (rb[..., 0].astype(np.int64) << 2) \
            | (rb[..., 1].astype(np.int64) << 3)
        hist += np.bincount(code.ravel(), minlength=16)
    nw = npan * len(pairs) * kt
    return dict(cta_clean=cta_clean, cta_total=npan, warp_clean=warp_clean, warp_total=npan * len(pairs),
                pat0=p0, pat15=p15, pat_total=nw, popcount_sum=pc, hist=hist.tolist())


def permuted(mask, panel, pairs):
    """Best-permutation counts: E0M3 granule rows clustered (exact for panels and warp spans), and rows paired by k_tile
    mask similarity for the patterns (a heuristic)."""
    rows_any = mask.any(axis=1)
    n = mask.shape[0]
    ndirty = int(rows_any.sum())
    npan = -(-n // panel)
    cta_clean = npan - (-(-ndirty // panel))
    nwarp = npan * len(pairs)
    warp_clean = nwarp - (-(-ndirty // 2))
    # patterns: order dirty rows by their k_tile mask (lexicographic on the packed mask, heaviest first) and pair
    # consecutive ones; clean rows pair with clean rows
    kt = mask.shape[1] // 2
    ktm = mask[:, :2 * kt].reshape(n, kt, 2)
    tile_any = ktm.any(axis=2)
    dirty = np.nonzero(rows_any)[0]
    order = sorted(dirty.tolist(), key=lambda r: (-int(tile_any[r].sum()), np.packbits(tile_any[r]).tobytes()))
    total_pairs = nwarp
    p0 = p15 = pc = 0
    pairs_d = [order[i:i + 2] for i in range(0, len(order), 2)]
    for pr in pairs_d:
        rows = ktm[pr]                      # (1 or 2, kt, 2)
        pop = rows.sum(axis=(0, 2))
        p0 += int((pop == 0).sum())
        p15 += int((pop == 4).sum())
        pc += int(pop.sum())
    clean_pairs = total_pairs - len(pairs_d)
    p0 += clean_pairs * kt
    return dict(cta_clean=cta_clean, cta_total=npan, warp_clean=warp_clean, warp_total=nwarp,
                pat0=p0, pat15=p15, pat_total=total_pairs * kt, popcount_sum=pc)


def aggregate(rows, key):
    """Plain share over modules and FLOP-weighted share, for a counts key pair."""
    num, den = key
    plain = float(np.mean([r[num] / r[den] for r in rows if r[den]]))
    wsum = sum(r['flops'] for r in rows if r[den])
    weighted = sum(r['flops'] * r[num] / r[den] for r in rows if r[den]) / wsum
    return plain, weighted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(20260930)
    res = dict(note=__doc__.split('\n\n')[0], models={})
    md = ['| model | unit | kernel layout | variant | E0M3 tiles | zero-E0M3 modules (plain / FLOP-w.) | all-E2M1 CTA panels | '
          'all-E2M1 warp spans | patterns all-E2M1 / all-E0M3 / other | mean taken branches per (warp, k_tile) |',
          '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for model in args.models.split(','):
        res['models'][model] = {}
        for unit in UNITS:
            path = ART / f'{model}_tc_{unit}.mixfp4map'
            header, masks, _ = mapio.read_map(path)
            tb = tuple(header['type_block'])
            per = {}
            layouts = dict(layout(unit), **adjacent_layout(unit))
            for mod in header['modules']:
                mask = masks[mod['name']].numpy().astype(bool)
                n, k = mod['weight_shape']
                # the 256x64 maps are stored as 16x64 granules already (policy 'TM-OPT+TC 256x64 as 16x64 granules'):
                # their masks need no expansion
                assert mask.shape[0] == -(-n // tb[0]), (mod['name'], mask.shape, tb)
                flops = n * k
                density = float(mask.mean())
                rand = rng.random(mask.shape) < density
                for lname, (panel, pairs) in layouts.items():
                    for variant, mm in (('map', mask), ('random', rand)):
                        c = metrics(mm, panel, pairs)
                        c.update(flops=flops, zero=int(not mm.any()), one=1, e0m3=int(mm.sum()), tiles=int(mm.size))
                        per.setdefault((lname, variant), []).append(c)
                    if not lname.endswith('adjacent'):
                        c = permuted(mask, panel, pairs)
                        c.update(flops=flops, zero=int(not mask.any()), one=1, e0m3=int(mask.sum()), tiles=int(mask.size))
                        per.setdefault((lname, 'perm'), []).append(c)
            out = res['models'][model][unit] = dict(map=str(path), type_block=list(tb), layouts={})
            for (lname, variant), rows in per.items():
                agg = {}
                for name, key in (('zero_module', ('zero', 'one')), ('cta_panel', ('cta_clean', 'cta_total')),
                                  ('warp_span', ('warp_clean', 'warp_total')), ('pattern_0', ('pat0', 'pat_total')),
                                  ('pattern_15', ('pat15', 'pat_total')), ('taken_branches', ('popcount_sum', 'pat_total'))):
                    agg[name] = dict(zip(('plain', 'flop_weighted'), aggregate(rows, key)))
                agg['pattern_other'] = {k: 1 - agg['pattern_0'][k] - agg['pattern_15'][k] for k in ('plain', 'flop_weighted')}
                agg['e0m3_share'] = sum(r['e0m3'] for r in rows) / sum(r['tiles'] for r in rows)
                if 'hist' in rows[0]:
                    # FLOP-weighted histogram of the 16 patterns (each module's shares weighted by out x in)
                    wsum = sum(r['flops'] for r in rows)
                    agg['pattern_hist_flop_weighted'] = [sum(r['flops'] * r['hist'][i] / r['pat_total'] for r in rows) / wsum
                                                         for i in range(16)]
                    agg['popcount_hist_flop_weighted'] = [sum(agg['pattern_hist_flop_weighted'][i] for i in range(16)
                                                              if bin(i).count('1') == c) for c in range(5)]
                agg['modules'] = len(rows)
                out['layouts'].setdefault(lname, {})[variant] = agg
                f = lambda d: f"{100 * d['plain']:.1f} / {100 * d['flop_weighted']:.1f} %"  # noqa: E731
                md.append(f"| {model} | {unit} | {lname} | {variant} | {100 * agg['e0m3_share']:.2f} % | {f(agg['zero_module'])} | "
                          f"{100 * agg['cta_panel']['flop_weighted']:.1f} % | {100 * agg['warp_span']['flop_weighted']:.1f} % | "
                          f"{100 * agg['pattern_0']['flop_weighted']:.1f} / {100 * agg['pattern_15']['flop_weighted']:.2f} / "
                          f"{100 * agg['pattern_other']['flop_weighted']:.1f} % | {agg['taken_branches']['flop_weighted']:.3f} |")
            print(model, unit, 'done', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    if args.md:
        args.md.write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
