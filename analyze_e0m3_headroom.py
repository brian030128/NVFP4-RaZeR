"""How much room does E0M3 have once the NVFP4 block scale is chosen per block?

For every 16-element scale block of every text linear layer, compute the block's squared
error under three candidates that share the NVFP4 global scale:
  S6  E2M1, block max -> 6 (plain NVFP4)
  S4  E2M1, block max -> 4 (FourOverSix's alternative)
  E0  E0M3, block max -> 7 (the MixFP4 alternative)
both unweighted (weight MSE) and weighted by per-input-channel E[x^2] from C4 train
windows, the diagonal-Hessian estimate of the layer output error. Reports how often and by
how much E0 beats S6 alone versus the best of {S6, S4}, and what survives when one E0M3
decision must cover a whole 8x64 or 256x64 tile. Also describes the blocks E0 wins on.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6_pair
from run_math_code_calibration import load_model
from run_train_map import CALIBRATIONS, c4_train_windows

TILES = {'8x64': (8, 64), '256x64': (256, 64)}


def block_sse(q, w, imp):
    d = (q.float() - w.float()).square()
    if imp is not None:
        d = d * imp[None, :]
    return d.reshape(-1, 16).sum(-1)


def tile_sum(x, shape, rows, cols):
    o, k = shape
    x = x.reshape(o, k // 16)
    x = torch.nn.functional.pad(x, (0, 0, 0, (-o) % rows))
    return x.reshape(-1, rows, k // cols, cols // 16).sum((1, 3))


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='llama1b_ins', choices=tuple(CALIBRATIONS))
    ap.add_argument('--windows', type=int, default=64)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    prior = json.loads((CALIBRATIONS[args.model] / 'report.json').read_text())
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    windows, _ = c4_train_windows(tok, args.windows)
    device = model.get_input_embeddings().weight.device
    sq = {n: torch.zeros(m.weight.shape[1], dtype=torch.float64, device=device) for n, m in modules.items()}
    count = [0]

    def hook(n):
        def f(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            sq[n] += x.square().sum(0)
        return f
    handles = [m.register_forward_pre_hook(hook(n)) for n, m in modules.items()]
    for w in windows:
        model(input_ids=w.to(device), use_cache=False)
        count[0] += w.shape[1]
    for h in handles:
        h.remove()

    kinds = {}
    for n, m in modules.items():
        kind = n.split('.')[-1]
        w = m.weight
        chosen, other, select_4 = quant_nvfp4_4over6_pair(w, 4, 16)
        s = select_4.reshape(-1, 1)
        s6 = torch.where(s, other.reshape(-1, 16), chosen.reshape(-1, 16)).view(w.shape)
        s4 = torch.where(s, chosen.reshape(-1, 16), other.reshape(-1, 16)).view(w.shape)
        e0 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        imp = (sq[n] / count[0]).float()
        # Block shape statistic: max / RMS (Gaussian-like 16 samples ~2.5, flat blocks lower).
        blocks = w.float().reshape(-1, 16)
        crest = blocks.abs().amax(-1) / blocks.square().mean(-1).sqrt().clamp_min(1e-30)
        k = kinds.setdefault(kind, {})
        for metric, weight in (('mse', None), ('hess', imp)):
            a6, a4, a0 = (block_sse(q, w, weight).double() for q in (s6, s4, e0))
            best2 = torch.minimum(a6, a4)
            best3 = torch.minimum(best2, a0)
            r = k.setdefault(metric, dict(blocks=0, s6=0., s4=0., e0=0., best2=0., best3=0.,
                                          e0_beats_s6=0, e0_beats_best2=0, gain_where_e0_beats_best2=0.,
                                          crest_e0win=0., crest_all=0.,
                                          **{f'tile_{t}': dict(tiles=0, e0_tiles=0, gain=0.) for t in TILES}))
            r['blocks'] += a6.numel()
            for key, v in (('s6', a6), ('s4', a4), ('e0', a0), ('best2', best2), ('best3', best3)):
                r[key] += float(v.sum())
            win = a0 < best2
            r['e0_beats_s6'] += int((a0 < a6).sum())
            r['e0_beats_best2'] += int(win.sum())
            r['gain_where_e0_beats_best2'] += float((best2 - a0)[win].sum())
            r['crest_e0win'] += float(crest[win].double().sum())
            r['crest_all'] += float(crest.double().sum())
            for t, (rows, cols) in TILES.items():
                # One E0M3 decision per tile, against the per-block best of {S6, S4}.
                g = tile_sum(best2 - a0, w.shape, rows, cols)
                tt = r[f'tile_{t}']
                tt['tiles'] += g.numel()
                tt['e0_tiles'] += int((g > 0).sum())
                tt['gain'] += float(g.clamp_min(0).sum())
        del chosen, other, s6, s4, e0
    summary = {}
    for metric in ('mse', 'hess'):
        tot = {key: sum(k[metric][key] for k in kinds.values())
               for key in ('blocks', 's6', 's4', 'e0', 'best2', 'best3', 'e0_beats_s6', 'e0_beats_best2',
                           'gain_where_e0_beats_best2', 'crest_e0win', 'crest_all')}
        out = dict(
            blocks=tot['blocks'],
            e0_beats_s6_fraction=tot['e0_beats_s6'] / tot['blocks'],
            e0_beats_best_e2m1_fraction=tot['e0_beats_best2'] / tot['blocks'],
            error_rel_to_s6=dict(s4=tot['s4'] / tot['s6'], e0=tot['e0'] / tot['s6'],
                                 best_e2m1_per_block=tot['best2'] / tot['s6'],
                                 best_of_three_per_block=tot['best3'] / tot['s6']),
            e0_extra_reduction_over_best_e2m1_per_block=1 - tot['best3'] / tot['best2'],
            crest_factor_mean=dict(all=tot['crest_all'] / tot['blocks'],
                                   e0_wins=tot['crest_e0win'] / max(tot['e0_beats_best2'], 1)))
        for t in TILES:
            tiles = sum(k[metric][f'tile_{t}']['tiles'] for k in kinds.values())
            e0t = sum(k[metric][f'tile_{t}']['e0_tiles'] for k in kinds.values())
            gain = sum(k[metric][f'tile_{t}']['gain'] for k in kinds.values())
            out[f'tile_{t}'] = dict(e0_tile_fraction=e0t / tiles,
                                    extra_reduction_over_best_e2m1=gain / tot['best2'],
                                    share_of_per_block_e0_gain_kept=gain / max(tot['best2'] - tot['best3'], 1e-30))
        out['per_kind_extra_reduction_per_block'] = {
            kind: 1 - k[metric]['best3'] / k[metric]['best2'] for kind, k in kinds.items()}
        out['per_kind_e0_beats_best_e2m1_fraction'] = {
            kind: k[metric]['e0_beats_best2'] / k[metric]['blocks'] for kind, k in kinds.items()}
        summary[metric] = out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model=args.model, job_id=os.environ['SLURM_JOB_ID'],
                                        c4_windows=args.windows, summary=summary), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
