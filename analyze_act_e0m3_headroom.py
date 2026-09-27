"""Can a STATIC per-K-strip E0M3/E2M1 type map help the ACTIVATION operand?

Activations are quantized at run time; a static map fixed offline makes the MMA instruction
variant a pure function of the K block (no run-time decision). For every text linear layer's
input X (tokens x K, one C4-train window at a time, one tensor-wide global scale per window
as in evaluation), each 16-channel block of each token is quantized two ways:
  base  FourOverSix E2M1 (the current activation quantizer)
  E0    E0M3, block max -> 7
The error is weighted by the weight column energy ||W[:, j]||^2 (how much channel j's
perturbation reaches the layer output), and also reported unweighted. With per-block gain
g = base - E0, on held-out TEST windows:
  per_block          sum of positive g                         (1x16 ceiling, not realizable)
  tile_16x64_runtime one decision per 16 tokens x 64 channels, oracle on the test data itself
                     (needs a run-time error computation, so not free)
  static_strip       one decision per 64-channel K strip, FIT on separate windows (realizable, free)
  static_strip_oracle the same, decided on the test data (in-sample upper bound)
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_math_code_calibration import load_model
from run_train_map import CALIBRATIONS, c4_train_windows


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='llama1b_ins', choices=tuple(CALIBRATIONS))
    ap.add_argument('--windows', type=int, default=64, help='Half fit, half test')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    prior = json.loads((CALIBRATIONS[args.model] / 'report.json').read_text())
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    windows, _ = c4_train_windows(tok, args.windows)
    device = model.get_input_embeddings().weight.device
    colw = {n: m.weight.float().square().sum(0) for n, m in modules.items()}
    acc = {}
    phase = ['fit']

    def hook(n):
        def f(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            t, k = x.shape
            base = quant_nvfp4_4over6(x, 4, 16).float()
            e0 = quant_mix_4_6(x, 4, 16, type_block=(1, 16), clip='a1', elect='always').float()
            xf = x.float()
            a = acc.setdefault(n, {})
            for metric, wgt in (('weighted', colw[n]), ('mse', None)):
                db, d0 = (base - xf).square(), (e0 - xf).square()
                if wgt is not None:
                    db, d0 = db * wgt, d0 * wgt
                eb = db.reshape(t, k // 16, 16).sum(-1).double()   # (tokens, blocks)
                g = eb - d0.reshape(t, k // 16, 16).sum(-1).double()
                strip = g.reshape(t, k // 64, 4).sum(-1)          # (tokens, strips)
                m = a.setdefault(metric, dict(fit_strip=torch.zeros(k // 64, dtype=torch.float64, device=x.device),
                                              test_strip=torch.zeros(k // 64, dtype=torch.float64, device=x.device),
                                              base=0., per_block=0., tile_16x64=0.))
                if phase[0] == 'fit':
                    m['fit_strip'] += strip.sum(0)
                    continue
                m['test_strip'] += strip.sum(0)
                m['base'] += float(eb.sum())
                m['per_block'] += float(g.clamp_min(0).sum())
                tp = (-t) % 16
                tiles = torch.nn.functional.pad(strip, (0, 0, 0, tp)).reshape(-1, 16, k // 64).sum(1)
                m['tile_16x64'] += float(tiles.clamp_min(0).sum())
        return f
    handles = [m.register_forward_pre_hook(hook(n)) for n, m in modules.items()]
    half = args.windows // 2
    for i, w in enumerate(windows):
        phase[0] = 'fit' if i < half else 'test'
        model(input_ids=w.to(device), use_cache=False)
    for h in handles:
        h.remove()

    summary = {}
    for metric in ('weighted', 'mse'):
        tot = dict(base=0., per_block=0., tile_16x64=0., static_strip=0., static_strip_oracle=0.,
                   strips=0, e0_strips=0)
        per_kind = {}
        for n, a in acc.items():
            m = a[metric]
            chosen = m['fit_strip'] > 0
            static = float(m['test_strip'][chosen].sum())
            oracle = float(m['test_strip'].clamp_min(0).sum())
            for key, v in (('base', m['base']), ('per_block', m['per_block']), ('tile_16x64', m['tile_16x64']),
                           ('static_strip', static), ('static_strip_oracle', oracle)):
                tot[key] += v
            tot['strips'] += chosen.numel()
            tot['e0_strips'] += int(chosen.sum())
            kind = n.split('.')[-1]
            pk = per_kind.setdefault(kind, dict(base=0., per_block=0., static_strip=0.))
            pk['base'] += m['base']
            pk['per_block'] += m['per_block']
            pk['static_strip'] += static
        summary[metric] = dict(
            e0m3_strip_fraction=tot['e0_strips'] / tot['strips'],
            extra_reduction_vs_four_over_six={k: tot[k] / tot['base'] for k in
                                              ('per_block', 'tile_16x64', 'static_strip', 'static_strip_oracle')},
            share_of_per_block_gain_kept={k: tot[k] / tot['per_block'] for k in
                                          ('tile_16x64', 'static_strip', 'static_strip_oracle')},
            per_kind={kind: dict(per_block=v['per_block'] / v['base'], static_strip=v['static_strip'] / v['base'])
                      for kind, v in per_kind.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model=args.model, job_id=os.environ['SLURM_JOB_ID'],
                                        fit_windows=half, test_windows=args.windows - half,
                                        summary=summary), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
