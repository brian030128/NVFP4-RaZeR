"""What the elected tiles look like as weights, and whether MSE would have found them.

The report argues the selector has to be a task loss because weight error is the
wrong objective. The election is now localized (analyze_kse_selection.py), so the
question is answerable directly: for every 8x64 tile, compare the squared weight
error of the canonical FourOverSix E2M1 candidate against the E0M3-alpha1
candidate, and ask how the tiles the k-SE rule elects sit in that distribution.

Weights are pristine and hash-checked against the calibration record, so this reads
exactly the matrices the scores were taken on.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_conditional_format import sha
from run_kse_paper import MODELS

TILE = (8, 64)


def tile_sum(x, rows=TILE[0], cols=TILE[1]):
    r, c = x.shape
    return x.reshape(r // rows, rows, c // cols, cols).sum((1, 3))


def load(model, source):
    if model == 'qwen27b':
        from transformers import Qwen3_5ForConditionalGeneration
        m = Qwen3_5ForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
        mods = {n: mod for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)
                and 'language_model' in n and 'head' not in n}
    else:
        m = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
        mods = {n: mod for n, mod in m.named_modules()
                if isinstance(mod, torch.nn.Linear) and mod is not m.get_output_embeddings()}
    m.eval().requires_grad_(False)
    return m, mods


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--selection', required=True, help='results/kse_selection/job_<id>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    sel = json.loads((Path(args.selection) / f'{args.model}.json').read_text())
    assert sel['status'] == 'complete' and sel['election_reproduced']
    old = Path(args.stage_root) / f'results/math_code_adaptive/calibration_{MODELS[args.model]}_{args.model}'
    prior = json.loads((old / 'report.json').read_text())
    assert transformers.__version__ == prior['transformers_version']
    _, mods = load(args.model, prior['source'])
    assert list(mods) == [m['name'] for m in sel['modules']]

    r = dict(status='running', model=args.model, job_id=os.environ['SLURM_JOB_ID'],
             transformers_version=transformers.__version__, selection=args.selection,
             k=sel['k'], selected=sel['counts'][str(sel['k'])], modules=[])
    top = []                      # (relative gain, elected?) for a model-wide ranking
    agg = dict(tiles=0, mse_prefers_e0m3=0, elected=0, elected_mse_prefers=0)
    for entry in sel['modules']:
        name = entry['name']
        w = mods[name].weight
        assert sha(w) == prior['matrices'][name]['source_sha256'], name
        base = quant_nvfp4_4over6(w, 4, 16)
        alt = quant_mix_4_6(w, 4, 16, type_block=TILE, clip='a1', elect='always')
        wf = w.float()
        eb = tile_sum((base.float() - wf).square())
        ea = tile_sum((alt.float() - wf).square())
        rel = ((eb - ea) / eb.clamp_min(torch.finfo(torch.float32).tiny))
        amax = wf.abs().reshape(w.shape[0] // 8, 8, w.shape[1] // 64, 64).amax((1, 3))
        rms = tile_sum(wf.square()).div(TILE[0] * TILE[1]).sqrt()
        peak = (amax / rms.clamp_min(torch.finfo(torch.float32).tiny))

        mask = torch.zeros(eb.shape, dtype=torch.bool, device=eb.device)
        if entry['selected']:
            mask[torch.tensor(entry['tiles_rows'], device=eb.device),
                 torch.tensor(entry['tiles_cols'], device=eb.device)] = True
        assert int(mask.sum()) == entry['selected'], name

        agg['tiles'] += eb.numel()
        agg['mse_prefers_e0m3'] += int((rel > 0).sum())
        agg['elected'] += entry['selected']
        agg['elected_mse_prefers'] += int((rel > 0)[mask].sum())
        flat_rel, flat_mask = rel.reshape(-1), mask.reshape(-1)
        n = min(r['selected'], flat_rel.numel())
        idx = torch.topk(flat_rel, n).indices
        top += list(zip(flat_rel[idx].tolist(), flat_mask[idx].tolist()))
        top = sorted(top, key=lambda t: -t[0])[:r['selected']]

        m = dict(name=name, kind=entry['kind'], layer=entry['layer'], selected=entry['selected'],
                 tiles=eb.numel(), mse_prefers_e0m3=int((rel > 0).sum()),
                 rel_gain_median=float(rel.median()), peak_median=float(peak.median()))
        if entry['selected']:
            m.update(elected_mse_prefers=int((rel > 0)[mask].sum()),
                     elected_rel_gain_median=float(rel[mask].median()),
                     elected_peak_median=float(peak[mask].median()),
                     elected_rel_gain_percentile=float(
                         (rel.reshape(-1)[None, :] < rel[mask][:, None]).float().mean()))
        r['modules'].append(m)
        del base, alt, wf, eb, ea, rel, amax, rms, peak, mask
        if len(r['modules']) % 25 == 0:
            print(f'{args.model} {len(r["modules"])}/{len(sel["modules"])} {name}', flush=True)

    r['aggregate'] = agg
    r['base_rate'] = agg['mse_prefers_e0m3'] / agg['tiles']
    r['elected_rate'] = agg['elected_mse_prefers'] / agg['elected']
    # How many of the model-wide top-N tiles by relative MSE gain the rule also elects.
    r['top_n_overlap'] = dict(n=len(top), elected=sum(1 for _, e in top if e))
    r['status'] = 'complete'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.model}.json').write_text(json.dumps(r, indent=1) + '\n')
    print(f'{args.model}: MSE prefers E0M3 for {100 * r["base_rate"]:.2f}% of all tiles, '
          f'{100 * r["elected_rate"]:.2f}% of elected; top-{len(top)} overlap '
          f'{r["top_n_overlap"]["elected"]}', flush=True)


if __name__ == '__main__':
    main()
