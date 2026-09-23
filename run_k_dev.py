"""Choose the 256x64 election threshold k on held-out math/code development documents.

The 192 development documents are the three recorded confirmation sets of the
transfer study; none of them is in the 128-sequence calibration set, and neither
WikiText nor C4 is read. Every k is scored by full-model W4A4 CE per document,
paired against FourOverSix, so k can be frozen before looking at test PPL.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_task_reorder_eval import mix_coarse, validate_compact_masks

ROOT = Path('/work/u4320956/task_reorder/transfer_20260920/llama8b')
DEVELOPMENT = ('confirmation', 'gate_up_confirm', 'tile_refine_confirm')


def load_development():
    records, provenance = [], []
    for name in DEVELOPMENT:
        report = json.loads((ROOT / name / 'report.json').read_text())
        assert report.get('status') == 'complete'
        path = ROOT / name / 'fresh.pt'
        assert digest_file(path) == report['fresh_sha256'], name
        rows = torch.load(path, map_location='cpu', weights_only=True)
        records.extend(rows)
        provenance.append(dict(name=name, documents=len(rows), sha256=report['fresh_sha256']))
    return records, provenance


def paired(a, b):
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    return dict(delta=float(x.mean()), two_se=float(2 * x.std(unbiased=True) / math.sqrt(len(x))))


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ks', default='1,1.5,2,2.5,3,3.5,4')
    ap.add_argument('--objective', choices=('both', 'ce', 'kl'), default='both')
    ap.add_argument('--scores', type=Path, default=Path('/work/u4320956/mixfp4_potential/llama8b_calibration/scores'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    calib = ROOT / 'calibration'
    prior = json.loads((calib / 'report.json').read_text())
    assert transformers.__version__ == prior['transformers_version']
    args.out.mkdir(parents=True, exist_ok=False)
    records, provenance = load_development()
    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], objective=args.objective,
                  development=provenance, documents=len(records), uses_wiki=False, uses_c4=False,
                  calibration_overlap=sum(r.get('document_sha256') in excluded for r in records),
                  source_sha256={p: digest_file(p) for p in ('run_k_dev.py', 'quantize/quantizer.py')},
                  arms={}, paired={})
    assert report['calibration_overlap'] == 0
    save(args.out, report)
    model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
                                                 torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and m is not model.get_output_embeddings()}
    assert list(modules) == list(prior['matrices'])
    frozen = validate_compact_masks(torch.load(calib / 'compact_masks.pt', map_location='cpu', weights_only=True), prior)
    base, alt, stats = {}, {}, {}
    for i, (n, m) in enumerate(modules.items()):
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        s = torch.load(args.scores / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert s['name'] == n
        stats[n] = {key: (s[key + '_seq256'].mean(0), s[key + '_seq256'].std(0, unbiased=True) / math.sqrt(128))
                    for key in ('ce', 'kl')}
        check = torch.maximum(*(mu + 3 * se for mu, se in stats[n].values())) < 0
        assert torch.equal(check, frozen['raw256'][n]), n
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for m in modules.values()]

    def mask_for(n, k):
        bounds = {key: mu + k * se for key, (mu, se) in stats[n].items()}
        return (torch.maximum(bounds['ce'], bounds['kl']) if args.objective == 'both' else bounds[args.objective]) < 0

    def losses():
        values = []
        for r in records:
            ids = r['ids'].cuda()
            logits = model(input_ids=ids, use_cache=False).logits
            values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))))
        return values

    arms = [('four_over_six', None)] + [(f'k={k}', float(k)) for k in args.ks.split(',')]
    for label, k in arms:
        tiles = 0
        for n, m in modules.items():
            if k is None:
                m.weight.copy_(base[n])
                continue
            mask = mask_for(n, k); tiles += int(mask.sum())
            m.weight.copy_(mix_coarse(base[n], alt[n], mask) if mask.any() else base[n])
        values = losses()
        report['arms'][label] = dict(k=k, tiles=tiles, nll=values, mean_nll=sum(values) / len(values),
                                     sources=[r['source'] for r in records])
        if k is not None:
            ref = report['arms']['four_over_six']['nll']
            report['paired'][label] = dict(tiles=tiles, all=paired(values, ref), **{
                src: paired([v for v, r in zip(values, records) if r['source'] == src],
                            [v for v, r in zip(ref, records) if r['source'] == src]) for src in ('math', 'code')})
            p = report['paired'][label]
            print(f'DEV {label} tiles={tiles} dCE={p["all"]["delta"]:+.6f}±{p["all"]["two_se"]:.6f} '
                  f'math={p["math"]["delta"]:+.6f} code={p["code"]["delta"]:+.6f}', flush=True)
        save(args.out, report)
    for h in handles:
        h.remove()
    report['selected_k_by_mean'] = min(report['paired'], key=lambda x: report['paired'][x]['all']['delta'])
    report['status'] = 'complete'
    save(args.out, report)
    print('SELECTED', report['selected_k_by_mean'], flush=True)


if __name__ == '__main__':
    main()
