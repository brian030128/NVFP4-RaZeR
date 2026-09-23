"""Cross-domain generalization of frozen MixFP4 maps (Llama-3.1-8B, 256x64, W4A4).

Every map is frozen before these documents are read. Domains: 64 fresh
OpenWebMath/CodeParrot windows (excluding calibration, every earlier fresh
manifest and this session's gate), and 64 windows each from PG-19 test,
arXiv articles and GovReport. One 512-token window per document (seed 20260923).
Per document: CE and KL(BF16 teacher || quantized). Paired against FourOverSix.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_fine_row_research import fresh_data, sequence_metadata
from run_math_code_calibration import load_model
from run_task_reorder_eval import mix_coarse

ROOT = Path('/work/u4320956/task_reorder')
LLAMA = ROOT / 'transfer_20260920/llama8b'
POTENTIAL = Path('/work/u4320956/mixfp4_potential')
EXTERNAL = {'books': ('emozilla/pg19-test', None, 'test', 'text'),
            'arxiv': ('ccdv/arxiv-summarization', None, 'test', 'article'),
            'govreport': ('ccdv/govreport-summarization', None, 'test', 'report')}


def external(tok, repo, config, split, field, count=64, length=512):
    stream = load_dataset(repo, config, split=split, streaming=True)
    generator = torch.Generator().manual_seed(20260923)
    rows = []
    for row in stream:
        text = row[field]
        ids = tok(text, return_tensors='pt').input_ids
        if ids.shape[1] < length:
            continue
        offset = int(torch.randint(ids.shape[1] - length + 1, (1,), generator=generator))
        rows.append(dict(ids=ids[:, offset:offset + length].clone(), offset=offset,
                         document_sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(rows) == count:
            break
    assert len(rows) == count, repo
    return rows


def paired(a, b):
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    return dict(delta=float(x.mean()), two_se=float(2 * x.std(unbiased=True) / math.sqrt(len(x))))


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads((LLAMA / 'calibration/report.json').read_text())
    maps = {'multiround_both': POTENTIAL / 'multiround_256x64_both/map.pt',
            'multiround_kl': POTENTIAL / 'multiround_256x64_kl/map.pt',
            'multiround_ce': POTENTIAL / 'multiround_256x64_ce/map.pt'}
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], map_sha256={k: digest_file(v) for k, v in maps.items()},
                  maps_frozen_before_data=True, arms={}, contrasts={}, data={})
    save(args.out, report)
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    # Fresh math/code excluding everything seen before, including this session's gate.
    published = json.loads(Path('results/kse_paper/job_336566/llama8b/report.json').read_text())
    excluded = [r['document_sha256'] for r in published['data']['c4_paper']['documents']]
    manifests = sorted({str(p.resolve()) for p in list(ROOT.rglob('fresh_manifest.json')) + list(POTENTIAL.rglob('fresh_manifest.json'))}
                       | set(json.loads(Path('results/task_reorder/transfer_20260920/exclude_manifests.json').read_text())))
    old_tokens = set(sequence_metadata(prior)[0])
    for path in manifests:
        records = json.loads(Path(path).read_text())['records']
        excluded.extend(r['document_sha256'] for r in records)
        old_tokens.update(r['token_sha256'] for r in records)
    qprior = json.loads((ROOT / 'pilot_20260919/qwen27b/calibration/report.json').read_text())
    excluded.extend(d['document_sha256'] for v in qprior['fit'].values() for d in v['documents'])
    fresh = fresh_data(tok, prior, excluded)
    assert not {r['token_sha256'] for r in fresh} & old_tokens
    domains = {'math': [r for r in fresh if r['source'] == 'math'], 'code': [r for r in fresh if r['source'] == 'code']}
    for name, spec in EXTERNAL.items():
        domains[name] = external(tok, *spec)
    report['data'] = {d: dict(documents=len(rs), document_sha256=[r['document_sha256'] for r in rs]) for d, rs in domains.items()}
    report['excluded_manifests'] = len(manifests)
    save(args.out, report)

    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    device = model.get_input_embeddings().weight.device
    teacher = {d: [model(input_ids=r['ids'].to(device), use_cache=False).logits[:, :-1].float().log_softmax(-1).bfloat16().cpu()
                   for r in rs] for d, rs in domains.items()}
    base, alt = {}, {}
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    frozen = torch.load(LLAMA / 'calibration/compact_masks.pt', map_location='cpu', weights_only=True)['raw256']
    scores = POTENTIAL / 'llama8b_calibration/scores'

    def k_mask(i, n, k):
        s = torch.load(scores / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert s['name'] == n
        b = [s[x + '_seq256'].mean(0) + k * s[x + '_seq256'].std(0, unbiased=True) / math.sqrt(128) for x in ('ce', 'kl')]
        return torch.maximum(*b) < 0

    arms = {'four_over_six': {n: None for n in modules}, 'oneshot_k3': dict(frozen),
            'oneshot_k1.5': {n: k_mask(i, n, 1.5) for i, n in enumerate(modules)}}
    for label, path in maps.items():
        arms[label] = torch.load(path, map_location='cpu', weights_only=True)
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for m in modules.values()]
    for label, masks in arms.items():
        tiles = 0
        for n, m in modules.items():
            mask = masks[n]
            if mask is None or not bool(mask.any()):
                m.weight.copy_(base[n]); continue
            tiles += int(mask.sum())
            m.weight.copy_(mix_coarse(base[n], alt[n], mask))
        result = dict(tiles=tiles)
        for d, rs in domains.items():
            ce, kl = [], []
            for r, t in zip(rs, teacher[d]):
                ids = r['ids'].to(device)
                lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1).reshape(-1, t.shape[-1])
                ce.append(float(F.nll_loss(lp, ids[:, 1:].reshape(-1))))
                kl.append(float(F.kl_div(lp, t.to(device).float().reshape(-1, t.shape[-1]), reduction='batchmean', log_target=True)))
            result[d] = dict(ce=ce, kl=kl)
        report['arms'][label] = result
        print(f'ARM {label} tiles={tiles} ' + ' '.join(f'{d}:CE={sum(result[d]["ce"])/64:.4f},KL={sum(result[d]["kl"])/64:.4f}'
                                                      for d in domains), flush=True)
        save(args.out, report)
    for h in handles:
        h.remove()
    base_arm = report['arms']['four_over_six']
    for label, result in report['arms'].items():
        if label == 'four_over_six':
            continue
        report['contrasts'][label] = {d: dict(ce=paired(result[d]['ce'], base_arm[d]['ce']),
                                               kl=paired(result[d]['kl'], base_arm[d]['kl'])) for d in domains}
        c = report['contrasts'][label]
        print(f'CONTRAST {label}: ' + ' '.join(f'{d} dCE={c[d]["ce"]["delta"]:+.4f}±{c[d]["ce"]["two_se"]:.4f} '
                                               f'dKL={c[d]["kl"]["delta"]:+.4f}±{c[d]["kl"]["two_se"]:.4f}' for d in domains), flush=True)
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()
