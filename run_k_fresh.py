"""Fresh CE-primary gate for the dev-selected 256x64 election threshold (Llama-3.1-8B).

The candidate (k=1.5, chosen by run_k_dev.py on 192 development documents) and the
criterion are written to plan.json before any fresh document is sampled. The 64
fresh OpenWebMath/CodeParrot windows exclude calibration, every earlier fresh
manifest, Qwen calibration and published C4 documents.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_fine_row_research import fresh_data, sequence_metadata
from run_math_code_calibration import load_model
from run_task_reorder_eval import mix_coarse, validate_compact_masks

ROOT = Path('/work/u4320956/task_reorder')
LLAMA = ROOT / 'transfer_20260920/llama8b'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def paired(a, b):
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    m = float(x.mean()); se = float(x.std(unbiased=True) / math.sqrt(len(x)))
    return dict(delta=m, two_se=2 * se, upper=m + 2 * se)


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-k', type=float, default=1.5)
    ap.add_argument('--secondary-k', type=float, default=2.0)
    ap.add_argument('--scores', type=Path, default=Path('/work/u4320956/mixfp4_potential/llama8b_calibration/scores'))
    ap.add_argument('--dev-report', type=Path, default=Path('/work/u4320956/mixfp4_potential/k_dev_both/report.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads((LLAMA / 'calibration/report.json').read_text())
    dev = json.loads(args.dev_report.read_text())
    assert dev['status'] == 'complete' and dev['selected_k_by_mean'] == f'k={args.candidate_k}'
    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(candidate=f'CE+KL 256x64 k={args.candidate_k}', secondary=f'k={args.secondary_k} (diagnostic)',
                selected_by=str(args.dev_report), dev_report_sha256=digest_file(args.dev_report),
                criterion='candidate pooled CE mean+2SE<0 vs raw256(k=3) and vs FourOverSix; '
                          'math and code CE means <=0 vs raw256; KL diagnostic only',
                source_sha256={p: digest_file(p) for p in ('run_k_fresh.py', 'quantize/quantizer.py',
                                                           'run_fine_row_research.py')})
    write(args.out / 'plan.json', plan)
    published = json.loads(Path('results/kse_paper/job_336566/llama8b/report.json').read_text())
    excluded = [r['document_sha256'] for r in published['data']['c4_paper']['documents']]
    manifests = sorted({str(p.resolve()) for p in ROOT.rglob('fresh_manifest.json')}
                       | set(json.loads(Path('results/task_reorder/transfer_20260920/exclude_manifests.json').read_text())))
    old_tokens = set(sequence_metadata(prior)[0])
    for path in manifests:
        records = json.loads(Path(path).read_text())['records']
        excluded.extend(r['document_sha256'] for r in records)
        old_tokens.update(r['token_sha256'] for r in records)
    qprior = json.loads((ROOT / 'pilot_20260919/qwen27b/calibration/report.json').read_text())
    excluded.extend(d['document_sha256'] for v in qprior['fit'].values() for d in v['documents'])
    old_tokens.update(sequence_metadata(qprior)[0])
    plan.update(exclude_manifests={p: digest_file(p) for p in manifests}, excluded_documents=len(set(excluded)))
    write(args.out / 'plan.json', plan)
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fresh = fresh_data(tok, prior, excluded)
    assert len(fresh) == 64 and not {r['document_sha256'] for r in fresh} & set(excluded)
    assert not {r['token_sha256'] for r in fresh} & old_tokens
    torch.save(fresh, args.out / 'fresh.pt')
    write(args.out / 'fresh_manifest.json', dict(records=[{k: v for k, v in r.items() if k != 'ids'} for r in fresh],
                                                 windows=64, window_tokens=512, seed=20260919))
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], plan_sha256=digest_file(args.out / 'plan.json'),
                  fresh_sha256=digest_file(args.out / 'fresh.pt'), sources=[r['source'] for r in fresh], arms={})
    save(args.out, report)

    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    teacher = []
    for r in fresh:
        logits = model(input_ids=r['ids'].cuda(), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    frozen = validate_compact_masks(torch.load(LLAMA / 'calibration/compact_masks.pt', map_location='cpu',
                                               weights_only=True), prior)
    base, alt, stats = {}, {}, {}
    for i, (n, m) in enumerate(modules.items()):
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        s = torch.load(args.scores / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert s['name'] == n
        stats[n] = [(s[k + '_seq256'].mean(0), s[k + '_seq256'].std(0, unbiased=True) / math.sqrt(128)) for k in ('ce', 'kl')]
        assert torch.equal(torch.maximum(*(mu + 3 * se for mu, se in stats[n])) < 0, frozen['raw256'][n]), n
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for m in modules.values()]
    arms = [('four_over_six', None), ('raw256', 3.0), ('candidate', args.candidate_k), ('secondary', args.secondary_k)]
    for label, k in arms:
        tiles = 0
        for n, m in modules.items():
            if k is None:
                m.weight.copy_(base[n]); continue
            mask = torch.maximum(*(mu + k * se for mu, se in stats[n])) < 0
            tiles += int(mask.sum())
            m.weight.copy_(mix_coarse(base[n], alt[n], mask) if mask.any() else base[n])
        ce, kl = [], []
        for r, t in zip(fresh, teacher):
            ids = r['ids'].cuda()
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            ce.append(float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), ids[:, 1:].reshape(-1))))
            kl.append(float(F.kl_div(lp.reshape(-1, lp.shape[-1]), t.cuda().float().reshape(-1, lp.shape[-1]),
                                     reduction='batchmean', log_target=True)))
        report['arms'][label] = dict(k=k, tiles=tiles, ce=ce, kl=kl)
        print(f'FRESH {label} k={k} tiles={tiles} meanCE={sum(ce)/len(ce):.6f} meanKL={sum(kl)/len(kl):.6f}', flush=True)
        save(args.out, report)
    for h in handles:
        h.remove()
    a = report['arms']
    contrasts = {}
    for label in ('candidate', 'secondary'):
        for ref in ('raw256', 'four_over_six'):
            contrasts[f'{label} - {ref}'] = dict(
                ce=paired(a[label]['ce'], a[ref]['ce']), kl=paired(a[label]['kl'], a[ref]['kl']),
                **{src + '_ce': paired([v for v, s in zip(a[label]['ce'], report['sources']) if s == src],
                                       [v for v, s in zip(a[ref]['ce'], report['sources']) if s == src])
                   for src in ('math', 'code')})
    c = contrasts
    report['contrasts'] = contrasts
    report['passed'] = (c['candidate - raw256']['ce']['upper'] < 0 and c['candidate - four_over_six']['ce']['upper'] < 0
                        and c['candidate - raw256']['math_ce']['delta'] <= 0 and c['candidate - raw256']['code_ce']['delta'] <= 0)
    report['status'] = 'complete'
    save(args.out, report)
    for key, v in contrasts.items():
        print(f'CONTRAST {key}: CE {v["ce"]["delta"]:+.6f}±{v["ce"]["two_se"]:.6f} KL {v["kl"]["delta"]:+.6f}±{v["kl"]["two_se"]:.6f} '
              f'math {v["math_ce"]["delta"]:+.6f} code {v["code_ce"]["delta"]:+.6f}', flush=True)
    print('PASSED' if report['passed'] else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
