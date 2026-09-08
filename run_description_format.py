"""Replay shared C4 scores, apply one fixed description-cost rule, evaluate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from run_conditional_format import REVISIONS, sha, save
from run_conditional_model import paired
from run_directed_format import fit_data
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.interacting_format import apply_mask
from quantize.description_format import elect


def data(tok):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=REVISIONS['wiki'], split='test')
    ids = tok('\n\n'.join(ds['text']), return_tensors='pt').input_ids
    result = {'wiki': [ids[:, i*512:(i+1)*512] for i in range(416, 448)]}
    assert all(b.numel() == 512 for b in result['wiki'])
    for domain, repo, config in [('math', 'openai/gsm8k', 'main'), ('code', 'google-research-datasets/mbpp', 'full')]:
        ds = load_dataset(repo, config, revision=REVISIONS[domain], split='test')
        texts = [('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain == 'math'
                 else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(432, 464))]
        result[domain] = [tok(t, return_tensors='pt').input_ids[:, :512] for t in texts]
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=['llama1b', 'opt350m', 'qwen06b'], required=True)
    ap.add_argument('--out', required=True); args = ap.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent; out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    old = root/f'results/relinearized_format/model_332316_{args.model}'
    prior = json.loads((old/'report.json').read_text()); assert prior['status'] == 'complete'
    files = ['run_description_format.py', 'quantize/description_format.py', 'quantize/relinearized_format.py',
             'quantize/quantizer.py', 'tests/test_description_format.py', 'results/description_format/PROTOCOL.md']
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), score_origin=str(old),
             source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
             evaluation={}, fit_audit={}, matrices=prior['matrices'], dataset_revisions=REVISIONS)
    assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']
    record = torch.load(old/'initial_scores.pt', map_location='cpu', weights_only=True)
    selected, r['election'] = elect(record['ce'], record['kl'], 511)
    bundle = torch.load(old/'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == r['source'] and bundle['revision'] == r['revision']
    maps = {p: bundle['maps'][p] for p in ('weight_mse', 'stale256', 'fixed_budget')}
    maps['description'] = {}
    for n, (lo, hi) in record['slices'].items():
        o, k = prior['matrices'][n]['shape']; maps['description'][n] = selected[lo:hi].reshape(o//8, k//64).clone()
    names = record['names']; del record, bundle
    kwargs = dict(revision=r['revision'], torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
    model = AutoModelForCausalLM.from_pretrained(r['source'], **kwargs).eval().requires_grad_(False)
    teacher = AutoModelForCausalLM.from_pretrained(r['source'], **kwargs).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == names
    base = {}; alt = {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256']
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8,64), clip='a1', elect='always')
    fit, r['fit'] = fit_data(tok); assert r['fit'] == prior['fit']
    torch.save(dict(source=r['source'], revision=r['revision'], maps=maps, baseline='FourOverSix', alternative='E0M3 alpha1', type_block=(8,64)), out/'maps.pt')
    r['maps_frozen'] = True; r['selected_tiles'] = {p: sum(int(m.sum()) for m in ms.values()) for p, ms in maps.items()}
    save(out, r); print('FROZEN '+json.dumps(r['election']), flush=True)
    def act(module, inputs): return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])
    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    batches = data(tok); r['eval_token_sha256'] = {d: [sha(b) for b in bs] for d, bs in batches.items()}
    policies = ('four_over_six', *maps)
    for p in policies:
        with torch.no_grad():
            for n, m in modules.items(): m.weight.copy_(base[n] if p == 'four_over_six' else apply_mask(base[n], alt[n], maps[p][n].cuda()))
            if p in ('four_over_six', 'description'):
                audit = []
                for batch in fit:
                    ids = batch.cuda(); teacher_log = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1).reshape(-1, teacher.config.vocab_size)
                    logits = model(input_ids=ids, use_cache=False).logits
                    logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
                    audit.append(dict(ce=float(F.nll_loss(logprob, ids[:, 1:].reshape(-1))), kl=float(F.kl_div(logprob, teacher_log, reduction='batchmean', log_target=True))))
                r['fit_audit'][p] = dict(examples=audit, ce=sum(a['ce'] for a in audit)/64, kl=sum(a['kl'] for a in audit)/64)
            result = {}
            for d, bs in batches.items():
                values = []
                for batch in bs:
                    ids = batch.cuda(); logits = model(input_ids=ids, use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))))
                result[d] = dict(nll=values, ppl=math.exp(sum(values)/len(values)))
        r['evaluation'][p] = result; save(out, r)
        print(f'EVAL {p} '+json.dumps({d: v['ppl'] for d, v in result.items()}), flush=True)
    for h in handles: h.remove()
    r['contrasts'] = {d: {p: paired(r['evaluation']['description'][d]['nll'], r['evaluation'][p][d]['nll']) for p in policies[:-1]} for d in batches}
    r['stale_contrasts'] = {d: paired(r['evaluation']['stale256'][d]['nll'], r['evaluation']['four_over_six'][d]['nll']) for d in batches}
    r['status'] = 'complete'; save(out, r)
    lines = [f'# Description-cost tile election: {args.model}', '', '```json', json.dumps(r['election'], indent=2), '```', '',
             '| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Description | ΔNLL ±2SE vs baseline |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in batches:
        c = r['contrasts'][d]['four_over_six']
        lines.append('| '+d+' | '+' | '.join(f'{r["evaluation"][p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', 'Fitting audit:', '', '```json', json.dumps({p: {k: v for k, v in a.items() if k != 'examples'} for p, a in r['fit_audit'].items()}, indent=2), '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
