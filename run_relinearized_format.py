"""Fixed-recipe current-model binary format updates; no test-driven election."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import HfApi
from run_conditional_format import CHECKPOINT, REVISIONS, sha, save
from run_conditional_model import paired
from run_directed_format import fit_data
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.interacting_format import apply_mask
from quantize.relinearized_format import next_bit, stale_map


def data(tok):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=REVISIONS['wiki'], split='test')
    ids = tok('\n\n'.join(ds['text']), return_tensors='pt').input_ids
    result = {'wiki': [ids[:, i*512:(i+1)*512] for i in range(352, 384)]}
    assert all(b.numel() == 512 for b in result['wiki'])
    for domain, repo, config in [('math', 'openai/gsm8k', 'main'), ('code', 'google-research-datasets/mbpp', 'full')]:
        ds = load_dataset(repo, config, revision=REVISIONS[domain], split='test')
        texts = [('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain == 'math'
                 else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(368, 400))]
        result[domain] = [tok(t, return_tensors='pt').input_ids[:, :512] for t in texts]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['llama1b', 'opt350m', 'qwen06b'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False); root = Path(__file__).resolve().parent
    source = {'llama1b': CHECKPOINT, 'opt350m': 'facebook/opt-350m', 'qwen06b': 'Qwen/Qwen3-0.6B'}[args.model]
    revision = Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
    files = ['run_relinearized_format.py', 'quantize/relinearized_format.py', 'run_directed_format.py',
             'quantize/quantizer.py', 'tests/test_relinearized_format.py', 'results/relinearized_format/PROTOCOL.md']
    r = dict(status='running', model=args.model, source=source, revision=revision,
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
             transformers_version=transformers.__version__,
             source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
             matrices={}, evaluation={}, fit_audit={}, dataset_revisions=REVISIONS, updates=[])
    save(out, r)
    kwargs = dict(revision=revision, torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs).eval().requires_grad_(False)
    teacher = AutoModelForCausalLM.from_pretrained(source, **kwargs).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n}
    names = list(modules); base = {}; alt = {}; mse_maps = {}; slices = {}; count = 0
    with torch.no_grad():
        for n, m in modules.items():
            w = m.weight.detach(); o, k = w.shape
            assert o % 8 == 0 and k % 64 == 0
            r['matrices'][n] = dict(shape=list(w.shape), source_sha256=sha(w))
            base[n] = quant_nvfp4_4over6(w, 4, 16)
            alt[n] = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            difference = (alt[n].float()-w.float()).square() - (base[n].float()-w.float()).square()
            mse_maps[n] = (difference.reshape(o//8, 8, k//64, 64).sum((1, 3)) < 0).cpu()
            tiles = o//8 * (k//64); slices[n] = (count, count+tiles); count += tiles
            m.weight.copy_(base[n])
    scoring = True
    def act(module, inputs):
        x = inputs[0]; q = quant_nvfp4_4over6(x.detach(), 4, 16)
        return (q+(x-x.detach()) if scoring else q, *inputs[1:])
    ah = [m.register_forward_pre_hook(act) for m in modules.values()]
    scores = {}; phase = 'ce'
    def make_hook(n):
        def forward(module, inputs, output):
            if not scoring: return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            def backward(dy):
                grad = dy.detach().reshape(-1, dy.shape[-1]).float().T @ x.float()
                o, k = grad.shape
                value = (grad * (alt[n].float()-base[n].float())).reshape(o//8, 8, k//64, 64).sum((1, 3))
                scores[phase][n].append(value.flatten().cpu())
            output.register_hook(backward)
        return forward
    handles = [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
    fit, r['fit'] = fit_data(tok)

    def score_batches(batches):
        nonlocal phase, scores
        scores = {p: {n: [] for n in names} for p in ('ce', 'kl')}; losses = []
        for batch in batches:
            ids = batch.cuda()
            with torch.no_grad():
                teacher_log = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1).reshape(-1, teacher.config.vocab_size)
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
            logits = model(inputs_embeds=embeds, use_cache=False).logits
            logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            ce = F.nll_loss(logprob, ids[:, 1:].reshape(-1))
            kl = F.kl_div(logprob, teacher_log, reduction='batchmean', log_target=True)
            phase = 'ce'; ce.backward(retain_graph=True)
            phase = 'kl'; kl.backward()
            losses.append(dict(ce=float(ce), kl=float(kl)))
        matrices = {p: torch.cat([torch.stack(scores[p][n]) for n in names], dim=1) for p in scores}
        scores = {}
        return matrices['ce'], matrices['kl'], losses

    start = time.perf_counter()
    initial_ce, initial_kl, r['initial_fit_losses'] = score_batches(fit)
    torch.save(dict(ce=initial_ce, kl=initial_kl, names=names, slices=slices), out/'initial_scores.pt')
    stale = stale_map(initial_ce, initial_kl)
    mean = initial_ce.mean(0); se = initial_ce.std(0, unbiased=True)/8
    eligible = mean+2*se < 0
    order = torch.argsort(torch.where(eligible, mean, torch.inf), stable=True)
    prefix = ((-mean[order]).clamp_min(0).cumsum(0) <= .1) & eligible[order]
    budget = torch.zeros(count, dtype=torch.bool); budget[order[prefix]] = True
    del initial_ce, initial_kl
    r['initial_score_seconds'] = time.perf_counter()-start
    selected = torch.zeros(count, dtype=torch.bool); rng = random.Random(20260924)
    touched = set(); start = time.perf_counter()
    for epoch in range(16):
        order = list(range(64)); rng.shuffle(order)
        for begin in range(0, 64, 4):
            indices = order[begin:begin+4]
            ce, kl, losses = score_batches([fit[i] for i in indices])
            index, upper = next_bit(ce, kl, selected)
            event = dict(step=len(r['updates']), examples=indices, bit=index, upper=upper, losses=losses)
            if index is not None:
                event['undo'] = bool(selected[index]); event['revisited'] = index in touched
                selected[index] ^= True; touched.add(index)
                for n, (lo, hi) in slices.items():
                    if lo <= index < hi:
                        m = modules[n]; o, k = m.weight.shape
                        row, col = divmod(index-lo, k//64)
                        chosen = alt[n] if selected[index] else base[n]
                        with torch.no_grad(): m.weight[row*8:(row+1)*8, col*64:(col+1)*64].copy_(chosen[row*8:(row+1)*8, col*64:(col+1)*64])
                        event.update(module=n, tile=[row, col]); break
            r['updates'].append(event)
        save(out, r)
        print(f'EPOCH {epoch+1}/16 selected={int(selected.sum())} revisits={sum(e.get("revisited", False) for e in r["updates"])}', flush=True)
    r['optimization_seconds'] = time.perf_counter()-start
    maps = {'weight_mse': mse_maps, 'stale256': {}, 'fixed_budget': {}, 'relinearized': {}}
    for n, (lo, hi) in slices.items():
        o, k = modules[n].weight.shape
        for p, flat in [('stale256', stale), ('fixed_budget', budget), ('relinearized', selected)]:
            maps[p][n] = flat[lo:hi].reshape(o//8, k//64).clone()
    torch.save(dict(source=source, revision=revision, maps=maps, baseline='FourOverSix', alternative='E0M3 alpha1', type_block=(8,64)), out/'maps.pt')
    r['maps_frozen'] = True; r['selected_tiles'] = {p: sum(int(m.sum()) for m in ms.values()) for p, ms in maps.items()}
    scoring = False
    for h in handles: h.remove()
    save(out, r)
    print('FROZEN '+json.dumps(r['selected_tiles']), flush=True)
    batches = data(tok); r['eval_token_sha256'] = {d: [sha(b) for b in bs] for d, bs in batches.items()}
    policies = ('four_over_six', 'weight_mse', 'stale256', 'fixed_budget', 'relinearized')
    for p in policies:
        with torch.no_grad():
            for n, m in modules.items(): m.weight.copy_(base[n] if p == 'four_over_six' else apply_mask(base[n], alt[n], maps[p][n].cuda()))
            if p in ('four_over_six', 'relinearized', 'stale256'):
                audit = []
                for batch in fit:
                    ids = batch.cuda(); teacher_log = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1).reshape(-1, teacher.config.vocab_size)
                    logits = model(input_ids=ids, use_cache=False).logits
                    logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
                    audit.append(dict(ce=float(F.nll_loss(logprob, ids[:, 1:].reshape(-1))), kl=float(F.kl_div(logprob, teacher_log, reduction='batchmean', log_target=True))))
                r['fit_audit'][p] = dict(examples=audit, ce=sum(a['ce'] for a in audit)/len(audit), kl=sum(a['kl'] for a in audit)/len(audit))
            result = {}
            for d, bs in batches.items():
                values = []
                for batch in bs:
                    ids = batch.cuda(); logits = model(input_ids=ids, use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))))
                result[d] = dict(nll=values, ppl=math.exp(sum(values)/len(values)))
        r['evaluation'][p] = result; save(out, r)
        print(f'EVAL {p} '+json.dumps({d: v['ppl'] for d, v in result.items()}), flush=True)
    for h in ah: h.remove()
    r['contrasts'] = {d: {p: paired(r['evaluation']['relinearized'][d]['nll'], r['evaluation'][p][d]['nll']) for p in policies[:-1]} for d in batches}
    r['status'] = 'complete'; save(out, r)
    lines = [f'# Relinearized format updates: {args.model}', '',
             '| Domain | FourOverSix | Weight MSE | Stale256 | Fixed budget | Relinearized | ΔNLL ±2SE vs baseline |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in batches:
        c = r['contrasts'][d]['four_over_six']
        lines.append('| '+d+' | '+' | '.join(f'{r["evaluation"][p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', 'Fit audits do not change the final map.', '', '```json', json.dumps({p: {k: v for k, v in a.items() if k != 'examples'} for p, a in r['fit_audit'].items()}, indent=2), '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
