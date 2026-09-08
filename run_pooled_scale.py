"""Frozen pooled-source scoring and causal replay at 4B and 8B scale."""
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
from quantize.relinearized_format import stale_map
from quantize.causal_four_over_six import quantize_rows


def shared_data(tok):
    web, web_meta = fit_data(tok)
    batches = {'web': web}; metadata = {'web': web_meta}
    seen = {d['document_sha256'] for d in web_meta['documents']}
    for domain, repo, field, suffix in [('math', 'open-web-math/open-web-math', 'text', '.parquet'),
                                      ('code', 'codeparrot/codeparrot-clean', 'content', '.json.gz')]:
        api = HfApi(); revision = api.dataset_info(repo).sha
        path = sorted(p for p in api.list_repo_files(repo, repo_type='dataset', revision=revision) if p.endswith(suffix))[0]
        stream = load_dataset(repo, revision=revision, data_files={'train': path}, split='train', streaming=True)
        docs = []; bs = []; rng = random.Random(20260926)
        for row in stream:
            text = row[field]; digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest); ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < 512: continue
            offset = rng.randrange(ids.shape[1]-512+1)
            bs.append(ids[:, offset:offset+512].clone()); docs.append(dict(document_sha256=digest, offset=offset))
            if len(bs) == 64: break
        assert len(bs) == 64
        batches[domain] = bs; metadata[domain] = dict(repo=repo, revision=revision, path=path, documents=docs, token_sha256=[sha(b) for b in bs])
    return batches, metadata


def data(tok, excluded):
    sources = [('literature', 'emozilla/pg19-test', None, 'text'),
               ('science', 'ccdv/arxiv-summarization', 'document', 'article'),
               ('government', 'ccdv/govreport-summarization', 'document', 'report')]
    earlier = json.loads(Path('results/pooled_confirmation/model_332349_llama1b/report.json').read_text())
    revisions = {m['repo']: m['revision'] for m in earlier['confirmation_data'].values()}
    result = {}; metadata = {}
    for domain, repo, config, field in sources:
        ds = load_dataset(repo, config, revision=revisions[repo], split='test', streaming=True)
        bs = []; docs = []; seen = set(); exclusions = []; rng = random.Random(20260925)
        for row in ds:
            text = row[field]; digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in excluded: exclusions.append(digest); continue
            if digest in seen: continue
            seen.add(digest); ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < 512: continue
            offset = rng.randrange(ids.shape[1]-512+1)
            bs.append(ids[:, offset:offset+512].clone()); docs.append(dict(document_sha256=digest, offset=offset))
            if len(bs) == 64: break
        assert len(bs) == 64
        result[domain] = bs
        metadata[domain] = dict(repo=repo, revision=revisions[repo], config=config, field=field,
                               documents=docs, excluded_training_overlap=exclusions, token_sha256=[sha(b) for b in bs])
    return result, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['qwen4b', 'llama8b'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False); root = Path(__file__).resolve().parent
    source = {'qwen4b': 'Qwen/Qwen3-4B',
              'llama8b': '/work/u4320956/hf/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'}[args.model]
    old = root/f'results/consensus_format/model_332332_{args.model}'
    prior = json.loads((old/'report.json').read_text()) if old.exists() else None
    revision = Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
    files = ['run_pooled_scale.py', 'quantize/relinearized_format.py', 'run_directed_format.py',
             'quantize/quantizer.py', 'quantize/causal_four_over_six.py', 'tests/test_relinearized_format.py',
             'tests/test_causal_four_over_six.py', 'results/pooled_scale/PROTOCOL.md']
    r = dict(status='running', model=args.model, source=source, revision=revision,
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
             transformers_version=transformers.__version__,
             source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
             matrices={}, evaluation={}, fit_audit={}, score_origin=str(old) if prior else 'one shared scoring pass')
    if prior:
        assert prior['status'] == 'complete' and prior['source'] == source
        assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']
    save(out, r)
    kwargs = dict(revision=revision, torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs).eval().requires_grad_(False)
    teacher = AutoModelForCausalLM.from_pretrained(source, **kwargs).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    names = list(modules); base = {}; alt = {}; mse_maps = {}; slices = {}; count = 0
    with torch.no_grad():
        for n, m in modules.items():
            w = m.weight.detach(); o, k = w.shape
            assert o % 8 == 0 and k % 64 == 0
            r['matrices'][n] = dict(shape=list(w.shape), source_sha256=sha(w))
            if prior: assert r['matrices'][n] == prior['matrices'][n]
            base[n] = quant_nvfp4_4over6(w, 4, 16)
            alt[n] = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            difference = (alt[n].float()-w.float()).square() - (base[n].float()-w.float()).square()
            mse_maps[n] = (difference.reshape(o//8, 8, k//64, 64).sum((1, 3)) < 0).cpu()
            tiles = o//8 * (k//64); slices[n] = (count, count+tiles); count += tiles
            m.weight.copy_(base[n])
    scoring = True; row_mode = False
    def act(module, inputs):
        x = inputs[0]; q = quantize_rows(x.detach()) if row_mode else quant_nvfp4_4over6(x.detach(), 4, 16)
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
    fit, r['fit'] = shared_data(tok)
    if prior: assert r['fit'] == prior['fit']

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
    if prior:
        record = torch.load(old/'scores.pt', map_location='cpu', weights_only=True)
        assert record['names'] == names and record['slices'] == slices and record['sources'] == list(fit)
        ce, kl = record['ce'], record['kl']; del record
        r['initial_fit_losses'] = prior['initial_fit_losses']
    else:
        ces = []; kls = []; r['initial_fit_losses'] = {}
        for source_name, bs in fit.items():
            ce, kl, losses = score_batches(bs); ces.append(ce); kls.append(kl)
            r['initial_fit_losses'][source_name] = losses
            print(f'SCORED {source_name} 64 documents', flush=True)
        ce = torch.stack(ces); kl = torch.stack(kls); del ces, kls
        torch.save(dict(ce=ce, kl=kl, sources=list(fit), names=names, slices=slices), out/'scores.pt')
    mixed_ce = torch.cat((ce[0, :22], ce[1, :21], ce[2, :21]))
    mixed_kl = torch.cat((kl[0, :22], kl[1, :21], kl[2, :21]))
    flat_maps = dict(c4_64=stale_map(ce[0], kl[0]), mixed64=stale_map(mixed_ce, mixed_kl),
                     pooled192=stale_map(ce.flatten(0,1), kl.flatten(0,1)))
    del ce, kl, mixed_ce, mixed_kl
    r['score_seconds'] = time.perf_counter()-start
    maps = {'weight_mse': mse_maps, **{p: {} for p in flat_maps}}
    for n, (lo, hi) in slices.items():
        o, k = modules[n].weight.shape
        for p, flat in flat_maps.items():
            maps[p][n] = flat[lo:hi].reshape(o//8, k//64).clone()
    if prior:
        old_bundle = torch.load(old/'maps.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(maps['pooled192'][n], old_bundle['maps']['all_pooled'][n]) for n in names)
        del old_bundle
        r['primary_replay_exact'] = True
    torch.save(dict(source=source, revision=revision, maps=maps, baseline='FourOverSix', alternative='E0M3 alpha1', type_block=(8,64)), out/'maps.pt')
    r['maps_frozen'] = True; r['selected_tiles'] = {p: sum(int(m.sum()) for m in ms.values()) for p, ms in maps.items()}
    scoring = False; row_mode = True
    for h in handles: h.remove()
    save(out, r)
    print('FROZEN '+json.dumps(r['selected_tiles']), flush=True)
    excluded = {doc['document_sha256'] for meta in r['fit'].values() for doc in meta['documents']}
    batches, r['confirmation_data'] = data(tok, excluded)
    r['eval_token_sha256'] = {d: [sha(b) for b in bs] for d, bs in batches.items()}
    policies = ('four_over_six', *maps)
    r['suffix_intervention'] = {}
    with torch.no_grad():
        ids = batches['literature'][0].cuda(); changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for p in ('four_over_six', 'pooled192'):
            for n, m in modules.items(): m.weight.copy_(base[n] if p == 'four_over_six' else apply_mask(base[n], alt[n], maps[p][n].cuda()))
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            equal = torch.equal(a, b)
            r['suffix_intervention'][p] = dict(equal=equal, max_logit_difference=float((a.float()-b.float()).abs().max()))
            assert equal and torch.isfinite(a).all() and torch.isfinite(b).all()
        del a, b
    save(out, r); print('PREFIX '+json.dumps(r['suffix_intervention']), flush=True)
    for p in policies:
        with torch.no_grad():
            for n, m in modules.items(): m.weight.copy_(base[n] if p == 'four_over_six' else apply_mask(base[n], alt[n], maps[p][n].cuda()))
            if p in ('four_over_six', 'pooled192'):
                r['fit_audit'][p] = {}
                for source_name, bs in fit.items():
                    audit = []
                    for batch in bs:
                        ids = batch.cuda(); teacher_log = teacher(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1).reshape(-1, teacher.config.vocab_size)
                        logits = model(input_ids=ids, use_cache=False).logits
                        logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
                        audit.append(dict(ce=float(F.nll_loss(logprob, ids[:, 1:].reshape(-1))), kl=float(F.kl_div(logprob, teacher_log, reduction='batchmean', log_target=True))))
                    r['fit_audit'][p][source_name] = dict(examples=audit, ce=sum(a['ce'] for a in audit)/len(audit), kl=sum(a['kl'] for a in audit)/len(audit))
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
    r['contrasts'] = {d: {p: paired(r['evaluation']['pooled192'][d]['nll'], r['evaluation'][p][d]['nll']) for p in policies[:-1]} for d in batches}
    r['diversity_contrasts'] = {d: paired(r['evaluation']['mixed64'][d]['nll'], r['evaluation']['c4_64'][d]['nll']) for d in batches}
    r['status'] = 'complete'; save(out, r)
    lines = [f'# Pooled-source causal scale transfer: {args.model}', '',
             'Window-factor scoring, frozen maps, per-token-factor evaluation. No recalibration or acceptance gate.', '',
             '| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs baseline |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in batches:
        c = r['contrasts'][d]['four_over_six']
        lines.append('| '+d+' | '+' | '.join(f'{r["evaluation"][p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', 'Fit audits do not change any map.', '', '```json', json.dumps({p: {s: {k: v for k, v in a.items() if k != 'examples'} for s, a in ss.items()} for p, ss in r['fit_audit'].items()}, indent=2), '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
