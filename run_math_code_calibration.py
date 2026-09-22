"""One shared causal math/code scoring pass for adaptive and fixed-count maps."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize.adaptive_prefix import derive_maps, source_subsets
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha

FINE_KS = (0, 1, 2, 3)
ORIGINS = {
    'qwen4b': 'results/pooled_scale/model_332389_qwen4b',
    'llama8b': 'results/pooled_scale/model_332389_llama8b',
    'qwen27b': 'results/pooled_qwen27b/model_332840',
}


def math_code_data(tok, previous_fit):
    batches, metadata = {}, {}
    for source in ('math', 'code'):
        meta = previous_fit[source]
        assert meta['repo'] == {'math': 'open-web-math/open-web-math',
                                'code': 'codeparrot/codeparrot-clean'}[source]
        wanted = {d['document_sha256']: d['offset'] for d in meta['documents']}
        assert len(wanted) == 64
        stream = load_dataset(meta['repo'], revision=meta['revision'],
                              data_files={'train': meta['path']}, split='train', streaming=True)
        found = {}
        for row in stream:
            text = row['text' if source == 'math' else 'content']
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in wanted or digest in found:
                continue
            ids = tok(text, return_tensors='pt').input_ids
            offset = wanted[digest]
            assert ids.shape[1] >= offset + 512
            found[digest] = ids[:, offset:offset+512].clone()
            if len(found) == 64:
                break
        assert set(found) == set(wanted)
        batches[source] = [found[d['document_sha256']] for d in meta['documents']]
        assert [sha(b) for b in batches[source]] == meta['token_sha256']
        metadata[source] = dict(meta)
    assert len({d['document_sha256'] for meta in metadata.values() for d in meta['documents']}) == 128
    return batches, metadata


def load_model(prior, target):
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], revision=prior['revision'], dtype=torch.bfloat16,
            attn_implementation='eager', device_map='balanced', max_memory={0:'65GiB', 1:'65GiB'},
            output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        modules = {n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
            torch_dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda')
        modules = {n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                   and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    assert list(modules) == list(prior['matrices'])
    return model, modules


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=ORIGINS, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-source', default=None,
                    help='Relocated checkpoint or Hub ID; revision and every weight hash remain checked.')
    ap.add_argument('--adaptive-only', action='store_true')
    ap.add_argument('--compact-scores', action='store_true',
        help='Persist fine pilot scores and validated 128-sequence 8x64/256x64 masks, but omit large historical score tables.')
    ap.add_argument('--summary-scores', action='store_true',
        help='With --compact-scores, also persist threshold-sweep statistics: per-8x64-tile CE/KL '
             'mean and std, and per-sequence CE/KL sums over 256x64 tiles (~0.7 GB for Llama-8B).')
    ap.add_argument('--fine-masks', action='store_true',
        help='Accumulate per-1x16-atom CE/KL mean and variance for every module on the GPU and '
             'persist bit-packed 1x16 election masks (both/ce/kl, k=3). Accuracy ceiling only.')
    ap.add_argument('--retry-incomplete', action='store_true',
                    help='Recompute an incomplete full-model export, preserving validated pilot shards.')
    ap.add_argument('--reorder-modules', default=None,
        help='Regex selecting modules whose CE/KL scores are additionally streamed as 1x16 '
             'atoms to reorder_scores/. Required for genuine column regrouping; historical '
             '8x64 shards cannot be split into these atoms. Start with a few pilot modules.')
    ap.add_argument('--allow-source-drift', action='store_true',
        help='Proceed when quantizer sources differ from the origin job, recording exactly which '
             'files differ. Intended for a re-run whose output is itself checked against the '
             'shipped map digest; the file hash is then the weaker of the two guards.')
    args = ap.parse_args()
    target = args.model == 'qwen27b'
    torch.set_num_threads(12 if target else 4); torch.backends.cuda.matmul.allow_tf32 = False
    old = Path(ORIGINS[args.model]); prior = json.loads((old/'report.json').read_text())
    origin_source = prior['source']
    if args.model_source:
        prior['source'] = args.model_source
    assert prior['status'] == 'complete' and transformers.__version__ == prior['transformers_version']
    out = Path(args.out)
    previous = None
    if args.retry_incomplete:
        if args.compact_scores:
            raise ValueError('Compact-score mode is only for a fresh calibration')
        if args.reorder_modules:
            raise ValueError('Retry preserves existing pilot shards; omit --reorder-modules')
        previous = json.loads((out / 'report.json').read_text())
        if previous['status'] == 'complete' or previous['model'] != args.model:
            raise ValueError('Only retry an incomplete export of the same model')
        pilot = json.loads((out.parent / 'pilot_calibration/report.json').read_text())
        if pilot['status'] != 'pilot_complete' or not pilot['pilot_scores_verified']:
            raise ValueError('Validate and preserve the pilot before retrying')
        for key in ('source', 'revision', 'transformers_version'):
            assert previous[key] == pilot[key]
        assert previous['source'] == prior['source'] and previous['revision'] == prior['revision']
    out.mkdir(parents=True, exist_ok=args.retry_incomplete)
    score_dir = out/'scores'; score_dir.mkdir(exist_ok=args.retry_incomplete)
    # df on the parent mount can report filesystem capacity instead of the
    # user's project quota. Check the actual output directory before compute.
    expected_bytes = sum(math.prod(m['shape']) // 512 * 128 * 3 * 4 for m in prior['matrices'].values())
    if args.compact_scores:
        expected_bytes = sum(math.prod(m['shape']) // 512 * 2 for m in prior['matrices'].values())
    if args.summary_scores:
        assert args.compact_scores, '--summary-scores requires --compact-scores'
        expected_bytes += sum(math.prod(m['shape']) // 512 * 4 * 4 + math.prod(m['shape']) // 16384 * 128 * 2 * 4
                              for m in prior['matrices'].values())
    if args.reorder_modules:
        expected_bytes += sum(math.prod(m['shape']) // 16 * 128 * 2 * 4
                              for n, m in prior['matrices'].items() if re.search(args.reorder_modules, n))
    reusable_bytes = sum(p.stat().st_size for p in score_dir.glob('*.pt'))
    if shutil.disk_usage(out).free + reusable_bytes < expected_bytes + 1024**3:
        raise RuntimeError('Insufficient output quota for complete calibration scores plus 1 GiB headroom')
    teacher_dir = Path(os.environ.get('TMPDIR', os.environ['HF_HOME']))/'adaptive_teacher'
    teacher_dir.mkdir()
    files = ('run_math_code_calibration.py', 'quantize/adaptive_prefix.py',
             'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'results/math_code_adaptive/PROTOCOL.md')
    if args.reorder_modules:
        files += ('quantize/task_reorder.py',)
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
        job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__, transformers_version=transformers.__version__,
        origin=str(old), source_sha256={f:digest_file(f) for f in files}, matrices={},
        activation_convention='causal per-token factors for scoring and evaluation',
        calibration_sources=['OpenWebMath','CodeParrot'], uses_c4_calibration=False, uses_wiki_calibration=False,
        subsets=source_subsets(), fixed256_comparison=not args.adaptive_only)
    r['historical_scores_persisted'] = not args.compact_scores
    r['origin_source'] = origin_source
    if previous:
        r['reorder_modules'] = previous['reorder_modules']
        r['reused_pilot_source_job'] = pilot['job_id']
        r['reused_pilot_validation'] = str(out.parent / 'pilot_calibration/report.json')
    save(out,r)
    # The origin job pinned the quantizer sources by digest. A later commit can invalidate that
    # digest without changing any function this pass calls -- 384b803 appended
    # quant_nvfp4_4over6_pair and touched nothing else -- so --allow-source-drift downgrades the
    # check to a recorded difference. It is not a way to skip verification: the caller is
    # expected to compare the resulting map digest against the shipped one, which tests the
    # thing the file hash is standing in for.
    drift = {f: dict(origin=prior['source_sha256'][f], now=r['source_sha256'][f])
             for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py')
             if r['source_sha256'][f] != prior['source_sha256'][f]}
    r['source_drift'] = drift
    if drift:
        assert args.allow_source_drift, f'quantizer sources differ from origin: {sorted(drift)}'
        print('SOURCE DRIFT ALLOWED: ' + ', '.join(sorted(drift)), flush=True)
    save(out, r)
    model, modules = load_model(prior,target)
    reorder_dirs = {}
    if args.reorder_modules:
        pattern = re.compile(args.reorder_modules)
        chosen = [name for name in modules if pattern.search(name)]
        if not chosen:
            raise ValueError('--reorder-modules matched no quantized modules')
        for index, name in enumerate(chosen):
            directory = out / 'reorder_scores' / f'{index:03d}'
            directory.mkdir(parents=True)
            reorder_dirs[name] = directory
        r['reorder_modules'] = {name: str(path) for name, path in reorder_dirs.items()}
    for name,m in modules.items():
        assert sha(m.weight) == prior['matrices'][name]['source_sha256']
        r['matrices'][name] = prior['matrices'][name]
    r['source_weights_verified'] = True
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    fit,r['fit'] = math_code_data(tok,prior['fit'])
    batches = [b for values in fit.values() for b in values]
    sequence_ids = [digest for meta in r['fit'].values() for digest in meta['token_sha256']]
    sequence_sources = [source for source, values in fit.items() for _ in values]
    if previous:
        assert r['fit'] == pilot['fit']
        for directory in r['reorder_modules'].values():
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            assert manifest['sequence_ids'] == sequence_ids and manifest['sequence_sources'] == sequence_sources
    fine_pending = {}
    fine_moments = {}
    device = model.get_input_embeddings().weight.device
    r['bf16_fit_nll'] = []
    with torch.no_grad():
        for i,batch in enumerate(batches):
            ids=batch.to(device); logits=model(input_ids=ids,use_cache=False).logits
            lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
            assert torch.isfinite(lp).all()
            r['bf16_fit_nll'].append(float(F.nll_loss(lp,ids[:,1:].reshape(-1).to(lp.device))))
            torch.save(lp.cpu(),teacher_dir/f'{i:03d}.pt')
            if (i+1)%16 == 0: print(f'TEACHER {i+1}/128',flush=True)
        del ids,logits,lp
    save(out,r)
    base,alt,directions,mse = {},{},{},{}
    cached_bytes = {}
    with torch.no_grad():
        for i,(name,m) in enumerate(modules.items()):
            w=m.weight.detach(); o,k=w.shape
            b=quant_nvfp4_4over6(w,4,16)
            a=quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always')
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            diff=(a.float()-w.float()).square()-(b.float()-w.float()).square()
            mse[name]=(diff.reshape(o//8,8,k//64,64).sum((1,3))<0).cpu()
            base[name],alt[name]=(b.cpu().pin_memory(),a.cpu().pin_memory()) if target else (b,a)
            key=str(w.device); size=w.numel()*4; limit=(12 if target else 4)*1024**3
            if cached_bytes.get(key,0)+size <= limit:
                directions[name]=a.float()-b.float(); cached_bytes[key]=cached_bytes.get(key,0)+size
            m.weight.copy_(b)
            if (i+1)%64 == 0: print(f'CANDIDATES {i+1}/{len(modules)}',flush=True)
        del a,b,w,diff
    tables={n:[torch.empty(128,m.weight.numel()//512) for _ in range(3)] for n,m in modules.items()}
    hits={n:[0,0,0] for n in modules}; phase=0; sequence=0

    def act(module,inputs):
        x=inputs[0]; q=quantize_rows(x.detach())
        return (q+(x-x.detach()),*inputs[1:])

    def make_hook(name):
        def forward(module,inputs,output):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
            def backward(dy):
                grad=dy.detach().reshape(-1,dy.shape[-1]).float().T@x.float()
                d=directions.get(name)
                if d is None:
                    d=alt[name].to(grad.device,non_blocking=True).float()-base[name].to(grad.device,non_blocking=True).float()
                o,k=grad.shape
                value=(grad*d).reshape(o//8,8,k//64,64).sum((1,3)).flatten()
                assert torch.isfinite(value).all(),name
                tables[name][phase][sequence].copy_(value.cpu()); hits[name][phase]+=1
                if args.fine_masks and phase in (0, 1):
                    atom = (grad*d).reshape(o,k//16,16).sum(-1).double()
                    acc = fine_moments.setdefault((name, phase), [torch.zeros_like(atom), torch.zeros_like(atom)])
                    acc[0] += atom; acc[1] += atom.square()
                if name in reorder_dirs and phase in (0, 1):
                    from quantize.task_reorder import scale_block_scores
                    fine = scale_block_scores(grad, d).cpu()
                    if not torch.isfinite(fine).all():
                        raise ValueError(f'Nonfinite fine score: {name}')
                    if phase == 0:
                        fine_pending[name] = fine
                    else:
                        torch.save(dict(ce=fine_pending.pop(name), kl=fine,
                                        sequence_id=sequence_ids[sequence]),
                                   reorder_dirs[name] / f'{sequence:03d}.pt')
            output.register_hook(backward)
        return forward

    handles=[m.register_forward_pre_hook(act) for m in modules.values()]
    handles += [m.register_forward_hook(make_hook(n)) for n,m in modules.items()]
    generator=None; r['initial_fit_losses']=[]; start=time.perf_counter()
    for sequence,batch in enumerate(batches):
        ids=batch.to(device); embeds=model.get_input_embeddings()(ids).detach().requires_grad_()
        logits=model(inputs_embeds=embeds,use_cache=False).logits
        lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
        path=teacher_dir/f'{sequence:03d}.pt'
        teacher=torch.load(path,map_location='cpu',weights_only=True).to(lp.device)
        ce=F.nll_loss(lp,ids[:,1:].reshape(-1).to(lp.device))
        kl=F.kl_div(lp,teacher,reduction='batchmean',log_target=True)
        phase=0; ce.backward(retain_graph=True)
        phase=1; kl.backward(retain_graph=True)
        if generator is None: generator=torch.Generator(device=lp.device).manual_seed(20260930)
        with torch.no_grad(): labels=torch.multinomial(lp.detach().exp(),1,generator=generator).squeeze(-1)
        sampled=F.nll_loss(lp,labels); phase=2; sampled.backward()
        assert all(math.isfinite(float(v)) for v in (ce,kl,sampled))
        r['initial_fit_losses'].append(dict(ce=float(ce),kl=float(kl),sampled_ce=float(sampled)))
        del embeds,logits,lp,teacher,ce,kl,sampled,labels
        path.unlink()
        if (sequence+1)%8 == 0:
            save(out,r); print(f'SCORED {sequence+1}/128 {time.perf_counter()-start:.1f}s',flush=True)
    assert all(v == [128,128,128] for v in hits.values())
    for h in handles: h.remove()
    del directions,model,modules,base,alt
    torch.cuda.empty_cache()
    r['score_seconds']=time.perf_counter()-start
    if args.fine_masks:
        import numpy as np
        fine_dir = out / 'fine_masks'; fine_dir.mkdir()
        counts = {}
        for i, name in enumerate(tables):
            bounds = {}
            for phase, key in ((0, 'ce'), (1, 'kl')):
                total, square = fine_moments.pop((name, phase))
                mean = total / 128
                std = ((square - 128 * mean.square()).clamp_min(0) / 127).sqrt()
                bounds[key] = (mean, std / math.sqrt(128))
            rules = {}
            for kk in FINE_KS:
                b = {key: m + kk * se for key, (m, se) in bounds.items()}
                suffix = '' if kk == 3 else f'_k{kk}'
                rules.update({'both' + suffix: torch.maximum(b['ce'], b['kl']) < 0,
                              'ce' + suffix: b['ce'] < 0, 'kl' + suffix: b['kl'] < 0})
            packed = {rule: torch.from_numpy(np.packbits(mask.cpu().numpy().reshape(-1)))
                      for rule, mask in rules.items()}
            for rule, mask in rules.items():
                counts[rule] = counts.get(rule, 0) + int(mask.sum())
            torch.save(dict(name=name, shape=list(rules['both'].shape), k=3, ks=list(FINE_KS), packed=packed),
                       fine_dir / f'{i:03d}.pt')
            del bounds, rules
        r['fine_mask_counts'] = counts
        save(out, r)
    for name, directory in reorder_dirs.items():
        manifest = dict(schema='mixfp4_reorder_scores_v1', status='complete', name=name,
                        atom_shape=[1, 16], weight_shape=r['matrices'][name]['shape'],
                        sequence_ids=sequence_ids, sequence_sources=sequence_sources,
                        source=r['source'], revision=r['revision'],
                        baseline='canonical FourOverSix', alternative='E0M3 alpha1',
                        activation_convention=r['activation_convention'],
                        source_sha256=r['source_sha256'],
                        weight_sha256=r['matrices'][name]['source_sha256'],
                        calibration_report=str(out / 'report.json'))
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    compact = None
    if args.compact_scores:
        from run_task_reorder_eval import coarse_mask, validate_compact_masks
        compact = dict(schema='mixfp4_compact_masks_v1', sequences=128, k=3, revision=r['revision'],
            accumulation='float64 per-sequence sums before mean+3SE', raw256={}, fine8x64={},
            weight_sha256={name: value['source_sha256'] for name, value in r['matrices'].items()})
    for i,(name,values) in enumerate(tables.items()):
        if args.summary_scores:
            shape = r['matrices'][name]['shape']
            summary = dict(name=name, shape=shape)
            for key, value in zip(('ce', 'kl'), values[:2]):
                v = value.double()
                summary[key + '_mean8'] = v.mean(0).float()
                summary[key + '_std8'] = v.std(0, unbiased=True).float()
                # Same float64 per-sequence aggregation as run_task_reorder_eval.coarse_mask.
                g = v.reshape(128, shape[0] // 8, shape[1] // 64)
                g = F.pad(g, (0, 0, 0, (-g.shape[1]) % 32)).reshape(128, -1, 32, shape[1] // 64).sum(2)
                summary[key + '_seq256'] = g
            torch.save(summary, score_dir / f'{i:03d}.pt')
        if compact is not None:
            shape = r['matrices'][name]['shape']
            compact['raw256'][name] = coarse_mask(values[0], values[1], shape)
            bounds = [v.double().mean(0) + 3 * v.double().std(0) / math.sqrt(128) for v in values[:2]]
            assert all(torch.isfinite(v).all() for v in bounds)
            compact['fine8x64'][name] = (torch.maximum(*bounds) < 0).reshape(shape[0] // 8, shape[1] // 64)
            continue
        destination = score_dir/f'{i:03d}.pt'
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(dict(name=name,ce=values[0],kl=values[1],fisher=values[2]), temporary)
        stored = torch.load(temporary, map_location='cpu', weights_only=True)
        assert stored['name'] == name
        assert all(torch.equal(stored[key], value) for key, value in zip(('ce', 'kl', 'fisher'), values))
        del stored
        temporary.replace(destination)
    if compact is not None:
        path = out / 'compact_masks.pt'
        torch.save(compact, path)
        stored = validate_compact_masks(torch.load(path, map_location='cpu', weights_only=True), r)
        assert all(torch.equal(stored[kind][name], mask) for kind in ('raw256', 'fine8x64')
                   for name, mask in compact[kind].items())
        r['compact_mask_sha256'] = digest_file(path)
        r['compact_mask_counts'] = {kind: sum(int(mask.sum()) for mask in compact[kind].values())
                                    for kind in ('raw256', 'fine8x64')}
        del stored, compact
    maps,stats=derive_maps(((n,r['matrices'][n]['shape'],*values) for n,values in tables.items()),
                           include_fixed=not args.adaptive_only)
    del tables
    total=sum(m['shape'][0]//8*(m['shape'][1]//64) for m in r['matrices'].values())
    r['block_statistics']=stats
    r['block_statistics']['four_over_six']=dict(selected_blocks=0,total_type_blocks=total,selected_fraction=0.)
    mse_count=sum(int(m.sum()) for m in mse.values())
    r['block_statistics']['weight_mse']=dict(selected_blocks=mse_count,total_type_blocks=total,selected_fraction=mse_count/total)
    torch.save(dict(source=r['source'],revision=r['revision'],weight_mse=mse),out/'weight_mse.pt')
    r['weight_mse_sha256']=digest_file(out/'weight_mse.pt')
    (out/'maps.json').write_text(json.dumps(dict(source=r['source'],revision=r['revision'],
        type_block=[8,64],baseline='FourOverSix',alternative='E0M3 alpha1',maps=maps),indent=2)+'\n')
    r['map_sha256']=digest_file(out/'maps.json'); r['maps_frozen']=True
    r['status']='complete'; save(out,r)
    print('FROZEN '+json.dumps({p:s['selected_blocks'] for p,s in r['block_statistics'].items()}),flush=True)


if __name__ == '__main__':
    main()
