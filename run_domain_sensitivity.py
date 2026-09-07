"""Prespecified domain-transfer and consensus experiment; Slurm only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit through Slurm, including tests and summaries.')
for key, subdir in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                    ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], subdir))
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
import analyze_task_sensitivity as task
from probe_qwen38 import NativeActivationQuantization, native_loss, text_modules
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from domain_score_backend import checked_scores


def digest(batches):
    return hashlib.sha256(torch.cat(batches, 1).numpy().tobytes()).hexdigest()


def c4_windows(tok, revision, split, shard, count, seed):
    path = f'en/c4-{split}.{shard:05d}-of-{1024 if split == "train" else 8:05d}.json.gz'
    stream = load_dataset('allenai/c4', revision=revision, data_files={split: path},
                          split=split, streaming=True).shuffle(seed=seed, buffer_size=1000)
    rng, batches, docs, seen = random.Random(seed), [], [], set()
    for row in stream:
        h = hashlib.sha256(row['text'].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        ids = tok(row['text'], return_tensors='pt').input_ids
        if ids.shape[1] < 2048:
            continue
        offset = rng.randrange(ids.shape[1]-2048+1)
        batches.append(ids[:, offset:offset+2048].clone())
        docs.append({'document_sha256': h, 'offset': offset})
        if len(batches) == count:
            return batches, {'revision': revision, 'path': path, 'seed': seed,
                             'documents': docs, 'tokens_sha256': digest(batches)}
    raise RuntimeError(f'Insufficient qualifying documents: {path}')


def pool(a, b, na=32, nb=32):
    """Exact sample mean/SE pooling, including between-group variation."""
    ma, sa = a
    mb, sb = b
    n = na+nb
    means = {k: (na*ma[k].double()+nb*mb[k].double())/n for k in ma}
    ses = {k: ((na*(na-1)*sa[k].double().square()
                + nb*(nb-1)*sb[k].double().square()
                + na*nb/n*(ma[k].double()-mb[k].double()).square())/(n*(n-1))).sqrt()
           for k in ma}
    return means, ses


def consensus_scores(a, b):
    means = {k: torch.maximum(a[0][k], b[0][k]) for k in a[0]}
    # Ineligible tiles are zeroed; trust_masks uses strict negativity.
    for k in means:
        eligible = (a[0][k]+2*a[1][k] < 0) & (b[0][k]+2*b[1][k] < 0)
        means[k] = torch.where(eligible, means[k], 0.)
    return means, {k: torch.zeros_like(v) for k, v in means.items()}


def consensus_fit(model, modules, base, alt, scores, fits, baselines, domain_scores):
    history = []
    for i in range(8):
        budget = 0.1*0.5**i
        masks = task.trust_masks(*scores, budget)
        summary = task.selected_summary(masks, scores[0])
        if not summary['tiles']:
            break
        task.install(modules, base, alt, masks)
        checks = {}
        for d in fits:
            actual = task.paired(task.evaluate(model, fits[d], f'consensus-fit/{i}/{d}'), baselines[d])
            predicted = task.selected_summary(masks, domain_scores[d][0])['predicted_delta_nll']
            checks[d] = {**actual, 'predicted': predicted, 'ratio': actual['mean']/predicted}
        accepted = all(v['mean']+2*v['se'] < 0 and v['ratio'] >= .25 for v in checks.values())
        history.append({**summary, 'budget': budget, 'domains': checks, 'accepted': accepted})
        if accepted:
            return masks, history
    return {k: torch.zeros_like(v, dtype=torch.bool) for k, v in scores[0].items()}, history


def export(path, masks, probe, fit_hash, rule):
    task.export_type_map(path, probe['model'], masks, fit_hash, probe['model_commit'], rule,
                         weight_baseline='nvfp4_4over6')
    spec = json.loads(path.read_text())
    spec.update(scope='qwen3_5_text_linear', apply_function='probe_qwen38.apply_native_type_map')
    task.atomic_json(path, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wiki-seeds', type=int, nargs='+', default=[20260912, 20260913])
    ap.add_argument('--out', default='results/task_sensitivity_domains')
    args = ap.parse_args()
    assert torch.cuda.device_count() == 2
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    probe = json.loads(Path('results/task_sensitivity_qwen38_probe/report.json').read_text())
    assert transformers.__version__ == probe['transformers']
    revisions = {n: HfApi().dataset_info(n).sha for n in ['allenai/c4', 'Salesforce/wikitext']}
    tok = AutoTokenizer.from_pretrained(probe['model'], revision=probe['model_commit'])
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        probe['model'], revision=probe['model_commit'], dtype=torch.bfloat16,
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}, output_loading_info=True)
    assert not loading['missing_keys']
    model.eval()
    modules = text_modules(model)
    assert {n: list(m.weight.shape) for n, m in modules.items()} == probe['quantized_shapes']
    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16).cpu()
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always').cpu()
            m.weight.copy_(base[n])
    wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
    task.activation_ste, task.loss = wrapper.backward, native_loss
    # Held-out data are shared; no evaluation occurs until all maps are frozen.
    data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                        revision=revisions['Salesforce/wikitext'], split='validation')
    ids = tok('\n\n'.join(data['text']), return_tensors='pt').input_ids
    wiki_eval = list(ids[:, :ids.shape[1]//2048*2048].split(2048, 1))
    c4_eval, eval_meta = c4_windows(tok, revisions['allenai/c4'], 'validation', 1, 256, 20260916)
    evals = {'wikitext': wiki_eval, 'c4': c4_eval}
    pending = []
    for seed in args.wiki_seeds:
        torch.manual_seed(seed)
        out = root/f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        assert not (out/'report.json').exists(), 'Use a fresh directory.'
        olddir = Path(f'results/task_sensitivity_four_over_six/seed{seed}')
        old = json.loads((olddir/'report.json').read_text())
        assert old['complete'] and old['model_commit'] == probe['model_commit']
        source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in old['dependency_sha256']}
        for p, h in old['dependency_sha256'].items():
            if p != 'analyze_task_sensitivity.py':
                assert source_hashes[p] == h, p
        wf, wv, wp, wh = task.data_splits(tok, 2048, 64, 16, seed, dataset_name='Salesforce/wikitext')
        assert wh == old['data_sha256']
        cb, cm = c4_windows(tok, revisions['allenai/c4'], 'train', 0, 88, seed+2)
        cf, cv, cp = cb[:64], cb[64:80], cb[80:]
        assert not ({x['document_sha256'] for x in cm['documents']}
                    & {x['document_sha256'] for x in eval_meta['documents']})
        r = {'complete': False, 'seed': seed, 'c4_seed': seed+2, 'model': probe['model'],
             'model_commit': probe['model_commit'], 'job_id': os.environ['SLURM_JOB_ID'],
             'transformers': transformers.__version__, 'torch': torch.__version__,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'dependency_sha256': source_hashes,
             'dataset_revisions': revisions, 'wiki_hashes': wh, 'c4_data': cm,
             'eval_c4_data': eval_meta, 'eval_wiki_sha256': digest(wiki_eval),
             'policies': {}, 'probes': [], 'final': {}}
        def save():
            task.atomic_json(out/'report.json', r)
        save()
        task.install(modules, base, alt)
        baselines = {d: task.evaluate(model, batches, 'baseline/'+d)
                     for d, batches in [('wiki_fit', wf), ('wiki_val', wv), ('wiki_probe', wp),
                                        ('c4_fit', cf), ('c4_val', cv), ('c4_probe', cp)]}
        assert baselines['wiki_fit'] == old['baseline']['fit']
        assert baselines['wiki_val'] == old['baseline']['val']
        r['baseline_calibration'] = baselines
        save()
        cache = torch.load(olddir/'scores.pt', weights_only=True)
        assert cache['hashes'] == wh and cache['model_commit'] == probe['model_commit']
        assert cache['args']['weight_baseline'] == 'nvfp4_4over6'
        if source_hashes['analyze_task_sensitivity.py'] != old['dependency_sha256']['analyze_task_sensitivity.py']:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                means, ses, losses = checked_scores(model, modules, base, alt, wf)
            assert losses == baselines['wiki_fit']
            r['wiki_scores_recomputed'] = True
            r['historical_score_max_abs_difference'] = max(float((means[n]-cache['means'][n]).abs().max()) for n in means)
            scores = {'wiki': (means, ses)}
            torch.save({'means': means, 'ses': ses, 'losses': losses, 'tokens_sha256': digest(wf)}, out/'wiki64_scores.pt')
        else:
            r['wiki_scores_recomputed'] = False
            scores = {'wiki': (cache['means'], cache['ses'])}
        for label, batches in [('wiki32', wf[:32]), ('c4a', cf[:32]), ('c4b', cf[32:])]:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                means, ses, losses = checked_scores(model, modules, base, alt, batches)
            expected = baselines['wiki_fit'][:32] if label == 'wiki32' else baselines['c4_fit'][0 if label == 'c4a' else 32:32 if label == 'c4a' else 64]
            assert losses == expected
            scores[label] = (means, ses)
            torch.save({'means': means, 'ses': ses, 'losses': losses, 'tokens_sha256': digest(batches)}, out/(label+'_scores.pt'))
        scores['c4'] = pool(scores['c4a'], scores['c4b'])
        scores['mixed'] = pool(scores['wiki32'], scores['c4a'])
        scores['consensus'] = consensus_scores(scores['wiki32'], scores['c4a'])
        policies = {}
        for label in ['wiki', 'c4', 'mixed', 'consensus']:
            task.install(modules, base, alt)
            fit_batches = wf if label == 'wiki' else cf if label == 'c4' else wf[:32]+cf[:32]
            fit_base = baselines['wiki_fit'] if label == 'wiki' else baselines['c4_fit'] if label == 'c4' else baselines['wiki_fit'][:32]+baselines['c4_fit'][:32]
            if label == 'wiki':
                masks = torch.load(olddir/'policies.pt', weights_only=True)['gradient_trust_backtracking']
                history = old['proposal_fit_calibration']
            elif label == 'consensus':
                masks, history = consensus_fit(model, modules, base, alt, scores[label],
                    {'wiki': wf[:32], 'c4': cf[:32]},
                    {'wiki': baselines['wiki_fit'][:32], 'c4': baselines['c4_fit'][:32]},
                    {'wiki': scores['wiki32'], 'c4': scores['c4a']})
            else:
                masks, history = task.calibrated_trust_masks(model, modules, base, alt, *scores[label], fit_batches, fit_base)
            task.install(modules, base, alt, masks)
            nv = 16 if label in ('wiki', 'c4') else 8
            vals = {d: task.evaluate(model, batches[:nv], label+'/val/'+d) for d, batches in [('wiki', wv), ('c4', cv)]}
            checks = {d: task.paired(v, baselines[d+'_val'][:nv]) for d, v in vals.items()}
            if label in ('wiki', 'c4'):
                gate = checks[label]
                accepted = gate['mean']+2*gate['se'] < 0
            elif label == 'mixed':
                gate = task.paired(vals['wiki']+vals['c4'], baselines['wiki_val'][:nv]+baselines['c4_val'][:nv])
                accepted = gate['mean']+2*gate['se'] < 0
            else:
                accepted = all(v['mean']+2*v['se'] < 0 for v in checks.values())
            if label == 'wiki':
                assert vals['wiki'] == old['results']['gradient_trust_backtracking']['val_nll']
                assert accepted == (old['export_decision'] == 'accepted')
            r['policies'][label] = {**task.selected_summary(masks, scores[label][0]),
                                    'history': history, 'validation': checks, 'accepted': accepted}
            policies[label] = masks
            export(out/(label+'_candidate.json'), masks, probe, digest(fit_batches), label)
            export(out/(label+'_export.json'), masks if accepted else {k: torch.zeros_like(v) for k, v in masks.items()}, probe, digest(fit_batches), label)
            save()
        torch.save(policies, out/'policies.pt')
        # Diagnostic selection fixed before measured isolated effects.
        names = list(scores['wiki'][0])
        wm, ws, cmu, cs = [torch.cat([s[n].flatten() for n in names]) for s in (*scores['wiki'], *scores['c4'])]
        categories = {'wiki_best': torch.ones_like(wm, dtype=torch.bool), 'c4_best': torch.ones_like(wm, dtype=torch.bool),
                      'wiki_good_c4_bad': (wm+2*ws < 0) & (cmu-2*cs > 0),
                      'c4_good_wiki_bad': (cmu+2*cs < 0) & (wm-2*ws > 0)}
        selected = set()
        r['score_sign_counts'] = {k: int(v.sum()) for k, v in categories.items() if 'bad' in k}
        r['diagnostic_categories'] = {}
        for category, valid in categories.items():
            inds = valid.nonzero().flatten()
            rank = cmu if category.startswith('c4') else wm
            indices = inds[torch.argsort(rank[inds])[:4]].tolist()
            r['diagnostic_categories'][category] = indices
            selected.update(indices)
        for flat_index in sorted(selected):
            offset = 0
            for n in names:
                size = scores['wiki'][0][n].numel()
                if flat_index < offset+size:
                    local = flat_index-offset
                    break
                offset += size
            mask = torch.zeros_like(scores['wiki'][0][n], dtype=torch.bool)
            mask.flatten()[local] = True
            task.install(modules, base, alt, {n: mask})
            entry = {'module': n, 'tile_flat_index': local, 'predicted': {}, 'measured': {}}
            for d, batches in [('wiki', wp), ('c4', cp)]:
                entry['predicted'][d] = {'mean': float(scores[d][0][n].flatten()[local]), 'se': float(scores[d][1][n].flatten()[local])}
                entry['measured'][d] = task.paired(task.evaluate(model, batches, 'isolated/'+d), baselines[d+'_probe'])
            r['probes'].append(entry)
            save()
        r['baseline_calibration'] = baselines
        save()
        pending.append((out, r, policies))
    # All seed proposals and probes are frozen before first held-out loss.
    task.install(modules, base, alt)
    baseline_eval = {d: task.evaluate(model, batches, 'heldout/baseline/'+d) for d, batches in evals.items()}
    for out, r, policies in pending:
        r['final']['baseline'] = {d: {'nll': v, 'ppl': math.exp(sum(v)/len(v))} for d, v in baseline_eval.items()}
        for label, masks in policies.items():
            task.install(modules, base, alt, masks)
            r['final'][label] = {}
            for d, batches in evals.items():
                values = task.evaluate(model, batches, 'heldout/'+label+'/'+d)
                r['final'][label][d] = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                                      'vs_baseline': task.paired(values, baseline_eval[d])}
                task.atomic_json(out/'report.json', r)
        r['complete'] = True
        task.atomic_json(out/'report.json', r)
        print('DONE', out, flush=True)
    wrapper.close()


if __name__ == '__main__':
    main()
