"""Frozen cross-domain rule transfer to native Qwen3 and Llama; Slurm only."""
import run_domain_sensitivity as domain
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from probe_qwen38 import NativeActivationQuantization, native_loss
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from domain_stress_data import load_stress
from domain_score_backend import checked_scores
import analyze_task_sensitivity as task


ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['qwen3-4b', 'llama-3.1-8b-local'])
args = ap.parse_args()
seed = 20260918
torch.manual_seed(seed)
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = False
out = Path('results/task_sensitivity_domains/panel')/args.model
out.mkdir(parents=True, exist_ok=True)
assert not (out/'report.json').exists()
source = json.loads(Path('model2path.json').read_text())[args.model]
commit = Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
tok = AutoTokenizer.from_pretrained(source, revision=commit)
model, loading = AutoModelForCausalLM.from_pretrained(source, revision=commit, dtype=torch.bfloat16,
                                                    device_map='cuda:0', output_loading_info=True)
assert not loading['missing_keys']
model.eval()
modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n}
base, alt = {}, {}
with torch.no_grad():
    for n, m in modules.items():
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16).cpu()
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always').cpu()
        m.weight.copy_(base[n])
wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
task.activation_ste, task.loss = wrapper.backward, native_loss
revisions = {n: HfApi().dataset_info(n).sha for n in ['allenai/c4', 'Salesforce/wikitext']}
wf, wv, _, wh = task.data_splits(tok, 2048, 64, 16, seed, dataset_name='Salesforce/wikitext')
cb, cm = domain.c4_windows(tok, revisions['allenai/c4'], 'train', 0, 88, seed+2)
cf, cv = cb[:64], cb[64:80]
r = {'complete': False, 'model': args.model, 'model_source': source, 'model_commit': commit,
     'seed': seed, 'c4_seed': seed+2, 'job_id': os.environ['SLURM_JOB_ID'],
     'transformers': transformers.__version__, 'torch': torch.__version__,
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'dependency_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['run_domain_sensitivity.py', 'domain_stress_data.py', 'quantize/quantizer.py']},
     'dataset_revisions': revisions, 'wiki_hashes': wh, 'c4_data': cm, 'policies': {}, 'final': {}}
def save():
    task.atomic_json(out/'report.json', r)
save()
baselines = {d: task.evaluate(model, batches, 'baseline/'+d) for d, batches in
             [('wiki_fit', wf), ('wiki_val', wv), ('c4_fit', cf), ('c4_val', cv)]}
scores = {}
for label, batches, expected in [('wa', wf[:32], baselines['wiki_fit'][:32]),
                                 ('wb', wf[32:], baselines['wiki_fit'][32:]),
                                 ('ca', cf[:32], baselines['c4_fit'][:32]),
                                 ('cb', cf[32:], baselines['c4_fit'][32:])]:
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        m, s, losses = checked_scores(model, modules, base, alt, batches)
    assert losses == expected
    scores[label] = (m, s)
    torch.save({'means': m, 'ses': s, 'tokens_sha256': domain.digest(batches)}, out/(label+'_scores.pt'))
scores['wiki'] = domain.pool(scores['wa'], scores['wb'])
scores['c4'] = domain.pool(scores['ca'], scores['cb'])
scores['mixed'] = domain.pool(scores['wa'], scores['ca'])
scores['consensus'] = domain.consensus_scores(scores['wa'], scores['ca'])
policies = {}
for label in ['wiki', 'c4', 'mixed', 'consensus']:
    task.install(modules, base, alt)
    fit_batches = wf if label == 'wiki' else cf if label == 'c4' else wf[:32]+cf[:32]
    fit_base = baselines['wiki_fit'] if label == 'wiki' else baselines['c4_fit'] if label == 'c4' else baselines['wiki_fit'][:32]+baselines['c4_fit'][:32]
    if label == 'consensus':
        masks, history = domain.consensus_fit(model, modules, base, alt, scores[label],
            {'wiki': wf[:32], 'c4': cf[:32]}, {'wiki': baselines['wiki_fit'][:32], 'c4': baselines['c4_fit'][:32]},
            {'wiki': scores['wa'], 'c4': scores['ca']})
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
    r['policies'][label] = {**task.selected_summary(masks, scores[label][0]), 'history': history,
                            'validation': checks, 'accepted': accepted}
    policies[label] = masks
    for suffix, chosen in [('candidate', masks), ('export', masks if accepted else {k: torch.zeros_like(v) for k, v in masks.items()})]:
        path = out/(label+'_'+suffix+'.json')
        task.export_type_map(path, args.model, chosen, domain.digest(fit_batches), commit, label, weight_baseline='nvfp4_4over6')
        spec = json.loads(path.read_text())
        spec.update(scope='all_linear_except_head', activation_dtype='nvfp4_4over6',
                    native_transformers=transformers.__version__)
        task.atomic_json(path, spec)
    save()
torch.save(policies, out/'policies.pt')
r['baseline_calibration'] = baselines
save()
# All proposals frozen before held-out evaluation.
data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=revisions['Salesforce/wikitext'], split='validation')
ids = tok('\n\n'.join(data['text']), return_tensors='pt').input_ids
wiki_eval = list(ids[:, :ids.shape[1]//2048*2048].split(2048, 1))
c4_eval, eval_meta = domain.c4_windows(tok, revisions['allenai/c4'], 'validation', 1, 256, 20260916)
assert not ({x['document_sha256'] for x in cm['documents']} & {x['document_sha256'] for x in eval_meta['documents']})
evals, stress_meta = load_stress(tok)
evals.update(wikitext=wiki_eval, c4=c4_eval)
r.update(eval_c4_data=eval_meta, eval_wiki_sha256=domain.digest(wiki_eval), stress_data=stress_meta)
for label, masks in {'baseline': None, **policies}.items():
    task.install(modules, base, alt, masks)
    r['final'][label] = {}
    for d, batches in evals.items():
        values = task.evaluate(model, batches, 'heldout/'+label+'/'+d)
        lengths = [b.numel()-1 for b in batches]
        entry = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                 'token_weighted_ppl': math.exp(sum(v*n for v, n in zip(values, lengths))/sum(lengths))}
        if label != 'baseline':
            entry['vs_baseline'] = task.paired(values, r['final']['baseline'][d]['nll'])
        r['final'][label][d] = entry
        save()
r['complete'] = True
save()
wrapper.close()
