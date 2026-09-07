"""Teacher-KL tile selection followed by independent actual-NLL validation."""
import run_domain_sensitivity as domain
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from probe_qwen38 import NativeActivationQuantization, native_loss
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from domain_score_backend import checked_scores
from domain_stress_data import load_stress
from domain_teacher_loss import teacher_kl
import analyze_task_sensitivity as task

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['qwen3-4b', 'llama-3.1-8b-local'])
args = ap.parse_args()
root = Path('results/task_sensitivity_domains')
parent_path = root/'panel'/args.model/'report.json'
parent = json.loads(parent_path.read_text())
assert parent['complete'] and parent['transformers'] == transformers.__version__
out = root/'teacher'/args.model
out.mkdir(parents=True, exist_ok=True)
assert not (out/'report.json').exists()
torch.manual_seed(parent['seed'])
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = False
source, commit = parent['model_source'], parent['model_commit']
tok = AutoTokenizer.from_pretrained(source, revision=commit)
model, loading = AutoModelForCausalLM.from_pretrained(source, revision=commit, dtype=torch.bfloat16,
                                                    device_map='cuda:0', output_loading_info=True)
assert not loading['missing_keys']
model.eval()
wf, wv, _, wh = task.data_splits(tok, 2048, 64, 16, parent['seed'], dataset_name='Salesforce/wikitext')
assert wh == parent['wiki_hashes']
cb, cm = domain.c4_windows(tok, parent['dataset_revisions']['allenai/c4'], 'train', 0, 88, parent['c4_seed'])
assert cm == parent['c4_data']
fits, vals = {'wiki': wf[:32], 'c4': cb[:32]}, {'wiki': wv[:8], 'c4': cb[64:72]}
r = {'complete': False, 'model': args.model, 'model_source': source, 'model_commit': commit,
     'seed': parent['seed'], 'job_id': os.environ['SLURM_JOB_ID'], 'parent_report': str(parent_path),
     'transformers': transformers.__version__, 'torch': torch.__version__,
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'dependency_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['run_domain_sensitivity.py', 'domain_teacher_loss.py', 'domain_score_backend.py',
                            'domain_stress_data.py', 'analyze_task_sensitivity.py', 'quantize/quantizer.py']},
     'fit_objective': 'KL(full_precision_teacher || quantized), nats per token',
     'validation_objective': 'actual next-token NLL', 'wiki_hashes': wh,
     'c4_data': cm, 'policies': {}, 'final': {}, 'teacher_nll': {}}
def save():
    task.atomic_json(out/'report.json', r)
save()
teacher = {}
with torch.no_grad():
    for split, domains in [('fit', fits), ('val', vals)]:
        r['teacher_nll'][split] = {}
        for d, batches in domains.items():
            losses = []
            for i, ids in enumerate(batches):
                logits = model(ids.to(model.device), use_cache=False).logits
                stored = logits.detach()
                if stored.dtype != torch.bfloat16:
                    compact = stored.to(torch.bfloat16)
                    if torch.equal(compact.to(stored.dtype), stored):
                        stored = compact  # Lossless storage only; never round teacher targets.
                teacher[id(ids)] = stored.cpu()
                losses.append(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              ids[:, 1:].reshape(-1).to(logits.device)).item())
                if (i+1) % 8 == 0:
                    print(f'TEACHER {split}/{d} {i+1}/{len(batches)}', flush=True)
            r['teacher_nll'][split][d] = losses
r['teacher_logit_bytes'] = sum(t.numel()*t.element_size() for t in teacher.values())
del logits
save()
modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n}
base, alt = {}, {}
with torch.no_grad():
    for n, m in modules.items():
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16).cpu()
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always').cpu()
        m.weight.copy_(base[n])
wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
task.activation_ste = wrapper.backward
def kl_loss(model, ids, use_cache=False):
    logits = model(ids.to(model.device), use_cache=use_cache).logits
    return teacher_kl(logits, teacher[id(ids)])
task.loss = native_loss
baseline_nll = {}
for split, domains in [('fit', fits), ('val', vals)]:
    baseline_nll[split] = {}
    for d, batches in domains.items():
        v = task.evaluate(model, batches, 'baseline-nll/'+split+'/'+d)
        assert v == parent['baseline_calibration'][d+'_'+split][:len(batches)]
        baseline_nll[split][d] = v
task.loss = kl_loss
baseline_kl = {split: {d: task.evaluate(model, batches, 'baseline-kl/'+split+'/'+d)
                       for d, batches in domains.items()} for split, domains in [('fit', fits), ('val', vals)]}
r['baseline_nll'], r['baseline_kl'] = baseline_nll, baseline_kl
save()
scores = {}
for d, batches in fits.items():
    m, s, losses = checked_scores(model, modules, base, alt, batches)
    assert losses == baseline_kl['fit'][d]
    scores[d] = (m, s)
    torch.save({'means': m, 'ses': s, 'losses': losses, 'tokens_sha256': domain.digest(batches)}, out/(d+'_scores.pt'))
scores['teacher_mixed'] = domain.pool(scores['wiki'], scores['c4'])
scores['teacher_consensus'] = domain.consensus_scores(scores['wiki'], scores['c4'])
policies = {}
for rule in ['teacher_mixed', 'teacher_consensus']:
    task.install(modules, base, alt)
    task.loss = kl_loss
    if rule == 'teacher_mixed':
        masks, history = task.calibrated_trust_masks(model, modules, base, alt, *scores[rule],
            fits['wiki']+fits['c4'], baseline_kl['fit']['wiki']+baseline_kl['fit']['c4'])
    else:
        masks, history = domain.consensus_fit(model, modules, base, alt, scores[rule], fits,
                                              baseline_kl['fit'], {'wiki': scores['wiki'], 'c4': scores['c4']})
    task.install(modules, base, alt, masks)
    kl_checks = {d: task.paired(task.evaluate(model, batches, rule+'/val-kl/'+d), baseline_kl['val'][d])
                 for d, batches in vals.items()}
    task.loss = native_loss
    actual = {d: task.evaluate(model, batches, rule+'/val-nll/'+d) for d, batches in vals.items()}
    checks = {d: task.paired(actual[d], baseline_nll['val'][d]) for d in vals}
    if rule == 'teacher_mixed':
        gate = task.paired(actual['wiki']+actual['c4'], baseline_nll['val']['wiki']+baseline_nll['val']['c4'])
        accepted = gate['mean']+2*gate['se'] < 0
    else:
        accepted = all(v['mean']+2*v['se'] < 0 for v in checks.values())
    r['policies'][rule] = {**task.selected_summary(masks, scores[rule][0]), 'history': history,
                           'validation': checks, 'validation_kl': kl_checks, 'accepted': accepted}
    policies[rule] = masks
    for suffix, chosen in [('candidate', masks), ('export', masks if accepted else {k: torch.zeros_like(v) for k, v in masks.items()})]:
        path = out/(rule+'_'+suffix+'.json')
        task.export_type_map(path, args.model, chosen, domain.digest(fits['wiki']+fits['c4']), commit, rule,
                             weight_baseline='nvfp4_4over6')
        spec = json.loads(path.read_text())
        spec.update(scope='all_linear_except_head', activation_dtype='nvfp4_4over6',
                    native_transformers=transformers.__version__, fit_objective='teacher_kl')
        task.atomic_json(path, spec)
    save()
torch.save(policies, out/'policies.pt')
# Drop cached teacher outputs before ordinary held-out evaluations.
teacher.clear()
task.loss = native_loss
save()
data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                    revision=parent['dataset_revisions']['Salesforce/wikitext'], split='validation')
ids = tok('\n\n'.join(data['text']), return_tensors='pt').input_ids
wiki_eval = list(ids[:, :ids.shape[1]//2048*2048].split(2048, 1))
assert domain.digest(wiki_eval) == parent['eval_wiki_sha256']
c4_eval, eval_meta = domain.c4_windows(tok, parent['dataset_revisions']['allenai/c4'], 'validation', 1, 256, 20260916)
assert eval_meta == parent['eval_c4_data']
evals, stress_meta = load_stress(tok, reference=parent['stress_data'])
assert stress_meta == parent['stress_data']
evals.update(wikitext=wiki_eval, c4=c4_eval)
r.update(eval_c4_data=eval_meta, eval_wiki_sha256=domain.digest(wiki_eval), stress_data=stress_meta)
for rule, masks in {'baseline': None, **policies}.items():
    task.install(modules, base, alt, masks)
    r['final'][rule] = {}
    for d, batches in evals.items():
        values = task.evaluate(model, batches, rule+'/'+d)
        lengths = [b.numel()-1 for b in batches]
        entry = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                 'token_weighted_ppl': math.exp(sum(v*n for v, n in zip(values, lengths))/sum(lengths))}
        if rule == 'baseline':
            assert values == parent['final']['baseline'][d]['nll']
        else:
            entry['vs_baseline'] = task.paired(values, r['final']['baseline'][d]['nll'])
            counterpart = rule.removeprefix('teacher_')
            entry['vs_ce_counterpart'] = task.paired(values, parent['final'][counterpart][d]['nll'])
        r['final'][rule][d] = entry
        save()
r['complete'] = True
save()
wrapper.close()
print('DONE', out, flush=True)
