"""Evaluate frozen cross-domain maps on math/code text; Slurm only."""
import run_domain_sensitivity as domain
import hashlib
import json
import math
import os
import random
from pathlib import Path
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from probe_qwen38 import NativeActivationQuantization, native_loss, text_modules
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
import analyze_task_sensitivity as task


root = Path('results/task_sensitivity_domains')
out = root/'stress.json'
assert not out.exists()
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = False
probe = json.loads(Path('results/task_sensitivity_qwen38_probe/report.json').read_text())
tok = AutoTokenizer.from_pretrained(probe['model'], revision=probe['model_commit'])
data, metadata = {}, {}
for name, repo, config in [('math', 'openai/gsm8k', 'main'), ('code', 'google-research-datasets/mbpp', 'full')]:
    revision = HfApi().dataset_info(repo).sha
    ds = load_dataset(repo, config, revision=revision, split='test')
    indices = random.Random(20260917).sample(range(len(ds)), 128)
    batches = []
    for i in indices:
        row = ds[i]
        text = ('Question: '+row['question']+'\nAnswer: '+row['answer'] if name == 'math'
                else 'Problem: '+row['text']+'\nCode:\n'+row['code'])
        ids = tok(text, return_tensors='pt').input_ids[:, :2048]
        assert ids.numel() >= 2
        batches.append(ids)
    data[name] = batches
    metadata[name] = {'repo': repo, 'config': config, 'revision': revision,
                      'row_indices': indices, 'lengths': [b.numel() for b in batches],
                      'token_sha256': [hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in batches]}
model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
    probe['model'], revision=probe['model_commit'], dtype=torch.bfloat16,
    device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}, output_loading_info=True)
assert not loading['missing_keys']
model.eval()
modules = text_modules(model)
base, alt = {}, {}
with torch.no_grad():
    for n, m in modules.items():
        base[n] = quant_nvfp4_4over6(m.weight, 4, 16).cpu()
        alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always').cpu()
        m.weight.copy_(base[n])
wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
task.loss = native_loss
r = {'complete': False, 'job_id': os.environ['SLURM_JOB_ID'], 'model_commit': probe['model_commit'],
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'data': metadata, 'results': {}, 'policy_sha256': {}}
policies = {'baseline': None}
for seed in [20260912, 20260913]:
    source = root/f'seed{seed}'
    assert json.loads((source/'report.json').read_text())['complete']
    for name, masks in torch.load(source/'policies.pt', weights_only=True).items():
        label = f'{seed}/{name}'
        policies[label] = masks
        r['policy_sha256'][label] = hashlib.sha256((source/(name+'_candidate.json')).read_bytes()).hexdigest()
for label, masks in policies.items():
    task.install(modules, base, alt, masks)
    r['results'][label] = {}
    for name, batches in data.items():
        values = task.evaluate(model, batches, label+'/'+name)
        lengths = [b.numel()-1 for b in batches]
        entry = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                 'token_weighted_ppl': math.exp(sum(v*n for v, n in zip(values, lengths))/sum(lengths))}
        if label != 'baseline':
            entry['vs_baseline'] = task.paired(values, r['results']['baseline'][name]['nll'])
        r['results'][label][name] = entry
        task.atomic_json(out, r)
r['complete'] = True
task.atomic_json(out, r)
wrapper.close()
