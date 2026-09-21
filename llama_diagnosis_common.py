"""Frozen diagnostic helpers. Model/data operations run only on Slurm H200s."""
import json
import os
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_conditional_format import sha
from run_c4_frozen import digest_file
from run_taskfit_diagnosis import get_data
from run_task_reorder_eval import load_layouts, raw256_masks
from scripts.ce_confirmation_gate import eligible

ROOT = Path('results/task_reorder/llama_diagnosis_20260920')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def checked_plan():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    plan = json.loads((ROOT / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['layout_sha256'],
                         **plan['exclusion_sources'], **plan['development_sources']}.items():
        assert digest_file(path) == digest, path
    assert eligible(Path(plan['confirmation']))
    prior = json.loads(Path(plan['calibration']).read_text())
    assert digest_file(plan['calibration']) == plan['calibration_sha256']
    assert prior['status'] == 'complete'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    return plan, prior


def verified_model(prior):
    assert transformers.__version__ == prior['transformers_version']
    model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == list(prior['matrices'])
    for name, module in modules.items():
        assert sha(module.weight) == prior['matrices'][name]['source_sha256'], name
    return model, modules


def controls(plan, prior):
    panels, hashes = load_layouts([f'both={plan["layouts"]}'], prior)
    assert hashes == plan['layout_sha256']
    raw = raw256_masks(Path(plan['calibration']).parent, prior)
    assert sum(int(m.sum()) for m in raw.values()) == 187
    layouts = panels['both']
    assert sum(int(m.sum()) for n, m in raw.items() if n not in layouts) + sum(
        int(l['mask'].sum()) for l in layouts.values()) == 147
    return layouts, raw


def records():
    manifest = json.loads((ROOT / 'data.json').read_text())
    assert manifest['status'] == 'complete'
    assert digest_file(ROOT / 'records.pt') == manifest['records_sha256']
    assert digest_file(ROOT / 'plan.json') == manifest['plan_sha256']
    return torch.load(ROOT / 'records.pt', map_location='cpu', weights_only=True)


def groups(rows):
    result = {key: [i for i, row in enumerate(rows) if row['split'] == key]
              for key in ('fit', 'election', 'development')}
    result.update({f'fresh_{domain}': [i for i, row in enumerate(rows)
        if row['split'] == 'fresh' and row['source'] == domain] for domain in ('math', 'code', 'general')})
    result['fresh_in_domain'] = result['fresh_math'] + result['fresh_code']
    return {k: v for k, v in result.items() if v}


def prepare():
    plan, prior = checked_plan()
    assert not (ROOT / 'data.json').exists()
    tokenizer = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    rows = get_data(tokenizer, prior, plan, ROOT)
    development = []
    for path in plan['development_sources']:
        old = torch.load(path, map_location='cpu', weights_only=True)
        for domain in ('math', 'code'):
            selected = [r for r in old if r['source'] == domain][:8]
            assert len(selected) == 8
            development.extend(dict(r, split='development', development_origin=path) for r in selected)
    rows = rows[:64] + development + rows[64:]
    assert len(rows) == 208 and len({r['document_sha256'] for r in rows}) == 208
    torch.save(rows, ROOT / 'records.pt')
    write(ROOT / 'data.json', dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
        plan_sha256=digest_file(ROOT / 'plan.json'), records_sha256=digest_file(ROOT / 'records.pt'),
        groups=groups(rows), records=[{k: v for k, v in row.items() if k != 'ids'} for row in rows]))
    checked_plan()
    print('DATA COMPLETE', {k: len(v) for k, v in groups(rows).items()}, flush=True)


if __name__ == '__main__':
    prepare()
