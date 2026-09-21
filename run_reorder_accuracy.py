"""Matched zero-shot task accuracy for frozen Qwen raw and arranged 256x64.

Four shards: full ARC-Challenge, and three disjoint sets of MMLU subjects.
No calibration, map changes, few-shot examples, or chat template.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import lm_eval_compat  # noqa: F401
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_c4_frozen import digest_file
from run_conditional_format import save
from run_fine_row_research import verified_model
from run_task_reorder_eval import load_layouts, mix_coarse


def compact_samples(samples):
    result = {}
    for task, records in samples.items():
        result[task] = []
        for row in records:
            fingerprint = hashlib.sha256(json.dumps(
                {k: row[k] for k in ('doc', 'target', 'arguments') if k in row},
                sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
            result[task].append(dict(doc_id=row['doc_id'], content_sha256=fingerprint,
                **{k: row[k] for k in ('acc', 'acc_norm', 'doc_hash', 'prompt_hash', 'target_hash') if k in row}))
    return result


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['layout_sha256']}.items():
        assert digest_file(path) == digest, path
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    tasks = ['arc_challenge', plan['mmlu_subjects'][0]] if args.smoke else plan['shards'][str(args.shard)]
    manager = lm_eval.tasks.TaskManager()
    # Prepare datasets before loading the model; retain their exact fingerprints.
    prepared_tasks = manager.load_task_or_group(tasks)
    assert set(prepared_tasks) == set(tasks)
    datasets = {name: dict(fingerprint=task.dataset['test']._fingerprint,
                           test_documents=len(task.dataset['test']))
                for name, task in prepared_tasks.items()}
    prior = json.loads((args.root / 'calibration/report.json').read_text())
    panels, hashes = load_layouts([f'arranged={plan["layouts"]}'], prior, allow_rowbands=True)
    layouts = panels['arranged']
    assert hashes == plan['layout_sha256']
    assert digest_file(plan['raw_masks']) == plan['raw_masks_sha256']
    raw_masks = torch.load(plan['raw_masks'], map_location='cpu', weights_only=True)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], shard=args.shard,
        smoke=args.smoke, plan_sha256=digest_file(args.plan), tasks=tasks, datasets=datasets,
        zero_shot=True, chat_template=False, native_backend=False, batch_size=plan['batch_size'],
        gpu=torch.cuda.get_device_name(), node=os.environ['SLURMD_NODENAME'],
        lm_eval_version=importlib.metadata.version('lm_eval'), source_weight_hashes_checked=False,
        raw_tiles=sum(int(mask.sum()) for mask in raw_masks.values()),
        arranged_tiles=sum(int(mask.sum()) for name, mask in raw_masks.items() if name not in layouts)
                       + sum(int(layout['mask'].sum()) for layout in layouts.values()), results={})
    assert report['raw_tiles'] == 195 and report['arranged_tiles'] == 212
    save(args.out, report)
    model, modules = verified_model(prior)
    tokenizer = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    weights = {'raw256': {}, 'arranged': {}}
    for name, module in modules.items():
        base = quant_nvfp4_4over6(module.weight, 4, 16)
        if name in layouts or raw_masks[name].any():
            alt = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            raw = mix_coarse(base, alt, raw_masks[name])
            if name in layouts:
                weights['raw256'][name] = raw
                weights['arranged'][name] = original_order_weight_reference(base, alt, layouts[name])
            base = raw
            del alt, raw
        module.weight.copy_(base)
    del base
    report['source_weight_hashes_checked'] = True
    handles = [module.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for module in modules.values()]
    fixture = tokenizer('The capital of France is Paris. A simple arithmetic example: two plus two equals four.',
                        return_tensors='pt').input_ids.cuda()
    baseline_logits = model(input_ids=fixture, use_cache=False).logits
    for policy in ('raw256', 'arranged'):
        for name, weight in weights[policy].items():
            modules[name].weight.copy_(weight)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=plan['batch_size'], max_length=2048)
        print('ACCURACY START', policy, tasks, flush=True)
        full = lm_eval.simple_evaluate(model=lm, tasks=list(prepared_tasks.values()), task_manager=manager,
            num_fewshot=0, batch_size=plan['batch_size'], bootstrap_iters=0, log_samples=True,
            limit=2 if args.smoke else None, random_seed=0, numpy_random_seed=0,
            torch_random_seed=0, fewshot_random_seed=0, apply_chat_template=False)
        samples = compact_samples(full['samples'])
        assert set(samples) == set(tasks)
        for task in tasks:
            assert len(samples[task]) == (2 if args.smoke else datasets[task]['test_documents'])
        (args.out / f'{policy}_samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        report['results'][policy] = full['results']
        report.setdefault('sample_sha256', {})[policy] = digest_file(args.out / f'{policy}_samples.json')
        report.setdefault('task_configs', {})[policy] = full['configs']
        save(args.out, report)
        print('ACCURACY RESULT', policy, json.dumps(full['results']), flush=True)
    for name, weight in weights['raw256'].items():
        modules[name].weight.copy_(weight)
    repeated_logits = model(input_ids=fixture, use_cache=False).logits
    assert torch.equal(baseline_logits, repeated_logits), 'Restoring raw weights changed audit logits'
    report['raw_restore_bitwise_audit'] = True
    left = json.loads((args.out / 'raw256_samples.json').read_text())
    right = json.loads((args.out / 'arranged_samples.json').read_text())
    for task in tasks:
        assert [(r['doc_id'], r['content_sha256']) for r in left[task]] == [
            (r['doc_id'], r['content_sha256']) for r in right[task]], task
    for handle in handles:
        handle.remove()
    for path, digest in {**plan['source_sha256'], **plan['layout_sha256']}.items():
        assert digest_file(path) == digest
    report['status'] = 'complete'
    save(args.out, report)
    print('ACCURACY COMPLETE', args.shard, flush=True)


if __name__ == '__main__':
    main()
