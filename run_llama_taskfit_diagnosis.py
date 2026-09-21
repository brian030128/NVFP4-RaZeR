"""Frozen-map diagnostic: quantization repair versus task-directed adaptation.

No optimization, candidate selection, PPL promotion, or native-kernel claim.
All model/data work must run on an H200 worker.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from llama_diagnosis_common import verified_model
from run_math_code_calibration import math_code_data
from run_reorder_replay import paired_bounds
from run_task_reorder import split_sequences
from run_task_reorder_eval import load_layouts, mix_coarse


def collect_hashes(value, key):
    result = set()
    if isinstance(value, dict):
        if key in value:
            if isinstance(value[key], str):
                result.add(value[key])
            elif isinstance(value[key], list):
                result.update(item for item in value[key] if isinstance(item, str))
        for child in value.values():
            result.update(collect_hashes(child, key))
    elif isinstance(value, list):
        for child in value:
            result.update(collect_hashes(child, key))
    return result


def shifted_weight(original, raw, candidate, sign=1):
    return (original.float() + sign * (candidate.float() - raw.float())).to(original.dtype)


def summarize(metrics, rows):
    from llama_diagnosis_common import groups
    comparisons = [(p, 'quant_raw') for p in metrics if p.startswith('quant_') and p != 'quant_raw']
    comparisons += [('bf16_plus_both', 'bf16'), ('bf16_minus_both', 'bf16')]
    result = {f'{p} - {ref}': {group: {key: paired_bounds(
        [metrics[p][key][i] for i in idx], [metrics[ref][key][i] for i in idx], k=2.)
        for key in ('ce', 'kl')} for group, idx in groups(rows).items()} for p, ref in comparisons}
    result['joint_minus_sum_isolated'] = {group: {key: paired_bounds([
        metrics['quant_both'][key][i] - sum(metrics[p][key][i] for p in ('quant_gate', 'quant_up', 'quant_down'))
        + 2 * metrics['quant_raw'][key][i] for i in idx], [0.] * len(idx), k=2.)
        for key in ('ce', 'kl')} for group, idx in groups(rows).items()}
    return result


@torch.no_grad()
def main():
    from llama_diagnosis_common import ROOT, checked_plan, controls, records as read_records
    plan, prior = checked_plan()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = ROOT / ('fit_smoke' if args.smoke else 'fit')
    out.mkdir(exist_ok=False)
    layout, raw_masks = controls(plan, prior)
    layouts = {'both': layout}
    names = list(layout)
    records = read_records()
    if args.smoke:
        records = [records[i] for i in (0, 1, 112, 113, 176, 177)]
    policies = ['bf16', 'bf16_plus_both', 'bf16_minus_both', 'quant_raw', 'quant_both',
                'quant_gate', 'quant_up', 'quant_down', 'quant_gate_up', 'quant_gate_down', 'quant_up_down']
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], plan_sha256=digest_file(ROOT / 'plan.json'),
        records_sha256=digest_file(ROOT / 'records.pt'), diagnostic_only=True, no_optimization=True,
        no_candidate_promotion=True, native_backend=False,
        records=[{k: v for k, v in r.items() if k != 'ids'} for r in records],
        metrics={p: dict(ce=[], kl=[]) for p in policies}, completed_bf16=0, completed_quant=0,
        audits=dict(bf16_replay=0, quant_raw_replay=0, changed_full=[]), weight_diagnostics={})
    save(out, report)
    model, modules = verified_model(prior)
    original = {name: modules[name].weight.clone() for name in names}
    prepared = {'bf16': original, **{p: {} for p in policies if p != 'bf16'}}
    for name in names:
        w = original[name]
        base = quant_nvfp4_4over6(w, 4, 16)
        alt = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        raw = mix_coarse(base, alt, raw_masks[name])
        prepared['quant_raw'][name] = raw
        report['weight_diagnostics'][name] = dict(raw_mse=float((raw.float() - w.float()).square().mean()))
        for family in ('both',):
            candidate = original_order_weight_reference(base, alt, layouts[family][name])
            prepared[f'quant_{family}'][name] = candidate
            prepared[f'bf16_plus_{family}'][name] = shifted_weight(w, raw, candidate)
            prepared[f'bf16_minus_{family}'][name] = shifted_weight(w, raw, candidate, -1)
            delta = candidate.float() - raw.float()
            report['weight_diagnostics'][name][family] = dict(
                candidate_mse=float((candidate.float() - w.float()).square().mean()),
                delta_squared_norm=float(delta.square().sum()),
                dot_with_quant_error_correction=float((delta * (w.float() - raw.float())).sum()))
        del base, alt, delta
    parts = {'quant_gate': (0,), 'quant_up': (1,), 'quant_down': (2,),
             'quant_gate_up': (0, 1), 'quant_gate_down': (0, 2), 'quant_up_down': (1, 2)}
    assert [n.split('.')[-1] for n in names] == ['gate_proj', 'up_proj', 'down_proj']
    for policy, selected in parts.items():
        prepared[policy] = {n: prepared['quant_both' if j in selected else 'quant_raw'][n]
                            for j, n in enumerate(names)}
    backbone = model.model
    last = backbone.layers[-1]
    head = model.get_output_embeddings()

    def install(policy):
        for name, weight in prepared[policy].items():
            modules[name].weight.copy_(weight)

    def capture_forward(ids):
        capture = {}
        def remember(key):
            return lambda module, inputs: capture.update({key: inputs[0].detach().clone()})
        hs = [last.post_attention_layernorm.register_forward_pre_hook(remember('residual')),
              last.mlp.register_forward_pre_hook(remember('input'))]
        logits = model(input_ids=ids, use_cache=False).logits
        for handle in hs:
            handle.remove()
        return logits, capture

    def replay(capture):
        return head(backbone.norm(capture['residual'] + last.mlp(capture['input'])))

    def losses(logits, ids, teacher):
        lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        result = dict(ce=float(F.nll_loss(lp, ids[:, 1:].reshape(-1))),
                      kl=float(F.kl_div(lp, teacher, reduction='batchmean', log_target=True)))
        assert all(math.isfinite(v) for v in result.values())
        return result

    def append(policy, logits, ids, teacher):
        for key, value in losses(logits, ids, teacher).items():
            report['metrics'][policy][key].append(value)

    with tempfile.TemporaryDirectory(prefix='taskfit_teacher_', dir=os.environ['TMPDIR']) as temporary:
        temporary = Path(temporary)
        vocab = model.config.vocab_size
        assert shutil.disk_usage(temporary).free > len(records) * 512 * vocab * 2 + 5 * 1024**3
        for i, row in enumerate(records):
            ids = row['ids'].cuda()
            install('bf16')
            direct, capture = capture_forward(ids)
            teacher = direct[:, :-1].float().reshape(-1, direct.shape[-1]).log_softmax(-1)
            torch.save(direct.cpu(), temporary / f'{i}.pt')
            for policy in policies[:3]:
                install(policy)
                logits = replay(capture)
                if policy == 'bf16':
                    assert torch.equal(logits, direct), 'BF16 replay mismatch'
                    report['audits']['bf16_replay'] += 1
                elif i in (0, 32, 64, 96, 128):
                    full = model(input_ids=ids, use_cache=False).logits
                    assert torch.equal(logits, full), policy
                    report['audits']['changed_full'].append([policy, i])
                    del full
                append(policy, logits, ids, teacher)
                del logits
            del direct, capture, teacher
            report['completed_bf16'] = i + 1
            if (i + 1) % 8 == 0:
                save(out, report)
                print('TASKFIT BF16', i + 1, '/', len(records), flush=True)
        install('bf16')
        for name, module in modules.items():
            base = quant_nvfp4_4over6(module.weight, 4, 16)
            if raw_masks[name].any():
                alt = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                base = mix_coarse(base, alt, raw_masks[name])
                del alt
            module.weight.copy_(base)
        del base
        handles = [module.register_forward_pre_hook(
            lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:]))
            for module in modules.values()]
        for i, row in enumerate(records):
            ids = row['ids'].cuda()
            path = temporary / f'{i}.pt'
            logits = torch.load(path, map_location='cuda', weights_only=True)
            teacher = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            del logits
            install('quant_raw')
            direct, capture = capture_forward(ids)
            for policy in policies[3:]:
                install(policy)
                logits = replay(capture)
                if policy == 'quant_raw':
                    assert torch.equal(logits, direct), 'Raw quantized replay mismatch'
                    report['audits']['quant_raw_replay'] += 1
                elif i in (0, 32, 64, 96, 128):
                    full = model(input_ids=ids, use_cache=False).logits
                    assert torch.equal(logits, full), policy
                    report['audits']['changed_full'].append([policy, i])
                    del full
                append(policy, logits, ids, teacher)
                del logits
            del direct, capture, teacher
            path.unlink()
            report['completed_quant'] = i + 1
            if (i + 1) % 8 == 0:
                save(out, report)
                print('TASKFIT QUANT', i + 1, '/', len(records), flush=True)
        for handle in handles:
            handle.remove()
    report['paired'] = summarize(report['metrics'], records)
    for path, digest in {**plan['source_sha256'], **plan['layout_sha256']}.items():
        assert digest_file(path) == digest, path
    checked_plan()
    report['smoke'] = args.smoke
    report['status'] = 'complete'
    save(out, report)
    print('TASKFIT COMPLETE ' + json.dumps(report['paired']), flush=True)


if __name__ == '__main__':
    main()
