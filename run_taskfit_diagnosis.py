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
from run_fine_row_research import fresh_data, verified_model
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


def summarize(metrics, records):
    comparisons = [('quant_both', 'quant_raw'), ('quant_rows', 'quant_raw')]
    comparisons += [(f'bf16_{sign}_{family}', 'bf16')
                    for sign in ('plus', 'minus') for family in ('both', 'rows')]
    groups = {
        'fit': [i for i, row in enumerate(records) if row['split'] == 'fit'],
        'election': [i for i, row in enumerate(records) if row['split'] == 'election'],
        'fresh_in_domain': [i for i, row in enumerate(records) if row['split'] == 'fresh' and row['source'] != 'general'],
        'fresh_math': [i for i, row in enumerate(records) if row['split'] == 'fresh' and row['source'] == 'math'],
        'fresh_code': [i for i, row in enumerate(records) if row['split'] == 'fresh' and row['source'] == 'code'],
        'fresh_general': [i for i, row in enumerate(records) if row['split'] == 'fresh' and row['source'] == 'general'],
    }
    return {f'{candidate} - {reference}': {
        group: {objective: paired_bounds([metrics[candidate][objective][i] for i in indices],
                                        [metrics[reference][objective][i] for i in indices], k=2.)
                for objective in ('ce', 'kl')} for group, indices in groups.items()}
        for candidate, reference in comparisons}


def get_data(tokenizer, prior, plan, out):
    excluded, old_tokens = set(), set()
    for path, digest in plan['exclusion_sources'].items():
        assert digest_file(path) == digest, path
        value = json.loads(Path(path).read_text())
        excluded.update(collect_hashes(value, 'document_sha256'))
        old_tokens.update(collect_hashes(value, 'token_sha256') - {None})
    batches, metadata = math_code_data(tokenizer, prior['fit'])
    assert metadata == prior['fit']
    original = [dict(ids=ids, source=source, document_sha256=meta['document_sha256'],
                     token_sha256=sha(ids), offset=meta['offset'])
                for source, values in batches.items()
                for ids, meta in zip(values, prior['fit'][source]['documents'])]
    manifest = dict(sequence_ids=[r['token_sha256'] for r in original],
                    sequence_sources=[r['source'] for r in original])
    fit, election = split_sequences(manifest, .5, 0)
    records = []
    for label, indices in [('fit', fit), ('election', election)]:
        for source in ('math', 'code'):
            chosen = [i for i in indices if original[i]['source'] == source][:16]
            assert len(chosen) == 16
            records.extend(dict(original[i], split=label) for i in chosen)
    fresh = fresh_data(tokenizer, prior, excluded)
    assert not {r['document_sha256'] for r in fresh} & excluded
    assert not {r['token_sha256'] for r in fresh} & old_tokens
    records.extend(dict(r, split='fresh') for r in fresh)
    excluded.update(r['document_sha256'] for r in fresh)
    general = plan['general_text']
    stream = load_dataset('allenai/c4', revision=general['revision'],
        data_files={'validation': general['path']}, split='validation', streaming=True)
    rng = random.Random(general['seed'])
    count = 0
    for doc in stream:
        digest = hashlib.sha256(doc['text'].encode()).hexdigest()
        if digest in excluded:
            continue
        ids = tokenizer(doc['text'], return_tensors='pt').input_ids
        if ids.shape[1] < 512:
            continue
        offset = rng.randrange(ids.shape[1] - 512 + 1)
        ids = ids[:, offset:offset + 512].clone()
        token_hash = sha(ids)
        if token_hash in old_tokens:
            continue
        records.append(dict(ids=ids, source='general', split='fresh', document_sha256=digest,
                            token_sha256=token_hash, offset=offset))
        excluded.add(digest)
        count += 1
        if count == 32:
            break
    assert count == 32 and len(records) == 160
    assert len({r['document_sha256'] for r in records}) == len(records)
    torch.save(records, out / 'records.pt')
    (out / 'fresh_manifest.json').write_text(json.dumps(dict(records=[
        {k: v for k, v in r.items() if k != 'ids'} for r in records if r['split'] == 'fresh'],
        windows=96, window_tokens=512, no_selection=True), indent=2) + '\n')
    return records


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    plan = json.loads((out / 'plan.json').read_text())
    assert not (out / 'report.json').exists()
    for path, digest in plan['source_sha256'].items():
        assert digest_file(path) == digest, path
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    root = Path(plan['root'])
    prior = json.loads((root / 'calibration/report.json').read_text())
    assert prior['status'] == 'complete'
    layouts = {}
    for label, directory in plan['layouts'].items():
        panels, hashes = load_layouts([f'{label}={directory}'], prior, allow_rowbands=True)
        assert all(plan['layout_sha256'][path] == digest for path, digest in hashes.items())
        layouts[label] = panels[label]
    names = list(layouts['both'])
    assert set(names) == set(layouts['rows']) == set(prior['reorder_modules'])
    assert digest_file(plan['raw_masks']) == plan['raw_masks_sha256']
    raw_masks = torch.load(plan['raw_masks'], map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    records = get_data(tokenizer, prior, plan, out)
    policies = ['bf16', 'bf16_plus_both', 'bf16_minus_both', 'bf16_plus_rows', 'bf16_minus_rows',
                'quant_raw', 'quant_both', 'quant_rows']
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], plan_sha256=digest_file(out / 'plan.json'),
        records_sha256=digest_file(out / 'records.pt'), diagnostic_only=True, no_optimization=True,
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
        for family in ('both', 'rows'):
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
    backbone = model.get_submodule('model.language_model')
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
        vocab = model.config.text_config.vocab_size
        assert shutil.disk_usage(temporary).free > len(records) * 512 * vocab * 2 + 5 * 1024**3
        for i, row in enumerate(records):
            ids = row['ids'].cuda()
            install('bf16')
            direct, capture = capture_forward(ids)
            teacher = direct[:, :-1].float().reshape(-1, direct.shape[-1]).log_softmax(-1)
            torch.save(direct.cpu(), temporary / f'{i}.pt')
            for policy in policies[:5]:
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
            for policy in policies[5:]:
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
    report['status'] = 'complete'
    save(out, report)
    print('TASKFIT COMPLETE ' + json.dumps(report['paired']), flush=True)


if __name__ == '__main__':
    main()
