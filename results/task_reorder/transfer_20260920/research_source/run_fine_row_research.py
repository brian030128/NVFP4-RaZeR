"""Deployment-matched 1x64 row search, fresh validation, and local controls.

All scoring, search, tokenization and evaluation must run on H200 workers.
Fine score tables exist only in job-local scratch; frozen layouts persist.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import SearchConfig, original_order_weight_reference
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import math_code_data
from run_reorder_replay import paired_bounds
from run_rowband_reorder import SCOPE
from run_task_reorder import run as search_run
from run_task_reorder_eval import coarse_mask, load_layouts, mix_coarse


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def scope_names(prior):
    import re
    return [name for name in prior['matrices'] if re.search(SCOPE, name)]


def sequence_metadata(prior):
    return ([x for v in prior['fit'].values() for x in v['token_sha256']],
            [s for s, v in prior['fit'].items() for _ in v['token_sha256']])


def identity_layout(prior, name, atom_rows=8):
    n, k = prior['matrices'][name]['shape']
    ids, sources = sequence_metadata(prior)
    return dict(schema='mixfp4_task_reorder_v1', name=name,
        config=asdict(SearchConfig(atom_rows=atom_rows, atom_cols=64, axes='rows')),
        weight_shape=[n, k], padded_shape=[math.ceil(n / 256) * 256, k],
        row_perm=torch.arange(n), col_perm=torch.arange(k),
        inverse_row_perm=torch.arange(n), inverse_col_perm=torch.arange(k),
        row_atom_perm=torch.arange(n // atom_rows), col_atom_perm=torch.arange(k // 64),
        fit_sequence_ids=[], election_sequence_ids=ids,
        provenance=dict(schema='mixfp4_reorder_scores_v1', status='complete', name=name,
            weight_shape=[n, k], atom_shape=[atom_rows, 64], sequence_ids=ids, sequence_sources=sources,
            source=prior['source'], revision=prior['revision'],
            weight_sha256=prior['matrices'][name]['source_sha256']))


def save_layout(path, layout):
    path.mkdir(parents=True, exist_ok=False)
    torch.save(layout, path / 'layout.pt')
    write_json(path / 'report.json', dict(status='complete', module=layout['name'],
        elected_tiles=int(layout['mask'].sum()), job_id=os.environ['SLURM_JOB_ID'],
        mask_selection=layout.get('mask_selection')))


def full_fine_mask(ce, kl, shape):
    n, width = shape
    bounds = [v.double().mean(0) + 3 * v.double().std(0) / math.sqrt(v.shape[0]) for v in (ce, kl)]
    assert all(torch.isfinite(v).all() for v in bounds)
    return (torch.maximum(*bounds) < 0).reshape(n // 8, width // 64)


def fresh_data(tok, prior, excluded_extra=()):
    excluded = {d['document_sha256'] for v in prior['fit'].values() for d in v['documents']} | set(excluded_extra)
    known_tokens = set(sequence_metadata(prior)[0])
    generator = torch.Generator().manual_seed(20260919)
    records = []
    for source in ('math', 'code'):
        meta = prior['fit'][source]
        stream = load_dataset(meta['repo'], revision=meta['revision'],
            data_files={'train': meta['path']}, split='train', streaming=True)
        count = 0
        for row in stream:
            content = row['text' if source == 'math' else 'content']
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in excluded:
                continue
            ids = tok(content, return_tensors='pt').input_ids
            if ids.shape[1] < 512:
                continue
            offset = int(torch.randint(ids.shape[1] - 512 + 1, (1,), generator=generator))
            ids = ids[:, offset:offset + 512].clone()
            token_hash = sha(ids)
            if token_hash in known_tokens:
                continue
            records.append(dict(source=source, document_sha256=digest, offset=offset,
                                token_sha256=token_hash, ids=ids))
            excluded.add(digest); known_tokens.add(token_hash); count += 1
            if count == 32:
                break
        assert count == 32, source
    return records


def prepare(root):
    prior = read_json(root / 'calibration/report.json')
    assert prior['status'] == 'complete'
    out = root / 'fine_rows_v2'
    out.mkdir(exist_ok=False)
    raw = {}; counts = dict(raw256=0, full8x64=0); sources = {}
    for index, (name, metadata) in enumerate(prior['matrices'].items()):
        path = root / 'calibration/scores' / f'{index:03d}.pt'
        sources[str(path)] = digest_file(path)
        scores = torch.load(path, map_location='cpu', weights_only=True)
        assert scores['name'] == name
        coarse = coarse_mask(scores['ce'], scores['kl'], metadata['shape'])
        fine = full_fine_mask(scores['ce'], scores['kl'], metadata['shape'])
        raw[name] = coarse
        layout = identity_layout(prior, name)
        layout.update(mask=coarse, identity_mask=coarse, identity_8x64_mask=fine,
                      mask_selection=dict(method='full 128-sequence k=3 election; no permutation search'))
        save_layout(out / 'full_controls/layouts' / f'{index:03d}', layout)
        counts['raw256'] += int(coarse.sum()); counts['full8x64'] += int(fine.sum())
        if (index + 1) % 64 == 0:
            print(f'CONTROL MAPS {index + 1}/{len(prior["matrices"])}', flush=True)
    torch.save(raw, out / 'raw256_masks.pt')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    published = read_json(Path('results/kse_paper/job_336969/qwen27b/report.json'))
    excluded = [d['document_sha256'] for d in published['data']['c4_paper']['documents']]
    fresh = fresh_data(tok, prior, excluded)
    torch.save(fresh, out / 'fresh_math_code.pt')
    write_json(out / 'fresh_manifest.json', dict(seed=20260919, windows=64, window_tokens=512,
        records=[{k: v for k, v in row.items() if k != 'ids'} for row in fresh],
        repositories={s: {k: v[k] for k in ('repo', 'revision', 'path')} for s, v in prior['fit'].items()},
        excluded_previous_fit=True, excluded_published_c4_documents=True))
    write_json(out / 'protocol.json', dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
        scope=SCOPE, scope_modules=scope_names(prior), candidates=['rows_last1', 'rows_last4', 'rows_last8', 'old_both_last1'],
        calibration_report_sha256=digest_file(root / 'calibration/report.json'),
        raw256_sha256=digest_file(out / 'raw256_masks.pt'), fresh_sha256=digest_file(out / 'fresh_math_code.pt'),
        counts=counts, historical_score_sha256=sources, score_atoms=[1, 64],
        scoring_activation='FourOverSix tensor-wide with straight-through input gradients', attention='sdpa',
        scoring_background='local raw256 outside layers 56-63 MLP; FourOverSix inside scope',
        election='64 fit / 64 separate election sequences, k=3',
        fresh_selection='64 fresh math/code sequences; improve paired CE and teacher KL mean+1SE vs BOTH raw256 and matched identity; minimum CE among eligible',
        search=asdict(SearchConfig(atom_rows=1, atom_cols=64, axes='rows', starts=8, rounds=12, scratch_mb=128))))
    print('PREPARED ' + json.dumps(counts), flush=True)


def verify_prepared(root):
    """Independently validate persisted outputs after a producer teardown error."""
    prior = read_json(root / 'calibration/report.json'); out = root / 'fine_rows_v2'
    protocol = read_json(out / 'protocol.json')
    assert protocol['status'] == 'complete'
    assert digest_file(root / 'calibration/report.json') == protocol['calibration_report_sha256']
    assert digest_file(out / 'raw256_masks.pt') == protocol['raw256_sha256']
    assert digest_file(out / 'fresh_math_code.pt') == protocol['fresh_sha256']
    raw = torch.load(out / 'raw256_masks.pt', map_location='cpu', weights_only=True)
    panels, hashes = load_layouts([f'fullmap={out / "full_controls/layouts"}'], prior, prior['matrices'], True)
    counts = dict(raw256=0, full8x64=0)
    for name, layout in panels['fullmap'].items():
        n, k = prior['matrices'][name]['shape']
        assert layout['mask'].dtype == layout['identity_8x64_mask'].dtype == torch.bool
        assert layout['mask'].shape == (math.ceil(n / 256), k // 64)
        assert layout['identity_8x64_mask'].shape == (n // 8, k // 64)
        assert torch.equal(raw[name], layout['mask']) and torch.equal(raw[name], layout['identity_mask'])
        counts['raw256'] += int(raw[name].sum()); counts['full8x64'] += int(layout['identity_8x64_mask'].sum())
    assert counts == protocol['counts']
    records = torch.load(out / 'fresh_math_code.pt', map_location='cpu', weights_only=True)
    manifest = read_json(out / 'fresh_manifest.json')
    assert [{k: v for k, v in row.items() if k != 'ids'} for row in records] == manifest['records']
    assert len(records) == len({row['document_sha256'] for row in records}) == 64
    for source in ('math', 'code'):
        assert sum(row['source'] == source for row in records) == 32
    assert all(row['ids'].shape == (1, 512) and sha(row['ids']) == row['token_sha256'] for row in records)
    assert not {row['token_sha256'] for row in records} & set(sequence_metadata(prior)[0])
    excluded = {d['document_sha256'] for source in prior['fit'].values() for d in source['documents']}
    published = read_json(Path('results/kse_paper/job_336969/qwen27b/report.json'))
    excluded |= {d['document_sha256'] for d in published['data']['c4_paper']['documents']}
    assert not {row['document_sha256'] for row in records} & excluded
    write_json(out / 'prepared_verification.json', dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
        producer_job=protocol['job_id'], producer_teardown_failed=True, protocol_sha256=digest_file(out / 'protocol.json'),
        control_layout_sha256=hashes, counts=counts, fresh_documents_verified=64,
        verification='Read back every control layout and all fresh tensors; shapes, hashes, source identity, counts and exclusions checked'))
    print('PREPARED VERIFIED ' + json.dumps(counts), flush=True)


def verified_model(prior):
    from transformers import Qwen3_5ForConditionalGeneration
    assert transformers.__version__ == prior['transformers_version']
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(prior['source'], revision=prior['revision'],
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda', output_loading_info=True)
    assert not info['missing_keys'] and not info.get('mismatched_keys') and not info.get('error_msgs')
    model.eval().requires_grad_(False)
    modules = {name: model.get_submodule(name) for name in prior['matrices']}
    for name, module in modules.items():
        assert sha(module.weight) == prior['matrices'][name]['source_sha256'], name
    return model, modules


def row_direction_scores(x, dy, direction):
    gradient = dy.detach().reshape(-1, dy.shape[-1]).float().T @ x.detach().reshape(-1, x.shape[-1]).float()
    n, k = direction.shape
    return (gradient * direction).reshape(n, k // 64, 64).sum(-1)


def activation_hook(module, inputs):
    x = inputs[0]
    q = quant_nvfp4_4over6(x.detach(), 4, 16)
    return (q + (x - x.detach()), *inputs[1:])


def arm_gradient(output, callback):
    # A frozen prefix needs no graph. Create leaves at the first selected
    # projections; downstream selected projections keep their existing graph.
    if not output.requires_grad:
        output.requires_grad_(True)
    output.register_hook(callback)
    return output


def layer(root, number, smoke=False):
    assert 56 <= number <= 63
    prior = read_json(root / 'calibration/report.json'); experiment = root / 'fine_rows_v2'
    protocol = read_json(experiment / 'protocol.json')
    assert digest_file(experiment / 'raw256_masks.pt') == protocol['raw256_sha256']
    assert digest_file(root / 'calibration/report.json') == protocol['calibration_report_sha256']
    raw = torch.load(experiment / 'raw256_masks.pt', map_location='cpu', weights_only=True)
    selected = [name for name in scope_names(prior) if f'.layers.{number}.' in name]
    assert len(selected) == 3
    out = experiment / ('smoke' if smoke else 'layers') / str(number)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], layer=number,
        source_sha256={p: digest_file(p) for p in ('run_fine_row_research.py', 'quantize/quantizer.py', 'quantize/task_reorder.py')},
        protocol_sha256=digest_file(experiment / 'protocol.json'), scored_sequences=0, losses=[])
    save(out, report)
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    batches, metadata = math_code_data(tok, prior['fit'])
    assert metadata == prior['fit']
    sequences = [ids for source in batches.values() for ids in source]
    ids_hashes, sources = sequence_metadata(prior)
    if smoke:
        sequences = sequences[:2]
    model, modules = verified_model(prior)
    local = Path(tempfile.mkdtemp(prefix=f'fine_layer{number}_', dir=os.environ['TMPDIR']))
    expected = len(sequences) * (511 * model.config.text_config.vocab_size * 4 + sum(
        math.prod(prior['matrices'][name]['shape']) // 64 * 8 for name in selected))
    assert shutil.disk_usage(local).free > expected + 2 * 1024**3, 'Insufficient node-local scratch'
    with torch.no_grad():
        for i, ids in enumerate(sequences):
            logits = model(input_ids=ids.cuda(), use_cache=False).logits
            lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            torch.save(lp.cpu(), local / f'teacher_{i:03d}.pt')
            del logits, lp
            if (i + 1) % 16 == 0:
                print(f'TEACHER {i + 1}/{len(sequences)}', flush=True)
    directions = {}
    with torch.no_grad():
        for name, module in modules.items():
            b = quant_nvfp4_4over6(module.weight, 4, 16)
            if name in selected or (name not in protocol['scope_modules'] and raw[name].any()):
                a = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                if name in selected:
                    directions[name] = a.float() - b.float()
                else:
                    b = mix_coarse(b, a, raw[name])
                del a
            module.weight.copy_(b)
        del b
    directories = {name: local / f'{i:03d}' for i, name in enumerate(selected)}
    for path in directories.values():
        path.mkdir()
    phase = 0; sequence = 0; pending = {}; hits = {name: [0, 0] for name in selected}
    def hook(name):
        def forward(module, inputs, output):
            x = inputs[0].detach()
            def backward(dy):
                score = row_direction_scores(x, dy, directions[name]).cpu()
                assert torch.isfinite(score).all(), name
                hits[name][phase] += 1
                if phase == 0:
                    pending[name] = score
                else:
                    torch.save(dict(ce=pending.pop(name), kl=score, sequence_id=ids_hashes[sequence]),
                               directories[name] / f'{sequence:03d}.pt')
            return arm_gradient(output, backward)
        return forward
    handles = [m.register_forward_pre_hook(activation_hook) for m in modules.values()]
    handles += [modules[name].register_forward_hook(hook(name)) for name in selected]
    for sequence, ids in enumerate(sequences):
        ids = ids.cuda()
        logits = model(input_ids=ids, use_cache=False).logits
        lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        teacher_path = local / f'teacher_{sequence:03d}.pt'
        teacher = torch.load(teacher_path, map_location='cpu', weights_only=True).cuda()
        ce = F.nll_loss(lp, ids[:, 1:].reshape(-1))
        kl = F.kl_div(lp, teacher, reduction='batchmean', log_target=True)
        phase = 0; ce.backward(retain_graph=True)
        phase = 1; kl.backward()
        report['losses'].append(dict(ce=float(ce.detach()), kl=float(kl.detach())))
        assert all(math.isfinite(v) for v in report['losses'][-1].values())
        report['scored_sequences'] = sequence + 1
        del logits, lp, teacher, ce, kl
        teacher_path.unlink()
        save(out, report)
        if smoke or (sequence + 1) % 8 == 0:
            print(f'SCORED layer={number} {sequence + 1}/{len(sequences)}', flush=True)
    assert all(value == [len(sequences)] * 2 for value in hits.values())
    for handle in handles:
        handle.remove()
    del model, modules, directions
    torch.cuda.empty_cache()
    if not smoke:
        for index, name in enumerate(selected):
            manifest = identity_layout(prior, name, atom_rows=1)['provenance']
            manifest.update(activation_convention=protocol['scoring_activation'],
                background=protocol['scoring_background'], source_sha256=report['source_sha256'],
                protocol_sha256=report['protocol_sha256'])
            write_json(directories[name] / 'manifest.json', manifest)
            search_run(directories[name], out / 'search' / f'{index:03d}', SearchConfig(**protocol['search']))
            write_json(out / 'score_manifests' / f'{index:03d}.json', manifest)
    shutil.rmtree(local)
    report.update(status='complete', worker_scratch_deleted=True, smoke_only=smoke)
    save(out, report)
    print(f'LAYER COMPLETE {number} smoke={smoke}', flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an H200 Slurm worker')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare', 'verify-prepared', 'layer'))
    ap.add_argument('root', type=Path)
    ap.add_argument('--layer', type=int, default=56)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False; torch.manual_seed(0)
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'verify-prepared':
        verify_prepared(args.root)
    else:
        layer(args.root, args.layer, args.smoke)


if __name__ == '__main__':
    main()
