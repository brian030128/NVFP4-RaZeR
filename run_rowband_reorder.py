"""Reuse historical task-gradient scores for a wider row-only MLP search.

Only complete eight-row atoms move. No column gathering is introduced.
Search and election use the original disjoint 64/64 source-stratified split.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from quantize.task_reorder import SearchConfig
from run_c4_frozen import digest_file
from run_task_reorder import run
from run_task_reorder_eval import load_layouts


SCOPE = r'^model\.language_model\.layers\.(5[6-9]|6[0-3])\.mlp\.(gate|up|down)_proj$'


def entries(prior):
    return [(index, name, metadata) for index, (name, metadata) in enumerate(prior['matrices'].items())
            if re.search(SCOPE, name)]


def search(root, task):
    prior = json.loads((root / 'calibration/report.json').read_text())
    assert prior['status'] == 'complete' and prior['model'] == 'qwen27b'
    selected = entries(prior)
    assert len(selected) == 24 and 0 <= task < len(selected)
    index, name, metadata = selected[task]
    source = root / 'calibration/scores' / f'{index:03d}.pt'
    source_hash = digest_file(source)
    values = torch.load(source, map_location='cpu', weights_only=True)
    assert values['name'] == name
    n, k = metadata['shape']
    sequence_ids = [digest for info in prior['fit'].values() for digest in info['token_sha256']]
    sequence_sources = [domain for domain, info in prior['fit'].items() for _ in info['token_sha256']]
    assert len(sequence_ids) == 128
    manifest = dict(schema='mixfp4_reorder_scores_v1', status='complete', name=name,
        atom_shape=[8, 64], weight_shape=[n, k], weight_sha256=metadata['source_sha256'],
        sequence_ids=sequence_ids, sequence_sources=sequence_sources,
        source=prior['source'], revision=prior['revision'], historical_score_path=str(source),
        historical_score_sha256=source_hash, calibration_report_sha256=digest_file(root / 'calibration/report.json'),
        baseline='canonical FourOverSix', alternative='E0M3 alpha1',
        source_sha256={p: digest_file(p) for p in ('run_rowband_reorder.py', 'run_task_reorder.py', 'quantize/task_reorder.py')})
    config = SearchConfig(atom_rows=8, atom_cols=64, axes='rows', starts=8, rounds=12, scratch_mb=128)
    output = root / 'rowband_last8/search' / f'{task:03d}'
    # Temporary input shards adapt the existing tested runner. Persistent
    # historical scores are never duplicated or modified.
    with tempfile.TemporaryDirectory(prefix='rowband_', dir=os.environ['TMPDIR']) as temporary:
        directory = Path(temporary)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        for objective in ('ce', 'kl'):
            assert values[objective].shape == (128, n // 8 * (k // 64))
        for i in range(128):
            torch.save(dict(sequence_id=sequence_ids[i],
                **{objective: values[objective][i].reshape(n // 8, k // 64).clone()
                   for objective in ('ce', 'kl')}), directory / f'{i:03d}.pt')
        del values
        report = run(directory, output, config)
    assert digest_file(source) == source_hash
    report.update(score_directory=None, historical_score_path=str(source), historical_score_sha256=source_hash,
                  temporary_score_adapter_deleted=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ROWBAND COMPLETE ' + json.dumps(report), flush=True)


def freeze_panels(root):
    prior = json.loads((root / 'calibration/report.json').read_text())
    selected = entries(prior)
    expected = [name for _, name, _ in selected]
    panels, hashes = load_layouts([f'rowband_last8={root / "rowband_last8/search"}'], prior, expected, True)
    base = panels['rowband_last8']
    fine_panels, fine_hashes = load_layouts([f'{label}={root / "search" / label}' for label in ('rows', 'both')], prior)
    out = root / 'rowband_last8/panels'
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for family in ('rowband_last8', 'rowband_last1', 'fine_rows_last1', 'fine_both_last1'):
        counts[family] = 0
        for task, (_, name, _) in enumerate(selected):
            layout = copy.deepcopy(base[name])
            is_last = '.layers.63.' in name
            if family.startswith('fine_') and is_last:
                fine = fine_panels['rows' if family == 'fine_rows_last1' else 'both'][name]
                assert fine['fit_sequence_ids'] == layout['fit_sequence_ids']
                assert fine['election_sequence_ids'] == layout['election_sequence_ids']
                # Different float32 summation order in 1x16 vs stored 8x64
                # atoms can affect ties. Reject mismatched controls explicitly.
                for key in ('identity_mask', 'identity_8x64_mask'):
                    assert torch.equal(fine[key], layout[key]), f'Fine/historical control mismatch: {name}/{key}'
                layout = copy.deepcopy(fine)
            elif family != 'rowband_last8' and not is_last:
                n, k = layout['weight_shape']
                layout.update(mask=layout['identity_mask'].clone(), row_perm=torch.arange(n),
                    inverse_row_perm=torch.arange(n), row_atom_perm=torch.arange(n // 8))
            layout['mask_selection'] = dict(method='directional CE/KL mean+3SE on 64 outer sequences',
                family=family, expanded_scope=SCOPE, search_uses_outer_sequences=False)
            directory = out / family / f'{task:03d}'
            directory.mkdir(parents=True)
            torch.save(layout, directory / 'layout.pt')
            count = int(layout['mask'].sum())
            counts[family] += count
            (directory / 'report.json').write_text(json.dumps(dict(status='complete', module=name,
                elected_tiles=count, job_id=os.environ['SLURM_JOB_ID']), indent=2) + '\n')
    assert all(digest_file(path) == value for path, value in {**hashes, **fine_hashes}.items())
    (out / 'report.json').write_text(json.dumps(dict(status='complete', module_regex=SCOPE,
        pilot_elected_tiles=counts, source_layout_sha256={**hashes, **fine_hashes},
        uses_wiki_selection=False, uses_c4_selection=False), indent=2) + '\n')
    print('ROWBAND PANELS ' + json.dumps(counts), flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run CPU-heavy row-band search through H200 Slurm')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('search', 'freeze'))
    ap.add_argument('root', type=Path)
    ap.add_argument('--task', type=int)
    args = ap.parse_args()
    torch.set_num_threads(8)
    if args.action == 'search':
        search(args.root, args.task)
    else:
        freeze_panels(args.root)


if __name__ == '__main__':
    main()
