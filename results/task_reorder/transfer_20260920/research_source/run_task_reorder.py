"""Search a single matrix's legal row/K-group layout and freeze its type map.

Run under Slurm. Input is a module directory emitted by
run_math_code_calibration.py --reorder-modules REGEX. Search and election use
disjoint, source-stratified sequences. This does not evaluate model quality.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from quantize.task_reorder import SearchConfig, elect_layout, search_layout


def split_sequences(manifest, fit_fraction, seed):
    ids, sources = manifest['sequence_ids'], manifest['sequence_sources']
    if len(ids) != len(sources) or len(ids) != len(set(ids)):
        raise ValueError('Sequence IDs must be unique and paired with sources')
    if not 0 < fit_fraction < 1:
        raise ValueError('fit-fraction must be strictly between zero and one')
    generator = torch.Generator().manual_seed(seed)
    fit, election = [], []
    for source in sorted(set(sources)):
        indices = [i for i, value in enumerate(sources) if value == source]
        count = int(len(indices) * fit_fraction)
        if count < 2 or len(indices) - count < 2:
            raise ValueError('Need >= 2 fit and election sequences per source')
        shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        fit.extend(shuffled[:count])
        election.extend(shuffled[count:])
    return sorted(fit), sorted(election)


def load_scores(directory, manifest, indices):
    shape = tuple(n // a for n, a in zip(manifest['weight_shape'], manifest['atom_shape']))
    ce = torch.empty((len(indices), *shape), dtype=torch.float32)
    kl = torch.empty_like(ce)
    for output, index in enumerate(indices):
        shard = torch.load(directory / f'{index:03d}.pt', map_location='cpu', weights_only=True)
        if shard['sequence_id'] != manifest['sequence_ids'][index]:
            raise ValueError('Sequence identity mismatch')
        if shard['ce'].shape != shape or shard['kl'].shape != shape:
            raise ValueError('Score atom shape mismatch')
        ce[output].copy_(shard['ce'])
        kl[output].copy_(shard['kl'])
    return ce, kl


def run(directory, output, config, fit_fraction=.5):
    config.validate()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'mixfp4_reorder_scores_v1' or manifest.get('status') != 'complete':
        raise ValueError('Need a complete fine-score manifest')
    if manifest['atom_shape'] != [config.atom_rows, config.atom_cols]:
        raise ValueError('Config atom shape differs from score manifest')
    if len(manifest['weight_shape']) != 2 or any(
            n < 1 or n % a for n, a in zip(manifest['weight_shape'], manifest['atom_shape'])):
        raise ValueError('Weight shape must be divisible by atom shape')
    fit, election = split_sequences(manifest, fit_fraction, config.seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(f'LOAD FIT {manifest["name"]}: {len(fit)} sequences, axes={config.axes}', flush=True)
    ce, kl = load_scores(directory, manifest, fit)
    layout = search_layout(ce, kl, config, progress=lambda event: print(
        'SEARCH ' + json.dumps(event), flush=True))
    del ce, kl
    print(f'LOAD ELECTION {manifest["name"]}: {len(election)} sequences', flush=True)
    ce, kl = load_scores(directory, manifest, election)
    result = elect_layout(ce, kl, layout)
    # Independent-election identity control is diagnostic only. Never select
    # permutations or tune search parameters using these scores.
    identity = dict(layout)
    identity['row_atom_perm'] = torch.arange(ce.shape[1])
    identity['col_atom_perm'] = torch.arange(ce.shape[2])
    control = elect_layout(ce, kl, identity)
    fine = dict(identity, config={**identity['config'], 'tile_rows': 8, 'tile_cols': 64})
    fine_control = elect_layout(ce, kl, fine)
    del ce, kl
    layout.update(mask=result['mask'], election_upper_ce=result['upper_ce'],
                  election_upper_kl=result['upper_kl'], name=manifest['name'],
                  identity_mask=control['mask'], identity_upper_ce=control['upper_ce'],
                  identity_upper_kl=control['upper_kl'],
                  identity_8x64_mask=fine_control['mask'],
                  provenance=manifest, fit_sequence_ids=[manifest['sequence_ids'][i] for i in fit],
                  election_sequence_ids=[manifest['sequence_ids'][i] for i in election])
    layout['search_source_sha256'] = {
        name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
        for name in ('run_task_reorder.py', 'quantize/task_reorder.py')}
    torch.save(layout, output / 'layout.pt')
    report = dict(status='complete', module=manifest['name'], score_directory=str(directory.resolve()),
                  job_id=os.environ.get('SLURM_JOB_ID'), config=asdict(config),
                  search_source_sha256=layout['search_source_sha256'],
                  fit_sequences=len(fit), election_sequences=len(election),
                  fit_objective=layout['fit_objective'], fit_identity_objective=layout['identity_objective'],
                  election_objective=result['objective'], election_identity_objective=control['objective'],
                  elected_tiles=int(result['mask'].sum()), identity_elected_tiles=int(control['mask'].sum()),
                  identity_8x64_elected_tiles=int(fine_control['mask'].sum()),
                  total_tiles=result['mask'].numel(), seconds=time.perf_counter()-started,
                  quality_evaluated=False, native_kernel_implemented=False,
                  activation_contract='Gather X by col_perm, or fold matching permutation into producer',
                  output_contract='Scatter new output column j to original row_perm[j] in epilogue')
    n, k = layout['weight_shape']
    active = bool(result['mask'].any())
    row_changed = active and not torch.equal(layout['row_perm'], torch.arange(n))
    col_changed = active and not torch.equal(layout['col_perm'], torch.arange(k))
    report['layout_cost'] = dict(
        rows_changed=row_changed, columns_changed=col_changed,
        identity_deployment_when_no_tiles=not active,
        permutation_index_bytes_int32=4 * (n * row_changed + k * col_changed),
        type_map_bytes_packed=(result['mask'].numel() + 7) // 8,
        extra_unfused_bf16_activation_bytes_per_token=4 * k * col_changed,
        extra_unfused_bf16_output_bytes_per_token=4 * n * row_changed,
        native_fused_overhead_measured=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this CPU-heavy search through Slurm')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fit-fraction', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=4)
    for name, default in asdict(SearchConfig()).items():
        ap.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    config = SearchConfig(**{name: getattr(args, name) for name in asdict(SearchConfig())})
    print(json.dumps(run(args.scores, args.out, config, args.fit_fraction), indent=2), flush=True)


if __name__ == '__main__':
    main()
