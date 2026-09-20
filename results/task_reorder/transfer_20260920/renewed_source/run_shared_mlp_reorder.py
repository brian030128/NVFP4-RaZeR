"""Fit one foldable MLP ordering with nested calibration validation, on Slurm."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import torch

from quantize.shared_mlp_reorder import SharedConfig, aggregate_atoms, search_shared, select_sequences
from quantize.task_reorder import SearchConfig, elect_layout
from run_task_reorder import load_scores, split_sequences


def nested_split(manifest, fit):
    generator = torch.Generator().manual_seed(7349)
    train, validation = [], []
    for source in sorted(set(manifest['sequence_sources'])):
        positions = [i for i, original in enumerate(fit) if manifest['sequence_sources'][original] == source]
        if len(positions) % 2 or len(positions) < 8:
            raise ValueError('Need equal inner train/validation counts per source')
        positions = [positions[i] for i in torch.randperm(len(positions), generator=generator).tolist()]
        train.extend(positions[:len(positions) // 2])
        validation.extend(positions[len(positions) // 2:])
    return sorted(train), sorted(validation)


def module_layout(manifest, group_perm, scales):
    n, k = manifest['weight_shape']
    kind = manifest['name'].rsplit('.', 1)[-1]
    channels = (group_perm[:, None] * 16 + torch.arange(16)).flatten()
    rp = torch.arange(n) if kind == 'down_proj' else channels
    cp = channels if kind == 'down_proj' else torch.arange(k)
    cfg = SearchConfig(axes='cols' if kind == 'down_proj' else 'rows')
    return dict(schema='mixfp4_task_reorder_v1', config=asdict(cfg), name=manifest['name'],
                row_perm=rp, col_perm=cp, inverse_row_perm=torch.argsort(rp), inverse_col_perm=torch.argsort(cp),
                row_atom_perm=rp, col_atom_perm=cp.reshape(-1, 16)[:, 0] // 16,
                weight_shape=[n, k], padded_shape=[n, k], objective_scales=scales,
                deployment='Shared gate/up output and down input order, folded into weights',
                shared_intermediate_group_perm=group_perm)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use Slurm')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--domain-robust', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(8)
    config = SharedConfig(domain_robust=args.domain_robust)
    started = time.perf_counter()
    entries = [(path.parent, json.loads(path.read_text())) for path in sorted(args.scores.glob('*/manifest.json'))]
    if len(entries) != 3 or any(m['status'] != 'complete' for _, m in entries):
        raise ValueError('Need the three complete MLP score manifests')
    reference = entries[0][1]
    fit, election = split_sequences(reference, .5, 0)
    train_indices, validation_indices = nested_split(reference, fit)
    args.out.mkdir(parents=True, exist_ok=False)
    train, validation, export_scales = {}, {}, {}
    for directory, manifest in entries:
        assert manifest['sequence_ids'] == reference['sequence_ids']
        assert manifest['sequence_sources'] == reference['sequence_sources']
        assert manifest['atom_shape'] == [1, 16]
        kind = manifest['name'].rsplit('.', 1)[-1]
        ce, kl = load_scores(directory, manifest, fit)
        export_scales[kind] = torch.stack([x.double().mean(0).square().mean().sqrt().clamp_min(1e-30) for x in (ce, kl)])
        features = aggregate_atoms(ce, kl, kind)
        del ce, kl
        train[kind] = select_sequences(features, train_indices)
        validation[kind] = select_sequences(features, validation_indices)
        del features
        print(f'LOADED NESTED {kind}: train={len(train_indices)}, validation={len(validation_indices)}', flush=True)
    result = search_shared(train, validation, config,
                           progress=lambda record: print('SHARED ' + json.dumps(record), flush=True))
    del train, validation
    freeze = dict(config=asdict(config), candidates=result['candidates'], selected=result['selected_validation'],
                  inner_train_sequence_ids=[reference['sequence_ids'][fit[i]] for i in train_indices],
                  inner_validation_sequence_ids=[reference['sequence_ids'][fit[i]] for i in validation_indices],
                  fit_sequence_ids=[reference['sequence_ids'][i] for i in fit],
                  election_sequence_ids=[reference['sequence_ids'][i] for i in election],
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ('run_shared_mlp_reorder.py', 'quantize/shared_mlp_reorder.py')},
                  selection='Signed joint CE/KL transfer of train-elected tiles on inner validation; outer election unseen')
    # Persist the selected permutation BEFORE opening any outer-election shard.
    torch.save(dict(group_perm=result['group_perm'], **freeze), args.out / 'shared_frozen.pt')
    reports = []
    for directory, manifest in entries:
        kind = manifest['name'].rsplit('.', 1)[-1]
        layout = module_layout(manifest, result['group_perm'], export_scales[kind])
        ce, kl = load_scores(directory, manifest, fit)
        fit_result = elect_layout(ce, kl, layout)
        del ce, kl
        ce, kl = load_scores(directory, manifest, election)
        elected = elect_layout(ce, kl, layout)
        identity = dict(layout, row_atom_perm=torch.arange(ce.shape[1]), col_atom_perm=torch.arange(ce.shape[2]))
        control = elect_layout(ce, kl, identity)
        fine = elect_layout(ce, kl, dict(identity, config={**identity['config'], 'tile_rows': 8}))
        del ce, kl
        layout.update(mask=elected['mask'], fit_mask=fit_result['mask'],
                      identity_mask=control['mask'], identity_8x64_mask=fine['mask'],
                      election_upper_ce=elected['upper_ce'], election_upper_kl=elected['upper_kl'],
                      identity_upper_ce=control['upper_ce'], identity_upper_kl=control['upper_kl'],
                      provenance=manifest, nested_selection=freeze, fit_objective=fit_result['objective'],
                      fit_sequence_ids=freeze['fit_sequence_ids'], election_sequence_ids=freeze['election_sequence_ids'])
        output = args.out / directory.name
        output.mkdir()
        torch.save(layout, output / 'layout.pt')
        report = dict(status='complete', module=manifest['name'], job_id=os.environ['SLURM_JOB_ID'],
                      elected_tiles=int(elected['mask'].sum()), identity_elected_tiles=int(control['mask'].sum()),
                      identity_8x64_elected_tiles=int(fine['mask'].sum()), election_objective=elected['objective'],
                      nested_selection=freeze, deployment=layout['deployment'],
                      runtime_activation_gathers=0, runtime_output_scatters=0, native_overhead_measured=False)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    (args.out / 'report.json').write_text(json.dumps(dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.perf_counter() - started, modules=reports, selected_validation=result['selected_validation'],
        identity_validation=result['identity_validation']), indent=2) + '\n')
    print('COMPLETE SHARED ' + json.dumps({r['module']: r['elected_tiles'] for r in reports}), flush=True)


if __name__ == '__main__':
    main()
