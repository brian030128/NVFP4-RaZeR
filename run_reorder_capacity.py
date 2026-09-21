"""Bracket how much a legal reordering could ever be worth, before searching for one.

The reordering programme has optimized toward a target of unknown size. This
measures the target. Three quantities bracket it, all on the same fit/election
split and the same objective the deployed search uses:

    identity      the floor: what the current tiling already achieves.
    rectangle     what a row and column permutation can reach. Not computed here;
                  it is what the searches report, and this script exists to give
                  those numbers a scale.
    free          the ceiling: atoms packed into tile-sized groups by rank,
                  ignoring the constraint that a tile is a row-group by
                  column-group rectangle. No permutation can beat this.
    atom          the absolute ceiling: every 1x16 atom elects for itself, which
                  is the nvif4 bound that no coarse geometry can reach.

`free` is the number that matters. A permutation preserves the multiset of atom
gains and can only rearrange them, and it must arrange them into rectangles. If
`free` is barely above `identity`, there is no prize and no search can find one.
If `free` is far above but the searches sit near `identity`, the prize exists and
the rectangle constraint or the search is what fails to reach it.

Every quantity is reported against a matched sign-flip null, because sorting
noise and grouping the top of it also produces a positive objective. The
groupings are derived on the fit sequences and scored on the disjoint election
sequences, matching the search protocol, so these are held-out numbers rather
than in-sample optima.

Also reported: the singular spectrum of the mean gain matrix against the same
null. Checkerboard structure is what makes rectangles work, and a spectrum that
sits where noise would put it means there is none to exploit.

No model, no teacher, no election of any deployable map.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from quantize.bicluster import structure
from quantize.task_reorder import SearchConfig, _features, _labels, _tiles, _values
from run_reorder_objective_ablation import transform
from run_task_reorder import load_scores, split_sequences


def grouped_objective(features, order, group_size, config, scales):
    """Objective for an arbitrary packing of atoms into equal-size groups.

    `features` is [atoms, 2, sequences]; `order` ranks atoms. This deliberately
    ignores the rectangle constraint, which is what makes it an upper bound on
    any permutation rather than a layout.
    """
    labels = torch.empty(features.shape[0], dtype=torch.long)
    labels[order] = torch.arange(features.shape[0]) // group_size
    phi = features.new_zeros((int(labels.max()) + 1, *features.shape[1:]))
    phi.index_add_(0, labels, features)
    return float(_values(phi, config.k, scales).sum()), int((_values(phi, config.k, scales) > 0).sum())


def bracket(fit_ce, fit_kl, election_ce, election_kl, config):
    """Floor, free-assignment ceiling and per-atom ceiling on held-out sequences."""
    fit = _features(fit_ce, fit_kl)
    held = _features(election_ce, election_kl)
    r, c = fit.shape[:2]
    rs, cs = config.tile_rows // config.atom_rows, config.tile_cols // config.atom_cols
    means = fit.reshape(r, c, 2, -1).mean(-1)
    scales = means.square().mean((0, 1)).sqrt().clamp_min(1e-30)

    identity_phi = _tiles(held, _labels(torch.arange(r), rs), _labels(torch.arange(c), cs))
    identity_value = _values(identity_phi, config.k, scales)
    result = dict(
        atoms=r * c, tile_atoms=rs * cs, row_groups=-(-r // rs), column_groups=-(-c // cs),
        identity=float(identity_value.sum()), identity_tiles=int((identity_value > 0).sum()))

    # Rank atoms on FIT, score the resulting packing on ELECTION, exactly as the
    # search ranks on fit and elects on the held-out half.
    gain = -(means / scales).amax(-1).flatten()
    flat_held = held.reshape(r * c, *held.shape[2:])
    order = torch.argsort(gain, descending=True, stable=True)
    free, free_tiles = grouped_objective(flat_held, order, rs * cs, config, scales)
    result.update(free=free, free_tiles=free_tiles)

    # Every atom its own tile: the nvif4 bound no coarse geometry can reach.
    atom_value = _values(flat_held, config.k, scales)
    result.update(atom=float(atom_value.sum()), atom_tiles=int((atom_value > 0).sum()))

    spectrum = structure((means / scales).amax(-1).neg(), rank=4)
    result['spectrum'] = spectrum
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fit-fraction', type=float, default=.5)
    ap.add_argument('--placebo-seed', type=int, default=1234)
    ap.add_argument('--threads', type=int, default=8)
    defaults = SearchConfig()
    for name, default in asdict(defaults).items():
        ap.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this CPU-heavy analysis through Slurm')
    torch.set_num_threads(args.threads)
    config = SearchConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    config.validate()

    manifest = json.loads((args.scores / 'manifest.json').read_text())
    if manifest.get('schema') != 'mixfp4_reorder_scores_v1' or manifest.get('status') != 'complete':
        raise ValueError('Need a complete fine-score manifest')
    fit, election = split_sequences(manifest, args.fit_fraction, config.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    print(f'LOAD {manifest["name"]}', flush=True)
    fit_ce, fit_kl = load_scores(args.scores, manifest, fit)
    election_ce, election_kl = load_scores(args.scores, manifest, election)

    report = dict(status='running', module=manifest['name'], config=asdict(config),
                  job_id=os.environ.get('SLURM_JOB_ID'), fit_sequences=len(fit),
                  election_sequences=len(election), model_forward_passes=0)
    real = bracket(fit_ce, fit_kl, election_ce, election_kl, config)
    # The same bracket on sign-flipped scores: sorting noise and grouping its top
    # also yields a positive objective, so the ceiling means nothing un-nulled.
    null_fit = transform('placebo', fit_ce, fit_kl, args.placebo_seed, 'fit')
    null_held = transform('placebo', election_ce, election_kl, args.placebo_seed, 'election')
    null = bracket(null_fit[0], null_fit[1], null_held[0], null_held[1], config)

    headroom = real['free'] - real['identity']
    report.update(real=real, null=null, status='complete',
                  free_over_identity=real['free'] / real['identity'] if real['identity'] > 0 else None,
                  free_headroom=headroom,
                  free_headroom_over_null=headroom / max(null['free'] - null['identity'], 1e-12),
                  seconds=time.perf_counter() - started)
    args.out.joinpath('report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT ' + json.dumps({k: report[k] for k in (
        'free_over_identity', 'free_headroom', 'free_headroom_over_null')}
        | {'real': {k: real[k] for k in ('identity', 'free', 'atom', 'identity_tiles',
                                         'free_tiles', 'atom_tiles')},
           'null': {k: null[k] for k in ('identity', 'free', 'atom')}}), flush=True)


if __name__ == '__main__':
    main()
