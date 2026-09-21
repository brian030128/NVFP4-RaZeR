"""Measure how much of the reordering search's FIT gain survives to held-out data.

The deployed search maximizes a CE/KL conjunction bound on 64 fit sequences and
re-elects formats on 64 disjoint election sequences. The recorded
election/fit objective ratio is 0 for every Qwen matrix before layer 62 and only
0.04-0.06 at the final MLP, so the layout search is dominated by overfitting
rather than by depth. This script asks which selection objective generalizes
best, using the saved 1x16 score shards only. It runs no model and no teacher.

Every variant is an input transform on the score tensors; quantize/task_reorder.py
is untouched, so the `ce_kl` variant reproduces the deployed rule exactly.

    ce_kl    deployed conjunction, both bounds must clear zero
    kl       KL duplicated into both channels, so the conjunction is KL alone
    ce       CE duplicated into both channels
    shrunk   deployed conjunction on positive-part James-Stein shrunken fit means
    placebo  balanced per-sequence sign flip; the population mean is destroyed and
             the remaining fit objective is what the search manufactures from noise

`placebo` is the control the study never had. If its fit objective approaches the
real one, the fit objective is not evidence of a usable layout.

Read the ratio, not the absolute objective: each variant normalizes by its own
fit-derived scales. The identity election objective is the honest floor, since a
layout that does not beat it on held-out sequences is worse than not reordering.
This measures a surrogate on held-out scores. It is not model loss, not PPL, and
not a substitute for the frozen fresh-document gate.
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
from run_task_reorder import load_scores, split_sequences

VARIANTS = ('ce_kl', 'kl', 'ce', 'shrunk', 'placebo', 'placebo_kl', 'placebo_ce')

# Each real variant's matched null. Fit objectives are only comparable within a
# variant, because collapsing the conjunction onto one channel removes a
# constraint and raises the attainable objective on its own. Held-out election
# numbers need no such pairing.
MATCHED_NULL = {'ce_kl': 'placebo', 'shrunk': 'placebo',
                'kl': 'placebo_kl', 'ce': 'placebo_ce'}


def balanced_signs(count, seed):
    """Exactly half -1 so the across-sequence mean of a constant signal is zero."""
    if count < 2 or count % 2:
        raise ValueError('Balanced sign flip needs an even sequence count >= 2')
    signs = torch.ones(count, dtype=torch.float32)
    signs[:count // 2] = -1.0
    return signs[torch.randperm(count, generator=torch.Generator().manual_seed(seed))]


def shrink_(scores, floor=0.0):
    """Positive-part James-Stein shrinkage of each atom's mean, in place.

    Keeps every sequence's deviation from its atom mean, so the noise the search
    must overcome is unchanged and only the apparent signal is regularized. An
    atom whose mean is within one standard error of zero is erased.
    """
    n = scores.shape[0]
    if n < 3:
        raise ValueError('Shrinkage needs at least three sequences')
    # float64 throughout: these scores are gradient inner products, and squaring
    # a small float32 mean underflows to zero and turns the ratio into inf/nan.
    values = scores.double()
    mean = values.mean(0, keepdim=True)
    standard_error = values.std(0, unbiased=True, keepdim=True) / (n ** 0.5)
    square = mean.square()
    # An atom with no signal at all shrinks to zero rather than dividing by zero.
    factor = torch.where(square > 0,
                         (1 - standard_error.square() / square.clamp_min(torch.finfo(
                             torch.float64).tiny)).clamp_min(floor),
                         torch.zeros_like(square))
    shrunk = (values - mean + mean * factor).to(scores.dtype)
    if not torch.isfinite(shrunk).all():
        raise ValueError('Shrinkage produced a nonfinite score')
    scores.copy_(shrunk)
    return float(factor.mean())


def transform(variant, ce, kl, seed, split):
    """Return (ce, kl) for one variant plus a JSON-safe description of the change."""
    detail = {}
    # A matched null collapses the conjunction the same way its real variant does
    # before flipping, so the two searches face an identically shaped problem.
    if variant == 'placebo_kl':
        ce, variant = kl.clone(), 'placebo'
    elif variant == 'placebo_ce':
        kl, variant = ce.clone(), 'placebo'
    if variant == 'kl':
        ce = kl.clone()
    elif variant == 'ce':
        kl = ce.clone()
    elif variant == 'shrunk':
        detail['mean_shrinkage_ce'] = shrink_(ce)
        detail['mean_shrinkage_kl'] = shrink_(kl)
    elif variant == 'placebo':
        # One sign per sequence, applied to both objectives so their covariance
        # survives. Balanced within the split, so no atom keeps a true mean.
        signs = balanced_signs(ce.shape[0], seed + (0 if split == 'fit' else 1))
        ce = ce * signs[:, None, None]
        kl = kl * signs[:, None, None]
        detail['flipped_sequences'] = int((signs < 0).sum())
    elif variant != 'ce_kl':
        raise ValueError(f'Unknown variant {variant}')
    return ce, kl, detail


def run(directory, output, variant, config, fit_fraction=.5, placebo_seed=1234):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant}')
    config.validate()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'mixfp4_reorder_scores_v1' or manifest.get('status') != 'complete':
        raise ValueError('Need a complete fine-score manifest')
    if manifest['atom_shape'] != [config.atom_rows, config.atom_cols]:
        raise ValueError('Config atom shape differs from score manifest')
    # The split is the deployed one, so every variant sees the same sequences and
    # only the objective differs.
    fit, election = split_sequences(manifest, fit_fraction, config.seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    print(f'LOAD FIT {manifest["name"]} [{variant}]: {len(fit)} sequences', flush=True)
    ce, kl = load_scores(directory, manifest, fit)
    ce, kl, fit_detail = transform(variant, ce, kl, placebo_seed, 'fit')
    layout = search_layout(ce, kl, config, progress=lambda event: print(
        'SEARCH ' + json.dumps(event), flush=True))
    del ce, kl

    print(f'LOAD ELECTION {manifest["name"]} [{variant}]: {len(election)} sequences', flush=True)
    ce, kl = load_scores(directory, manifest, election)
    # Only the placebo touches election data: its null must hold on both splits.
    # Shrinkage is a fit-side regularizer, so held-out scoring stays untouched.
    election_variant = variant if variant in ('kl', 'ce', 'placebo') else 'ce_kl'
    ce, kl, election_detail = transform(election_variant, ce, kl, placebo_seed, 'election')
    result = elect_layout(ce, kl, layout)
    identity = dict(layout, row_atom_perm=torch.arange(ce.shape[1]),
                    col_atom_perm=torch.arange(ce.shape[2]))
    control = elect_layout(ce, kl, identity)
    del ce, kl

    fit_objective = layout['fit_objective']
    election_objective = result['objective']
    report = dict(
        status='complete', variant=variant, module=manifest['name'],
        score_directory=str(directory.resolve()), job_id=os.environ.get('SLURM_JOB_ID'),
        config=asdict(config), fit_sequences=len(fit), election_sequences=len(election),
        fit_objective=fit_objective, fit_identity_objective=layout['identity_objective'],
        election_objective=election_objective,
        election_identity_objective=control['objective'],
        # The headline: the share of the fitted surrogate gain that is real.
        election_fit_ratio=election_objective / fit_objective if fit_objective > 0 else None,
        beats_identity_on_election=election_objective > control['objective'],
        elected_tiles=int(result['mask'].sum()),
        identity_elected_tiles=int(control['mask'].sum()),
        total_tiles=int(result['mask'].numel()),
        fit_transform=fit_detail, election_transform=election_detail,
        election_transform_variant=election_variant,
        seconds=time.perf_counter() - started,
        quality_evaluated=False, model_forward_passes=0,
        source_sha256={name: hashlib.sha256(
            (Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ('run_reorder_objective_ablation.py', 'run_task_reorder.py',
                         'quantize/task_reorder.py')})
    torch.save(dict(layout, mask=result['mask'], identity_mask=control['mask'],
                    variant=variant, name=manifest['name']), output / 'layout.pt')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT ' + json.dumps({k: report[k] for k in (
        'variant', 'module', 'fit_objective', 'election_objective', 'election_fit_ratio',
        'election_identity_objective', 'elected_tiles', 'identity_elected_tiles')}), flush=True)
    return report


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this CPU-heavy search through Slurm')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--fit-fraction', type=float, default=.5)
    ap.add_argument('--placebo-seed', type=int, default=1234)
    ap.add_argument('--threads', type=int, default=8)
    defaults = SearchConfig()
    for name, default in asdict(defaults).items():
        ap.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    config = SearchConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    run(args.scores, args.out, args.variant, config, args.fit_fraction, args.placebo_seed)


if __name__ == '__main__':
    main()
