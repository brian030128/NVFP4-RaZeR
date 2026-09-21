"""Derive the three frozen mechanism experiment families from verified N16 moments.

No model is loaded and no outcome is read.  Every map is a deterministic function of
the frozen score moments, exact primary map, protocol JSON, and keyed seeds.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from campaign import mapio
from campaign import runtime


_workspace = Path(os.environ.get('MIXFP4_WORKSPACE_ROOT', Path.cwd()))
PARENT = Path(os.environ.get(
    'MIXFP4_PRIMARY_CAMPAIGN',
    _workspace / 'research_runs/mixfp4_n16k64_full_validation_20260911T065444Z',
))
MODEL_RUN = {
    'llama8b': ('V30_calib_llama8b_seed0_attempt1', 'llama8b'),
    'qwen4b': ('V30_calib_qwen4b_seed0_attempt1', 'qwen4b'),
    'mistral7b': ('V61_calib_mistral7b_seed0_attempt2', 'mistral7b'),
}
K = 3.0
GROUP_FRACTION = 0.25
VETO_FRACTION = 0.25
VETO_BINS = 5
SEED = 20260917


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def module_stratum(name):
    m = re.search(r'\.layers\.(\d+)\.', name)
    if not m:
        raise ValueError(f'cannot parse layer from {name}')
    projection = name.rsplit('.', 1)[-1]
    family = 'attention' if '.self_attn.' in name else 'mlp' if '.mlp.' in name else None
    if family is None:
        raise ValueError(f'neither attention nor MLP: {name}')
    return int(m.group(1)), projection, family


def mean_se(st, objective):
    n = int(st['n'])
    s, q = st[f'{objective}_sum'].double(), st[f'{objective}_sq'].double()
    mean = s / n
    var = ((q - s * s / n) / (n - 1)).clamp_min(0)
    return mean, (var / n).sqrt()


def largest_remainder(counts, fraction):
    total = sum(counts.values())
    target = int(math.floor(total * fraction))
    caps = {n: c // 3 for n, c in counts.items()}
    quota = {n: target * c / total if total else 0.0 for n, c in counts.items()}
    alloc = {n: min(caps[n], int(math.floor(quota[n]))) for n in counts}
    left = target - sum(alloc.values())
    order = sorted(counts, key=lambda n: (-(quota[n] - math.floor(quota[n])), n))
    while left:
        changed = False
        for n in order:
            if alloc[n] < caps[n]:
                alloc[n] += 1
                left -= 1
                changed = True
                if not left:
                    break
        if not changed:
            raise ValueError('25% group budget is infeasible under disjoint three-group caps')
    return target, alloc


def keyed_indices(indices, label):
    return sorted((int(i) for i in indices), key=lambda i: hashlib.sha256(f'{label}:{i}'.encode()).digest())


def empty_like(full):
    return {n: torch.zeros_like(m) for n, m in full.items()}


def clone_masks(masks):
    return {n: m.clone() for n, m in masks.items()}


def mask_union(a, b):
    return {n: a[n] | b[n] for n in a}


def score_summary(mask, stats):
    ce, kl, uce, ukl, comb = [], [], [], [], []
    for n, m in mask.items():
        flat = m.flatten()
        if flat.any():
            s = stats[n]
            ce.append(s['ce_mean'][flat]); kl.append(s['kl_mean'][flat])
            uce.append(s['uce'][flat]); ukl.append(s['ukl'][flat]); comb.append(s['combined_margin'][flat])
    def total(xs): return float(torch.cat(xs).sum()) if xs else 0.0
    def avg(xs): return float(torch.cat(xs).mean()) if xs else None
    return {'tiles': sum(int(x.sum()) for x in mask.values()),
            'sum_ce_first_order_mean': total(ce), 'sum_kl_first_order_mean': total(kl),
            'mean_ce_upper': avg(uce), 'mean_kl_upper': avg(ukl),
            'mean_combined_margin': avg(comb)}


def derive(model, protocol, out_dir, source_manifest):
    run_name, stem = MODEL_RUN[model]
    source_run = PARENT / 'runs' / run_name
    moments_path = source_run / 'calibration/moments/moments_full.pt'
    base_path = source_run / f'maps/{stem}_seed0_n16_k3.mixfp4map'
    mom = torch.load(moments_path, map_location='cpu', weights_only=True, mmap=True)
    base_header, base_masks, base_sha = mapio.read_map(base_path)
    names = mom['names']
    if names != list(base_masks):
        raise ValueError(f'{model}: moment/map module order mismatch')
    stats = {}
    reconstructed = {}
    score_rows = []
    for name in names:
        st = mom['n16'][name]
        cm, cs = mean_se(st, 'ce'); km, ks = mean_se(st, 'kl')
        uce, ukl = cm + K * cs, km + K * ks
        combined = -torch.maximum(uce, ukl)
        mask = (uce < 0) & (ukl < 0)
        reconstructed[name] = mask.reshape(base_masks[name].shape)
        stats[name] = {'ce_mean': cm, 'ce_se': cs, 'ce_standardized': torch.where(cs > 0, cm / cs, torch.sign(cm) * torch.inf),
                       'ce_margin': -uce, 'uce': uce, 'kl_mean': km, 'kl_se': ks,
                       'kl_standardized': torch.where(ks > 0, km / ks, torch.sign(km) * torch.inf),
                       'kl_margin': -ukl, 'ukl': ukl, 'combined_margin': combined}
    mismatches = sum(int((reconstructed[n] != base_masks[n]).sum()) for n in names)
    if mismatches:
        raise ValueError(f'{model}: reconstructed primary map differs at {mismatches} tiles')
    full = reconstructed
    selected_counts = {n: int(full[n].sum()) for n in names}

    # A: composition-matched, mutually disjoint strongest / weakest / random groups.
    group_target, alloc = largest_remainder(selected_counts, GROUP_FRACTION)
    strongest, weakest, random_group = empty_like(full), empty_like(full), empty_like(full)
    excluded = []
    for name in names:
        q = alloc[name]
        chosen = torch.nonzero(full[name].flatten(), as_tuple=False).flatten()
        if q == 0:
            if chosen.numel(): excluded.append({'module': name, 'selected_support': int(chosen.numel()), 'reason': 'largest-remainder allocation is zero'})
            continue
        margin = stats[name]['combined_margin'][chosen]
        order = torch.argsort(margin, descending=True, stable=True)
        strong_idx = chosen[order[:q]]
        weak_idx = chosen[order[-q:]]
        reserved = set(map(int, strong_idx.tolist())) | set(map(int, weak_idx.tolist()))
        middle = [int(i) for i in chosen.tolist() if int(i) not in reserved]
        rand_idx = keyed_indices(middle, f'ranking-random:{SEED}:{model}:{name}')[:q]
        for dst, ix in ((strongest, strong_idx.tolist()), (weakest, weak_idx.tolist()), (random_group, rand_idx)):
            dst[name].view(-1)[torch.tensor(ix, dtype=torch.long)] = True
    for name in names:
        q = alloc[name]
        if not (int(strongest[name].sum()) == int(weakest[name].sum()) == int(random_group[name].sum()) == q):
            raise AssertionError(f'{model} composition mismatch {name}')
        if (strongest[name] & weakest[name]).any() or (strongest[name] & random_group[name]).any() or (weakest[name] & random_group[name]).any():
            raise AssertionError(f'{model} non-disjoint ranking groups {name}')

    # B: strongest approving-objective veto add-backs and disjoint exact-cell random controls.
    veto_masks = {}
    veto_records = {}
    for label, approve in (('kl_vetoed_ce_approved', 'ce'), ('ce_vetoed_kl_approved', 'kl')):
        cls = {}
        values = []
        locs = []
        for name in names:
            uce, ukl = stats[name]['uce'], stats[name]['ukl']
            c = ((uce < 0) & (ukl >= 0)) if approve == 'ce' else ((ukl < 0) & (uce >= 0))
            cls[name] = c
            ix = torch.nonzero(c, as_tuple=False).flatten()
            margin = (-uce if approve == 'ce' else -ukl)[ix]
            for i, v in zip(ix.tolist(), margin.tolist()):
                locs.append((name, int(i))); values.append(float(v))
        values_np = np.asarray(values, np.float64)
        if values_np.size == 0:
            raise ValueError(f'{model}: empty {label} class')
        edges = np.quantile(values_np, np.linspace(0, 1, VETO_BINS + 1), method='linear')
        bins = np.searchsorted(edges[1:-1], values_np, side='right')
        target = min(int(math.floor(sum(selected_counts.values()) * VETO_FRACTION)), len(locs))
        order = sorted(range(len(locs)), key=lambda j: (-values[j], locs[j][0], locs[j][1]))[:target]
        proposed = set(order)
        cell_all, cell_prop = {}, {}
        for j, ((name, idx), b) in enumerate(zip(locs, bins.tolist())):
            cell = (name, int(b))
            cell_all.setdefault(cell, []).append(j)
            if j in proposed: cell_prop.setdefault(cell, []).append(j)
        actual, control = empty_like(full), empty_like(full)
        reductions = []
        for cell in sorted(cell_all):
            all_j = cell_all[cell]
            prop = sorted(cell_prop.get(cell, []), key=lambda j: (-values[j], locs[j][1]))
            keep_n = min(len(prop), len(all_j) // 2)
            keep = prop[:keep_n]
            keep_set = set(keep)
            remaining = [j for j in all_j if j not in keep_set]
            random_j = keyed_indices(remaining, f'veto-random:{SEED}:{model}:{label}:{cell[0]}:bin{cell[1]}')[:keep_n]
            # keyed_indices returned indices in the `j` coordinate, exactly what is needed here.
            for j in keep:
                name, idx = locs[j]; actual[name].view(-1)[idx] = True
            for j in random_j:
                name, idx = locs[j]; control[name].view(-1)[idx] = True
            if keep_n < len(prop):
                reductions.append({'module': cell[0], 'bin': cell[1], 'proposed': len(prop), 'kept': keep_n,
                                   'class_support': len(all_j), 'reason': 'disjoint exact-match common support'})
        for name in names:
            if int(actual[name].sum()) != int(control[name].sum()):
                raise AssertionError(f'{model} veto module mismatch {label} {name}')
            if (actual[name] & control[name]).any():
                raise AssertionError(f'{model} veto/control overlap {label} {name}')
        veto_masks[label] = (actual, control)
        veto_records[label] = {'approving_objective': approve, 'class_tiles': len(locs), 'target_before_common_support': target,
                               'actual_tiles': sum(int(x.sum()) for x in actual.values()), 'random_tiles': sum(int(x.sum()) for x in control.values()),
                               'margin_bin_edges': edges.tolist(), 'bins': VETO_BINS, 'reductions': reductions}

    # C: actual attention/MLP partition and per-module count-matched random partition.
    attention, mlp = empty_like(full), empty_like(full)
    rand_attention, rand_mlp = empty_like(full), empty_like(full)
    for name in names:
        _, _, family = module_stratum(name)
        (attention if family == 'attention' else mlp)[name] = full[name].clone()
        count = int(full[name].sum())
        dst = rand_attention if family == 'attention' else rand_mlp
        if count:
            seed = int(hashlib.sha256(f'interaction-random:{SEED}:{model}:{name}'.encode()).hexdigest()[:16], 16)
            g = torch.Generator().manual_seed(seed)
            dst[name].view(-1)[torch.randperm(full[name].numel(), generator=g)[:count]] = True
    rand_union = mask_union(rand_attention, rand_mlp)
    if any((attention[n] & mlp[n]).any() for n in names):
        raise AssertionError('attention/MLP overlap')
    if any(not torch.equal(attention[n] | mlp[n], full[n]) for n in names):
        raise AssertionError('attention/MLP union differs from full')

    maps = {
        'full': full,
        'group_only_strongest': strongest, 'group_only_weakest': weakest, 'group_only_random': random_group,
        'full_minus_strongest': {n: full[n] & ~strongest[n] for n in names},
        'full_minus_weakest': {n: full[n] & ~weakest[n] for n in names},
        'full_minus_random': {n: full[n] & ~random_group[n] for n in names},
        'full_plus_kl_vetoed_ce_approved': mask_union(full, veto_masks['kl_vetoed_ce_approved'][0]),
        'full_plus_kl_vetoed_matched_random': mask_union(full, veto_masks['kl_vetoed_ce_approved'][1]),
        'full_plus_ce_vetoed_kl_approved': mask_union(full, veto_masks['ce_vetoed_kl_approved'][0]),
        'full_plus_ce_vetoed_matched_random': mask_union(full, veto_masks['ce_vetoed_kl_approved'][1]),
        'attention_only': attention, 'mlp_only': mlp,
        'matched_random_attention': rand_attention, 'matched_random_mlp': rand_mlp, 'matched_random_union': rand_union,
    }
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    shape = {n: tuple(mom['shapes'][n]) for n in names}
    entries = []
    for policy, masks in maps.items():
        grid = {n: masks[n].reshape(base_masks[n].shape) for n in names}
        family = ('ranking' if policy.startswith(('group_', 'full_minus')) else
                  'veto' if policy.startswith('full_plus') else
                  'interaction' if policy in ('attention_only', 'mlp_only', 'matched_random_attention', 'matched_random_mlp', 'matched_random_union') else
                  'shared_reference')
        header = mapio.build_header(protocol_id=protocol['protocol_id'],
            policy={'name': policy, 'family': family, 'frozen_k': 3, 'seed': SEED},
            model=base_header['model'], type_block=(16, 64), masks=grid, weight_shapes=shape,
            source_manifest_sha256=source_manifest,
            calibration_manifest_sha256=base_header['calibration_manifest_sha256'])
        path = model_dir / f'{model}_{policy}.mixfp4map'
        digest, path = mapio.write_map(path, header, grid, provenance={
            'protocol_sha256': file_sha(protocol['protocol_path']), 'source_primary_map': str(base_path),
            'source_primary_map_sha256': base_sha, 'source_moments': str(moments_path),
            'source_moments_sha256': file_sha(moments_path), 'anchor_mask_mismatches': mismatches,
            'definition': protocol['map_definitions'].get(family, protocol['map_definitions']['shared_reference'])})
        entries.append({'model': model, 'policy': policy, 'family': family, 'path': path, 'sha256': digest,
                        'type_block': [16, 64], 'selected_tiles': header['totals']['selected_tiles'],
                        'total_tiles': header['totals']['total_tiles'], 'score_summary': score_summary(grid, stats)})

    # Selected-tile score components are compact enough to retain in full.
    cols = {k: [] for k in ('module_index', 'tile_index', 'layer', 'family_code', 'ce_mean', 'ce_se', 'ce_standardized',
                             'ce_margin', 'kl_mean', 'kl_se', 'kl_standardized', 'kl_margin', 'combined_margin')}
    for mi, name in enumerate(names):
        ix = torch.nonzero(full[name].flatten(), as_tuple=False).flatten()
        layer, _, family = module_stratum(name)
        for k in ('ce_mean', 'ce_se', 'ce_standardized', 'ce_margin', 'kl_mean', 'kl_se', 'kl_standardized', 'kl_margin', 'combined_margin'):
            cols[k].append(stats[name][k][ix].numpy())
        cols['module_index'].append(np.full(ix.numel(), mi, np.int32)); cols['tile_index'].append(ix.numpy().astype(np.int64))
        cols['layer'].append(np.full(ix.numel(), layer, np.int16)); cols['family_code'].append(np.full(ix.numel(), 0 if family == 'attention' else 1, np.int8))
    npz = model_dir / f'{model}_selected_score_components.npz'
    np.savez_compressed(npz, module_names=np.asarray(names), **{k: np.concatenate(v) for k, v in cols.items()})
    details = {
        'model': model, 'source_primary_map': str(base_path), 'source_primary_map_sha256': base_sha,
        'source_moments': str(moments_path), 'source_moments_sha256': file_sha(moments_path),
        'anchor_reproduction': {'tile_mismatches': mismatches, 'passed': mismatches == 0},
        'ranking': {'fraction': GROUP_FRACTION, 'target_tiles_each': group_target, 'actual_tiles_each': int(sum(alloc.values())),
                    'per_module_allocation': alloc, 'excluded_strata': excluded, 'groups_disjoint': True},
        'veto': veto_records,
        'interaction': {'attention_tiles': sum(int(x.sum()) for x in attention.values()), 'mlp_tiles': sum(int(x.sum()) for x in mlp.values()),
                        'random_attention_tiles': sum(int(x.sum()) for x in rand_attention.values()),
                        'random_mlp_tiles': sum(int(x.sum()) for x in rand_mlp.values()), 'full_union_exact': True},
        'score_components_npz': {'path': str(npz), 'sha256': file_sha(npz), 'rows': int(sum(selected_counts.values()))},
    }
    (model_dir / 'GROUP_DEFINITIONS.json').write_text(json.dumps(details, indent=1, sort_keys=True) + '\n')
    del mom
    return entries, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--models', default='llama8b,qwen4b,mistral7b')
    args = ap.parse_args()
    protocol_path = Path(args.protocol)
    if file_sha(protocol_path) != args.protocol_sha256:
        raise SystemExit('protocol hash mismatch')
    protocol = json.loads(protocol_path.read_text())
    protocol['protocol_path'] = str(protocol_path)
    launch = json.loads((runtime.run_dir / 'launch_record.json').read_text())
    out_dir = runtime.out_dir('derived_maps')
    all_entries, definitions = [], {}
    for model in args.models.split(','):
        entries, detail = derive(model, protocol, out_dir, launch['source_manifest_sha256'])
        all_entries += entries; definitions[model] = detail
    runtime.atomic_json(out_dir / 'map_manifest.json', all_entries)
    runtime.atomic_json(out_dir / 'group_definitions.json', definitions)
    runtime.atomic_json(runtime.run_dir / 'job_result.json', {
        'protocol_id': protocol['protocol_id'], 'protocol_freeze_sha256': args.protocol_sha256,
        'source': {'model_id': None, 'model_revision': None, 'tokenizer_revision': None, 'model_class': None,
                   'module_manifest_sha256': None, 'source_manifest_sha256': launch['source_manifest_sha256']},
        'environment': runtime.environment(), 'data': {'calibration_manifest_sha256': None, 'evaluation_manifest_sha256': None,
        'token_hashes': {}, 'overlap_audit': None}, 'policies': [
            {'name': e['policy'], 'weight_format': 'FourOverSix/E0M3 tile mix',
             'activation_format': 'four_over_six_rows', 'scale_block': 16,
             'type_block': e['type_block'], 'map_path': e['path'], 'map_sha256': e['sha256'],
             'selected_tiles': e['selected_tiles'], 'total_tiles': e['total_tiles'],
             'map_reloaded_for_evaluation': False} for e in all_entries],
        'results': {'raw_outputs': [str(p) for p in sorted(out_dir.rglob('*')) if p.is_file()],
                    'summary': {'models': args.models.split(','), 'maps': len(all_entries),
                                'anchor_reproductions_passed': all(d['anchor_reproduction']['passed'] for d in definitions.values())},
                    'uncertainty': {}, 'attempted_endpoints': ['map_derivation'], 'missing_endpoints': []},
        'logs': [], 'failures': []})
    print(json.dumps({'maps': len(all_entries), 'models': list(definitions),
                      'anchors': {m: d['anchor_reproduction'] for m, d in definitions.items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
