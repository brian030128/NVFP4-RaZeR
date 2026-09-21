"""V72 N8/N16 structure and V73 selection-statistics (multiplicity, joint null, sign-flip) from calibration outputs (CPU)."""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from campaign import mapio as MIO
from campaign import runtime
from campaign import stats as S
from campaign import tiles as T
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
CALIB = {'qwen4b': 'V30_calib_qwen4b_seed0', 'llama8b': 'V30_calib_llama8b_seed0', 'qwen27b': 'V30_calib_qwen27b_seed0',
         'mistral7b': 'V61_calib_mistral7b_seed0', 'phi4': 'V61_calib_phi4_seed0', 'olmo2_13b': 'V61_calib_olmo2_13b_seed0'}


def load_maps(run, policies):
    man = {e['policy']: e for e in json.loads((run / 'calibration' / 'map_manifest.json').read_text())}
    out = {}
    for p in policies:
        h, masks, d = MIO.read_map(man[p]['path'], expected_sha256=man[p]['sha256'])
        out[p] = (h, masks)
    return out


def layer_type(name):
    layer = int(name.split('layers.')[1].split('.')[0]) if 'layers.' in name else -1
    return layer, name.split('.')[-1]


def analyze_model(model):
    run = latest_complete_run(CR, CALIB[model])
    crep = json.loads((run / 'calibration' / 'calibration_report.json').read_text())
    mom = torch.load(run / 'calibration' / 'moments' / 'moments_full.pt', weights_only=False)
    names, shapes = mom['names'], {n: tuple(s) for n, s in mom['shapes'].items()}
    maps = load_maps(run, ['n8_k3', 'n16_k3'])
    n8 = {n: maps['n8_k3'][1][n].reshape(-1) for n in names}
    n16 = {n: maps['n16_k3'][1][n].reshape(-1) for n in names}
    structure = S.structure(n8, n16, shapes)
    # cancellation / amplification using float64 moments
    canc = dict(parent_selected_no_child=0, parent_selected_one_child=0, parent_selected_both=0, child_selected_parent_rejected=0,
                both_children_selected_parent_rejected=0)
    heat = defaultdict(lambda: dict(n8=0, n16=0, n8_tiles=0, n16_tiles=0))
    t_parent_gain = []
    for n in names:
        o, k = shapes[n]
        m8 = n8[n].reshape(o // 16, 2, k // 64)
        m16 = n16[n].reshape(o // 16, k // 64)
        c = m8.sum(1)
        canc['parent_selected_no_child'] += int((m16 & (c == 0)).sum())
        canc['parent_selected_one_child'] += int((m16 & (c == 1)).sum())
        canc['parent_selected_both'] += int((m16 & (c == 2)).sum())
        canc['child_selected_parent_rejected'] += int((~m16 & (c >= 1)).sum())
        canc['both_children_selected_parent_rejected'] += int((~m16 & (c == 2)).sum())
        L, ty = layer_type(n)
        heat[(L, ty)]['n8'] += int(n8[n].sum())
        heat[(L, ty)]['n16'] += int(n16[n].sum())
        heat[(L, ty)]['n8_tiles'] += n8[n].numel()
        heat[(L, ty)]['n16_tiles'] += n16[n].numel()
        u8 = T.upper_bound(T.Moments.from_state(mom['n8'][n]), 3).reshape(o // 16, 2, k // 64)
        u16 = T.upper_bound(T.Moments.from_state(mom['n16'][n]), 3).reshape(o // 16, k // 64)
        best_child = u8.min(1).values
        sel = m16 | (c >= 1)
        if sel.any():
            t_parent_gain.append(torch.stack([best_child[sel], u16[sel]], 1))
    tp = torch.cat(t_parent_gain) if t_parent_gain else torch.empty(0, 2)
    structure['cancellation_amplification'] = dict(canc, note='counts over N16 parents; "no child" = amplification by summation; "child selected parent rejected" = cancellation or SE inflation')
    if tp.numel():
        structure['u3_parent_vs_best_child'] = dict(pairs=int(tp.shape[0]), parent_better_fraction=float((tp[:, 1] < tp[:, 0]).double().mean()),
                                                    median_parent_minus_best_child=float((tp[:, 1] - tp[:, 0]).median()))
    structure['heatmap'] = [dict(layer=L, module=ty, **v) for (L, ty), v in sorted(heat.items())]
    structure['per_module'] = None  # kept compact; per-module counts are in heatmap
    # multiplicity (V73)
    st8 = {n: S.tile_statistics(mom['n8'][n]) for n in names}
    st16 = {n: S.tile_statistics(mom['n16'][n]) for n in names}
    mult = dict(n8=S.multiplicity_summary(st8), n16=S.multiplicity_summary(st16))
    # sign flip on the stored stratified sample
    raw = torch.load(run / 'calibration' / 'moments' / 'raw_scores_sample.pt', weights_only=False)
    ce8 = torch.cat([raw['scores'][n]['ce'] for n in names], 1)
    kl8 = torch.cat([raw['scores'][n]['kl'] for n in names], 1)
    ce16 = torch.cat([raw['scores'][n]['ce'].reshape(raw['scores'][n]['ce'].shape[0], -1, 2).double().sum(2) for n in names], 1)
    kl16 = torch.cat([raw['scores'][n]['kl'].reshape(raw['scores'][n]['kl'].shape[0], -1, 2).double().sum(2) for n in names], 1)
    flip = dict(n8=S.sign_flip(ce8, kl8, R=1000), n16=S.sign_flip(ce16, kl16, R=1000),
                sample_rule=raw['rule'], sample_tiles_n16=int(ce16.shape[1]), total_n16=crep['n16_total'])
    # verify sample moments agree with the full moments (sanity)
    return run, crep, structure, mult, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='qwen4b,llama8b,qwen27b,mistral7b,phi4,olmo2_13b')
    args = ap.parse_args()
    out = runtime.out_dir('analysis_selection')
    structure, stats_all = {}, {}
    for m in args.models.split(','):
        try:
            run, crep, st, mult, flip = analyze_model(m)
        except FileNotFoundError as exc:
            structure[m] = dict(status='missing', error=str(exc))
            stats_all[m] = dict(status='missing', error=str(exc))
            continue
        structure[m] = dict(run=run.name, n8_total=crep['n8_total'], n16_total=crep['n16_total'], **st)
        stats_all[m] = dict(run=run.name, multiplicity=mult, sign_flip=flip, sequences=crep['sequences'])
        print(m, json.dumps(dict(n8=st['totals']['n8'], n16=st['totals']['n16'], jac_any=st['jaccard_n16_vs_n8_any'],
                                 bh=mult['n16']['normal']['BH_q0.05'], flip=flip['n16']['estimated_false_discovery_proportion'])), flush=True)
    runtime.atomic_json(out / 'N8_N16_STRUCTURE.json', dict(matrix_id='V72', freeze_sha256=FSHA, models=structure))
    runtime.atomic_json(out / 'SELECTION_STATISTICS.json', dict(matrix_id='V73', freeze_sha256=FSHA, models=stats_all))
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='statistics', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / 'N8_N16_STRUCTURE.json'), str(out / 'SELECTION_STATISTICS.json')], summary={}, uncertainty={},
                                  attempted_endpoints=args.models.split(','),
                                  missing_endpoints=sorted(set(m for m, v in structure.items() if 'run' not in v)
                                                           | set(runtime.collect_missing(out)))),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
