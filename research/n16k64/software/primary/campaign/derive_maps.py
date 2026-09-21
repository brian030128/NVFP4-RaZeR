"""V51/V52 CPU job: k=3 CE-intersect-KL maps for calibration-size and domain subsets from stored prefix moments.

Subsets of the seed0 64 OpenWebMath + 64 CodeParrot pass (sequence order is the archived manifest order):
  mc32   = first 16 math + first 16 code
  mc64   = first 32 math + first 32 code
  math64 = all 64 math          code64 = all 64 code
  (mc128 = the primary map of the calibration run)
"""
import argparse
import json
import os
from pathlib import Path

import torch

from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime
from campaign import tiles as T
from campaign.policies import latest_complete_run

SUBSETS = {'mc32': (('math', 16), ('code', 16)), 'mc64': (('math', 32), ('code', 32)), 'math64': (('math', 64),), 'code64': (('code', 64),)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibration-job', required=True)
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    args = ap.parse_args()
    if runtime.sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit('freeze digest mismatch')
    run = latest_complete_run(os.environ['CAMPAIGN_ROOT'], args.calibration_job)
    crep = json.loads((run / 'calibration' / 'calibration_report.json').read_text())
    mfile = run / 'calibration' / 'moments' / 'moments_subsets.pt'
    if runtime.sha256_file(mfile) != crep['moment_files']['moments_subsets.pt']:
        raise SystemExit('subset moments digest mismatch')
    sub = torch.load(mfile, weights_only=False)
    names, shapes = sub['names'], {n: tuple(s) for n, s in sub['shapes'].items()}
    spec = MOD.REGISTRY[args.model]
    header_model = dict(model_id=spec['model_id'], revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=crep['model_class'])
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    out = runtime.out_dir('derived_maps')
    entries = []
    for sname, parts in SUBSETS.items():
        for ri, (res, tb) in enumerate((('n8', T.N8), ('n16', T.N16))):
            masks = {}
            for n in names:
                acc = None
                for dom, cnt in parts:
                    st = sub[f'{dom}64'][ri][n] if cnt == 64 else sub['snapshots'][f'{dom}{cnt}'][ri][n]
                    m = T.Moments.from_state(st)
                    acc = m if acc is None else acc.add(m)
                masks[n] = T.elect(acc, 3, 'ce_kl').reshape(shapes[n][0] // tb[0], shapes[n][1] // tb[1])
            pol = dict(name=f'{res}_k3_{sname}', rule='ce_kl', k=3, resolution=res, subset=sname,
                       subset_sequences={dom: cnt for dom, cnt in parts}, draw='seed0')
            header = MIO.build_header(protocol_id='aligned-robustness', policy=pol, model=header_model, type_block=tb, masks=masks,
                                      weight_shapes={n: shapes[n] for n in names}, source_manifest_sha256=source_manifest,
                                      calibration_manifest_sha256=crep['calibration_manifest_sha256'])
            digest, path = MIO.write_map(out / f'{args.model}_seed0_{pol["name"]}.mixfp4map', header, masks,
                                         provenance=dict(run_id=os.environ.get('CAMPAIGN_RUN_ID'), calibration_run=run.name, moments_sha256=crep['moment_files']['moments_subsets.pt']))
            entries.append(dict(policy=pol['name'], type_block=list(tb), path=path, sha256=digest, selected_tiles=header['totals']['selected_tiles'],
                                total_tiles=header['totals']['total_tiles'], selected_weights=header['totals']['selected_weights']))
            print(f'MAP {pol["name"]} {header["totals"]["selected_tiles"]}', flush=True)
    runtime.atomic_json(out / 'map_manifest.json', entries)
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='aligned-robustness', protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=crep['model_class'],
                    module_manifest_sha256=crep['module_manifest_sha256'], source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend=None, activation_quantizer=None),
        data=dict(calibration_manifest_sha256=crep['calibration_manifest_sha256'], evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[dict(name=e['policy'], weight_format='FourOverSix/E0M3 tile mix', activation_format='four_over_six_rows', scale_block=16,
                       type_block=e['type_block'], map_path=e['path'], map_sha256=e['sha256'], selected_tiles=e['selected_tiles'],
                       total_tiles=e['total_tiles'], map_reloaded_for_evaluation=None) for e in entries],
        results=dict(raw_outputs=[str(out / 'map_manifest.json')], summary={e['policy']: e['selected_tiles'] for e in entries}, uncertainty={},
                     attempted_endpoints=[e['policy'] for e in entries], missing_endpoints=[]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
