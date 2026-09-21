"""V42 frozen checksum controls on a real model (PROTOCOL_FREEZE.json#policies.selector_controls):
all-false map == FourOverSix and all-true map == all-E0M3, for N8K64 and N16K64, through the full write -> reload ->
verify -> install path. Compares installed-weight digests and the logits of one 512-token WikiText crop bitwise.
For models evaluated with the CPU candidate cache it also checks that the GPU-recomputation path (cache='none', used for
>=13B models) installs byte-identical weights."""
import argparse
import json
import os

import torch

from campaign import data as D
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import policies as P
from campaign import runtime
from campaign import tiles as T


def _policy_record(i):
    """RESULT_SCHEMA policy entry built from an Installer.install() info dict (same fields as evaluate_ppl/evaluate_lmeval)."""
    return dict(name=i['name'], weight_format=str(i['weight']), activation_format=str(i['activation']), scale_block=16,
                type_block=i.get('type_block'), map_path=i.get('map_path'), map_sha256=i.get('map_sha256'),
                selected_tiles=i.get('selected_tiles'), total_tiles=i.get('total_tiles'),
                map_reloaded_for_evaluation=i.get('map_reloaded_for_evaluation'))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = MOD.REGISTRY[args.model]
    out = runtime.out_dir('checksum')
    tok = MOD.load_tokenizer(args.model)
    wins, meta = D.wiki_windows(tok, 2048)
    x = wins[0][:, :512]
    runtime.phase('load_model')
    model, _ = MOD.load_model(args.model, device_map='cuda')
    modules = MOD.scope(model, args.model)
    shapes = {n: list(m.weight.shape) for n, m in modules.items()}
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    ident = dict(model_id=spec['model_id'], revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=type(model).__name__)
    plans = [dict(name='four_over_six', kind='four_over_six'), dict(name='all_e0m3', kind='all_e0m3')]
    for res, tb in (('n8', T.N8), ('n16', T.N16)):
        for fill in (False, True):
            name = f'{res}_all_{"true" if fill else "false"}'
            grid = {n: torch.full((s[0] // tb[0], s[1] // tb[1]), fill, dtype=torch.bool) for n, s in shapes.items()}
            header = MIO.build_header(protocol_id='aligned-control', policy=dict(name=name, rule=f'constant {fill}', resolution=res), model=ident,
                                      type_block=tb, masks=grid, weight_shapes=shapes, source_manifest_sha256=source_manifest,
                                      calibration_manifest_sha256=None)
            digest, path = MIO.write_map(out / f'{args.model}_{name}.mixfp4map', header, grid, provenance=dict(run_id=os.environ.get('CAMPAIGN_RUN_ID')))
            plans.append(dict(name=name, kind='map', map_path=path, map_sha256=digest, map_policy=name, type_block=list(tb),
                              expected_total_tiles=header['totals']['total_tiles']))
    caches = ['cpu', 'none'] if args.model not in ('phi4', 'olmo2_13b', 'qwen27b') else ['none']
    results = {}
    for cache in caches:
        inst = P.Installer(args.model, model, modules, cache=cache, protocol_id='aligned-control', campaign_root=os.environ.get('CAMPAIGN_ROOT'))
        res = {}
        for pol in plans:
            runtime.phase(f'{cache}_{pol["name"]}')
            info = inst.install(pol)
            logits = model(x.to(model.get_input_embeddings().weight.device)).logits.float().cpu()
            res[pol['name']] = dict(installed_weight_sha256=info['installed_weight_sha256'], selected_tiles=info.get('selected_tiles'), logits=logits, info=info)
        inst.remove()
        for n, m in modules.items():
            m.weight.copy_(inst.pristine[n].to(m.weight.device))
        del inst
        results[cache] = res
    checks = {}
    for cache, res in results.items():
        for res_name in ('n8', 'n16'):
            f, t = res[f'{res_name}_all_false'], res[f'{res_name}_all_true']
            checks[f'{cache}:{res_name}_all_false==four_over_six'] = dict(weights=f['installed_weight_sha256'] == res['four_over_six']['installed_weight_sha256'],
                                                                         logits=bool(torch.equal(f['logits'], res['four_over_six']['logits'])))
            checks[f'{cache}:{res_name}_all_true==all_e0m3'] = dict(weights=t['installed_weight_sha256'] == res['all_e0m3']['installed_weight_sha256'],
                                                                   logits=bool(torch.equal(t['logits'], res['all_e0m3']['logits'])))
    if len(results) == 2:
        for pol in plans:
            a, b = results['cpu'][pol['name']], results['none'][pol['name']]
            checks[f'cache_cpu==cache_none:{pol["name"]}'] = dict(weights=a['installed_weight_sha256'] == b['installed_weight_sha256'],
                                                                   logits=bool(torch.equal(a['logits'], b['logits'])))
    passed = all(v['weights'] and v['logits'] for v in checks.values())
    report = dict(matrix_id='V42', model=args.model, freeze_sha256=args.freeze_sha256, window_token_sha256=D.sha(x), checks=checks, passed=passed,
                  digests={c: {k: v['installed_weight_sha256'] for k, v in r.items()} for c, r in results.items()},
                  maps={p['name']: dict(sha256=p['map_sha256'], total_tiles=p['expected_total_tiles']) for p in plans if p['kind'] == 'map'})
    runtime.atomic_json(out / 'CHECKSUM_CONTROLS.json', report)
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='aligned-control', protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=type(model).__name__,
                    module_manifest_sha256=MOD.module_manifest(modules)[0], source_manifest_sha256=source_manifest),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes=dict(wiki_window0_512=D.sha(x)), overlap_audit=None),
        policies=[_policy_record(results[caches[-1]][p['name']]['info']) for p in plans],
        results=dict(raw_outputs=[str(out / 'CHECKSUM_CONTROLS.json')], summary=dict(passed=passed), uncertainty={}, attempted_endpoints=list(checks),
                     missing_endpoints=[]), logs=[], failures=([] if passed else [k for k, v in checks.items() if not (v['weights'] and v['logits'])])))
    print(json.dumps(dict(passed=passed, checks=checks), indent=1))
    if not passed:
        raise SystemExit('checksum controls failed')


if __name__ == '__main__':
    main()
