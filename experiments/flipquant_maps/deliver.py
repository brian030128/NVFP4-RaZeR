#!/usr/bin/env python3
"""flipquant-maps delivery (results/flipquant_maps/PROTOCOL.md): write a TM-OPT+TC map into ~/flipquant's
flipquant-map/1 format and verify it against their quantizable_linears on a meta-device model.

    python experiments/flipquant_maps/deliver.py --key qwen3-1.7b --unit 8x64 --run RUN_DIR --revision REV --out FILE
    python experiments/flipquant_maps/deliver.py --key phi4-14b --unit 8x64 --reuse PAPER_MAP_DIR --revision REV --out FILE
    python experiments/flipquant_maps/deliver.py --key qwen3-1.7b --verify FILE [--revision REV]

- --run: a run_train_map.py output directory (map.pt + report.json, status complete). --reuse: a paper map directory
  (/home/dev/n16k64_campaign/paper/maps/<model>_<unit>: map.pt + calibration.json); the map is read (copied), never
  moved, and must have the committed sha256 (experiments/paper/maps.sha256.json).
- The file is written with ~/flipquant's flipquant.maps.save; an existing file is never overwritten.
- Verification, before and after writing: ~/flipquant's maps.load reads it, and its module names and tile shapes equal
  quantizable_linears(model, spec) (models/loader.py) for a meta-device model built from the registry's config at the
  given revision (the registry's class: Qwen3_5ForConditionalGeneration for loader qwen3_5, else
  AutoModelForCausalLM). Every tile tensor must be bool of shape maps.tile_shape(out, in, unit).
- With --run, --commit names the commit the run was launched at; the run's recorded source sha256 must equal the
  files at that commit.
- Every delivered file is appended to results/flipquant_maps/delivery.json (file, sha256, source, E0M3 share).
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers

REPO = Path(__file__).resolve().parents[2]
FLIPQUANT = Path.home() / 'flipquant'
sys.path.insert(0, str(FLIPQUANT))
from flipquant import maps as FM  # noqa: E402
from models import registry as R  # noqa: E402
from models.loader import quantizable_linears  # noqa: E402

PAPER_MANIFEST = REPO / 'experiments' / 'paper' / 'maps.sha256.json'
# results/nodev_cost: the paper maps re-trained with --no-dev --no-eval and found bitwise equal on this GPU
NODEV_EQUAL = {f'{m}_{u}' for m in ('llama8b', 'mistral7b', 'phi4') for u in ('8x64', '16x64', '256x64')} | {'qwen27b_8x64'}
DELIVERY = REPO / 'results' / 'flipquant_maps' / 'delivery.json'
SETTINGS = ('param', 'optimizer', 'lr', 'eps', 'betas', 'schedule', 'init_logit', 'epochs', 'batch', 'accum', 'seed',
            'tm_opt', 'tile_grad_tc', 'no_dev', 'no_eval', 'unit')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*a):
    return subprocess.check_output(['git', *a], cwd=REPO, text=True).strip()


def expected_shapes(spec, revision):
    """{module name: (out_features, in_features)} of ~/flipquant's quantizable_linears on a meta-device model."""
    config = transformers.AutoConfig.from_pretrained(spec.hf_id, revision=revision)
    if spec.loader == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        cls = transformers.AutoModelForCausalLM
    with torch.device('meta'):
        model = cls.from_config(config) if hasattr(cls, 'from_config') else cls._from_config(config)
    return {n: (m.out_features, m.in_features) for n, m in quantizable_linears(model, spec).items()}


def check(tiles, unit, shapes):
    """The problems of a tile dict against the expected module set (empty = ok)."""
    bad = []
    if set(tiles) != set(shapes):
        extra, missing = sorted(set(tiles) - set(shapes)), sorted(set(shapes) - set(tiles))
        bad.append(f'module set differs: {len(extra)} extra (e.g. {extra[:3]}), {len(missing)} missing (e.g. {missing[:3]})')
    for n in sorted(set(tiles) & set(shapes)):
        want = FM.tile_shape(*shapes[n], unit)
        if tiles[n].dtype != torch.bool or tuple(tiles[n].shape) != want:
            bad.append(f'{n}: {tiles[n].dtype} {tuple(tiles[n].shape)}, expected bool {want}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key', required=True, help="~/flipquant registry key (models/registry.py)")
    ap.add_argument('--unit', choices=('8x64', '16x64', '256x64'))
    ap.add_argument('--revision', default=None, help="model revision (default: the registry's pin)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--run', type=Path)
    src.add_argument('--reuse', type=Path)
    src.add_argument('--verify', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--commit', default=None, help='the commit the run was launched at (with --run; its run log)')
    args = ap.parse_args()
    spec = R.get(args.key)
    revision = args.revision or spec.revision
    assert revision, f'{args.key}: the registry pins no revision; pass --revision'
    shapes = expected_shapes(spec, revision)
    if args.verify:
        tiles, unit, meta = FM.load(args.verify)
        bad = check(tiles, unit, shapes)
        print(json.dumps(dict(file=str(args.verify), unit=unit, ok=not bad, problems=bad[:10], **FM.summary(tiles))))
        sys.exit(1 if bad else 0)
    assert args.unit and args.out, '--unit and --out are required to deliver'
    assert not args.out.exists(), f'{args.out} exists; never overwritten'
    if args.run:
        report = json.loads((args.run / 'report.json').read_text())
        assert report['status'] == 'complete', args.run
        a = report['args']
        assert a['unit'] == args.unit and a['epochs'] == 20 and a['tm_opt'] and a['tile_grad_tc'], a
        src_map = args.run / 'map.pt'
        assert sha(src_map) == report['map_sha256'], 'map.pt does not match its report'
        prior = json.loads((Path(a['data_root']) / a['model'] / 'calibration' / 'report.json').read_text())
        assert prior['revision'] == revision, (prior['revision'], revision)
        source = dict(kind='trained', run=str(args.run), report_sha256=sha(args.run / 'report.json'),
                      calibration_record=str(Path(a['data_root']) / a['model'] / 'calibration' / 'report.json'),
                      calibration_record_sha256=sha(Path(a['data_root']) / a['model'] / 'calibration' / 'report.json'))
        calibration_time = dict(wall_seconds=report.get('resources', {}).get('total_seconds'),
                                setup_seconds=report.get('setup_seconds'), training_seconds=report.get('training_seconds'))
        assert args.commit, '--commit is required with --run'
        trainer_commit = git('rev-parse', args.commit)
        # the run's recorded source hashes must be the files at that commit
        for path, digest in report['source_sha256'].items():
            blob = subprocess.check_output(['git', 'show', f'{trainer_commit}:{path}'], cwd=REPO)
            assert hashlib.sha256(blob).hexdigest() == digest, f'{path} differs from {trainer_commit}'
    else:
        rec = json.loads((args.reuse / 'calibration.json').read_text())
        key = f"{rec['model']}_{rec['unit']}"
        want = json.loads(PAPER_MANIFEST.read_text())['maps'][key]
        src_map = args.reuse / 'map.pt'
        assert rec['unit'] == args.unit and sha(src_map) == want['map_sha256'] == rec['map_sha256'], (key, want)
        report = json.loads((REPO / want['record']).read_text())
        a = report['args']
        source = dict(kind='reused paper map (copied, not moved)', path=str(src_map), paper_map_sha256=want['map_sha256'],
                      paper_record=want['record'], paper_record_commit=git('log', '-1', '--format=%H', '--', want['record']),
                      note=('the paper map, calibrated with a development set as monitor only (last-epoch map). '
                            + ('Re-trained with --no-dev --no-eval on this GPU: bitwise equal (results/nodev_cost).'
                               if key in NODEV_EQUAL else
                               'Not re-trained without the monitor; it never touches training (results/nodev_cost: all '
                               '10 re-trained maps bitwise equal).')))
        calibration_time = dict(wall_seconds=report.get('resources', {}).get('total_seconds'),
                                setup_seconds=report.get('setup_seconds'), training_seconds=report.get('training_seconds'),
                                note='the recorded run monitored a development set; without it the paper cost table applies')
        trainer_commit = source['paper_record_commit']
    raw = torch.load(src_map, map_location='cpu', weights_only=True)
    tiles = {n: m.bool() for n, m in raw.items()}
    bad = check(tiles, args.unit, shapes)
    assert not bad, bad[:10]
    summ = FM.summary(tiles)
    meta = dict(
        method='FlipQuant = TM-OPT+TC (the paper\'s calibration; docs/FLIPQUANT_CALIBRATION_zh.md in NVFP4-RaZeR)',
        trainer='NVFP4-RaZeR run_train_map.py --tm-opt --tile-grad-tc',
        repo='NVFP4-RaZeR', branch='flipquant-maps' if args.run else 'tm-opt', commit=trainer_commit,
        settings=dict({k: a.get(k) for k in SETTINGS}, **report.get('settings', {}),
                      fit_set='128 windows x 512 tokens: 64 open-web-math @ fde8ef8 + 64 codeparrot-clean @ 35a59fb',
                      activations='per-token FourOverSix fake quant (STE)', loss='KL to the BF16 teacher',
                      output='last-epoch map (theta > 0 -> E0M3)'),
        model_id=spec.hf_id, revision=revision, registry_key=spec.key, unit=args.unit,
        e0m3_tiles=summ['e0m3_tiles'], tiles=summ['tiles'], e0m3_share=summ['e0m3_fraction'], modules=summ['modules'],
        calibration_time=calibration_time, transformers=report.get('host', {}).get('transformers'),
        torch=report.get('host', {}).get('torch'), gpu=(report.get('host', {}).get('gpus') or [None])[0],
        source_map_sha256=sha(src_map), source=source,
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    FM.save(args.out, tiles, args.unit, meta)
    tiles2, unit2, meta2 = FM.load(args.out)
    bad = check(tiles2, unit2, shapes)
    assert not bad and unit2 == args.unit and all(torch.equal(tiles2[n], tiles[n]) for n in tiles), bad[:10]
    entry = dict(file=str(args.out), sha256=sha(args.out), key=spec.key, unit=args.unit, model_id=spec.hf_id,
                 revision=revision, source=source, e0m3_share=summ['e0m3_fraction'], modules=summ['modules'],
                 verified='maps.load + quantizable_linears (meta device): module set and tile shapes equal',
                 written_utc=meta['written_utc'])
    log = json.loads(DELIVERY.read_text()) if DELIVERY.exists() else []
    log.append(entry)
    DELIVERY.write_text(json.dumps(log, indent=1) + '\n')
    print(json.dumps({k: entry[k] for k in ('file', 'sha256', 'key', 'unit', 'e0m3_share', 'modules')}))


if __name__ == '__main__':
    main()
