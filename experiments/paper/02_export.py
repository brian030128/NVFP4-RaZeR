#!/usr/bin/env python3
"""Step 02: deployment artifacts. The TM-OPT+TC maps of step 01 and the NVFP4 / FourOverSix baselines, each exported
once and checked for ownership.

    PAPER_PYTHON experiments/paper/02_export.py [--models ...] [--units ...] [--policies nvfp4,fo6,tc] [--smoke]

Per model, into <out>/artifacts/:
- <model>_nvfp4, <model>_fo6: export_map_artifact.py --kind nvfp4 | four_over_six --ownership (checked on stock_wA),
  then ownership_wb.py (the same check on stock_wB, the 8x64 same-placement latency reference);
- <model>_tc_<unit> (+ .mixfp4map): export_map_artifact.py --map <out>/maps/<model>_<unit>/map.pt --unit <unit>
  --ownership, checked on the map's own kernel: n16k64_wA for 16x64 and 256x64 (exported as 16x64 granules),
  n8k64_wB for 8x64.
One artifact serves both placements: its scale layout does not depend on the operand (sf_buffer_size(rows, k)), and
NativeLinear places the scales for the kernel it is installed on.
An artifact is done when artifact.json and its ownership record(s) exist and passed. An unfinished one is moved aside
(<dir>.failed_<time>), and so is a finished one redone with --force (<dir>.previous_<time>). The weight hashes are compared with the Parts 2-3 artifacts (PAPER_ARTIFACTS_REF), recorded
in <out>/artifacts/<name>.export.json; a difference is reported, not an error (a retrained map is expected to match).
--smoke: Llama-3.1-8B; the two baselines and the 8x64 and 16x64 maps (256x64 runs the 16x64 path).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

REF = Path(os.environ.get('PAPER_ARTIFACTS_REF', '/home/dev/n16k64_campaign/deploy_eval/artifacts'))
KIND = {'nvfp4': 'nvfp4', 'fo6': 'four_over_six'}


def passed(path):
    try:
        s = json.loads(Path(path).read_text())['summary']
    except (OSError, ValueError, KeyError):
        return False
    return s.get('all_exact') is True and not s.get('format_mismatches') and s.get('returncode', 0) == 0


def ownership_files(name):
    return ['ownership.json'] + (['ownership_stock_wB.json'] if name in ('nvfp4', 'fo6') else [])


def main():
    ap = P.parser(__doc__)
    args = P.setup(ap.parse_args())
    kinds = (args.policies or 'nvfp4,fo6,tc').split(',')
    models = ['llama8b'] if args.smoke else args.models
    for model in models:
        names = [k for k in ('nvfp4', 'fo6') if k in kinds]
        names += [f'tc_{u}' for u in (['8x64', '16x64'] if args.smoke else args.units) if 'tc' in kinds]
        for name in names:
            art = P.artifact(args.out, model, name)
            done = (art / 'artifact.json').exists() and all(passed(art / f) for f in ownership_files(name))
            if done and not args.force:
                print(f'{model}_{name}: done')
                continue
            if art.exists():                  # an earlier attempt: keep it aside, labeled by its state
                tag = f"{'previous' if done else 'failed'}_{int(time.time())}"
                art.rename(art.with_name(f'{art.name}.{tag}'))
                for extra in (art.with_name(art.name + '.mixfp4map'), art.with_name(art.name + '.mixfp4map.provenance.json')):
                    if extra.exists():
                        extra.rename(extra.with_name(f'{extra.name}.{tag}'))
            art.parent.mkdir(parents=True, exist_ok=True)
            cmd = [P.PY, 'export_map_artifact.py', '--model', model, '--data-root', P.DATA[model]]
            if name.startswith('tc_'):
                unit = name[3:]
                m = args.out / 'maps' / f'{model}_{unit}' / 'map.pt'
                if not m.exists():
                    P.die(f'{m} missing: run step 01 first')
                cmd += ['--map', m, '--unit', unit]
            else:
                cmd += ['--kind', KIND[name]]
            t0 = time.time()
            rc = P.run(args.out, f'02_export_{model}_{name}', [*cmd, '--ownership', '--out', art], extra_env=P.ACCURACY_ENV)
            if rc != 0 or not passed(art / 'ownership.json'):
                P.die(f'export or ownership check failed: {model}_{name} (log: {args.out}/logs)')
            if name in ('nvfp4', 'fo6'):
                rc = P.run(args.out, f'02_ownership_wB_{model}_{name}',
                           [P.PY, Path(__file__).resolve().parent / 'ownership_wb.py', '--artifact', art,
                            '--out', art / 'ownership_stock_wB.json'])
                if rc != 0 or not passed(art / 'ownership_stock_wB.json'):
                    P.die(f'stock_wB ownership check failed: {model}_{name}')
            meta = json.loads((art / 'artifact.json').read_text())
            ref = REF / f'{model}_{name}' / 'artifact.json'
            ref_meta = json.loads(ref.read_text()) if ref.exists() else None
            rec = dict(model=model, artifact=str(art), weights_sha256=meta['weights_sha256'],
                       map_sha256=(meta.get('map') or {}).get('sha256'), e0m3_tiles=sum(m['e0m3_tiles'] for m in meta['modules']),
                       ownership={f: json.loads((art / f).read_text())['summary'] for f in ownership_files(name)},
                       reference=str(ref) if ref_meta else None,
                       equals_reference=None if ref_meta is None else ref_meta['weights_sha256'] == meta['weights_sha256'],
                       seconds=round(time.time() - t0, 1))
            for o in rec['ownership'].values():
                o.pop('failing_modules', None)
            (art.parent / f'{model}_{name}.export.json').write_text(json.dumps(rec, indent=1) + '\n')
            print(f"{model}_{name}: exported, {rec['e0m3_tiles']} E0M3 tiles, equals Parts 2-3 artifact: "
                  f"{rec['equals_reference']}", flush=True)


if __name__ == '__main__':
    main()
