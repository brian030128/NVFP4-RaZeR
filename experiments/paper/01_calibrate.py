#!/usr/bin/env python3
"""Step 01: calibration. TM-OPT+TC (the final method), per model and unit, without a development set, deterministic.

    PAPER_PYTHON experiments/paper/01_calibrate.py [--models ...] [--units ...] [--train] [--smoke]

For each (model, unit) the output is <out>/maps/<model>_<unit>/map.pt with calibration.json:
- **done** when map.pt exists with its record;
- **reused** when PAPER_MAPS_REF/<reference_dir>/map.pt has the committed sha256 (maps.sha256.json): it is copied;
- **trained** otherwise, or with --train: run_train_map.py with the TM-OPT+TC flags, --no-dev --no-eval, deterministic;
  Qwen3.8-27B with micro-batch 2 x accumulation 4. The record says whether the trained map equals the committed one
  bitwise (Task 1 found it does on this GPU: Llama, Mistral and Phi-4 at every unit, Qwen at 8x64).
--smoke: Llama-3.1-8B 8x64 and 16x64: the reuse check (the maps steps 02-06 then use), and for 16x64 also a 1-epoch
training (not the method's 20) into the smoke directory, to prove the training path. Its map feeds the smoke only
when no committed map is available.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

MANIFEST = json.loads((Path(__file__).resolve().parent / 'maps.sha256.json').read_text())['maps']


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def train(args, model, unit, target, epochs):
    run_dir = target / ('train' if epochs == 20 else f'train_{epochs}epoch')
    if run_dir.exists():                                  # an earlier attempt: keep it aside, labeled by its state
        state = 'previous' if P.complete(run_dir / 'report.json') else 'failed'
        run_dir.rename(run_dir.with_name(run_dir.name + f'.{state}_{int(time.time())}'))
    flags = list(P.TM_OPT_TC)
    flags[flags.index('--epochs') + 1] = str(epochs)
    cmd = [P.PY, 'run_train_map.py', '--model', model, '--data-root', P.DATA[model], *P.DEVIATION.get(model, []), '--unit', unit,
           *flags, *P.TRAIN_BATCH.get(model, []), '--out', run_dir]
    rc = P.run(args.out, f'01_train_{model}_{unit}' + ('' if epochs == 20 else f'_{epochs}epoch'), cmd,
               extra_env=P.ACCURACY_ENV)
    if rc != 0 or not P.complete(run_dir / 'report.json'):
        P.die(f'training failed: {model} {unit} (log: {args.out}/logs)')
    return run_dir


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--train', action='store_true', help='train even when a committed map could be reused')
    args = P.setup(ap.parse_args())
    pairs = [('llama8b', '8x64'), ('llama8b', '16x64')] if args.smoke else [(m, u) for m in args.models for u in args.units]
    for model, unit in pairs:
        key = f'{model}_{unit}'
        target = args.out / 'maps' / key
        target.mkdir(parents=True, exist_ok=True)
        record = target / 'calibration.json'
        want = MANIFEST[key]['map_sha256']
        if (target / 'map.pt').exists() and record.exists() and not args.force and not args.train:
            print(f'{key}: done ({json.loads(record.read_text())["source"]})')
            continue
        ref = P.MAPS_REF / MANIFEST[key]['reference_dir'] / 'map.pt'
        rec = dict(model=model, unit=unit, committed_map_sha256=want, committed_record=MANIFEST[key]['record'])
        ref_ok = ref.exists() and sha(ref) == want
        if ref.exists() and not ref_ok:
            rec['reference_mismatch'] = str(ref)
            print(f'{key}: {ref} does not have the committed sha256; training instead', flush=True)
        if ref_ok and not args.train:
            shutil.copyfile(ref, target / 'map.pt')
            rec.update(source='reused', reused_from=str(ref), map_sha256=sha(target / 'map.pt'))
            P.log(args.out, f'REUSE {key} {ref} sha256={want}')
        else:
            epochs = 1 if args.smoke else 20
            t0 = time.time()
            run_dir = train(args, model, unit, target, epochs)
            got = sha(run_dir / 'map.pt')
            shutil.copyfile(run_dir / 'map.pt', target / 'map.pt')    # a smoke's 1-epoch map only feeds the smoke
            rec.update(source='trained', epochs=epochs, run=str(run_dir), map_sha256=got, equals_committed=got == want,
                       seconds=time.time() - t0)
        if args.smoke and rec['source'] == 'reused' and unit == '16x64':
            # prove the training path as well: one epoch, into the smoke directory (its map is not used)
            t0 = time.time()
            run_dir = train(args, model, unit, target, 1)
            rec['smoke_training'] = dict(epochs=1, run=str(run_dir), seconds=time.time() - t0,
                                         e0m3_tiles=json.loads((run_dir / 'report.json').read_text())['final_e0m3_units'])
        record.write_text(json.dumps(rec, indent=1) + '\n')
        print(f"{key}: {rec['source']} sha256 {rec['map_sha256'][:12]} (committed {want[:12]})", flush=True)


if __name__ == '__main__':
    main()
