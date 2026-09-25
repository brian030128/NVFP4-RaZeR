#!/usr/bin/env python3
"""Export a native deployment artifact from a pinned model and a frozen format map.

    python sm120/eval/export_artifact.py --model qwen4b --map sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
        --out sm120/artifacts/qwen4b_n16_k3
    python sm120/eval/export_artifact.py --model qwen4b --kind four_over_six --out sm120/artifacts/qwen4b_fo6

Every packed weight is checked bit for bit against the fake-quant weight the map defines, every
tag against the map's tiles; the artifact records the map's sha256 and the model revision.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from mixfp4_sm120 import artifact as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(C.MODELS))
    ap.add_argument('--kind', default='map', choices=('map', 'four_over_six', 'nvfp4'))
    ap.add_argument('--map', default=None)
    ap.add_argument('--map-sha256', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    model = C.load_model(args.model)
    mods = C.scope(model, args.model)
    spec = C.MODELS[args.model]
    info = dict(model_id=spec['model_id'], revision=spec['revision'], model_class=type(model).__name__, key=args.model)
    note = None
    if args.kind == 'map':
        if not args.map:
            raise SystemExit('--map is required for --kind map')
        C.read_map_for(args.model, args.map, mods)
        prov = Path(args.map + '.provenance.json')
        note = json.loads(prov.read_text()) if prov.exists() else None
    meta = A.export(args.out, mods, kind=args.kind, map_path=args.map, map_sha256=args.map_sha256, model_info=info,
                    verify_fake=True, note=dict(map_provenance=note, export_seconds=None))
    meta['note']['export_seconds'] = round(time.time() - t0, 1)
    Path(args.out, 'artifact.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    s = meta['sizes_bytes']
    print(json.dumps(dict(out=args.out, modules=len(meta['modules']), e0m3_tiles=sum(m['e0m3_tiles'] for m in meta['modules']),
                          packed_GB=s['packed'] / 1e9, scales_GB=s['scales'] / 1e9, bf16_GB=s['bf16_weights'] / 1e9,
                          weights_sha256=meta['weights_sha256'], seconds=meta['note']['export_seconds']), indent=1))


if __name__ == '__main__':
    main()
