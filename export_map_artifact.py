"""B2: a trained tile map (run_train_map.py / run_multiround.py map.pt) -> an SM120 deployment artifact.

    python export_map_artifact.py --model llama8b --data-root DATA_ROOT --map RUN/map.pt --unit 8x64 --out ARTIFACT
    python export_map_artifact.py --model llama8b --data-root DATA_ROOT --kind four_over_six --out ARTIFACT   # baselines:
                                                                                               # four_over_six | nvfp4
Steps, each checked:
1. The map (--map, --unit): one bool grid per quantized Linear, E0M3 where True, in the calibration record's module
   order, of shape (ceil(rows / unit rows), cols / 64). It is written as a MIXFP4MAP/1 file (ARTIFACT.mixfp4map, with a
   .provenance.json naming the source) for sm120:
     8x64    type block (8, 64)   -> kernel n8k64_wB   (weights on operand B)
     16x64   type block (16, 64)  -> kernel n16k64_wA  (weights on operand A)
     256x64  runs as 16x64 granules on n16k64_wA: every 256-row tile becomes its 16x64 granules, and the last tile of a
             matrix whose rows are not a multiple of 256 keeps only its real rows. Checked: expanded to rows, the
             16x64 map equals the 256x64 map, so the weights are unchanged.
2. The weights: every scoped Linear of the pinned model (sm120/eval/common.py MODELS) must have the calibration
   record's sha256.
3. The candidates: the calibration's (run_train_map.py: FourOverSix E2M1 = quant_nvfp4_4over6(w, 4, 16); E0M3 alpha=1 =
   quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')) must equal sm120's fake-quant candidates
   (mixfp4_sm120.artifact.fake_quant_weight) bitwise, per module; for --kind nvfp4, quant_nvfp4(w, 4, 16).
4. The artifact: mixfp4_sm120.artifact.export (sm120/eval/export_artifact.py's exporter), which checks every packed
   weight bit for bit against the fake-quant weight the map defines.
5. --ownership (B3): sm120/eval/ownership_check.py on the artifact (map artifacts), or its per-module check with an
   all-E2M1 map on the stock kernel (baseline artifacts): every weight element's hardware-decoded value and executed
   format are compared with the stored codes and the map. Written to ARTIFACT/ownership.json.

--scales STATE (results/scale_additivity/PROTOCOL.md): learned per-block scale factors, the state.pt of
run_cost_distill.py --arm scale ({module: factor per 16-element block}).
- The weight is quantize/learned_scale.deployed: scale = e4m3(clamp(f * pre)) and round to nearest, E2M1 on the E2M1
  blocks and E0M3 on the map's tiles.
- --scales-apply e2m1 (default): the factors apply to the E2M1 blocks; E0M3 tiles keep the E0M3 alpha=1 candidate
  (f = 1). This is SCALE→OURS, and SCALE itself without --map.
- --scales-apply all: every block's factor is used (OURS→SCALE).
- The packing is mixfp4_sm120.artifact.export's, with its pack_module replaced. The codes and UE4M3 bytes (bit 7 =
  E0M3 flag) come from the learned scales, and numerics.decode_fake_order must equal the learned fake-quant weight
  (value-equal, as the exporter's own check: -0 packs as +0).
- Step 3 then also checks that the learned parametrization at f = 1 equals the standard candidates on every module.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
SM120 = REPO / 'sm120'
sys.path.insert(0, str(SM120))
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from quantize import learned_scale as L  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6  # noqa: E402
from run_conditional_format import sha  # noqa: E402
from run_multiround import data_paths  # noqa: E402

UNITS = {'8x64': (8, 64), '16x64': (16, 64), '256x64': (256, 64)}
KERNEL = {(8, 64): 'n8k64_wB', (16, 64): 'n16k64_wA'}


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row_mask(mask, rows, height):
    return mask.repeat_interleave(rows, 0)[:height]


def convert(masks, unit, shapes):
    """(masks at the kernel's type block, type block, expansion record) for a map at `unit`."""
    rows, cols = UNITS[unit]
    assert list(masks) == list(shapes), 'map modules differ from the calibration record'
    for n, m in masks.items():
        o, k = shapes[n]
        assert m.dtype == torch.bool and tuple(m.shape) == (-(-o // rows), k // cols), (n, tuple(m.shape), unit)
    if rows != 256:
        return masks, (rows, cols), None
    fine = {n: row_mask(m, 16, -(-shapes[n][0] // 16)) for n, m in masks.items()}
    for n in fine:
        assert torch.equal(row_mask(fine[n], 16, shapes[n][0]), row_mask(masks[n], 256, shapes[n][0])), n
    expansion = dict(from_unit='256x64', to_unit='16x64', source_e0m3_tiles=sum(int(m.sum()) for m in masks.values()),
                     e0m3_weights=sum(int(row_mask(m, 256, shapes[n][0]).sum()) * cols for n, m in masks.items()),
                     check='expanded to rows, the 16x64 map equals the 256x64 map on every matrix')
    return fine, (16, cols), expansion


@torch.no_grad()
def check_candidates(mods, kind, type_block):
    """The calibration's candidates equal sm120's fake-quant candidates bitwise, per module."""
    bad = []
    for n, lin in mods.items():
        w = lin.weight
        if kind == 'nvfp4':
            ok = torch.equal(quant_nvfp4(w, 4, 16), A.fake_quant_weight(w, 'nvfp4'))
        else:
            ok = torch.equal(quant_nvfp4_4over6(w, 4, 16), A.fake_quant_weight(w, 'four_over_six'))
            if type_block is not None:
                o, k = w.shape
                everywhere = torch.ones(-(-o // type_block[0]), k // type_block[1], dtype=torch.bool)
                ok = ok and torch.equal(quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'),
                                        A.fake_quant_weight(w, 'map', everywhere, type_block))
        if not ok:
            bad.append(n)
    return dict(modules=len(mods), mismatching_modules=bad)


def ownership(out, map_file):
    """B3 on one artifact; returns the summary written to ARTIFACT/ownership.json."""
    path = out / 'ownership.json'
    if map_file is not None:
        rc = subprocess.run([sys.executable, str(SM120 / 'eval' / 'ownership_check.py'), '--artifact', str(out),
                             '--map', str(map_file), '--out', str(path)]).returncode
        summary = json.loads(path.read_text())['summary']
        summary['returncode'] = rc
        return summary
    # an E2M1-only artifact: the same per-module check, every tile E2M1, on the stock weights-on-A kernel
    from mixfp4_sm120.lib import Kernel
    oc = load_path('ownership_check', SM120 / 'eval' / 'ownership_check.py')
    kern = Kernel.load('stock_wA')
    t0 = time.time()
    meta, weights = A.load(out)
    rows = {}
    for name, pw in weights.items():
        n, k = pw.shape
        pw16 = dataclasses.replace(pw, type_block=(16, 64))
        rows[name] = oc.check_module(kern, pw16, torch.zeros(-(-n // 16), k // 64, dtype=torch.bool))
    summary = dict(modules=len(rows), all_exact=all(r['exact'] for r in rows.values()),
                   format_mismatches=sum(r['format_mismatches'] for r in rows.values()),
                   informative_elements=sum(r['informative_elements'] for r in rows.values()),
                   e0m3_elements_observed=sum(r['e0m3_elements_observed'] for r in rows.values()),
                   failing_modules=sorted(n for n, r in rows.items() if not r['exact'] or r['format_mismatches']),
                   kernel=kern.cfg.name, kernel_sha256=kern.sha256, artifact_weights_sha256=meta['weights_sha256'],
                   check='sm120/eval/ownership_check.check_module with an all-E2M1 map (E2M1-only artifact)',
                   seconds=round(time.time() - t0, 1), returncode=0 if not any(
                       not r['exact'] or r['format_mismatches'] for r in rows.values()) else 1)
    path.write_text(json.dumps(dict(summary=summary, modules=rows), indent=1) + '\n')
    return summary


@torch.no_grad()
def learned_export(args, mods, record, map_file):
    """Replace artifact.pack_module by a packing of the learned scales; check the parametrization at f = 1."""
    state = torch.load(args.scales, map_location='cpu', weights_only=True)
    assert list(state) == list(mods), 'scale state modules differ from the model scope'
    masks, type_block = {}, None
    if map_file is not None:
        header, masks, _ = mapio.read_map(map_file)
        type_block = tuple(header['type_block'])
    learned, init_mismatch = {}, []
    counts = dict(blocks=0, blocks_scale_moved=0, e0m3_blocks=0, e0m3_blocks_scale_moved=0)
    for n, lin in mods.items():
        w = lin.weight
        flags = L.tile_flags(masks[n].to(w.device), *type_block, w.shape) if masks else None
        gs, pre = L.block_pre(w, flags)
        f = state[n].to(w.device).reshape(-1).float()
        if args.scales_apply == 'e2m1' and flags is not None:
            f = torch.where(flags.reshape(-1), 1.0, f)
        one = L.deployed(w, torch.ones_like(pre), pre, gs, flags)[0]
        standard = A.fake_quant_weight(w, 'map' if flags is not None else 'four_over_six', masks.get(n), type_block)
        if not torch.equal(one, standard):
            init_mismatch.append(n)
        weight, codes, scales = L.deployed(w, f, pre, gs, flags)
        moved = scales != L.e4m3_scales(torch.ones_like(pre), pre)
        counts['blocks'] += moved.numel()
        counts['blocks_scale_moved'] += int(moved.sum())
        if flags is not None:
            counts['e0m3_blocks'] += int(flags.sum())
            counts['e0m3_blocks_scale_moved'] += int((moved & flags.reshape(-1)).sum())
        if flags is None:
            nib, sbytes = N.e2m1_nibbles(codes), N.scale_bytes(scales.reshape(w.shape[0], -1))
        else:
            per_elem = flags.repeat_interleave(16, 1)
            nib = torch.where(per_elem, N.e0m3_nibbles(torch.where(per_elem, codes, 0.)),
                              N.e2m1_nibbles(torch.where(per_elem, 0., codes)))
            sbytes = N.scale_bytes(scales.reshape(w.shape[0], -1), flags)
        learned[n] = (nib, sbytes, gs.float(), weight)
    assert not init_mismatch, f'the learned parametrization at f = 1 differs from the standard candidates: {init_mismatch[:3]}'
    record['learned_scales'] = dict(state=str(args.scales), state_sha256=file_sha(args.scales), apply=args.scales_apply,
                                    init_equals_standard_candidates=True,
                                    factors_changed=int(sum(int((state[n].reshape(-1) != 1).sum()) for n in state)), **counts)

    def pack_module(name, weight, bias, kind, mask=None, type_block=None, expected=None):
        nib, sbytes, gs, want = learned[name]
        pw = A.PackedWeight(name, N.pack_nibbles(nib), sbytes, float(gs),
                            None if bias is None else bias.detach().to(torch.bfloat16).contiguous(),
                            tuple(type_block) if type_block else None, int(mask.sum()) if mask is not None else 0)
        got = N.decode_fake_order(nib, sbytes, gs)
        if not torch.equal(got, want):
            raise A.ArtifactError(f'{name}: packed learned-scale weight differs from its fake-quant weight in '
                                  f'{int((got != want).sum())} elements')
        return pw

    A.pack_module = pack_module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', type=Path, required=True, help='the calibration data (prepare_*_data.py)')
    ap.add_argument('--map', type=Path, default=None, help='map.pt of run_train_map.py / run_multiround.py')
    ap.add_argument('--unit', choices=tuple(UNITS), default=None)
    ap.add_argument('--kind', choices=('map', 'four_over_six', 'nvfp4'), default='map')
    ap.add_argument('--policy-name', default='TM-OPT+TC', help='recorded in the MIXFP4MAP header')
    ap.add_argument('--ownership', action='store_true', help='B3: run the ownership check on the artifact')
    ap.add_argument('--scales', type=Path, default=None, help='learned block-scale factors (run_cost_distill.py state.pt)')
    ap.add_argument('--scales-apply', choices=('e2m1', 'all'), default='e2m1',
                    help='e2m1: factors on the E2M1 blocks only (E0M3 tiles at alpha=1); all: on every block')
    ap.add_argument('--out', type=Path, required=True, help='artifact directory (must not exist or be empty)')
    args = ap.parse_args()
    if args.scales is not None and args.kind == 'nvfp4':
        raise SystemExit('--scales starts from FourOverSix scales: use --kind four_over_six (E2M1 only) or --kind map')
    assert (args.kind == 'map') == (args.map is not None and args.unit is not None), '--map and --unit go with --kind map'
    t0 = time.time()
    C = load_path('sm120_eval_common', SM120 / 'eval' / 'common.py')
    spec = C.MODELS[args.model]
    calibration, _ = data_paths(args.model, args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    shapes = {n: tuple(m['shape']) for n, m in prior['matrices'].items()}
    record = dict(model=args.model, model_id=spec['model_id'], revision=spec['revision'], kind=args.kind,
                  calibration_record=str(calibration / 'report.json'), calibration_sha256=file_sha(calibration / 'report.json'))
    map_file = None
    if args.kind == 'map':
        masks, type_block, expansion = convert(torch.load(args.map, map_location='cpu', weights_only=True), args.unit, shapes)
        header = mapio.build_header(protocol_id='export_map_artifact.py (B2)', policy=f'{args.policy_name} {args.unit}' +
                                    (' as 16x64 granules' if expansion else ''),
                                    model=dict(model_id=spec['model_id'], revision=spec['revision']), type_block=type_block,
                                    masks=masks, weight_shapes=shapes, source_manifest_sha256=file_sha(args.map),
                                    calibration_manifest_sha256=record['calibration_sha256'])
        if expansion:
            assert header['totals']['selected_weights'] == expansion['e0m3_weights'], 'E0M3 area changed'
        map_file = args.out.with_name(args.out.name + '.mixfp4map')
        digest, _ = mapio.write_map(map_file, header, masks, provenance=dict(source=str(args.map.resolve()), unit=args.unit,
                                                                             expansion=expansion))
        record.update(map=str(args.map), unit=args.unit, map_file=str(map_file), map_sha256=digest, type_block=list(type_block),
                      kernel=KERNEL[type_block], expansion=expansion, totals=header['totals'])
        print('MAP', json.dumps({k: record[k] for k in ('map_file', 'type_block', 'kernel', 'totals', 'expansion')}), flush=True)
    model = C.load_model(args.model)
    mods = C.scope(model, args.model)
    assert list(mods) == list(shapes), 'model scope differs from the calibration record'
    wrong = [n for n, m in mods.items() if sha(m.weight) != prior['matrices'][n]['source_sha256']]
    assert not wrong, f'weights differ from the calibration record: {wrong[:3]}'
    record['weights_equal_calibration_record'] = True
    record['candidates'] = check_candidates(mods, args.kind, record.get('type_block') and tuple(record['type_block']))
    assert not record['candidates']['mismatching_modules'], record['candidates']
    print('CANDIDATES equal sm120 fake-quant candidates on all', record['candidates']['modules'], 'modules', flush=True)
    info = dict(model_id=spec['model_id'], revision=spec['revision'], model_class=type(model).__name__, key=args.model)
    if args.scales is not None:
        learned_export(args, mods, record, map_file)
        meta = A.export(args.out, mods, kind=args.kind, map_path=map_file, model_info=info, verify_fake=False,
                        note=dict(exporter='export_map_artifact.py (B2) --scales', record=record))
        meta['numerics'].update(weight_baseline=f'learned per-block scales ({args.scales.name}, apply={args.scales_apply}; '
                                                'quantize/learned_scale.deployed)',
                                verified_against_fake_quant='learned-scale fake-quant weight, value-equal, per module')
    else:
        meta = A.export(args.out, mods, kind=args.kind, map_path=map_file, model_info=info, verify_fake=True,
                        note=dict(exporter='export_map_artifact.py (B2)', record=record))
    meta['note']['export_seconds'] = round(time.time() - t0, 1)
    Path(args.out, 'artifact.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    del model, mods
    torch.cuda.empty_cache()
    s = meta['sizes_bytes']
    print('ARTIFACT', json.dumps(dict(out=str(args.out), modules=len(meta['modules']), e0m3_tiles=sum(m['e0m3_tiles'] for m in meta['modules']),
                                      packed_GB=s['packed'] / 1e9, weights_sha256=meta['weights_sha256'])), flush=True)
    if args.ownership:
        summary = ownership(args.out, map_file)
        print('OWNERSHIP', json.dumps(summary), flush=True)
        if summary['returncode'] != 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
