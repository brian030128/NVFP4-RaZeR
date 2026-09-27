"""R2 (PROTOCOL_QR.md): the TM-OPT and TM-OPT+TC maps as MIXFP4MAP/1 files for the SM120 exporter.

8x64 and 16x64 maps are written unchanged. A 256x64 map is written as a 16x64 map, because n16k64_wA executes
16x64 granules: every 256-row tile becomes its 16 granules, and the last tile of a matrix whose row count is not a
multiple of 256 (Qwen3.8-27B's 48-row in_proj_a/b) keeps only the granules of its real rows. The expansion is
checked per matrix: expanded to rows, the 16x64 map equals the 256x64 map, so the exporter's fake-quant check
verifies the same weights the 256x64 map defines. The header carries the SM120 model id and revision, and the
sha256 of the source map and of the calibration record; a sidecar .provenance.json names the source run.

python results/tm_opt/latency/convert_maps.py MODEL OUT_DIR
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

SM120 = Path('/home/dev/n16k64_campaign/sm120_bench/sm120')
sys.path.insert(0, str(SM120))
from mixfp4_sm120 import mapio  # noqa: E402

TM = Path('/home/dev/n16k64_campaign/tm_opt/runs')
CALIBRATION = {'llama8b': Path('/home/dev/n16k64_campaign/cost_comparison/data/llama8b/calibration/report.json'),
               'mistral7b': Path('/home/dev/n16k64_campaign/multimodel/data/mistral7b/calibration/report.json'),
               'phi4': Path('/home/dev/n16k64_campaign/multimodel/data/phi4/calibration/report.json'),
               'qwen27b': Path('/home/dev/n16k64_campaign/multimodel/data/qwen27b/calibration/report.json')}
METHODS = {'tmopt': 'TM-OPT', 'tc': 'TM-OPT+TC'}
UNITS = ('8x64', '16x64', '256x64')


def sm120_models():
    spec = importlib.util.spec_from_file_location('sm120_eval_common', SM120 / 'eval' / 'common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODELS


def source(model, method, unit):
    if model == 'qwen27b':
        return TM / f'q_{method}_qwen27b_{unit}' / 'map.pt'
    if method == 'tmopt':
        return TM / 'g2_tmopt' / 'map.pt' if (model == 'llama8b' and unit == '8x64') else TM / f'tm_{model}_{unit}' / 'map.pt'
    return TM / f'tc_{model}_{unit}' / 'map.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row_mask(mask, rows, height):
    return mask.repeat_interleave(rows, 0)[:height]


def main():
    model, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    spec = sm120_models()[model]
    prior = json.loads(CALIBRATION[model].read_text())
    shapes = {n: tuple(m['shape']) for n, m in prior['matrices'].items()}
    for method, title in METHODS.items():
        for unit in UNITS:
            rows, cols = (int(v) for v in unit.split('x'))
            src = source(model, method, unit)
            masks = torch.load(src, map_location='cpu', weights_only=True)
            assert list(masks) == list(shapes), 'map modules differ from the calibration record'
            expansion = None
            if rows == 256:
                wide = masks
                masks = {n: row_mask(m, 16, -(-shapes[n][0] // 16)) for n, m in wide.items()}
                for n in masks:
                    assert torch.equal(row_mask(masks[n], 16, shapes[n][0]), row_mask(wide[n], 256, shapes[n][0])), n
                expansion = dict(from_unit='256x64', to_unit='16x64', source_e0m3_tiles=int(sum(int(m.sum()) for m in wide.values())),
                                 source_tiles=int(sum(m.numel() for m in wide.values())),
                                 e0m3_weights=int(sum(int(row_mask(m, 256, shapes[n][0]).sum()) * cols for n, m in wide.items())),
                                 check='expanded to rows, the 16x64 map equals the 256x64 map on every matrix')
                rows = 16
            header = mapio.build_header(protocol_id='tm_opt PROTOCOL_QR.md R2 (converted map.pt)',
                                        policy=f'{title} {unit}' + (' as 16x64 granules' if expansion else ''),
                                        model=dict(model_id=spec['model_id'], revision=spec['revision']), type_block=(rows, cols),
                                        masks=masks, weight_shapes=shapes, source_manifest_sha256=sha(src),
                                        calibration_manifest_sha256=sha(CALIBRATION[model]))
            if expansion:
                assert header['totals']['selected_weights'] == expansion['e0m3_weights'], 'E0M3 area changed in the expansion'
            path = out / f'{model}_{method}_{unit}.mixfp4map'
            path.write_bytes(mapio.serialize(header, masks))
            head, back, _ = mapio.read_map(path)
            assert all(torch.equal(back[n], masks[n]) for n in masks), path
            Path(str(path) + '.provenance.json').write_text(json.dumps(dict(source=str(src), source_sha256=sha(src), totals=head['totals'],
                                                                            expansion=expansion), indent=1) + '\n')
            print(path, json.dumps(head['totals']), json.dumps(expansion), flush=True)


if __name__ == '__main__':
    main()
