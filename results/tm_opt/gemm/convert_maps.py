"""#3 (PROTOCOL_ITEMS.md): our tile maps (run_multiround.py / run_train_map.py map.pt) as MIXFP4MAP/1 files for the
SM120 exporter. The tiles are unchanged; the header carries the model id and revision the SM120 code pins, and the
sha256 of the source map and of the calibration record. A sidecar .provenance.json names the source run.

python results/tm_opt/gemm/convert_maps.py MODEL OUT_DIR [METHOD-UNIT ...]   (default: mropt/tmopt x 8x64/16x64)
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

MR = Path('/home/dev/n16k64_campaign/mr_variants/runs')
TM = Path('/home/dev/n16k64_campaign/tm_opt/runs')
CALIBRATION = {'llama8b': Path('/home/dev/n16k64_campaign/cost_comparison/data/llama8b/calibration/report.json'),
               'phi4': Path('/home/dev/n16k64_campaign/multimodel/data/phi4/calibration/report.json'),
               'mistral7b': Path('/home/dev/n16k64_campaign/multimodel/data/mistral7b/calibration/report.json')}


def sm120_models():
    spec = importlib.util.spec_from_file_location('sm120_eval_common', SM120 / 'eval' / 'common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODELS


def source(model, method, unit):
    if method == 'mropt':
        return MR / model / f'mropt_{unit}' / 'map.pt'
    return TM / 'g2_tmopt' / 'map.pt' if (model == 'llama8b' and unit == '8x64') else TM / f'tm_{model}_{unit}' / 'map.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    model, out = sys.argv[1], Path(sys.argv[2])
    wanted = sys.argv[3:] or ['mropt-8x64', 'tmopt-8x64', 'mropt-16x64', 'tmopt-16x64']
    out.mkdir(parents=True, exist_ok=True)
    spec = sm120_models()[model]
    prior = json.loads(CALIBRATION[model].read_text())
    shapes = {n: tuple(m['shape']) for n, m in prior['matrices'].items()}
    for item in wanted:
        method, unit = item.split('-')
        rows, cols = (int(v) for v in unit.split('x'))
        src = source(model, method, unit)
        masks = torch.load(src, map_location='cpu', weights_only=True)
        assert list(masks) == list(shapes), 'map modules differ from the calibration record'
        header = mapio.build_header(protocol_id='tm_opt PROTOCOL_ITEMS.md #3 (converted map.pt)', policy=f'{method.upper()} {unit}',
                                    model=dict(model_id=spec['model_id'], revision=spec['revision']), type_block=(rows, cols),
                                    masks=masks, weight_shapes=shapes, source_manifest_sha256=sha(src),
                                    calibration_manifest_sha256=sha(CALIBRATION[model]))
        path = out / f'{model}_{method}_{unit}.mixfp4map'
        path.write_bytes(mapio.serialize(header, masks))
        head, back, _ = mapio.read_map(path)
        assert all(torch.equal(back[n], masks[n]) for n in masks), path
        Path(str(path) + '.provenance.json').write_text(json.dumps(dict(source=str(src), source_sha256=sha(src),
                                                                        totals=head['totals']), indent=1) + '\n')
        print(path, json.dumps(head['totals']), flush=True)


if __name__ == '__main__':
    main()
