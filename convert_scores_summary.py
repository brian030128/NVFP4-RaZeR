"""Convert full per-sequence 8x64 CE/KL score shards into threshold-sweep statistics.

Output matches run_math_code_calibration.py --summary-scores. Each shard is checked:
k=3 CE+KL computed from the summary must equal run_task_reorder_eval.coarse_mask
(the historical raw256 map) and the 8x64 k=3 map from the full table.
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from run_task_reorder_eval import coarse_mask


def main():
    calib, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((calib / 'report.json').read_text())
    counts = dict(raw256=0, fine8x64=0)
    for i, (name, meta) in enumerate(prior['matrices'].items()):
        shard = torch.load(calib / 'scores' / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert shard['name'] == name
        shape = meta['shape']
        summary = dict(name=name, shape=shape)
        bounds256, bounds8 = [], []
        for key in ('ce', 'kl'):
            v = shard[key].double()
            summary[key + '_mean8'] = v.mean(0).float()
            summary[key + '_std8'] = v.std(0, unbiased=True).float()
            bounds8.append(v.mean(0) + 3 * v.std(0, unbiased=True) / math.sqrt(128))
            g = v.reshape(128, shape[0] // 8, shape[1] // 64)
            g = F.pad(g, (0, 0, 0, (-g.shape[1]) % 32)).reshape(128, -1, 32, shape[1] // 64).sum(2)
            summary[key + '_seq256'] = g
            bounds256.append(g.mean(0) + 3 * g.std(0, unbiased=True) / math.sqrt(128))
        raw = coarse_mask(shard['ce'], shard['kl'], shape)
        assert torch.equal(torch.maximum(*bounds256) < 0, raw), name
        counts['raw256'] += int(raw.sum()); counts['fine8x64'] += int((torch.maximum(*bounds8) < 0).sum())
        torch.save(summary, out / f'{i:03d}.pt')
        if (i + 1) % 64 == 0:
            print(f'CONVERTED {i + 1}/{len(prior["matrices"])} {counts}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(dict(source=str(calib), counts=counts,
                                                       matrices=len(prior['matrices'])), indent=2) + '\n')
    print('DONE', counts, flush=True)


if __name__ == '__main__':
    main()
