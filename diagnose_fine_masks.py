"""Where do 1x16 task-loss elections land, and how large is the E0M3 change there?"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6


@torch.no_grad()
def main():
    fine_dir = Path(sys.argv[1]); rules = sys.argv[2].split(',')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B', revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
                                                 torch_dtype=torch.bfloat16, device_map='cuda')
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and m is not model.get_output_embeddings()}
    rows = []
    for i, (n, m) in enumerate(modules.items()):
        f = torch.load(fine_dir / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert f['name'] == n
        w = m.weight
        b = quant_nvfp4_4over6(w, 4, 16).float(); a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always').float()
        d = (a - b).reshape(-1, 16).square().sum(-1); e = (b - w.float()).reshape(-1, 16).square().sum(-1)
        row = dict(name=n)
        for rule in rules:
            count = f['shape'][0] * f['shape'][1]
            mask = torch.from_numpy(np.unpackbits(f['packed'][rule].numpy())[:count].astype(bool)).cuda()
            row[rule] = int(mask.sum())
            if mask.any():
                row[rule + '_delta_over_base_err'] = float(d[mask].sum() / e[mask].sum().clamp_min(1e-30))
        rows.append(row)
    rows.sort(key=lambda r: -max(r[x] for x in rules))
    for r in rows[:25]:
        print(json.dumps(r))
    print('totals', {x: sum(r[x] for r in rows) for x in rules})


if __name__ == '__main__':
    main()
