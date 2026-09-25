"""Phase 1 unit tests (results/speedups/PROTOCOL_PHASE1.md): bitwise, over every distinct Linear shape (N, K) of
the four final models (Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B), at 8x512 and 16x512 tokens.

- fused per-token FourOverSix (quantize.fused_fourover6.fourover6_rows) vs quantize_rows;
- fused per-document FourOverSix (fourover6(x, documents)) vs quant_per_document, 512 tokens per document;
- the native evaluator's single-pass epilogue, torch.mul(D, gs, out=bf16), vs (D.float() * gs).to(bf16).
Bitwise means equal int16 bit patterns (signed zeros included). The activations mix a heavy-tailed body, per-channel
scales over four decades, outlier channels, exact zeros, one all-zero row and values that round to (signed) zero.

python repro_local/realquant/test_phase1_speedups.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from quantize.causal_four_over_six import quantize_rows  # noqa: E402
from quantize.fast_act import quant_per_document  # noqa: E402
from quantize.fused_fourover6 import fourover6, fourover6_rows  # noqa: E402

RECORDS = {'llama8b': '/home/dev/n16k64_campaign/cost_comparison/data/llama8b/calibration/report.json',
           'mistral7b': '/home/dev/n16k64_campaign/multimodel/data/mistral7b/calibration/report.json',
           'phi4': '/home/dev/n16k64_campaign/multimodel/data/phi4/calibration/report.json',
           'qwen27b': '/home/dev/n16k64_campaign/multimodel/data/qwen27b/calibration/report.json'}
TOKENS = (8 * 512, 16 * 512)


def activations(t, k, g):
    x = torch.randn(t, k, generator=g, device='cuda') * torch.logspace(-2, 2, k, device='cuda')[torch.randperm(k, generator=g, device='cuda')]
    x = x * (torch.rand(t, 1, generator=g, device='cuda') * 4 + 0.25)             # per-token scale
    x[:, torch.randperm(k, generator=g, device='cuda')[:max(1, k // 512)]] *= 50       # outlier channels
    x[torch.rand(t, k, generator=g, device='cuda') < 0.01] = 0.0                       # exact zeros
    x[torch.rand(t, k, generator=g, device='cuda') < 0.01] *= 1e-6                     # values that round to zero
    x[t // 3] = 0.0                                                                    # an all-zero token row
    return x.bfloat16()


def same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    g = torch.Generator(device='cuda').manual_seed(0)
    shapes = {}
    for model, path in RECORDS.items():
        for name, m in json.loads(Path(path).read_text())['matrices'].items():
            shapes.setdefault(tuple(m['shape']), []).append(f'{model}:{name.rsplit(".", 2)[-2]}.{name.rsplit(".", 1)[-1]}')
    result = dict(distinct_shapes=len(shapes), tokens=list(TOKENS), shapes={}, failures=[])
    for (n, k), users in sorted(shapes.items()):
        row = dict(users=sorted(set(users))[:6], per_token=[], per_document=[], epilogue=[])
        for t in TOKENS:
            x = activations(t, k, g)
            row['per_token'].append(same(fourover6_rows(x), quantize_rows(x)))
            x3 = x.reshape(t // 512, 512, k)
            row['per_document'].append(same(fourover6(x3, documents=t // 512), quant_per_document(x3)))
            d = (torch.randn(t, n, generator=g, device='cuda') * torch.logspace(-3, 3, n, device='cuda')).bfloat16()
            d[torch.rand(t, n, generator=g, device='cuda') < 0.01] = 0.0
            d[torch.rand(t, n, generator=g, device='cuda') < 0.01] = -0.0
            gs = (torch.rand(t, generator=g, device='cuda') * 1e-3 + 1e-6).float()
            two_pass = (d.float() * gs[:, None]).to(torch.bfloat16)
            one_pass = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
            torch.mul(d, gs[:, None], out=one_pass)
            row['epilogue'].append(same(one_pass, two_pass))
            del x, x3, d, two_pass, one_pass
        ok = all(row['per_token']) and all(row['per_document']) and all(row['epilogue'])
        row['passed'] = ok
        if not ok:
            result['failures'].append([n, k])
        result['shapes'][f'{n}x{k}'] = row
        print(f'{n:6d} x {k:6d}  per-token {row["per_token"]}  per-document {row["per_document"]}  '
              f'epilogue {row["epilogue"]}  ({", ".join(row["users"][:3])})', flush=True)
    result['passed'] = not result['failures']
    out.write_text(json.dumps(result, indent=1) + '\n')
    print('PASS' if result['passed'] else f'FAIL {result["failures"]}')
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
