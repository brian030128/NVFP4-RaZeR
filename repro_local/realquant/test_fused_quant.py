"""Bitwise check of fused_quant against the reference packing path on real Llama-3.1-8B activations.

Pre-hooks on all 224 scoped projections of the BF16 model see the real inputs of one WikiText and
one C4 window (2048 tokens each). For every input and both modes (nvfp4_rows, four_over_six_rows) the
fused kernel's packed codes, scale-factor buffer (GEMM layout) and per-token global scales must equal
the reference: rq.act_* -> e2m1_nibbles -> pack_nibbles -> scale_bytes -> place_scales, which the
accuracy runs verified value-for-value against the campaign quantizers.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fused_quant as FQ  # noqa: E402
import rq  # noqa: E402
from campaign import evaluate_ppl as EP  # noqa: E402
from campaign import models as MOD  # noqa: E402


@torch.no_grad()
def main():
    model, _ = MOD.load_model('llama8b', attn='sdpa', device_map='cuda')
    tok = MOD.load_tokenizer('llama8b')
    modules = MOD.scope(model, 'llama8b')
    kern = rq.Kernel('stock')                 # activations as operand A (weights on B)
    stats = {m: dict(inputs=0, elements=0, blocks=0, packed_bytes_diff=0, sf_bytes_diff=0, gs_diff=0)
             for m in FQ.MODES}

    def hook(mod, args):
        x2 = args[0].reshape(-1, args[0].shape[-1]).contiguous()
        t, k = x2.shape
        n = mod.weight.shape[0]
        idx, size = kern.sf_index(0, t, n, k, x2.device)
        for mode in FQ.MODES:
            code, scale, gs = rq.ACT[mode](x2)
            ref_packed = rq.pack_nibbles(rq.e2m1_nibbles(code))
            ref_sf = kern.place_scales(rq.scale_bytes(scale), 0, t, n, k)
            packed, sf, fgs = FQ.quantize(x2, mode, idx, size)
            s = stats[mode]
            s['inputs'] += 1
            s['elements'] += t * k
            s['blocks'] += t * k // 16
            s['packed_bytes_diff'] += int((packed != ref_packed).sum())
            s['sf_bytes_diff'] += int((sf != ref_sf).sum())
            s['gs_diff'] += int((fgs != gs).sum())

    handles = [m.register_forward_pre_hook(hook) for m in modules.values()]
    for dom in ('wiki', 'c4'):
        wins, _ = EP.windows_for(tok, dom, 2048, 'llama8b')
        model(input_ids=wins[0].cuda(), use_cache=False)
    for h in handles:
        h.remove()
    ok = all(s['packed_bytes_diff'] == 0 and s['sf_bytes_diff'] == 0 and s['gs_diff'] == 0 for s in stats.values())
    for mode, s in stats.items():
        print(mode, json.dumps(s))
    print('FUSED QUANT BITWISE', 'PASS' if ok else 'FAIL')
    out = Path(__file__).resolve().parents[1] / 'results/fused_quant_bitwise_llama8b.json'
    out.write_text(json.dumps(dict(stats=stats, bitwise_equal=ok), indent=1))


if __name__ == '__main__':
    main()
