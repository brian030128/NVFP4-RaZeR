"""Diagnose W4A4 KV-cache vs full-forward divergence: semantic bug or amplified kernel noise?

For BF16 and W4A4 (four_over_six_rows), in bfloat16 and float32 compute dtype, compare logits for
positions < P obtained by (a) full forward on 64 tokens, (b) full forward on the 48-token prefix,
(c) 48-token prefill + 16 cached decode steps. (a) vs (b) is a pure causal-prefix comparison without
any cache; if W4A4 diverges there as much as with the cache, the cache path is not the cause.
Also checks that per-token activation quantization of a row never depends on other rows.
"""
import argparse
import json

import torch

from campaign import data as D
from campaign import models as MOD
from campaign import policies as P
from campaign import quant as Q
from campaign import runtime


def stats(a, b):
    a, b = a.float(), b.float()
    return dict(max_abs=float((a - b).abs().max()), mean_abs=float((a - b).abs().mean()),
                argmax_agree=float((a.argmax(-1) == b.argmax(-1)).float().mean()),
                first_pos_gt_1em3=int(((a - b).abs().amax(-1) > 1e-3).float().argmax()) if bool(((a - b).abs().amax(-1) > 1e-3).any()) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen4b')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('diag')
    tok = MOD.load_tokenizer(args.model)
    wiki, _ = D.wiki_windows(tok)
    res = {}
    # row independence of the activation quantizer on real-scale inputs
    x = torch.randn(3, 64, 2560, device='cuda', dtype=torch.bfloat16) * 3
    y = Q.ACTIVATION['four_over_six_rows'](x)
    z = Q.ACTIVATION['four_over_six_rows'](x[:, :48])
    res['row_independence_prefix_equal'] = bool(torch.equal(y[:, :48], z))
    runtime.phase('load_model')
    for dtype in (torch.bfloat16, torch.float32):
        model, _ = MOD.load_model(args.model, attn='sdpa', device_map='cuda', dtype=dtype)
        modules = MOD.scope(model, args.model)
        inst = P.Installer(args.model, model, modules, cache='none')
        for pol in (dict(name='bf16', kind='bf16'), dict(name='four_over_six', kind='four_over_six')):
            inst.install(pol)
            with torch.no_grad():
                for w in range(3):
                    ids = wiki[w][:, :64].cuda()
                    full64 = model(input_ids=ids, use_cache=False).logits[0]
                    full48 = model(input_ids=ids[:, :48], use_cache=False).logits[0]
                    o = model(input_ids=ids[:, :48], use_cache=True)
                    past, inc = o.past_key_values, [o.logits[0]]
                    for j in range(48, 64):
                        o = model(input_ids=ids[:, j:j + 1], past_key_values=past, use_cache=True)
                        past = o.past_key_values
                        inc.append(o.logits[0])
                    inc = torch.cat(inc)
                    key = f'{str(dtype).split(".")[-1]}/{pol["name"]}/window{w}'
                    res[key] = dict(prefix48_vs_full64_first48=stats(full48, full64[:48]),
                                    cache_vs_full64_all64=stats(inc, full64),
                                    cache_prefill_vs_full48=stats(inc[:48], full48),
                                    cache_decode_positions_vs_full64=stats(inc[48:], full64[48:]))
            inst.remove()
        del model, inst
        torch.cuda.empty_cache()
    runtime.atomic_json(out / 'causality_diag.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
