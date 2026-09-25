#!/usr/bin/env python3
"""Prefill + token-by-token decode with the KV cache on the native path (checklist section 7).

For a batch of real prompts (WikiText windows), greedy-generates `--gen` tokens with a DynamicCache
(prefill of the prompt, then one token per forward) and compares, at every generated position, the
logits of that incremental decode with the logits of ONE full forward over prompt + generated
tokens without a cache. Both runs execute the same native Linears on the same token rows (the
native path is batch-invariant, tests/test_select.py), so any gap comes from attention over the
cache; the same comparison on the BF16 model gives the reference size of that gap. Also records
native call counts per decode step and that generate() runs end to end.

    python sm120/eval/decode_check.py --model qwen4b --artifact sm120/artifacts/qwen4b_n16_k3 \
        --out sm120/results/decode/qwen4b_n16_k3.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from mixfp4_sm120 import model as NM  # noqa: E402


@torch.no_grad()
def incremental_vs_full(model, prompts, gen):
    from transformers import DynamicCache
    cache = DynamicCache()
    out = model(input_ids=prompts, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    inc_logits, toks = [out.logits[:, -1].float()], [nxt]
    calls = []
    for _ in range(gen - 1):
        before = sum(m.calls for m in NM.native_modules(model).values())
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        calls.append(sum(m.calls for m in NM.native_modules(model).values()) - before)
        nxt = out.logits[:, -1:].argmax(-1)
        inc_logits.append(out.logits[:, -1].float())
        toks.append(nxt)
    seq = torch.cat([prompts] + toks[:-1], dim=1)
    full = model(input_ids=seq, use_cache=False).logits[:, prompts.shape[1] - 1:].float()
    inc = torch.stack(inc_logits, dim=1)
    diff = (inc - full).abs()
    agree = (inc.argmax(-1) == full.argmax(-1)).double().mean()
    kl = (full.log_softmax(-1).exp() * (full.log_softmax(-1) - inc.log_softmax(-1))).sum(-1)
    return dict(max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
                top1_agreement=float(agree), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                native_calls_per_decode_step=sorted(set(calls)) if calls else None,
                generated_tokens=int(gen * prompts.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(C.MODELS))
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--kernel', default='auto')
    ap.add_argument('--map', default=None, help='also run the fake-quant model of this map as a control')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--prompt', type=int, default=256)
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = C.load_tokenizer(args.model)
    wins, _, _ = C.windows(tok, args.model, 'wiki')
    prompts = torch.cat([w[:, :args.prompt] for w in wins[:args.batch]]).cuda()
    model = C.load_model(args.model)
    res = dict(model=args.model, batch=args.batch, prompt=args.prompt, gen=args.gen)
    res['bf16'] = incremental_vs_full(model, prompts, args.gen)
    print('bf16', res['bf16'], flush=True)
    if args.map:
        # control: the fake-quant W4A4 model under the same comparison
        mods = C.scope(model, args.model)
        header, masks, _ = C.read_map_for(args.model, args.map, mods)
        fq = C.FakeQuant(mods)
        fq.install('map', masks, tuple(header['type_block']))
        res['fake'] = incremental_vs_full(model, prompts, args.gen)
        print('fake', res['fake'], flush=True)
        fq.restore()
        del fq
    rep = NM.install(model, args.artifact, kernel=args.kernel, loader=C.MODELS[args.model]['loader'])
    res['install'] = rep.as_dict()
    res['native'] = incremental_vs_full(model, prompts, args.gen)
    print('native', res['native'], flush=True)
    # generate() end to end (HF's own loop, sampling off)
    out = model.generate(prompts[:1], max_new_tokens=16, do_sample=False)
    res['generate_ok'] = bool(out.shape[1] == args.prompt + 16)
    res['generated_text'] = tok.decode(out[0, args.prompt:])
    first = next(iter(NM.native_modules(model).values()))
    res['kernel_set'] = first.kernel_set.describe() if first.kernel_set is not None else rep.kernel_set
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
