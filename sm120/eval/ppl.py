#!/usr/bin/env python3
"""Perplexity of BF16, fake-quant and native policies on one loaded model, same windows.

    python sm120/eval/ppl.py --model qwen4b \
        --policy bf16=bf16 \
        --policy fake_n16=fake:map:sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
        --policy native_n16=native:sm120/artifacts/qwen4b_n16_k3 \
        --out sm120/results/ppl/qwen4b.json

Policy kinds: `bf16`; `fake:<map|four_over_six|nvfp4>[:<map file>]` (the campaign's fake quant);
`native:<artifact dir>[:<kernel config>]`. Native policies run last (they replace the modules).
Per window it records the NLL sum; per native policy the install report, the per-forward native
call coverage, and a profiler audit of every dense GEMM op launched in one forward (the only BF16
GEMM allowed is the output head). A native policy whose artifact was exported from the same map as
a fake policy is compared with it window by window (paired mean and 2 SE of the NLL difference).
"""
import argparse
import datetime
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402

GEMM_OPS = ('aten::mm', 'aten::addmm', 'aten::bmm', 'aten::matmul', 'aten::linear', 'aten::baddbmm', 'aten::_scaled_mm')


def parse_policy(s):
    name, _, spec = s.partition('=')
    parts = spec.split(':')
    if parts[0] == 'bf16':
        return dict(name=name, kind='bf16')
    if parts[0] == 'fake':
        return dict(name=name, kind='fake', weight=parts[1], map=parts[2] if len(parts) > 2 else None)
    if parts[0] == 'native':
        return dict(name=name, kind='native', artifact=parts[1], kernel=parts[2] if len(parts) > 2 else 'n16k64_wA')
    raise SystemExit(f'bad policy {s!r}')


@torch.no_grad()
def window_nll(model, ids, dev, chunk=512):
    logits = model(input_ids=ids.to(dev), use_cache=False).logits[0, :-1]
    y = ids[0, 1:].to(dev)
    tot = 0.0
    for s in range(0, logits.shape[0], chunk):
        lp = logits[s:s + chunk].float().log_softmax(-1)
        tot += float(-lp.gather(-1, y[s:s + chunk, None])[:, 0].double().sum())
    return tot, logits


@torch.no_grad()
def gemm_audit(model, ids, dev):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        model(input_ids=ids.to(dev), use_cache=False)
    ops = {}
    for e in prof.key_averages(group_by_input_shape=True):
        if e.key in GEMM_OPS:
            ops[f'{e.key} {e.input_shapes}'] = ops.get(f'{e.key} {e.input_shapes}', 0) + e.count
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(C.MODELS))
    ap.add_argument('--policy', action='append', required=True)
    ap.add_argument('--domains', default='wiki,c4')
    ap.add_argument('--limit-windows', type=int, default=None)
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    pols = [parse_policy(p) for p in args.policy]
    pols = [p for p in pols if p['kind'] != 'native'] + [p for p in pols if p['kind'] == 'native']
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(model=args.model, spec=C.MODELS[args.model], policies=pols, domains=args.domains.split(','),
                  limit_windows=args.limit_windows, attn=args.attn, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                  host=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, results={}, native={})
    import transformers
    report['transformers'] = transformers.__version__
    save = lambda: out.write_text(json.dumps(report, indent=1) + '\n')  # noqa: E731
    tok = C.load_tokenizer(args.model)
    data = {}
    for dom in report['domains']:
        wins, hashes, status = C.windows(tok, args.model, dom, check=True)
        if args.limit_windows:
            wins = wins[:args.limit_windows]
        data[dom] = wins
        report.setdefault('windows', {})[dom] = dict(count=len(wins), check=status)
    save()
    model = C.load_model(args.model, attn=args.attn)
    dev = model.get_input_embeddings().weight.device
    mods = C.scope(model, args.model)
    fq = C.FakeQuant(mods)
    maps = {}
    first_logits = {}
    for pol in pols:
        t0 = time.time()
        if pol['kind'] == 'bf16':
            fq.install('bf16')
        elif pol['kind'] == 'fake':
            masks = tb = None
            if pol['weight'] == 'map':
                header, masks, digest = C.read_map_for(args.model, pol['map'], mods)
                tb = tuple(header['type_block'])
                pol['map_sha256'] = digest
                maps[digest] = pol['name']
            pol['installed_weight_sha256'] = fq.install(pol['weight'], masks, tb)
        else:
            fq.remove()
            rep = NM.install(model, pol['artifact'], kernel=pol['kernel'], loader=C.MODELS[args.model]['loader'])
            meta = json.loads(Path(pol['artifact'], 'artifact.json').read_text())
            pol['artifact_map_sha256'] = (meta.get('map') or {}).get('sha256')
            pol['artifact_weight_kind'] = meta['weight_kind']
            report['native'][pol['name']] = dict(install=rep.as_dict(), artifact_weights_sha256=meta['weights_sha256'],
                                                 artifact_sizes=meta['sizes_bytes'])
            NM.reset_counters(model)
        res = {}
        total_calls = total_tokens = 0
        for dom, wins in data.items():
            rows = []
            if pol['kind'] == 'native':
                cov = NM.coverage(model)
                total_calls, total_tokens = total_calls + cov['calls'], total_tokens + cov['tokens']
                NM.reset_counters(model)
            for wi, ids in enumerate(wins):
                nll, logits = window_nll(model, ids, dev)
                if not math.isfinite(nll):
                    raise SystemExit(f'non-finite NLL: {pol["name"]} {dom} window {wi}')
                rows.append(nll)
                if wi == 0:
                    first_logits[(pol['name'], dom)] = logits.float().cpu()
                    if pol['kind'] == 'native':
                        cov = NM.coverage(model)
                        report['native'][pol['name']].setdefault('coverage_first_forward', {})[dom] = cov
                        if cov['native_called'] != cov['native'] or cov['calls'] != cov['native']:
                            raise SystemExit(f'native coverage incomplete: {cov}')
                del logits
            ntok = sum(w.shape[1] - 1 for w in wins)
            res[dom] = dict(ppl=math.exp(sum(rows) / ntok), mean_nll=sum(rows) / ntok, tokens=ntok, window_nll_sum=rows)
            print(f'{pol["name"]} {dom}: ppl {res[dom]["ppl"]:.4f} ({len(rows)} windows, {time.time() - t0:.0f}s)', flush=True)
        res['seconds'] = time.time() - t0
        if pol['kind'] == 'native':
            report['native'][pol['name']]['gemm_audit'] = gemm_audit(model, data[report['domains'][0]][0], dev)
            cov = NM.coverage(model)
            report['native'][pol['name']]['coverage_total'] = dict(
                calls=total_calls + cov['calls'], tokens=total_tokens + cov['tokens'],
                expected_calls=cov['native'] * (sum(len(w) for w in data.values()) + 1),   # + the audit forward
                remaining_bf16_linears=cov['remaining_bf16_linears'])
        report['results'][pol['name']] = res
        save()
    # paired comparisons: native vs the fake policy of the same map, and vs bf16
    comp = {}
    for pol in pols:
        if pol['kind'] != 'native':
            continue
        if pol.get('artifact_map_sha256') is not None:
            partner = maps.get(pol['artifact_map_sha256'])
        else:   # E2M1-only artifact: the fake policy with the same weight quantizer
            partner = next((p['name'] for p in pols if p['kind'] == 'fake' and p['weight'] == pol['artifact_weight_kind']), None)
        if partner is None:
            continue
        c = {}
        for dom in data:
            a = torch.tensor(report['results'][pol['name']][dom]['window_nll_sum'], dtype=torch.float64)
            b = torch.tensor(report['results'][partner][dom]['window_nll_sum'], dtype=torch.float64)
            per_tok = (a - b) / (data[dom][0].shape[1] - 1)
            se = float(per_tok.std() / math.sqrt(len(per_tok))) if len(per_tok) > 1 else float('nan')
            fl, nl = first_logits[(partner, dom)], first_logits[(pol['name'], dom)]
            c[dom] = dict(partner=partner, delta_mean_nll=float(per_tok.mean()), two_se=2 * se,
                          positive=int((per_tok > 0).sum()), negative=int((per_tok < 0).sum()),
                          ppl_ratio=report['results'][pol['name']][dom]['ppl'] / report['results'][partner][dom]['ppl'],
                          first_window_logits_max_abs_diff=float((fl - nl).abs().max()),
                          first_window_top1_agreement=float((fl.argmax(-1) == nl.argmax(-1)).double().mean()))
        comp[pol['name']] = c
    report['native_vs_fake'] = comp
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    save()
    print(json.dumps({p: {d: round(v[d]['ppl'], 4) for d in data} for p, v in report['results'].items()}, indent=1))
    print(json.dumps(comp, indent=1))


if __name__ == '__main__':
    main()
