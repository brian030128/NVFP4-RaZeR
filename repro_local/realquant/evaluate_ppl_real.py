"""Exact-map perplexity with every scoped Linear executed on the native mixfp4 SM120 kernel.

Mirrors campaign.evaluate_ppl (same plan format and map resolution, same map verification, same
windows and token-hash checks, same per-token NLL and aggregation) but replaces the fake-quant
weight install + activation pre-hook with repro_local/realquant/rq.RealInstaller:

    map policies     type_block (16, 64) -> kernel 'wt_as_A' (weights on A, 16 rows x 64 K)
                     type_block  (8, 64) -> kernel 'b8x64'   (weights on B,  8 cols x 64 K)
    nvfp4 / four_over_six                -> kernel given by the plan entry's "kernel" field
                                            (default wt_as_A), with every format flag clear

Every packed weight is asserted value-equal to the fake-quant weight the campaign installs, and
the first `--check-calls` activation quantizations of every module are asserted value-equal to the
fake-quant activation, so the only difference from the fake path is the GEMM itself (FP4 x FP4
tensor-core products accumulated in FP32, bf16 output) and the per-token rescale after it.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402
from campaign import data as D  # noqa: E402
from campaign import evaluate_ppl as EP  # noqa: E402
from campaign import mapio as MIO  # noqa: E402
from campaign import models as MOD  # noqa: E402
from campaign import policies as P  # noqa: E402
from campaign import quant as Q  # noqa: E402
from campaign import runtime  # noqa: E402
from campaign import tiles as T  # noqa: E402

KERNEL_FOR_BLOCK = {(16, 64): 'wt_as_A', (8, 64): 'b8x64'}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_masks(entry, spec, shapes, protocol_id, campaign_root):
    header, masks, digest = MIO.read_map(entry['map_path'], expected_sha256=entry['map_sha256'])
    MIO.verify_for_model(header, model_id=spec['model_id'], model_revision=spec['revision'],
                         tokenizer_revision=spec['revision'], weight_shapes=shapes,
                         type_block=tuple(entry['type_block']), protocol_id=entry.get('protocol_id') or protocol_id,
                         policy_name=entry['map_policy'],
                         known_source_manifests=P.known_source_manifests(campaign_root) if campaign_root else None,
                         expected_total_tiles=entry.get('expected_total_tiles'))
    return header, masks, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MOD.REGISTRY))
    ap.add_argument('--plan', required=True)
    ap.add_argument('--domains', default='wiki,c4')
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--protocol-id', default='aligned-primary')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--check-calls', type=int, default=2)
    ap.add_argument('--reference-run', default=None,
                    help='fake-quant evaluate_ppl run dir whose windows_*.json token hashes must match')
    ap.add_argument('--limit-windows', type=int, default=None, help='smoke tests only')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.set_num_threads(8)
    out = runtime.out_dir('ppl_real')
    spec = MOD.REGISTRY[args.model]
    campaign_root = os.environ.get('CAMPAIGN_ROOT')
    plan = P.resolve_plan(json.loads(Path(args.plan).read_text()), campaign_root)
    kernels = {}
    libs = {}
    for cfg in ('wt_as_A', 'b8x64'):
        path = rq.LIB_DIR / f'lib{cfg}.so'
        libs[cfg] = dict(path=str(path), sha256=sha256_file(path))
    report = dict(status='running', model=args.model, spec=spec, plan=plan, domains=args.domains.split(','),
                  length=args.length, protocol_id=args.protocol_id, attention_backend=args.attn,
                  execution='native mixfp4 SM120 W4A4 kernel (repro_local/realquant)',
                  kernel_libraries=libs, mixfp4_commit=(rq.LIB_DIR.parent / 'MIXFP4_COMMIT').read_text().strip(),
                  epilogue='D_bf16 = gs_w * acc_fp32 (kernel); y = bf16(D * gs_x[token]) (per-token rescale)',
                  evaluation={}, installs=[])
    save = lambda: runtime.atomic_json(out / 'ppl_report.json', report)
    save()
    tok = MOD.load_tokenizer(args.model)
    data = {}
    for dom in report['domains']:
        wins, meta = EP.windows_for(tok, dom, args.length, args.model)
        if args.reference_run:
            ref = json.loads((Path(args.reference_run) / 'ppl' / f'windows_{dom}.json').read_text())
            if ref['token_sha256'] != meta['token_sha256']:
                raise SystemExit(f'{dom}: windows differ from the reference fake-quant run')
        if args.limit_windows:
            wins = wins[:args.limit_windows]
            meta = dict(meta, token_sha256=meta['token_sha256'][:args.limit_windows], limited_to=args.limit_windows)
        data[dom] = (wins, meta)
        runtime.atomic_json(out / f'windows_{dom}.json', meta)
    report['evaluation_manifest_sha256'] = D.manifest_sha256({d: m for d, (w, m) in data.items()})
    report['windows'] = {d: len(w) for d, (w, m) in data.items()}
    save()

    model, info = MOD.load_model(args.model, attn=args.attn, device_map='cuda')
    modules = MOD.scope(model, args.model)
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    mm_sha, entries = MOD.module_manifest(modules)
    report.update(module_manifest_sha256=mm_sha, model_class=type(model).__name__,
                  attn_implementation=model.config._attn_implementation)
    if spec['panel'] == 'development':
        prior = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
        if [e['weight_sha256'] for e in entries] != [prior['matrices'][n]['source_sha256'] for n in modules]:
            raise SystemExit('source weights differ from archived hashes')
    save()
    inst = rq.RealInstaller(modules, tokens=args.length, check_calls=args.check_calls)
    dev0 = model.get_input_embeddings().weight.device
    first_window = {}
    for pi, pol in enumerate(plan):
        t0 = time.time()
        kind = pol['kind']
        masks, info_map = None, {}
        if kind in ('map', 'hist_map'):
            tb = tuple(pol['type_block'])
            header, masks, digest = load_masks(pol, spec, shapes, args.protocol_id, campaign_root)
            cfg = KERNEL_FOR_BLOCK[tb]
            info_map = dict(map_path=pol['map_path'], map_sha256=digest, selected_tiles=header['totals']['selected_tiles'],
                            total_tiles=header['totals']['total_tiles'], type_block=list(tb))
            kind = 'map'
        elif kind in ('nvfp4', 'four_over_six'):
            cfg = pol.get('kernel', 'wt_as_A')
        else:
            raise SystemExit(f'policy kind {kind} is not supported on the native path')
        if cfg not in kernels:
            kernels[cfg] = rq.Kernel(cfg)
        kern = kernels[cfg]
        tb_kernel = rq.TYPE_BLOCK[cfg]

        def expected(name, w, kind=kind, masks=masks, tb=tb_kernel):
            if kind == 'nvfp4':
                return Q.nvfp4(w)
            base = Q.four_over_six(w)
            if kind == 'map' and bool(masks[name].any()):
                return T.apply_mask(base, Q.e0m3(w), masks[name].to(w.device), tb)
            return base

        inst_info = inst.install(kind, kern, masks, expected)
        if kind == 'map' and inst_info['e0m3_tiles'] != info_map['selected_tiles']:
            raise SystemExit('installed E0M3 tile count differs from the map header')
        rec = dict(name=pol['name'], kind=pol['kind'], kernel=cfg, weights_bitwise_equal_fake=True,
                   install_seconds=time.time() - t0, **inst_info, **info_map)
        report['installs'].append(rec)
        save()
        ev = {}
        for dom, (wins, meta) in data.items():
            rows = []
            for wi, ids in enumerate(wins):
                if D.sha(ids) != meta['token_sha256'][wi]:
                    raise SystemExit('token hash drift')
                with torch.no_grad():
                    logits = model(input_ids=ids.to(dev0), use_cache=False).logits[0, :-1]
                st = EP.token_stats(logits, ids[0, 1:], None)
                if not torch.isfinite(st['nll']).all():
                    raise SystemExit(f'non-finite NLL in {pol["name"]} {dom} window {wi}')
                rows.append(dict(window=wi, tokens=st['nll'].numel(), nll_sum=float(st['nll'].double().sum()),
                                 nll_mean=float(st['nll'].double().mean()),
                                 token_accuracy=float(st['correct'].double().mean())))
                if pi == 0 and wi == 0:
                    first_window[dom] = rows[-1]['nll_mean']
                del logits, st
            total_nll = sum(r['nll_sum'] for r in rows)
            total_tok = sum(r['tokens'] for r in rows)
            ev[dom] = dict(ppl=math.exp(total_nll / total_tok), mean_nll=total_nll / total_tok, windows=rows,
                           tokens=total_tok)
            print(f'REAL {pol["name"]} {dom} ppl={ev[dom]["ppl"]:.6f} windows={len(rows)}', flush=True)
        ev['seconds'] = time.time() - t0
        ev['activation_checks'] = sum(inst.activation_checks().values())
        report['evaluation'][pol['name']] = ev
        save()
    # determinism: reinstall the first policy and recompute its first window
    pol = plan[0]
    kind = 'map' if pol['kind'] in ('map', 'hist_map') else pol['kind']
    masks = load_masks(pol, spec, shapes, args.protocol_id, campaign_root)[1] if kind == 'map' else None
    cfg = KERNEL_FOR_BLOCK[tuple(pol['type_block'])] if kind == 'map' else pol.get('kernel', 'wt_as_A')
    inst.install(kind, kernels[cfg], masks, None)
    again = {}
    for dom, (wins, meta) in data.items():
        with torch.no_grad():
            logits = model(input_ids=wins[0].to(dev0), use_cache=False).logits[0, :-1]
        again[dom] = float(EP.token_stats(logits, wins[0][0, 1:])['nll'].double().mean())
    inst.remove()
    report['reinstall_check'] = dict(policy=pol['name'], first_window_nll=again, original_first_window_nll=first_window,
                                     identical=all(again[d] == first_window[d] for d in again))
    report['status'] = 'complete'
    save()
    print('REAL DONE ' + json.dumps({p: {d: v['ppl'] for d, v in ev.items() if isinstance(v, dict) and 'ppl' in v}
                                     for p, ev in report['evaluation'].items()}), flush=True)


if __name__ == '__main__':
    main()
