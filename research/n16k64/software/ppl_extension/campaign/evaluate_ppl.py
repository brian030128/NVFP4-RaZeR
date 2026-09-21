"""Exact-map perplexity with per-window/per-token NLL and token diagnostics against a BF16 teacher.

Plan file (JSON list) entries: {"name", "kind", ["map_path", "map_sha256", "map_policy", "type_block"]}.
Every map is re-read from disk and verified (digest, model/tokenizer revision, module names/shapes,
type block, protocol id, policy name, archived source manifest) before installation.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from campaign import data as D
from campaign import models as MOD
from campaign import policies as P
from campaign import runtime


def calibration_document_hashes(key, tok):
    """Documents used by this model's seed0 and draw1-4 calibration sets (same chaining rule as calibrate.calibration_inputs)."""
    if MOD.REGISTRY[key]['panel'] == 'development':
        base = D.archived_manifest(key)[0]
    else:
        base = D.builder_seed0(tok)[1]
    used = {d['document_sha256'] for dom in ('math', 'code') for d in base[dom]['documents']}
    for s in range(1, 5):
        _, pm = D.keyed_draw(tok, f'draw{s}', set(used))
        used |= {d['document_sha256'] for dom in ('math', 'code') for d in pm[dom]['documents']}
    return used


def windows_for(tok, domain, length, key=None):
    if domain in ('math_eval', 'code_eval'):
        return D.domain_eval_windows(tok, domain[:4], calibration_document_hashes(key, tok), length=length)
    if domain == 'wiki':
        return D.wiki_windows(tok, length)
    if domain == 'c4':
        return D.c4_windows(tok, length)
    if domain.startswith('pg19_'):
        L = {'pg19_4k': 4096, 'pg19_8k': 8192}[domain]
        return D.pg19_frozen_books(tok, L, minimum_books=20)
    raise ValueError(domain)


@torch.no_grad()
def token_stats(logits, labels, teacher_logits=None, chunk=512):
    """Per-token float metrics computed in float32 over chunks of positions."""
    out = dict(nll=[], correct=[], entropy=[], confidence=[], argmax=[])
    if teacher_logits is not None:
        out.update(kl=[], top1_agree=[], teacher_nll=[], teacher_entropy=[])
    for s in range(0, logits.shape[0], chunk):
        lp = logits[s:s + chunk].float().log_softmax(-1)
        y = labels[s:s + chunk].to(lp.device)
        p = lp.exp()
        am = lp.argmax(-1)
        out['nll'].append(-lp.gather(-1, y[:, None])[:, 0])
        out['correct'].append(am == y)
        out['entropy'].append(-(p * lp).sum(-1))
        out['confidence'].append(p.amax(-1))
        out['argmax'].append(am)
        if teacher_logits is not None:
            tlp = teacher_logits[s:s + chunk].to(lp.device).float().log_softmax(-1)
            tp = tlp.exp()
            out['kl'].append((tp * (tlp - lp)).sum(-1))
            out['top1_agree'].append(am == tlp.argmax(-1))
            out['teacher_nll'].append(-tlp.gather(-1, y[:, None])[:, 0])
            out['teacher_entropy'].append(-(tp * tlp).sum(-1))
        del lp, p
    return {k: torch.cat(v).cpu() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MOD.REGISTRY))
    ap.add_argument('--plan', required=True)
    ap.add_argument('--domains', default='wiki,c4')
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--protocol-id', default='aligned-primary')
    ap.add_argument('--matrix-policies-protocol', default=None, help='protocol id stored in the maps (defaults to --protocol-id)')
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--teacher', choices=('instance', 'cache', 'none'), default='instance')
    ap.add_argument('--teacher-windows', type=int, default=32, help='cache mode: first N windows per corpus')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--max-memory-gib', type=float, default=None)
    ap.add_argument('--limit-windows', type=int, default=None, help='smoke tests only')
    ap.add_argument('--no-token-arrays', action='store_true')
    args = ap.parse_args()
    if args.freeze_sha256 != 'SMOKE' and runtime.sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit('PROTOCOL_FREEZE.json digest mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.set_num_threads(8)
    out = runtime.out_dir('ppl')
    spec = MOD.REGISTRY[args.model]
    plan = P.resolve_plan(json.loads(Path(args.plan).read_text()), os.environ.get('CAMPAIGN_ROOT'))
    report = dict(status='running', model=args.model, spec=spec, plan=plan, domains=args.domains.split(','),
                  length=args.length, protocol_id=args.protocol_id, teacher=args.teacher,
                  teacher_windows=(args.teacher_windows if args.teacher == 'cache' else None), attention_backend=args.attn,
                  freeze_sha256=args.freeze_sha256, evaluation={}, installs=[])
    save = lambda: runtime.atomic_json(out / 'ppl_report.json', report)
    save()
    tok = MOD.load_tokenizer(args.model)
    data = {}
    for dom in report['domains']:
        wins, meta = windows_for(tok, dom, args.length if not dom.startswith('pg19') else None, args.model)
        if args.limit_windows:
            wins = wins[:args.limit_windows]
            meta = dict(meta, token_sha256=meta['token_sha256'][:args.limit_windows], limited_to=args.limit_windows)
        data[dom] = (wins, meta)
        runtime.atomic_json(out / f'windows_{dom}.json', meta)
    report['evaluation_manifest_sha256'] = D.manifest_sha256({d: m for d, (w, m) in data.items()})
    report['windows'] = {d: len(w) for d, (w, m) in data.items()}
    save()

    runtime.phase('load_model')
    ngpu = torch.cuda.device_count()
    if args.teacher == 'instance':
        cap = args.max_memory_gib or (40 if ngpu > 1 else None)
        mm = {i: f'{cap}GiB' for i in range(ngpu)} if cap else None
        model, info = MOD.load_model(args.model, attn=args.attn, device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
        teacher, _ = MOD.load_model(args.model, attn=args.attn, device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
    else:
        mm = {i: f'{args.max_memory_gib}GiB' for i in range(ngpu)} if (ngpu > 1 and args.max_memory_gib) else None
        model, info = MOD.load_model(args.model, attn=args.attn, device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
        teacher = None
    cached_teacher = {}
    if args.teacher == 'cache' and plan[0]['kind'] != 'bf16':
        raise SystemExit('cache teacher mode requires bf16 as the first policy')
    modules = MOD.scope(model, args.model)
    mm_sha, _ = MOD.module_manifest(modules)
    report.update(module_manifest_sha256=mm_sha, model_class=type(model).__name__,
                  attn_implementation=model.config._attn_implementation, device_map=getattr(model, 'hf_device_map', None))
    if spec['panel'] == 'development':
        prior = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
        _, entries = MOD.module_manifest(modules)
        if [e['weight_sha256'] for e in entries] != [prior['matrices'][n]['source_sha256'] for n in modules]:
            raise SystemExit('source weights differ from archived hashes')
    save()
    inst = P.Installer(args.model, model, modules, cache=('none' if args.model in ('phi4', 'olmo2_13b', 'qwen27b') else 'cpu'),
                       protocol_id=(args.matrix_policies_protocol or args.protocol_id), campaign_root=os.environ.get('CAMPAIGN_ROOT'))
    dev0 = model.get_input_embeddings().weight.device
    tdev0 = teacher.get_input_embeddings().weight.device if teacher is not None else None
    first_window_check = {}
    for pi, pol in enumerate(plan):
        runtime.phase(f'policy_{pi:02d}')
        t0 = time.time()
        info = inst.install(pol)
        report['installs'].append(info)
        ev = {}
        for dom, (wins, meta) in data.items():
            rows = []
            arrays = {}
            for wi, ids in enumerate(wins):
                if D.sha(ids) != meta['token_sha256'][wi]:
                    raise SystemExit('token hash drift')
                x = ids.to(dev0)
                logits = model(input_ids=x, use_cache=False).logits[0, :-1]
                tlog = None
                if teacher is not None and pol.get('teacher', True):
                    tlog = teacher(input_ids=ids.to(tdev0), use_cache=False).logits[0, :-1]
                elif args.teacher == 'cache' and wi < args.teacher_windows:
                    if pi == 0:
                        cached_teacher[(dom, wi)] = logits.detach().to('cpu', copy=True)
                    if pol.get('teacher', True):
                        tlog = cached_teacher[(dom, wi)]
                st = token_stats(logits, ids[0, 1:], tlog)
                n = st['nll'].numel()
                if not torch.isfinite(st['nll']).all():
                    raise SystemExit(f'non-finite NLL in {pol["name"]} {dom} window {wi}')
                row = dict(window=wi, tokens=n, nll_sum=float(st['nll'].double().sum()), nll_mean=float(st['nll'].double().mean()),
                           token_accuracy=float(st['correct'].double().mean()), entropy_mean=float(st['entropy'].double().mean()),
                           confidence_mean=float(st['confidence'].double().mean()))
                if tlog is not None:
                    row['teacher_diagnostics'] = True
                    row.update(kl_mean=float(st['kl'].double().mean()), top1_agreement=float(st['top1_agree'].double().mean()),
                               teacher_nll_mean=float(st['teacher_nll'].double().mean()))
                rows.append(row)
                if not args.no_token_arrays:
                    for k2, v in st.items():
                        arrays.setdefault(k2, []).append(v.numpy().astype(np.float32 if v.dtype.is_floating_point else np.int32))
                if pi == 0 and wi == 0:
                    first_window_check[dom] = row['nll_mean']
                del logits, tlog, st
            total_nll = sum(r['nll_sum'] for r in rows)
            total_tok = sum(r['tokens'] for r in rows)
            ev[dom] = dict(ppl=math.exp(total_nll / total_tok), mean_nll=total_nll / total_tok, windows=rows, tokens=total_tok,
                           ppl_archived_aggregation=float(torch.exp(torch.tensor([r['nll_mean'] for r in rows], dtype=torch.float32).mean())))
            if arrays:
                fn = out / f'tokens_{pol["name"]}_{dom}.npz'
                np.savez_compressed(fn, **{k2: np.concatenate(v) for k2, v in arrays.items()})
                ev[dom]['token_arrays'] = dict(path=str(fn), sha256=runtime.sha256_file(fn))
            print(f'PPL {pol["name"]} {dom} windows={len(rows)} done', flush=True)
        ev['seconds'] = time.time() - t0
        report['evaluation'][pol['name']] = ev
        save()
    # reinstall the first policy and confirm its first window is bitwise identical
    runtime.phase('reinstall_check')
    info = inst.install(plan[0])
    re = {}
    for dom, (wins, meta) in data.items():
        x = wins[0].to(dev0)
        logits = model(input_ids=x, use_cache=False).logits[0, :-1]
        re[dom] = float(token_stats(logits, wins[0][0, 1:])['nll'].double().mean())
    report['reinstall_check'] = dict(policy=plan[0]['name'], checksum_equal=info['installed_weight_sha256'] == report['installs'][0]['installed_weight_sha256'],
                                     first_window_nll=re, original_first_window_nll=first_window_check,
                                     identical=all(re[d] == first_window_check[d] for d in re))
    inst.remove()
    report['status'] = 'complete'
    save()
    policies = []
    for info in report['installs']:
        policies.append(dict(name=info['name'], weight_format=str(info['weight']), activation_format=str(info['activation']), scale_block=16,
                             weight_scale_multiplier=info.get('weight_scale_multiplier', 1.0),
                             type_block=info.get('type_block'), map_path=info.get('map_path'), map_sha256=info.get('map_sha256'),
                             selected_tiles=info.get('selected_tiles'), total_tiles=info.get('total_tiles'),
                             map_reloaded_for_evaluation=info.get('map_reloaded_for_evaluation')))
    summary = {p: {d: dict(ppl=v['ppl'], mean_nll=v['mean_nll']) for d, v in ev.items() if d != 'seconds'} for p, ev in report['evaluation'].items()}
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id=args.protocol_id, protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
                    model_class=type(model).__name__, module_manifest_sha256=mm_sha, source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend=args.attn, activation_quantizer='per policy (see policies)'),
        data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=report['evaluation_manifest_sha256'],
                  token_hashes={d: m['token_sha256'] for d, (w, m) in data.items()}, overlap_audit=None),
        policies=policies,
        results=dict(raw_outputs=[str(out / 'ppl_report.json')] + [str(p) for p in sorted(out.glob('tokens_*.npz'))],
                     summary=summary, uncertainty=dict(note='paired clustered bootstrap computed in V73 from raw per-window NLL'),
                     attempted_endpoints=[f'{p["name"]}:{d}' for p in plan for d in data], missing_endpoints=[]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
