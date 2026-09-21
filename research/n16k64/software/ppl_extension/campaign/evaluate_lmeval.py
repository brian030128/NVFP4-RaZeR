"""Exact-map downstream accuracy / generation with lm-eval (per-example samples saved).

Plan entries as in evaluate_ppl. Task groups are frozen in PROTOCOL_FREEZE.json; every task that fails to
load or run is recorded as a failure with its traceback and the job exits non-zero (never silently skipped).
"""
import argparse
import gzip
import hashlib
import importlib.metadata as md
import json
import os
import time
import traceback
from pathlib import Path

import torch

from campaign import models as MOD
from campaign import policies as P
from campaign import runtime
from campaign import data as D

SUITES = {
    'full8': [dict(tasks=['arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa', 'mmlu'], num_fewshot=0)],
    'representative': [dict(tasks=['arc_challenge', 'piqa', 'winogrande', 'boolq'], num_fewshot=0)],
    'gsm8k': [dict(tasks=['gsm8k'], num_fewshot=5,
                    gen_kwargs={'do_sample': False,
                                'until': ['Question:', '</s>', '<|im_end|>'],
                                'max_gen_toks': 256})],
}
PRIMARY_METRIC = dict(arc_easy='acc_norm', arc_challenge='acc_norm', hellaswag='acc_norm', openbookqa='acc_norm', piqa='acc_norm',
                      boolq='acc', winogrande='acc', mmlu='acc', gsm8k='exact_match,flexible-extract')


def jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [jsonable(v) for v in x]
        return repr(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MOD.REGISTRY))
    ap.add_argument('--plan', required=True)
    ap.add_argument('--suite', required=True, choices=sorted(SUITES))
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--protocol-id', default='aligned-primary')
    ap.add_argument('--matrix-policies-protocol', default=None)
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--limit', type=int, default=None, help='smoke only')
    ap.add_argument('--max-memory-gib', type=float, default=None)
    args = ap.parse_args()
    if args.freeze_sha256 != 'SMOKE' and runtime.sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit('PROTOCOL_FREEZE.json digest mismatch')
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = runtime.out_dir('lmeval')
    spec = MOD.REGISTRY[args.model]
    plan = P.resolve_plan(json.loads(Path(args.plan).read_text()), os.environ.get('CAMPAIGN_ROOT'))
    rep = dict(status='running', model=args.model, suite=args.suite, suite_spec=SUITES[args.suite], batch_size=args.batch_size,
               lm_eval_version=md.version('lm_eval'), plan=plan, results={}, failures=[], installs=[], limit=args.limit)
    save = lambda: runtime.atomic_json(out / 'lmeval_report.json', rep)
    save()
    runtime.phase('load_model')
    ngpu = torch.cuda.device_count()
    mm = {i: f'{args.max_memory_gib}GiB' for i in range(ngpu)} if (ngpu > 1 and args.max_memory_gib) else None
    model, info = MOD.load_model(args.model, attn='sdpa', device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
    tok = MOD.load_tokenizer(args.model)
    modules = MOD.scope(model, args.model)
    mm_sha, entries = MOD.module_manifest(modules)
    if spec['panel'] == 'development':
        prior = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
        if [e['weight_sha256'] for e in entries] != [prior['matrices'][n]['source_sha256'] for n in modules]:
            raise SystemExit('source weights differ from archived hashes')
    rep.update(module_manifest_sha256=mm_sha, model_class=type(model).__name__, attn_implementation=model.config._attn_implementation)
    save()
    inst = P.Installer(args.model, model, modules, cache=('none' if args.model in ('phi4', 'olmo2_13b', 'qwen27b') else 'cpu'),
                       protocol_id=(args.matrix_policies_protocol or args.protocol_id), campaign_root=os.environ.get('CAMPAIGN_ROOT'))
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    for pi, pol in enumerate(plan):
        runtime.phase(f'policy_{pi:02d}')
        info = inst.install(pol)
        rep['installs'].append(info)
        rep['results'][pol['name']] = {}
        for group in SUITES[args.suite]:
            t0 = time.time()
            try:
                res = lm_eval.simple_evaluate(model=lm, tasks=list(group['tasks']), num_fewshot=group['num_fewshot'], batch_size=args.batch_size,
                                              log_samples=True, limit=args.limit, bootstrap_iters=1000, random_seed=0, numpy_random_seed=1234,
                                              torch_random_seed=1234, fewshot_random_seed=1234, gen_kwargs=group.get('gen_kwargs'))
            except Exception as exc:
                rep['failures'].append(dict(policy=pol['name'], tasks=group['tasks'], error=repr(exc), traceback=traceback.format_exc()[-6000:]))
                save()
                continue
            samples = res.pop('samples', {})
            for task, rows in samples.items():
                fn = out / f'samples_{pol["name"]}_{task}.jsonl.gz'
                h = hashlib.sha256()
                with gzip.open(fn, 'wt') as f:
                    for row in rows:
                        line = json.dumps(jsonable(row), sort_keys=True)
                        h.update(line.encode())
                        f.write(line + '\n')
                rep['results'][pol['name']].setdefault('samples', {})[task] = dict(path=str(fn), rows=len(rows), content_sha256=h.hexdigest(),
                                                                                     file_sha256=runtime.sha256_file(fn))
            rep['results'][pol['name']].setdefault('lm_eval', {})[','.join(group['tasks'])] = jsonable(dict(
                results=res.get('results'), groups=res.get('groups'), versions=res.get('versions'), n_shot=res.get('n-shot'),
                n_samples=res.get('n-samples'), configs=res.get('configs'), higher_is_better=res.get('higher_is_better'), seconds=time.time() - t0))
            save()
            print(f'LMEVAL {pol["name"]} {group["tasks"]} done {time.time() - t0:.0f}s', flush=True)
    inst.remove()
    expected = {t for g in SUITES[args.suite] for t in g['tasks']}
    missing = [f'{p["name"]}:{t}' for p in plan for t in expected
               if not any(t in k.split(',') for k in rep['results'].get(p['name'], {}).get('lm_eval', {}))]
    rep['missing_endpoints'] = missing
    rep['status'] = 'complete' if not rep['failures'] and not missing else 'failed'
    save()
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    policies = [dict(name=i['name'], weight_format=str(i['weight']), activation_format=str(i['activation']), scale_block=16,
                     type_block=i.get('type_block'), map_path=i.get('map_path'), map_sha256=i.get('map_sha256'),
                     selected_tiles=i.get('selected_tiles'), total_tiles=i.get('total_tiles'),
                     map_reloaded_for_evaluation=i.get('map_reloaded_for_evaluation')) for i in rep['installs']]
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id=args.protocol_id, protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
                    model_class=type(model).__name__, module_manifest_sha256=mm_sha, source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend='sdpa', activation_quantizer='per policy'),
        data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=policies,
        results=dict(raw_outputs=[str(out / 'lmeval_report.json')] + [s['path'] for p in rep['results'].values() for s in p.get('samples', {}).values()],
                     summary={p: {k: v.get('results') for k, v in r.get('lm_eval', {}).items()} for p, r in rep['results'].items()},
                     uncertainty=dict(note='lm-eval stderr in summary; paired bootstrap from samples in V73'),
                     attempted_endpoints=[f'{p["name"]}:{t}' for p in plan for t in expected], missing_endpoints=missing),
        logs=[], failures=rep['failures']))
    if rep['status'] != 'complete':
        raise SystemExit(f'lm-eval failures: {len(rep["failures"])} missing: {missing[:5]}')


if __name__ == '__main__':
    main()
