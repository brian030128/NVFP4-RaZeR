#!/usr/bin/env python3
"""Downstream accuracy (lm-eval 0.4.11) of BF16, fake-quant and native policies on one model.

Suites and primary metrics are the N16K64 campaign's (campaign/evaluate_lmeval.py):
    representative  arc_challenge, piqa, winogrande, boolq      (0-shot)
    full8           + arc_easy, hellaswag, openbookqa, mmlu     (0-shot)
Policies use the same spec strings as eval/ppl.py; native policies run last. Per task it stores
the primary metric, its stderr, and each example's correctness, so native and fake quant of the
same map are compared example by example (paired difference and 2 SE).

    python sm120/eval/lmeval.py --model qwen4b --suite representative \
        --policy fake_n16_k3=fake:map:sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
        --policy native_n16_k3=native:sm120/artifacts/qwen4b_n16_k3 --out sm120/results/lmeval/qwen4b.json
"""
import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from ppl import parse_policy  # noqa: E402

from mixfp4_sm120 import model as NM  # noqa: E402

SUITES = {
    'full8': ['arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa', 'mmlu'],
    'representative': ['arc_challenge', 'piqa', 'winogrande', 'boolq'],
}
PRIMARY = dict(arc_easy='acc_norm', arc_challenge='acc_norm', hellaswag='acc_norm', openbookqa='acc_norm', piqa='acc_norm',
               boolq='acc', winogrande='acc', mmlu='acc')


def per_example(samples, metric):
    """{doc_id: 0/1 correctness under the task's primary metric}."""
    return {s['doc_id']: float(s[metric]) for s in samples if s.get(metric) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(C.MODELS))
    ap.add_argument('--suite', default='representative', choices=sorted(SUITES))
    ap.add_argument('--policy', action='append', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import importlib.metadata as md

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pols = [parse_policy(p) for p in args.policy]
    pols = [p for p in pols if p['kind'] != 'native'] + [p for p in pols if p['kind'] == 'native']
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = dict(model=args.model, spec=C.MODELS[args.model], suite=args.suite, tasks=SUITES[args.suite], batch_size=args.batch_size,
               limit=args.limit, lm_eval=md.version('lm_eval'), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), policies=pols, results={})
    save = lambda: out.write_text(json.dumps(rep, indent=1) + '\n')  # noqa: E731
    model = C.load_model(args.model)
    tok = C.load_tokenizer(args.model)
    mods = C.scope(model, args.model)
    fq = C.FakeQuant(mods)
    samples = {}
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
            fq.install(pol['weight'], masks, tb)
        else:
            if fq.modules:                  # first native policy: free the BF16 Linear weights
                fq.release()
                mods = None
                torch.cuda.empty_cache()
            r = NM.install(model, pol['artifact'], kernel=pol['kernel'], loader=C.MODELS[args.model]['loader'])
            pol['install'] = r.as_dict()
            pol['artifact_map_sha256'] = r.map_sha256
            NM.reset_counters(model)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=SUITES[args.suite], num_fewshot=0, limit=args.limit,
                                      log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                      fewshot_random_seed=0)
        pr = {}
        for task in SUITES[args.suite]:
            m = PRIMARY[task]
            tr = res['results'][task]
            pr[task] = dict(metric=m, value=tr.get(f'{m},none'), stderr=tr.get(f'{m}_stderr,none'))
            if task in res.get('samples', {}):
                samples[(pol['name'], task)] = per_example(res['samples'][task], m)
                pr[task]['n'] = len(samples[(pol['name'], task)])
                # per-example correctness, doc_id order: lets separate runs be paired later
                ids = sorted(samples[(pol['name'], task)])
                pr[task]['doc_ids_sha'] = __import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest()
                pr[task]['correct'] = ''.join('1' if samples[(pol['name'], task)][i] else '0' for i in ids)
        rec = dict(primary=pr, seconds=time.time() - t0)
        if pol['kind'] == 'native':
            cov = NM.coverage(model)
            rec['coverage'] = dict(calls=cov['calls'], tokens=cov['tokens'], native=cov['native'],
                                   native_not_called=cov['native_not_called'], remaining_bf16_linears=cov['remaining_bf16_linears'])
        rep['results'][pol['name']] = rec
        print(pol['name'], {t: round(v['value'], 4) for t, v in pr.items() if v['value'] is not None}, flush=True)
        save()
    comp = {}
    for pol in pols:
        if pol['kind'] != 'native':
            continue
        partner = next((p['name'] for p in pols if p['kind'] == 'fake' and p.get('map_sha256') == pol.get('artifact_map_sha256')
                        and (p['weight'] == 'map') == (pol.get('artifact_map_sha256') is not None)), None)
        if partner is None:
            continue
        c = {}
        for task in SUITES[args.suite]:
            a, b = samples.get((pol['name'], task)), samples.get((partner, task))
            if not a or not b:
                continue
            ids = sorted(set(a) & set(b))
            d = torch.tensor([a[i] - b[i] for i in ids], dtype=torch.float64)
            c[task] = dict(partner=partner, n=len(ids), delta=float(d.mean()), two_se=float(2 * d.std() / math.sqrt(len(ids))),
                           disagreements=int((d != 0).sum()))
        comp[pol['name']] = c
    rep['native_vs_fake'] = comp
    rep['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    save()
    print(json.dumps(comp, indent=1))


if __name__ == '__main__':
    main()
