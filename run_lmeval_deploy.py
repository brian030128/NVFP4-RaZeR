"""Task 3: downstream accuracy through lm-eval's HFLM, on BF16, fake (c) and NativeLinear (c) policies.

    python run_lmeval_deploy.py --model llama8b --data-root DATA_ROOT \\
        --evaluate BF16=bf16 --evaluate FourOverSix=native:ARTIFACT_FO6 --evaluate tc-16x64=native:ARTIFACT \\
        --evaluate tc-16x64-fake=fake:map:ARTIFACT.mixfp4map \\
        --tasks arc_easy,arc_challenge,hellaswag,openbookqa,boolq,winogrande,piqa,gsm8k --limit 20 --out DIR

- **Harness:** lm-eval 0.4.11 (the pinned version, sm120/requirements.lock.txt), with the datasets pinned by
  lm_eval_datasets.py. The model loading is run_ppl_deploy.py's, and so are the policies and their kernel choice:
  - bf16: the model as loaded;
  - fake:four_over_six | fake:nvfp4 | fake:map:<.mixfp4map>: fake quant with per-token activations (convention (c));
  - native:<artifact>[:<kernel>]: NativeLinear (c).
  Fake and BF16 policies run first; native policies replace the Linears, so they run last.
- **Task settings:** the lm-eval defaults. The log-likelihood tasks are 0-shot; the primary metric is acc_norm where
  the task defines it, else acc. gsm8k is 5-shot generate_until with greedy decoding, scored by exact_match
  (strict-match and flexible-extract).
- **Recorded per policy and task:**
  - the metrics and their stderr;
  - every example's correctness under the primary metric (paired comparisons);
  - for gsm8k, the generated answers;
  - the wall time, the time inside the model calls (loglikelihood / generate_until, with their request counts) and
    the peak GPU memory.
- **Recorded per native policy:** coverage. Every quantized Linear must run natively, with no fallback. Linears
  outside the quantization scope (Qwen's vision tower; the head is not a scoped Linear) stay BF16 and must never run.
- **--repeat LABEL** evaluates that policy a second time in the same process (a determinism check).
- **--no-bos** passes add_bos_token=False to HFLM: contexts without BOS, lm-eval 0.4.5 / 0.4.9.1's behaviour. 0.4.11's
  default keeps the tokenizer's special tokens (BOS for Llama-3.1 and Mistral).
- **Older lm-eval** (a cross-check): run with its install first on PYTHONPATH; lm_eval_compat is imported for
  versions before 0.4.10 (transformers 5).
"""
import argparse
import datetime
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

import lm_eval_datasets  # noqa: F401  (pins the datasets before lm-eval builds its tasks)
import run_ppl_deploy as D

PRIMARY = dict(arc_easy='acc_norm', arc_challenge='acc_norm', hellaswag='acc_norm', openbookqa='acc_norm', piqa='acc_norm',
               boolq='acc', winogrande='acc', gsm8k='exact_match')
CHANCE = dict(arc_easy=0.25, arc_challenge=0.25, hellaswag=0.25, openbookqa=0.25, piqa=0.5, boolq=0.5, winogrande=0.5, gsm8k=0.0)
GSM8K_FILTERS = ('strict-match', 'flexible-extract')     # gsm8k.yaml's filter_list, in order


def summarize(results, samples):
    out = {}
    for task, metrics in results['results'].items():
        entry = dict(metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'alias'})
        rows = samples.get(task, [])
        if task == 'gsm8k':
            # lm-eval logs one sample per (filter, document), in the task's filter order: strict-match, then
            # flexible-extract. 0.4.11 names the filter; 0.4.5 does not, so the order is used there.
            examples, seen = {}, {}
            for s in rows:
                doc = str(s['doc_id'])
                seen[doc] = seen.get(doc, -1) + 1
                e = examples.setdefault(doc, dict(raw=s['resps'][0][0] if s.get('resps') else None))
                e[s.get('filter') or GSM8K_FILTERS[seen[doc]]] = dict(
                    exact_match=float(s.get('exact_match', float('nan'))),
                    response=s['filtered_resps'][0] if s.get('filtered_resps') else None)
            entry['examples'] = examples
        else:
            key = PRIMARY[task]
            entry['examples'] = {str(s['doc_id']): float(s[key]) for s in rows if key in s}
        out[task] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--evaluate', action='append', required=True, metavar='LABEL=SPEC')
    ap.add_argument('--tasks', default='arc_easy,arc_challenge,hellaswag,openbookqa,boolq,winogrande,piqa,gsm8k')
    ap.add_argument('--limit', type=int, default=None, help='examples per task (smoke tests); default: all')
    ap.add_argument('--gsm8k-limit', type=int, default=None, help='examples for gsm8k (default: --limit)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--repeat', action='append', default=[], metavar='LABEL', help='evaluate this policy twice')
    ap.add_argument('--transformers-deviation', action='store_true')
    ap.add_argument('--no-bos', action='store_true', help='HFLM add_bos_token=False (lm-eval 0.4.5 / 0.4.9.1 behaviour)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if tuple(int(x) for x in md.version('lm_eval').split('.')[:3]) < (0, 4, 10):
        import lm_eval_compat  # noqa: F401  (0.4.5 / 0.4.9.1 under transformers 5)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = D.load_path('sm120_eval_common', D.SM120 / 'eval' / 'common.py')
    policies = [D.parse(s) for s in args.evaluate]
    policies = [p for p in policies if p['kind'] != 'native'] + [p for p in policies if p['kind'] == 'native']
    runs = []
    for p in policies:
        runs.append(p)
        if p['label'] in args.repeat:
            runs.append(dict(p, label=p['label'] + '#repeat'))
    tasks = args.tasks.split(',')
    calibration, _ = D.data_paths(args.model, args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    if transformers.__version__ != prior['transformers_version']:
        assert args.transformers_deviation, (transformers.__version__, prior['transformers_version'])
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', model=args.model, evaluate=args.evaluate, tasks=tasks, limit=args.limit,
                  gsm8k_limit=args.gsm8k_limit, batch_size=args.batch_size, no_bos=args.no_bos, lm_eval_path=lm_eval.__file__,
                  versions=dict(lm_eval=md.version('lm_eval'), transformers=transformers.__version__, torch=torch.__version__,
                                datasets=md.version('datasets')),
                  dataset_pins=lm_eval_datasets.PINS, host=dict(hostname=platform.node(), gpu=torch.cuda.get_device_name(0)),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), evaluations={})
    save = lambda: (args.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')  # noqa: E731
    t0 = time.time()
    model, modules = D.load_for_evaluation(args.model, prior, C)
    scoped = set(modules)          # the quantized Linears; anything else (Qwen's vision tower, the head) stays BF16
    report['load_seconds'] = time.time() - t0
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fq = C.FakeQuant(modules)
    for pol in runs:
        entry = report['evaluations'][pol['label']] = dict(pol)
        t1 = time.time()
        if pol['kind'] == 'native':
            if fq.modules:
                fq.release()
                modules = None
                torch.cuda.empty_cache()
            entry.update(D.install_native(model, pol, args.model, C))
            from mixfp4_sm120 import model as NM
            NM.reset_counters(model)
            # the Linears left in BF16 (outside the scope) must never run during the evaluation
            head = model.get_output_embeddings()
            unscoped_calls = {}
            unscoped_hooks = [m.register_forward_hook(lambda mod, i, o, _n=n: unscoped_calls.__setitem__(_n, unscoped_calls.get(_n, 0) + 1))
                              for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head]
        else:
            entry.update(D.install_fake(fq, pol, args.model, modules, C))
        entry['install_seconds'] = time.time() - t1
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, backend='causal',
                  **(dict(add_bos_token=False) if args.no_bos else {}))
        if 'tokenization' not in report:
            # lm-eval 0.4.11 keeps the tokenizer's default special tokens (add_bos_token=None); 0.4.5 / 0.4.9.1 add none
            probe = lm.tok_encode('Question: 1 + 1 =')
            report['tokenization'] = dict(add_bos_token=lm.add_bos_token, bos_token=tok.bos_token, bos_token_id=tok.bos_token_id,
                                          context_starts_with_bos=tok.bos_token_id is not None and probe[0] == tok.bos_token_id,
                                          probe_ids=probe[:4])
        # the model's share of the time: lm-eval dispatches every request type through getattr(lm, type)
        timing = {}

        def timed(fn, kind):
            def call(requests, *a, **k):
                t = time.time()
                result = fn(requests, *a, **k)
                torch.cuda.synchronize()
                timing[kind + '_seconds'] = timing.get(kind + '_seconds', 0.0) + time.time() - t
                timing[kind + '_requests'] = timing.get(kind + '_requests', 0) + len(requests)
                return result
            return call
        lm.loglikelihood = timed(lm.loglikelihood, 'loglikelihood')
        lm.generate_until = timed(lm.generate_until, 'generate_until')
        entry['tasks'] = {}
        for task in tasks:
            timing.clear()
            limit = args.gsm8k_limit if (task == 'gsm8k' and args.gsm8k_limit is not None) else args.limit
            torch.cuda.reset_peak_memory_stats()
            t2 = time.time()
            # lm-eval's defaults. Every metric here aggregates by mean, whose stderr is the closed form (no bootstrap
            # draws); bootstrap_iters=0 would drop the stderr altogether.
            res = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, log_samples=True, bootstrap_iters=100000,
                                          random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                                          fewshot_random_seed=1234)
            summary = summarize(res, res.get('samples', {}))[task]
            summary.update(seconds=time.time() - t2, examples_evaluated=len(summary['examples']),
                           peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, n_shot=res.get('n-shot', {}).get(task),
                           chance=CHANCE[task], model_timing=dict(timing))
            entry['tasks'][task] = summary
            print(f"LMEVAL {pol['label']} {task} {PRIMARY[task]} {summary['metrics'].get(PRIMARY[task] + ',none', summary['metrics'].get(PRIMARY[task] + ',strict-match'))} "
                  f"({summary['examples_evaluated']} examples, {summary['seconds']:.0f}s)", flush=True)
            save()
        if pol['kind'] == 'native':
            from mixfp4_sm120 import model as NM
            cov = NM.coverage(model)
            entry['coverage'] = {k: v for k, v in cov.items() if k != 'remaining_bf16_linears'}
            entry['coverage']['remaining_bf16_linears'] = cov['remaining_bf16_linears']
            entry['coverage']['remaining_bf16_linears_called'] = dict(unscoped_calls)
            for h in unscoped_hooks:
                h.remove()
            # every quantized Linear ran natively; no Linear of the scope is left in BF16; the unscoped ones never ran
            assert cov['native_called'] == cov['native'] == len(scoped), cov
            assert not scoped & set(cov['remaining_bf16_linears']) and not unscoped_calls, (cov, unscoped_calls)
        entry['seconds'] = time.time() - t1
        save()
    report['dataset_loads'] = lm_eval_datasets.LOADS
    report['status'] = 'complete'
    report['seconds'] = time.time() - t0
    save()


if __name__ == '__main__':
    main()
