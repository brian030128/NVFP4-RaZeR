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
  Per task (`native_gemm`): the native GEMM calls and tokens, the model forwards by token count (the selection
  buckets of mixfp4_sm120.select; a gsm8k decode step is a forward of batch-size tokens), whether every quantized
  Linear ran in every forward, and, for a width-selecting KernelSet ('auto', 'auto_stock'), the GEMM calls by CTA
  tile width. The install record's `kernel_set` is replaced after the policy by its cumulative description.
- **--repeat LABEL** evaluates that policy a second time in the same process (a determinism check).
- **--num-fewshot TASK=N** sets that task's few-shot count (lm-eval's default otherwise); the paper runs mmlu=5.
- **Group tasks (mmlu):** lm-eval evaluates the 57 subjects; the entry `mmlu` holds lm-eval's own group aggregate
  (weight_by_size: the micro-average over all questions), the per-subject and per-category metrics, and the examples of
  every subject keyed `<subject>:<doc_id>`.
- **Per-choice log-likelihoods** of every multiple-choice example are recorded under `loglikelihoods`
  ({key: {target, ll: [per choice]}}), next to the per-example correctness in `examples`.
- **What was evaluated:** per task, `sample_digest` hashes lm-eval's doc / prompt / target hashes of every example, and
  `prompt_hash` keeps a 12-character prefix per example. Paired comparisons require equal digests.
- **--resume** keeps the (policy, task) entries an earlier run of the same --out already finished. Every task entry
  records the batch size it ran with.
- **--no-bos** passes add_bos_token=False to HFLM: contexts without BOS, lm-eval 0.4.5 / 0.4.9.1's behaviour. 0.4.11's
  default keeps the tokenizer's special tokens (BOS for Llama-3.1 and Mistral).
- **Older lm-eval** (a cross-check): run with its install first on PYTHONPATH; lm_eval_compat is imported for
  versions before 0.4.10 (transformers 5).
"""
import argparse
import datetime
import hashlib
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
PRIMARY['mmlu'], CHANCE['mmlu'] = 'acc', 0.25


def primary(task):
    return 'acc' if task.startswith('mmlu') else PRIMARY[task]


def mc_record(s, key):
    """(correctness, {target, ll}) of one multiple-choice sample: lm-eval's per-choice (log-likelihood, is_greedy)."""
    lls = [float(r[0]) for r in s['filtered_resps']] if s.get('filtered_resps') else None
    return float(s[key]), dict(target=s.get('target'), ll=lls)


def sample_hashes(entry, keyed_rows):
    """What each example was evaluated on: lm-eval's per-sample doc / prompt / target hashes (sha256).
    `sample_digest` hashes them all, sorted by example key, so two policies saw the same samples iff their digests are
    equal; `prompt_hash` keeps a 12-character prefix per example to locate a difference."""
    lines, prefix = [], {}
    for k, s in keyed_rows:
        if k in prefix or 'prompt_hash' not in s:
            continue
        prefix[k] = s['prompt_hash'][:12]
        lines.append(f"{k} {s.get('doc_hash')} {s['prompt_hash']} {s.get('target_hash')}")
    if lines:
        entry['sample_digest'] = hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()
        entry['prompt_hash'] = prefix


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
            key = primary(task)
            entry['examples'], entry['loglikelihoods'] = {}, {}
            for s in rows:
                if key in s:
                    entry['examples'][str(s['doc_id'])], entry['loglikelihoods'][str(s['doc_id'])] = mc_record(s, key)
        sample_hashes(entry, [(str(s['doc_id']), s) for s in rows])
        out[task] = entry
    return out


def summarize_group(res, task):
    """A group task (mmlu): lm-eval's group aggregate, the metrics of every subtask and subgroup, and the examples of
    every subtask keyed '<subtask>:<doc_id>'."""
    results = res['results']
    groups = res.get('groups', {})
    top = results.get(task) or groups.get(task)
    entry = dict(metrics={k: v for k, v in top.items() if isinstance(v, (int, float)) and k != 'alias'},
                 subtasks={t: {k: v for k, v in m.items() if isinstance(v, (int, float)) and k != 'alias'}
                           for t, m in results.items() if t != task},
                 examples={}, loglikelihoods={})
    for sub, rows in res.get('samples', {}).items():
        for s in rows:
            if 'acc' in s:
                k = f'{sub}:{s["doc_id"]}'
                entry['examples'][k], entry['loglikelihoods'][k] = mc_record(s, 'acc')
    sample_hashes(entry, [(f'{sub}:{s["doc_id"]}', s) for sub, rows in res.get('samples', {}).items() for s in rows])
    return entry


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
    ap.add_argument('--num-fewshot', action='append', default=[], metavar='TASK=N', help='few-shot count of one task')
    ap.add_argument('--resume', action='store_true', help='keep the finished (policy, task) entries of an earlier run')
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
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), evaluations={},
                  num_fewshot=dict(spec.split('=') for spec in args.num_fewshot))
    report['num_fewshot'] = {k: int(v) for k, v in report['num_fewshot'].items()}
    previous = {}
    if args.resume and (args.out / 'report.json').exists():
        old = json.loads((args.out / 'report.json').read_text())
        assert old['model'] == args.model and old.get('num_fewshot', {}) == report['num_fewshot'] and old['limit'] == args.limit, \
            'resume: the earlier run used other settings'
        previous = old['evaluations']
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
            from mixfp4_sm120.select import bucket
            nat = NM.native_modules(model)
            ksets = list({id(m.kernel_set): m.kernel_set for m in nat.values() if m.kernel_set is not None}.values())
            # every quantized Linear of a forward sees the same token count, so the first one's input gives the forward's
            forwards = {}

            def count_forward(mod, inputs):
                b = bucket(inputs[0].numel() // inputs[0].shape[-1])
                forwards[b] = forwards.get(b, 0) + 1
            probe = next(iter(nat.values())).register_forward_pre_hook(count_forward)

            def native_counts():
                widths = {}
                for ks in ksets:
                    for w, c in ks.stats.items():
                        widths[w] = widths.get(w, 0) + c
                return dict(calls=sum(m.calls for m in nat.values()), tokens=sum(m.tokens for m in nat.values()),
                            calls_by_width=widths, forwards_by_token_bucket=dict(forwards))
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
            ids = lm.tok_encode('Question: 1 + 1 =')       # not `probe`: that is the native policy's forward hook
            report['tokenization'] = dict(add_bos_token=lm.add_bos_token, bos_token=tok.bos_token, bos_token_id=tok.bos_token_id,
                                          context_starts_with_bos=tok.bos_token_id is not None and ids[0] == tok.bos_token_id,
                                          probe_ids=ids[:4])
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
            done = previous.get(pol['label'], {}).get('tasks', {}).get(task)
            if done is not None and done.get('examples'):
                entry['tasks'][task] = dict(done, resumed=True)
                print(f"LMEVAL {pol['label']} {task} resumed from the earlier run", flush=True)
                continue
            timing.clear()
            limit = args.gsm8k_limit if (task == 'gsm8k' and args.gsm8k_limit is not None) else args.limit
            before = native_counts() if pol['kind'] == 'native' else None
            torch.cuda.reset_peak_memory_stats()
            t2 = time.time()
            # lm-eval's defaults. Every metric here aggregates by mean, whose stderr is the closed form (no bootstrap
            # draws); bootstrap_iters=0 would drop the stderr altogether.
            res = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, log_samples=True, bootstrap_iters=100000,
                                          random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                                          fewshot_random_seed=1234, num_fewshot=report['num_fewshot'].get(task))
            if task in res.get('samples', {}):
                summary = summarize(res, res.get('samples', {}))[task]
            else:                                   # a group of subtasks (mmlu)
                summary = summarize_group(res, task)
                summary['aggregate'] = 'lm-eval group aggregate, weight_by_size (micro-average over all questions)'
            n_shot = res.get('n-shot', {}).get(task)
            if n_shot is None and 'subtasks' in summary:    # a group: the distinct few-shot counts of its subtasks
                n_shot = sorted({v for t, v in res.get('n-shot', {}).items() if t in summary['subtasks']})
            summary.update(seconds=time.time() - t2, examples_evaluated=len(summary['examples']), batch_size=args.batch_size,
                           peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, n_shot=n_shot,
                           chance=CHANCE.get(task, 0.25), model_timing=dict(timing))
            if before is not None:
                after = native_counts()
                delta = lambda a, b: {k: a[k] - b.get(k, 0) for k in a if a[k] != b.get(k, 0)}  # noqa: E731
                g = summary['native_gemm'] = dict(calls=after['calls'] - before['calls'], tokens=after['tokens'] - before['tokens'],
                                                  forwards_by_token_bucket=delta(after['forwards_by_token_bucket'], before['forwards_by_token_bucket']),
                                                  calls_by_width=delta(after['calls_by_width'], before['calls_by_width']) if ksets else None,
                                                  kernel=entry['install']['kernel'])
                g['forwards'] = sum(g['forwards_by_token_bucket'].values())
                g['every_linear_every_forward'] = g['calls'] == len(nat) * g['forwards']
            entry['tasks'][task] = summary
            print(f"LMEVAL {pol['label']} {task} {primary(task)} {summary['metrics'].get(primary(task) + ',none', summary['metrics'].get(primary(task) + ',strict-match'))} "
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
            probe.remove()
            if ksets:
                entry['install']['kernel_set'] = ksets[0].describe() if len(ksets) == 1 else [k.describe() for k in ksets]
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
