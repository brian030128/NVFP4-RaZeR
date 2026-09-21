"""Pre-fetch lm-eval task datasets (host, network) and record their Hub revisions and document digests.

Run once online to populate $HF_DATASETS_CACHE, then again with --offline to prove every task
loads from cache alone. No model is evaluated here.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import time
import traceback

TASKS = ['arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa', 'mmlu', 'gsm8k']


def leaves(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, 'config') and hasattr(v, 'dataset'):
                yield k, v
            else:
                yield from leaves(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    rec = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), offline=args.offline,
               lm_eval=md.version('lm_eval'), datasets=md.version('datasets'), tasks={}, failures={})
    api = None
    if not args.offline:
        from huggingface_hub import HfApi
        api = HfApi()
    tm = TaskManager()
    for t in TASKS:
        try:
            td = get_task_dict([t], tm)
            for name, task in leaves(td):
                cfg = task.config
                path, dname = cfg.dataset_path, cfg.dataset_name
                split = cfg.test_split or cfg.validation_split
                docs = list(task.test_docs() if task.has_test_docs() else task.validation_docs())
                h = hashlib.sha256()
                for d in docs:
                    h.update(json.dumps(d, sort_keys=True, default=str).encode())
                entry = dict(dataset_path=path, dataset_name=dname, split=split, docs=len(docs), docs_sha256=h.hexdigest(),
                             num_fewshot_default=cfg.num_fewshot, output_type=cfg.output_type,
                             metrics=[m['metric'] for m in (cfg.metric_list or [])], version=(cfg.metadata or {}).get('version'),
                             fewshot_split=cfg.fewshot_split, task_family=t)
                if api is not None:
                    try:
                        entry['hub_revision'] = api.dataset_info(path).sha
                    except Exception as exc:
                        entry['hub_revision_error'] = repr(exc)
                rec['tasks'][name] = entry
            print(f'TASK {t} ok ({sum(1 for _ in leaves(td))} leaf tasks)', flush=True)
        except Exception as exc:
            rec['failures'][t] = dict(error=repr(exc), traceback=traceback.format_exc()[-3000:])
            print(f'TASK {t} FAILED {exc!r}', flush=True)
    rec['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(rec, open(args.out, 'w'), indent=1, sort_keys=True)
    print(json.dumps(dict(ok=len(rec['tasks']), failures=list(rec['failures'])), indent=1))


if __name__ == '__main__':
    main()
