"""Pinned datasets for lm-eval (Task 3): import this before lm-eval builds its tasks.

The downstream suite is:
- arc_easy, arc_challenge, hellaswag, openbookqa, boolq, winogrande, piqa (0-shot log-likelihood);
- gsm8k (generative).

lm-eval 0.4.11, the pinned version, names every dataset by its namespaced Hugging Face repo but loads the repo's
current head. This module pins each repo to the commit recorded below, by patching datasets.load_dataset:
- a call without an explicit revision gets the pinned one;
- every call is recorded in LOADS, so the caller can report what was loaded.

For lm-eval 0.4.5 and 0.4.9.1 (reproducing older runs), the legacy un-namespaced ids are also resolved to the repos
lm-eval 0.4.11 uses, and trust_remote_code is dropped (datasets 4 no longer runs dataset scripts). The task
definitions are otherwise identical across these three versions: the prompts, targets, choices and metrics, and the
hellaswag / winogrande preprocessing are byte-identical (results/lmeval_ready/READINESS.md). This supersedes the
lm_eval_dataset_ids.py of the zero-shot WIP (stash on repro/n16k64-rtx-pro-6000).
"""
import datasets

# dataset repo -> commit (huggingface_hub dataset_info(...).sha on 2026-09-27)
PINS = {
    'allenai/ai2_arc': '210d026faf9955653af8916fad021475a3f00453',
    'Rowan/hellaswag': '218ec52e09a7e7462a5400043bb9a69a41d06b76',
    'allenai/openbookqa': '388097ea7776314e93a529163e0fea805b8a6454',
    'aps/super_glue': '3de24cf8022e94f4ee4b9d55a6f539891524d646',
    'allenai/winogrande': '01e74176c63542e6b0bcb004dcdea22d94fb67b5',
    'baber/piqa': '142f6d7367fd9877f0fb3b5734ea6a545f54cdd1',
    'openai/gsm8k': '740312add88f781978c0658806c59bc2815b9866',
}
LEGACY_TO_NAMESPACED = {'ai2_arc': 'allenai/ai2_arc', 'hellaswag': 'Rowan/hellaswag', 'openbookqa': 'allenai/openbookqa',
                        'super_glue': 'aps/super_glue', 'winogrande': 'allenai/winogrande', 'piqa': 'baber/piqa',
                        'gsm8k': 'openai/gsm8k'}
LOADS = []
_original = datasets.load_dataset


def load_dataset(path, *args, **kwargs):
    target = LEGACY_TO_NAMESPACED.get(path, path)
    dropped = kwargs.pop('trust_remote_code', None)
    if target in PINS and kwargs.get('revision') is None:
        kwargs['revision'] = PINS[target]
    result = _original(target, *args, **kwargs)
    LOADS.append(dict(requested=path, loaded=target, name=kwargs.get('name', args[0] if args else None),
                      revision=kwargs.get('revision'), dropped_trust_remote_code=dropped))
    return result


datasets.load_dataset = load_dataset
