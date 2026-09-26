"""Zero-shot accuracy of frozen MixFP4 tile maps, paired with FourOverSix in the same run.

lm-eval 0.4.5, 0-shot, arc_easy / arc_challenge / hellaswag / openbookqa / boolq /
winogrande, acc_norm where defined else acc, unweighted mean. FourOverSix weights with
E0M3 alpha=1 on the mapped tiles. Activation fake-quantization (FourOverSix, or NVFP4
for the NVFP4 row) uses one tensor-wide scale per document, so results do not depend
on the lm-eval batch size, unlike run_zeroshot_kse.py whose scale spans the batch.

  --map LABEL=PATH:TILE_ROWS   frozen {module: bool[ceil(N/rows), K/64]} map, repeatable
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.fast_act import quant_per_document
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import sha

CALIBRATIONS = {'llama8b': '/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration',
                'qwen27b': '/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration',
                'llama8b_ins': '/work/u4320956/mixfp4_potential/llama8b_ins_calibration'}
TASKS = ('arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande')
METRIC_KEYS = ('acc_norm,none', 'acc,none')


def expand(mask, rows, height):
    return mask.repeat_interleave(rows, 0).repeat_interleave(64, 1)[:height]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    import lm_eval_compat  # noqa: F401  (needed before lm_eval under transformers 5)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=tuple(CALIBRATIONS), required=True)
    ap.add_argument('--map', action='append', default=[], help='LABEL=PATH:TILE_ROWS')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--baselines', default='bf16,nvfp4,four_over_six',
                    help='Comma-separated baseline policies evaluated before the maps')
    ap.add_argument('--samples-dir', type=Path, default=None)
    ap.add_argument('--tasks', default=','.join(TASKS),
                    help='Comma-separated lm-eval tasks (e.g. gsm8k_llama,mmlu_cot_llama for the RaZeR CoT benchmark)')
    ap.add_argument('--chat', action='store_true',
                    help='apply_chat_template and fewshot_as_multiturn, as in run_llama_cot.py')
    ap.add_argument('--limit', type=int, default=None, help='Documents per task (smoke runs only)')
    args = ap.parse_args()
    tasks = tuple(t for t in args.tasks.split(',') if t)
    torch.backends.cuda.matmul.allow_tf32 = False
    qwen = args.model == 'qwen27b'
    prior = json.loads((Path(CALIBRATIONS[args.model]) / 'report.json').read_text())
    specs = {}
    for spec in args.map:
        label, _, rest = spec.partition('=')
        path, _, rows = rest.rpartition(':')
        specs[label] = (Path(path), int(rows))
    args.out.mkdir(parents=True, exist_ok=True)
    r = dict(status='running', model=args.model, job_id=os.environ['SLURM_JOB_ID'], tasks=list(tasks), chat=args.chat, limit=args.limit,
             num_fewshot=None if args.chat else 0, batch_size=args.batch_size, lm_eval_version=lm_eval.__version__ if hasattr(lm_eval, '__version__') else None,
             transformers_version=transformers.__version__,
             maps={k: dict(path=str(p), tile_rows=rows, sha256=digest_file(p)) for k, (p, rows) in specs.items()},
             accuracy={})

    def save():
        (args.out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
    save()
    if qwen:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(prior['source'], revision=prior['revision'],
                                                                dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                device_map='cuda')
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
                                                     torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                     device_map='cuda')
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    assert list(modules) == list(prior['matrices'])
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    maps = {label: torch.load(p, map_location='cpu', weights_only=True) for label, (p, _) in specs.items()}
    for label, m in maps.items():
        assert set(m) == set(modules), label
        r['maps'][label]['e0m3_tiles'] = sum(int(v.sum()) for v in m.values())
    # Candidates live on the CPU; each policy install decodes one module at a time.
    pristine = {}
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        pristine[n] = m.weight.detach().to('cpu', copy=True)
    save()
    handles = []

    def install(policy):
        nonlocal handles
        for h in handles:
            h.remove()
        with torch.no_grad():
            for n, m in modules.items():
                w = pristine[n].to(m.weight.device)
                if policy in ('bf16', 'nvfp4'):
                    m.weight.copy_(w if policy == 'bf16' else quant_nvfp4(w, 4, 16))
                    del w
                    continue
                b = quant_nvfp4_4over6(w, 4, 16)
                if policy != 'four_over_six':
                    mask = maps[policy][n]
                    if bool(mask.any()):
                        a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                        b = torch.where(expand(mask.to(w.device), specs[policy][1], w.shape[0]), a, b)
                        del a
                m.weight.copy_(b)
                del w, b
        # Activation scales are per document (not per lm-eval batch), so the batch size
        # only changes speed. NVFP4 activations for the NVFP4 row, FourOverSix otherwise.
        if policy == 'bf16':
            handles = []
        elif policy == 'nvfp4':
            handles = [m.register_forward_pre_hook(lambda module, inputs: (torch.stack(
                [quant_nvfp4(x, 4, 16) for x in inputs[0]]) if inputs[0].dim() >= 3 else quant_nvfp4(inputs[0], 4, 16),
                *inputs[1:])) for m in modules.values()]
        else:
            handles = [m.register_forward_pre_hook(
                lambda module, inputs: (quant_per_document(inputs[0]), *inputs[1:])) for m in modules.values()]

    baselines = [b for b in args.baselines.split(',') if b]
    assert set(baselines) <= {'bf16', 'nvfp4', 'four_over_six'}
    r['activation_scale'] = 'per document (padding positions included)'
    for policy in baselines + list(specs):
        install(policy)
        torch.cuda.empty_cache()
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
        extra = dict(apply_chat_template=True, fewshot_as_multiturn=True) if args.chat else dict(num_fewshot=0)
        full = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), batch_size=args.batch_size, limit=args.limit,
                                       bootstrap_iters=0, log_samples=args.samples_dir is not None, **extra)
        if args.samples_dir:
            args.samples_dir.mkdir(parents=True, exist_ok=True)
            keep = {task: [{'doc_id': rec.get('doc_id'), **{k: v for k, v in rec.items() if k in ('acc', 'acc_norm', 'exact_match')}}
                           for rec in records] for task, records in (full.get('samples') or {}).items()}
            (args.samples_dir / f'{policy}.json').write_text(json.dumps(keep))
        # Every numeric metric of every returned task/group, e.g. exact_match with its
        # strict/flexible filters for generation tasks.
        r.setdefault('metrics', {})[policy] = {
            task: {k: v for k, v in values.items() if isinstance(v, (int, float))}
            for task, values in full['results'].items()}
        if set(tasks) <= set(TASKS):
            acc = {}
            for task in tasks:
                values = full['results'][task]
                key = next(k for k in METRIC_KEYS if k in values)
                acc[task] = dict(metric=key.split(',')[0], value=values[key],
                                 stderr=values.get(key.replace(',none', '_stderr,none')))
            acc['mean'] = sum(acc[t]['value'] for t in tasks) / len(tasks)
            r['accuracy'][policy] = acc
            print(f'ACC {policy} mean {acc["mean"]:.4f} ' + ' '.join(f'{t}={acc[t]["value"]:.4f}' for t in tasks), flush=True)
        else:
            for task in tasks:
                em = {k: round(v, 4) for k, v in r['metrics'][policy].get(task, {}).items()
                      if k.startswith('exact_match') and 'stderr' not in k}
                print(f'ACC {policy} {task} {json.dumps(em)}', flush=True)
        save()
    for h in handles:
        h.remove()
    r['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
