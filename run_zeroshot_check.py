"""Does the Qwen perplexity gain show up in downstream accuracy?

Qwen3-4B MixFP4 perplexity falls below the unquantized BF16 reference, which
perplexity alone cannot adjudicate: a model that becomes less overconfident
scores better on next-token loss without predicting better. This evaluates the
same frozen policies on zero-shot multiple-choice tasks, where a smoothing
artefact should not help.

Policies: BF16, FourOverSix W4A4, MixFP4 at 256 tiles, MixFP4 at the
calibration-chosen count. Weight and activation quantization match the
paper-aligned evaluator exactly.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_adaptive_paper import MODELS, elect, name
from run_c4_frozen import digest_file
from run_conditional_format import sha

# Script-based loaders are gone from datasets 4.x, so a task can fail to download
# for reasons unrelated to the model. Each is evaluated independently and any that
# cannot load is recorded and skipped rather than failing the run.
TASKS = ('arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--counts', default='256,65536',
                    help='comma separated MixFP4 tile counts to evaluate')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    old = Path(args.stage_root) / f'results/math_code_adaptive/calibration_{MODELS[args.model]}_{args.model}'
    prior = json.loads((old / 'report.json').read_text())
    bundle = json.loads((old / 'maps.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert digest_file(old / 'maps.json') == prior['map_sha256']
    assert transformers.__version__ == prior['transformers_version'], transformers.__version__

    counts = [int(c) for c in args.counts.split(',')]
    r = dict(status='running', model=args.model, source=prior['source'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__,
             lm_eval_version=importlib.metadata.version('lm_eval'),
             tasks=list(TASKS), counts=counts, batch_size=args.batch_size,
             calibration=str(old), accuracy={})

    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')

    save()
    upper, order, slices, names = elect(old, prior)
    r['eligible_tiles'] = int((upper < 0).sum())
    r['total_tiles'] = int(upper.numel())
    maps = {}
    for count in counts:
        flat = torch.zeros(upper.numel(), dtype=torch.bool)
        take = order[:count]
        flat[take[upper[take] < 0]] = True
        maps[name(count)] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
        r[f'selected_{name(count)}'] = int(flat.sum())
    del upper, order

    model = AutoModelForCausalLM.from_pretrained(
        prior['source'], torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(prior['source'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == names
    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    if 'n256' in maps:
        for n, m in modules.items():
            want = torch.zeros(maps['n256'][n].numel(), dtype=torch.bool)
            want[bundle['maps']['fixed256_math_code128'][n]] = True
            assert torch.equal(maps['n256'][n].reshape(-1), want), n
        r['frozen_map_reproduced'] = True

    pristine, base, alt = {}, {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            pristine[n] = m.weight.detach().clone()
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    r['source_weights_verified'] = True
    save()

    def act(module, inputs):
        return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])

    handles = []

    def install(policy):
        """BF16 restores pristine weights and removes activation quantization."""
        nonlocal handles
        for h in handles:
            h.remove()
        handles = []
        with torch.no_grad():
            for n, m in modules.items():
                if policy == 'bf16':
                    m.weight.copy_(pristine[n])
                elif policy == 'four_over_six':
                    m.weight.copy_(base[n])
                else:
                    m.weight.copy_(apply_mask(base[n], alt[n], maps[policy][n].cuda()))
        if policy != 'bf16':
            handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    usable, skipped = [], {}
    for task in TASKS:
        try:
            lm_eval.tasks.TaskManager().load_task_or_group([task])
            usable.append(task)
        except Exception as exc:  # dataset availability, not a model property
            skipped[task] = repr(exc)[:200]
            print(f'SKIP {task}: {repr(exc)[:120]}', flush=True)
    assert usable, f'No task could be loaded: {skipped}'
    r['tasks_evaluated'], r['tasks_skipped'] = usable, skipped
    print(f'TASKS {usable}', flush=True)
    save()

    for policy in ['bf16', 'four_over_six', *[name(c) for c in counts]]:
        install(policy)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=list(usable), num_fewshot=0,
                                      batch_size=args.batch_size)['results']
        acc = {}
        for task, values in res.items():
            for key in ('acc_norm,none', 'acc,none'):
                if key in values:
                    acc[task] = dict(metric=key.split(',')[0], value=values[key],
                                     stderr=values.get(key.replace(',none', '_stderr,none')))
                    break
        acc['mean'] = sum(v['value'] for k, v in acc.items() if k != 'mean') / len(acc)
        r['accuracy'][policy] = acc
        save()
        print(f'ACC {policy} mean {acc["mean"]:.4f} ' +
              ' '.join(f'{k}={v["value"]:.4f}' for k, v in acc.items() if k != 'mean'), flush=True)
    for h in handles:
        h.remove()
    r['status'] = 'complete'
    save()

    policies = list(r['accuracy'])
    lines = [f'# Zero-shot accuracy check: {args.model}', '',
             'Perplexity cannot distinguish a genuinely better model from a less overconfident',
             'one. These are the same frozen policies on zero-shot multiple choice, where a',
             'smoothing artefact should not help. BF16 restores pristine weights and removes',
             'activation quantization; every other row uses the paper-aligned W4A4 path.', '',
             '| Policy | ' + ' | '.join(usable) + ' | mean |',
             '|---' * (len(usable) + 2) + '|']
    for p in policies:
        a = r['accuracy'][p]
        lines.append(f'| {p} | ' + ' | '.join(f'{a[t]["value"]:.4f}' for t in usable) +
                     f' | {a["mean"]:.4f} |')
    if skipped:
        lines += ['', 'Tasks skipped because their dataset could not be loaded in this '
                  'environment: ' + ', '.join(sorted(skipped)) + '.']
    lines += ['', 'If accuracy tracks the perplexity ordering, the perplexity gain reflects a',
              'better model. If accuracy is flat or lower while perplexity improves sharply, the',
              'perplexity gain on this model is a metric artefact and must be reported as such.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
