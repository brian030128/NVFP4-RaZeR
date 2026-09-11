"""
    Zero-shot accuracy for the four policies of MIXFP4_REPORT.md, at k = 3.

    The report is perplexity only. Perplexity is a next-token loss: a quantizer that makes the
    model less confident lowers it without the model predicting anything better, and the k-SE
    rule elects tiles by a calibration score that is itself a loss, so the possibility is real
    rather than hypothetical. Zero-shot multiple choice does not reward smoothing, so it is the
    check the report is missing.

    The four rows, built exactly as `run_kse_paper.py` builds them so the accuracy numbers sit on
    the same weights as the perplexity numbers:

      BF16 reference          pristine weights, no activation quantization
      NVFP4 W4A4              quant_nvfp4 weights, quant_nvfp4 activations
      NVFP4 FourOverSix W4A4  quant_nvfp4_4over6 weights and activations -- MixFP4's own base
      MixFP4 (k=3)            the FourOverSix weights with the elected tiles switched to E0M3

    The k = 3 map is re-elected from the calibration scores rather than assumed: the same
    `max(mean CE + k SE, mean KL + k SE) < 0` rule, the same frozen calibration. Two gates make
    that safe. The shipped score must be reproduced exactly at k = 2, and the 256-tile prefix of
    the k = 2 ranking must reproduce the frozen `fixed256_math_code128` map bitwise -- both are
    `run_kse_paper.py`'s own assertions, reused here rather than reimplemented.

        python run_zeroshot_kse.py --model llama8b --calib <dir> --out <dir>
"""

import argparse
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
from run_adaptive_paper import FROZEN
from run_c4_frozen import digest_file
from run_conditional_format import sha
from run_kse_paper import MODELS, STREAMED, elect_k

K = 3

# Script-based loaders are gone from datasets 4.x, so a task can fail to download for reasons
# unrelated to the model. Each is probed and any that cannot load is recorded and skipped.
TASKS = ('arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa')

# lm-eval names its metric differently per task family; generative tasks report exact_match with
# an extraction suffix. Tried in order, first hit wins.
METRIC_KEYS = ('acc_norm,none', 'acc,none',
               'exact_match,strict-match', 'exact_match,flexible-extract', 'exact_match,none')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm -- see /home/u4320956/CLAUDE.md'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--calib', required=True,
                    help='A freshly regenerated math/code calibration directory containing '
                         'scores/ -- the shipped one under results/math_code_adaptive/ no '
                         'longer has its score shards.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--tasks', type=lambda v: tuple(x.strip() for x in v.split(',') if x.strip()),
                    default=TASKS,
                    help='Task list. The default panel is 4-way multiple choice, which resolves '
                         'differences of about 0.004 and no smaller; generative chain-of-thought '
                         'tasks (gsm8k) are far more sensitive to a small per-token degradation '
                         'because errors compound over the chain.')
    ap.add_argument('--num-fewshot', type=int, default=0,
                    help='0 for the zero-shot panel; 8 is conventional for gsm8k, 5 for mmlu.')
    ap.add_argument('--samples-dir', default=None,
                    help='Write per-document outcomes per policy here. All four policies are '
                         'scored on the same documents, so their differences are paired, and '
                         'the per-task standard error lm-eval prints is the error of one '
                         'measurement rather than of the difference.')
    ap.add_argument('--allow-map-drift', action='store_true',
                    help='Evaluate even when the re-election does not reproduce the shipped '
                         'frozen map, recording exactly how far off it is. The result is then '
                         'the k-SE rule at this k re-derived here, NOT the shipped artifact, '
                         'and must be labelled as such wherever it is reported.')
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()

    import lm_eval_compat  # noqa: F401  -- restores a name transformers 5 dropped; see module
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib = Path(args.calib)
    prior = json.loads((calib / 'report.json').read_text())
    fresh_bundle = json.loads((calib / 'maps.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert digest_file(calib / 'maps.json') == prior['map_sha256']
    assert transformers.__version__ == prior['transformers_version'], transformers.__version__

    # The shipped calibration for this model, kept only as a reference digest now that its
    # score shards are gone. If the regenerated maps.json matches it, the re-election below is
    # the shipped election and not merely a re-derivation of it.
    shipped_dir = Path(args.stage_root) / \
        f'results/math_code_adaptive/calibration_{MODELS[args.model]}_{args.model}'
    shipped_report = shipped_dir / 'report.json'
    shipped_sha = json.loads(shipped_report.read_text())['map_sha256'] \
        if shipped_report.is_file() else None

    # The election is validated against the SHIPPED frozen map, not the freshly generated one.
    # Checking the fresh re-election against the fresh bundle only proves internal consistency;
    # what has to hold is that it lands on the artifact the report was built from. maps.json
    # carries 20 maps and only fixed256_math_code128 cross-checks this election, so comparing
    # whole-file digests is the wrong test -- a difference in the adaptive_* searches, which
    # nothing here uses, would fail it for no reason. Qwen3.8-27B does exactly that.
    shipped_maps = shipped_dir / 'maps.json'
    bundle = json.loads(shipped_maps.read_text()) if shipped_maps.is_file() else fresh_bundle
    frozen_source = 'shipped' if shipped_maps.is_file() else 'fresh'

    r = dict(status='running', model=args.model, source=prior['source'], k=K,
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__,
             lm_eval_version=__import__('importlib.metadata', fromlist=['version'])
             .version('lm_eval'),
             calibration=str(calib), map_sha256=prior['map_sha256'],
             shipped_map_sha256=shipped_sha,
             calibration_maps_json_matches_shipped=(shipped_sha == prior['map_sha256']),
             frozen_map_source=frozen_source,
             batch_size=args.batch_size, samples_dir=args.samples_dir,
             tasks_requested=list(args.tasks), num_fewshot=args.num_fewshot,
             election={}, accuracy={})

    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')

    save()
    print(f'maps.json matches shipped: {r["calibration_maps_json_matches_shipped"]} '
          f'(informational -- covers 20 maps, 19 unused here)', flush=True)
    print(f'frozen map for validation taken from: {frozen_source}', flush=True)

    # Re-elect. k = 2 is asserted inside elect_k to equal the shipped score exactly.
    uppers, slices, names = elect_k(calib, prior, (2, K))
    r['shipped_score_identical_at_k2'] = True
    r['total_tiles'] = int(uppers[2].numel())
    flat_k = uppers[K] < 0
    maps = {f'k{K}': {n: flat_k[lo:hi].clone() for n, (lo, hi) in slices.items()}}
    r['election'][f'k{K}'] = dict(k=K, selected=int(flat_k.sum()),
                                  fraction=float(flat_k.sum()) / flat_k.numel())
    print(f'ELECT k={K}: {int(flat_k.sum()):,} of {flat_k.numel():,} tiles', flush=True)

    # The 256-tile prefix of the k = 2 ranking must be the frozen map, bitwise.
    order = torch.argsort(uppers[2], stable=True)[:256]
    flat256 = torch.zeros(uppers[2].numel(), dtype=torch.bool)
    flat256[order[uppers[2][order] < 0]] = True
    maps['n256'] = {n: flat256[lo:hi].clone() for n, (lo, hi) in slices.items()}
    del uppers, order
    save()

    # Task datasets are prepared BEFORE the model is loaded, and the order matters. `datasets`
    # forks worker processes to build a dataset, and a fork from a parent holding the model --
    # on the GPU plus its CPU-side variants -- fails to reserve memory:
    #     File "multiprocessing/popen_fork.py", line 66, in _launch
    #       self.pid = os.fork()
    #     OSError: [Errno 12] Cannot allocate memory
    # Probing first leaves the parent small at fork time, and everything is cached by the time
    # evaluation runs. Each task is probed independently so one unavailable dataset is recorded
    # and skipped rather than failing the run.
    usable, skipped = [], {}
    for task in args.tasks:
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

    target = args.model in STREAMED
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda', output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys')
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(
            prior['source'], torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda')
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(prior['source'])
    r['model_class'] = type(model).__name__
    assert list(modules) == names

    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8,
                                                      m.weight.shape[1] // 64)
    drift = {}
    for n, m in modules.items():
        want = torch.zeros(maps['n256'][n].numel(), dtype=torch.bool)
        want[bundle['maps'][FROZEN][n]] = True
        got = maps['n256'][n].reshape(-1)
        if not torch.equal(got, want):
            drift[n] = dict(shipped_only=int((want & ~got).sum()),
                            reelected_only=int((got & ~want).sum()))
    r['frozen_map_reproduced'] = not drift
    if drift:
        # Quantified, not waved away: how many tiles moved, and where.
        r['frozen_map_drift'] = dict(
            modules_differing=len(drift),
            tiles_in_shipped_only=sum(d['shipped_only'] for d in drift.values()),
            tiles_in_reelected_only=sum(d['reelected_only'] for d in drift.values()),
            detail=drift)
        msg = (f'RE-ELECTION DIFFERS FROM {frozen_source.upper()} {FROZEN}: '
               f'{len(drift)} module(s), '
               f'{r["frozen_map_drift"]["tiles_in_shipped_only"]} tile(s) only in the shipped '
               f'map, {r["frozen_map_drift"]["tiles_in_reelected_only"]} only in the '
               f're-election')
        print(msg, flush=True)
        assert args.allow_map_drift, (
            msg + ' -- refusing to report this as the shipped policy. Pass --allow-map-drift to '
            'evaluate it anyway, clearly labelled as a re-derivation.')
        print('PROCEEDING UNDER --allow-map-drift: results are the k-SE rule re-derived here, '
              'not the shipped artifact', flush=True)
    else:
        print(f'REELECTION AT 256 MATCHES {frozen_source.upper()} {FROZEN} -- this election is '
              f'the shipped k = {K} policy', flush=True)
    save()

    # Weight variants, always on CPU. Keeping them on the GPU costs three extra copies of the
    # model there, which likelihood scoring tolerates and generation does not: gsm8k's
    # generate_until with a KV cache over 8-shot prompts ran the 8B model out of 79 GiB. They are
    # read once per policy install, so the transfer is negligible next to the evaluation.
    pristine, nvfp4_w, base, alt = {}, {}, {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            pristine[n] = m.weight.detach().to('cpu', copy=True)
            nvfp4_w[n] = quant_nvfp4(m.weight, 4, 16).to('cpu', copy=True)
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16).to('cpu', copy=True)
            if not target:
                alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1',
                                       elect='always').to('cpu', copy=True)
    r['source_weights_verified'] = True
    save()

    # Activation quantization matches the weight format of the row, as in the released cases:
    # plain NVFP4 activations for the NVFP4 row, FourOverSix for FourOverSix and for MixFP4.
    handles = []

    def install(policy):
        nonlocal handles
        for h in handles:
            h.remove()
        handles = []
        with torch.no_grad():
            for n, m in modules.items():
                if policy == 'bf16':
                    w = pristine[n]
                elif policy == 'nvfp4':
                    w = nvfp4_w[n]
                elif policy == 'four_over_six':
                    w = base[n]
                else:
                    b = base[n].to(m.weight.device)
                    if target:
                        a = quant_mix_4_6(pristine[n].to(m.weight.device), 4, 16,
                                          type_block=(8, 64), clip='a1', elect='always') \
                            if bool(maps[policy][n].any()) else None
                        w = apply_mask(b, a, maps[policy][n].cuda()) if a is not None else b
                    else:
                        w = apply_mask(b, alt[n].to(m.weight.device),
                                       maps[policy][n].to(m.weight.device))
                m.weight.copy_(w.to(m.weight.device))
        if policy == 'bf16':
            return
        q = quant_nvfp4 if policy == 'nvfp4' else quant_nvfp4_4over6

        def act(module, inputs):
            return (q(inputs[0], 4, 16), *inputs[1:])

        handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    for policy in ('bf16', 'nvfp4', 'four_over_six', f'k{K}'):
        install(policy)
        torch.cuda.empty_cache()
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
        # bootstrap_iters=0 disables lm-eval's bootstrap standard errors. Two reasons: they
        # spawn a multiprocessing Pool after evaluation, and forking from a parent that holds
        # the model fails with "OSError: [Errno 12] Cannot allocate memory" -- which killed an
        # mmlu run at the aggregation step, after all the compute was already spent. And they
        # are not the error model used here: differences between policies are tested paired,
        # with an exact McNemar test on per-document outcomes, which the bootstrap over a single
        # policy's score cannot substitute for. Per-task acc stderr is analytic and unaffected.
        full = lm_eval.simple_evaluate(model=lm, tasks=list(usable),
                                       num_fewshot=args.num_fewshot,
                                       batch_size=args.batch_size,
                                       bootstrap_iters=0,
                                       log_samples=args.samples_dir is not None)
        if args.samples_dir:
            sdir = Path(args.samples_dir)
            sdir.mkdir(parents=True, exist_ok=True)
            keep = {}
            for task, records in (full.get('samples') or {}).items():
                # doc_id and the scored metric are all a paired test needs; prompts and raw
                # continuations are identical across policies and large.
                keep[task] = [{'doc_id': rec.get('doc_id'),
                               **{kk: vv for kk, vv in rec.items()
                                  if kk in ('acc', 'acc_norm', 'exact_match')}}
                              for rec in records]
            (sdir / f'{policy}.json').write_text(json.dumps(keep))
        res = full['results']
        acc = {}
        for task, values in res.items():
            for key in METRIC_KEYS:
                if key in values:
                    acc[task] = dict(metric=key.split(',')[0], value=values[key],
                                     stderr=values.get(key.replace(',none', '_stderr,none')))
                    break
        # Mean over the tasks that were ASKED for, not over every key lm-eval returns. A group
        # task reports its aggregate and each of its members: mmlu comes back as `mmlu` plus 57
        # subject rows plus 4 category rows, so averaging everything counts the subjects twice
        # and silently reweights the panel towards whichever task happens to be a group.
        named = [k for k in acc if k in usable]
        acc['mean'] = (sum(acc[k]['value'] for k in named) / len(named)) if named else \
            sum(v['value'] for k, v in acc.items() if k != 'mean') / max(len(acc), 1)
        acc['mean_over'] = named
        r['accuracy'][policy] = acc
        save()
        print(f'ACC {policy} mean {acc["mean"]:.4f} ' +
              ' '.join(f'{k}={v["value"]:.4f}' for k, v in acc.items() if k != 'mean'), flush=True)

    for h in handles:
        h.remove()
    r['status'] = 'complete'
    save()
    print(f'Done. {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
