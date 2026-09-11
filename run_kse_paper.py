"""Measure the search-free k-SE count rule under the paper-aligned protocol.

The shipped election keeps tiles with max(mean_CE + 2 SE, mean_KL + 2 SE) < 0 and
then truncates to a fixed 256. Raising the constant from 2 to k tightens the same
rule instead, needs no cap and no per-model search, and adapts on its own because
SE scales with each model's score noise. The offline screen suggested k around 4;
this measures the rule exactly, electing every tile that passes.

k = 2 must reproduce the shipped scores exactly, and the 256-tile prefix of the
k = 2 ranking must reproduce the frozen fixed256_math_code128 map bitwise.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.relinearized_format import common_descent_scores
from run_adaptive_paper import LENGTH, FROZEN
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import sha

MODELS = {'llama8b': '333779', 'qwen4b': '333779', 'qwen27b': '333787'}
K_VALUES = (2, 3, 4, 5, 6)
# 24.35B parameters will not hold a resident baseline and alternative alongside the
# model, so the target streams each module's candidates from a CPU copy instead.
STREAMED = ('qwen27b',)


def upper_scores(ce, kl, k):
    """max(mean + k SE) over the two objectives; k = 2 is the shipped score."""
    out = []
    for x in (ce, kl):
        out.append(x.mean(0) + k * x.std(0, unbiased=True) / math.sqrt(x.shape[0]))
    return torch.maximum(*out)


def elect_k(old, prior, ks):
    """Per-module scores rebuilt from the frozen shards, for every k at once."""
    names = list(prior['matrices'])
    parts = {k: [] for k in ks}
    slices, offset = {}, 0
    for i, n in enumerate(names):
        shard = torch.load(old / 'scores' / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert shard['name'] == n
        ce, kl = shard['ce'], shard['kl']
        assert ce.shape[0] == 128 and ce.shape == kl.shape
        for k in ks:
            u = upper_scores(ce, kl, k)
            assert torch.isfinite(u).all()
            if k == 2:
                # The shipped path must agree exactly, or the rest is not comparable.
                assert torch.equal(u, common_descent_scores(
                    ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))), n
            parts[k].append(u)
        slices[n] = (offset, offset + ce.shape[1])
        offset += ce.shape[1]
        del shard, ce, kl
    return {k: torch.cat(v) for k, v in parts.items()}, slices, names


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calib', default=None,
                    help='Calibration directory to elect from. Defaults to the shipped one for '
                         'this model, which no longer carries its score shards, so a freshly '
                         'regenerated directory has to be passed explicitly.')
    ap.add_argument('--k-values', default=None,
                    type=lambda v: tuple(int(x) for x in v.split(',') if x.strip()),
                    help='Restrict which k are evaluated. k = 2 is always elected anyway, '
                         'because the frozen-map check depends on it.')
    ap.add_argument('--objectives', default='max',
                    type=lambda v: tuple(x.strip() for x in v.split(',') if x.strip()),
                    help="Election objectives to evaluate: 'max' is the shipped rule, which "
                         "requires both CE and KL to be confidently negative; 'kl' and 'ce' "
                         'threshold one alone. The report argues for the conjunction but never '
                         'measured the alternatives.')
    ap.add_argument('--skip-n256', action='store_true',
                    help='Skip the fixed-256 row, which is a reference point rather than part '
                         'of the k-SE rule.')
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    old = Path(args.calib) if args.calib else (
        Path(args.stage_root)
        / f'results/math_code_adaptive/calibration_{MODELS[args.model]}_{args.model}')
    prior = json.loads((old / 'report.json').read_text())
    bundle = json.loads((old / 'maps.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert digest_file(old / 'maps.json') == prior['map_sha256']
    assert transformers.__version__ == prior['transformers_version']

    r = dict(status='running', model=args.model, source=prior['source'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__, calibration=str(old),
             map_sha256=prior['map_sha256'], length=LENGTH,
             activation='FourOverSix tensor-wide factor', wiki_use_cache=True, c4_use_cache=False,
             calibration_sources=['OpenWebMath', 'CodeParrot'],
             uses_c4_calibration=False, uses_wiki_calibration=False,
             k_values=list(K_VALUES), election={}, evaluation={})

    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')

    save()
    # k = 2 is always elected: the frozen-map cross-check is defined on its ranking.
    k_values = tuple(sorted(set(args.k_values or K_VALUES) | {2}))
    r['k_values'] = list(k_values)
    r['objectives'] = list(args.objectives)
    uppers, slices, names = elect_k(old, prior, k_values)
    r['shipped_score_identical_at_k2'] = True
    r['total_tiles'] = int(uppers[2].numel())

    maps = {}
    obj_upper = None
    if any(o != 'max' for o in args.objectives):
        # Per-objective bounds; elect_k only keeps the maximum of the two.
        from analyze_objective_ablation import scores_for
        obj_upper, obj_slices, obj_names = scores_for(old, prior, k_values)
        assert obj_names == names and obj_slices == slices
    for k in k_values:
        for objective in args.objectives:
            flat = (uppers[k] < 0) if objective == 'max' else (obj_upper[(objective, k)] < 0)
            name = f'k{k}' if objective == 'max' else f'k{k}_{objective}'
            maps[name] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
            r['election'][name] = dict(k=k, objective=objective, selected=int(flat.sum()),
                                       fraction=float(flat.sum()) / flat.numel())
            print(f'ELECT {name} (objective={objective}): {int(flat.sum()):,} tiles', flush=True)
    order = torch.argsort(uppers[2], stable=True)[:256]
    flat = torch.zeros(uppers[2].numel(), dtype=torch.bool)
    flat[order[uppers[2][order] < 0]] = True
    maps['n256'] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
    r['election']['n256'] = dict(k=2, selected=int(flat.sum()), fraction=float(flat.sum()) / flat.numel())
    del uppers, order

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
    r['attention_backend'] = model.config._attn_implementation
    assert list(modules) == names
    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    for n, m in modules.items():
        want = torch.zeros(maps['n256'][n].numel(), dtype=torch.bool)
        want[bundle['maps'][FROZEN][n]] = True
        assert torch.equal(maps['n256'][n].reshape(-1), want), n
    r['frozen_map_reproduced'] = True
    print(f'REELECTION AT 256 MATCHES {FROZEN}', flush=True)
    save()

    base, alt, pristine = {}, {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            if target:
                # Baseline cached on CPU so it is not recomputed once per policy;
                # the E0M3 side is rebuilt on demand only where a policy switches.
                pristine[n] = m.weight.detach().to('cpu', copy=True)
                base[n] = quant_nvfp4_4over6(m.weight, 4, 16).to('cpu', copy=True)
            else:
                base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
                alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1',
                                       elect='always')
    r['source_weights_verified'] = True
    batches, r['data'] = data(tok, prior, LENGTH)
    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert excluded.isdisjoint(d['document_sha256'] for d in r['data']['c4_paper']['documents'])
    save()

    def act(module, inputs):
        return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n, m in modules.items():
            if not target:
                m.weight.copy_(base[n] if policy == 'four_over_six'
                               else apply_mask(base[n], alt[n], maps[policy][n].cuda()))
                continue
            b = base[n].cuda()
            if policy != 'four_over_six' and bool(maps[policy][n].any()):
                w = pristine[n].cuda()
                a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                b = apply_mask(b, a, maps[policy][n].cuda())
                del a, w
            m.weight.copy_(b)
            del b

    def ppl(values):
        t = torch.tensor(values, dtype=torch.float32) * LENGTH
        return float(torch.exp(t.sum() / (len(values) * LENGTH)))

    policies = ['four_over_six']
    if not args.skip_n256:
        policies.append('n256')
    for k in k_values:
        for objective in args.objectives:
            policies.append(f'k{k}' if objective == 'max' else f'k{k}_{objective}')
    r['policies'] = policies
    print(f'POLICIES {policies}', flush=True)
    with torch.no_grad():
        for policy in policies:
            install(policy)
            ev = {}
            for domain, bs in batches.items():
                values = []
                for ids in bs:
                    ids = ids.cuda()
                    logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
                    v = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              ids[:, 1:].reshape(-1)))
                    assert math.isfinite(v)
                    values.append(v)
                    del logits
                key = 'c4' if domain == 'c4_paper' else domain
                ev[key] = dict(ppl=ppl(values), nll=values, windows=len(values))
                print(f'PPL {policy} {key} {ev[key]["ppl"]:.6f}', flush=True)
            r['evaluation'][policy] = ev
            save()
    for h in handles:
        h.remove()

    if args.model == 'llama8b':
        ref = json.loads(Path('results/released_reproduction/job_335297/'
                              'llama-3.1-8b_four_over_six_w4a4/report.json').read_text())
        assert r['evaluation']['four_over_six']['wiki']['ppl'] == ref['ppl']['wikitext']
        assert r['evaluation']['four_over_six']['c4']['ppl'] == ref['ppl']['c4']
        r['released_baseline_exact'] = True
    assert digest_file(old / 'maps.json') == prior['map_sha256']
    r['status'] = 'complete'
    save()

    b = r['evaluation']['four_over_six']
    lines = [f'# Search-free k-SE count rule: {args.model}', '',
             f'Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no '
             f'cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; '
             f'its 256-tile prefix reproduces the frozen {FROZEN} map bitwise. Evaluation is the '
             f'released {LENGTH}-token protocol.', '',
             '| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for p in policies:
        e, el = r['evaluation'][p], r['election'].get(p)
        n = el['selected'] if el else 0
        frac = f"{100 * el['fraction']:.6f}%" if el else '0'
        lines.append(f"| {p} | {n:,} | {frac} | {e['wiki']['ppl']:.6f} | "
                     f"{e['wiki']['ppl'] - b['wiki']['ppl']:+.6f} | {e['c4']['ppl']:.6f} | "
                     f"{e['c4']['ppl'] - b['c4']['ppl']:+.6f} |")
    lines += ['', f'Scope: {len(modules)} text linear matrices, {r["total_tiles"]:,} type blocks '
              f'of 8x64.', '',
              'The constant k is fixed in advance and identical for every model; it is not chosen '
              'from these results. Two-SE intervals in the machine-readable report are descriptive '
              'evaluation-window intervals.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
