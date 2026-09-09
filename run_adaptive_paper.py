"""Paper-aligned adaptive tile count: choose N on calibration data, evaluate at 2048.

Re-elects the unchanged CE/KL two-SE rule from the frozen math/code score shards
at several nested counts, picks the count with the lowest actual loss on
OpenWebMath and CodeParrot documents disjoint from the 128 scoring documents,
and evaluates the chosen map under the released 2048-token protocol. WikiText-2
and C4 are never consulted for the choice.

Re-election at 256 must reproduce the frozen fixed256_math_code128 map bitwise.
"""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.relinearized_format import common_descent_scores
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import sha

MODELS = {'llama8b': '333779', 'qwen4b': '333779'}
COUNTS = (256, 1024, 4096, 16384, 65536, None)
FROZEN = 'fixed256_math_code128'
LENGTH = 2048
SELECT_DOCS = 64  # per source


def name(count):
    return 'n_all' if count is None else f'n{count}'


def selection_data(tok, prior, seed=20261002):
    """64 OpenWebMath + 64 CodeParrot documents disjoint from the scoring set."""
    used = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
    assert len(used) == 128
    batches, meta = [], {}
    for source, field in (('math', 'text'), ('code', 'content')):
        m = prior['fit'][source]
        stream = load_dataset(m['repo'], revision=m['revision'],
                              data_files={'train': m['path']}, split='train', streaming=True)
        rng = random.Random(seed)
        picked, docs = [], []
        for row in stream:
            text = row[field]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in used:
                continue
            used.add(digest)
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < LENGTH + 1:
                continue
            offset = rng.randrange(ids.shape[1] - LENGTH + 1)
            picked.append(ids[:, offset:offset + LENGTH].clone())
            docs.append(dict(document_sha256=digest, offset=offset))
            if len(picked) == SELECT_DOCS:
                break
        assert len(picked) == SELECT_DOCS, source
        batches += picked
        meta[source] = dict(repo=m['repo'], revision=m['revision'], path=m['path'],
                            documents=docs, token_sha256=[sha(b) for b in picked])
    return batches, meta


def elect(old, prior):
    """Global CE/KL two-SE ranking rebuilt from the per-module score shards."""
    names = list(prior['matrices'])
    uppers, slices, offset = [], {}, 0
    for i, n in enumerate(names):
        shard = torch.load(old / 'scores' / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert shard['name'] == n, (shard['name'], n)
        ce, kl = shard['ce'], shard['kl']
        assert ce.shape[0] == 128 and ce.shape == kl.shape
        u = common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))
        assert torch.isfinite(u).all()
        uppers.append(u)
        slices[n] = (offset, offset + u.numel())
        offset += u.numel()
        del shard, ce, kl
    upper = torch.cat(uppers)
    del uppers
    return upper, torch.argsort(upper, stable=True), slices, names


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    old = Path(args.stage_root) / f'results/math_code_adaptive/calibration_{MODELS[args.model]}_{args.model}'
    prior = json.loads((old / 'report.json').read_text())
    bundle = json.loads((old / 'maps.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert digest_file(old / 'maps.json') == prior['map_sha256']
    assert transformers.__version__ == prior['transformers_version']

    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__, calibration=str(old),
             map_sha256=prior['map_sha256'], length=LENGTH,
             activation='FourOverSix tensor-wide factor', wiki_use_cache=True, c4_use_cache=False,
             calibration_sources=['OpenWebMath', 'CodeParrot'],
             uses_c4_calibration=False, uses_wiki_calibration=False,
             counts=[name(c) for c in COUNTS], election={}, selection_loss={},
             evaluation={}, chosen={})

    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')

    save()
    upper, order, slices, names = elect(old, prior)
    eligible = int((upper < 0).sum())
    r['eligible_tiles'] = eligible
    r['total_tiles'] = int(upper.numel())
    print(f'ELIGIBLE {eligible}/{upper.numel()}', flush=True)

    maps = {}
    for count in COUNTS:
        flat = torch.zeros(upper.numel(), dtype=torch.bool)
        take = order if count is None else order[:count]
        flat[take[upper[take] < 0]] = True
        maps[name(count)] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
        r['election'][name(count)] = dict(requested=count, selected=int(flat.sum()))
    del upper, order

    model = AutoModelForCausalLM.from_pretrained(
        prior['source'], torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(prior['source'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == names
    r['attention_backend'] = model.config._attn_implementation

    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    # The frozen map is stored as flat indices; rebuilding it must agree exactly.
    for n, m in modules.items():
        want = torch.zeros(maps['n256'][n].numel(), dtype=torch.bool)
        want[bundle['maps'][FROZEN][n]] = True
        assert torch.equal(maps['n256'][n].reshape(-1), want), n
    r['frozen_map_reproduced'] = True
    print(f'REELECTION AT 256 MATCHES {FROZEN}', flush=True)
    save()

    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(base[n]).all() and torch.isfinite(alt[n]).all()
    r['source_weights_verified'] = True

    select, r['selection_data'] = selection_data(tok, prior)
    batches, r['data'] = data(tok, prior, LENGTH)
    excluded = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
    assert excluded.isdisjoint(d['document_sha256'] for d in r['data']['c4_paper']['documents'])
    save()
    print(f'DATA select={len(select)} wiki={len(batches["wiki"])} c4={len(batches["c4_paper"])}',
          flush=True)

    def act(module, inputs):
        return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n, m in modules.items():
            m.weight.copy_(base[n] if policy == 'four_over_six'
                           else apply_mask(base[n], alt[n], maps[policy][n].cuda()))

    def losses(bs, use_cache):
        values = []
        for ids in bs:
            ids = ids.cuda()
            logits = model(input_ids=ids, use_cache=use_cache).logits
            v = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                      ids[:, 1:].reshape(-1)))
            assert math.isfinite(v)
            values.append(v)
            del logits
        return values

    def ppl(values):
        # Same FP32 aggregation as the released run_ppl.py.
        t = torch.tensor(values, dtype=torch.float32) * LENGTH
        return float(torch.exp(t.sum() / (len(values) * LENGTH)))

    with torch.no_grad():
        for policy in r['counts']:
            install(policy)
            v = losses(select, use_cache=False)
            r['selection_loss'][policy] = dict(nll=v, mean=sum(v) / len(v))
            save()
            print(f'SELECT {policy} {r["selection_loss"][policy]["mean"]:.6f}', flush=True)
        best = min(r['counts'], key=lambda p: r['selection_loss'][p]['mean'])
        r['chosen'] = dict(policy=best, **r['election'][best])
        print(f'CHOSEN {best} ({r["election"][best]["selected"]} tiles)', flush=True)
        save()

        for policy in dict.fromkeys(['four_over_six', 'n256', best]):
            install(policy)
            ev = {}
            for domain, bs in batches.items():
                v = losses(bs, use_cache=(domain == 'wiki'))
                key = 'c4' if domain == 'c4_paper' else domain
                ev[key] = dict(ppl=ppl(v), nll=v, windows=len(v))
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

    def row(policy, label):
        e = r['evaluation'][policy]
        return (f"| {label} | {r['election'].get(policy, {}).get('selected', 0)} | "
                f"{e['wiki']['ppl']:.6f} | {e['c4']['ppl']:.6f} |")

    lines = [f'# Paper-aligned adaptive tile count: {args.model}', '',
             f'Counts are chosen by the lowest actual loss on {len(select)} OpenWebMath and '
             f'CodeParrot documents disjoint from the 128 scoring documents. WikiText-2 and C4 '
             f'are never consulted for the choice. Evaluation is the released {LENGTH}-token '
             f'protocol. {eligible} of {r["total_tiles"]} tiles are eligible; re-election at 256 '
             f'reproduces the frozen {FROZEN} map bitwise.', '',
             '| Count | Selected | Selection NLL |', '|---|---:|---:|']
    lines += [f"| {p} | {r['election'][p]['selected']:,} | {r['selection_loss'][p]['mean']:.6f} |"
              for p in r['counts']]
    lines += ['', f"Chosen: **{best}** ({r['election'][best]['selected']:,} tiles).", '',
              '| Policy | E0M3 blocks | WikiText-2 | C4 |', '|---|---:|---:|---:|',
              row('four_over_six', 'FourOverSix baseline'), row('n256', 'MixFP4 fixed 256')]
    if best != 'n256':
        lines.append(row(best, f'MixFP4 adaptive ({best})'))
    lines += ['', 'The count is selected on calibration-source documents only; these perplexities '
              'are a readout of that decision, not the criterion for it. Two-SE intervals in the '
              'machine-readable report are descriptive and do not cover calibration-draw or '
              'selection-draw variability.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
