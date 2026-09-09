"""Choose the tile count from calibration data, then read out held-out C4.

The count sweep showed the count/gain curve has a model-dependent interior
optimum. This script asks whether a rule that never sees the evaluation set can
find it: measure each nested prefix's actual loss on calibration-source
documents and take the argmin. Two selection sets are measured -- the 192
scoring documents themselves (`fit`, which the scores were computed on) and a
disjoint 192 drawn from the same three sources (`sel`). Only `sel` is a
legitimate selection set; `fit` is reported to show the overfitting it invites.
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

from quantize.causal_four_over_six import quantize_rows
from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.relinearized_format import common_descent_scores
from run_cap_sweep import COUNTS, MODELS, heldout_data, policy_name
from run_conditional_format import save, sha
from run_conditional_model import paired

WEB_SEED = 20260920
OTHER_SEED = 20260926


def digest_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect(tok, repo, revision, path, split, field, seed, seen, total):
    """Replay a recorded calibration stream and continue it past the first 64."""
    stream = load_dataset(repo, revision=revision, data_files={split: path},
                          split=split, streaming=True)
    rng = random.Random(seed)
    batches, docs = [], []
    for row in stream:
        text = row[field]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(text, return_tensors='pt').input_ids
        if ids.shape[1] < 512:
            continue
        offset = rng.randrange(ids.shape[1] - 512 + 1)
        batches.append(ids[:, offset:offset + 512].clone())
        docs.append(dict(document_sha256=digest, offset=offset))
        if len(batches) == total:
            return batches, docs
    raise RuntimeError(f'Insufficient documents in {repo}')


def calibration_sets(tok, fit_meta):
    """Rebuild the recorded 192 scoring documents plus a disjoint 192."""
    sources = [
        ('web', 'allenai/c4', fit_meta['web']['revision'], fit_meta['web']['path'],
         'train', 'text', WEB_SEED),
        ('math', fit_meta['math']['repo'], fit_meta['math']['revision'],
         fit_meta['math']['path'], 'train', 'text', OTHER_SEED),
        ('code', fit_meta['code']['repo'], fit_meta['code']['revision'],
         fit_meta['code']['path'], 'train', 'content', OTHER_SEED),
    ]
    seen = set()
    fit, sel, meta = [], [], {}
    for name, repo, revision, path, split, field, seed in sources:
        batches, docs = collect(tok, repo, revision, path, split, field, seed, seen, 128)
        recorded = fit_meta[name]['documents']
        assert docs[:64] == recorded, f'{name} calibration stream did not replay'
        fit += batches[:64]
        sel += batches[64:]
        meta[name] = dict(repo=repo, revision=revision, path=path,
                          selection_documents=docs[64:],
                          selection_token_sha256=[sha(b) for b in batches[64:]])
    assert len(fit) == 192 and len(sel) == 192
    fit_hashes = {d['document_sha256'] for m in fit_meta.values() for d in m['documents']}
    sel_hashes = {d['document_sha256'] for m in meta.values() for d in m['selection_documents']}
    assert len(sel_hashes) == 192 and not (fit_hashes & sel_hashes), 'Selection set overlaps scoring set'
    return fit, sel, meta, fit_hashes | sel_hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    stage = 'pooled_scale/model_332389' if args.model in ('qwen4b', 'llama8b') else 'pooled_confirmation/model_332349'
    old = Path(args.stage_root) / f'results/{stage}_{args.model}'
    prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    files = ('run_adaptive_count.py', 'run_cap_sweep.py', 'quantize/causal_four_over_six.py',
             'quantize/quantizer.py', 'quantize/interacting_format.py',
             'quantize/relinearized_format.py')
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
             transformers_version=transformers.__version__, frozen_map_origin=str(old),
             map_file_sha256=digest_file(old / 'maps.pt'), score_file_sha256=digest_file(old / 'scores.pt'),
             source_sha256={f: digest_file(root / f) for f in files},
             counts=[policy_name(c) for c in COUNTS],
             election={}, calibration_loss={}, evaluation={}, chosen={})
    assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']

    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    frozen = bundle['maps']['pooled192']
    record = torch.load(old / 'scores.pt', map_location='cpu', weights_only=True)
    names, slices = record['names'], record['slices']
    ce, kl = record['ce'].flatten(0, 1), record['kl'].flatten(0, 1)
    del record
    assert ce.shape[0] == 192 and ce.shape == kl.shape
    upper = common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))
    del ce, kl
    assert torch.isfinite(upper).all()
    order = torch.argsort(upper, stable=True)
    r['eligible_tiles'] = int((upper < 0).sum())
    r['total_tiles'] = int(upper.numel())

    maps = {}
    for count in COUNTS:
        policy = policy_name(count)
        flat = torch.zeros(upper.numel(), dtype=torch.bool)
        if count != 0:
            take = order if count is None else order[:count]
            flat[take[upper[take] < 0]] = True
        maps[policy] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
        r['election'][policy] = dict(requested=count, selected=int(flat.sum()))
    del upper, order

    model = AutoModelForCausalLM.from_pretrained(
        r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert set(modules) == set(prior['matrices']) == set(names)
    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    assert all(torch.equal(maps['n256'][n], frozen[n]) for n in names), \
        'Re-election at 256 does not reproduce the frozen pooled192 map'
    r['frozen_map_reproduced'] = True
    save(out, r)

    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    r['source_weights_verified'] = True

    fit, sel, r['selection_data'], excluded = calibration_sets(tok, prior['fit'])
    print(f'DATA fit=192 sel=192 excluded={len(excluded)}', flush=True)
    heldout, r['c4_data'] = heldout_data(tok, excluded)
    save(out, r)

    def act(module, inputs):
        return (quantize_rows(inputs[0]), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n, m in modules.items():
            m.weight.copy_(base[n] if policy == 'four_over_six'
                           else apply_mask(base[n], alt[n], maps[policy][n].cuda()))

    def losses(batches):
        values = []
        for batch in batches:
            ids = batch.cuda()
            logits = model(input_ids=ids, use_cache=False).logits
            loss = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                         ids[:, 1:].reshape(-1)))
            assert math.isfinite(loss)
            values.append(loss)
        return values

    policies = [policy_name(c) for c in COUNTS]
    with torch.no_grad():
        for p in policies:
            install(p)
            r['calibration_loss'][p] = dict(fit=losses(fit), sel=losses(sel))
            means = {k: sum(v) / len(v) for k, v in r['calibration_loss'][p].items()}
            save(out, r)
            print(f'CAL {p} fit {means["fit"]:.6f} sel {means["sel"]:.6f}', flush=True)

        for key in ('fit', 'sel'):
            best = min(policies, key=lambda p: sum(r['calibration_loss'][p][key]) / 192)
            r['chosen'][key] = dict(policy=best, requested=r['election'][best]['requested'],
                                    selected=r['election'][best]['selected'])
            print(f'CHOSEN[{key}] {best}', flush=True)

        readout = ['four_over_six', 'n256', r['chosen']['sel']['policy'], r['chosen']['fit']['policy']]
        for p in dict.fromkeys(readout):
            install(p)
            values = losses(heldout)
            r['evaluation'][p] = dict(nll=values, ppl=math.exp(sum(values) / len(values)),
                                      scored_tokens=len(values) * 511)
            save(out, r)
            print(f'PPL {p} {r["evaluation"][p]["ppl"]:.6f}', flush=True)
    for h in handles:
        h.remove()

    r['contrasts'] = {p: paired(r['evaluation'][p]['nll'], r['evaluation']['four_over_six']['nll'])
                      for p in r['evaluation'] if p != 'four_over_six'}
    r['adaptive_vs_fixed'] = {
        key: paired(r['evaluation'][r['chosen'][key]['policy']]['nll'], r['evaluation']['n256']['nll'])
        for key in ('fit', 'sel')}
    r['status'] = 'complete'
    save(out, r)

    lines = [f'# Calibration-chosen tile count: {args.model}', '',
             f'Counts are chosen by the lowest actual next-token loss over calibration-source '
             f'documents; the held-out C4 set is never consulted for selection. `sel` is 192 '
             f'documents disjoint from the 192 scoring documents; `fit` is the scoring set itself. '
             f'{r["eligible_tiles"]} of {r["total_tiles"]} tiles are eligible.', '',
             '| Count | Selected | fit NLL | sel NLL |', '|---|---:|---:|---:|']
    for p in policies:
        c = r['calibration_loss'][p]
        lines.append(f'| {p} | {r["election"][p]["selected"]} | '
                     f'{sum(c["fit"]) / 192:.6f} | {sum(c["sel"]) / 192:.6f} |')
    lines += ['', '| Policy | Count | C4 PPL | ΔPPL vs FourOverSix | ΔNLL ±2SE |', '|---|---:|---:|---:|---:|']
    for p in r['evaluation']:
        sel_n = r['election'][p]['selected']
        ppl = r['evaluation'][p]['ppl']
        if p == 'four_over_six':
            lines.append(f'| FourOverSix | 0 | {ppl:.6f} | — | — |')
        else:
            c = r['contrasts'][p]
            lines.append(f'| {p} | {sel_n} | {ppl:.6f} | {c["ppl_delta"]:+.6f} | '
                         f'{c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    for key in ('sel', 'fit'):
        a = r['adaptive_vs_fixed'][key]
        lines += ['', f'Chosen on `{key}`: {r["chosen"][key]["policy"]} '
                  f'({r["chosen"][key]["selected"]} tiles). Against the fixed 256 map: '
                  f'ΔPPL {a["ppl_delta"]:+.6f}, ΔNLL {a["mean_nll"]:+.6f} ±{a["two_se"]:.6f}.']
    lines += ['', 'The count is selected on calibration-source documents only. These held-out C4 '
              'numbers are a readout of that decision, not the criterion for it. Two-SE intervals '
              'are descriptive and do not account for calibration-draw variability. A count rule '
              'validated here still requires confirmation on the untouched literature, science, '
              'government and WikiText families.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
