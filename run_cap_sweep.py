"""Tile-count sweep for the frozen CE/KL election; re-election only, no rescoring.

Reuses each model's saved 192-sequence CE/KL score table and re-elects the same
rule at several counts. The 256 point must reproduce the already-frozen
pooled192 map exactly; that equality is the integrity check for this path.
Evaluation is the unchanged held-out C4 recipe from results/c4_frozen.
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
from run_conditional_format import save, sha
from run_conditional_model import paired

MODELS = ('olmo1b', 'pythia14b', 'qwen4b', 'llama8b')
COUNTS = (0, 16, 64, 256, 1024, 4096, 16384, 65536, None)
REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
C4_PATH = 'en/c4-validation.00001-of-00008.json.gz'


def policy_name(count):
    return 'four_over_six' if count == 0 else ('n_all' if count is None else f'n{count}')


def digest_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def heldout_data(tok, excluded):
    """Identical to results/c4_frozen: same shard, seed, exclusions and crop rule."""
    stream = load_dataset('allenai/c4', revision=REVISION,
                          data_files={'validation': C4_PATH}, split='validation', streaming=True)
    rng = random.Random(20260928)
    batches, docs, seen = [], [], set()
    for row in stream:
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        if digest in excluded or digest in seen:
            continue
        seen.add(digest)
        ids = tok(row['text'], return_tensors='pt').input_ids
        if ids.shape[1] < 512:
            continue
        offset = rng.randrange(ids.shape[1] - 512 + 1)
        batches.append(ids[:, offset:offset + 512].clone())
        docs.append(dict(document_sha256=digest, offset=offset))
        if len(batches) == 256:
            break
    assert len(batches) == 256, 'Insufficient held-out C4 documents'
    assert not ({d['document_sha256'] for d in docs} & excluded)
    return batches, dict(repo='allenai/c4', revision=REVISION, path=C4_PATH,
                         split='validation', seed=20260928, documents=docs,
                         token_sha256=[sha(b) for b in batches],
                         calibration_hash_overlap=0, excluded_calibration_documents=len(excluded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR',
                    help='checkout holding the frozen scores.pt/maps.pt artifacts')
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
    files = ('run_cap_sweep.py', 'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'quantize/interacting_format.py', 'quantize/relinearized_format.py',
             'run_conditional_model.py', 'run_conditional_format.py')
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
             transformers_version=transformers.__version__, frozen_map_origin=str(old),
             map_file_sha256=digest_file(old / 'maps.pt'), score_file_sha256=digest_file(old / 'scores.pt'),
             prior_report_sha256=digest_file(old / 'report.json'),
             source_sha256={f: digest_file(root / f) for f in files},
             activation_convention='per-token FP32 factor; per-16-element E4M3 block scales',
             counts=[policy_name(c) for c in COUNTS], election={}, evaluation={}, suffix_intervention={})
    assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']

    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == r['source'] and bundle['revision'] == r['revision']
    assert bundle['baseline'] == 'FourOverSix' and bundle['alternative'] == 'E0M3 alpha1'
    assert tuple(bundle['type_block']) == (8, 64)
    frozen = bundle['maps']['pooled192']

    # --- re-election from the saved table; one reduction, several prefixes ---
    record = torch.load(old / 'scores.pt', map_location='cpu', weights_only=True)
    names, slices = record['names'], record['slices']
    ce, kl = record['ce'].flatten(0, 1), record['kl'].flatten(0, 1)
    del record
    assert ce.shape[0] == 192 and ce.shape == kl.shape, 'Expected a 192-sequence paired table'
    unselected = torch.zeros(ce.shape[1], dtype=torch.bool)
    upper = common_descent_scores(ce, kl, unselected)
    mean_ce = ce.mean(0)
    del ce, kl
    assert torch.isfinite(upper).all()
    order = torch.argsort(upper, stable=True)
    eligible = int((upper < 0).sum())
    r['eligible_tiles'] = eligible
    r['total_tiles'] = int(upper.numel())
    print(f'ELIGIBLE {eligible}/{upper.numel()}', flush=True)

    maps = {}
    for count in COUNTS:
        policy = policy_name(count)
        flat = torch.zeros_like(unselected)
        if count != 0:
            take = order if count is None else order[:count]
            flat[take[upper[take] < 0]] = True
        maps[policy] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
        selected = int(flat.sum())
        r['election'][policy] = dict(
            requested=count, selected=selected,
            predicted_ce=float(mean_ce[flat].sum()) if selected else 0.0,
            upper_sum=float(upper[flat].sum()) if selected else 0.0,
            cap_binds=(count not in (0, None) and selected == count))
    del upper, order, mean_ce

    model = AutoModelForCausalLM.from_pretrained(
        r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert set(modules) == set(prior['matrices']) == set(names)

    # reshape flat tile vectors into the (rows/8, cols/64) grids the mask path expects
    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    assert all(torch.equal(maps['n256'][n], frozen[n]) for n in names), \
        'Re-election at 256 does not reproduce the frozen pooled192 map'
    r['frozen_map_reproduced'] = True
    print('REELECTION MATCHES FROZEN pooled192', flush=True)
    save(out, r)

    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(base[n]).all() and torch.isfinite(alt[n]).all()
    r['source_weights_verified'] = True

    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert len(excluded) == 192
    batches, r['c4_data'] = heldout_data(tok, excluded)
    save(out, r)
    print('DATA ' + json.dumps({k: v for k, v in r['c4_data'].items()
                                if k not in ('documents', 'token_sha256')}), flush=True)

    def act(module, inputs):
        return (quantize_rows(inputs[0]), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n, m in modules.items():
            m.weight.copy_(base[n] if policy == 'four_over_six'
                           else apply_mask(base[n], alt[n], maps[policy][n].cuda()))

    policies = [policy_name(c) for c in COUNTS]
    with torch.no_grad():
        ids = batches[0].cuda()
        changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for p in ('four_over_six', 'n256', 'n_all'):
            install(p)
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            equal = torch.equal(a, b)
            r['suffix_intervention'][p] = dict(
                equal=equal, max_logit_difference=float((a.float() - b.float()).abs().max()),
                prefix_tokens=128, replacement_token_id=replacement)
            assert equal, f'Prefix depends on suffix: {p}'
        del a, b
        save(out, r)
        for p in policies:
            install(p)
            values = []
            for i, batch in enumerate(batches):
                ids = batch.cuda()
                logits = model(input_ids=ids, use_cache=False).logits
                loss = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             ids[:, 1:].reshape(-1)))
                assert math.isfinite(loss)
                values.append(loss)
                if (i + 1) % 128 == 0:
                    print(f'EVAL {p} {i + 1}/256', flush=True)
            r['evaluation'][p] = dict(nll=values, ppl=math.exp(sum(values) / len(values)),
                                      scored_tokens=len(values) * 511)
            save(out, r)
            print(f'PPL {p} {r["evaluation"][p]["ppl"]:.6f}', flush=True)
    for h in handles:
        h.remove()

    r['contrasts'] = {p: paired(r['evaluation'][p]['nll'], r['evaluation']['four_over_six']['nll'])
                      for p in policies if p != 'four_over_six'}
    assert digest_file(old / 'maps.pt') == r['map_file_sha256']
    r['status'] = 'complete'
    save(out, r)

    lines = [f'# Tile-count sweep on held-out C4: {args.model}', '',
             f'Re-election of the unchanged CE/KL two-SE rule from the saved 192-sequence score '
             f'table at several counts. The 256 row reproduces the frozen pooled192 map exactly. '
             f'{eligible} of {r["total_tiles"]} tiles have a negative two-SE score.', '',
             'ΔPPL and ΔNLL are relative to FourOverSix; negative is better. Two-SE intervals are '
             'descriptive evaluation-window intervals and do not account for calibration-draw '
             'variability or multiple comparisons across counts.', '',
             '| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE |', '|---|---:|---:|---:|---:|']
    for p in policies:
        sel = r['election'][p]['selected']
        ppl = r['evaluation'][p]['ppl']
        if p == 'four_over_six':
            lines.append(f'| baseline | {sel} | {ppl:.6f} | — | — |')
        else:
            c = r['contrasts'][p]
            lines.append(f'| {p[1:]} | {sel} | {ppl:.6f} | {c["ppl_delta"]:+.6f} | '
                         f'{c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', 'Selection uses only the saved calibration score table; C4 evaluation documents '
              'have zero hash overlap with the 192 calibration documents. This sweep measures the '
              'shape of the count/gain curve. Choosing a count from this table would be selection '
              'on the evaluation set; any count rule derived here requires validation on domains '
              'untouched by this sweep.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
