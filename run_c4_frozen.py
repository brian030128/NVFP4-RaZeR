"""Held-out C4 measurement of previously frozen format maps; no calibration."""
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
from run_conditional_format import save, sha
from run_conditional_model import paired

MODELS = ('opt350m', 'qwen06b', 'llama1b', 'olmo1b', 'pythia14b', 'qwen4b', 'llama8b')
POLICIES = ('four_over_six', 'pooled192', 'c4_64', 'mixed64', 'weight_mse')
REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
C4_PATH = 'en/c4-validation.00001-of-00008.json.gz'


def digest_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def heldout_data(tok, excluded):
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
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    stage = 'pooled_scale/model_332389' if args.model in ('qwen4b', 'llama8b') else 'pooled_confirmation/model_332349'
    old = root / f'results/{stage}_{args.model}'
    prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    files = ('run_c4_frozen.py', 'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'quantize/interacting_format.py', 'run_conditional_model.py', 'run_conditional_format.py',
             'results/c4_frozen/PROTOCOL.md')
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
             transformers_version=transformers.__version__, frozen_map_origin=str(old),
             map_file_sha256=digest_file(old / 'maps.pt'), prior_report_sha256=digest_file(old / 'report.json'),
             source_sha256={f: digest_file(root / f) for f in files},
             activation_convention='per-token FP32 factor; per-16-element E4M3 block scales',
             evaluation={}, suffix_intervention={})
    assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']
    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == r['source'] and bundle['revision'] == r['revision']
    assert bundle['baseline'] == 'FourOverSix' and bundle['alternative'] == 'E0M3 alpha1'
    assert tuple(bundle['type_block']) == (8, 64)
    maps = bundle['maps']
    r['selected_tiles'] = {p: sum(int(m.sum()) for m in maps[p].values()) for p in POLICIES[1:]}
    assert r['selected_tiles'] == prior['selected_tiles']
    assert r['selected_tiles']['pooled192'] <= 256
    save(out, r)
    model = AutoModelForCausalLM.from_pretrained(
        r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert set(modules) == set(prior['matrices'])
    assert all(set(ms) == set(modules) for ms in maps.values())
    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(base[n]).all() and torch.isfinite(alt[n]).all()
            assert all(ms[n].dtype == torch.bool and
                       ms[n].shape == (m.weight.shape[0] // 8, m.weight.shape[1] // 64)
                       for ms in maps.values())
    r['source_weights_verified'] = True
    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert len(excluded) == 192
    batches, r['c4_data'] = heldout_data(tok, excluded)
    save(out, r)
    print('DATA ' + json.dumps({k: v for k, v in r['c4_data'].items() if k not in ('documents', 'token_sha256')}), flush=True)

    def act(module, inputs):
        return (quantize_rows(inputs[0]), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n, m in modules.items():
            m.weight.copy_(base[n] if policy == 'four_over_six'
                           else apply_mask(base[n], alt[n], maps[policy][n].cuda()))

    with torch.no_grad():
        ids = batches[0].cuda()
        changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for p in ('four_over_six', 'pooled192'):
            install(p)
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            equal = torch.equal(a, b)
            r['suffix_intervention'][p] = dict(equal=equal, max_logit_difference=float((a.float() - b.float()).abs().max()),
                                                prefix_tokens=128, replacement_token_id=replacement)
            assert equal, f'Prefix depends on suffix: {p}'
        del a, b
        save(out, r)
        for p in POLICIES:
            install(p)
            values = []
            for i, batch in enumerate(batches):
                ids = batch.cuda()
                logits = model(input_ids=ids, use_cache=False).logits
                loss = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)))
                assert math.isfinite(loss)
                values.append(loss)
                if (i + 1) % 64 == 0:
                    print(f'EVAL {p} {i + 1}/256', flush=True)
            r['evaluation'][p] = dict(nll=values, ppl=math.exp(sum(values) / len(values)), scored_tokens=len(values) * 511)
            save(out, r)
            print(f'PPL {p} {r["evaluation"][p]["ppl"]:.6f}', flush=True)
    for h in handles:
        h.remove()
    r['contrasts'] = {p: paired(r['evaluation']['pooled192']['nll'], r['evaluation'][p]['nll'])
                      for p in POLICIES if p != 'pooled192'}
    assert digest_file(old / 'maps.pt') == r['map_file_sha256']
    r['frozen_map_unchanged'] = True
    r['status'] = 'complete'
    save(out, r)
    c = r['contrasts']['four_over_six']
    lines = [f'# Frozen-map held-out C4: {args.model}', '',
             '256 distinct validation documents, 512 tokens each; causal activation factors; no recalibration.', '',
             '| Policy | C4 PPL |', '|---|---:|']
    lines += [f'| {p} | {r["evaluation"][p]["ppl"]:.6f} |' for p in POLICIES]
    lines += ['', f'Pooled192 minus FourOverSix: ΔPPL {c["ppl_delta"]:+.6f}; '
              f'ΔNLL {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} (descriptive 2SE).', '',
              'Source weights verified; frozen map unchanged; calibration hash overlap zero; '
              'baseline and selected prefix independence exact.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
