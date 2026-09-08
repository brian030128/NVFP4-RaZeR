"""Replay frozen maps and exact inputs with token-local activation factors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from run_conditional_format import sha, save
from run_conditional_model import paired
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.interacting_format import apply_mask
from quantize.causal_four_over_six import quantize_rows


def replay_data(tok, recorded):
    result = {}
    for domain, meta in recorded.items():
        wanted = {d['document_sha256']: d for d in meta['documents']}; found = {}
        ds = load_dataset(meta['repo'], meta['config'], revision=meta['revision'], split='test', streaming=True)
        for row in ds:
            text = row[meta['field']]; digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in wanted: continue
            ids = tok(text, return_tensors='pt').input_ids; offset = wanted[digest]['offset']
            found[digest] = ids[:, offset:offset+512].clone()
            if len(found) == len(wanted): break
        bs = [found[d['document_sha256']] for d in meta['documents']]
        assert [sha(b) for b in bs] == meta['token_sha256']
        result[domain] = bs
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent; out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    old = root/f'results/pooled_confirmation/model_332349_{args.model}'
    prior = json.loads((old/'report.json').read_text()); assert prior['status'] == 'complete'
    files = ['run_causal_replay.py', 'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'tests/test_causal_four_over_six.py', 'results/causal_replay/PROTOCOL.md']
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), frozen_map_origin=str(old),
             map_file_sha256=hashlib.sha256((old/'maps.pt').read_bytes()).hexdigest(),
             source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
             scale_convention='one FP32 activation factor per token row; per16-element E4M3 scales',
             evaluation={}, suffix_intervention={}, confirmation_data=prior['confirmation_data'])
    assert r['source_sha256']['quantize/quantizer.py'] == prior['source_sha256']['quantize/quantizer.py']
    save(out, r)
    bundle = torch.load(old/'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == r['source'] and bundle['revision'] == r['revision']
    maps = bundle['maps']
    model = AutoModelForCausalLM.from_pretrained(r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
                                               device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert set(modules) == set(prior['matrices'])
    base = {}; alt = {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256']
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8,64), clip='a1', elect='always')
            assert all(ms[n].shape == (m.weight.shape[0]//8, m.weight.shape[1]//64) for ms in maps.values())
    batches = replay_data(tok, r['confirmation_data']); r['exact_input_replay'] = True
    row_mode = True
    def act(module, inputs):
        x = inputs[0]
        return (quantize_rows(x) if row_mode else quant_nvfp4_4over6(x, 4, 16), *inputs[1:])
    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    def install(policy):
        for n, m in modules.items():
            m.weight.copy_(base[n] if policy == 'four_over_six' else apply_mask(base[n], alt[n], maps[policy][n].cuda()))
    with torch.no_grad():
        ids = batches['literature'][0].cuda(); changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for policy, mode in [('four_over_six', False), ('four_over_six', True), ('pooled192', True)]:
            install(policy); row_mode = mode
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            equal = torch.equal(a, b)
            key = policy+('_row' if mode else '_window')
            r['suffix_intervention'][key] = dict(equal=equal, max_logit_difference=float((a.float()-b.float()).abs().max()),
                                                prefix_tokens=128, replacement_token_id=replacement)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            if mode: assert equal, f'Prefix depends on suffix under row scaling: {key}'
        del a, b
    row_mode = True; save(out, r)
    print('PREFIX '+json.dumps(r['suffix_intervention']), flush=True)
    policies = ('four_over_six', 'weight_mse', 'c4_64', 'mixed64', 'pooled192')
    for p in policies:
        with torch.no_grad():
            install(p); result = {}
            for d, bs in batches.items():
                values = []
                for batch in bs:
                    ids = batch.cuda(); logits = model(input_ids=ids, use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))))
                result[d] = dict(nll=values, ppl=math.exp(sum(values)/len(values)))
        r['evaluation'][p] = result; save(out, r)
        print(f'EVAL {p} '+json.dumps({d: v['ppl'] for d, v in result.items()}), flush=True)
    for h in handles: h.remove()
    r['contrasts'] = {d: {p: paired(r['evaluation']['pooled192'][d]['nll'], r['evaluation'][p][d]['nll']) for p in policies[:-1]} for d in batches}
    r['prior_contrasts'] = prior['contrasts']
    r['status'] = 'complete'; save(out, r)
    lines = [f'# Frozen-map causal replay: {args.model}', '',
             'All policies use one FP32 activation factor per token; maps and inputs replay unchanged.', '',
             '| Domain | FourOverSix | Weight MSE | C4-64 | Mixed64 | Pooled192 | ΔNLL ±2SE vs matched baseline |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in batches:
        c = r['contrasts'][d]['four_over_six']
        lines.append('| '+d+' | '+' | '.join(f'{r["evaluation"][p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', 'Suffix intervention:', '', '```json', json.dumps(r['suffix_intervention'], indent=2), '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
