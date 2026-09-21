"""Causal WikiText-2 replay of existing pooled maps, without calibration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from quantize.causal_four_over_six import quantize_rows
from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_conditional_model import paired

POLICIES = ('four_over_six', 'pooled192', 'c4_64', 'mixed64', 'weight_mse')
WIKI_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def wiki_data(tok, excluded):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=WIKI_REVISION, split='test')
    texts = list(ds['text'])
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts if t.strip()]
    overlap = sorted(set(hashes) & excluded)
    assert not overlap, 'A test row exactly matches a calibration document'
    text = '\n\n'.join(texts)
    ids = tok(text, return_tensors='pt').input_ids
    count = ids.shape[1] // 512
    assert count >= 2
    batches = [ids[:, i*512:(i+1)*512].clone() for i in range(count)]
    return batches, dict(repo='Salesforce/wikitext', config='wikitext-2-raw-v1', revision=WIKI_REVISION,
                         split='test', concatenation='two newlines between every dataset row, including empty rows',
                         text_sha256=hashlib.sha256(text.encode()).hexdigest(), all_tokens_sha256=sha(ids),
                         nonempty_row_sha256=hashes, calibration_exact_row_overlap=overlap,
                         excluded_calibration_documents=len(excluded), total_tokens=ids.shape[1],
                         window_tokens=512, windows=count, omitted_tail_tokens=ids.shape[1] % 512,
                         offsets=[i*512 for i in range(count)], token_sha256=[sha(b) for b in batches])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('qwen4b', 'llama8b', 'qwen27b'), required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(12 if args.model == 'qwen27b' else 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent; out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    target = args.model == 'qwen27b'
    old = root / ('results/pooled_qwen27b/model_332840' if target else f'results/pooled_scale/model_332389_{args.model}')
    prior = json.loads((old / 'report.json').read_text())
    c4_path = old / 'report.json' if target else root / f'results/c4_frozen/model_332781_{args.model}/report.json'
    c4 = json.loads(c4_path.read_text())
    assert prior['status'] == c4['status'] == 'complete' and prior['maps_frozen']
    assert transformers.__version__ == c4['transformers_version']
    files = ('run_wiki_frozen.py', 'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'quantize/interacting_format.py', 'run_conditional_model.py', 'run_conditional_format.py',
             'results/wiki_frozen/PROTOCOL.md')
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__, transformers_version=transformers.__version__,
             frozen_map_origin=str(old), prior_report_sha256=digest_file(old / 'report.json'),
             map_file_sha256=digest_file(old / 'maps.pt'), source_sha256={f: digest_file(root / f) for f in files},
             activation_convention='per-token FP32 factor; per-16-element E4M3 block scales',
             evaluation={}, suffix_intervention={})
    assert r['map_file_sha256'] == c4['map_file_sha256']
    for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py'):
        assert r['source_sha256'][f] == c4['source_sha256'][f]
    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == r['source'] and bundle['revision'] == r['revision']
    assert bundle['baseline'] == 'FourOverSix' and bundle['alternative'] == 'E0M3 alpha1'
    assert tuple(bundle['type_block']) == (8, 64)
    maps = bundle['maps']
    r['selected_tiles'] = {p: sum(int(m.sum()) for m in maps[p].values()) for p in POLICIES[1:]}
    assert r['selected_tiles'] == prior['selected_tiles'] and r['selected_tiles']['pooled192'] <= 256
    save(out, r)
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            r['source'], revision=r['revision'], dtype=torch.bfloat16, attn_implementation='eager',
            device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}, output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        model.eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(
            r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert set(modules) == set(prior['matrices']) and all(set(ms) == set(modules) for ms in maps.values())
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    input_device = model.get_input_embeddings().weight.device
    base, alt = {}, {}
    with torch.no_grad():
        for i, (n, m) in enumerate(modules.items()):
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            b = quant_nvfp4_4over6(m.weight, 4, 16)
            a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(b).all() and torch.isfinite(a).all()
            assert all(ms[n].dtype == torch.bool and ms[n].shape == (m.weight.shape[0]//8, m.weight.shape[1]//64)
                       for ms in maps.values())
            base[n], alt[n] = (b.cpu().pin_memory(), a.cpu().pin_memory()) if target else (b, a)
            if (i+1) % 64 == 0: print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
        del a, b
    r['source_weights_verified'] = True
    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert len(excluded) == 192
    batches, r['wiki_data'] = wiki_data(tok, excluded)
    print(f'WIKI {len(batches)} windows; omitted tail {r["wiki_data"]["omitted_tail_tokens"]} tokens', flush=True)
    save(out, r)
    def act(module, inputs): return (quantize_rows(inputs[0]), *inputs[1:])
    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    def install(policy):
        for n, m in modules.items():
            b = base[n].to(m.weight.device, non_blocking=True)
            m.weight.copy_(b if policy == 'four_over_six' else apply_mask(
                b, alt[n].to(m.weight.device, non_blocking=True), maps[policy][n].to(m.weight.device)))
    with torch.no_grad():
        ids = batches[0].to(input_device); changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for p in ('four_over_six', 'pooled192'):
            install(p)
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            r['suffix_intervention'][p] = dict(equal=torch.equal(a, b), max_logit_difference=float((a.float()-b.float()).abs().max()),
                                               prefix_tokens=128, replacement_token_id=replacement)
            assert torch.equal(a, b), p
        del a, b
        save(out, r)
        for p in POLICIES:
            install(p); values = []
            for i, batch in enumerate(batches):
                ids = batch.to(input_device); logits = model(input_ids=ids, use_cache=False).logits
                loss = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              ids[:, 1:].reshape(-1).to(logits.device)))
                assert math.isfinite(loss); values.append(loss)
                if (i+1) % 64 == 0: print(f'EVAL {p} {i+1}/{len(batches)}', flush=True)
            r['evaluation'][p] = dict(nll=values, ppl=math.exp(sum(values)/len(values)), scored_tokens=len(values)*511)
            save(out, r); print(f'PPL {p} {r["evaluation"][p]["ppl"]:.6f}', flush=True)
    for h in handles: h.remove()
    r['contrasts'] = {p: paired(r['evaluation']['pooled192']['nll'], r['evaluation'][p]['nll']) for p in POLICIES if p != 'pooled192'}
    assert digest_file(old / 'maps.pt') == r['map_file_sha256']
    r['frozen_map_unchanged'] = True; r['status'] = 'complete'; save(out, r)
    c = r['contrasts']['four_over_six']
    lines = [f'# Frozen-map causal WikiText-2: {args.model}', '',
             f'{len(batches)} nonoverlapping 512-token test windows; {r["wiki_data"]["omitted_tail_tokens"]} trailing tokens omitted. '
             'No calibration or map selection. All policies use causal per-token activation factors.', '',
             '| Policy | WikiText-2 PPL |', '|---|---:|']
    lines += [f'| {p} | {r["evaluation"][p]["ppl"]:.6f} |' for p in POLICIES]
    lines += ['', f'Pooled192 minus FourOverSix: ΔPPL {c["ppl_delta"]:+.6f}; '
              f'ΔNLL {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} (descriptive paired 2SE).', '',
              'Pinned source weights, quantizer code and frozen maps verified. Exact prefix independence passes for '
              'baseline and pooled192. No exact overlap between nonempty WikiText rows and calibration document hashes. '
              'Adjacent windows may share articles; two-SE does not account for that dependence. '
              'These are simulated nonhead-text-linear W4A4 reference-text scores.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
