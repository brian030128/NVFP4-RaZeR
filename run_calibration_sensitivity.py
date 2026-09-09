"""Evaluate every frozen calibration subset using the current causal convention."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from calibration_sensitivity import calibration_subsets, validate_losses
from prepare_calibration_sensitivity import ALIASES
from quantize.causal_four_over_six import quantize_rows
from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file, heldout_data
from run_conditional_format import save, sha
from run_conditional_model import paired
from run_pooled_scale import data as transfer_data
from run_wiki_frozen import wiki_data

DOMAINS = ('c4', 'wiki', 'literature', 'science', 'government')
POLICIES = ('four_over_six', *calibration_subsets(), 'weight_mse')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    args = ap.parse_args()
    assert len(args.domains) == len(set(args.domains))
    prepared = Path(args.prepared) / 'maps.json'
    frozen = json.loads(prepared.read_text())
    assert frozen['status'] == 'complete' and frozen['original_maps_reproduced_exactly']
    assert frozen['subsets'] == calibration_subsets()
    target = frozen['model'] == 'qwen27b'
    torch.set_num_threads(12 if target else 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    old = Path(frozen['origin'])
    prior = json.loads((old / 'report.json').read_text())
    assert digest_file(old / 'report.json') == frozen['prior_report_sha256']
    assert digest_file(old / 'maps.pt') == frozen['original_maps_sha256']
    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    assert bundle['source'] == frozen['source'] and bundle['revision'] == frozen['revision']
    assert bundle['baseline'] == 'FourOverSix' and bundle['alternative'] == 'E0M3 alpha1'
    assert tuple(bundle['type_block']) == (8, 64)
    assert transformers.__version__ == prior['transformers_version']
    files = ('run_calibration_sensitivity.py', 'calibration_sensitivity.py',
             'quantize/causal_four_over_six.py', 'quantize/quantizer.py', 'quantize/interacting_format.py',
             'run_c4_frozen.py', 'run_wiki_frozen.py', 'run_pooled_scale.py',
             'results/calibration_sensitivity/PROTOCOL.md')
    r = dict(status='running', model=frozen['model'], source=frozen['source'], revision=frozen['revision'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__, prepared=str(prepared),
             prepared_sha256=digest_file(prepared), source_sha256={f: digest_file(f) for f in files},
             activation_convention='per-token FP32 factor; per-16-element E4M3 block scales',
             block_statistics=frozen['block_statistics'], evaluation={}, data={}, suffix_intervention={},
             evaluated_domains=args.domains)
    for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py'):
        assert r['source_sha256'][f] == prior['source_sha256'][f]
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
        r['device_map'] = model.hf_device_map
    else:
        model = AutoModelForCausalLM.from_pretrained(
            r['source'], revision=r['revision'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == list(frozen['matrices'])
    maps = {}
    for policy, sparse in frozen['maps'].items():
        assert set(sparse) == set(modules)
        maps[policy] = {}
        for name, m in modules.items():
            shape = (m.weight.shape[0] // 8, m.weight.shape[1] // 64)
            flat = torch.zeros(shape[0] * shape[1], dtype=torch.bool)
            indices = sparse[name]
            assert len(indices) == len(set(indices)) and all(0 <= i < flat.numel() for i in indices)
            flat[indices] = True
            maps[policy][name] = flat.reshape(shape)
        assert sum(int(v.sum()) for v in maps[policy].values()) == r['block_statistics'][policy]['selected_blocks'] <= 256
    maps['weight_mse'] = bundle['maps']['weight_mse']
    for policy, alias in ALIASES.items():
        assert all(torch.equal(maps[policy][n], bundle['maps'][alias][n]) for n in modules)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    input_device = model.get_input_embeddings().weight.device
    base, alt = {}, {}
    with torch.no_grad():
        for i, (n, m) in enumerate(modules.items()):
            assert sha(m.weight) == frozen['matrices'][n]['source_sha256'], n
            b = quant_nvfp4_4over6(m.weight, 4, 16)
            a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(b).all() and torch.isfinite(a).all()
            base[n], alt[n] = (b.cpu().pin_memory(), a.cpu().pin_memory()) if target else (b, a)
            if (i + 1) % 64 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
        del a, b
    r['source_weights_verified'] = True
    excluded = {d['document_sha256'] for meta in frozen['fit'].values() for d in meta['documents']}
    assert len(excluded) == 192
    batches = {}
    batches['c4'], r['data']['c4'] = heldout_data(tok, excluded)
    batches['wiki'], r['data']['wiki'] = wiki_data(tok, excluded)
    transfer, meta = transfer_data(tok, excluded)
    batches.update(transfer); r['data'].update(meta)
    assert tuple(batches) == DOMAINS
    for domain, bs in batches.items():
        assert r['data'][domain]['token_sha256'] == [sha(b) for b in bs]
        r['data'][domain]['windows'] = len(bs)
        assert all(b.shape == (1, 512) for b in bs)
    references = {}
    c4_path = old / 'report.json' if target else Path(f'results/c4_frozen/model_332781_{r["model"]}/report.json')
    wiki_job = '332976' if target else '332974'
    wiki_path = Path(f'results/wiki_frozen/model_{wiki_job}_{r["model"]}/report.json')
    for domain, path in [('c4', c4_path), ('wiki', wiki_path)]:
        ref = json.loads(path.read_text())
        references[domain] = (ref, path, ref[f'{domain}_data'], ref['evaluation'])
    if not target:
        for domain in DOMAINS[2:]:
            references[domain] = (prior, old / 'report.json', prior['confirmation_data'][domain],
                                  {p: v[domain] for p, v in prior['evaluation'].items()})
    for domain, (ref, path, metadata, evaluation) in references.items():
        assert ref['status'] == 'complete'
        assert ref['source'] == r['source'] and ref['revision'] == r['revision']
        assert ref['transformers_version'] == r['transformers_version'] and ref['torch_version'] == r['torch_version']
        for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py'):
            assert ref['source_sha256'][f] == r['source_sha256'][f]
        assert metadata['token_sha256'] == r['data'][domain]['token_sha256']
        if domain in ('c4', 'wiki'):
            assert ref['map_file_sha256'] == frozen['original_maps_sha256']
    save(out, r)
    print('DATA ' + json.dumps({d: len(bs) for d, bs in batches.items()}), flush=True)

    def act(module, inputs):
        return (quantize_rows(inputs[0]), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for name, m in modules.items():
            b = base[name].to(m.weight.device, non_blocking=True)
            m.weight.copy_(b if policy == 'four_over_six' else apply_mask(
                b, alt[name].to(m.weight.device, non_blocking=True), maps[policy][name].to(m.weight.device)))

    def nll(batch):
        ids = batch.to(input_device)
        logits = model(input_ids=ids, use_cache=False).logits
        value = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                      ids[:, 1:].reshape(-1).to(logits.device)))
        assert math.isfinite(value)
        return value

    aliases = {'four_over_six': 'four_over_six', 'weight_mse': 'weight_mse', **ALIASES}
    with torch.no_grad():
        for policy in POLICIES:
            install(policy)
            ids = batches['c4'][0].to(input_device); changed = ids.clone()
            replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
            changed[:, 128:] = replacement
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.equal(a, b), policy
            r['suffix_intervention'][policy] = dict(equal=True, prefix_tokens=128,
                replacement_token_id=replacement, max_logit_difference=float((a.float()-b.float()).abs().max()))
            del a, b
            r['evaluation'][policy] = {}
            for domain, bs in batches.items():
                if domain not in args.domains:
                    continue
                if policy in aliases and domain in references:
                    ref, path, metadata, evaluation = references[domain]
                    values = evaluation[aliases[policy]]['nll']
                    first = nll(bs[0])
                    assert abs(first - values[0]) <= 1e-6, (policy, domain, first, values[0])
                    origin = dict(reused=True, report=str(path), report_sha256=digest_file(path),
                                  policy=aliases[policy], first_window_nll=first,
                                  first_window_absolute_error=abs(first-values[0]))
                else:
                    values = []
                    for i, batch in enumerate(bs):
                        values.append(nll(batch))
                        if (i+1) % 128 == 0:
                            print(f'EVAL {policy} {domain} {i+1}/{len(bs)}', flush=True)
                    origin = dict(reused=False)
                ppl = validate_losses(values, len(bs))
                r['evaluation'][policy][domain] = dict(nll=values, ppl=ppl, scored_tokens=len(bs)*511, origin=origin)
                save(out, r)
                print(f'PPL {policy} {domain} {ppl:.6f} reused={origin["reused"]}', flush=True)
    for h in handles:
        h.remove()
    r['contrasts'] = {p: {d: paired(r['evaluation'][p][d]['nll'], r['evaluation']['four_over_six'][d]['nll'])
                          for d in args.domains} for p in POLICIES if p != 'four_over_six'}
    assert digest_file(prepared) == r['prepared_sha256']
    assert digest_file(old / 'maps.pt') == frozen['original_maps_sha256']
    r['maps_unchanged'] = True
    r['status'] = 'complete'; save(out, r)
    print('COMPLETE ' + r['model'], flush=True)


if __name__ == '__main__':
    main()
