"""Sweep the elected tile COUNT for one basis, reusing its frozen scores.

The 256-tile comparison is the wrong test of the alpha basis. E0M3 costs a
format bit per elected tile, so a small count is the point of it; the alpha
choice costs NOTHING -- it only changes the value written into the ue4m3 scale
field NVFP4 already stores -- so there is no reason to cap it at 256. This
re-elects each basis at every count in run_cap_sweep.COUNTS from the score
tables job 337178 already wrote, so no re-scoring happens and the arms stay
frozen. The election reproduces derive_maps' math on the math_code128 subset,
which is all 128 calibration sequences.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from quantize.basis import BASES, build_pair
from quantize.causal_four_over_six import quantize_rows
from quantize.interacting_format import apply_mask
from run_c4_frozen import digest_file, heldout_data
from run_cap_sweep import COUNTS, policy_name
from run_conditional_format import save, sha
from run_conditional_model import paired
from run_math_code_calibration import load_model
from run_wiki_frozen import wiki_data


def upper_scores(score_dir, names):
    """max(CE, KL) mean + 2 SE over all 128 sequences, exactly as derive_maps."""
    parts, slices, offset = [], {}, 0
    for i, name in enumerate(names):
        rec = torch.load(score_dir / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert rec['name'] == name, (rec['name'], name)
        ce, kl = rec['ce'].double(), rec['kl'].double()
        assert ce.shape[0] == 128 and ce.shape == kl.shape
        upper = torch.maximum(ce.mean(0) + 2 * ce.std(0, unbiased=True) / math.sqrt(128),
                              kl.mean(0) + 2 * kl.std(0, unbiased=True) / math.sqrt(128))
        assert torch.isfinite(upper).all()
        slices[name] = (offset, offset + upper.numel()); offset += upper.numel()
        parts.append(upper)
    return torch.cat(parts), slices


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    old = Path(args.calibration); prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen'] and prior['source_weights_verified']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert transformers.__version__ == prior['transformers_version']
    basis = prior['basis']; assert basis in BASES
    target = prior['model'] == 'qwen27b'
    torch.set_num_threads(12 if target else 4); torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    files = ('run_basis_count_sweep.py', 'run_math_code_evaluation.py', 'quantize/basis.py',
             'quantize/causal_four_over_six.py', 'quantize/quantizer.py', 'quantize/interacting_format.py')
    r = dict(status='running', model=prior['model'], source=prior['source'], revision=prior['revision'],
             basis=basis, basis_definition=BASES[basis], job_id=os.environ['SLURM_JOB_ID'],
             torch_version=torch.__version__, transformers_version=transformers.__version__,
             calibration=str(old), calibration_report_sha256=digest_file(old / 'report.json'),
             source_sha256={f: digest_file(f) for f in files},
             counts=[policy_name(c) for c in COUNTS], election={}, evaluation={}, data={})
    for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py'):
        assert r['source_sha256'][f] == prior['source_sha256'][f]
    save(out, r)

    model, modules = load_model(prior, target)
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    names = list(modules)
    upper, slices = upper_scores(old / 'scores', names)
    order = torch.argsort(upper, stable=True)
    eligible = int((upper < 0).sum())
    r['eligible_tiles'] = eligible; r['total_tiles'] = int(upper.numel())
    print(f'ELIGIBLE {eligible}/{upper.numel()}', flush=True)

    maps = {}
    for count in COUNTS:
        policy = policy_name(count)
        flat = torch.zeros(upper.numel(), dtype=torch.bool)
        if count != 0:
            take = order if count is None else order[:count]
            flat[take[upper[take] < 0]] = True
        maps[policy] = {n: flat[lo:hi].clone().reshape(modules[n].weight.shape[0] // 8,
                                                       modules[n].weight.shape[1] // 64)
                        for n, (lo, hi) in slices.items()}
        r['election'][policy] = dict(requested=count, selected=int(flat.sum()))
    # The frozen 256-tile map must reproduce, or the re-election drifted.
    frozen = json.loads((old / 'maps.json').read_text())['maps']['fixed256_math_code128']
    got = sum(int(m.sum()) for m in maps['n256'].values())
    want = sum(len(v) for v in frozen.values())
    assert got == want == 256, (got, want)
    for n in names:
        idx = maps['n256'][n].flatten().nonzero().flatten().tolist()
        assert idx == sorted(frozen[n]), n
    r['frozen_256_reproduced'] = True
    save(out, r)

    base, alt = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            b, a = build_pair(m.weight, basis)
            base[n], alt[n] = (b.cpu().pin_memory(), a.cpu().pin_memory()) if target else (b, a)
    r['source_weights_verified'] = True

    excluded = {d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert len(excluded) == 128
    batches = {}
    for domain in ('wiki', 'c4'):
        batches[domain], r['data'][domain] = (heldout_data if domain == 'c4' else wiki_data)(tok, excluded)
        r['data'][domain]['windows'] = len(batches[domain])
        assert [sha(b) for b in batches[domain]] == r['data'][domain]['token_sha256']
    save(out, r)

    handles = [m.register_forward_pre_hook(lambda mod, inp: (quantize_rows(inp[0]), *inp[1:]))
               for m in modules.values()]
    device = model.get_input_embeddings().weight.device

    def nll(batch):
        ids = batch.to(device)
        logits = model(input_ids=ids, use_cache=False).logits
        value = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                      ids[:, 1:].reshape(-1)))
        assert math.isfinite(value)
        return value

    with torch.no_grad():
        for count in COUNTS:
            policy = policy_name(count)
            for n, m in modules.items():
                m.weight.copy_(base[n].to(m.weight.device) if count == 0 else
                               apply_mask(base[n].to(m.weight.device), alt[n].to(m.weight.device),
                                          maps[policy][n].to(m.weight.device)))
            r['evaluation'][policy] = {}
            for domain in ('wiki', 'c4'):
                values = [nll(b) for b in batches[domain]]
                ppl = math.exp(sum(values) / len(values))
                r['evaluation'][policy][domain] = dict(nll=values, ppl=ppl)
                print(f'PPL {policy} {domain} {ppl:.6f} blocks={r["election"][policy]["selected"]}', flush=True)
            save(out, r)
    for h in handles:
        h.remove()

    ref = r['evaluation']['four_over_six']
    for policy in r['evaluation']:
        if policy == 'four_over_six':
            continue
        for domain in ('wiki', 'c4'):
            r['evaluation'][policy][domain]['paired'] = paired(
                r['evaluation'][policy][domain]['nll'], ref[domain]['nll'])
    r['status'] = 'complete'; save(out, r)
    print('SWEEP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
