"""Derive all prespecified maps from the existing shared calibration scores."""
import argparse
import json
import os
from pathlib import Path
import torch
from calibration_sensitivity import calibration_subsets, select_stream, selection_overlap
from run_c4_frozen import digest_file

ORIGINS = {
    'qwen4b': 'results/pooled_scale/model_332389_qwen4b',
    'llama8b': 'results/pooled_scale/model_332389_llama8b',
    'qwen27b': 'results/pooled_qwen27b/model_332840',
}
ALIASES = {'c4_64': 'c4_64', 'pooled64': 'mixed64', 'pooled192': 'pooled192'}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=ORIGINS, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(12)
    old, out = Path(ORIGINS[args.model]), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    bundle = torch.load(old / 'maps.pt', map_location='cpu', weights_only=True)
    names = list(prior['matrices'])
    provenance = {}

    def modules():
        if args.model == 'qwen27b':
            for i, name in enumerate(names):
                path = old / 'scores' / f'{i:03d}.pt'
                provenance[str(path)] = digest_file(path)
                record = torch.load(path, map_location='cpu', weights_only=True)
                assert record['name'] == name
                yield name, record['ce'], record['kl']
        else:
            path = old / 'scores.pt'
            provenance[str(path)] = digest_file(path)
            record = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert record['names'] == names and record['sources'] == ['web', 'math', 'code']
            ce, kl = record['ce'].flatten(0, 1), record['kl'].flatten(0, 1)
            assert list(record['slices']) == names
            for name, (lo, hi) in record['slices'].items():
                yield name, ce[:, lo:hi].contiguous(), kl[:, lo:hi].contiguous()

    sparse, counts, sizes = select_stream(modules())
    for name in names:
        o, k = prior['matrices'][name]['shape']
        assert sizes[name] == (o // 8) * (k // 64)
    for policy, old_policy in ALIASES.items():
        for name in names:
            actual = bundle['maps'][old_policy][name].flatten().nonzero().flatten().tolist()
            assert sparse[policy][name] == actual, (policy, name)
    overlap = {p: {q: selection_overlap(sparse[p], sparse[q]) for q in sparse} for p in sparse}
    counts['four_over_six'] = dict(selected_blocks=0, total_type_blocks=sum(sizes.values()), selected_fraction=0.)
    mse_count = sum(int(m.sum()) for m in bundle['maps']['weight_mse'].values())
    counts['weight_mse'] = dict(selected_blocks=mse_count, total_type_blocks=sum(sizes.values()),
                              selected_fraction=mse_count / sum(sizes.values()))
    r = dict(status='complete', model=args.model, source=prior['source'], revision=prior['revision'],
             job_id=os.environ['SLURM_JOB_ID'], origin=str(old),
             prior_report_sha256=digest_file(old / 'report.json'), original_maps_sha256=digest_file(old / 'maps.pt'),
             score_artifact_sha256=provenance, fit=prior['fit'], matrices=prior['matrices'],
             scoring_convention='Original window-wide factor score tables; no new scoring or calibration.',
             source_order=['C4', 'OpenWebMath', 'CodeParrot'], subsets=calibration_subsets(),
             maps=sparse, block_statistics=counts, overlap=overlap, original_maps_reproduced_exactly=True,
             source_sha256={f: digest_file(f) for f in ('calibration_sensitivity.py',
                 'prepare_calibration_sensitivity.py', 'quantize/relinearized_format.py',
                 'results/calibration_sensitivity/PROTOCOL.md')})
    (out / 'maps.json').write_text(json.dumps(r, indent=2) + '\n')
    print('MAPS FROZEN ' + json.dumps({p: v['selected_blocks'] for p, v in counts.items()}), flush=True)


if __name__ == '__main__':
    main()
