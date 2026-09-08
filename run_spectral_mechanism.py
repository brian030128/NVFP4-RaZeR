"""Frozen synthetic mechanism panel. No language data or configuration election."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

from quantize.spectral_selector import candidates, mse_map, reconstruct, select, spectral_squared


def digest(tensor):
    return hashlib.sha256(tensor.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def metrics(w, quantized, directions):
    error = quantized.double() - w.double()
    spec = spectral_squared(error)
    frob = error.square().sum().item()
    return dict(spectral_squared=spec, frobenius_squared=frob,
                stable_rank=frob/spec if spec else 0.,
                isotropic_output_error=frob/w.shape[1],
                **{name: (error @ vectors).square().sum().item()/vectors.shape[1]
                   for name, vectors in directions.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    sources = ['quantize/spectral_selector.py', 'quantize/quantizer.py',
               'run_spectral_mechanism.py', 'results/spectral_selection/PROTOCOL.md']
    report = dict(status='running', scope='synthetic mechanism only',
                  torch_version=torch.__version__, python_version=platform.python_version(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  source_sha256={s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
                  cases=[])
    start = time.time()
    for seed in (101, 102, 103):
        for family in ('gaussian', 'heavy_tail', 'column_outliers', 'row_correlated'):
            gen = torch.Generator().manual_seed(seed)
            w = torch.randn(32, 128, generator=gen)
            if family == 'heavy_tail':
                w /= torch.randn(32, 128, generator=gen).abs().clamp_min(.1)
            elif family == 'column_outliers':
                w[:, ::17] *= 12
            elif family == 'row_correlated':
                w += 2 * torch.randn(1, 128, generator=gen)
            w = w.bfloat16()
            base, alt = candidates(w)
            mask, trace = select(w, base, alt)
            # All maps are frozen before constructing any evaluation directions.
            masks = dict(four_over_six=torch.zeros_like(mask), all_e0m3=torch.ones_like(mask),
                         weight_mse=mse_map(w, base, alt), spectral=mask)
            rg = torch.Generator().manual_seed(9000+seed)
            random_map = torch.zeros(mask.numel(), dtype=torch.bool)
            random_map[torch.randperm(mask.numel(), generator=rg)[:int(mask.sum())]] = True
            masks['random_count_matched'] = random_map.reshape_as(mask)
            # Exhaustive spectral optimum is a diagnostic, never a policy choice.
            oracle_value, oracle_mask = float('inf'), None
            for bits in range(1 << mask.numel()):
                candidate = torch.tensor([(bits >> j) & 1 for j in range(mask.numel())],
                                         dtype=torch.bool).reshape_as(mask)
                value = spectral_squared(reconstruct(base, alt, candidate).double()-w.double())
                if value < oracle_value:
                    oracle_value, oracle_mask = value, candidate
            masks['exhaustive_spectral_oracle'] = oracle_mask
            eg = torch.Generator().manual_seed(20000+seed)
            v = torch.randn(128, 4, generator=eg, dtype=torch.float64)
            v = torch.linalg.qr(v).Q
            directions = dict(random_rank4_output_error=v,
                              coordinate_shift_output_error=torch.eye(128, dtype=torch.float64)[:, ::17])
            case = dict(seed=seed, family=family, shape=list(w.shape), weight_sha256=digest(w),
                        solver=trace, policies={})
            for name, policy in masks.items():
                q = reconstruct(base, alt, policy)
                case['policies'][name] = dict(metrics(w, q, directions),
                    selected_tiles=int(policy.sum()), mask=policy.int().tolist(), quantized_sha256=digest(q))
            assert trace['objective_history'][-1] >= oracle_value - 1e-9
            case['solver_relative_oracle_gap'] = trace['objective_history'][-1]/oracle_value-1 if oracle_value else 0.
            report['cases'].append(case)
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'{family} seed={seed} spectral/base={trace["objective_history"][-1]/case["policies"]["four_over_six"]["spectral_squared"]:.6f} oracle_gap={case["solver_relative_oracle_gap"]:.6f}', flush=True)
    report.update(status='complete', elapsed_seconds=time.time()-start)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Synthetic mechanism results', '',
             'These are small synthetic matrices, not evidence of LLM or domain generalization.', '',
             '| Family / seed | Spectral / baseline | MSE policy spectral / baseline | Solver / oracle − 1 | Frobenius / baseline |',
             '|---|---:|---:|---:|---:|']
    for c in report['cases']:
        p = c['policies']; b = p['four_over_six']; s = p['spectral']
        lines.append(f'| {c["family"]} / {c["seed"]} | {s["spectral_squared"]/b["spectral_squared"]:.6f} | {p["weight_mse"]["spectral_squared"]/b["spectral_squared"]:.6f} | {c["solver_relative_oracle_gap"]:.6f} | {s["frobenius_squared"]/b["frobenius_squared"]:.6f} |')
    lines += ['', 'All maps, shifted-input output errors, hashes, and solver traces are in report.json.',
              'The exhaustive optimum is an objective diagnostic; it does not select an exported policy.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
