"""Post-freeze queue additions. Never rewrites an existing job or plan; every addition is logged in
provenance/PROTOCOL_AMENDMENTS.jsonl (hash-chained) by campaign.amend before it is queued."""
import json
import os
from pathlib import Path

from campaign.plan_campaign import CALIB_JOB, CR, JOBS, PLANS, job, mp

FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
F = ['--freeze', str(CR / 'freeze' / 'PROTOCOL_FREEZE.json'), '--freeze-sha256', FSHA]


def new_plan(name, entries):
    p = PLANS / f'{name}.json'
    if not p.exists():
        p.write_text(json.dumps(entries, indent=1) + '\n')
    elif json.loads(p.read_text()) != entries:
        raise SystemExit(f'plan {p} exists with different content')
    return str(p)


def synthesis(job_id, matrix, protocol, module, args=()):
    p = JOBS / f'{job_id}.json'
    if p.exists():
        return
    p.write_text(json.dumps(dict(job_id=job_id, matrix_id=matrix, protocol_id=protocol, gpus=0, gpu_model='a6000', env='main', cpus=16, memory='120g',
                                 priority=99, depends_on=[], set_env=[f'FREEZE_SHA256={FSHA}'], command=['-m', module, *args], hold=True,
                                 max_invalid_retries=25), indent=1) + '\n')


def main():
    # A03: V52 in-domain held-out PPL (math/code documents disjoint from every calibration draw)
    for m in ('qwen4b', 'mistral7b'):
        derive = f'V51_derive_subset_maps_{m}'
        seed0, held = CALIB_JOB(m), CALIB_JOB(m, 'heldout')
        # the balanced reference arms come from the seed0 calibration, whose map headers carry aligned-primary
        entries = [dict(name='four_over_six', kind='four_over_six'),
                   mp('n8_k3', seed0, protocol_id='aligned-primary'), mp('n16_k3', seed0, protocol_id='aligned-primary')]
        for s in ('math64', 'code64'):
            entries += [mp(f'n8_k3_{s}', derive), mp(f'n16_k3_{s}', derive)]
        entries += [mp('n8_k3_heldout', held, 'n8_k3'), mp('n16_k3_heldout', held, 'n16_k3')]
        plan = new_plan(f'crossdomain_ppl_{m}', entries)
        job(f'V52_ppl_crossdomain_{m}', 'V52', 'aligned-robustness', 1, 'a6000',
            ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan, '--domains', 'math_eval,code_eval', '--teacher', 'none', '--no-token-arrays',
             '--matrix-policies-protocol', 'aligned-robustness', '--protocol-id', 'aligned-robustness', *F],
            deps=[derive, seed0, held], priority=40, memory='110g', retries=25)
    # A06: frozen all-false/all-true checksum controls on the real models (V42)
    for m in ('qwen4b', 'llama8b', 'mistral7b', 'phi4', 'olmo2_13b'):
        job(f'V42_checksum_controls_{m}', 'V42', 'aligned-control', 1, 'a6000', ['-m', 'campaign.checksum_controls', '--model', m, '--freeze-sha256', FSHA],
            priority=30, memory=('150g' if m in ('phi4', 'olmo2_13b') else '110g'), retries=25)
    # A07: V43 noise-floor control. The V43 singles compare one intervention measurement against a
    # baseline measured once, at the start of a multi-hour run. This measures the noise of a single
    # measurement, the GPU-side restore path and the end-of-run drift, so the singles can be read
    # against their own resolution. Diagnostic on the V43 row; no confirmatory endpoint uses it.
    # qwen4b was added after its V43 fidelity run completed, so the control covers all three REP3 models rather than two
    # (amendment 30); it is also the model whose single-tile rank correlation is highest, i.e. the hardest case for the
    # per-tile refutation. job() never rewrites the two entries queued earlier.
    for m in ('mistral7b', 'llama8b', 'qwen4b'):
        job(f'V43_fidelity_noise_{m}', 'V43', 'aligned-analysis', 1, 'a6000',
            ['-m', 'campaign.fidelity_noise', '--model', m, '--calibration-run', f'@latest:{CALIB_JOB(m)}', *F],
            deps=[CALIB_JOB(m)], priority=35, memory=('120g' if m == 'llama8b' else '110g'), retries=25)
    # synthesis jobs (held until released)
    synthesis('V90_assemble_deliverables', 'V90', 'synthesis', 'campaign.assemble_deliverables')
    synthesis('V90_final_reports', 'V90', 'synthesis', 'campaign.final_all')


if __name__ == '__main__':
    main()
