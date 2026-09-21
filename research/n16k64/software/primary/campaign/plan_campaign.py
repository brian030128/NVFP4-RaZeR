"""Generate policy plans and queue jobs for the GPU rows of EXPERIMENT_MATRIX.csv (after the freeze)."""
import argparse
import json
import os
from pathlib import Path

CR = Path(os.environ['CAMPAIGN_ROOT'])
PLANS = CR / 'plans'
JOBS = CR / 'queue' / 'jobs'
SMALL = ('qwen4b', 'llama8b', 'mistral7b')
LARGE1 = ('phi4', 'olmo2_13b')
CALIB_JOB = lambda m, draw='seed0': f'{"V30" if m in ("qwen4b", "llama8b", "qwen27b") else "V61"}_calib_{m}_{draw}'


def mp(name, job, policy=None, **kw):
    return dict(name=name, kind='map', from_job=job, map_policy=policy or name, **kw)


def write_plan(name, entries):
    PLANS.mkdir(parents=True, exist_ok=True)
    p = PLANS / f'{name}.json'
    p.write_text(json.dumps(entries, indent=1) + '\n')
    return str(p)


def job(job_id, matrix, protocol, gpus, model, command, deps=(), priority=50, env='main', cpus=12, memory='96g', retries=3, set_env=(), reserve=None):
    JOBS.mkdir(parents=True, exist_ok=True)
    p = JOBS / f'{job_id}.json'
    if p.exists():
        return job_id
    p.write_text(json.dumps(dict(job_id=job_id, matrix_id=matrix, protocol_id=protocol, gpus=gpus, gpu_model=model, env=env, cpus=cpus,
                                 memory=memory, priority=priority, depends_on=list(deps), command=command, max_invalid_retries=retries,
                                 set_env=list(set_env), reserve=(gpus > 1 if reserve is None else reserve)), indent=1) + '\n')
    return job_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=str(CR / 'freeze' / 'PROTOCOL_FREEZE.json'))
    args = ap.parse_args()
    fsha = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
    F = ['--freeze', args.freeze, '--freeze-sha256', fsha]
    mem = dict(qwen4b='110g', llama8b='120g', mistral7b='110g', phi4='150g', olmo2_13b='150g', qwen27b='190g')

    # ---------------- historical (V20-V23)
    for m in ('qwen4b', 'llama8b'):
        job(f'V21_hist_calib_{m}', 'V21', 'historical', 1, 'a6000', ['-m', 'campaign.historical', '--model', m, '--freeze-sha256', fsha],
            priority=10, env='hist', memory=mem[m])
        job(f'V22_hist_eval_{m}', 'V22', 'historical', 1, 'a6000',
            ['-m', 'campaign.historical_eval', '--model', m, '--policies', 'four_over_six,archived_fixed256,hist_n8_n256,hist_n8_k2,hist_n8_k3,hist_n8_k4,hist_n8_k5,hist_n8_k6',
             '--maps-from-job', f'V21_hist_calib_{m}', '--freeze-sha256', fsha], deps=[f'V21_hist_calib_{m}'], priority=15, env='hist', memory=mem[m])
    job('V23_hist_eval_qwen4b_ada', 'V23', 'historical', 1, 'ada',
        ['-m', 'campaign.historical_eval', '--model', 'qwen4b', '--policies', 'four_over_six,archived_fixed256,hist_n8_k3', '--maps-from-job', 'V21_hist_calib_qwen4b',
         '--freeze-sha256', fsha], deps=['V21_hist_calib_qwen4b'], priority=15, env='hist', memory=mem['qwen4b'], retries=20)
    job('V21_hist_calib_qwen27b', 'V21', 'historical', 2, 'a6000', ['-m', 'campaign.historical', '--model', 'qwen27b', '--stream', '--max-memory-gib', '44',
        '--direction-cache-gib', '2', '--freeze-sha256', fsha], priority=22, memory=mem['qwen27b'], cpus=16)
    job('V22_hist_eval_qwen27b', 'V22', 'historical', 2, 'a6000', ['-m', 'campaign.historical_eval', '--model', 'qwen27b', '--policies',
        'four_over_six,archived_fixed256,hist_n8_k3', '--maps-from-job', 'V21_hist_calib_qwen27b', '--max-memory-gib', '44', '--freeze-sha256', fsha],
        deps=['V21_hist_calib_qwen27b'], priority=23, memory=mem['qwen27b'], cpus=16)

    # ---------------- aligned calibrations (V30 / V61 / V50 / V52 / V82 / V81)
    for m in ('qwen4b', 'llama8b', 'mistral7b', 'phi4', 'olmo2_13b'):
        extra = []
        if m in ('qwen4b', 'mistral7b'):
            extra += ['--subset-moments']
        extra += ['--raw', 'full' if m == 'qwen4b' else 'sample']
        job(CALIB_JOB(m), 'V30' if m in ('qwen4b', 'llama8b') else 'V61', 'aligned-primary', 1, 'a6000',
            ['-m', 'campaign.calibrate', '--model', m, '--draw', 'seed0', '--protocol-id', 'aligned-primary'] + extra + F, priority=10, memory=mem[m])
    job(CALIB_JOB('qwen27b'), 'V30', 'aligned-primary', 2, 'a6000', ['-m', 'campaign.calibrate', '--model', 'qwen27b', '--draw', 'seed0', '--protocol-id',
        'aligned-primary', '--raw', 'sample', '--max-memory-gib', '44', '--moments-device', 'cpu'] + F, priority=21, memory=mem['qwen27b'], cpus=16)
    for m in SMALL:
        for d in ('draw1', 'draw2', 'draw3', 'draw4'):
            job(CALIB_JOB(m, d), 'V50', 'aligned-robustness', 1, 'a6000', ['-m', 'campaign.calibrate', '--model', m, '--draw', d, '--protocol-id', 'aligned-robustness',
                '--raw', 'sample'] + F, priority=40, memory=mem[m])
    for m in ('qwen4b', 'mistral7b'):
        job(CALIB_JOB(m, 'heldout'), 'V52', 'aligned-robustness', 1, 'a6000', ['-m', 'campaign.calibrate', '--model', m, '--draw', 'heldout', '--n-per-domain', '32',
            '--protocol-id', 'aligned-robustness', '--raw', 'sample'] + F, priority=40, memory=mem[m])
        job(f'V51_derive_subset_maps_{m}', 'V51', 'aligned-robustness', 0, 'a6000', ['-m', 'campaign.derive_maps', '--model', m, '--calibration-job', CALIB_JOB(m)] + F,
            deps=[CALIB_JOB(m)], priority=12, memory='64g')
    job('V82_calib_qwen4b_seed0_repeat', 'V82', 'aligned-primary', 1, 'a6000', ['-m', 'campaign.calibrate', '--model', 'qwen4b', '--draw', 'seed0', '--protocol-id',
        'aligned-primary', '--raw', 'sample', '--tag', 'determinism_repeat'] + F, deps=[CALIB_JOB('qwen4b')], priority=45, memory=mem['qwen4b'])
    job('V81_calib_qwen4b_seed0_ada', 'V81', 'aligned-primary', 1, 'ada', ['-m', 'campaign.calibrate', '--model', 'qwen4b', '--draw', 'seed0', '--protocol-id',
        'aligned-primary', '--raw', 'sample', '--tag', 'ada_portability'] + F, deps=[CALIB_JOB('qwen4b')], priority=45, memory=mem['qwen4b'], retries=20)

    # ---------------- primary PPL (V31 / V62)
    for m in SMALL + LARGE1 + ('qwen27b',):
        cj = CALIB_JOB(m)
        plan = write_plan(f'primary_ppl_{m}', [dict(name='bf16', kind='bf16'), dict(name='nvfp4', kind='nvfp4'), dict(name='four_over_six', kind='four_over_six'),
                                               dict(name='all_e0m3', kind='all_e0m3'), mp('n8_k3', cj), mp('n16_k3', cj)])
        if m in SMALL:
            cmd, g, mm = ['--teacher', 'instance'], 1, mem[m]
        elif m in LARGE1:
            cmd, g, mm = ['--teacher', 'cache', '--teacher-windows', '32'], 1, mem[m]
        else:
            cmd, g, mm = ['--teacher', 'cache', '--teacher-windows', '16', '--max-memory-gib', '44'], 2, mem[m]
        job(f'{"V31" if m in ("qwen4b", "llama8b", "qwen27b") else "V62"}_ppl_primary_{m}', 'V31' if m in ('qwen4b', 'llama8b', 'qwen27b') else 'V62',
            'aligned-primary', g, 'a6000', ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan] + cmd + F, deps=[cj], priority=(20 if m != 'qwen27b' else 24),
            memory=mm, cpus=(16 if g > 1 else 12))
    # ---------------- accuracy (V32 / V63), split per policy
    for m in SMALL + LARGE1 + ('qwen27b',):
        cj = CALIB_JOB(m)
        mid = 'V32' if m in ('qwen4b', 'llama8b', 'qwen27b') else 'V63'
        for pol in (dict(name='bf16', kind='bf16'), dict(name='four_over_six', kind='four_over_six'), mp('n8_k3', cj), mp('n16_k3', cj)):
            plan = write_plan(f'acc_{m}_{pol["name"]}', [pol])
            g = 2 if m == 'qwen27b' else 1
            job(f'{mid}_acc_full8_{m}_{pol["name"]}', mid, 'aligned-primary', g, 'a6000',
                ['-m', 'campaign.evaluate_lmeval', '--model', m, '--plan', plan, '--suite', 'full8'] + (['--max-memory-gib', '44'] if g > 1 else []) + F,
                deps=[cj], priority=(25 if m in ('mistral7b', 'phi4', 'olmo2_13b') else 27) if m != 'qwen27b' else 63, memory=mem[m], cpus=(16 if g > 1 else 12))
    # ---------------- generation (V33 / V64)
    for m in ('qwen4b', 'llama8b', 'mistral7b'):
        cj = CALIB_JOB(m)
        mid = 'V64' if m == 'mistral7b' else 'V33'
        for pol in (dict(name='bf16', kind='bf16'), dict(name='four_over_six', kind='four_over_six'), mp('n8_k3', cj), mp('n16_k3', cj)):
            plan = write_plan(f'gen_{m}_{pol["name"]}', [pol])
            job(f'{mid}_gsm8k_{m}_{pol["name"]}', mid, 'aligned-primary', 1, 'a6000', ['-m', 'campaign.evaluate_lmeval', '--model', m, '--plan', plan, '--suite', 'gsm8k'] + F,
                deps=[cj], priority=30, memory=mem[m])
    # ---------------- selector controls, k sweep, objectives (V40-V42), baselines (V80)
    for m in SMALL:
        cj = CALIB_JOB(m)
        ctrl_a = [dict(name='four_over_six', kind='four_over_six')] + [mp(f'n16_k{k}', cj) for k in (2, 3, 4, 5, 6)] + [mp(f'n8_k{k}', cj) for k in (2, 4, 5, 6)]
        ctrl_b = [dict(name='four_over_six', kind='four_over_six')] + [mp(f'n16_k3_{r}', cj) for r in ('ce_only', 'kl_only', 'mean_only')] + \
                 [mp(f'n16_random_s{s}', cj) for s in range(5)] + [mp(p, cj) for p in ('n16_weight_mse', 'n16_magnitude', 'n16_change_norm', 'n16_density_matched_n8k3')]
        base = [dict(name='four_over_six', kind='four_over_six'), dict(name='razer_wonly_shared_act', kind='razer_wonly_shared_act'),
                dict(name='razer_native_rows', kind='razer_native_rows'), dict(name='nover6_wonly_shared_act', kind='nover6_wonly_shared_act'),
                dict(name='nover6_native_rows', kind='nover6_native_rows'), mp('n16_k3', cj)]
        for tag, entries, mid, proto in (('ksweep', ctrl_a, 'V40', 'aligned-ablation'), ('controls', ctrl_b, 'V42', 'aligned-control'), ('baselines', base, 'V80', 'aligned-baseline')):
            plan = write_plan(f'{tag}_ppl_{m}', entries)
            job(f'{mid}_ppl_{tag}_{m}', mid, 'aligned-analysis' if proto == 'aligned-baseline' else proto, 1, 'a6000',
                ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan, '--teacher', 'none', '--no-token-arrays', '--matrix-policies-protocol', 'aligned-primary',
                 '--protocol-id', proto if proto != 'aligned-baseline' else 'aligned-analysis'] + F, deps=[cj], priority=35, memory=mem[m], retries=20)
            rep_entries = [e for e in entries if e['name'] != 'four_over_six' or tag == 'ksweep']
            for e in rep_entries:
                plan1 = write_plan(f'rep_{m}_{e["name"]}', [e])
                job(f'{mid}_acc_rep_{m}_{e["name"]}', mid, proto if proto != 'aligned-baseline' else 'aligned-analysis', 1, 'a6000',
                    ['-m', 'campaign.evaluate_lmeval', '--model', m, '--plan', plan1, '--suite', 'representative', '--matrix-policies-protocol', 'aligned-primary',
                     '--protocol-id', proto if proto != 'aligned-baseline' else 'aligned-analysis'] + F, deps=[cj], priority=50, memory=mem[m])
    # ---------------- calibration draws (V50)
    for m in SMALL:
        entries = [dict(name='four_over_six', kind='four_over_six')] + [mp(f'{res}_k3_{d}', CALIB_JOB(m, d), policy=f'{res}_k3') for d in ('draw1', 'draw2', 'draw3', 'draw4') for res in ('n8', 'n16')]
        plan = write_plan(f'draws_ppl_{m}', entries)
        job(f'V50_ppl_draws_{m}', 'V50', 'aligned-robustness', 1, 'a6000', ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan, '--teacher', 'none', '--no-token-arrays',
            '--matrix-policies-protocol', 'aligned-robustness', '--protocol-id', 'aligned-robustness'] + F,
            deps=[CALIB_JOB(m, d) for d in ('draw1', 'draw2', 'draw3', 'draw4')], priority=40, memory=mem[m], retries=20)
        for e in entries[1:]:
            plan1 = write_plan(f'rep_{m}_{e["name"]}', [e])
            job(f'V50_acc_rep_{m}_{e["name"]}', 'V50', 'aligned-robustness', 1, 'a6000', ['-m', 'campaign.evaluate_lmeval', '--model', m, '--plan', plan1, '--suite',
                'representative', '--matrix-policies-protocol', 'aligned-robustness', '--protocol-id', 'aligned-robustness'] + F, deps=[e['from_job']], priority=52, memory=mem[m])
    # ---------------- calibration size / domain (V51 / V52)
    for m in ('qwen4b', 'mistral7b'):
        dj = f'V51_derive_subset_maps_{m}'
        entries = [dict(name='four_over_six', kind='four_over_six')]
        for sub in ('mc32', 'mc64', 'math64', 'code64'):
            for res in ('n8', 'n16'):
                entries.append(mp(f'{res}_k3_{sub}', dj))
        entries += [mp(f'{res}_k3_heldout', CALIB_JOB(m, 'heldout'), policy=f'{res}_k3') for res in ('n8', 'n16')]
        plan = write_plan(f'sizedomain_ppl_{m}', entries)
        job(f'V51_ppl_sizedomain_{m}', 'V51', 'aligned-robustness', 1, 'a6000', ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan, '--teacher', 'none',
            '--no-token-arrays', '--matrix-policies-protocol', 'aligned-robustness', '--protocol-id', 'aligned-robustness'] + F, deps=[dj, CALIB_JOB(m, 'heldout')],
            priority=40, memory=mem[m], retries=20)
        for e in entries[1:]:
            plan1 = write_plan(f'rep_{m}_{e["name"]}', [e])
            job(f'V52_acc_rep_{m}_{e["name"]}', 'V52' if ('math' in e['name'] or 'code' in e['name'] or 'heldout' in e['name']) else 'V51', 'aligned-robustness', 1, 'a6000',
                ['-m', 'campaign.evaluate_lmeval', '--model', m, '--plan', plan1, '--suite', 'representative', '--matrix-policies-protocol', 'aligned-robustness',
                 '--protocol-id', 'aligned-robustness'] + F, deps=[e['from_job']], priority=53, memory=mem[m])
    # ---------------- first-order fidelity (V43) and long context (V71)
    for m in SMALL:
        job(f'V43_fidelity_{m}', 'V43', 'aligned-analysis', 1, 'a6000', ['-m', 'campaign.fidelity', '--model', m, '--calibration-run',
            f'@latest:{CALIB_JOB(m)}'] + F, deps=[CALIB_JOB(m)], priority=45, memory=mem[m])
        plan = write_plan(f'long_ppl_{m}', [dict(name='bf16', kind='bf16'), dict(name='four_over_six', kind='four_over_six'), mp('n8_k3', CALIB_JOB(m)), mp('n16_k3', CALIB_JOB(m))])
        job(f'V71_ppl_long_{m}', 'V71', 'aligned-analysis', 1, 'a6000', ['-m', 'campaign.evaluate_ppl', '--model', m, '--plan', plan, '--domains', 'pg19_4k,pg19_8k',
            '--teacher', 'none', '--matrix-policies-protocol', 'aligned-primary', '--protocol-id', 'aligned-analysis'] + F, deps=[CALIB_JOB(m)], priority=45, memory=mem[m], retries=20)
    # ---------------- cross-GPU aligned portability (V81): same frozen A6000 map and the Ada-generated map, evaluated on Ada
    plan = write_plan('portability_ppl_qwen4b', [dict(name='four_over_six', kind='four_over_six'), mp('n16_k3', CALIB_JOB('qwen4b')),
                                                 mp('n16_k3_ada', 'V81_calib_qwen4b_seed0_ada', policy='n16_k3')])
    job('V81_ppl_portability_qwen4b_ada', 'V81', 'aligned-primary', 1, 'ada', ['-m', 'campaign.evaluate_ppl', '--model', 'qwen4b', '--plan', plan, '--teacher', 'none',
        '--no-token-arrays'] + F, deps=[CALIB_JOB('qwen4b'), 'V81_calib_qwen4b_seed0_ada'], priority=46, memory=mem['qwen4b'], retries=20)
    job('V81_ppl_portability_qwen4b_a6000', 'V81', 'aligned-primary', 1, 'a6000', ['-m', 'campaign.evaluate_ppl', '--model', 'qwen4b', '--plan', plan, '--teacher', 'none',
        '--no-token-arrays'] + F, deps=[CALIB_JOB('qwen4b'), 'V81_calib_qwen4b_seed0_ada'], priority=46, memory=mem['qwen4b'])
    print(json.dumps(dict(jobs=len(list(JOBS.glob('*.json'))), plans=len(list(PLANS.glob('*.json'))))))


if __name__ == '__main__':
    main()
