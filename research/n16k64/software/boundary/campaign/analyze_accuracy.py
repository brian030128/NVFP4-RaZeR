"""V32/V63 accuracy, V33/V64 generation and representative-suite comparisons from per-example lm-eval samples (CPU)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
METRIC = dict(arc_easy='acc_norm', arc_challenge='acc_norm', hellaswag='acc_norm', openbookqa='acc_norm', piqa='acc_norm', boolq='acc',
              winogrande='acc', mmlu='acc', gsm8k='exact_match,flexible-extract')
MARGIN_PP = FREEZE['success_criteria']['sesoi']['accuracy_macro_margin_pp']


def sample_rows(job, policy):
    run = latest_complete_run(CR, job)
    rep = json.loads((run / 'lmeval' / 'lmeval_report.json').read_text())
    if rep['status'] != 'complete':
        raise FileNotFoundError(f'{job} not complete')
    return run, rep, rep['results'][policy]['samples']


def task_rows(samples, task):
    if task == 'mmlu':
        rows = {}
        for t, s in samples.items():
            if t.startswith('mmlu_'):
                for k, v in S.load_samples(s['path'], 'acc').items():
                    rows[(t, k)] = v
        return rows
    if task == 'gsm8k':
        rows = S.load_samples(samples['gsm8k']['path'], 'exact_match')
        strict = {}
        import gzip
        with gzip.open(samples['gsm8k']['path'], 'rt') as f:
            for line in f:
                r = json.loads(line)
                # lm-eval stores one value per filter as separate keys for generate_until tasks
                strict[r.get('doc_id')] = r
        return rows
    return S.load_samples(samples[task]['path'], METRIC[task])


def gsm8k_filters(path):
    import gzip
    flex, strict = {}, {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            key = (r.get('doc_id'), r.get('doc_hash'))
            if r.get('filter') == 'flexible-extract':
                flex[key] = float(r['exact_match'])
            elif r.get('filter') == 'strict-match':
                strict[key] = float(r['exact_match'])
    return flex, strict


def compare_policies(model, job_of, pa, pb, tasks, B=10000):
    ra, repa, sa = sample_rows(job_of(model, pa), pa)
    rb, repb, sb = sample_rows(job_of(model, pb), pb)
    per = {}
    for t in tasks:
        if t == 'gsm8k':
            fa, sta = gsm8k_filters(sa['gsm8k']['path'])
            fb, stb = gsm8k_filters(sb['gsm8k']['path'])
            per['gsm8k_flexible'] = S.paired_accuracy(fa, fb, B=B)
            per['gsm8k_strict'] = S.paired_accuracy(sta, stb, B=B)
            continue
        per[t] = S.paired_accuracy(task_rows(sa, t), task_rows(sb, t), B=B)
    out = {t: {k: v for k, v in r.items() if k != 'boot'} for t, r in per.items()}
    mac_tasks = {t: r for t, r in per.items() if not t.startswith('gsm8k')}
    if len(mac_tasks) > 1:
        out['macro'] = S.macro_accuracy(mac_tasks)
        out['tasks_with_ci_below_zero'] = [t for t, r in mac_tasks.items() if r['ci95'][1] < 0]
        out['tasks_with_ci_above_zero'] = [t for t, r in mac_tasks.items() if r['ci95'][0] > 0]
    out['runs'] = dict(a=ra.name, b=rb.name)
    out['lm_eval_version'] = repa['lm_eval_version']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', required=True, choices=('primary', 'generation', 'representative', 'all'))
    args = ap.parse_args()
    out = runtime.out_dir('analysis_accuracy')
    full8 = ['arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa', 'mmlu']
    if args.what in ('primary', 'all'):
        for models, mid, name in ((('llama8b', 'qwen4b', 'qwen27b'), 'V32', 'LEGACY_PANEL_ACCURACY.json'), (('mistral7b', 'phi4', 'olmo2_13b'), 'V63', 'CONFIRMATORY_ACCURACY.json')):
            res = dict(matrix_id=mid, freeze_sha256=FSHA, margin_macro_pp=MARGIN_PP, models={})
            for m in models:
                job_of = lambda mm, p: f'{mid}_acc_full8_{mm}_{p}'
                res['models'][m] = {}
                for a, b in (('n16_k3', 'four_over_six'), ('n8_k3', 'four_over_six'), ('n16_k3', 'n8_k3'), ('four_over_six', 'bf16'), ('n16_k3', 'bf16')):
                    try:
                        c = compare_policies(m, job_of, a, b, full8)
                        if a == 'n16_k3' and b == 'four_over_six' and 'macro' in c:
                            c['noninferior_macro'] = bool(c['macro']['ci95'][0] * 100 > MARGIN_PP)
                            c['fewer_than_3_tasks_clear_loss'] = len(c['tasks_with_ci_below_zero']) < 3
                        res['models'][m][f'{a}-{b}'] = c
                    except (FileNotFoundError, KeyError, ValueError) as exc:
                        res['models'][m][f'{a}-{b}'] = dict(status='missing', error=repr(exc))
            runtime.atomic_json(out / name, res)
    if args.what in ('generation', 'all'):
        for models, mids, name in ((('qwen4b', 'llama8b'), 'V33', 'LEGACY_GENERATION.json'), (('mistral7b',), 'V64', 'CONFIRMATORY_GENERATION.json')):
            res = dict(matrix_id=mids, freeze_sha256=FSHA, models={})
            for m in models:
                job_of = lambda mm, p: f'{mids}_gsm8k_{mm}_{p}'
                res['models'][m] = {}
                for a, b in (('n16_k3', 'four_over_six'), ('n8_k3', 'four_over_six'), ('n16_k3', 'n8_k3'), ('four_over_six', 'bf16'), ('n16_k3', 'bf16')):
                    try:
                        res['models'][m][f'{a}-{b}'] = compare_policies(m, job_of, a, b, ['gsm8k'])
                    except (FileNotFoundError, KeyError, ValueError) as exc:
                        res['models'][m][f'{a}-{b}'] = dict(status='missing', error=repr(exc))
            runtime.atomic_json(out / name, res)
    if args.what in ('representative', 'all'):
        rep4 = ['arc_challenge', 'piqa', 'winogrande', 'boolq']
        groups = {
            'K_SENSITIVITY_ACCURACY.json': ('V40', ('llama8b', 'qwen4b', 'mistral7b'), [f'n16_k{k}' for k in (2, 3, 4, 5, 6)] + [f'n8_k{k}' for k in (2, 4, 5, 6)]),
            'SELECTOR_CONTROLS_ACCURACY.json': ('V42', ('llama8b', 'qwen4b', 'mistral7b'), ['n16_k3_ce_only', 'n16_k3_kl_only', 'n16_k3_mean_only'] +
                                                [f'n16_random_s{s}' for s in range(5)] + ['n16_weight_mse', 'n16_magnitude', 'n16_change_norm', 'n16_density_matched_n8k3']),
            'ADDITIONAL_BASELINES_ACCURACY.json': ('V80', ('llama8b', 'qwen4b', 'mistral7b'), ['razer_wonly_shared_act', 'razer_native_rows', 'nover6_wonly_shared_act', 'nover6_native_rows']),
            'CALIBRATION_SEED_STABILITY_ACCURACY.json': ('V50', ('llama8b', 'qwen4b', 'mistral7b'), [f'{r}_k3_draw{d}' for d in (1, 2, 3, 4) for r in ('n8', 'n16')]),
            'CALIBRATION_SIZE_DOMAIN_ACCURACY.json': ('V51/V52', ('qwen4b', 'mistral7b'), [f'{r}_k3_{s}' for s in ('mc32', 'mc64', 'math64', 'code64', 'heldout') for r in ('n8', 'n16')]),
        }
        for name, (mid, models, pols) in groups.items():
            res = dict(matrix_id=mid, freeze_sha256=FSHA, reference='four_over_six (V40 representative run)', models={})
            for m in models:
                res['models'][m] = {}
                for p in pols + ['n16_k3']:
                    def job_of(mm, pp, _mid=mid):
                        if pp == 'four_over_six' or pp.startswith('n16_k') and len(pp) == 6 or pp.startswith('n8_k') and len(pp) == 5:
                            return f'V40_acc_rep_{mm}_{pp}'
                        if pp.startswith(('razer', 'nover6')):
                            return f'V80_acc_rep_{mm}_{pp}'
                        if 'draw' in pp:
                            return f'V50_acc_rep_{mm}_{pp}'
                        # Every size/domain accuracy job is named V52_acc_rep_* by plan_campaign.py; only the
                        # MATRIX ID differs (V51 owns mc32/mc64, V52 owns math64/code64/heldout). Routing mc32/mc64 to
                        # a V51_acc_rep_* id - which has never existed as a spec or a run - made 8 cells permanently
                        # missing. Merged rather than deleted: without this branch mc32/mc64 would fall through to the
                        # V42_acc_rep_* default, which is equally nonexistent but fails silently in the same way.
                        if any(s in pp for s in ('mc32', 'mc64', 'math64', 'code64', 'heldout')):
                            return f'V52_acc_rep_{mm}_{pp}'
                        return f'V42_acc_rep_{mm}_{pp}'
                    try:
                        res['models'][m][p] = compare_policies(m, job_of, p, 'four_over_six', rep4, B=2000)
                    except (FileNotFoundError, KeyError, ValueError) as exc:
                        res['models'][m][p] = dict(status='missing', error=repr(exc))
            runtime.atomic_json(out / name, res)
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='statistics', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(p) for p in sorted(out.iterdir())], summary={}, uncertainty={}, attempted_endpoints=[args.what], missing_endpoints=runtime.collect_missing(out)),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
