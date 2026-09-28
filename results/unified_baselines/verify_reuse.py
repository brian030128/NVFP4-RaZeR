"""May earlier records stand in for unified-condition runs? (PROTOCOL.md, 'Reused runs')

Checks, from the run records:
- OURS (TM-OPT+TC, the committed maps; results/tm_opt) for Llama-3.1-8B and Mistral-7B-v0.3 at 8x64 and 16x64:
  - per-token (c) training activations (fused_act_quant: fourover6_rows);
  - 20 epochs, deterministic, optimizer batch 8 (micro-batch 8, no accumulation), the 128 fit sequences;
  - lr 0.02;
  - without a development set: Task 1's --no-dev rerun wrote a bitwise-equal map (results/nodev_cost).
- SCALE for Llama (Task 2, results/scale_additivity/runs/scale_nodev):
  - arm scale, per-token (c) training activations (--act-rows), 20 epochs of the c1 budget, deterministic;
  - batch and micro-batch 8, --no-dev;
  - the learning rate chosen by Task 2's rule on the development set;
  - the per-step training KL equal to the chosen development run's.

    python results/unified_baselines/verify_reuse.py   -> reuse.json (and exit status 1 if any check fails)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TM = Path('/home/dev/n16k64_campaign/tm_opt/runs')
ND = HERE.parent / 'nodev_cost' / 'runs'
SA = HERE.parent / 'scale_additivity' / 'runs'


def load(p):
    return json.loads(Path(p).read_text())


def ours(model, unit):
    r = load(TM / f'tc_{model}_{unit}' / 'report.json')
    nd = load(ND / f'nd_tc_{model}_{unit}' / 'report.json')
    a, st = r['args'], r['settings']
    checks = dict(configuration=r['configuration'] == 'TM-OPT+TC', per_token_training_activations=st['fused_act_quant'] is True,
                  epochs_20=a['epochs'] == 20 and len(r['epochs']) == 20, deterministic=st['deterministic'] is True,
                  batch_8=a['batch'] == 8 and a['accum'] == 1, lr_002=a['lr'] == 0.02, fit_sequences_128=r['epochs'][0]['step'] == 16,
                  nodev_map_bitwise_equal=nd['map_sha256'] == r['map_sha256'] and nd['args']['no_dev'] is True)
    return dict(run=str(TM / f'tc_{model}_{unit}'), nodev_run=str(ND / f'nd_tc_{model}_{unit}'), checks=checks)


def scale_llama():
    r = load(SA / 'scale_nodev' / 'report.json')
    c = r['config']
    choice = load(SA / 'lr_choice.json')
    dev = load(SA / f"scale_dev_lr{choice['chosen']}" / 'report.json')
    checks = dict(arm_scale=c['arm'] == 'scale', per_token_training_activations=c['act_rows'] is True,
                  epochs_20=c['budget'] == 'c1' and c['epochs'] == 20 and r['counts']['optimizer_steps'] == 320,
                  deterministic=c['deterministic'] is True, batch_8=c['batch'] == 8 and c['micro_batch'] == 8,
                  no_dev=c['no_dev'] is True, lr_is_the_rule_choice=float(choice['chosen']) == c['lr'],
                  per_step_kl_equals_chosen_dev_run=[x['kl'] for x in r['log']] == [x['kl'] for x in dev['log']])
    return dict(run=str(SA / 'scale_nodev'), lr_choice=choice, checks=checks)


def main():
    out = dict(ours={f'{m} {u}': ours(m, u) for m in ('llama8b', 'mistral7b') for u in ('8x64', '16x64')},
               scale={'llama8b': scale_llama()})
    ok = all(all(v['checks'].values()) for group in out.values() for v in group.values())
    out['all_pass'] = ok
    (HERE / 'reuse.json').write_text(json.dumps(out, indent=1) + '\n')
    for group in ('ours', 'scale'):
        for k, v in out[group].items():
            print(group, k, 'PASS' if all(v['checks'].values()) else f"FAIL {[c for c, x in v['checks'].items() if not x]}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
