"""The baselines' learning rate (PROTOCOL.md): the lowest final development KL, with registered edge extensions.

    python results/unified_baselines/choose_lr.py next RUNS ARM   -> prints the next rate to run, or 'DONE <rate>', or
                                                                     'STOP <reason>'; writes RUNS/lr_choice_<ARM>.json
                                                                     once the choice is made or the rule stops

Runs are RUNS/<ARM>_dev_lr<rate>/report.json (run_cost_distill.py with the development set).
- scale: grid {1e-4, 3e-4, 1e-3}, one extension (x3: 3e-5 below, 3e-3 above) -- Task 2's rule.
- qat: grid {1e-6, 1e-5, 1e-4} (the cost study's registered arm-C grid), up to two extensions by the grid's own
  step (x10: 1e-7 then 1e-8 below, 1e-3 then 1e-2 above).
- Each extension is run only while the lowest KL of the rates run so far sits at an edge, in that edge's direction.
- If the lowest KL still sits at an edge once that direction's extensions are used up, the rule stops: no choice.
- Ties go to the smaller rate. A non-finite KL disqualifies a run.
"""
import json
import math
import sys
from pathlib import Path

RULES = {'scale': dict(grid=('1e-4', '3e-4', '1e-3'), below=('3e-5',), above=('3e-3',)),
         'qat': dict(grid=('1e-6', '1e-5', '1e-4'), below=('1e-7', '1e-8'), above=('1e-3', '1e-2'))}


def final_kl(runs, arm, rate):
    p = runs / f'{arm}_dev_lr{rate}' / 'report.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    assert r['status'] == 'complete' and r['fused_quantizer_bitwise_on_final_weights'], (arm, rate)
    kl = r['final_dev']['kl']
    return kl if math.isfinite(kl) else float('inf')


def main():
    what, runs, arm = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    assert what == 'next'
    rule = RULES[arm]
    for rate in rule['grid']:
        if final_kl(runs, arm, rate) is None:
            print(rate)
            return
    tried = list(rule['grid'])
    extensions = []
    while True:
        kls = {rate: final_kl(runs, arm, rate) for rate in tried}
        best = min(sorted(tried, key=float), key=lambda rate: kls[rate])
        low, high = min(tried, key=float), max(tried, key=float)
        direction = 'below' if best == low else 'above' if best == high else None
        if direction is None:
            verdict = dict(chosen=best)
            break
        ladder = [r for r in rule[direction] if r not in tried]
        if not ladder:
            verdict = dict(chosen=None, stop=f'the lowest KL ({best}) is still at the {direction} edge after the extensions')
            break
        nxt = ladder[0]
        if final_kl(runs, arm, nxt) is None:
            print(nxt)
            return
        tried.append(nxt)
        extensions.append(nxt)
    out = dict(arm=arm, grid=list(rule['grid']), ladders=dict(below=list(rule['below']), above=list(rule['above'])),
               extensions_run=extensions, final_dev_kl={r: final_kl(runs, arm, r) for r in tried},
               rule='lowest final development KL (per-token activations, the deployed weight); edge extensions as '
                    'registered; ties to the smaller rate', **verdict)
    (runs / f'lr_choice_{arm}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f"DONE {verdict['chosen']}" if verdict['chosen'] else f"STOP {verdict['stop']}")


if __name__ == '__main__':
    main()
