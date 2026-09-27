"""SCALE's learning rate (PROTOCOL.md): the lowest final development KL over the grid, with one edge extension.

    python results/scale_additivity/choose_lr.py RUNS edge     -> prints the extension rate, or nothing
    python results/scale_additivity/choose_lr.py RUNS choose   -> writes RUNS/lr_choice.json, prints the chosen rate

The grid is {1e-4, 3e-4, 1e-3} (runs RUNS/scale_dev_lr<rate>). If its lowest final development KL is at an edge, the
grid is extended once by the next rate in that direction (x3: 3e-3 above, 3e-5 below), and the choice is the lowest of
the extended grid. Ties go to the smaller rate.
"""
import json
import sys
from pathlib import Path

GRID = ('1e-4', '3e-4', '1e-3')
EDGE = {'1e-3': '3e-3', '1e-4': '3e-5'}


def final_kl(runs, rate):
    r = json.loads((runs / f'scale_dev_lr{rate}' / 'report.json').read_text())
    assert r['status'] == 'complete' and r['fused_quantizer_bitwise_on_final_weights'], rate
    return r['final_dev']['kl'], r['initial_dev']['kl']


def lowest(runs, rates):
    kls = {rate: final_kl(runs, rate) for rate in rates}
    return min(sorted(rates, key=float), key=lambda rate: kls[rate][0]), kls


def main():
    runs, what = Path(sys.argv[1]), sys.argv[2]
    best, _ = lowest(runs, GRID)
    extension = EDGE.get(best)
    if what == 'edge':
        print(extension or '')
        return
    rates = GRID + ((extension,) if extension else ())
    chosen, kls = lowest(runs, rates)
    out = dict(grid=list(GRID), extension=extension, final_dev_kl={k: v[0] for k, v in kls.items()},
               initial_dev_kl={k: v[1] for k, v in kls.items()}, chosen=chosen,
               rule='lowest final development KL (math/code, per-token activations); one edge extension; ties to the smaller rate')
    (runs / 'lr_choice.json').write_text(json.dumps(out, indent=1) + '\n')
    print(chosen)


if __name__ == '__main__':
    main()
