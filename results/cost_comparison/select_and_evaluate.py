"""PROTOCOL.md section 4: for one grid, select the run with the lowest mean development KL
(ties: smaller LR; non-finite or incomplete runs are not eligible) and print its directory.

python results/cost_comparison/select_and_evaluate.py RUNS_DIR GRID_PREFIX   (e.g. grid_C1)
"""
import json
import math
import sys
from pathlib import Path


def main():
    runs, prefix = Path(sys.argv[1]), sys.argv[2]
    rows = []
    for p in sorted(runs.glob(f'{prefix}_lr*/report.json')):
        if p.parent.name.endswith('_eval'):
            continue
        r = json.loads(p.read_text())
        kl = r.get('final_dev', {}).get('kl')
        ok = r['status'] == 'complete' and kl is not None and math.isfinite(kl) and r.get('state_sha256')
        rows.append((kl if ok else math.inf, r['config']['lr'], p.parent.name, r['status']))
        print(f'{p.parent.name:32s} status {r["status"]:9s} lr {r["config"]["lr"]:<8g} dev KL {kl}', file=sys.stderr)
    best = min(rows)
    assert math.isfinite(best[0]), 'no eligible run'
    print(best[2])


if __name__ == '__main__':
    main()
