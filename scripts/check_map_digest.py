"""
    Does a regenerated math/code calibration reproduce the shipped map exactly?

    The shipped calibration directories under results/math_code_adaptive/ kept only report.json
    and maps.json; their scores/ shards, which the k-SE election reads, are gone. So the k = 3
    election has to be rebuilt from a fresh scoring pass -- and a fresh pass is only the shipped
    policy if it lands on the shipped map.

    This is the gate for that, and it is the reason the calibration's own source-digest check can
    be relaxed: comparing what the pass produced is a stronger statement than comparing the
    bytes of the file that produced it.

        python scripts/check_map_digest.py <fresh calibration dir> <model>
"""
import json
import sys
from pathlib import Path

# The job that produced each shipped calibration, per run_kse_paper.MODELS.
SHIPPED_JOB = {'llama8b': '333779', 'qwen4b': '333779', 'qwen27b': '333787'}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    calib, model = Path(sys.argv[1]), sys.argv[2]
    if model not in SHIPPED_JOB:
        print(f'FATAL: unknown model {model}')
        return 1

    fresh = json.loads((calib / 'report.json').read_text())['map_sha256']
    shipped_path = Path(
        f'results/math_code_adaptive/calibration_{SHIPPED_JOB[model]}_{model}/report.json')
    if not shipped_path.is_file():
        print(f'FATAL: no shipped calibration to compare against at {shipped_path}')
        return 1
    shipped = json.loads(shipped_path.read_text())['map_sha256']

    print(f'MAP DIGEST fresh   = {fresh}')
    print(f'MAP DIGEST shipped = {shipped}')
    if fresh != shipped:
        print('FATAL: the regenerated calibration does not reproduce the shipped map, so the '
              'election built from it is not the shipped k = 3 policy.')
        return 1
    print('MAP DIGEST MATCHES SHIPPED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
