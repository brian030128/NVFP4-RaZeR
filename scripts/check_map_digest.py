"""
    Does a regenerated math/code calibration reproduce the shipped map exactly?

    The shipped calibration directories under results/math_code_adaptive/ kept only report.json
    and maps.json; their scores/ shards, which the k-SE election reads, are gone. So the k = 3
    election has to be rebuilt from a fresh scoring pass -- and a fresh pass is only the shipped
    policy if it lands on the shipped map.

    This compares the whole of maps.json, which is INFORMATIONAL rather than decisive: the file
    holds 20 maps and only fixed256_math_code128 cross-checks the k-SE election, so a difference
    in the adaptive_* searches -- which nothing here uses -- shows up as a mismatch that means
    nothing for this experiment. Qwen3.8-27B does exactly that while reproducing all 128 teacher
    losses bit for bit and matching weight_mse_sha256.

    The gate that decides lives in run_zeroshot_kse.py: the re-elected 256-tile prefix must equal
    the SHIPPED frozen map bitwise, or it refuses to evaluate.

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
        print('DIFFERS: maps.json is not identical to the shipped one. This alone does not mean '
              'the election differs -- 19 of its 20 maps are unused here. The frozen-map gate in '
              'run_zeroshot_kse.py is what decides.')
        return 1
    print('MAP DIGEST MATCHES SHIPPED (all 20 maps identical)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
