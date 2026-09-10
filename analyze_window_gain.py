"""Is the below-BF16 gain uniform, or concentrated where BF16 is already surprised?

A temperature-like flattening pays off in proportion to how much probability mass
sits off the true token, so its per-window gain should grow with the window's own
BF16 NLL. A genuine modelling improvement need not.

Reads only committed per-window NLL from the released BF16 reproduction and the
paper-aligned adaptive run; windows are matched by token hash.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BF16 = ROOT / 'results/released_reproduction/job_335302/qwen3-4b_bf16/report.json'
ADAPT = ROOT / 'results/adaptive_paper/job_335993/qwen4b/report.json'


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def slope(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    return (sum((a - mx) * (b - my) for a, b in zip(x, y))
            / sum((a - mx) ** 2 for a in x))


def main():
    bf = json.loads(BF16.read_text())
    ad = json.loads(ADAPT.read_text())
    by_hash = {w['input_sha256']: w['nll'] for w in bf['windows']}

    for ds, key in (('wiki', 'wiki'), ('c4', 'c4_paper')):
        hashes = ad['data'][key]['token_sha256']
        assert all(h in by_hash for h in hashes), ds
        b = [by_hash[h] for h in hashes]
        q = ad['evaluation']['n65536'][ds]['nll']
        f = ad['evaluation']['four_over_six'][ds]['nll']
        gain = [bi - qi for bi, qi in zip(b, q)]
        damage = [fi - bi for fi, bi in zip(f, b)]
        n = len(b)
        print(f'--- {ds} ({n} windows) ---')
        print(f'  bf16 mean NLL {sum(b) / n:.4f}  range {min(b):.2f}..{max(b):.2f}')
        print(f'  n65536 gain over bf16: mean {sum(gain) / n:+.4f} nats, '
              f'corr with bf16 NLL {corr(b, gain):+.3f}, slope {slope(b, gain):+.4f}')
        print(f'  fourOverSix damage vs bf16: mean {sum(damage) / n:+.4f} nats, '
              f'corr with bf16 NLL {corr(b, damage):+.3f}, slope {slope(b, damage):+.4f}')
        print(f'  windows where n65536 beats bf16: {sum(g > 0 for g in gain)}/{n}')
        print()


if __name__ == '__main__':
    main()
