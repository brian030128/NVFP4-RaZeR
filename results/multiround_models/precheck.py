"""Stage 4 of PROTOCOL.md: native vs fake WikiText-2 / C4 NLL at the FourOverSix map.

python results/multiround_models/precheck.py NATIVE_EVAL FAKE_EVAL > precheck.json

Both are run_multiround.py --evaluate-map FourOverSix=fourover6 runs (native / fake backend) on
identical windows (token hashes compared). PASS iff on each corpus the per-window mean NLL
difference native - fake is within +-0.01 nats, and the native run's map check (0 mismatches) and
activation checks (the first 64 calls, bitwise) passed. Exit 1 on FAIL.
"""
import json
import math
import sys
from pathlib import Path

THRESHOLD = 0.01


def main():
    native, fake = (json.loads((Path(p) / 'report.json').read_text()) for p in sys.argv[1:3])
    assert native['status'] == fake['status'] == 'complete'
    same_windows = all(native['data'][d]['token_sha256'] == fake['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper'))
    n, f = native['evaluations']['FourOverSix'], fake['evaluations']['FourOverSix']
    out = dict(threshold_nats=THRESHOLD, same_windows=same_windows, corpora={})
    ok = same_windows
    for d in ('wiki', 'c4'):
        diff = [a - b for a, b in zip(n['evaluation'][d]['nll'], f['evaluation'][d]['nll'])]
        mean = sum(diff) / len(diff)
        sd = math.sqrt(sum((x - mean) ** 2 for x in diff) / (len(diff) - 1))
        within = abs(mean) <= THRESHOLD
        ok &= within
        out['corpora'][d] = dict(windows=len(diff), native_ppl=n['evaluation'][d]['ppl'], fake_ppl=f['evaluation'][d]['ppl'],
                                 mean_native_minus_fake=mean, two_se=2 * sd / math.sqrt(len(diff)), within_threshold=within)
    out['native_map_mismatches'] = n.get('native_map_mismatches')
    out['native_activation_checks'] = n.get('native_activation_checks')
    out['native_start_verification'] = native.get('native', {}).get('verification')
    ok &= n.get('native_map_mismatches') == [] and (n.get('native_activation_checks') or 0) >= 64
    out['passed'] = ok
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
