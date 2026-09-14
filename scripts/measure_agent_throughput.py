"""
    Time a served policy at the shape a Terminal-Bench agent turn actually has, and convert
    that into hours per trial.

    An agent turn is a long prompt -- the conversation plus the terminal state -- and a
    short-to-medium completion, repeated up to the turn cap. So throughput at that shape, not
    peak throughput, is what decides whether a comparison is affordable. The 4B model was
    measured at about three hours per trial this way; this exists so the 24B figure is measured
    rather than extrapolated from it.

        python scripts/measure_agent_throughput.py --port 18137 --out <dir>
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

# (approximate prompt tokens, completion cap) -- two turn shapes plus a longer-context one.
SHAPES = ((2000, 128), (2000, 512), (6000, 512))

# A terminus-2 turn generates on the order of this many tokens.
TOKENS_PER_TURN = 400


def call(url, ctx_tokens, max_tokens):
    filler = 'The quick brown fox jumps over the lazy dog. ' * max(ctx_tokens // 10, 1)
    body = json.dumps({
        'model': 'qmodel', 'max_tokens': max_tokens, 'temperature': 0,
        'messages': [{'role': 'user',
                      'content': filler + '\nSummarise the above in one line.'}],
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    dt = time.time() - t0
    usage = r.get('usage', {})
    return dt, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    url = f'http://127.0.0.1:{args.port}/v1/chat/completions'

    rows = []
    for ctx, mx in SHAPES:
        dt, n_in, n_out = call(url, ctx, mx)
        tps = n_out / dt if dt > 0 else 0.0
        rows.append(dict(ctx_requested=ctx, max_tokens=mx, seconds=round(dt, 1),
                         prompt_tokens=n_in, completion_tokens=n_out,
                         tokens_per_s=round(tps, 2)))
        print(f'ctx~{ctx:5d} max_new={mx:4d}  {dt:7.1f}s  in={n_in:6d} out={n_out:5d}  '
              f'{tps:6.2f} tok/s', flush=True)

    rates = [r['tokens_per_s'] for r in rows if r['tokens_per_s'] > 0]
    if not rates:
        print('no successful generation; cannot estimate')
        return 1
    slow = min(rates)
    per_turn = TOKENS_PER_TURN / slow
    print(f'\nat {slow:.2f} tok/s, a {TOKENS_PER_TURN}-token turn takes {per_turn / 60:.1f} min')
    est = {}
    for turns in (10, 15, 20):
        trial_h = turns * per_turn / 3600
        est[turns] = dict(hours_per_trial=round(trial_h, 2),
                          hours_15_tasks=round(15 * trial_h, 1),
                          hours_3_policies=round(3 * 15 * trial_h, 1))
        print(f'  {turns:2d} turns/trial -> {trial_h:5.2f} h per trial, '
              f'{15 * trial_h:6.1f} h for 15 tasks, '
              f'{3 * 15 * trial_h:6.1f} h for 3 policies')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'throughput.json').write_text(
        json.dumps(dict(measurements=rows, tokens_per_turn=TOKENS_PER_TURN,
                        slowest_tokens_per_s=slow, estimates=est), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
