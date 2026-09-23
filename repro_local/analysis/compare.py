"""Compare local fake/real reproduction runs with the original N16K64 primary-campaign results.

Criteria are the original campaign's own pre-registered cross-GPU tolerances
(research/n16k64/campaigns/primary/PROTOCOL_FREEZE.json, numerical_tolerances):
  PPL (W4A4)      |relative difference| <= 0.5%;  BF16 <= 0.1%
  paired effect   same sign and |local - original| <= max(0.25 |original|, 0.002)  (dlogPPL)
  k=3 tile count  |relative difference| <= 5%   (historical tierB k3_count_rel)
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/dev/NVFP4-RaZeR-n16k64')
RUNS = Path('/home/dev/n16k64_campaign/runs')
ORIG = json.load(open(ROOT / 'repro_local/original_reference.json'))
LEG = {}
for f in ('LEGACY_PANEL_PPL', 'CONFIRMATORY_PPL'):
    LEG.update(json.load(open(ROOT / f'research/n16k64/campaigns/primary/analysis_ppl/{f}.json'))['models'])


def latest(name):
    runs = sorted(RUNS.glob(f'{name}_attempt*'), key=lambda p: int(p.name.rsplit('attempt', 1)[1]))
    runs = [r for r in runs if json.load(open(r / 'launch_record.json')).get('status') == 'complete']
    return runs[-1] if runs else None


def load_eval(run, sub):
    return json.load(open(run / sub / 'ppl_report.json'))['evaluation']


def nll(ev, pol, dom):
    return np.array([w['nll_mean'] for w in ev[pol][dom]['windows']])


def paired(a, b):
    d = a - b
    return float(d.mean()), float(2 * d.std(ddof=1) / math.sqrt(len(d)))


def orig_effect(model, a, b, dom):
    c = LEG[model]['contrasts'].get(f'{a}-{b}')
    return None if c is None else c[dom]['estimate']


def tol_effect(orig):
    return max(0.25 * abs(orig), 0.002)


def main(model, sources):
    """sources: list of (label, run_name, subdir, {local_policy: canonical_policy})"""
    out = dict(model=model, rows=[], effects=[])
    evs = {}
    for label, name, sub, polmap in sources:
        run = latest(name)
        if run is None:
            print(f'{label}: {name} not complete'); continue
        ev = load_eval(run, sub)
        for lp, cp in polmap.items():
            if lp in ev:
                evs[(label, cp)] = (ev, lp)
    for (label, cp), (ev, lp) in sorted(evs.items()):
        for dom in ('wiki', 'c4'):
            got = ev[lp][dom]['ppl']; want = ORIG[model]['ppl'][cp][dom]
            rel = got / want - 1
            tol = 0.001 if cp == 'bf16' else 0.005
            out['rows'].append(dict(source=label, policy=cp, local_name=lp, corpus=dom, local=got, original=want,
                                    rel=rel, pass_=abs(rel) <= tol))
    for label in sorted({k[0] for k in evs}):
        for a, b in (('n8_k3', 'four_over_six'), ('n16_k3', 'four_over_six'), ('n16_k3', 'n8_k3'), ('nvfp4', 'four_over_six')):
            if (label, a) not in evs or (label, b) not in evs:
                continue
            for dom in ('wiki', 'c4'):
                (eva, la), (evb, lb) = evs[(label, a)], evs[(label, b)]
                est, se2 = paired(nll(eva, la, dom), nll(evb, lb, dom))
                o = orig_effect(model, a, b, dom)
                ok = None if o is None else (np.sign(est) == np.sign(o) and abs(est - o) <= tol_effect(o))
                out['effects'].append(dict(source=label, contrast=f'{a}-{b}', corpus=dom, local=est, local_2se=se2,
                                           original=o, tol=None if o is None else tol_effect(o), pass_=ok))
    return out


def show(res):
    print(f"== {res['model']}: PPL (criterion |rel| <= 0.5% W4A4, 0.1% BF16)")
    for r in res['rows']:
        print(f"  {r['source']:5s} {r['policy']:14s} {r['corpus']:4s} local {r['local']:.6f} orig {r['original']:.6f} rel {100*r['rel']:+.3f}% {'PASS' if r['pass_'] else 'FAIL'}")
    print(f"== {res['model']}: paired effects dlogPPL (criterion: same sign, |diff| <= max(0.25|orig|, 0.002))")
    for e in res['effects']:
        o = 'n/a' if e['original'] is None else f"{e['original']:+.5f}"
        p = '' if e['pass_'] is None else ('PASS' if e['pass_'] else 'FAIL')
        print(f"  {e['source']:5s} {e['contrast']:26s} {e['corpus']:4s} local {e['local']:+.5f} (2SE {e['local_2se']:.5f}) orig {o} {p}")


if __name__ == '__main__':
    model = sys.argv[1]
    src = json.loads(sys.argv[2])
    res = main(model, src)
    show(res)
    Path(ROOT / 'repro_local/analysis').mkdir(exist_ok=True)
    json.dump(res, open(ROOT / f'repro_local/analysis/compare_{model}.json', 'w'), indent=1, default=float)
