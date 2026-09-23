"""Aggregate the local N16K64 reproduction (fake quant + native mixfp4 kernel) into REPORT.md.

Criteria are the original campaign's pre-registered cross-GPU tolerances
(research/n16k64/campaigns/primary/PROTOCOL_FREEZE.json -> numerical_tolerances):
  PPL W4A4 |rel| <= 0.5%, BF16 <= 0.1%; paired effect same sign and |diff| <= max(0.25|orig|, 0.002);
  k=3 tile count |rel| <= 5% (historical tierB k3_count_rel).
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/home/dev/NVFP4-RaZeR-n16k64')
RUNS = Path('/home/dev/n16k64_campaign/runs')
ORIG = json.load(open(ROOT / 'repro_local/original_reference.json'))
LEG = {}
for f in ('LEGACY_PANEL_PPL', 'CONFIRMATORY_PPL'):
    LEG.update(json.load(open(ROOT / f'research/n16k64/campaigns/primary/analysis_ppl/{f}.json'))['models'])
MODELS = [('llama8b', 'Llama-3.1-8B'), ('qwen27b', 'Qwen3.8-27B'), ('qwen4b', 'Qwen3-4B'),
          ('mistral7b', 'Mistral-7B-v0.3'), ('phi4', 'Phi-4')]
POLS = ['nvfp4', 'four_over_six', 'n8_k3', 'n16_k3']
HEAD = {'nvfp4': 'NVFP4', 'four_over_six': '4Over6', 'n8_k3': 'N8K64(k=3)', 'n16_k3': 'N16K64(k=3)'}
CORP = {'wiki': 'Wiki', 'c4': 'C4'}


def latest(name):
    runs = sorted(RUNS.glob(f'{name}_attempt*'), key=lambda p: int(p.name.rsplit('attempt', 1)[1]))
    runs = [r for r in runs if json.load(open(r / 'launch_record.json')).get('status') == 'complete']
    return runs[-1] if runs else None


def evaluation(names, sub):
    ev = {}
    for n in names:
        r = latest(n)
        if r is not None:
            for k, v in json.load(open(r / sub / 'ppl_report.json'))['evaluation'].items():
                ev.setdefault(k, v)
    return ev


def windows(ev, pol, dom):
    return np.array([w['nll_mean'] for w in ev[pol][dom]['windows']])


def calib_counts(model):
    r = latest(f'calib_{model}')
    if r is None:
        return None
    return json.load(open(r / 'calibration' / 'calibration_report.json'))['counts']


def collect(model):
    if model == 'llama8b':
        fake = evaluation(['fake_baselines_llama8b', 'fake_maps_llama8b'], 'ppl')
        real = evaluation(['real_baselines_llama8b', 'real_maps_llama8b'], 'ppl_real')
    else:
        fake = evaluation([f'fake_{model}'], 'ppl')
        real = evaluation([f'real_{model}'], 'ppl_real')
    # native: NVFP4 / FourOverSix / N16 on the weights-as-A kernel, N8 on the weights-as-B kernel;
    # each map's native effect is paired with FourOverSix on the SAME kernel.
    native = {p: p for p in ('nvfp4', 'four_over_six', 'n16_k3', 'n8_k3')}
    native_fo6_for = {'n16_k3': 'four_over_six', 'n8_k3': 'four_over_six_b8x64', 'nvfp4': 'four_over_six'}
    return fake, real, native, native_fo6_for


def fmt(x, nd=4):
    return '—' if x is None else f'{x:.{nd}f}'


def main():
    lines = ['# N16K64 reproduction on one RTX PRO 6000 (fake quant + native mixfp4 kernel)', '']
    summary, eff_rows, tile_rows, nf_rows = [], [], [], []
    for key, label in MODELS:
        if key not in ORIG:
            continue
        fake, real, native, fo6_for = collect(key)
        counts = calib_counts(key)
        for dom in ('wiki', 'c4'):
            row = [label, CORP[dom]]
            for p in POLS:
                o = ORIG[key]['ppl'][p][dom]
                f = fake.get(p, {}).get(dom, {}).get('ppl')
                r = real.get(native[p], {}).get(dom, {}).get('ppl')
                row.append((o, f, r))
            summary.append(row)
            for a, b in (('n8_k3', 'four_over_six'), ('n16_k3', 'four_over_six'), ('nvfp4', 'four_over_six')):
                c = LEG[key]['contrasts'].get(f'{a}-{b}')
                o = None if c is None else c[dom]['estimate']
                res = []
                for src, ev, bpol in (('fake', fake, b), ('native', real, fo6_for[a])):
                    if a in ev and bpol in ev and dom in ev[a]:
                        d = windows(ev, a, dom) - windows(ev, bpol, dom)
                        res.append((src, float(d.mean()), float(2 * d.std(ddof=1) / math.sqrt(len(d)))))
                for src, est, se2 in res:
                    ok = o is not None and np.sign(est) == np.sign(o) and abs(est - o) <= max(0.25 * abs(o), 0.002)
                    eff_rows.append((label, CORP[dom], f'{a}−{b}', src, est, se2, o, ok))
            for p in POLS:
                if p in fake and native[p] in real and dom in fake[p]:
                    d = windows(real, native[p], dom) - windows(fake, p, dom)
                    nf_rows.append((label, CORP[dom], p, float(d.mean()), float(2 * d.std(ddof=1) / math.sqrt(len(d)))))
        if counts:
            for p in ('n8_k3', 'n16_k3'):
                o = LEG[key]['selected_tiles'][p]
                tile_rows.append((label, p, o, counts[p], counts[p] / o - 1))

    lines += ['## PPL: original / fake quant (this machine) / native kernel (this machine)', '',
              '| Model | Corpus | ' + ' | '.join(HEAD[p] for p in POLS) + ' |',
              '|---|---|' + '---:|' * len(POLS)]
    for row in summary:
        cells = []
        for (o, f, r) in row[2:]:
            cells.append(f'{fmt(o)} / {fmt(f)} / {fmt(r)}')
        lines.append(f'| {row[0]} | {row[1]} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Each cell: original / fake quant / native kernel. Tolerance: |rel| ≤ 0.5%.', '']
    lines += ['| Model | Corpus | Policy | fake rel | native rel |', '|---|---|---|---:|---:|']
    for row in summary:
        for p, (o, f, r) in zip(POLS, row[2:]):
            fr = None if f is None else f / o - 1
            rr = None if r is None else r / o - 1
            mark = lambda x: '—' if x is None else f"{100 * x:+.3f}%{'' if abs(x) <= 0.005 else ' ✗'}"
            lines.append(f'| {row[0]} | {row[1]} | {HEAD[p]} | {mark(fr)} | {mark(rr)} |')
    lines += ['', '## Tile counts (k=3, CE+KL conjunction)', '', '| Model | Map | original | this machine | rel |',
              '|---|---|---:|---:|---:|']
    for label, p, o, got, rel in tile_rows:
        lines.append(f"| {label} | {HEAD[p]} | {o:,} | {got:,} | {100 * rel:+.1f}%{'' if abs(rel) <= 0.05 else ' ✗'} |")
    lines += ['', '## Paired effects, ΔlogPPL vs FourOverSix (tierC: same sign, |diff| ≤ max(0.25|orig|, 0.002))', '',
              '| Model | Corpus | Contrast | Source | local ± 2SE | original | tierC |', '|---|---|---|---|---:|---:|---|']
    for label, corp, con, src, est, se2, o, ok in eff_rows:
        lines.append(f"| {label} | {corp} | {con} | {src} | {est:+.5f} ± {se2:.5f} | {fmt(o, 5)} | {'PASS' if ok else 'FAIL'} |")
    lines += ['', '## Native kernel − fake quant, per-window paired ΔNLL', '', '| Model | Corpus | Policy | mean ± 2SE |',
              '|---|---|---|---:|']
    for label, corp, p, m, se2 in nf_rows:
        lines.append(f'| {label} | {corp} | {HEAD[p]} | {m:+.5f} ± {se2:.5f} |')
    out = ROOT / 'repro_local' / 'REPORT_TABLES.md'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
