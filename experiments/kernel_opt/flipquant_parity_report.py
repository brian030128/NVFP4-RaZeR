#!/usr/bin/env python3
"""Tables and criteria of the flipquant GEMM parity check (results/kernel_opt/flipquant_parity/NOTE.md). CPU only.

    python experiments/kernel_opt/flipquant_parity_report.py --src /home/dev/n16k64_campaign/kernel_opt/fq_parity \
        --out-dir results/kernel_opt/flipquant_parity --runs llama8b,llama8b_r2,llama8b_r3,llama8b_aa

Reads gemm_<run>.json (flipquant_parity_gemm.py) and binaries_<run>.json (flipquant_parity_binaries.py) of each run.
Writes parity.md (the per-unit tables with every run's ratio, the criteria per run, the cell-level spread per run),
cells.md (every cell of every run), parity.json (the summary), and copies the raw records as gemm_<run>_raw.json and
binaries_<run>.json. A run made with --control aa is the A/A control: its 'fq' side is a second NVFP4-RaZeR process.

Per-forward GEMM time of a policy at T: the sum over the projections of (modules of the projection) x (the median GEMM
time of its timed module), per side; per round, the same with the round's medians.
Criteria (as asked):
- the same kernel selection: per cell, the same build, width and scheduler row on both sides, in every round, from
  tables with the same content; the same activation quantizer (the same kernel, from the same build, the same mode);
- the same binaries: per loaded build, the same SASS (manifest and recomputed) and the same device and host code;
- the per-forward GEMM sum of every (model, policy, T) within +-1 %;
- no cell where flipquant is slower than RaZeR by more than 2 % in every round.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SUM_TOL, CELL_TOL = 0.01, 0.02
UNIT = {'8x64': '8x64 map, mixed_wB_ko', '16x64': '16x64 map, mixed_ko', '256x64': '256x64 map, mixed256_ko',
        'stock_ko': 'FourOverSix, stock_ko', 'stock_wB_ko': 'FourOverSix, stock_wB_ko (weights on B)'}


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).exists() else None


def analyze(gemm, binaries):
    tags = gemm['setup']['rz']['tags']
    pols = list(gemm['policies'])
    tokens = gemm['tokens']
    rows = {(r['policy'], r['proj'], r['tokens'], r['side']): r for r in gemm['rows']}
    cells = {(c['policy'], c['proj'], c['tokens']): c for c in gemm['checks']['cells']}
    out = dict(model=gemm['model'], policies={}, cells=[], criteria={})
    # ---- selection, quantizer, outputs per cell
    sel_ok, quant_ok, out_ok, every_round_ok, table_ok = True, True, True, True, True
    tables = {}
    for (pol, proj, t), c in cells.items():
        f, z = c['fq'], c['rz']
        for side in ('fq', 'rz'):
            tables.setdefault(c[side]['table'], file_sha(c[side]['table']))
        rf, rr = rows[(pol, proj, t, 'fq')], rows[(pol, proj, t, 'rz')]
        same_sel = all(f[k] == z[k] for k in ('kernel', 'width', 'schedule', 'set', 'extra_defines', 'sass_sha256',
                                              'blob_gen', 'compiled_description'))
        same_rounds = rf['same_launch_every_round'] and rr['same_launch_every_round'] and \
            (rf['kernel'], rf['width'], rf['schedule']) == (f['kernel'], f['width'], f['schedule']) and \
            (rr['kernel'], rr['width'], rr['schedule']) == (z['kernel'], z['width'], z['schedule'])
        same_table = tables[f['table']] == tables[z['table']]
        same_quant = (f['quant_mode'] == z['quant_mode'] and rf['quant_names'] == rr['quant_names'] ==
                      f['forward_quant_names'] == z['forward_quant_names'] and len(rf['quant_names']) == 1)
        same_gemm_name = rf['gemm_names'] == rr['gemm_names'] == f['forward_gemm_names'] == z['forward_gemm_names'] and \
            len(rf['gemm_names']) == 1
        sel_ok &= same_sel and same_gemm_name
        every_round_ok &= same_rounds
        table_ok &= same_table
        quant_ok &= same_quant
        out_ok &= c['y_equal'] and c['isolated_equals_forward']
        ratio_rounds = [a['gemm_us'] / b['gemm_us'] for a, b in zip(sorted(rf['rounds'], key=lambda x: x['round']),
                                                                    sorted(rr['rounds'], key=lambda x: x['round']))]
        out['cells'].append(dict(policy=pol, proj=proj, tokens=t, kernel=f['kernel'], width=f['width'],
                                 schedule=f['schedule'], rz_us=rr['gemm_us'], fq_us=rf['gemm_us'],
                                 ratio=rf['gemm_us'] / rr['gemm_us'], round_ratios=ratio_rounds,
                                 slower_every_round=all(x > 1 + CELL_TOL for x in ratio_rounds),
                                 faster_every_round=all(x < 1 - CELL_TOL for x in ratio_rounds),
                                 rz_quant_us=rr['quant_us'], fq_quant_us=rf['quant_us'],
                                 same_selection=same_sel, same_gemm_kernel_name=same_gemm_name, same_quantizer=same_quant,
                                 same_launch_every_round=same_rounds, same_table_content=same_table,
                                 outputs_equal=c['y_equal'], isolated_equals_forward=c['isolated_equals_forward']))
    # ---- per-forward sums
    sums_ok = True
    for pol in pols:
        projs = list(tags[pol])
        per = []
        for t in tokens:
            s = {}
            for side in ('fq', 'rz'):
                s[side] = sum(tags[pol][p]['modules'] * rows[(pol, p, t, side)]['gemm_us'] for p in projs)
                s[side + '_rounds'] = [sum(tags[pol][p]['modules'] * sorted(rows[(pol, p, t, side)]['rounds'],
                                                                         key=lambda x: x['round'])[r]['gemm_us'] for p in projs)
                                       for r in range(gemm['method']['rounds'])]
                s[side + '_quant'] = sum(tags[pol][p]['modules'] * rows[(pol, p, t, side)]['quant_us'] for p in projs)
            ratio = s['fq'] / s['rz']
            rr = [a / b for a, b in zip(s['fq_rounds'], s['rz_rounds'])]
            ok = abs(ratio - 1) <= SUM_TOL
            sums_ok &= ok
            per.append(dict(tokens=t, rz_us=s['rz'], fq_us=s['fq'], ratio=ratio, round_ratios=rr, within_1pct=ok,
                            rz_quant_us=s['rz_quant'], fq_quant_us=s['fq_quant'],
                            widths={p: rows[(pol, p, t, 'rz')]['width'] for p in projs}))
        out['policies'][pol] = dict(label=UNIT[pol], modules={p: tags[pol][p]['module'] for p in projs},
                                    modules_per_projection={p: tags[pol][p]['modules'] for p in projs},
                                    e0m3_tiles={p: tags[pol][p]['e0m3_tiles'] for p in projs},
                                    tiles_per_module={p: tags[pol][p]['tiles_per_module'] for p in projs},
                                    per_forward=per,
                                    routing=dict(fq=gemm['setup']['fq']['routing'][pol],
                                                 rz=gemm['setup']['rz']['routing'][pol]))
    slow = [c for c in out['cells'] if c['slower_every_round']]
    b = binaries['builds']
    out['tables'] = tables
    out['binaries'] = {n: dict(identical=r['identical'], sass_equal=r['sass_recomputed']['equal'],
                               sass_equals_manifest=r['sass_recomputed']['equals_manifest'],
                               device_code_equal=r['device_code_equal'], host_code_equal=r['host_code_equal'],
                               file_bytes_equal=r['library_sha256']['equal'],
                               host_sections_differing=sorted(r['host']['differing']),
                               cubin_sections_differing=sorted({s for c in r['cubins'] for s in c['differing']}),
                               extra_defines=r['manifest']['extra_defines']) for n, r in b.items()}
    out['criteria'] = {
        'operands equal bit for bit (every timed module)': all(all(v.values()) for v in gemm['checks']['operands'].values()),
        'outputs equal bit for bit (fq vs rz; isolated path vs the fused forward)': out_ok,
        'same kernel selection: build, width, scheduler row, defines, SASS hash, GEMM kernel name': sel_ok,
        'same launch in every round': every_round_ok,
        'the tile tables have the same content': table_ok and len(set(tables.values())) == 1,
        'same activation quantizer (kernel name, build, mode)': quant_ok,
        'same binaries (SASS, device code, host code; loaded bytes up to the compilation id)': binaries['all_identical'],
        'per-forward GEMM sum within +-1 % for every policy and T': sums_ok,
        'no cell slower in every round by more than 2 %': not slow}
    out['passed'] = all(out['criteria'].values())
    out['slower_every_round'] = slow
    out['faster_every_round'] = [c for c in out['cells'] if c['faster_every_round']]
    out['max_abs_cell_deviation'] = max(abs(c['ratio'] - 1) for c in out['cells'])
    out['max_abs_sum_deviation'] = max(abs(p['ratio'] - 1) for v in out['policies'].values() for p in v['per_forward'])
    out['run'] = dict(started=gemm['started_utc'], finished=gemm['finished_utc'], sampler=gemm.get('sampler'),
                      razer=gemm['razer'], flipquant=gemm['flipquant'], flipquant_after=gemm['flipquant_after'],
                      method=gemm['method'], power_limit_w=gemm.get('power_limit_w'), gpu=gemm['setup']['rz']['gpu'])
    return out


NAMES = dict(fq='flipquant', rz='RaZeR harness', rzm='RaZeR model path')


def label(stem, a):
    pair = a.get('pair') or dict(fq='rz' if a['control'] == 'aa' else 'fq', rz='rz')
    return f"{stem} ({NAMES[pair['fq']]} / {NAMES[pair['rz']]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/fq_parity'))
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'results' / 'kernel_opt' /
                    'flipquant_parity')
    ap.add_argument('--runs', default='llama8b', help='comma-separated run stems (gemm_<stem>.json, binaries_<stem>.json)')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for stem in args.runs.split(','):
        g, b = args.src / f'gemm_{stem}.json', args.src / f'binaries_{stem}.json'
        gemm, binaries = json.loads(g.read_text()), json.loads(b.read_text())
        assert gemm['status'] == 'complete', (stem, gemm['status'])
        a = analyze(gemm, binaries)
        a['control'] = gemm.get('control', 'none')
        a['sides'] = gemm.get('sides')
        a['pair'] = gemm.get('pair') or dict(fq='rz' if a['control'] == 'aa' else 'fq', rz='rz')
        runs[stem] = a
        shutil.copyfile(g, args.out_dir / f'gemm_{stem}_raw.json')
        shutil.copyfile(b, args.out_dir / f'binaries_{stem}.json')
    pairs = [s for s, a in runs.items() if a['pair'] == dict(fq='fq', rz='rz')]
    controls = [s for s in runs if s not in pairs]
    first = runs[pairs[0]]
    md = ['# flipquant GEMM parity: tables\n',
          'Per-forward GEMM time of a policy at T: the sum over the projections of (modules) x (the median GEMM time of the '
          'timed module), in µs. RaZeR and flipquant are the values of ' + pairs[0] + '; then the flipquant / RaZeR ratio '
          'of every flipquant / RaZeR-harness run, and the ratio of every other pairing (first slot / second slot): the '
          'A/A control (two RaZeR harness processes), flipquant against RaZeR\'s own model path, RaZeR\'s model path '
          'against its harness.\n']
    for pol, p in first['policies'].items():
        md += [f"## {p['label']}\n",
               '| T | RaZeR | flipquant | ' + ' | '.join(f'{s}: fq/rz' for s in pairs)
               + ''.join(f" | {s}: {NAMES[runs[s]['pair']['fq']]} / {NAMES[runs[s]['pair']['rz']]}" for s in controls)
               + ' |', '|---:|---:|---:|' + '---:|' * (len(pairs) + len(controls))]
        for i, r in enumerate(p['per_forward']):
            cells = [f"{runs[s]['policies'][pol]['per_forward'][i]['ratio']:.4f}" for s in pairs + controls]
            md.append(f"| {r['tokens']} | {r['rz_us']:,.1f} | {r['fq_us']:,.1f} | " + ' | '.join(cells) + ' |')
        md.append('')
    md += ['## Criteria per run\n', '| criterion | ' + ' | '.join(label(s, a) for s, a in runs.items()) + ' |',
           '|---|' + ':-:|' * len(runs)]
    for k in first['criteria']:
        md.append(f'| {k} | ' + ' | '.join('pass' if a['criteria'][k] else '**FAIL**' for a in runs.values()) + ' |')
    md += ['', '## Cell-level spread per run\n',
           '| run | cells | max abs deviation | cells beyond 1 % | beyond 2 % | slower (fq) by > 2 % in every round | faster by > 2 % '
           'in every round | per-forward sums: max abs deviation |', '|---|---:|---:|---:|---:|---|---|---:|']
    for s, a in runs.items():
        n1 = sum(1 for c in a['cells'] if abs(c['ratio'] - 1) > 0.01)
        n2 = sum(1 for c in a['cells'] if abs(c['ratio'] - 1) > 0.02)
        fmt = lambda cs: ', '.join(f"{c['policy']} {c['proj']} T={c['tokens']} ({c['ratio']:.3f})" for c in cs) or '—'
        md.append(f"| {label(s, a)} | {len(a['cells'])} | {a['max_abs_cell_deviation'] * 100:.1f} % | {n1} | {n2} | "
                  f"{fmt(a['slower_every_round'])} | {fmt(a['faster_every_round'])} | {a['max_abs_sum_deviation'] * 100:.2f} % |")
    md.append('')
    cells = []
    for s, a in runs.items():
        cells += [f'## {label(s, a)}\n', '| policy | projection | T | build | width | scheduler | RaZeR (µs) | flipquant (µs) | '
                  'ratio | per round | quantizer RaZeR / flipquant (µs) |', '|---|---|---:|---|---|---|---:|---:|---:|---|---|']
        for c in a['cells']:
            cells.append(f"| {c['policy']} | {c['proj']} | {c['tokens']} | {c['kernel']} | {c['width']} | {c['schedule']} | "
                         f"{c['rz_us']:.2f} | {c['fq_us']:.2f} | {c['ratio']:.4f} | "
                         + ' / '.join(f'{x:.3f}' for x in c['round_ratios'])
                         + f" | {c['rz_quant_us']:.2f} / {c['fq_quant_us']:.2f} |")
        cells.append('')
    res = dict(sum_tolerance=SUM_TOL, cell_tolerance=CELL_TOL, runs=runs, pairs=pairs, controls=controls,
               passed={s: a['passed'] for s, a in runs.items()})
    (args.out_dir / 'parity.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    (args.out_dir / 'parity.md').write_text('\n'.join(md) + '\n')
    (args.out_dir / 'cells.md').write_text('\n'.join(cells) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
