#!/usr/bin/env python3
"""flipquant GEMM parity: the diagnostics' summary (results/kernel_opt/flipquant_parity/NOTE.md). CPU only.

    python experiments/kernel_opt/flipquant_parity_diagnostics.py --src /home/dev/n16k64_campaign/kernel_opt/fq_parity \
        --out-dir results/kernel_opt/flipquant_parity

Reads the reduced driver runs of the bisection (<src>/bisect/*.json, flipquant_parity_gemm.py with the options in
BISECT) and the in-process diagnostics (flipquant_parity_placement.py, _codeplace.py, _actplace.py, _alloc.py), and
writes diagnostics.md and diagnostics.json, plus gzipped copies of every record under diagnostics/.
"""
import argparse
import gzip
import json
import shutil
import statistics
from pathlib import Path

CELL_TOL = 0.02
# run -> (what was set up and timed, the per-cell check made before each (projection, T))
BISECT = {
    'D_all_t1_16': ('every policy and projection, T in {1, 16}', 'full check'),
    'E_allpol_k': ('every policy, k_proj, T in {1, 16}', 'full check'),
    'A_256_k': ('256x64 only, k_proj, T in {1, 16}', 'full check'),
    'F_256_allproj': ('256x64 only, every projection, T in {1, 16}', 'full check'),
    'E1_rehome': ('as E_allpol_k, every timed buffer moved to fresh segments first (--rehome)', 'full check'),
    'E2_nochecks': ('as E_allpol_k', 'none (--no-checks)'),
    'E3_noprof': ('as E_allpol_k', 'fused forward + isolated GEMM, no profiler'),
    'E3_forward': ('as E_allpol_k', 'fused forward under the profiler only'),
    'E3_isolated': ('as E_allpol_k', 'isolated GEMM under the profiler only'),
    'E4_P_I': ('as E_allpol_k', "profiled forward + isolated GEMM (the check's first two parts)"),
    'E4_P_I_H': ('as E_allpol_k', '... + the output hash (D2H copy)'),
    'E4_P_I_E': ('as E_allpol_k', '... + torch.equal of the two outputs'),
    'E4_P_I_E_H': ('as E_allpol_k', '... + torch.equal + the output hash (= the full check)'),
    'E5_full_addr': ('as E_allpol_k, buffer addresses recorded', 'full check'),
    'E5_noeq_addr': ('as E_allpol_k, buffer addresses recorded', '... + the output hash, no torch.equal'),
}


def cell_table(g):
    rows = {(r['policy'], r['proj'], r['tokens'], r['side']): r for r in g['rows']}
    out = []
    for (pol, proj, t, side), rf in rows.items():
        if side != 'fq':
            continue
        rr = rows[(pol, proj, t, 'rz')]
        rounds = [a['gemm_us'] / b['gemm_us'] for a, b in zip(sorted(rf['rounds'], key=lambda x: x['round']),
                                                              sorted(rr['rounds'], key=lambda x: x['round']))]
        out.append(dict(policy=pol, proj=proj, tokens=t, fq=rf['gemm_us'], rz=rr['gemm_us'],
                        ratio=rf['gemm_us'] / rr['gemm_us'], rounds=rounds,
                        every_round=('slower' if all(x > 1 + CELL_TOL for x in rounds) else
                                     'faster' if all(x < 1 - CELL_TOL for x in rounds) else None)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/fq_parity'))
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'results' / 'kernel_opt' /
                    'flipquant_parity')
    args = ap.parse_args()
    dst = args.out_dir / 'diagnostics'
    dst.mkdir(parents=True, exist_ok=True)
    res, md = dict(bisect={}), ['# flipquant GEMM parity: diagnostics\n']
    md += ['## Bisection with the parity driver (flipquant worker against the RaZeR harness)\n',
           'Values: the median GEMM time (µs) of 256x64 k_proj, flipquant / RaZeR, and the cells that differ by more '
           'than 2 % in every round.\n',
           '| run | set-up and timed | per-cell check | T=1 | T=16 | cells beyond 2 % in every round |',
           '|---|---|---|---|---|---|']
    for name, (setup, check) in BISECT.items():
        g = json.loads((args.src / 'bisect' / f'{name}.json').read_text())
        cells = cell_table(g)
        k = {c['tokens']: c for c in cells if c['policy'] == '256x64' and c['proj'] == 'k_proj'}
        off = [c for c in cells if c['every_round']]
        res['bisect'][name] = dict(setup=setup, check=check, k_proj=k, offsets=off,
                                   max_abs_deviation=max(abs(c['ratio'] - 1) for c in cells))
        fmt = lambda c: f"{c['fq']:.2f} / {c['rz']:.2f} ({c['ratio']:.3f})"
        md.append(f"| {name} | {setup} | {check} | {fmt(k[1])} | {fmt(k[16])} | "
                  + (', '.join(f"{c['policy']} {c['proj']} T={c['tokens']} ({c['ratio']:.3f})" for c in off) or '—') + ' |')
        with gzip.open(dst / f'bisect_{name}.json.gz', 'wt') as f:
            f.write(json.dumps(g))
    # the timed buffers' addresses with and without the trigger (E5)
    md += ['', '## The timed buffers with and without the trigger (E5, flipquant worker, 256x64 k_proj)\n',
           'Offsets of the first timed launch\'s buffers within their 16 MiB-aligned region (the high bits are the '
           "process's own address-space base).\n", '| run | T | GEMM (µs) | x | packed | sf | gs | y |',
           '|---|---:|---:|---|---|---|---|---|']
    addr = {}
    for name in ('E5_full_addr', 'E5_noeq_addr'):
        g = json.loads((args.src / 'bisect' / f'{name}.json').read_text())
        for r in g['rows']:
            if r['policy'] == '256x64' and r['side'] == 'fq':
                b = r['rounds'][0]
                low = {k: hex(int(v, 16) % (1 << 24)) for k, v in b['addresses'].items()}
                addr[f"{name}/T={r['tokens']}"] = dict(gemm_us=r['gemm_us'], low=low)
                md.append(f"| {name} | {r['tokens']} | {r['gemm_us']:.2f} | " + ' | '.join(low[k] for k in
                                                                                       ('x', 'packed', 'sf', 'gs', 'y')) + ' |')
    res['addresses'] = addr
    # in-process diagnostics on one RaZeR process
    md += ['', '## In one NVFP4-RaZeR process: what moves these cells\n',
           '| diagnostic | what changes between measurements | cell | min–max (µs) | spread |', '|---|---|---|---|---:|']
    pl = json.loads((args.src / 'placement_rz.json').read_text())
    cp = json.loads((args.src / 'codeplace_llama8b.json').read_text())
    ac = json.loads((args.src / 'actplace_llama8b.json').read_text())
    res['in_process'] = {}
    for key in ('256x64/k_proj/1', '256x64/k_proj/16', '256x64/q_proj/1', 'stock_wB_ko/gate_proj/16'):
        s = pl['summary'][key]
        md.append(f"| placement (9 trials) | the rotation copies cloned again after a pad, or the activation buffers "
                  f"reallocated after a pad | {key} | {s['min']:.2f}–{s['max']:.2f} | {s['spread'] * 100:.1f} % |")
        res['in_process'][f'placement/{key}'] = s
    for row in cp['rows']:
        if row['cell'] in ('256x64/k_proj/1', '256x64/k_proj/16', '256x64/q_proj/1', 'stock_wB_ko/gate_proj/16'):
            v = list(row['medians'].values())
            md.append(f"| code instances ({len(v)}) | the same builds loaded again from byte-identical copies | {row['cell']} | "
                      f"{min(v):.2f}–{max(v):.2f} | {row['spread'] * 100:.1f} % |")
            res['in_process'][f'codeplace/{row["cell"]}'] = dict(min=min(v), max=max(v), spread=row['spread'])
    for key, c in ac['cells'].items():
        md.append(f"| activation placement ({len(c['offsets'])} offsets) | the input, quantized activation, scales and "
                  f"output carved at seeded 4 KiB-aligned offsets of a 512 MiB arena | {key} | {c['min']:.2f}–{c['max']:.2f} "
                  f"| {c['spread'] * 100:.1f} % |")
        res['in_process'][f'actplace/{key}'] = dict(min=c['min'], max=c['max'], spread=c['spread'], allocator=c['allocator'])
    # the paired allocation diagnostics
    md += ['', '## Paired allocation diagnostics (flipquant worker against the RaZeR harness, no per-cell checks)\n',
           '| record | phase | segments holding the copies: fq / rz | 256x64 k_proj T=1 | T=16 | largest |dev| |',
           '|---|---|---|---|---|---:|']
    res['alloc'] = {}
    for fname in ('alloc_llama8b.json', 'alloc_full_localize.json'):
        a = json.loads((args.src / fname).read_text())
        for ph in a['phases']:
            c = ph['cells']
            seg = ph['segments']
            dev = max(abs(v['ratio'] - 1) for v in c.values())
            k1, k16 = c['256x64/k_proj/1'], c['256x64/k_proj/16']
            md.append(f"| {fname} | {ph['phase']} | {seg['fq']['copies']['segments']} (largest "
                      f"{seg['fq']['copies']['largest_gib']:.2f} GiB) / {seg['rz']['copies']['segments']} (largest "
                      f"{seg['rz']['copies']['largest_gib']:.2f} GiB) | {k1['fq']:.2f} / {k1['rz']:.2f} | "
                      f"{k16['fq']:.2f} / {k16['rz']:.2f} | {dev * 100:.1f} % |")
            res['alloc'][f"{fname}/{ph['phase']}"] = dict(max_abs_deviation=dev, k1=k1, k16=k16)
    for fname in ('placement_rz.json', 'codeplace_llama8b.json', 'actplace_llama8b.json', 'alloc_llama8b.json',
                  'alloc_full_localize.json'):
        with open(args.src / fname, 'rb') as fi, gzip.open(dst / f'{fname}.gz', 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    (args.out_dir / 'diagnostics.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    (args.out_dir / 'diagnostics.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
