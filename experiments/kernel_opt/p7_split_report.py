#!/usr/bin/env python3
"""Kernel-opt amendment 15 (the 8x64 plan's P7) report on the in-graph splits (p7_split.py): per model, shape and policy,
the captures' wall, GEMM, quantizer, other and idle times and the SM clock, against fo6-ko.

    python experiments/kernel_opt/p7_split_report.py --src JSON [--src JSON ...] --out-dir results/kernel_opt/w8/p7

For each policy the medians are over its captures; the per-capture idle times are listed (D4's bimodality). Changes are
the medians' differences against fo6-ko, in µs and in % of fo6-ko's wall (GEMM also in % of fo6-ko's GEMM). SM clock
and power are the medians of each capture's NVML samples.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REF = 'fo6-ko'
PARTS = ('wall_us', 'gemm_us', 'quant_us', 'other_us', 'idle_us')


def med(v):
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, action='append', required=True)
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8' / 'p7')
    args = ap.parse_args()
    md, data = ['## The in-graph split (D4\'s method)\n'], {}
    for src in args.src:
        rec = json.loads(src.read_text())
        assert rec.get('status') == 'complete', src
        model = rec['model']
        for shape, pols in rec['shapes'].items():
            md.append(f"### {model} {shape} ({len(next(iter(pols.values())))} captures per policy × {rec['reps']} replays)\n")
            md.append('| policy | wall µs | GEMM µs | quantizer µs | other µs | idle µs | SM MHz | power W | vs fo6-ko: wall | '
                      'GEMM | idle µs per capture |')
            md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
            m = {p: {k: med([c[k] for c in caps]) for k in PARTS} for p, caps in pols.items()}
            clk = {p: med([c['telemetry']['sm_mhz_median'] for c in caps]) for p, caps in pols.items()}
            pw = {p: med([c['telemetry']['power_w_median'] for c in caps]) for p, caps in pols.items()}
            rows = {}
            for p, caps in pols.items():
                r = m[p]
                dw = 100 * (r['wall_us'] / m[REF]['wall_us'] - 1)
                dg = 100 * (r['gemm_us'] / m[REF]['gemm_us'] - 1)
                idle = [round(c['idle_us']) for c in caps]
                rows[p] = dict(**r, sm_mhz=clk[p], power_w=pw[p], wall_vs_ref_pct=dw, gemm_vs_ref_pct=dg, idle_per_capture=idle,
                               captures=len(caps))
                md.append(f"| {p} | {r['wall_us']:.0f} | {r['gemm_us']:.0f} | {r['quant_us']:.0f} | {r['other_us']:.0f} | "
                          f"{r['idle_us']:.0f} | {clk[p]:.0f} | {pw[p]:.0f} | {dw:+.2f} % | {dg:+.2f} % | {idle} |")
            md.append('')
            data.setdefault(model, {})[shape] = rows
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'p7_split.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'p7_split_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
