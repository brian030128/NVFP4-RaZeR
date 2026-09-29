#!/usr/bin/env python3
"""C2-lite report: results/paper_extra/C2/C2.md from C2_time.py and C2_sass.py. CPU, seconds.

    PAPER_PYTHON experiments/paper_extra/C2_report.py --time C2_time.json --sass C2_sass.json [--dest ...]

Every table carries the caveat (the user's condition for C2-lite).
Tensor-pipe estimate at M = N = K = 4096 for the 128 x 128 CTA tile with 8 warps (m16n8k64 atoms, 16 per warp per
64-K block: 2 x 8 in the 4 x 2 arrangement of the stock and wA builds, 8 x 2 in n8k64_wB's 1 x 8): CTAs x warps x
(K / K per iteration) x OMMAs per iteration, where the SASS census gives the OMMAs per steady-state iteration on every
path (32 = two 64-K blocks).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402
from C2_time import CAVEAT  # noqa: E402

# sm120/kernel/docs/mixed_nvfp4_report.md (RTX 5090, ncu), cited, not re-measured
HISTORICAL = [
    ('stock NVFP4', '1207', '8,388,608', 'math_pipe_throttle 4.42', '§1 table'),
    ('mixed, branch per MMA (C++ if/else, 4 asm blocks)', '504', '16,777,216 (2x: 128 @P0 + 128 @!P0 OMMAs)',
     'wait 3.96, branch_resolving 1.04', '§1 tables'),
    ('mixed, flat 4-way compound predicates', '357', '(four predicated OMMAs per useful one)', '—', '§1 table'),
    ('mixed, brx.idx per MMA', '296', '(1x: if-conversion defeated)', '—', 'scripts/gen_mixed_mma_ptx.py'),
    ('mixed, one brx.idx per k_block', '865', '—', '—', 'scripts/gen_mixed_mma_ptx.py'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--time', type=Path, required=True)
    ap.add_argument('--sass', type=Path, required=True)
    ap.add_argument('--dest', type=Path, default=P.REPO / 'results' / 'paper_extra' / 'C2')
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    tm, sa = json.loads(args.time.read_text()), json.loads(args.sass.read_text())
    cav = f'> **Caveat.** {CAVEAT}\n'
    size = tm['size']
    stock = next(r for r in tm['rows'] if r['kernel'] == 'stock_wA')
    md = ['# C2-lite: the mixed kernel at M = N = K = 4096 on the RTX PRO 6000\n', cav,
          f"## Kernel time (CUPTI, median of {tm['rounds']} rotated rounds of 20 calls; 128-wide CTA tile)\n", cav,
          '| kernel | build | µs | TFLOP/s | vs stock_wA | per-round µs |', '|---|---|---:|---:|---:|---|']
    for r in tm['rows']:
        md.append(f"| {r['kernel']} | {r['build']} | {r['time_us']:.2f} | {r['tflops']:.1f} | "
                  f"{100 * (r['time_us'] / stock['time_us'] - 1):+.1f} % | {', '.join(f'{v:.2f}' for v in r['rounds_us'])} |")
    md.append(f"\n`real` = the tags of {tm['layer']} in the committed FlipQuant (ours) Llama-3.1-8B 16x64 map (TM-OPT+TC); E0M3 share "
              f"{100 * tm['layer_e0m3_share']:.2f} % of its tiles.\n")
    ctas, warps = (size // 128) * (size // 128), 8
    md += ['## Static SASS census of the GEMM function\n', cav,
           '| build | instructions | OMMA (static) | formats | predicated OMMA | WARPSYNC | BRX | steady k-loop: OMMA per iteration, min / max over paths | est. tensor-pipe instructions at 4096³ |',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for cfg, k in sa['kernels'].items():
        if 'error' in k:
            md.append(f"| {cfg} | not built: {k['error'][:80]} | | | | | | | |")
            continue
        steady = [lp for lp in k['omma_loops'] if lp['inner_back_edges'] == 0 and lp['omma_per_iteration_min']]
        steady = min(steady, key=lambda lp: lp['blocks']) if steady else None
        per = (steady['omma_per_iteration_min'], steady['omma_per_iteration_max']) if steady else (None, None)
        est = (ctas * warps * (size // 128) * per[0]) if per[0] and per[0] == per[1] else None
        md.append(f"| {cfg} | {k['instructions']} | {k['omma']} | {', '.join(f'{a} {b}' for a, b in k['omma_formats'].items())} | "
                  f"{k['omma_predicated']} | {k['warpsync']} | {k['brx']} | {per[0]} / {per[1]} | "
                  f"{'—' if est is None else f'{est:,}'} |")
    md += ['\nThe estimate assumes 8 warps per 128 x 128 CTA and 128 K per steady-state iteration (32 OMMAs = two 64-K '
           'blocks of 16 m16n8k64 atoms per warp: 2 x 8 in the 4 x 2 warp arrangement of the stock and wA builds, 8 x 2 in '
           'n8k64_wB\'s 1 x 8); it equals the historical ncu count of the stock '
           'kernel (8,388,608, RTX 5090). Every path through the mixed kernels\' steady-state iteration issues the same 32 '
           'OMMAs as stock, so the E0M3 tags do not change the tensor-pipe instruction count.\n',
           '## Historical: the per-MMA-branch kernel and its variants (cited, not re-measured)\n', cav,
           'From sm120/kernel/docs/mixed_nvfp4_report.md, RTX 5090, 4096³, ncu. The per-MMA-branch kernel cannot be '
           'built from the current sources: the generator emits only the per-k_block dispatch.\n',
           '| kernel | TFLOP/s | sm__inst_executed_pipe_tensor | dominant warp stall | source |', '|---|---:|---|---|---|']
    md += [f'| {a} | {b} | {c} | {d} | {e} |' for a, b, c, d, e in HISTORICAL]
    (args.dest / 'C2.md').write_text('\n'.join(md) + '\n')
    (args.dest / 'C2.json').write_text(json.dumps(dict(caveat=CAVEAT, time=tm, sass=sa), indent=1) + '\n')
    print('wrote', args.dest / 'C2.md')


if __name__ == '__main__':
    main()
