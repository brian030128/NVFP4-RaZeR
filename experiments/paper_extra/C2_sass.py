#!/usr/bin/env python3
"""C2-lite static SASS census: OMMA instructions per path through the GEMM main loop (results/paper_extra/C2).

    PAPER_PYTHON experiments/paper_extra/C2_sass.py --configs stock_wA,n16k64_wA,n16k64_wA_nodisp --out JSON

CAVEAT: these are static counts from the disassembly, not measured counters (ncu is unavailable on this machine).

For each kernel build (cuobjdump --dump-sass of its library), the GEMM function (cutlass::device_kernel) is split into
basic blocks (leaders: branch targets and the instructions after a branch or EXIT). Loops are the back edges
(a BRA to a lower address). For the loop whose body holds the OMMAs, the minimum and the maximum OMMA count over every
path from the loop head to the back edge are computed on the body's acyclic graph (inner back edges removed; reported).
Also recorded: the opcode mix of the function, OMMAs by format, predicated OMMAs, WARPSYNC, BRX / BRA counts.
If every path through one iteration issues the same number of OMMAs, the tensor-pipe instruction count of a GEMM is
(OMMAs per iteration) x (iterations) x (warps) x (CTAs), whatever the tags: the census estimate reported by C2.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
import paper_common as P  # noqa: E402

INSN = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T\d]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);')


def gemm_function(sass):
    parts = re.split(r'\n\s*Function : ', sass)
    for p in parts:
        if p.startswith('_ZN7cutlass13device_kernel'):
            return p
    raise SystemExit('no cutlass::device_kernel in the disassembly')


def parse(text):
    out = []
    for m in INSN.finditer(text):
        addr, pred, op, args = int(m.group(1), 16), (m.group(2) or '').strip(), m.group(3), m.group(4)
        target = None
        if op.split('.')[0] in ('BRA', 'BRX', 'JMP', 'CALL'):
            t = re.search(r'0x([0-9a-f]+)', args)
            target = int(t.group(1), 16) if t else None
        out.append(dict(addr=addr, pred=pred, op=op, base=op.split('.')[0], args=args.strip(), target=target))
    return out


def blocks(insns):
    addrs = [i['addr'] for i in insns]
    index = {a: k for k, a in enumerate(addrs)}
    leaders = {addrs[0]}
    for k, i in enumerate(insns):
        if i['base'] in ('BRA', 'BRX', 'JMP', 'EXIT', 'RET'):
            if i['target'] is not None and i['target'] in index:
                leaders.add(i['target'])
            if k + 1 < len(insns):
                leaders.add(addrs[k + 1])
    starts = sorted(leaders)
    bl = {}
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else addrs[-1] + 16
        body = [i for i in insns[index[s]:] if i['addr'] < e]
        last = body[-1]
        succ = []
        uncond = not last['pred'] or last['pred'] in ('@PT', '@UPT')
        conditional_uniform = '.U' in last['op'] and ('!UP' in last['args'] or 'UP' in last['args'])
        if last['base'] in ('BRA', 'JMP'):
            if last['target'] is not None:
                succ.append(last['target'])
            if not uncond or conditional_uniform:
                succ.append(e)
        elif last['base'] in ('EXIT', 'RET'):
            if not uncond:
                succ.append(e)
        elif last['base'] == 'BRX':
            succ.append(('BRX', last['addr']))
        else:
            succ.append(e)
        bl[s] = dict(start=s, end=e, omma=sum(1 for i in body if i['base'] == 'OMMA'), succ=[x for x in succ if x in index or isinstance(x, tuple)])
    return bl


def loop_paths(bl):
    """For each back edge (u -> h, h <= u) whose body holds OMMAs: min / max OMMAs over head->u paths in the body."""
    res = []
    order = sorted(bl)
    for u in order:
        for h in bl[u]['succ']:
            if isinstance(h, tuple) or h > u:
                continue
            body = [b for b in order if h <= b <= u]
            omma = sum(bl[b]['omma'] for b in body)
            if not omma:
                continue
            # DP over the body in address order (forward edges only; inner back edges are dropped and counted)
            lo, hi = {h: bl[h]['omma']}, {h: bl[h]['omma']}
            inner_back = 0
            for b in body:
                if b not in lo:
                    continue
                for s in bl[b]['succ']:
                    if isinstance(s, tuple):
                        continue
                    if s <= b:
                        if not (b == u and s == h):
                            inner_back += 1
                        continue
                    if s > u:
                        continue               # leaves the loop
                    lo[s] = min(lo.get(s, 1 << 30), lo[b] + bl[s]['omma'])
                    hi[s] = max(hi.get(s, -1), hi[b] + bl[s]['omma'])
            res.append(dict(head=hex(h), latch=hex(u), blocks=len(body), omma_in_body=omma,
                            omma_per_iteration_min=lo.get(u), omma_per_iteration_max=hi.get(u),
                            inner_back_edges=inner_back))
    return res


def census(cfg, cuobjdump):
    lib = next((REPO / 'sm120' / 'build' / cfg).glob('libmixfp4_sm120_*.so'))
    sass = subprocess.run([cuobjdump, '--dump-sass', str(lib)], capture_output=True, text=True, check=True).stdout
    text = gemm_function(sass)
    insns = parse(text)
    ops = Counter(i['base'] for i in insns)
    omma = [i for i in insns if i['base'] == 'OMMA']
    formats = Counter(re.search(r'OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)', i['op'] + i['args']).groups() if
                      re.search(r'OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)', i['op'] + i['args']) else ('?', '?') for i in omma)
    bl = blocks(insns)
    loops = loop_paths(bl)
    return dict(config=cfg, library=str(lib), instructions=len(insns), omma=len(omma),
                omma_formats={f'{a}x{b}': n for (a, b), n in formats.items()},
                omma_predicated=sum(1 for i in omma if i['pred']), warpsync=ops.get('WARPSYNC', 0), brx=ops.get('BRX', 0),
                bra=ops.get('BRA', 0), blocks=len(bl), omma_loops=loops, opcodes=dict(ops.most_common(20)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', default='stock_wA,n16k64_wA,n16k64_wA_nodisp,n8k64_wB,stock_wB')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cuobjdump = Path(P.CUDA_HOME) / 'bin' / 'cuobjdump'
    res = dict(caveat='static SASS census, not measured counters (ncu unavailable on this machine)', kernels={})
    for cfg in args.configs.split(','):
        try:
            res['kernels'][cfg] = census(cfg, str(cuobjdump))
        except (StopIteration, subprocess.CalledProcessError) as e:
            res['kernels'][cfg] = dict(error=f'{type(e).__name__}: {e}')
        k = res['kernels'][cfg]
        print(cfg, json.dumps({x: k.get(x) for x in ('instructions', 'omma', 'omma_formats', 'omma_predicated', 'warpsync',
                                                      'brx', 'omma_loops', 'error')}), flush=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
