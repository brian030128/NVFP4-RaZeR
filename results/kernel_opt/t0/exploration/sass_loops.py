"""Parse cuobjdump -sass output: take the GEMM kernel (device_kernel), list its instructions, find loops (backward
branches), and print per-loop opcode counts. Usage: python sass_loops.py FILE.sass [--dump-loop IDX]"""
import collections
import re
import sys

INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')


def load(path):
    text = open(path).read()
    funcs = re.split(r'\n\s*Function : ', text)
    for f in funcs:
        if f.startswith('_ZN7cutlass13device_kernel'):
            name = f.split('\n', 1)[0]
            ins = []
            for m in INS.finditer(f):
                addr = int(m.group(1), 16)
                body = m.group(2)
                ins.append((addr, body))
            return name, ins
    raise SystemExit('no device_kernel')


def opcode(body):
    b = body
    pred = ''
    if b.startswith('@'):
        pred, b = b.split(None, 1)
    op = b.split(None, 1)[0]
    return pred, op


def loops(ins):
    out = []
    for addr, body in ins:
        pred, op = opcode(body)
        if op.startswith('BRA') or op == 'BRX':
            m = re.search(r'0x([0-9a-f]+)', body.split(op, 1)[1])
            if m:
                tgt = int(m.group(1), 16)
                if tgt <= addr:
                    out.append((tgt, addr, pred, body))
    return out


def counts(ins, lo, hi):
    c = collections.Counter()
    for addr, body in ins:
        if lo <= addr <= hi:
            pred, op = opcode(body)
            c[op.split('.')[0]] += 1
    return c


if __name__ == '__main__':
    path = sys.argv[1]
    name, ins = load(path)
    print(name[:120], len(ins), 'instructions')
    ls = loops(ins)
    for i, (lo, hi, pred, body) in enumerate(ls):
        c = counts(ins, lo, hi)
        n = sum(c.values())
        print(f'loop {i}: [{lo:#06x}, {hi:#06x}] {n} ins, back-branch {pred} {body[:60]}')
        print('   ', ', '.join(f'{k}:{v}' for k, v in c.most_common(30)))
    if '--dump-loop' in sys.argv:
        i = int(sys.argv[sys.argv.index('--dump-loop') + 1])
        lo, hi = ls[i][0], ls[i][1]
        for addr, body in ins:
            if lo <= addr <= hi:
                print(f'{addr:#06x}  {body}')
