#!/usr/bin/env python3
"""flipquant GEMM parity, the code-placement diagnostic (results/kernel_opt/flipquant_parity/NOTE.md): within ONE
process, does the GEMM time of a cell change when the same build is loaded again as another library instance?

    python experiments/kernel_opt/flipquant_parity_codeplace.py --copies 4 --out JSON

The parity driver's rz worker (flipquant_parity_gemm.py) builds the timed modules and their rotation copies once. The
builds of each set are then copied, byte for byte, into --copies scratch build directories, and each copy is loaded as
its own library instance (another path, so another dlopen and another CUDA module: the same SASS, placed elsewhere in
device memory). Every instance times the same modules on the same rotation copies, input pool and flush buffer; only
the instance of the code differs. Rounds rotate over the instances (build_V's own included as instance 0).
"""
import argparse
import copy
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flipquant_parity_gemm as D  # noqa: E402

CELLS = [('256x64', 'q_proj', 1), ('256x64', 'k_proj', 1), ('256x64', 'v_proj', 1), ('256x64', 'gate_proj', 1),
         ('256x64', 'k_proj', 16), ('256x64', 'v_proj', 16), ('stock_wB_ko', 'gate_proj', 16),
         ('16x64', 'q_proj', 1), ('16x64', 'k_proj', 1), ('16x64', 'k_proj', 16)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--copies', type=int, default=4)
    ap.add_argument('--scratch', type=Path, required=True, help='where the copied build directories go')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import os
    build_v = Path(os.environ['SM120_BUILD_DIR'])
    pols = [p for p in D.POLICIES if p in {c[0] for c in CELLS}]
    projs = sorted({c[1] for c in CELLS})
    wargs = argparse.Namespace(side='rz', model=args.model, fq_root=None, fq_build=None, flush_mib=512, policies=pols,
                               projections=projs)
    w = D.Worker(wargs)
    setup = w.setup({})
    sys.path.insert(0, str(D.RAZER / 'sm120' / 'bench'))
    from mixfp4_sm120.select import KernelSet
    # the instances: build_V's own, then byte-for-byte copies of the same builds in scratch roots
    families = {pol: w.lins[(pol, projs[0])].kernel_set.family for pol in pols}
    builds = sorted({Path(k.path).parent.name for pol in pols for k in w.lins[(pol, projs[0])].kernel_set.kernels.values()})
    roots = [build_v]
    for i in range(1, args.copies + 1):
        root = args.scratch / f'root{i}'
        for b in builds:
            if not (root / b).exists():
                shutil.copytree(build_v / b, root / b)
        roots.append(root)
    inst = {}
    for i, root in enumerate(roots):
        for pol in pols:
            ks = KernelSet(families[pol], build_root=root) if i else w.lins[(pol, projs[0])].kernel_set
            assert ks.sha256 == w.lins[(pol, projs[0])].kernel_set.sha256        # the same bytes
            for proj in projs:
                if i == 0:
                    inst[(i, pol, proj)] = (pol, proj)
                    continue
                key = (f'{pol}@{i}', proj)
                lin = copy.copy(w.lins[(pol, proj)])
                lin.kernel, lin.kernel_set = ks, ks
                w.lins[key] = lin
                w.copies[key] = w.copies[(pol, proj)]                            # the same weight buffers
                w.counter[key] = 0
                inst[(i, pol, proj)] = key
    res = dict(roots=[str(r) for r in roots], builds=builds, cells=[list(c) for c in CELLS], gpu=setup.get('gpu'),
               method=dict(reps=args.reps, warmup=args.warmup, rounds=args.rounds), rows=[])
    for pol, proj, t in CELLS:
        vals = {i: [] for i in range(len(roots))}
        per_round = {i: [] for i in range(len(roots))}
        libs = {}
        for r in range(args.rounds):
            order = list(range(len(roots)))
            shift = r * len(order) // args.rounds
            for i in order[shift:] + order[:shift]:
                key = inst[(i, pol, proj)]
                got = w.time(dict(policy=key[0], proj=key[1], t=t, reps=args.reps, warmup=args.warmup))
                vals[i] += got['gemm_us']
                per_round[i].append(statistics.median(got['gemm_us']))
                libs[i] = str(w.lins[key].kernel_set.pick(w.lins[key].out_features, w.lins[key].in_features, t).path)
        med = {i: statistics.median(v) for i, v in vals.items()}
        row = dict(cell=f'{pol}/{proj}/{t}', medians=med, rounds=per_round, libraries=libs,
                   spread=(max(med.values()) - min(med.values())) / statistics.median(list(med.values())))
        res['rows'].append(row)
        print(f"{row['cell']:28s} " + ' '.join(f'{med[i]:7.2f}' for i in range(len(roots)))
              + f"   spread {row['spread'] * 100:.1f} %", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
