#!/usr/bin/env python3
"""flipquant GEMM parity, the activation-placement diagnostic (results/kernel_opt/flipquant_parity/NOTE.md): within ONE
NVFP4-RaZeR process, the GEMM time of a cell against the device address of its activation-side buffers, everything else
fixed (the module, its rotation copies used in the same order, the kernel, the width, the scheduler row, the flush).

    python experiments/kernel_opt/flipquant_parity_actplace.py --out JSON

The parity driver's rz worker (flipquant_parity_gemm.py) builds the timed modules and their rotation copies. Per cell,
the activation-side buffers -- the BF16 input, the quantized activation (packed codes, scale bytes in the kernel layout,
per-token scales) and the BF16 output -- are carved, as one contiguous block, out of a 512 MiB arena at --offsets
seeded 4 KiB-aligned offsets, plus 'allocator' (the caching allocator's own choice, as in the parity run). Each
placement is timed with the parity run's repetition (flush, input copy, quantizer, GEMM; CUPTI time of the GEMM; the
same rotation copies from copy 0), in two passes over the placements in seeded orders.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flipquant_parity_gemm as D  # noqa: E402

CELLS = [('256x64', 'k_proj', 1), ('256x64', 'q_proj', 1), ('256x64', 'k_proj', 16), ('256x64', 'gate_proj', 1),
         ('16x64', 'k_proj', 1), ('stock_wB_ko', 'gate_proj', 16)]
ARENA = 512 * 2 ** 20


def align(v, a=256):
    return (v + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--offsets', type=int, default=24)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile
    pols = [p for p in D.POLICIES if p in {c[0] for c in CELLS}]
    projs = sorted({c[1] for c in CELLS})
    w = D.Worker(argparse.Namespace(side='rz', model=args.model, fq_root=None, fq_build=None, flush_mib=512,
                                    policies=pols, projections=projs))
    setup = w.setup({})
    from mixfp4_sm120.lib import sf_buffer_size
    arena = torch.empty(ARENA, dtype=torch.uint8, device='cuda')
    rng = random.Random(args.seed)
    stream = torch.cuda.current_stream()
    res = dict(gpu=setup.get('gpu'), arena_base=hex(arena.data_ptr()), method=vars(args) | {'out': str(args.out)}, cells={})
    for pol, proj, t in CELLS:
        key = (pol, proj)
        lin, n, k, kern, sched, width = w.call(key, t)
        pool, _ = w.pools(n, k, t)
        sizes = dict(x=t * k * 2, packed=t * k // 2, sf=sf_buffer_size(t, k), gs=4 * t, y=t * n * 2)
        need = sum(align(v) for v in sizes.values())
        offs = sorted({rng.randrange(0, (ARENA - need) // 4096) * 4096 for _ in range(args.offsets)})
        places = ['allocator'] + offs

        def buffers(place):
            if place == 'allocator':
                x = torch.empty((t, k), dtype=torch.bfloat16, device='cuda')
                return x, None, torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
            o, views = place, {}
            for name, size in sizes.items():
                views[name] = arena[o:o + size]
                o += align(size)
            x = views['x'].view(torch.bfloat16).view(t, k)
            y = views['y'].view(torch.bfloat16).view(t, n)
            return x, (views['packed'], views['sf'], views['gs'].view(torch.float32)), y

        def run(place):
            x, q, y = buffers(place)
            copies = w.copies[key]

            def rep(i):
                w.flush.sum()
                torch.cuda.synchronize()
                x.copy_(pool[i % 2])
                torch.cuda.synchronize()
                if q is None:
                    qq = kern.quant_rows(x, lin.act_kind)
                else:
                    rc = kern.lib.sm120_quant_rows(x.data_ptr(), x.stride(0), t, k, kern.QUANT_MODES[lin.act_kind],
                                                   q[0].data_ptr(), q[1].data_ptr(), q[2].data_ptr(), stream.cuda_stream)
                    assert rc == 0
                    qq = q
                torch.cuda.synchronize()
                wp, wsf = copies[i % len(copies)]
                w.gemm(lin, kern, sched, qq, wp, wsf, y, n, k, t)
                torch.cuda.synchronize()
            for i in range(args.warmup):
                rep(i)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(args.reps):
                    rep(args.warmup + i)
            g = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and D.classify(e.name) == 'gemm']
            assert len(g) == args.reps, len(g)
            return statistics.median(g), y.clone()

        per = {str(p): [] for p in places}
        ref = None
        for ps in range(args.passes):
            order = places[:]
            random.Random(args.seed + 1 + ps).shuffle(order)
            for p in order:
                us, y = run(p)
                per[str(p)].append(us)
                if ref is None:
                    ref = y
                assert torch.equal(ref.view(torch.int16), y.view(torch.int16)), (pol, proj, t, p)   # same output bits
        med = {p: statistics.median(v) for p, v in per.items()}
        vals = [med[str(p)] for p in offs]
        res['cells'][f'{pol}/{proj}/{t}'] = dict(kernel=kern.cfg.name, width=width, schedule=list(sched), sizes=sizes,
                                                 allocator=med['allocator'], allocator_passes=per['allocator'],
                                                 offsets={str(p): per[str(p)] for p in offs},
                                                 min=min(vals), max=max(vals), median=statistics.median(vals),
                                                 spread=(max(vals) - min(vals)) / statistics.median(vals),
                                                 pass_agreement=statistics.median(abs(a - b) / b for a, b in
                                                                                  (per[str(p)][:2] for p in offs)))
        c = res['cells'][f'{pol}/{proj}/{t}']
        print(f"{pol}/{proj}/{t}: allocator {c['allocator']:.2f}; carved min {c['min']:.2f} median {c['median']:.2f} "
              f"max {c['max']:.2f} (spread {c['spread'] * 100:.1f} %, median pass-to-pass change "
              f"{c['pass_agreement'] * 100:.2f} %)", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
