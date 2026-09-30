#!/usr/bin/env python3
"""Kernel-opt correctness gate: candidate weights-on-B builds against the current n8k64_wB, bitwise.

    python experiments/kernel_opt/check_bitwise.py --build-root /home/dev/n16k64_campaign/kernel_opt/build \
        --candidates n8k64_wB,n8k64_wB_m64,n8k64_wB_m32,n8k64_wB_m16 --set mixed_wB --out JSON

Protocol: results/kernel_opt/PROTOCOL.md (gate G4). A failure is reported and stops the optimization's adoption.

- Reference: n8k64_wB loaded from sm120/build, the build behind every paper result (library sha256 recorded).
- Candidates: the builds of --build-root named by --candidates, each alone, and the width-selecting KernelSet --set
  loaded from --build-root (with the tile table in sm120/configs, if it has rows for that family).
- Weights: per model (llama8b, mistral7b, phi4, qwen27b) and per distinct projection shape of its TC 8x64 artifact:
  - real: the densest module (the first maximum of E0M3 tiles) and the lower-median module (ties: layer order), with
    their own tags, global scale and bias; each is read tensor by tensor and checked against the artifact's sha256s
    and tile counts;
  - synthetic, once per distinct shape: a seeded N(0, 0.02) weight packed with an 8x64 map that selects no tile
    (all E2M1), every tile (all E0M3), or a seeded random 30 % of tiles; the random map also with a seeded bias.
- Tokens: TOKENS. Inputs are seeded BF16 N(0, 1) rows with 8 seeded outlier channels scaled by 30.
- Paths, all through NativeLinear with activation quantizer four_over_six_rows:
  - fused: sm120_linear (the library's fused quantizer and GEMM), with share_input off;
  - reuse: a second call on the same input tensor with share_input on, i.e. the GEMM on the cached quantized input;
  - unfused: quant_act.quantize followed by Kernel.gemm (fused off);
  - graph: the fused forward captured in a CUDA graph and replayed; real modules at GRAPH_TOKENS only.
- Rule: every candidate output equals the reference's output on the same path bitwise (int16 view).
- Recorded, not part of the rule: whether the reference's own paths agree with each other.
"""
import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear, clear_quant_cache  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
ARTIFACTS = Path('/home/dev/n16k64_campaign/paper/artifacts')
TOKENS = (1, 7, 16, 17, 33, 64, 65, 100, 128, 200, 256, 512, 1000, 2048, 8192)
GRAPH_TOKENS = (1, 16, 64)
ACT = 'four_over_six_rows'
PATHS = ('fused', 'reuse', 'unfused')


def read_module(art, meta, name):
    """PackedWeight of one module of a TC artifact, checked against its manifest entry."""
    m = next(e for e in meta['modules'] if e['name'] == name)
    tb = tuple(meta['type_block'])
    with safe_open(str(art / 'weights.safetensors'), 'pt') as f:
        keys = set(f.keys())
        packed, scales = f.get_tensor(f'{name}.packed'), f.get_tensor(f'{name}.scales')
        gs = float(f.get_tensor(f'{name}.global_scale')[0])
        bias = f.get_tensor(f'{name}.bias') if f'{name}.bias' in keys else None
    assert A.tensor_sha256(packed) == m['packed_sha256'] and A.tensor_sha256(scales) == m['scales_sha256'], name
    assert gs == m['global_scale'] and int(A.tile_flags(scales, tb).sum()) == m['e0m3_tiles'], name
    return A.PackedWeight(name, packed.cuda(), scales.cuda(), gs, None if bias is None else bias.cuda(), tb,
                          m['e0m3_tiles'])


def real_modules(model):
    """[(label, PackedWeight)]: per distinct shape, the densest and the lower-median module of the TC 8x64 artifact."""
    art = ARTIFACTS / f'{model}_tc_8x64'
    meta = json.loads((art / 'artifact.json').read_text())
    assert tuple(meta['type_block']) == (8, 64), meta['type_block']
    by = collections.OrderedDict()
    for i, m in enumerate(meta['modules']):
        by.setdefault(tuple(m['shape']), []).append((m['e0m3_tiles'], i, m['name']))
    out = []
    for shape, mods in by.items():
        worst = max(mods, key=lambda v: (v[0], -v[1]))
        typical = sorted(mods)[(len(mods) - 1) // 2]
        for label, (tiles, _, name) in (('densest', worst), ('median', typical)):
            out.append((f'{model} {name} ({label}, {tiles} E0M3 tiles)', read_module(art, meta, name)))
    return out, meta['weights_sha256']


def synthetic(shape):
    n, k = shape
    g = torch.Generator('cpu').manual_seed(n * 7 + k)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    bias = (torch.randn(n, generator=g) * 0.1).cuda().bfloat16()
    nb = -(-n // 8)
    rnd = torch.rand(nb, k // 64, generator=g) < 0.3
    out = []
    for tag, mask, b in (('all E2M1', torch.zeros(nb, k // 64, dtype=torch.bool), None),
                         ('all E0M3', torch.ones(nb, k // 64, dtype=torch.bool), None),
                         ('random 30 %', rnd, None), ('random 30 % + bias', rnd, bias)):
        out.append((f'synthetic {n}x{k} {tag}', A.pack_module('synthetic', w, b, 'map', mask, (8, 64))))
    return out


def inputs(t, k):
    g = torch.Generator('cpu').manual_seed(1000 * t + k)
    x = torch.randn(t, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:8]] *= 30
    return x.to('cuda', torch.bfloat16)


def run_path(lin, x, path):
    clear_quant_cache()
    lin.fused, lin.share_input = path != 'unfused', path == 'reuse'
    y = lin(x)
    if path == 'reuse':
        before = lin.quant_reused
        y = lin(x)
        assert lin.quant_reused == before + 1, 'the second call did not reuse the quantized input'
    torch.cuda.synchronize()
    return y


def run_graph(lin, x):
    clear_quant_cache()
    lin.fused, lin.share_input = True, True
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            lin(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = lin(x)
    g.replay()
    torch.cuda.synchronize()
    out = y.clone()
    del g
    return out


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-root', required=True)
    ap.add_argument('--candidates', required=True, help='comma-separated build names in --build-root')
    ap.add_argument('--set', default=None, help='a select.FAMILIES family to check as a KernelSet (from --build-root)')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--tokens', default=','.join(map(str, TOKENS)))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = Kernel.load('n8k64_wB')
    assert Path(ref.path).parent.parent == REPO / 'sm120' / 'build', f'reference is not sm120/build: {ref.path}'
    cands = {n: Kernel.load(n, build_root=args.build_root) for n in args.candidates.split(',')}
    if args.set:
        cands[f'set:{args.set}'] = KernelSet(args.set, build_root=args.build_root)
    for c in cands.values():
        assert (c.weight_operand, c.type_block) == (ref.weight_operand, ref.type_block), c.cfg.name
    tokens = [int(v) for v in args.tokens.split(',')]
    res = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               gpu=B.gpu_info(), reference=dict(name='n8k64_wB', library=str(ref.path), sha256=ref.sha256),
               candidates={n: (c.sha256 if isinstance(c, Kernel) else c.describe()) for n, c in cands.items()},
               tokens=tokens, graph_tokens=list(GRAPH_TOKENS), activation_quantizer=ACT, paths=list(PATHS) + ['graph'],
               artifacts={}, weights=[], comparisons=collections.Counter(), failures=[], reference_path_disagreements=[])
    work, seen = [], set()
    for model in args.models.split(','):
        real, wsha = real_modules(model)
        res['artifacts'][model] = dict(path=str(ARTIFACTS / f'{model}_tc_8x64'), weights_sha256=wsha)
        for label, pw in real:
            work.append((label, pw, True))
            if pw.shape not in seen:
                seen.add(pw.shape)
                work += [(lab, p, False) for lab, p in synthetic(pw.shape)]
    for label, pw, is_real in work:
        n, k = pw.shape
        res['weights'].append(dict(label=label, shape=[n, k], e0m3_tiles=pw.e0m3_tiles, bias=pw.bias is not None))
        rlin = NativeLinear(pw, ref, ACT, 'ref')
        clins = {c: NativeLinear(pw, kern, ACT, c) for c, kern in cands.items()}
        for t in tokens:
            x = inputs(t, k)
            outs = {p: run_path(rlin, x, p) for p in PATHS}
            for p in PATHS[1:]:
                if not same(outs[p], outs['fused']):
                    res['reference_path_disagreements'].append(dict(weight=label, tokens=t, path=p))
            for c, lin in clins.items():
                for p in PATHS:
                    y = run_path(lin, x, p)
                    res['comparisons'][f'{c} {p}'] += 1
                    if not same(y, outs[p]):
                        d = (y.float() - outs[p].float()).abs().max().item() if y.shape == outs[p].shape else None
                        res['failures'].append(dict(candidate=c, path=p, weight=label, tokens=t, max_abs_diff=d))
                if is_real and t in GRAPH_TOKENS:
                    g_ref = run_graph(rlin, x)
                    y = run_graph(lin, x)
                    res['comparisons'][f'{c} graph'] += 1
                    if not (same(y, g_ref) and same(g_ref, outs['fused'])):
                        res['failures'].append(dict(candidate=c, path='graph', weight=label, tokens=t,
                                                    graph_equals_eager_reference=same(g_ref, outs['fused'])))
            del x, outs
        print(f'{label}: {sum(res["comparisons"].values())} comparisons so far, {len(res["failures"])} failures',
              flush=True)
        del rlin, clins
        torch.cuda.empty_cache()
    if args.set:
        res['candidates'][f'set:{args.set}'] = cands[f'set:{args.set}'].describe()
    res['comparisons'] = dict(res['comparisons'])
    res['total_comparisons'] = sum(res['comparisons'].values())
    res['passed'] = not res['failures']
    res['status'] = 'complete'
    res['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    B.write(args.out, res)
    print(f"{res['total_comparisons']} comparisons, {len(res['failures'])} not bitwise equal; reference path "
          f"disagreements: {len(res['reference_path_disagreements'])}; {'PASSED' if res['passed'] else 'FAILED'}")
    if not res['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
