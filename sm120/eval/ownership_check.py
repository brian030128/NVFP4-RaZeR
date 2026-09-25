#!/usr/bin/env python3
"""Full-model format-ownership proof on a real artifact (checklist section 3).

For every module of an exported artifact, the GEMM is run with the activation operand an exact
identity (code 1.0, scale 1.0, unit epilogue scales, no global scale), so the output equals each
weight element's hardware-decoded value `codebook[format](nibble) * scale` exactly. The check
requires, for every element:
  * output == decode(stored nibble, stored scale, tag)       (bitwise; both are exact in bf16),
  * the format the tensor core used == the selector map's tile for that element, observed wherever
    the two formats disagree (every nibble except +-0),
so for any weight the selector tile, the stored tag and the executed MMA format are traced to each
other. The map is re-read from its file and compared with the artifact's tags as well.

    python sm120/eval/ownership_check.py --artifact sm120/artifacts/qwen4b_n16_k3 \
        --map sm120/maps/qwen4b_seed0_n16_k3.mixfp4map --out sm120/results/ownership/qwen4b_n16_k3.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales  # noqa: E402


def identity(t, k, off, dev):
    nib = torch.zeros((t, k), dtype=torch.uint8, device=dev)
    i = torch.arange(t, device=dev)
    nib[i, off + i] = 2
    sb = torch.full((t, k // 16), 0x38, dtype=torch.uint8, device=dev)
    return N.pack_nibbles(nib), place_scales(sb, k)


@torch.no_grad()
def check_module(kern, pw, mask, chunk=1024):
    n, k = pw.shape
    dev = pw.packed.device
    sf = place_scales(pw.scales, k)
    got = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    for off in range(0, k, chunk):
        t = min(chunk, k - off)
        xp, xsf = identity(t, k, off, dev)
        if kern.weight_operand == 0:
            d = kern.gemm(pw.packed, sf, xp, xsf, n, t, k)
        else:
            d = kern.gemm(xp, xsf, pw.packed, sf, t, n, k)
        got[:, off:off + t] = d.t()
    nib = N.unpack_nibbles(pw.packed)
    want = N.decode_exact(nib, pw.scales, 1.0)
    exact = bool(torch.equal(got.double(), want))
    as_e2m1 = N.decode_exact(nib, pw.scales & 0x7F, 1.0)
    informative = (nib & 7) != 0
    executed_e0m3 = (got.double() != as_e2m1) & informative
    bm, bk = pw.type_block
    tile_e0m3 = mask.to(dev).repeat_interleave(bm, 0).repeat_interleave(bk, 1)
    mismatch = int((executed_e0m3 != (tile_e0m3 & informative)).sum())
    return dict(exact=exact, format_mismatches=mismatch, informative_elements=int(informative.sum()),
                e0m3_elements_observed=int(executed_e0m3.sum()), e0m3_tiles=int(mask.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--kernel', default=None, help='default: n16k64_wA for 16x64 maps, n8k64_wB for 8x64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    meta, weights = A.load(args.artifact)
    header, masks, digest = mapio.read_map(args.map)
    if meta['map']['sha256'] != digest:
        raise SystemExit('artifact was not exported from this map')
    tb = tuple(header['type_block'])
    kern = Kernel.load(args.kernel or {(16, 64): 'n16k64_wA', (8, 64): 'n8k64_wB'}[tb])
    tags = A.masks_from_artifact(weights)
    rows, bad = {}, []
    for name, pw in weights.items():
        if not torch.equal(tags[name], masks[name]):
            bad.append(name)
        rows[name] = check_module(kern, pw, masks[name])
        if not rows[name]['exact'] or rows[name]['format_mismatches']:
            bad.append(name)
    summary = dict(modules=len(rows), all_exact=all(r['exact'] for r in rows.values()),
                   format_mismatches=sum(r['format_mismatches'] for r in rows.values()),
                   informative_elements=sum(r['informative_elements'] for r in rows.values()),
                   e0m3_elements_observed=sum(r['e0m3_elements_observed'] for r in rows.values()),
                   e0m3_tiles=sum(r['e0m3_tiles'] for r in rows.values()), map_tiles=header['totals']['selected_tiles'],
                   tags_equal_map=not any(not torch.equal(tags[n], masks[n]) for n in weights),
                   failing_modules=sorted(set(bad)), kernel=kern.cfg.name, kernel_sha256=kern.sha256,
                   map_sha256=digest, artifact_weights_sha256=meta['weights_sha256'], seconds=round(time.time() - t0, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, modules=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
