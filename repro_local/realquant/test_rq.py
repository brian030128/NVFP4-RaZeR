"""GPU unit gate for the real-quant path (run from repro_local/env.sh).

For each kernel configuration and several Llama-shaped problems:
  * weights with a random tile map at the configuration's granule are packed, and decode(packed)
    must equal the campaign's fake-quant weight (FourOverSix base, E0M3 alternative, apply_mask)
    bit for bit -- pack_weight asserts this;
  * activations are packed with the per-token FourOverSix rule and checked against quantize_rows;
  * the native GEMM output is compared with an FP64 reference computed from the exact decoded
    operands (relative Frobenius error), alongside the fake-quant F.linear output;
  * negative control: the UNPATCHED library (every E0M3 site silently decoded as E2M1) must be
    clearly worse than the patched one whenever E0M3 tiles are present.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402
from campaign import quant as Q  # noqa: E402
from campaign import tiles as T  # noqa: E402
from quantize.causal_four_over_six import quantize_rows  # noqa: E402


def exact(code, scale, gs):
    return (code.double() * scale.double().repeat_interleave(16, -1)) * gs.double()


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@torch.no_grad()
def run_case(cfg, t, n, k, p_tile, seed, results):
    torch.manual_seed(seed)
    dev = 'cuda'
    kern = rq.Kernel(cfg)
    bad = rq.Kernel(cfg, rq.LIB_DIR / f'lib{cfg}.so.unpatched')
    tb = rq.TYPE_BLOCK[cfg]
    w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
    # a few heavy input channels, as in real activations
    x = torch.randn(t, k, device=dev)
    x[:, torch.randperm(k, device=dev)[:8]] *= 20
    x = x.bfloat16()
    mask = torch.rand(n // tb[0], k // tb[1], device=dev) < p_tile
    fake_w = T.apply_mask(Q.four_over_six(w), Q.e0m3(w), mask, tb)
    pw, sbytes = rq.pack_weight(w, 'map', mask, tb, expected=fake_w)
    # exact FP64 operands
    c4, s4, g4 = rq.weight_four_over_six(w)
    c0, s0, g0 = rq.weight_e0m3(w)
    f = mask.repeat_interleave(tb[0], 0).repeat_interleave(tb[1] // 16, 1)
    wd = torch.where(f.repeat_interleave(16, 1), exact(c0, s0, g0), exact(c4, s4, g4))
    ac, asc, ag = rq.act_four_over_six_rows(x)
    xd = exact(ac, asc, ag[:, None])
    ref = xd @ wd.t()
    out = {}
    for name, kk in (('patched', kern), ('unpatched', bad)):
        if kk.weight_operand == 0:
            pw.sf_bytes = kk.place_scales(sbytes, 0, n, t, k)
        else:
            pw.sf_bytes = kk.place_scales(sbytes, 1, t, n, k)
        lin = rq.RealLinear(kk, pw, None, 'four_over_six_rows', check_calls=1)
        y = lin(x)
        out[name] = rel(y, ref)
        if name == 'patched':
            assert lin.checked == 1
            y_real = y
    fake = F.linear(quantize_rows(x), fake_w)
    out['fake_linear'] = rel(fake, ref)
    out['real_vs_fake'] = rel(y_real, fake)
    row = dict(cfg=cfg, t=t, n=n, k=k, e0m3_tiles=int(mask.sum()), tiles=mask.numel(), **out)
    results.append(row)
    print(json.dumps(row), flush=True)
    return row


def main():
    results = []
    shapes = [(2048, 4096, 4096), (2048, 1024, 4096), (2048, 14336, 4096), (2048, 4096, 14336), (512, 256, 1024)]
    for cfg in ('wt_as_A', 'b8x64'):
        for i, (t, n, k) in enumerate(shapes):
            run_case(cfg, t, n, k, 0.3, 100 + i, results)
        run_case(cfg, 2048, 4096, 4096, 0.0, 7, results)     # no E0M3: patched == unpatched
        run_case(cfg, 2048, 4096, 4096, 1.0, 8, results)     # all E0M3
    ok = True
    for r in results:
        if r['patched'] > 5e-3:
            ok = False
        if r['e0m3_tiles'] > 0 and not r['unpatched'] > 10 * r['patched']:
            ok = False
        if r['e0m3_tiles'] == 0 and r['unpatched'] != r['patched']:
            ok = False
    print('GATE', 'PASS' if ok else 'FAIL', flush=True)
    json.dump(dict(results=results, gate='PASS' if ok else 'FAIL'),
              open('/home/dev/n16k64_campaign/realquant/unit_gate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
