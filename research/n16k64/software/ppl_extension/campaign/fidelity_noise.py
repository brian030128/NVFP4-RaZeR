"""V43 noise-floor control for the first-order fidelity instrument.

V43 estimates a tile's causal effect as the difference between one loss measurement with the tile
switched and one baseline measurement taken once, at the start of the run. This control measures how
large the numerical noise of a single measurement is, under the exact V43 conditions (FourOverSix
weights, causal row-wise activation quantization, no STE, SDPA attention, BF16 teacher for KL), so
that the V43 singles can be read against their own resolution rather than against zero.

Blocks:
  repeat - R consecutive baseline measurements with nothing touched in between. The spread of these
           is the noise floor of one measurement of the mean loss.
  cycle  - R baseline measurements, each preceded by a switch(tile, on) + switch(tile, off) pair, so
           the GPU-side restore path is exercised exactly the way V43 exercises it. The CPU proof
           that the restore is bitwise is in campaign/tests; this checks it on device.
  tiles  - a few N16 tiles measured with replication, interleaved with baselines, so a single-tile
           effect can be estimated against replicate noise instead of against a single reference.
  drift  - R further baseline measurements at the end, which bound the error V43 makes by comparing
           interventions against a baseline measured hours earlier.

Nothing here changes a confirmatory endpoint; it is a diagnostic on the V43 row.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from campaign import calibrate as C
from campaign import models as MOD
from campaign import quant as Q
from campaign import runtime
from campaign import tiles as T


def losses(model, seqs, teacher_logits, batch):
    """Identical to campaign.fidelity.losses; kept in sync so the two measure the same quantity."""
    dev = model.get_input_embeddings().weight.device
    ce, kl = [], []
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            ids = torch.cat(seqs[s:s + batch]).to(dev)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
            for j in range(ids.shape[0]):
                lp = logits[j].float().log_softmax(-1)
                tl = teacher_logits[s + j].to(lp.device).float().log_softmax(-1)
                ce.append(float(F.nll_loss(lp, ids[j, 1:].to(lp.device), reduction='mean').double()))
                kl.append(float((tl.exp() * (tl - lp)).sum(-1).double().mean()))
            del logits
    return torch.tensor(ce, dtype=torch.float64), torch.tensor(kl, dtype=torch.float64)


def block_stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return dict(n=0)
    sd = None
    if a.size > 1:
        # bitwise-identical values have zero spread; a computed mean would leave a ~1e-16 residue
        sd = 0.0 if a.min() == a.max() else float(a.std(ddof=1))
    return dict(n=int(a.size), mean=float(a.mean()), sd=sd,
                min=float(a.min()), max=float(a.max()), range=float(a.max() - a.min()))


def pooled_within_sd(groups):
    """SD of one measurement, pooled about each block's own mean (so a block-to-block shift does not
    inflate it). A block whose values are bitwise identical contributes exactly zero: subtracting a
    computed mean would otherwise report ~1e-16 of noise where the measurements are provably equal.
    groups: list of lists of measured means."""
    num = den = 0.0
    for g in groups:
        a = np.asarray(g, dtype=np.float64)
        if a.size > 1:
            den += a.size - 1
            if a.min() != a.max():
                num += float(((a - a.mean()) ** 2).sum())
    return math.sqrt(num / den) if den > 0 else None


def replicates_for_3se(sd, effect):
    """Replicates per arm needed for a 3-SE separation of a two-arm difference of means."""
    if not sd or not effect:
        return None
    return int(math.ceil(18.0 * (sd ** 2) / (effect ** 2)))


def summarize(measurements, tiles):
    out = {}
    base_of = lambda blk: [m for m in measurements if m['block'] == blk and m['state'] == 'baseline']
    for blk in ('repeat', 'cycle', 'tiles', 'drift'):
        sel = base_of(blk)
        if sel:
            out[blk] = {o: block_stats([m[o] for m in sel]) for o in ('ce', 'kl')}
    noise = {}
    for o in ('ce', 'kl'):
        groups = [[m[o] for m in base_of(b)] for b in ('repeat', 'cycle', 'tiles', 'drift')]
        groups = [g for g in groups if len(g) > 1]
        sd = pooled_within_sd(groups)
        first = measurements[0][o] if measurements else None
        allb = [m[o] for m in base_of('repeat') + base_of('cycle') + base_of('tiles') + base_of('drift')]
        noise[o] = dict(single_measurement_sd=sd,
                        difference_of_two_measurements_sd=(sd * math.sqrt(2) if sd is not None else None),
                        max_abs_deviation_from_first_baseline=(max(abs(v - first) for v in allb) if allb and first is not None else None),
                        all_baselines=block_stats(allb))
        r, d = base_of('repeat'), base_of('drift')
        if r and d:
            ra, da = np.asarray([m[o] for m in r]), np.asarray([m[o] for m in d])
            se = math.sqrt((ra.var(ddof=1) / ra.size if ra.size > 1 else 0) + (da.var(ddof=1) / da.size if da.size > 1 else 0))
            noise[o]['drift_end_minus_start'] = dict(delta=float(da.mean() - ra.mean()), se=(se or None),
                                                     t=(float((da.mean() - ra.mean()) / se) if se else None),
                                                     note='V43 compares every intervention against a baseline measured before this shift')
    out['noise'] = noise
    rows = []
    for t in tiles:
        on = [m for m in measurements if m['block'] == 'tiles' and m['state'] == 'tile_on'
              and m['tile'] and m['tile']['module'] == t['module'] and m['tile']['tile'] == t['tile']]
        ref = base_of('tiles')
        row = dict(t)
        for o in ('ce', 'kl'):
            a = np.asarray([m[o] for m in on], dtype=np.float64)
            b = np.asarray([m[o] for m in ref], dtype=np.float64)
            if a.size and b.size:
                se = math.sqrt((a.var(ddof=1) / a.size if a.size > 1 else 0) + (b.var(ddof=1) / b.size if b.size > 1 else 0))
                eff = float(a.mean() - b.mean())
                row[o] = dict(replicates=int(a.size), baselines=int(b.size), effect=eff, se=(se or None),
                              t=(float(eff / se) if se else None), predicted=t.get('pred_' + o),
                              replicates_for_3se_of_prediction=replicates_for_3se(noise[o]['single_measurement_sd'], t.get('pred_' + o)))
        rows.append(row)
    out['tiles'] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibration-run', required=True)
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--cycles', type=int, default=8)
    ap.add_argument('--tile-replicates', type=int, default=4)
    ap.add_argument('--drift-repeats', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20260913)
    args = ap.parse_args()
    if runtime.sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit('freeze digest mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('fidelity_noise')
    if args.calibration_run.startswith('@latest:'):
        from campaign.policies import latest_complete_run
        crun = latest_complete_run(os.environ['CAMPAIGN_ROOT'], args.calibration_run.split(':', 1)[1])
    else:
        crun = Path(args.calibration_run)
    crep = json.loads((crun / 'calibration' / 'calibration_report.json').read_text())
    if crep['status'] != 'complete' or crep['model'] != args.model or crep['draw'] != 'seed0':
        raise SystemExit('need a complete seed0 aligned calibration of this model')
    mom = torch.load(crun / 'calibration' / 'moments' / 'moments_full.pt', weights_only=False)
    if runtime.sha256_file(crun / 'calibration' / 'moments' / 'moments_full.pt') != crep['moment_files']['moments_full.pt']:
        raise SystemExit('moments digest mismatch')
    names = mom['names']
    shapes = {n: tuple(mom['shapes'][n]) for n in names}
    rep = dict(status='running', model=args.model, calibration_run=str(crun), seed=args.seed,
               config=dict(repeats=args.repeats, cycles=args.cycles, tile_replicates=args.tile_replicates,
                           drift_repeats=args.drift_repeats, batch=args.batch, type_block='n16'),
               measurements=[], tiles=[])
    save = lambda: runtime.atomic_json(out / 'fidelity_noise_report.json', rep)
    save()
    tok = MOD.load_tokenizer(args.model)
    seqs, domains, cmeta = C.calibration_inputs(args.model, 'seed0', tok)
    from campaign import data as D
    if [D.sha(s) for s in seqs] != crep['sequence_token_sha256']:
        raise SystemExit('calibration sequences differ from the calibration run')
    runtime.phase('load_model')
    model, _ = MOD.load_model(args.model, attn='sdpa', device_map='cuda')
    modules = MOD.scope(model, args.model)
    if list(modules) != names:
        raise SystemExit('scope mismatch')
    teacher = []
    dev0 = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for s in range(0, len(seqs), args.batch):
            ids = torch.cat(seqs[s:s + args.batch]).to(dev0)
            lg = model(input_ids=ids, use_cache=False).logits[:, :-1]
            teacher.extend(lg[j].detach().to('cpu', copy=True) for j in range(ids.shape[0]))
            del lg
    pristine = {n: m.weight.detach().to('cpu', copy=True) for n, m in modules.items()}
    with torch.no_grad():
        for n, m in modules.items():
            m.weight.copy_(Q.four_over_six(m.weight))
    act = Q.ActivationQuant(modules, 'four_over_six_rows', ste=False)

    alt_cache = {}

    def alt_of(n):
        if n not in alt_cache:
            alt_cache.clear()
            alt_cache[n] = Q.e0m3(pristine[n].to(modules[n].weight.device))
        return alt_cache[n]

    def tile_slice(n, idx, tb):
        gk = shapes[n][1] // tb[1]
        r, c = divmod(int(idx), gk)
        return slice(r * tb[0], (r + 1) * tb[0]), slice(c * tb[1], (c + 1) * tb[1])

    def switch(n, idx, on, tb=T.N16):
        with torch.no_grad():
            rs, cs = tile_slice(n, idx, tb)
            w = modules[n].weight
            w[rs, cs] = alt_of(n)[rs, cs] if on else Q.four_over_six(pristine[n].to(w.device))[rs, cs]

    def measure(block, state, tile=None):
        t0 = time.time()
        ce, kl = losses(model, seqs, teacher, args.batch)
        rec = dict(block=block, state=state, tile=tile, ce=float(ce.mean()), kl=float(kl.mean()),
                   ce_per_seq=ce.tolist(), kl_per_seq=kl.tolist(), seconds=time.time() - t0,
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        rep['measurements'].append(rec)
        save()
        return rec

    # pick N16 tiles: the largest predicted gain, a median selected tile, and a rejected tile
    m16 = {n: T.Moments.from_state(mom['n16'][n]) for n in names}
    U = torch.cat([T.upper_bound(m16[n], 3, 'ce_kl') for n in names])
    mce = torch.cat([m16[n].mean_se('ce')[0] for n in names])
    mkl = torch.cat([m16[n].mean_se('kl')[0] for n in names])
    owner, offs, o = [], {}, 0
    for n in names:
        k = m16[n].ce_sum.numel()
        owner.extend([n] * k)
        offs[n] = o
        o += k
    sel = (U < 0).nonzero().flatten()
    sel = sel[torch.argsort(U[sel], stable=True)]
    g = torch.Generator().manual_seed(args.seed)
    pos = (U > 0).nonzero().flatten()
    chosen = []
    if sel.numel():
        chosen.append(('selected_best', int(sel[0])))
        chosen.append(('selected_median', int(sel[sel.numel() // 2])))
    if pos.numel():
        chosen.append(('rejected', int(pos[torch.randint(pos.numel(), (1,), generator=g)])))
    for label, gi in chosen:
        n = owner[gi]
        rep['tiles'].append(dict(label=label, module=n, tile=gi - offs[n], layer=(int(n.split('layers.')[1].split('.')[0]) if 'layers.' in n else -1),
                                 module_type=n.split('.')[-1], U3=float(U[gi]), pred_ce=float(mce[gi]), pred_kl=float(mkl[gi])))
    save()

    runtime.phase('baseline')
    measure('baseline', 'baseline')
    runtime.phase('repeat')
    for _ in range(args.repeats):
        measure('repeat', 'baseline')
    runtime.phase('cycle')
    if rep['tiles']:
        t0 = rep['tiles'][0]
        for _ in range(args.cycles):
            switch(t0['module'], t0['tile'], True)
            switch(t0['module'], t0['tile'], False)
            measure('cycle', 'baseline', dict(module=t0['module'], tile=t0['tile'], cycled=True))
    runtime.phase('tiles')
    for _ in range(args.tile_replicates):
        measure('tiles', 'baseline')
        for t in rep['tiles']:
            switch(t['module'], t['tile'], True)
            measure('tiles', 'tile_on', dict(module=t['module'], tile=t['tile']))
            switch(t['module'], t['tile'], False)
    runtime.phase('drift')
    for _ in range(args.drift_repeats):
        measure('drift', 'baseline')
    rep['summary'] = summarize(rep['measurements'], rep['tiles'])
    act.remove()
    rep['status'] = 'complete'
    save()
    spec = MOD.REGISTRY[args.model]
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='aligned-analysis', protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=type(model).__name__,
                    module_manifest_sha256=crep['module_manifest_sha256'], source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend='sdpa', activation_quantizer='four_over_six_rows'),
        data=dict(calibration_manifest_sha256=crep['calibration_manifest_sha256'], evaluation_manifest_sha256=None,
                  token_hashes=dict(calibration=crep['sequence_token_sha256']), overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / 'fidelity_noise_report.json')], summary={}, uncertainty={},
                                  attempted_endpoints=['repeat', 'cycle', 'tiles', 'drift'],
                                  missing_endpoints=sorted(runtime.collect_missing(out))),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
