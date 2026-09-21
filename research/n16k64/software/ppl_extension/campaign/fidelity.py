"""V43 first-order selector fidelity: predicted CE/KL directional change vs actual tile interventions.

Predicted change of switching a tile = mean over the 128 calibration sequences of its directional score
(first-order Taylor term for the full FourOverSix -> E0M3 step). Actual change = float64 mean over the same
sequences of loss(with switch) - loss(baseline), W4A4 causal evaluation mode (no STE), BF16 teacher for KL.
Actual changes include discontinuous activation-rounding effects (see V14 causality diagnostic).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from campaign import calibrate as C
from campaign import models as MOD
from campaign import quant as Q
from campaign import runtime
from campaign import tiles as T

PREFIX_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def losses(model, seqs, teacher_logits, batch):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibration-run', required=True)
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--per-stratum', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20260912)
    args = ap.parse_args()
    if runtime.sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit('freeze digest mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('fidelity')
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
    rep = dict(status='running', model=args.model, calibration_run=str(crun), per_stratum=args.per_stratum, seed=args.seed, results={})
    save = lambda: runtime.atomic_json(out / 'fidelity_report.json', rep)
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
    runtime.phase('baseline')
    t0 = time.time()
    base_ce, base_kl = losses(model, seqs, teacher, args.batch)
    rep['baseline'] = dict(ce=float(base_ce.mean()), kl=float(base_kl.mean()), seconds=time.time() - t0)
    save()
    alt_cache = {}

    def alt_of(n):
        if n not in alt_cache:
            alt_cache.clear()
            alt_cache[n] = Q.e0m3(pristine[n].to(modules[n].weight.device))
        return alt_cache[n]

    def tile_slice(n, idx, tb):
        go, gk = shapes[n][0] // tb[0], shapes[n][1] // tb[1]
        r, c = divmod(int(idx), gk)
        return slice(r * tb[0], (r + 1) * tb[0]), slice(c * tb[1], (c + 1) * tb[1])

    def switch(tiles, tb, on):
        with torch.no_grad():
            for n, idx in tiles:
                rs, cs = tile_slice(n, idx, tb)
                w = modules[n].weight
                if on:
                    w[rs, cs] = alt_of(n)[rs, cs]
                else:
                    w[rs, cs] = Q.four_over_six(pristine[n].to(w.device))[rs, cs]

    g = torch.Generator().manual_seed(args.seed)
    layer_of = lambda n: int(n.split('layers.')[1].split('.')[0]) if 'layers.' in n else -1
    nlayers = max(layer_of(n) for n in names) + 1
    for res, tb in (('n8', T.N8), ('n16', T.N16)):
        m = {n: T.Moments.from_state(mom[res][n]) for n in names}
        U = torch.cat([T.upper_bound(m[n], 3, 'ce_kl') for n in names])
        mce = torch.cat([m[n].mean_se('ce')[0] for n in names])
        mkl = torch.cat([m[n].mean_se('kl')[0] for n in names])
        owner = []
        for n in names:
            owner.extend([n] * m[n].ce_sum.numel())
        offs, o = {}, 0
        for n in names:
            offs[n] = o
            o += m[n].ce_sum.numel()
        pos = U[U >= 0]
        sub = pos[torch.randperm(pos.numel(), generator=g)[:2_000_000]] if pos.numel() > 2_000_000 else pos
        q10, med = float(torch.quantile(sub, 0.10)), float(torch.quantile(sub, 0.5))
        strata = dict(selected=(U < 0).nonzero().flatten(), near_threshold=((U >= 0) & (U < q10)).nonzero().flatten(),
                      rejected=(U > med).nonzero().flatten(), random=torch.arange(U.numel()))
        picks = {}
        for sname, pool in strata.items():
            k = min(args.per_stratum, pool.numel())
            picks[sname] = pool[torch.randperm(pool.numel(), generator=g)[:k]]
        records = []
        for sname, idxs in picks.items():
            runtime.phase(f'{res}_{sname}')
            for gi in idxs.tolist():
                n = owner[gi]
                li = gi - offs[n]
                switch([(n, li)], tb, True)
                ce, kl = losses(model, seqs, teacher, args.batch)
                switch([(n, li)], tb, False)
                d_ce, d_kl = ce - base_ce, kl - base_kl
                mc, sc = m[n].mean_se('ce')
                mk, sk = m[n].mean_se('kl')
                records.append(dict(stratum=sname, module=n, layer=layer_of(n), layer_third=min(2, 3 * layer_of(n) // nlayers) if nlayers > 0 else None,
                                    module_type=n.split('.')[-1], tile=li, U3=float(U[gi]), pred_ce=float(mc[li]), se_ce=float(sc[li]),
                                    pred_kl=float(mk[li]), se_kl=float(sk[li]), actual_ce=float(d_ce.mean()), actual_kl=float(d_kl.mean()),
                                    actual_ce_se=float(d_ce.std() / math.sqrt(len(d_ce))), actual_kl_se=float(d_kl.std() / math.sqrt(len(d_kl))),
                                    actual_ce_per_seq=d_ce.tolist(), actual_kl_per_seq=d_kl.tolist()))
            rep['results'].setdefault(res, {})['singles'] = records
            save()
        # batched prefixes of selected tiles ranked by U3
        sel = strata['selected'][torch.argsort(U[strata['selected']], stable=True)]
        sizes = [s for s in PREFIX_SIZES if s < sel.numel()] + [int(sel.numel())]
        batched = []
        runtime.phase(f'{res}_batched')
        for s in sizes:
            tiles = [(owner[gi], gi - offs[owner[gi]]) for gi in sel[:s].tolist()]
            switch(tiles, tb, True)
            ce, kl = losses(model, seqs, teacher, args.batch)
            switch(tiles, tb, False)
            batched.append(dict(size=s, actual_ce=float((ce - base_ce).mean()), actual_kl=float((kl - base_kl).mean()),
                                pred_ce_sum=float(mce[sel[:s]].sum()), pred_kl_sum=float(mkl[sel[:s]].sum())))
            rep['results'][res]['batched'] = batched
            save()
        # restore full baseline (guards against any partial switch)
        with torch.no_grad():
            for n, mm_ in modules.items():
                mm_.weight.copy_(Q.four_over_six(pristine[n].to(mm_.weight.device)))
        chk_ce, _ = losses(model, seqs[:8], teacher[:8], args.batch)
        rep['results'][res]['baseline_restored_exact'] = bool(torch.equal(chk_ce, base_ce[:8]))
        rep['results'][res]['thresholds'] = dict(q10_positive_U3=q10, median_positive_U3=med, selected_total=int(strata['selected'].numel()))
        save()
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
        policies=[], results=dict(raw_outputs=[str(out / 'fidelity_report.json')], summary={}, uncertainty={},
                                  attempted_endpoints=['n8 singles', 'n8 batched', 'n16 singles', 'n16 batched'], missing_endpoints=[]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
