"""Aligned causal calibration: CE and teacher-KL directional tile scores, N8 and N16 maps and controls.

Protocol (frozen in PROTOCOL_FREEZE.json):
  * teacher: pristine BF16 weights, no activation quantization, SDPA attention;
  * student: FourOverSix weights on every scoped Linear, causal per-token FourOverSix activations
    (quantize_rows) with an identity straight-through derivative, SDPA attention;
  * per calibration sequence i and N8 tile j:  g[i,j] = < dLoss_i/dW_j , (E0M3 - FourOverSix)_j >  (float32),
    for Loss in {mean next-token CE, batchmean KL(teacher || student)};
  * N16 per-sequence scores are float64 sums of the two vertically adjacent N8 children;
  * moments (float64) -> mean, SE -> U_k = max(mean_CE + k SE_CE, mean_KL + k SE_KL); elect iff U_k < 0.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from campaign import data as D
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import quant as Q
from campaign import runtime
from campaign import tiles as T

K_VALUES = (2, 3, 4, 5, 6)
SUBSET_SNAPSHOTS = (8, 16, 32)  # per-domain prefix sizes kept in addition to the full 64+64


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def calibration_inputs(key, draw, tok, n_per_domain=64):
    spec = MOD.REGISTRY[key]
    if draw == 'seed0':
        if spec['panel'] == 'development':
            fit, _ = D.archived_manifest(key)
            batches, meta = D.crops_from_manifest(tok, fit)
            meta['rule'] = 'archived manifest (document hash + token offset), token hashes verified'
        else:
            batches, meta = D.builder_seed0(tok)
    elif draw.startswith('draw'):
        s = int(draw[4:])
        exclude = set()
        if spec['panel'] == 'development':
            fit, _ = D.archived_manifest(key)
            base_meta = fit
        else:
            _, base_meta = D.builder_seed0(tok)
        for dom in ('math', 'code'):
            exclude |= {d['document_sha256'] for d in base_meta[dom]['documents']}
        for prev in range(1, s):
            _, pm = D.keyed_draw(tok, f'draw{prev}', exclude)
            for dom in ('math', 'code'):
                exclude |= {d['document_sha256'] for d in pm[dom]['documents']}
        batches, meta = D.keyed_draw(tok, draw, exclude)
    elif draw == 'heldout':
        b, meta = D.keyed_draw(tok, 'heldout', set(), count=n_per_domain // 2 if n_per_domain > 32 else n_per_domain,
                               domains=('arxiv', 'govreport'))
        batches = {'math': b['arxiv'], 'code': b['govreport']}
        meta = {'math': dict(meta['arxiv'], source='arxiv (slot math)'), 'code': dict(meta['govreport'], source='govreport (slot code)')}
    else:
        raise ValueError(draw)
    seqs = batches['math'][:n_per_domain] + batches['code'][:n_per_domain]
    domains = ['math'] * len(batches['math'][:n_per_domain]) + ['code'] * len(batches['code'][:n_per_domain])
    return seqs, domains, meta


class TileSample:
    """Predeclared stratified raw-score sample: in every module, N16 parents chosen by a keyed hash."""

    def __init__(self, name, o, k, fraction, cap=4096, floor=32):
        n16 = (o // 16) * (k // 64)
        want = min(n16, max(floor, min(cap, int(round(n16 * fraction)))))
        g = torch.Generator().manual_seed(int(hashlib.sha256(f'tile-sample:{name}'.encode()).hexdigest()[:15], 16))
        self.parents = torch.randperm(n16, generator=g)[:want].sort().values
        cols = k // 64
        r, c = self.parents // cols, self.parents % cols
        self.children = torch.stack([(2 * r) * cols + c, (2 * r + 1) * cols + c], 1).reshape(-1)


def build_candidates(modules, alt_on_gpu):
    alt, stats = {}, {}
    for i, (n, m) in enumerate(modules.items()):
        w = m.weight.detach()
        b = Q.four_over_six(w)
        a = Q.e0m3(w)
        wf = w.float()
        eb = (b.float() - wf).square()
        ea = (a.float() - wf).square()
        mse_gain8 = T.tile_sum((eb - ea).double(), T.N8)          # > 0: E0M3 has lower weight MSE on the tile
        l2_8 = T.tile_sum(wf.double().square(), T.N8)               # tile weight energy (magnitude heuristic)
        dnorm8 = T.tile_sum((a.float() - b.float()).double().square(), T.N8)
        o, k = w.shape
        stats[n] = dict(mse_gain8=mse_gain8.cpu(), l2_8=l2_8.cpu(), dnorm8=dnorm8.cpu(),
                        mse_gain16=T.aggregate_rows_1d(mse_gain8, o, k).cpu(), l2_16=T.aggregate_rows_1d(l2_8, o, k).cpu(),
                        dnorm16=T.aggregate_rows_1d(dnorm8, o, k).cpu())
        alt[n] = a if alt_on_gpu else a.cpu().pin_memory()
        m.weight.copy_(b)
        del w, b, wf, eb, ea
        if (i + 1) % 64 == 0:
            print(f'CANDIDATES {i + 1}/{len(modules)}', flush=True)
    return alt, stats


def per_layer_topk(values, counts, names, largest=True):
    masks = {}
    for n in names:
        v = values[n]
        c = counts[n]
        m = torch.zeros(v.numel(), dtype=torch.bool)
        if c:
            order = torch.sort(-v if largest else v, stable=True).indices[:c]
            m[order] = True
        masks[n] = m
    return masks


def run_scoring(model, modules, seqs, domains, teacher_provider, alt, raw='sample', sample_fraction=0.01, subset_moments=False, progress=None, moments_device=None):
    device0 = model.get_input_embeddings().weight.device
    names = list(modules)
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    dev = {n: (torch.device(moments_device) if moments_device else m.weight.device) for n, m in modules.items()}
    tiles8 = {n: (shapes[n][0] // 8) * (shapes[n][1] // 64) for n in names}
    tiles16 = {n: (shapes[n][0] // 16) * (shapes[n][1] // 64) for n in names}
    full8 = {n: T.Moments(tiles8[n], device=dev[n]) for n in names}
    full16 = {n: T.Moments(tiles16[n], device=dev[n]) for n in names}
    run8 = {d: {n: T.Moments(tiles8[n], device=dev[n]) for n in names} for d in ('math', 'code')} if subset_moments else None
    run16 = {d: {n: T.Moments(tiles16[n], device=dev[n]) for n in names} for d in ('math', 'code')} if subset_moments else None
    snapshots = {}
    samples = {n: TileSample(n, *shapes[n], sample_fraction) for n in names}
    nseq = len(seqs)
    raw_sample = {n: dict(ce=torch.empty(nseq, samples[n].children.numel()), kl=torch.empty(nseq, samples[n].children.numel()))
                  for n in names} if raw in ('sample', 'full') else None
    raw_full = None
    if raw == 'full':
        raw_full = {n: dict(ce=torch.empty(nseq, tiles8[n]), kl=torch.empty(nseq, tiles8[n])) for n in names}
    phase_scores = {}
    state = dict(phase=None, seq=0)
    hits = {n: [0, 0] for n in names}

    act = Q.ActivationQuant(modules, 'four_over_six_rows', ste=True)

    def make_hook(name):
        m = modules[name]

        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])

            def backward(dy):
                grad = dy.detach().reshape(-1, dy.shape[-1]).float().T @ x.float()
                a = alt[name]
                d = a.to(grad.device, non_blocking=True).float() - m.weight.detach().float()
                v = T.directional_scores(grad, d, T.N8)
                if not torch.isfinite(v).all():
                    raise FloatingPointError(f'non-finite score in {name}')
                phase_scores.setdefault(name, {})[state['phase']] = v
                hits[name][0 if state['phase'] == 'ce' else 1] += 1
            output.register_hook(backward)
        return forward

    handles = [modules[n].register_forward_hook(make_hook(n)) for n in names]
    runtime.phase('scoring')
    fit_losses = []
    score_hash = {n: hashlib.sha256() for n in names}
    start = time.time()
    for i, s in enumerate(seqs):
        state['seq'] = i
        ids = s.to(device0)
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        tl = teacher_provider(i).to(lp.device)
        if tl.dtype != torch.float32 or tl.ndim != 2:
            tl = tl.float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        ce = F.nll_loss(lp, ids[:, 1:].reshape(-1).to(lp.device))
        kl = F.kl_div(lp, tl, reduction='batchmean', log_target=True)
        state['phase'] = 'ce'
        ce.backward(retain_graph=True)
        state['phase'] = 'kl'
        kl.backward()
        cev, klv = float(ce.detach()), float(kl.detach())
        if not (math.isfinite(cev) and math.isfinite(klv)):
            raise SystemExit('non-finite fit loss')
        fit_losses.append(dict(ce=cev, kl=klv))
        dom = domains[i]
        for n in names:
            c8, k8 = phase_scores[n]['ce'], phase_scores[n]['kl']
            o, kk = shapes[n]
            if c8.device != dev[n]:
                c8, k8 = c8.to(dev[n]), k8.to(dev[n])
            c16 = T.aggregate_n8_to_n16(c8.double().reshape(1, -1), o, kk).reshape(-1)
            k16 = T.aggregate_n8_to_n16(k8.double().reshape(1, -1), o, kk).reshape(-1)
            full8[n].update(c8, k8)
            full16[n].update(c16, k16)
            if run8 is not None:
                run8[dom][n].update(c8, k8)
                run16[dom][n].update(c16, k16)
            c8c, k8c = c8.cpu(), k8.cpu()
            score_hash[n].update(c8c.numpy().tobytes())
            score_hash[n].update(k8c.numpy().tobytes())
            if raw_sample is not None:
                raw_sample[n]['ce'][i] = c8c[samples[n].children]
                raw_sample[n]['kl'][i] = k8c[samples[n].children]
            if raw_full is not None:
                raw_full[n]['ce'][i] = c8c
                raw_full[n]['kl'][i] = k8c
        phase_scores.clear()
        if run8 is not None:
            done = domains[:i + 1].count(dom)
            if done in SUBSET_SNAPSHOTS:
                snapshots[f'{dom}{done}'] = ({n: {a: v.cpu().clone() if torch.is_tensor(v) else v for a, v in run8[dom][n].state().items()} for n in names},
                                             {n: {a: v.cpu().clone() if torch.is_tensor(v) else v for a, v in run16[dom][n].state().items()} for n in names})
        del embeds, logits, lp, tl, ce, kl
        if (i + 1) % 8 == 0:
            if progress is not None:
                progress(i, time.time() - start)
            print(f'SCORED {i + 1}/{nseq} {time.time() - start:.1f}s', flush=True)
    for h in handles:
        h.remove()
    act.remove()
    if not all(v == [nseq, nseq] for v in hits.values()):
        raise SystemExit('some modules did not receive exactly one CE and one KL gradient per sequence')
    return dict(names=names, shapes=shapes, tiles8=tiles8, tiles16=tiles16, full8=full8, full16=full16, run8=run8, run16=run16,
                snapshots=snapshots, samples=samples, raw_sample=raw_sample, raw_full=raw_full, fit_losses=fit_losses,
                score_seconds=time.time() - start, score_stream_sha256={n: score_hash[n].hexdigest() for n in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MOD.REGISTRY))
    ap.add_argument('--draw', default='seed0')
    ap.add_argument('--n-per-domain', type=int, default=64)
    ap.add_argument('--protocol-id', default='aligned-primary')
    ap.add_argument('--freeze', required=True, help='path to PROTOCOL_FREEZE.json (hash is recorded and checked)')
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--raw', choices=('none', 'sample', 'full'), default='sample')
    ap.add_argument('--sample-fraction', type=float, default=0.01)
    ap.add_argument('--subset-moments', action='store_true')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--max-memory-gib', type=float, default=None)
    ap.add_argument('--alt-on-gpu', action='store_true')
    ap.add_argument('--teacher', choices=('ram', 'disk'), default='ram')
    ap.add_argument('--moments-device', default=None, help='e.g. cpu for the 27B model')
    ap.add_argument('--limit-sequences', type=int, default=None, help='smoke tests only')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    out = runtime.out_dir('calibration')
    import shutil
    free_gib = shutil.disk_usage(os.environ.get('CAMPAIGN_ROOT', '.')).free / 2 ** 30
    raw_requested = args.raw
    if args.raw == 'full' and free_gib < 80:
        args.raw = 'sample'  # frozen storage rule: full raw scores only if /home has >= 80 GiB free at launch
    fz = Path(args.freeze)
    if args.freeze_sha256 != 'SMOKE' and runtime.sha256_file(fz) != args.freeze_sha256:
        raise SystemExit('PROTOCOL_FREEZE.json digest mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.set_num_threads(8)
    t_start = time.time()
    spec = MOD.REGISTRY[args.model]
    report = dict(status='running', model=args.model, spec=spec, draw=args.draw, protocol_id=args.protocol_id,
                  freeze=str(fz), freeze_sha256=args.freeze_sha256, attention_backend=args.attn,
                  activation_quantizer='four_over_six_rows (causal per-token, STE identity backward)',
                  weight_baseline='four_over_six', alternative='e0m3 alpha=1', type_blocks=[[8, 64], [16, 64]],
                  n16_construction='per-sequence float64 sum of vertically adjacent N8 tile scores, then moments',
                  k_values=list(K_VALUES), raw_policy=args.raw, raw_requested=raw_requested, free_disk_gib_at_launch=free_gib, sample_fraction=args.sample_fraction)
    save = lambda: runtime.atomic_json(out / 'calibration_report.json', report)
    save()

    tok = MOD.load_tokenizer(args.model)
    seqs, domains, cmeta = calibration_inputs(args.model, args.draw, tok, args.n_per_domain)
    if args.limit_sequences:
        half = args.limit_sequences // 2
        seqs = seqs[:half] + seqs[args.n_per_domain:args.n_per_domain + half]
        domains = domains[:half] + domains[args.n_per_domain:args.n_per_domain + half]
    cal_sha = D.manifest_sha256(cmeta)
    runtime.atomic_json(out / 'calibration_manifest.json', cmeta)
    report.update(calibration_manifest_sha256=cal_sha, sequences=len(seqs),
                  sequence_token_sha256=[D.sha(s) for s in seqs], domains=domains)
    save()

    runtime.phase('load_model')
    ngpu = torch.cuda.device_count()
    mm = None
    if ngpu > 1:
        cap = args.max_memory_gib or 40
        mm = {i: f'{cap}GiB' for i in range(ngpu)}
    model, loading = MOD.load_model(args.model, attn=args.attn, device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
    modules = MOD.scope(model, args.model)
    mm_sha, mm_entries = MOD.module_manifest(modules)
    tok_sha, tok_files = MOD.tokenizer_manifest(args.model)
    runtime.atomic_json(out / 'module_manifest.json', mm_entries)
    total8, bad8 = MOD.tile_totals(modules, (8, 64))
    total16, bad16 = MOD.tile_totals(modules, (16, 64))
    if bad16:
        raise SystemExit(f'modules not divisible by 16x64: {bad16[:5]}')
    if spec['n8_total'] is not None and (total8 != spec['n8_total'] or total16 != spec['n16_total']):
        raise SystemExit(f'tile totals {total8}/{total16} != audited {spec["n8_total"]}/{spec["n16_total"]}')
    if spec['panel'] == 'development':
        prior = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
        if list(prior['matrices']) != list(modules):
            raise SystemExit('module scope/order differs from the archived calibration')
        badw = [n for n, e in zip(modules, mm_entries) if e['weight_sha256'] != prior['matrices'][n]['source_sha256']]
        if badw:
            raise SystemExit(f'weights differ from archived source hashes: {badw[:3]}')
        report['source_weights_match_archived'] = True
    report.update(model_class=type(model).__name__, module_manifest_sha256=mm_sha, tokenizer_manifest_sha256=tok_sha,
                  tokenizer_files=tok_files, n8_total=total8, n16_total=total16, modules=len(modules),
                  device_map=getattr(model, 'hf_device_map', None), loading_info=loading,
                  attn_implementation=model.config._attn_implementation)
    save()
    device0 = model.get_input_embeddings().weight.device

    # ---- teacher
    runtime.phase('teacher')
    teacher, teacher_dir = [], None
    if args.teacher == 'disk':
        teacher_dir = runtime.out_dir('teacher_tmp')
    bf16_nll = []
    with torch.no_grad():
        for i, s in enumerate(seqs):
            ids = s.to(device0)
            logits = model(input_ids=ids, use_cache=False).logits
            lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            if not torch.isfinite(lp).all():
                raise SystemExit('non-finite teacher log-probabilities')
            bf16_nll.append(float(F.nll_loss(lp, ids[:, 1:].reshape(-1).to(lp.device))))
            # BF16 logits; `.float().reshape(-1, V).log_softmax(-1)` at use time is bitwise identical to storing lp
            if teacher_dir is None:
                teacher.append(logits[:, :-1].detach().to('cpu', copy=True))
            else:
                torch.save(logits[:, :-1].detach().cpu(), teacher_dir / f'{i:03d}.pt')
            del logits, lp
    report['bf16_fit_nll'] = bf16_nll
    save()

    # ---- candidates
    runtime.phase('candidates')
    alt, wstats = build_candidates(modules, args.alt_on_gpu)
    report['candidate_weight_sha256'] = MOD.module_manifest(modules)[0]
    save()

    sc = run_scoring(model, modules, seqs, domains, (lambda i: teacher[i]) if teacher_dir is None else (lambda i: torch.load(teacher_dir / f'{i:03d}.pt', weights_only=True)),
                     alt, args.raw, args.sample_fraction, args.subset_moments, progress=lambda i, dt: (report.update(score_seconds_so_far=dt), save()),
                     moments_device=args.moments_device)
    names, shapes, full8, full16, run8, run16, snapshots, samples, raw_sample, raw_full = (sc[k] for k in ('names', 'shapes', 'full8', 'full16', 'run8', 'run16', 'snapshots', 'samples', 'raw_sample', 'raw_full'))
    tiles16 = sc['tiles16']
    report.update(fit_losses=sc['fit_losses'], score_seconds=sc['score_seconds'], score_stream_sha256=sc['score_stream_sha256'])
    report['score_stream_sha256_all'] = sha_bytes(''.join(report['score_stream_sha256'][n] for n in names).encode())
    save()
    if teacher_dir is not None:
        for p in teacher_dir.glob('*.pt'):
            p.unlink()
        teacher_dir.rmdir()
    del teacher

    # ---- persist moments / raw scores
    runtime.phase('elections')
    mom_dir = runtime.out_dir('calibration', 'moments')
    to_cpu = lambda d: {n: {a: (v.cpu() if torch.is_tensor(v) else v) for a, v in m.state().items()} for n, m in d.items()}
    torch.save(dict(names=names, shapes=shapes, n8=to_cpu(full8), n16=to_cpu(full16)), mom_dir / 'moments_full.pt')
    if run8 is not None:
        torch.save(dict(names=names, shapes=shapes, snapshots=snapshots,
                        math64=(to_cpu(run8['math']), to_cpu(run16['math'])), code64=(to_cpu(run8['code']), to_cpu(run16['code']))),
                   mom_dir / 'moments_subsets.pt')
    torch.save(dict(names=names, shapes=shapes, **{k: {n: v[k] for n, v in wstats.items()} for k in next(iter(wstats.values()))}),
               mom_dir / 'weight_tile_stats.pt')
    if raw_sample is not None:
        torch.save(dict(names=names, shapes=shapes, sample_parents={n: samples[n].parents for n in names},
                        sample_children={n: samples[n].children for n in names}, scores=raw_sample,
                        rule='keyed hash sha256(tile-sample:<module>) randperm of N16 parents; fraction/cap/floor recorded'),
                   mom_dir / 'raw_scores_sample.pt')
    if raw_full is not None:
        torch.save(dict(names=names, shapes=shapes, scores=raw_full), mom_dir / 'raw_scores_full.pt')
    report['moment_files'] = {p.name: runtime.sha256_file(p) for p in sorted(mom_dir.iterdir())}
    save()

    # ---- elections and maps
    model_ident = dict(model_id=spec['model_id'], revision=spec['revision'], tokenizer_revision=spec['revision'],
                       model_class=type(model).__name__)
    source_manifest = os.environ.get('CAMPAIGN_SOURCE_MANIFEST_SHA256') or json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    shapes2 = {n: list(shapes[n]) for n in names}
    maps_dir = runtime.out_dir('maps')
    entries = []

    def write(policy, tb, masks_flat, rule_desc):
        grid = {n: masks_flat[n].reshape(shapes[n][0] // tb[0], shapes[n][1] // tb[1]).cpu() for n in names}
        header = MIO.build_header(protocol_id=args.protocol_id, policy=dict(policy, draw=args.draw), model=model_ident,
                                  type_block=tb, masks=grid, weight_shapes=shapes2, source_manifest_sha256=source_manifest,
                                  calibration_manifest_sha256=cal_sha)
        fname = f'{args.model}_{args.draw}_{policy["name"]}.mixfp4map'
        digest, path = MIO.write_map(maps_dir / fname, header, grid, provenance=dict(
            run_id=os.environ.get('CAMPAIGN_RUN_ID'), rule=rule_desc, freeze_sha256=args.freeze_sha256,
            moments_sha256=report['moment_files'], score_stream_sha256_all=report['score_stream_sha256_all']))
        e = dict(policy=policy['name'], type_block=list(tb), path=path, sha256=digest, selected_tiles=header['totals']['selected_tiles'],
                 total_tiles=header['totals']['total_tiles'], selected_weights=header['totals']['selected_weights'],
                 per_module_selected={m['name']: m['selected'] for m in header['modules']})
        entries.append(e)
        return e

    upper3 = {}
    for res, moms, tb in (('n8', full8, T.N8), ('n16', full16, T.N16)):
        for kk in K_VALUES:
            masks = {n: T.elect(moms[n], kk, 'ce_kl').cpu() for n in names}
            write(dict(name=f'{res}_k{kk}', rule='ce_kl', k=kk, resolution=res), tb, masks, f'max(mean_CE+{kk}SE, mean_KL+{kk}SE)<0')
        for rule in ('ce_only', 'kl_only', 'mean_only'):
            masks = {n: T.elect(moms[n], 3, rule).cpu() for n in names}
            write(dict(name=f'{res}_k3_{rule}', rule=rule, k=3, resolution=res), tb, masks, f'{rule} at k=3')
        upper3[res] = {n: T.upper_bound(moms[n], 3, 'ce_kl').cpu() for n in names}
    primary16 = {n: (upper3['n16'][n] < 0) for n in names}
    counts16 = {n: int(primary16[n].sum()) for n in names}
    n8k3_count = sum(int((upper3['n8'][n] < 0).sum()) for n in names)
    # per-layer count-matched controls for N16 k=3
    for seed in range(5):
        g = torch.Generator().manual_seed(1000 + seed)
        masks = {}
        for n in names:
            m = torch.zeros(tiles16[n], dtype=torch.bool)
            if counts16[n]:
                m[torch.randperm(tiles16[n], generator=g)[:counts16[n]]] = True
            masks[n] = m
        write(dict(name=f'n16_random_s{seed}', rule='per-layer count-matched uniform random', matched_to='n16_k3', seed=1000 + seed,
                   resolution='n16'), T.N16, masks, 'torch.randperm per module with Generator(1000+seed), module order')
    write(dict(name='n16_weight_mse', rule='per-layer count-matched largest weight-MSE gain of E0M3 over FourOverSix', matched_to='n16_k3', resolution='n16'),
          T.N16, per_layer_topk({n: wstats[n]['mse_gain16'] for n in names}, counts16, names), 'stable sort descending mse_gain16')
    write(dict(name='n16_magnitude', rule='per-layer count-matched largest tile weight energy sum(w^2)', matched_to='n16_k3', resolution='n16'),
          T.N16, per_layer_topk({n: wstats[n]['l2_16'] for n in names}, counts16, names), 'stable sort descending l2_16')
    write(dict(name='n16_change_norm', rule='per-layer count-matched largest ||E0M3-FourOverSix||^2 on the tile', matched_to='n16_k3', resolution='n16'),
          T.N16, per_layer_topk({n: wstats[n]['dnorm16'] for n in names}, counts16, names), 'stable sort descending dnorm16')
    # density-matched: N16 tiles ranked globally by U_3 until selected weights match N8 k3 as closely as legal
    target16 = int(round(n8k3_count * 512 / 1024))
    flat_u = torch.cat([upper3['n16'][n] for n in names])
    order = torch.sort(flat_u, stable=True).indices[:target16]
    flat = torch.zeros(flat_u.numel(), dtype=torch.bool)
    flat[order] = True
    masks, off = {}, 0
    for n in names:
        masks[n] = flat[off:off + tiles16[n]]
        off += tiles16[n]
    write(dict(name='n16_density_matched_n8k3', rule='global ascending U_3 (N16), count = round(N8k3 tiles * 512 / 1024)',
               target_tiles=target16, n8k3_tiles=n8k3_count, resolution='n16'), T.N16, masks, 'stable global sort of U_3')
    runtime.atomic_json(out / 'map_manifest.json', entries)
    report.update(maps=entries, status='complete', wall_seconds=time.time() - t_start,
                  counts={e['policy']: e['selected_tiles'] for e in entries})
    save()

    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id=args.protocol_id, protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
                    model_class=type(model).__name__, module_manifest_sha256=mm_sha, source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend=args.attn, activation_quantizer='four_over_six_rows+STE'),
        data=dict(calibration_manifest_sha256=cal_sha, evaluation_manifest_sha256=None,
                  token_hashes=dict(calibration=report['sequence_token_sha256']), overlap_audit=None),
        policies=[dict(name=e['policy'], weight_format='FourOverSix/E0M3 tile mix', activation_format='four_over_six_rows',
                       scale_block=16, type_block=e['type_block'], map_path=e['path'], map_sha256=e['sha256'],
                       selected_tiles=e['selected_tiles'], total_tiles=e['total_tiles'], map_reloaded_for_evaluation=None)
                  for e in entries],
        results=dict(raw_outputs=[str(p) for p in sorted(mom_dir.iterdir())] + [str(out / 'calibration_report.json')],
                     summary=dict(counts=report['counts'], n8_total=total8, n16_total=total16, score_seconds=report['score_seconds']),
                     uncertainty={}, attempted_endpoints=['calibration_scores', 'maps'], missing_endpoints=[]),
        logs=[], failures=[]))
    print('MAPS ' + json.dumps({e['policy']: e['selected_tiles'] for e in entries}), flush=True)


if __name__ == '__main__':
    main()
