"""Historical-protocol reconstruction (V20/V21): archived causal/eager calibration, k-SE elections, anchors.

Faithful replication of run_math_code_calibration.main + run_kse_paper.elect_k with these documented,
numerically-neutral deviations:
  D-H1 inputs are rebuilt from the pinned local files (crops_from_manifest) instead of streaming
       load_dataset; the archived per-sequence token SHA-256 lists are asserted equal.
  D-H2 the model is loaded from the pinned local snapshot directory (weights asserted equal to the
       archived per-matrix SHA-256).
  D-H3 teacher logits are kept as BF16 tensors in host RAM and converted with the identical
       `.float().reshape(-1, V).log_softmax(-1)` at use time (bitwise identical to the archived
       float32 files, half the memory).
  D-H4 'SLURM_JOB_ID' is not required; the campaign lease id is recorded instead.
  D-H5 (27B only, --stream) raw 128-row score tables are not held in RAM; float64 per-subset
       sums/squares are streamed instead. Shard-file hashes are then not reproducible and are
       reported as not attempted; subset means/SEs differ from torch.std only by float64 rounding.
"""
import argparse
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from campaign import data as D
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime
from campaign import tiles as T
from quantize.adaptive_prefix import adaptive_prefix, derive_maps, source_subsets
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.relinearized_format import common_descent_scores

ORIGINS = {'qwen4b': 'results/pooled_scale/model_332389_qwen4b', 'llama8b': 'results/pooled_scale/model_332389_llama8b',
           'qwen27b': 'results/pooled_qwen27b/model_332840'}
K_VALUES = (2, 3, 4, 5, 6)
FROZEN = 'fixed256_math_code128'


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(p):
    return runtime.sha256_file(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(ORIGINS), required=True)
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--stream', action='store_true', help='D-H5 streaming moments (27B)')
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--direction-cache-gib', type=float, default=None)
    ap.add_argument('--attn', default='eager', help="investigation only: 'sdpa' changes the kernel path, not the protocol semantics")
    ap.add_argument('--compare-run', default=None, help='investigation: job id of the eager A6000 historical run to compare maps with')
    args = ap.parse_args()
    src = D.SOURCE_ROOT
    target = args.model == 'qwen27b'
    torch.set_num_threads(12 if target else 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('historical')
    spec = MOD.REGISTRY[args.model]
    arch = json.loads((src / spec['archived_calibration'] / 'report.json').read_text())
    arch_maps = json.loads((src / spec['archived_calibration'] / 'maps.json').read_text())
    prior = json.loads((src / ORIGINS[args.model] / 'report.json').read_text())
    kse_dir = {'llama8b': 'results/kse_paper/job_336566/llama8b', 'qwen4b': 'results/kse_paper/job_336566/qwen4b',
               'qwen27b': 'results/kse_paper/job_336969/qwen27b'}[args.model]
    kse = json.loads((src / kse_dir / 'report.json').read_text())
    audit = json.loads((src / 'results/math_code_adaptive/summary_333786_333788/curvature_audit.json').read_text())
    arch_shard_sha = {Path(k).name: v for k, v in audit['score_artifact_sha256'].items() if f'_{args.model}/scores/' in k}
    r = dict(status='running', model=args.model, protocol_id='historical', deviations=__doc__.split('documented,')[1],
             torch_version=torch.__version__, transformers_version=transformers.__version__,
             archived_transformers_version=prior['transformers_version'], stream=args.stream,
             lease=os.environ.get('CAMPAIGN_LEASE_ID'), freeze_sha256=args.freeze_sha256, attn=args.attn,
             role=('anchor' if args.attn == 'eager' else 'investigation: kernel-path perturbation on the same GPU'))
    save = lambda: runtime.atomic_json(out / 'historical_report.json', r)
    save()
    if transformers.__version__ != prior['transformers_version']:
        raise SystemExit(f'transformers {transformers.__version__} != archived {prior["transformers_version"]}')
    runtime.phase('load_model')
    ngpu = torch.cuda.device_count()
    path = str(MOD.snapshot_path(args.model))
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=args.attn,
            device_map='balanced', max_memory={i: f'{args.max_memory_gib}GiB' for i in range(ngpu)}, output_loading_info=True)
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=args.attn, device_map='cuda')
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    if list(modules) != list(prior['matrices']):
        raise SystemExit('module scope differs from archived')
    for n, m in modules.items():
        if sha(m.weight) != prior['matrices'][n]['source_sha256']:
            raise SystemExit(f'weight hash mismatch {n}')
    r['source_weights_verified'] = True
    tok = AutoTokenizerFrom(path)
    batches_by, fit_meta = D.crops_from_manifest(tok, prior['fit'])
    for dom in ('math', 'code'):
        if fit_meta[dom]['token_sha256'] != prior['fit'][dom]['token_sha256'] or fit_meta[dom]['token_sha256'] != arch['fit'][dom]['token_sha256']:
            raise SystemExit(f'{dom} calibration tokens differ from archived')
    batches = [b for dom in ('math', 'code') for b in batches_by[dom]]
    r['calibration_tokens_equal_archived'] = True
    save()
    device = model.get_input_embeddings().weight.device
    runtime.phase('teacher')
    teacher_logits, bf16_nll = [], []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            ids = batch.to(device)
            logits = model(input_ids=ids, use_cache=False).logits
            lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            bf16_nll.append(float(F.nll_loss(lp, ids[:, 1:].reshape(-1).to(lp.device))))
            teacher_logits.append(logits[:, :-1].detach().to('cpu', copy=True))
            del ids, logits, lp
    r['bf16_fit_nll'] = bf16_nll
    r['bf16_fit_nll_vs_archived'] = dict(max_abs=max(abs(a - b) for a, b in zip(bf16_nll, arch['bf16_fit_nll'])),
                                         mean_signed=sum(a - b for a, b in zip(bf16_nll, arch['bf16_fit_nll'])) / 128,
                                         exact_equal=bf16_nll == arch['bf16_fit_nll'])
    save()
    runtime.phase('candidates')
    alt, directions, mse, base_cpu = {}, {}, {}, {}
    cached = {}
    limit = int((args.direction_cache_gib if args.direction_cache_gib is not None else (12 if target else 4)) * 1024 ** 3)
    with torch.no_grad():
        for i, (name, m) in enumerate(modules.items()):
            w = m.weight.detach()
            o, k = w.shape
            b = quant_nvfp4_4over6(w, 4, 16)
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            diff = (a.float() - w.float()).square() - (b.float() - w.float()).square()
            mse[name] = (diff.reshape(o // 8, 8, k // 64, 64).sum((1, 3)) < 0).cpu()
            alt[name] = a.cpu().pin_memory() if target else a
            key = str(w.device)
            size = w.numel() * 4
            if cached.get(key, 0) + size <= limit:
                directions[name] = a.float() - b.float()
                cached[key] = cached.get(key, 0) + size
            m.weight.copy_(b)
        del a, b, w, diff
    # weight_mse.pt bytes use the archived source string so the file digest is comparable
    wm = out / 'weight_mse.pt'
    torch.save(dict(source=arch['source'], revision=arch['revision'], weight_mse=mse), wm)
    r['weight_mse_sha256'] = file_sha(wm)
    r['weight_mse_equal_archived'] = r['weight_mse_sha256'] == arch['weight_mse_sha256']
    save()
    names = list(modules)
    subsets = source_subsets()
    stream = args.stream
    if not stream:
        tables = {n: [torch.empty(128, m.weight.numel() // 512) for _ in range(3)] for n, m in modules.items()}
    else:
        prefixes = {('math', c): list(range(c)) for c in (8, 16, 32, 64)} | {('code', c): list(range(64, 64 + c)) for c in (8, 16, 32, 64)}
        acc = {n: {p: dict(cs=torch.zeros(m.weight.numel() // 512, dtype=torch.float64), cq=torch.zeros(m.weight.numel() // 512, dtype=torch.float64),
                           ks=torch.zeros(m.weight.numel() // 512, dtype=torch.float64), kq=torch.zeros(m.weight.numel() // 512, dtype=torch.float64),
                           fq=torch.zeros(m.weight.numel() // 512, dtype=torch.float64)) for p in prefixes} for n, m in modules.items()}
        k2_ce = {n: T.Moments(m.weight.numel() // 512) for n, m in modules.items()}
    hits = {n: [0, 0, 0] for n in modules}
    state = dict(phase=0, sequence=0)
    current = {}

    def act(module, inputs):
        x = inputs[0]
        q = quantize_rows(x.detach())
        return (q + (x - x.detach()), *inputs[1:])

    def make_hook(name):
        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])

            def backward(dy):
                grad = dy.detach().reshape(-1, dy.shape[-1]).float().T @ x.float()
                d = directions.get(name)
                if d is None:
                    d = alt[name].to(grad.device, non_blocking=True).float() - modules[name].weight.detach().float()
                o, k = grad.shape
                value = (grad * d).reshape(o // 8, 8, k // 64, 64).sum((1, 3)).flatten()
                assert torch.isfinite(value).all(), name
                if not stream:
                    tables[name][state['phase']][state['sequence']].copy_(value.cpu())
                else:
                    current.setdefault(name, {})[state['phase']] = value.cpu()
                hits[name][state['phase']] += 1
            output.register_hook(backward)
        return forward

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
    runtime.phase('scoring')
    generator = None
    fit_losses = []
    start = time.perf_counter()
    for sequence, batch in enumerate(batches):
        state['sequence'] = sequence
        ids = batch.to(device)
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        teacher = teacher_logits[sequence].to(lp.device).float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        ce = F.nll_loss(lp, ids[:, 1:].reshape(-1).to(lp.device))
        kl = F.kl_div(lp, teacher, reduction='batchmean', log_target=True)
        state['phase'] = 0
        ce.backward(retain_graph=True)
        state['phase'] = 1
        kl.backward(retain_graph=True)
        if generator is None:
            generator = torch.Generator(device=lp.device).manual_seed(20260930)
        with torch.no_grad():
            labels = torch.multinomial(lp.detach().exp(), 1, generator=generator).squeeze(-1)
        sampled = F.nll_loss(lp, labels)
        state['phase'] = 2
        sampled.backward()
        fit_losses.append(dict(ce=float(ce.detach()), kl=float(kl.detach()), sampled_ce=float(sampled.detach())))
        if stream:
            for n in names:
                c, kk_, f = (current[n][p].double() for p in (0, 1, 2))
                k2_ce[n].update(c, kk_)
                for (dom, cnt), ids_ in prefixes.items():
                    if sequence in ids_:
                        a_ = acc[n][(dom, cnt)]
                        a_['cs'] += c; a_['cq'] += c * c; a_['ks'] += kk_; a_['kq'] += kk_ * kk_; a_['fq'] += f * f
            current.clear()
        teacher_logits[sequence] = None
        del embeds, logits, lp, teacher, ce, kl, sampled, labels
        if (sequence + 1) % 8 == 0:
            r['initial_fit_losses'] = fit_losses
            save()
            print(f'SCORED {sequence + 1}/128 {time.perf_counter() - start:.1f}s', flush=True)
    assert all(v == [128, 128, 128] for v in hits.values())
    for h in handles:
        h.remove()
    r['score_seconds'] = time.perf_counter() - start
    r['initial_fit_losses'] = fit_losses
    r['initial_fit_losses_vs_archived'] = {k2: dict(max_abs=max(abs(a[k2] - b[k2]) for a, b in zip(fit_losses, arch['initial_fit_losses'])),
                                                    mean_signed=sum(a[k2] - b[k2] for a, b in zip(fit_losses, arch['initial_fit_losses'])) / 128,
                                                    exact_equal=[a[k2] for a in fit_losses] == [b[k2] for b in arch['initial_fit_losses']])
                                          for k2 in ('ce', 'kl', 'sampled_ce')}
    save()
    del directions, alt
    torch.cuda.empty_cache()
    runtime.phase('elections')
    shapes = {n: list(prior['matrices'][n]['shape']) for n in names}
    # ---- shard hashes (non-stream only): write each shard exactly as archived, hash, delete
    shard_dir = runtime.out_dir('historical', 'scores_tmp')
    if not stream:
        match = 0
        shard_hash = {}
        for i, (name, values) in enumerate(tables.items()):
            p = shard_dir / f'{i:03d}.pt'
            torch.save(dict(name=name, ce=values[0], kl=values[1], fisher=values[2]), p)
            shard_hash[p.name] = file_sha(p)
            match += int(arch_shard_sha.get(p.name) == shard_hash[p.name])
            p.unlink()
        r['shard_sha256'] = shard_hash
        r['shard_sha256_equal_archived'] = dict(equal=match, total=len(tables), archived_available=len(arch_shard_sha))
        maps, stats = derive_maps(((n, shapes[n], *v) for n, v in tables.items()), include_fixed=True)
    else:
        r['shard_sha256_equal_archived'] = dict(equal=None, total=len(names), not_attempted='D-H5 streaming mode', archived_available=len(arch_shard_sha))
        maps, stats = derive_maps_streamed(names, shapes, acc, subsets)
    total = sum(s[0] // 8 * (s[1] // 64) for s in shapes.values())
    stats['four_over_six'] = dict(selected_blocks=0, total_type_blocks=total, selected_fraction=0.)
    mse_count = sum(int(m.sum()) for m in mse.values())
    stats['weight_mse'] = dict(selected_blocks=mse_count, total_type_blocks=total, selected_fraction=mse_count / total)
    mj = out / 'maps.json'
    mj.write_text(json.dumps(dict(source=arch['source'], revision=arch['revision'], type_block=[8, 64], baseline='FourOverSix',
                                  alternative='E0M3 alpha1', maps=maps), indent=2) + '\n')
    r['maps_json_sha256'] = file_sha(mj)
    r['maps_json_equal_archived'] = r['maps_json_sha256'] == arch['map_sha256']
    cmp = {}
    for pol, mp in maps.items():
        a_set = {(n, i) for n, ix in mp.items() for i in ix}
        b_set = {(n, i) for n, ix in arch_maps['maps'][pol].items() for i in ix}
        cmp[pol] = dict(regenerated=len(a_set), archived=len(b_set), intersection=len(a_set & b_set),
                        jaccard=(len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 1.0), bitwise_equal=a_set == b_set)
    r['map_comparison'] = cmp
    r['block_statistics'] = stats
    r['block_statistics_selected_vs_archived'] = {p: (stats[p]['selected_blocks'], arch['block_statistics'][p]['selected_blocks']) for p in stats if p in arch['block_statistics']}
    save()
    # ---- run_kse_paper.elect_k on the regenerated scores
    uppers = {k: [] for k in K_VALUES}
    slices, off = {}, 0
    k2_identity = True
    for n in names:
        if not stream:
            ce, kl = tables[n][0], tables[n][1]
            for kk in K_VALUES:
                u = T.archived_upper_scores(ce, kl, kk)
                if kk == 2 and not torch.equal(u, common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))):
                    k2_identity = False
                uppers[kk].append(u)
            width = ce.shape[1]
        else:
            m128 = k2_ce[n]
            for kk in K_VALUES:
                uppers[kk].append(T.upper_bound(m128, kk, 'ce_kl').float())
            width = m128.n and m128.ce_sum.numel()
        slices[n] = (off, off + width)
        off += width
    uppers = {k: torch.cat(v) for k, v in uppers.items()}
    r['k2_equals_common_descent_scores'] = (k2_identity if not stream else 'not bitwise testable in streaming mode (float64 moments)')
    election = {}
    maps_k = {}
    for kk in K_VALUES:
        flat = uppers[kk] < 0
        election[f'k{kk}'] = dict(selected=int(flat.sum()), archived=kse['election'][f'k{kk}']['selected'])
        maps_k[f'k{kk}'] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
    order = torch.argsort(uppers[2], stable=True)[:256]
    flat = torch.zeros(uppers[2].numel(), dtype=torch.bool)
    flat[order[uppers[2][order] < 0]] = True
    maps_k['n256'] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
    election['n256'] = dict(selected=int(flat.sum()), archived=kse['election']['n256']['selected'])
    frozen_equal = True
    frozen_regenerated_equal = True
    for n in names:
        want = torch.zeros(maps_k['n256'][n].numel(), dtype=torch.bool)
        want[arch_maps['maps'][FROZEN][n]] = True
        frozen_equal &= bool(torch.equal(maps_k['n256'][n], want))
        want2 = torch.zeros(maps_k['n256'][n].numel(), dtype=torch.bool)
        want2[maps[FROZEN][n]] = True
        frozen_regenerated_equal &= bool(torch.equal(maps_k['n256'][n], want2))
    r['kse_election'] = election
    r['n256_prefix_equals_archived_frozen_fixed256'] = frozen_equal
    r['n256_prefix_equals_regenerated_fixed256'] = frozen_regenerated_equal
    # rank of archived fixed-256 tiles inside the regenerated k=2 ranking (diagnostic if not bitwise)
    rank = torch.empty_like(uppers[2], dtype=torch.long)
    rank[torch.argsort(uppers[2], stable=True)] = torch.arange(uppers[2].numel())
    arch_ranks = sorted(int(rank[slices[n][0] + i]) for n, ix in arch_maps['maps'][FROZEN].items() for i in ix)
    r['archived_fixed256_ranks_in_regenerated'] = dict(max=arch_ranks[-1] if arch_ranks else None, within_256=sum(x < 256 for x in arch_ranks),
                                                      within_512=sum(x < 512 for x in arch_ranks))
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    maps_dir = runtime.out_dir('maps')
    entries = []
    ident = dict(model_id=spec['model_id'], revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=type(model).__name__)
    cal_sha = D.manifest_sha256(fit_meta)
    for pol, mk in maps_k.items():
        grid = {n: mk[n].reshape(shapes[n][0] // 8, shapes[n][1] // 64) for n in names}
        header = MIO.build_header(protocol_id='historical', policy=dict(name=f'hist_n8_{pol}', rule='archived run_kse_paper', k=(2 if pol == 'n256' else int(pol[1:]))),
                                  model=ident, type_block=(8, 64), masks=grid, weight_shapes={n: tuple(s) for n, s in shapes.items()},
                                  source_manifest_sha256=source_manifest, calibration_manifest_sha256=cal_sha)
        digest, pth = MIO.write_map(maps_dir / f'{args.model}_historical_{pol}.mixfp4map', header, grid,
                                    provenance=dict(run_id=os.environ.get('CAMPAIGN_RUN_ID'), stream=stream))
        entries.append(dict(policy=f'hist_n8_{pol}', path=pth, sha256=digest, selected=header['totals']['selected_tiles'], total=header['totals']['total_tiles']))
    r['maps'] = entries
    torch.save(dict(names=names, slices=slices, upper_k2=uppers[2], upper_k3=uppers[3]), out / 'upper_k2_k3.pt')
    r['upper_file_sha256'] = file_sha(out / 'upper_k2_k3.pt')
    if args.compare_run:
        from campaign.policies import latest_complete_run
        other = json.loads((latest_complete_run(os.environ['CAMPAIGN_ROOT'], args.compare_run) / 'historical' / 'historical_report.json').read_text())
        o = torch.load(Path(other['maps'][0]['path']).parent.parent / 'historical' / 'upper_k2_k3.pt', weights_only=False)
        comp = {}
        for kk, key in ((2, 'upper_k2'), (3, 'upper_k3')):
            a, b = uppers[kk] < 0, o[key] < 0
            comp[f'k{kk}'] = dict(this=int(a.sum()), other=int(b.sum()), intersection=int((a & b).sum()), jaccard=float((a & b).sum() / max(1, (a | b).sum())))
        oa = torch.argsort(o['upper_k2'], stable=True)[:256]
        ta = torch.argsort(uppers[2], stable=True)[:256]
        comp['fixed256_prefix_k2'] = dict(intersection=len(set(oa.tolist()) & set(ta.tolist())), jaccard=len(set(oa.tolist()) & set(ta.tolist())) / len(set(oa.tolist()) | set(ta.tolist())))
        comp['fit_ce_vs_other'] = dict(max_abs=max(abs(a['ce'] - b['ce']) for a, b in zip(fit_losses, other['initial_fit_losses'])))
        comp['bf16_fit_nll_vs_other'] = dict(max_abs=max(abs(a - b) for a, b in zip(bf16_nll, other['bf16_fit_nll'])))
        r['same_gpu_comparison_with'] = dict(run=args.compare_run, **comp)
    r['status'] = 'complete'
    save()
    shard_dir.rmdir()
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='historical', protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
                    model_class=type(model).__name__, module_manifest_sha256=MOD.module_manifest(modules, with_weight_hash=False)[0],
                    source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend='eager', activation_quantizer='quantize_rows (causal per-token) + STE'),
        data=dict(calibration_manifest_sha256=cal_sha, evaluation_manifest_sha256=None, token_hashes=dict(math=fit_meta['math']['token_sha256'], code=fit_meta['code']['token_sha256']), overlap_audit=None),
        policies=[dict(name=e['policy'], weight_format='FourOverSix/E0M3 8x64', activation_format='quantize_rows', scale_block=16, type_block=[8, 64],
                       map_path=e['path'], map_sha256=e['sha256'], selected_tiles=e['selected'], total_tiles=e['total'], map_reloaded_for_evaluation=None) for e in entries],
        results=dict(raw_outputs=[str(out / 'historical_report.json'), str(mj), str(wm), str(out / 'upper_k2_k3.pt')],
                     summary={k: r[k] for k in ('bf16_fit_nll_vs_archived', 'initial_fit_losses_vs_archived', 'kse_election', 'n256_prefix_equals_archived_frozen_fixed256',
                                                'maps_json_equal_archived', 'weight_mse_equal_archived', 'shard_sha256_equal_archived')},
                     uncertainty={}, attempted_endpoints=['scores', 'fixed256', 'k-elections', 'shard hashes'], missing_endpoints=[]),
        logs=[], failures=[]))
    print('HISTORICAL ' + json.dumps({k: r[k] for k in ('kse_election', 'n256_prefix_equals_archived_frozen_fixed256', 'maps_json_equal_archived')}, default=str), flush=True)


def AutoTokenizerFrom(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def derive_maps_streamed(names, shapes, acc, subsets):
    """derive_maps with subset moments assembled from streamed float64 prefix sums (D-H5)."""
    vectors = {p: {'upper': [], 'diagonal': [], 'global_indices': []} for p in subsets}
    slices, offset = {}, 0
    for n in names:
        width = (shapes[n][0] // 8) * (shapes[n][1] // 64)
        slices[n] = (offset, offset + width)
        for policy, ids in subsets.items():
            if policy.startswith('math') and not policy.startswith('math_code'):
                parts = [('math', int(policy[4:]))]
            elif policy.startswith('code'):
                parts = [('code', int(policy[4:]))]
            else:
                half = int(policy[len('math_code'):]) // 2
                parts = [('math', half), ('code', half)]
            cs = sum(acc[n][p]['cs'] for p in parts); cq = sum(acc[n][p]['cq'] for p in parts)
            ks = sum(acc[n][p]['ks'] for p in parts); kq = sum(acc[n][p]['kq'] for p in parts)
            fq = sum(acc[n][p]['fq'] for p in parts)
            cnt = len(ids)
            def up(s, q):
                var = ((q - s * s / cnt) / (cnt - 1)).clamp_min(0)
                return s / cnt + 2 * var.sqrt() / math.sqrt(cnt)
            upper = torch.maximum(up(cs, cq), up(ks, kq))
            diagonal = 511 * fq / cnt
            eligible = (upper < 0).nonzero().flatten()
            vectors[policy]['upper'].append(upper[eligible])
            vectors[policy]['diagonal'].append(diagonal[eligible])
            vectors[policy]['global_indices'].append(eligible + offset)
        offset += width
    maps, statistics = {}, {}
    for setting, parts in vectors.items():
        upper = torch.cat(parts['upper']); diagonal = torch.cat(parts['diagonal']); gi = torch.cat(parts['global_indices'])
        chosen, stats = adaptive_prefix(upper, diagonal)
        modes = {f'adaptive_{setting}': (chosen, stats)}
        fixed = torch.argsort(upper, stable=True)[:256]
        modes[f'fixed256_{setting}'] = (fixed, dict(selected_blocks=len(fixed), eligible_blocks=len(upper), count_cap=256))
        for policy, (indices, st) in modes.items():
            cg = gi[indices].sort().values
            maps[policy] = {n: (cg[(cg >= lo) & (cg < hi)] - lo).tolist() for n, (lo, hi) in slices.items()}
            statistics[policy] = dict(**st, calibration_sequences=len(subsets[setting]), total_type_blocks=offset,
                                      selected_fraction=len(indices) / offset, selected_weights=len(indices) * 512, selected_scale_blocks=len(indices) * 32)
    return maps, statistics


if __name__ == '__main__':
    main()
