"""Predict legal 8x64 type switches from final-loss sensitivity.

Run ONLY in a Slurm allocation. Forward quantization is unchanged; activation
backward uses an explicitly labelled identity STE. Discrete interventions and
held-out NLL, not this surrogate, determine whether the prediction works.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time

if __name__ == '__main__' and not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit through Slurm; even CPU work must not run on the login node.')

import torch
import torch.nn.functional as F

from quantize import QuantConfig, collect_importance
from quantize.quantizer import quant_act, quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
from utils import load_model_and_tokenizer, set_seed


class IdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q):
        return q

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste_quant_act(x, config):
    return IdentityBackward.apply(x, quant_act(x, config))


@contextlib.contextmanager
def activation_ste():
    import models.qmodule_llama as llama
    import models.qmodule_qwen3 as qwen
    modules = [llama, qwen]
    previous = [m.quant_act for m in modules]
    try:
        for m in modules:
            m.quant_act = ste_quant_act
        yield
    finally:
        for m, old in zip(modules, previous):
            m.quant_act = old


def tile_sum(x):
    m, k = x.shape
    assert m % 8 == 0 and k % 64 == 0, (m, k)
    return x.reshape(m // 8, 8, k // 64, 64).sum(dim=(1, 3))


def expand_mask(mask):
    return mask.repeat_interleave(8, 0).repeat_interleave(64, 1)


def atomic_json(path, obj):
    with open(str(path) + '.tmp', 'w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(str(path) + '.tmp', path)


def mean_se(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {'mean': x.mean().item(),
            'se': (x.std(unbiased=True) / math.sqrt(len(x))).item() if len(x) > 1 else None,
            'n': len(x)}


def paired(candidate, baseline):
    assert len(candidate) == len(baseline)
    out = mean_se([a - b for a, b in zip(candidate, baseline)])
    out['ppl_delta'] = math.exp(sum(candidate) / len(candidate)) - math.exp(sum(baseline) / len(baseline))
    return out


def loss(model, ids, use_cache=False):
    ids = ids.to(model.device)
    logits = model(ids, use_cache=use_cache).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate(model, batches, label, use_cache=False):
    out = []
    for i, ids in enumerate(batches):
        out.append(loss(model, ids, use_cache=use_cache).item())
        if (i + 1) % 16 == 0:
            print(f'EVAL {label} {i+1}/{len(batches)} nll={sum(out)/len(out):.6f}', flush=True)
    return out


def data_splits(tokenizer, seq, nfit, nval, seed, dataset_name='wikitext'):
    from datasets import load_dataset
    data = load_dataset(dataset_name, 'wikitext-2-raw-v1', split='train')
    ids = tokenizer('\n\n'.join(data['text']), return_tensors='pt').input_ids
    windows = list(ids[:, :ids.shape[1] // seq * seq].split(seq, dim=1))
    # Entire contiguous thirds are separated before sampling, preventing overlapping windows.
    third = len(windows) // 3
    assert third >= max(nfit, nval)
    rng = random.Random(seed)
    # Preserve the original 16-sequence experiment exactly. Larger budgets
    # append independent windows without changing its validation/probe draws.
    first_indices = rng.sample(range(third), min(nfit, 16))
    fit = [windows[i] for i in first_indices]
    val = rng.sample(windows[third:2*third], nval)
    probe = rng.sample(windows[2*third:], min(8, nval))
    if nfit > 16:
        remaining = [i for i in range(third) if i not in set(first_indices)]
        random.Random(seed+1).shuffle(remaining)
        fit.extend(windows[i] for i in remaining[:nfit-16])
    def digest(bs):
        return hashlib.sha256(torch.cat(bs, 1).numpy().tobytes()).hexdigest()
    return fit, val, probe, {k: digest(v) for k, v in [('fit', fit), ('val', val), ('probe', probe)]}


@torch.no_grad()
def install(modules, base, alt, masks=None):
    for name, mod in modules.items():
        b = base[name].to(mod.weight.device)
        if masks is not None and name in masks:
            a = alt[name].to(mod.weight.device)
            b = torch.where(expand_mask(masks[name].to(b.device)), a, b)
        mod.weight.copy_(b)


def score_tiles(model, modules, base, alt, batches):
    """One backward per sequence, streaming per-module weight gradients to tile scores.

    Parameters stay frozen. The input embedding output starts the autograd graph.
    No full-model parameter-gradient buffer or full Hessian is allocated.
    """
    sums, squares, handles = {}, {}, []
    for p in model.parameters():
        p.requires_grad_(False)
    handles.append(model.get_input_embeddings().register_forward_hook(
        lambda _m, _i, o: o.requires_grad_(True)))
    counts = {n: 0 for n in modules}

    def forward_for(name):
        def forward(_m, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            def backward(dy):
                with torch.no_grad():
                    grad_w = dy.reshape(-1, dy.shape[-1]).float().T @ x.float()
                    delta = alt[name].to(x.device).float() - base[name].to(x.device).float()
                    s = tile_sum(grad_w * delta).cpu()
                    assert torch.isfinite(s).all(), name
                    if name not in sums:
                        sums[name] = s.clone()
                        squares[name] = s.square()
                    else:
                        sums[name].add_(s)
                        squares[name].add_(s.square())
                    counts[name] += 1
                return dy
            output.register_hook(backward)
        return forward

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(forward_for(name)))
    fit_losses = []
    try:
        with activation_ste():
            for i, ids in enumerate(batches):
                start = time.time()
                l = loss(model, ids)
                fit_losses.append(l.item())
                l.backward()
                del l
                print(f'BACKWARD {i+1}/{len(batches)} {time.time()-start:.1f}s nll={fit_losses[-1]:.6f}', flush=True)
    finally:
        for h in handles:
            h.remove()
    assert all(c == len(batches) for c in counts.values()), counts
    n = len(batches)
    means = {k: s / n for k, s in sums.items()}
    ses = {k: ((squares[k] - s.square()/n).clamp_min(0) / (n*(n-1))).sqrt()
           for k, s in sums.items()}
    return means, ses, fit_losses


def selected_summary(masks, means):
    return {'tiles': sum(int(m.sum()) for m in masks.values()),
            'fraction': sum(int(m.sum()) for m in masks.values()) / sum(m.numel() for m in masks.values()),
            'predicted_delta_nll': sum(float(means[n][m].sum()) for n, m in masks.items())}


def intervention_masks(means, count, beneficial):
    names = list(means)
    flat = torch.cat([means[n].flatten() for n in names])
    selected = torch.zeros_like(flat, dtype=torch.bool)
    selected[torch.topk(flat, min(count, flat.numel()), largest=not beneficial).indices] = True
    out, offset = {}, 0
    for n in names:
        out[n] = selected[offset:offset+means[n].numel()].reshape_as(means[n])
        offset += means[n].numel()
    return out


def trust_masks(means, ses, budget=0.1):
    """Fixed global budget in predicted NLL units, not a per-model tile quota.

    Only sequence-stable improvements are eligible. Rank by predicted benefit,
    then retain the largest prefix whose summed predicted reduction is <= budget.
    This bounds extrapolation of the first-order model; it is not a loss bound.
    """
    names = list(means)
    flat = torch.cat([means[n].flatten() for n in names])
    se = torch.cat([ses[n].flatten() for n in names])
    eligible = torch.nonzero(flat + 2*se < 0).flatten()
    order = eligible[torch.argsort(flat[eligible])]
    selected = torch.zeros_like(flat, dtype=torch.bool)
    selected[order[(-flat[order]).double().cumsum(0) <= budget]] = True
    out, offset = {}, 0
    for n in names:
        out[n] = selected[offset:offset+means[n].numel()].reshape_as(means[n])
        offset += means[n].numel()
    return out


def matched_random(masks, seed):
    rng = torch.Generator().manual_seed(seed)
    out = {}
    for n, m in masks.items():
        r = torch.zeros(m.numel(), dtype=torch.bool)
        count = int(m.sum())
        if count:
            r[torch.randperm(m.numel(), generator=rng)[:count]] = True
        out[n] = r.reshape_as(m)
    return out


def calibrated_trust_masks(model, modules, base, alt, means, ses, fit, baseline_fit):
    """Backtrack using FIT loss only; validation never chooses the step size."""
    history = []
    for attempt in range(8):
        budget = 0.1 * 0.5**attempt
        masks = trust_masks(means, ses, budget)
        summary = selected_summary(masks, means)
        if not summary['tiles']:
            break
        install(modules, base, alt, masks)
        values = evaluate(model, fit, f'trust-fit/{attempt}')
        delta = paired(values, baseline_fit)
        ratio = delta['mean'] / summary['predicted_delta_nll']
        accepted = ratio >= 0.25 and delta['mean'] + 2*delta['se'] < 0
        entry = {**summary, 'budget': budget, 'actual_fit': delta,
                 'actual_over_predicted': ratio, 'accepted': accepted}
        history.append(entry)
        print(f'TRUST STEP {entry}', flush=True)
        if accepted:
            return masks, history
    install(modules, base, alt)
    return {n: torch.zeros_like(s, dtype=torch.bool) for n, s in means.items()}, history


def export_type_map(path, model_name, masks, fit_hash, model_commit=None, rule='gradient_trust_0p1', weight_baseline='nvfp4'):
    value = {'model': model_name, 'model_commit': model_commit, 'rule': rule,
             'weight_type_block': [8, 64], 'scale_block': 16, 'alpha': 1.,
             'default': 'E2M1', 'fit_sha256': fit_hash,
             'modules': {n: {'tile_grid_shape': list(v.shape),
                             'e0m3_flat_indices': v.flatten().nonzero().flatten().tolist()}
                         for n, v in masks.items() if v.any()}}
    assert weight_baseline in ('nvfp4', 'nvfp4_4over6')
    if weight_baseline == 'nvfp4_4over6':
        value.pop('alpha')
        value.update(baseline_weight_dtype=weight_baseline, e2m1_alphas=[1., 1.5], e0m3_alpha=1.)
    atomic_json(path, value)


@torch.no_grad()
def apply_type_map(model, type_map):
    """Apply an exported map to PRISTINE model weights, never already-quantized weights.

    This is an offline weight rewrite. Activations must separately use the
    nvfp4_4over6 configuration used during calibration. No runtime reordering.
    """
    assert type_map.get('scope', 'all_linear_except_head') == 'all_linear_except_head', 'Use the architecture-specific map loader.'
    assert type_map['weight_type_block'] == [8, 64] and type_map['scale_block'] == 16
    assert type_map['default'] == 'E2M1'
    baseline = type_map.get('baseline_weight_dtype', 'nvfp4')
    assert baseline in ('nvfp4', 'nvfp4_4over6')
    if baseline == 'nvfp4':
        assert type_map['alpha'] == 1.
    else:
        assert type_map['e2m1_alphas'] == [1., 1.5] and type_map['e0m3_alpha'] == 1.
    commit = type_map.get('model_commit')
    if commit is not None:
        assert getattr(model.config, '_commit_hash', None) == commit, 'Source model revision differs.'
    entries = type_map['modules']
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and 'head' not in n}
    assert set(entries) <= set(modules), 'Unknown matrix in type map.'
    for n, mod in modules.items():
        w = mod.weight.detach()
        q = (quant_nvfp4_4over6 if baseline == 'nvfp4_4over6' else quant_nvfp4)(w, 4, 16)
        if n in entries:
            shape = [w.shape[0]//8, w.shape[1]//64]
            assert list(entries[n]['tile_grid_shape']) == shape
            assert w.shape[0] % 8 == 0 and w.shape[1] % 64 == 0
            mask = torch.zeros(shape[0]*shape[1], dtype=torch.bool, device=w.device)
            indices = entries[n]['e0m3_flat_indices']
            assert len(indices) == len(set(indices))
            assert all(0 <= i < mask.numel() for i in indices)
            mask[indices] = True
            alt = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            q = torch.where(expand_mask(mask.reshape(shape)), alt, q)
        mod.weight.copy_(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--fit', type=int, default=16)
    ap.add_argument('--val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--weight-baseline', choices=['nvfp4', 'nvfp4_4over6'], default='nvfp4')
    ap.add_argument('--test-limit', type=int, default=0)
    ap.add_argument('--c4-samples', type=int, default=64)
    ap.add_argument('--skip-final', action='store_true')
    ap.add_argument('--resume-scores', help='Reuse a previous scores.pt with identical model, data, and seed.')
    ap.add_argument('--policy-set', choices=['initial', 'sparse'], default='initial')
    ap.add_argument('--skip-interventions', action='store_true')
    ap.add_argument('--final-policies', default='', help='Optional comma-separated subset for a predeclared replication.')
    ap.add_argument('--calibrate-only', action='store_true',
                    help='Export the frozen sparse type map without control sweeps or final test evaluation.')
    ap.add_argument('--validate-selected', action='store_true',
                    help='With --calibrate-only, estimate the chosen map\'s loss change on separate calibration windows.')
    ap.add_argument('--backtrack-proposal', action='store_true',
                    help='Calibrate the discrete step using fit loss before independent validation.')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit through Slurm; this program must not run on the login node.')
    assert args.fit >= 2
    assert args.val >= 2 and args.seq_len >= 2
    assert args.c4_samples >= 0 and args.test_limit >= 0
    assert not args.validate_selected or args.calibrate_only
    assert not args.backtrack_proposal or args.calibrate_only or args.policy_set == 'sparse'
    assert args.weight_baseline == 'nvfp4' or args.backtrack_proposal, 'FourOverSix uses the frozen sparse backtracking path.'
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, 'report.json')):
        raise FileExistsError('Use a new --out directory; completed measurements must not be overwritten.')
    torch.set_num_threads(min(12, int(os.environ.get('SLURM_CPUS_PER_TASK', 8))))
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(args.seed)
    report = {'args': vars(args), 'job_id': os.environ['SLURM_JOB_ID'],
              'method': 'NLL gradient at W4A4 NVFP4 reference; identity activation STE; alpha=1; 8x64',
              'results': {}, 'interventions': {}}
    report['source_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    if args.weight_baseline == 'nvfp4_4over6':
        report['method'] = 'NLL gradient at W4A4 FourOverSix reference; identity activation STE; E0M3 alpha=1; 8x64'
    assert torch.cuda.device_count() == 1, 'Each sensitivity job must expose exactly one allocated GPU.'
    report['gpu'] = torch.cuda.get_device_name(0)
    assert 'H100' in report['gpu'], report['gpu']
    dest = os.path.join(args.out, 'report.json')
    def save():
        atomic_json(dest, report)
    config = QuantConfig(w_bits=4, w_dtype=args.weight_baseline, a_bits=16, a_dtype='fp16',
                         w_groupsize=16, a_groupsize=16, w_type_block='8x64')
    print(f'LOAD {args.model}', flush=True)
    start = time.time()
    model, tok = load_model_and_tokenizer(args.model, config, device_map='cuda:0')
    report['model_load_seconds'] = time.time() - start
    report['model_commit'] = getattr(model.config, '_commit_hash', None)
    model.config.use_cache = False
    fit, val, probe, hashes = data_splits(tok, args.seq_len, args.fit, args.val, args.seed)
    report['data_sha256'] = hashes
    start = time.time()
    skip_controls = args.calibrate_only or args.backtrack_proposal
    imp = {} if skip_controls else collect_importance(model, fit[:4], device=model.device)
    report['importance_seconds'] = time.time() - start
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and 'head' not in n}
    base, alt = {}, {}
    controls = {} if skip_controls else {'hess_h1.5': {}, 'hess_impg16_h10': {}}
    print(f'CANDIDATES {len(modules)} matrices', flush=True)
    start = time.time()
    with torch.no_grad():
        for i, (n, mod) in enumerate(modules.items()):
            w = mod.weight.detach()
            b = (quant_nvfp4_4over6 if args.weight_baseline == 'nvfp4_4over6' else quant_nvfp4)(w, 4, 16)
            b2 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='never')
            if args.weight_baseline == 'nvfp4':
                assert torch.equal(b, b2), f'NVFP4 identity failed: {n}'
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            base[n], alt[n] = b.cpu(), a.cpu()
            control_specs = [] if skip_controls else [('hess_h1.5', 1.5, 0), ('hess_impg16_h10', 10., 16)]
            for label, margin, gran in control_specs:
                q = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1',
                                  importance=imp[n].to(w.device), elect='harm', margin=margin, imp_gran=gran)
                # Recover a tile's elected candidate; identical candidates need no switch.
                mask = tile_sum((q != b).float()) > 0
                assert torch.equal(q, torch.where(expand_mask(mask), a, b)), n
                controls[label][n] = mask.cpu()
                del q
            mod.weight.copy_(b)
            if (i+1) % 32 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
    del imp, w, a, b, b2
    report['candidate_and_controls_seconds'] = time.time() - start
    config.a_bits, config.a_dtype = 4, 'nvfp4_4over6'
    torch.cuda.empty_cache()
    # Forward equality must be exact; the STE changes only the backward.
    with torch.no_grad():
        plain = loss(model, fit[0]).item()
        with activation_ste():
            surrogate = loss(model, fit[0]).item()
        cache_enabled = loss(model, fit[0], use_cache=True).item()
    assert plain == surrogate, (plain, surrogate)
    report['ste_forward_equal'] = True
    report['cache_enabled_minus_disabled_nll_first_fit'] = cache_enabled - plain
    start = time.time()
    if args.resume_scores:
        cached = torch.load(args.resume_scores, map_location='cpu', weights_only=True)
        assert cached['hashes'] == hashes
        assert cached['args'].get('weight_baseline', 'nvfp4') == args.weight_baseline
        if cached.get('model_commit') is not None:
            assert cached['model_commit'] == report['model_commit'], 'Cached scores use different source weights.'
        for key in ['model', 'seq_len', 'fit', 'val', 'seed']:
            assert cached['args'][key] == vars(args)[key], key
        assert set(cached['means']) == set(modules)
        means, ses = cached['means'], cached['ses']
        fit_losses = cached.get('fit_losses')
    else:
        means, ses, fit_losses = score_tiles(model, modules, base, alt, fit)
    report['scores_seconds'] = time.time() - start
    report['scores_reused'] = bool(args.resume_scores)
    torch.save({'means': means, 'ses': ses, 'controls': controls, 'hashes': hashes,
                'args': vars(args), 'fit_losses': fit_losses, 'model_commit': report['model_commit']},
               os.path.join(args.out, 'scores.pt'))
    trust_label = 'gradient_trust_backtracking' if args.backtrack_proposal else 'gradient_trust_0p1'
    trust = None
    if args.backtrack_proposal:
        baseline_fit = evaluate(model, fit, 'trust-fit/baseline')
        trust, report['proposal_fit_calibration'] = calibrated_trust_masks(
            model, modules, base, alt, means, ses, fit, baseline_fit)
        install(modules, base, alt)
        save()
    if args.calibrate_only:
        masks = trust if trust is not None else trust_masks(means, ses)
        export_type_map(os.path.join(args.out, 'candidate_type_map.json'), args.model, masks, hashes['fit'], report['model_commit'], trust_label, args.weight_baseline)
        report['proposal_selection'] = selected_summary(masks, means)
        report['export_decision'] = 'unvalidated'
        if args.validate_selected:
            baseline_val = evaluate(model, val, 'forecast/baseline')
            install(modules, base, alt, masks)
            chosen_val = evaluate(model, val, 'forecast/selected')
            forecast = paired(chosen_val, baseline_val)
            mu, se = forecast['mean'], forecast['se']
            forecast['predicted_relative_ppl_change_percent'] = 100*math.expm1(mu)
            forecast['relative_change_interval_percent'] = [100*math.expm1(mu-2*se), 100*math.expm1(mu+2*se)]
            forecast['improvement_supported'] = mu + 2*se < 0
            report['validation_forecast'] = forecast
            report['validation_baseline_nll'] = baseline_val
            report['validation_proposal_nll'] = chosen_val
            report['export_decision'] = 'accepted' if forecast['improvement_supported'] else 'fallback_'+args.weight_baseline
            if not forecast['improvement_supported']:
                masks = {n: torch.zeros_like(m) for n, m in masks.items()}
                install(modules, base, alt)
        export_type_map(os.path.join(args.out, 'type_map.json'), args.model, masks, hashes['fit'], report['model_commit'], trust_label, args.weight_baseline)
        report['selection'] = selected_summary(masks, means)
        report['calibration_only'] = True
        report['complete'] = True
        report['gpu_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 2**30
        save()
        print(f'CALIBRATED {report["selection"]}; decision={report["export_decision"]}; exported {args.out}/type_map.json', flush=True)
        return
    report['backward_fit_nll'] = fit_losses
    baseline = {'fit': evaluate(model, fit, 'baseline/fit'), 'val': evaluate(model, val, 'baseline/val'),
                'probe': evaluate(model, probe, 'baseline/probe')}
    report['grad_vs_nograd_forward_max_delta'] = (max(abs(a-b) for a, b in zip(fit_losses, baseline['fit']))
                                                 if fit_losses is not None else None)
    report['baseline'] = baseline
    report['per_projection_score'] = {
        n: {'negative_fraction': float((s < 0).float().mean()),
            'l1': float(s.abs().sum()), 'sum': float(s.sum())} for n, s in means.items()}
    save()
    # Fixed, bounded interventions validate sign and approximation before wholesale switching.
    interventions = [] if args.skip_interventions else [
        ('beneficial_256', True, 256), ('harmful_256', False, 256),
        ('beneficial_4096', True, 4096), ('harmful_4096', False, 4096)]
    for label, beneficial, count in interventions:
        masks = intervention_masks(means, count, beneficial)
        install(modules, base, alt, masks)
        entry = selected_summary(masks, means)
        for split, batches in [('fit', fit), ('probe', probe)]:
            vals = evaluate(model, batches, label+'/'+split)
            entry[split] = paired(vals, baseline[split][:len(vals)])
            entry[split+'_nll'] = vals
        report['interventions'][label] = entry
        print(f'INTERVENTION {label} ' + str({k: v for k, v in entry.items() if not k.endswith('_nll')}), flush=True)
        save()
    if args.policy_set == 'initial':
        policies = {'gradient_sign': {n: s < 0 for n, s in means.items()},
                    'gradient_confident': {n: s + 2*ses[n] < 0 for n, s in means.items()}, **controls}
    else:
        trust = trust if trust is not None else trust_masks(means, ses)
        export_type_map(os.path.join(args.out, 'type_map.json'), args.model, trust, hashes['fit'], report['model_commit'], trust_label, args.weight_baseline)
        policies = {trust_label: trust}
        if not args.backtrack_proposal:
            policies.update({'matched_random': matched_random(trust, args.seed+1), **controls})
    for label, masks in policies.items():
        install(modules, base, alt, masks)
        entry = selected_summary(masks, means)
        entry['per_projection'] = {
            n: {'tiles': int(m.sum()), 'predicted_delta_nll': float(means[n][m].sum())}
            for n, m in masks.items() if m.any()}
        for split, batches in [('fit', fit), ('val', val)]:
            vals = evaluate(model, batches, label+'/'+split)
            entry[split] = paired(vals, baseline[split])
            entry[split+'_nll'] = vals
        report['results'][label] = entry
        print(f'POLICY {label} ' + str({k: entry[k] for k in
              ['tiles', 'fraction', 'predicted_delta_nll', 'fit', 'val']}), flush=True)
        save()
    # Choose solely on validation. Test sets never influence selection.
    winner = min(['baseline'] + list(policies), key=lambda k:
                 0 if k == 'baseline' else report['results'][k]['val']['mean'])
    report['validation_winner'] = winner
    eligible = ['baseline'] + [k for k in policies if
        report['results'][k]['val']['mean'] + 2*report['results'][k]['val']['se'] < 0]
    report['validation_guarded_winner'] = min(eligible, key=lambda k:
        0 if k == 'baseline' else report['results'][k]['val']['mean'])
    torch.save(policies, os.path.join(args.out, 'policies.pt'))
    save()
    if not args.skip_final:
        from run_ppl_sweep import build_wikitext, build_c4
        test = build_wikitext(tok, args.seq_len)
        if args.test_limit:
            test = test[:, :args.seq_len*args.test_limit]
        test = list(test[:, :test.shape[1]//args.seq_len*args.seq_len].split(args.seq_len, 1))
        datasets = {'wikitext': test}
        if args.c4_samples:
            c4 = build_c4(tok, args.seq_len, args.c4_samples)
            datasets['c4'] = list(c4.split(args.seq_len, 1))
        report['final'] = {}
        # Report both predeclared new rules even when validation rejects them.
        final_labels = (args.final_policies.split(',') if args.final_policies else ['baseline'] + list(policies))
        assert final_labels[0] == 'baseline' and set(final_labels) <= {'baseline', *policies}
        for label in final_labels:
            install(modules, base, alt, policies.get(label))
            report['final'][label] = {}
            for ds, batches in datasets.items():
                vals = evaluate(model, batches, label+'/'+ds)
                entry = {'nll': vals, 'ppl': math.exp(sum(vals)/len(vals))}
                if label != 'baseline':
                    entry['vs_baseline'] = paired(vals, report['final']['baseline'][ds]['nll'])
                report['final'][label][ds] = entry
                print(f'FINAL {label} {ds} ppl={entry["ppl"]:.6f} delta={entry.get("vs_baseline")}', flush=True)
                save()
        if trust_label in policies:
            # Isolate numerical execution-path sensitivity on the same validation
            # windows, without changing the map or the primary cache-off test.
            install(modules, base, alt)
            cached_base = evaluate(model, val, 'cache-enabled/baseline', use_cache=True)
            install(modules, base, alt, policies[trust_label])
            cached_selected = evaluate(model, val, 'cache-enabled/selected', use_cache=True)
            report['cache_robustness'] = {
                'baseline_nll': cached_base, 'selected_nll': cached_selected,
                'selected_vs_baseline': paired(cached_selected, cached_base)}
            print(f'CACHE CHECK {report["cache_robustness"]["selected_vs_baseline"]}', flush=True)
    report['complete'] = True
    report['gpu_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 2**30
    save()
    print(f'DONE {dest}', flush=True)


if __name__ == '__main__':
    main()
