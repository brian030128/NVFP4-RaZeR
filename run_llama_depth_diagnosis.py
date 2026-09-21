"""Fixed-layout depth probe; diagnostic transfer, never a trained early-layer map.

Compare the STE directional prediction with finite CE changes. A second forward
holds each activation quantizer's baseline residual fixed, q0 + (x - x0), so its
baseline is bitwise exact and its local derivative is identity. This is an
artificial control, not an implementable FP4 policy or a new quality candidate.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from llama_diagnosis_common import ROOT, checked_plan, controls, records, verified_model, write
from run_c4_frozen import digest_file
from run_fine_row_research import arm_gradient
from run_reorder_replay import paired_bounds
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.task_reorder import original_order_weight_reference
from run_task_reorder_eval import mix_coarse


def ce(logits, ids):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def main():
    plan, prior = checked_plan()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layer', type=int, choices=(0, 15, 31), required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = ROOT / (f'depth_smoke_{args.layer}' if args.smoke else f'depth_{args.layer}')
    out.mkdir(exist_ok=False)
    layouts, masks = controls(plan, prior)
    all_records = records()
    chosen = []
    for split, domain, count in [('fit', 'math', 4), ('fit', 'code', 4),
                                  ('fresh', 'math', 8), ('fresh', 'code', 8), ('fresh', 'general', 8)]:
        chosen += [r for r in all_records if r['split'] == split and r['source'] == domain][:count]
    assert len(chosen) == 32
    if args.smoke:
        chosen = chosen[:2]
    model, modules = verified_model(prior)
    names = [f'model.layers.{args.layer}.mlp.{part}_proj' for part in ('gate', 'up', 'down')]
    prepared = {}; directions = {}; diagnostics = {}
    with torch.no_grad():
        for name, module in modules.items():
            base = quant_nvfp4_4over6(module.weight, 4, 16)
            if name in names or masks[name].any():
                alt = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                raw = mix_coarse(base, alt, masks[name])
                if name in names:
                    template = layouts[name.replace(f'.{args.layer}.', '.31.')]
                    candidate = original_order_weight_reference(base, alt, template)
                    prepared[name] = {0.: raw, 1.: candidate,
                        .25: (raw.float() + .25 * (candidate.float() - raw.float())).to(raw.dtype)}
                    directions[name] = {scale: prepared[name][scale].float() - raw.float() for scale in (.25, 1.)}
                    diagnostics[name] = dict(changed_weights=int((candidate != raw).sum()),
                        delta_squared_norm=float(directions[name][1.].square().sum()),
                        source='Final-layer frozen permutation/mask transferred diagnostically; not optimized here')
                base = raw
                del alt, raw
            module.weight.copy_(base)
        del base
    policies = {'quarter_joint': (.25, (0, 1, 2)), 'joint': (1., (0, 1, 2)),
                'gate': (1., (0,)), 'up': (1., (1,)), 'down': (1., (2,))}
    def install(scale=0., selected=()):
        with torch.no_grad():
            for j, name in enumerate(names):
                modules[name].weight.copy_(prepared[name][scale if j in selected else 0.])
    state = {'mode': 'capture'}; saved = {}
    def quant_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            if state['mode'] == 'frozen':
                x0, q0 = saved[name]
                value = q0 + (x - x0)
            else:
                value = quant_nvfp4_4over6(x.detach(), 4, 16)
                if state['mode'] == 'capture':
                    assert name not in saved
                    saved[name] = (x.detach().clone(), value.detach().clone())
                elif state['mode'] == 'ste':
                    value = value + (x - x.detach())
            return (value, *inputs[1:])
        return hook
    handles = [m.register_forward_pre_hook(quant_hook(n)) for n, m in modules.items()]
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], layer=args.layer,
        smoke=args.smoke, plan_sha256=digest_file(ROOT / 'plan.json'),
        diagnostic_only=True, no_selection=True, template_transfer=args.layer != 31,
        weight_diagnostics=diagnostics, sequences=[], audits=dict(frozen_baseline=0, raw_restore=0, ste_baseline=0))
    write(out / 'report.json', report)
    for index, row in enumerate(chosen):
        ids = row['ids'].cuda()
        install(); saved.clear(); state['mode'] = 'capture'
        with torch.no_grad():
            baseline = model(input_ids=ids, use_cache=False).logits
            raw_ce = float(ce(baseline, ids))
            assert set(saved) == set(modules)
            state['mode'] = 'frozen'
            check = model(input_ids=ids, use_cache=False).logits
            assert torch.equal(check, baseline), 'Frozen residual baseline is not exact'
            report['audits']['frozen_baseline'] += 1
            del check
        prediction = {}
        def gradient_hook(name):
            def hook(module, inputs, output):
                x = inputs[0].detach()
                def backward(dy):
                    prediction[name] = {scale: float((dy.detach().float() * F.linear(x.float(), d)).sum())
                                        for scale, d in directions[name].items()}
                return arm_gradient(output, backward)
            return hook
        gradients = [modules[n].register_forward_hook(gradient_hook(n)) for n in names]
        state['mode'] = 'ste'
        ste_logits = model(input_ids=ids, use_cache=False).logits
        ste_ce = ce(ste_logits, ids)
        assert torch.equal(ste_logits, baseline), 'STE forward differs from the measured baseline'
        report['audits']['ste_baseline'] += 1
        ste_ce.backward()
        assert set(prediction) == set(names)
        for h in gradients:
            h.remove()
        del ste_logits, ste_ce
        result = dict(source=row['source'], split=row['split'], document_sha256=row['document_sha256'],
            token_sha256=row['token_sha256'], raw_ce=raw_ce, policies={})
        with torch.no_grad():
            for policy, (scale, selected) in policies.items():
                install(scale, selected)
                values = dict(predicted=sum(prediction[names[j]][scale] for j in selected))
                for mode in ('actual', 'frozen'):
                    state['mode'] = mode
                    logits = model(input_ids=ids, use_cache=False).logits
                    values[mode] = float(ce(logits, ids)) - raw_ce
                    del logits
                result['policies'][policy] = values
            install(); state['mode'] = 'actual'
            check = model(input_ids=ids, use_cache=False).logits
            assert torch.equal(check, baseline), 'Raw restore differs'
            report['audits']['raw_restore'] += 1
            del check, baseline
        saved.clear(); report['sequences'].append(result)
        write(out / 'report.json', report)
        print('DEPTH', args.layer, index + 1, '/', len(chosen), flush=True)
    for h in handles:
        h.remove()
    report['groups'] = {}
    for group in ('all', 'fit', 'fresh', 'general'):
        rows = [r for r in report['sequences'] if group == 'all' or r['split'] == group or (group == 'general' and r['source'] == 'general')]
        if len(rows) < 2:
            continue
        summary = {}
        for policy in policies:
            values = [r['policies'][policy] for r in rows]
            summary[policy] = {key: paired_bounds([v[key] for v in values], [0.] * len(values), k=2.)
                               for key in ('predicted', 'actual', 'frozen')}
            for mode in ('actual', 'frozen'):
                errors = [abs(v[mode] - v['predicted']) for v in values]
                summary[policy][mode + '_prediction_mae'] = sum(errors) / len(errors)
                summary[policy][mode + '_sign_agreement'] = sum((v[mode] < 0) == (v['predicted'] < 0) for v in values) / len(values)
        summary['interaction'] = {mode: paired_bounds([
            r['policies']['joint'][mode] - sum(r['policies'][p][mode] for p in ('gate', 'up', 'down')) for r in rows],
            [0.] * len(rows), k=2.) for mode in ('actual', 'frozen')}
        report['groups'][group] = summary
    checked_plan()
    report['status'] = 'complete'
    write(out / 'report.json', report)
    print('DEPTH COMPLETE', args.layer, flush=True)


if __name__ == '__main__':
    main()
