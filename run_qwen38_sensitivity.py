"""Frozen MixFP4 calibration on the native hybrid Qwen3.8 text pathway."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit all compute through Slurm.')
for key, subdir in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                    ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], subdir))
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4, quant_nvfp4_4over6, quant_mix_4_6
import analyze_task_sensitivity as task
from probe_qwen38 import NativeActivationQuantization, native_loss, native_wikitext, text_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--fit', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bf16-reference', action='store_true')
    ap.add_argument('--weight-baseline', choices=['nvfp4', 'nvfp4_4over6'], default='nvfp4')
    args = ap.parse_args()
    assert args.fit >= 2
    assert torch.cuda.device_count() == 2
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'report.json').exists(), 'Use a fresh output directory.'
    probe = json.loads(Path('results/task_sensitivity_qwen38_probe/report.json').read_text())
    assert probe['complete'] and probe['real_forward_equal'] and probe['forward_backward_nll_equal']
    assert probe['transformers'] == transformers.__version__
    report = {'args': {**vars(args), 'model': probe['model'], 'seq_len': 2048, 'val': 16},
              'model_commit': probe['model_commit'], 'transformers': transformers.__version__,
              'torch': torch.__version__, 'job_id': os.environ['SLURM_JOB_ID'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                     for p in ['probe_qwen38.py', 'analyze_task_sensitivity.py', 'quantize/quantizer.py']},
              'quantization_scope': 'Text nn.Linear weights and inputs, excluding head; native recurrent/conv/norm/vision.',
              'complete': False, 'results': {}, 'final': {}}
    def save():
        task.atomic_json(out/'report.json', report)
    save()
    start = time.time()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        probe['model'], revision=probe['model_commit'], dtype=torch.bfloat16,
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}, output_loading_info=True)
    assert not loading['missing_keys'], loading['missing_keys']
    model.eval()
    assert all(m.norm.activation in ('silu', 'swish') for m in model.modules()
               if m.__class__.__name__ == 'Qwen3_5GatedDeltaNet')
    report['model_load_seconds'] = time.time()-start
    report['device_map'] = model.hf_device_map
    tok = AutoTokenizer.from_pretrained(probe['model'], revision=probe['model_commit'])
    task.loss = native_loss
    fit, val, _, hashes = task.data_splits(tok, 2048, args.fit, 16, args.seed, dataset_name='Salesforce/wikitext')
    report['data_sha256'] = hashes
    with torch.no_grad():
        report['first_bf16_fit_nll'] = native_loss(model, fit[0]).item()
    from run_ppl_sweep import build_c4
    test = native_wikitext(tok)
    test = list(test[:, :test.shape[1]//2048*2048].split(2048, 1))
    c4 = list(build_c4(tok, 2048, 64).split(2048, 1))
    report['eval_data_sha256'] = {n: hashlib.sha256(torch.cat(v, 1).numpy().tobytes()).hexdigest()
                                  for n, v in [('wikitext', test), ('c4', c4)]}
    if args.bf16_reference:
        report['bf16_reference'] = {}
        for ds, batches in [('wikitext', test), ('c4', c4)]:
            values = task.evaluate(model, batches, 'bf16/'+ds)
            report['bf16_reference'][ds] = {'nll': values, 'ppl': math.exp(sum(values)/len(values))}
            save()
    modules = text_modules(model)
    assert {n: list(m.weight.shape) for n, m in modules.items()} == probe['quantized_shapes']
    base, alt = {}, {}
    with torch.no_grad():
        for i, (n, m) in enumerate(modules.items()):
            w = m.weight.detach()
            b = (quant_nvfp4_4over6 if args.weight_baseline == 'nvfp4_4over6' else quant_nvfp4)(w, 4, 16)
            b2 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='never')
            if args.weight_baseline == 'nvfp4':
                assert torch.equal(b, b2), n
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            base[n], alt[n] = b.cpu(), a.cpu()
            m.weight.copy_(b)
            if (i+1) % 32 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
    del a, b, b2, w
    wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
    task.activation_ste = wrapper.backward
    task.loss = native_loss
    if args.weight_baseline == 'nvfp4_4over6':
        # Measure the stronger comparator first. Held-out results never select masks.
        report['final']['baseline'] = {}
        old = json.loads(Path('results/task_sensitivity_qwen38/seed20260909/report.json').read_text())
        assert old['eval_data_sha256'] == report['eval_data_sha256']
        for ds, batches in [('wikitext', test), ('c4', c4)]:
            values = task.evaluate(model, batches, 'four_over_six/'+ds)
            result = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                      'vs_plain_nvfp4': task.paired(values, old['final']['baseline'][ds]['nll']),
                      'vs_previous_sparse': task.paired(values, old['final']['gradient_trust_backtracking'][ds]['nll'])}
            report['final']['baseline'][ds] = result
            save()
            print(f'FOUR_OVER_SIX {ds} {result["ppl"]} {result["vs_previous_sparse"]}', flush=True)
    with torch.no_grad():
        plain = task.loss(model, fit[0]).item()
        with wrapper.backward():
            ste = task.loss(model, fit[0]).item()
    assert plain == ste
    report['ste_forward_equal'] = True
    start = time.time()
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        means, ses, fit_losses = task.score_tiles(model, modules, base, alt, fit)
    report['scores_seconds'] = time.time()-start
    torch.save({'means': means, 'ses': ses, 'hashes': hashes, 'fit_losses': fit_losses,
                'model_commit': probe['model_commit'], 'args': report['args']}, out/'scores.pt')
    baseline_fit = task.evaluate(model, fit, 'fit/baseline')
    report['grad_vs_nograd_forward_max_delta'] = max(abs(a-b) for a, b in zip(fit_losses, baseline_fit))
    assert report['grad_vs_nograd_forward_max_delta'] == 0
    masks, history = task.calibrated_trust_masks(model, modules, base, alt, means, ses, fit, baseline_fit)
    report['proposal_fit_calibration'] = history
    rule = 'gradient_trust_backtracking'
    entry = task.selected_summary(masks, means)
    entry['fit'] = history[-1]['actual_fit'] if history and history[-1]['accepted'] else task.paired(baseline_fit, baseline_fit)
    task.install(modules, base, alt)
    baseline_val = task.evaluate(model, val, 'val/baseline')
    task.install(modules, base, alt, masks)
    chosen_val = task.evaluate(model, val, 'val/proposal')
    entry['val'] = task.paired(chosen_val, baseline_val)
    entry['val_nll'] = chosen_val
    report['baseline'] = {'fit': baseline_fit, 'val': baseline_val}
    report['results'][rule] = entry
    accepted = entry['val']['mean'] + 2*entry['val']['se'] < 0
    report['export_decision'] = 'accepted' if accepted else 'fallback_'+args.weight_baseline
    report['validation_forecast_relative_ppl_percent'] = 100*math.expm1(entry['val']['mean'])
    task.export_type_map(out/'candidate_type_map.json', probe['model'], masks, hashes['fit'], probe['model_commit'], rule)
    exported = masks if accepted else {n: torch.zeros_like(m) for n, m in masks.items()}
    task.export_type_map(out/'type_map.json', probe['model'], exported, hashes['fit'], probe['model_commit'], rule)
    for filename in ['candidate_type_map.json', 'type_map.json']:
        spec = json.loads((out/filename).read_text())
        spec['scope'] = 'qwen3_5_text_linear'
        spec['apply_function'] = 'probe_qwen38.apply_native_type_map'
        if args.weight_baseline == 'nvfp4_4over6':
            spec.pop('alpha')
            spec['baseline_weight_dtype'] = 'nvfp4_4over6'
            spec['e2m1_alphas'] = [1., 1.5]
            spec['e0m3_alpha'] = 1.
        task.atomic_json(out/filename, spec)
    torch.save({rule: masks}, out/'policies.pt')
    save()
    print(f'PROPOSAL {entry}; {report["export_decision"]}', flush=True)
    for label in ['baseline', rule]:
        if label == 'baseline' and args.weight_baseline == 'nvfp4_4over6':
            continue
        task.install(modules, base, alt, masks if label == rule else None)
        report['final'][label] = {}
        for ds, batches in [('wikitext', test), ('c4', c4)]:
            vals = task.evaluate(model, batches, label+'/'+ds)
            result = {'nll': vals, 'ppl': math.exp(sum(vals)/len(vals))}
            if label != 'baseline':
                result['vs_baseline'] = task.paired(vals, report['final']['baseline'][ds]['nll'])
            report['final'][label][ds] = result
            save()
            print(f'FINAL {label} {ds} {result.get("vs_baseline", result["ppl"])}', flush=True)
    report['peak_gpu_gib'] = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]
    report['complete'] = True
    save()
    print(f'DONE {out}', flush=True)


if __name__ == '__main__':
    main()
