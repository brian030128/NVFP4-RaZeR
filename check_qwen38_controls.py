"""Predeclared MSE/random controls and native exported-map replay, Slurm only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit all compute through Slurm.')
for key, subdir in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                    ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], subdir))
import hashlib
import json
import math
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4, quant_mix_4_6
import analyze_task_sensitivity as task
from probe_qwen38 import NativeActivationQuantization, native_loss, native_wikitext, text_modules, apply_native_type_map


def main():
    assert torch.cuda.device_count() == 2
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path('results/task_sensitivity_qwen38/seed20260909')
    reference = json.loads((source/'report.json').read_text())
    assert reference['complete'] and reference['export_decision'] == 'accepted'
    assert transformers.__version__ == reference['transformers']
    out = Path('results/task_sensitivity_qwen38/controls')
    out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists(), 'Use a fresh result directory.'
    report = {'complete': False, 'job_id': os.environ['SLURM_JOB_ID'],
              'model_commit': reference['model_commit'], 'random_seed': 20260911,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'final': {}, 'tile_counts': {}}
    def save():
        task.atomic_json(out/'report.json', report)
    save()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        reference['args']['model'], revision=reference['model_commit'], dtype=torch.bfloat16,
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}).eval()
    tok = AutoTokenizer.from_pretrained(reference['args']['model'], revision=reference['model_commit'])
    modules = text_modules(model)
    base, alt, mse = {}, {}, {}
    with torch.no_grad():
        for i, (name, module) in enumerate(modules.items()):
            w = module.weight.detach()
            b = quant_nvfp4(w, 4, 16)
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            mse[name] = (task.tile_sum((a.float()-w.float()).square()) <
                         task.tile_sum((b.float()-w.float()).square())).cpu()
            base[name], alt[name] = b.cpu(), a.cpu()
            # Keep native weights pristine for the actual exported-map loader below.
            if (i+1) % 32 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
    del w, a, b
    wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
    task.loss = native_loss
    from run_ppl_sweep import build_c4
    ids = native_wikitext(tok)
    test = list(ids[:, :ids.shape[1]//2048*2048].split(2048, 1))
    c4 = list(build_c4(tok, 2048, 64).split(2048, 1))
    for ds, batches in [('wikitext', test), ('c4', c4)]:
        assert hashlib.sha256(torch.cat(batches, 1).numpy().tobytes()).hexdigest() == reference['eval_data_sha256'][ds]
    apply_native_type_map(model, json.loads((source/'type_map.json').read_text()))
    replay = task.evaluate(model, test[:16], 'export-replay')
    expected = reference['final']['gradient_trust_backtracking']['wikitext']['nll'][:16]
    report['export_replay_max_nll_delta'] = max(abs(a-b) for a, b in zip(replay, expected))
    assert report['export_replay_max_nll_delta'] == 0
    task.install(modules, base, alt)
    replay = task.evaluate(model, test[:16], 'baseline-replay')
    expected = reference['final']['baseline']['wikitext']['nll'][:16]
    report['baseline_replay_max_nll_delta'] = max(abs(a-b) for a, b in zip(replay, expected))
    assert report['baseline_replay_max_nll_delta'] == 0
    selected = torch.load(source/'policies.pt', map_location='cpu', weights_only=True)['gradient_trust_backtracking']
    policies = {'matched_random': task.matched_random(selected, 20260911), 'weight_mse': mse}
    for label, masks in policies.items():
        report['tile_counts'][label] = sum(int(m.sum()) for m in masks.values())
        task.install(modules, base, alt, masks)
        report['final'][label] = {}
        for ds, batches in [('wikitext', test), ('c4', c4)]:
            values = task.evaluate(model, batches, label+'/'+ds)
            result = {'nll': values, 'ppl': math.exp(sum(values)/len(values)),
                      'vs_baseline': task.paired(values, reference['final']['baseline'][ds]['nll'])}
            report['final'][label][ds] = result
            save()
            print(f'FINAL {label} {ds} {result["vs_baseline"]}', flush=True)
    report['complete'] = True
    save()
    print('CONTROLS AND EXPORT REPLAY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
