"""Baseline-conditional channel ablations and activation anatomy, Slurm only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit through Slurm.')
for key, subdir in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                    ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], subdir))
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4, quant_nvfp4_4over6, quant_mix_4_6
import analyze_task_sensitivity as task
from probe_qwen38 import NativeActivationQuantization, native_loss, text_modules, apply_native_type_map


def variants(masks, modules):
    dominant, complement, random = {}, {}, {}
    generator = torch.Generator().manual_seed(20260914)
    for name, mask in masks.items():
        region = torch.zeros_like(mask)
        if modules[name].weight.shape[1] == 5120:
            region[:, 62] = True
        dominant[name] = mask & region
        complement[name] = mask & ~region
        shuffled = torch.zeros_like(mask)
        # Preserve the exact input-column counts, randomize output row groups.
        for column in mask.any(dim=0).nonzero().flatten().tolist():
            count = int(mask[:, column].sum())
            shuffled[torch.randperm(mask.shape[0], generator=generator)[:count], column] = True
        random[name] = shuffled
    return {'full': masks, 'channels_3968_4031_only': dominant,
            'without_channels_3968_4031': complement, 'random_rows_same_columns': random}


def main():
    assert torch.cuda.device_count() == 2
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path('results/task_sensitivity_four_over_six')
    out = root/'mechanism'
    out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists()
    olddir = Path('results/task_sensitivity_qwen38/seed20260909')
    reference = json.loads((olddir/'report.json').read_text())
    newdir = root/'seed20260912'
    new_report = json.loads((newdir/'report.json').read_text())
    assert new_report['complete']
    report = {'complete': False, 'job_id': os.environ['SLURM_JOB_ID'], 'results': {}}
    def save():
        task.atomic_json(out/'report.json', report)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        reference['args']['model'], revision=reference['model_commit'], dtype=torch.bfloat16,
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}).eval()
    tokenizer = AutoTokenizer.from_pretrained(reference['args']['model'], revision=reference['model_commit'])
    _, _, probe, hashes = task.data_splits(tokenizer, 2048, 64, 16, 20260909, dataset_name='Salesforce/wikitext')
    report['probe_sha256'] = hashes['probe']
    assert hashes['probe'] == reference['data_sha256']['probe']
    modules = text_modules(model)
    old_masks = torch.load(olddir/'policies.pt', map_location='cpu', weights_only=True)['gradient_trust_backtracking']
    new_masks = torch.load(newdir/'policies.pt', map_location='cpu', weights_only=True)['gradient_trust_backtracking']
    selected_indices = {name: (old_masks[name] | new_masks[name]).nonzero().tolist() for name in modules}
    selected_columns = {name: sorted({column for _, column in pairs}) for name, pairs in selected_indices.items()}
    tile_errors = {}
    base_nv, base_four, alt = {}, {}, {}
    with torch.no_grad():
        for i, (name, module) in enumerate(modules.items()):
            w = module.weight.detach()
            base_nv[name] = quant_nvfp4(w, 4, 16).cpu()
            base_four[name] = quant_nvfp4_4over6(w, 4, 16).cpu()
            alt[name] = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always').cpu()
            for row, column in selected_indices[name]:
                region = (slice(8*row, 8*row+8), slice(64*column, 64*column+64))
                pristine = w[region].float().cpu()
                tile_errors[(name, row, column)] = {
                    'nvfp4': base_nv[name][region].float()-pristine,
                    'four_over_six': base_four[name][region].float()-pristine,
                    'e0m3': alt[name][region].float()-pristine}
            if (i+1) % 32 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
    del w
    # Real exported FourOverSix map applies to pristine weights and must
    # match direct candidate installation over the complete model exactly.
    apply_native_type_map(model, json.loads((newdir/'type_map.json').read_text()))
    exported_masks = new_masks if new_report['export_decision'] == 'accepted' else {n: torch.zeros_like(m) for n, m in new_masks.items()}
    for name, module in modules.items():
        expected = torch.where(task.expand_mask(exported_masks[name]), alt[name], base_four[name])
        assert torch.equal(module.weight.detach().cpu(), expected), name
    report['four_over_six_export_all_weights_exact'] = True
    del expected
    wrapper = NativeActivationQuantization(modules, QuantConfig(a_bits=4, a_dtype='nvfp4_4over6', a_groupsize=16))
    task.loss = native_loss
    energies, handles, covariances, counts = {}, [], {}, {}
    def hook(name):
        def collect(_module, inputs, _output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            energy = x.float().square().sum(0).cpu()
            energies[name] = energies.get(name, torch.zeros_like(energy)) + energy
            counts[name] = counts.get(name, 0) + len(x)
            for column in selected_columns[name]:
                block = x[:, 64*column:64*column+64].float()
                covariance = (block.T @ block).cpu()
                key = (name, column)
                covariances[key] = covariances.get(key, torch.zeros_like(covariance)) + covariance
        return collect
    for name, module in modules.items():
        handles.append(module.register_forward_hook(hook(name)))
    task.install(modules, base_nv, alt)
    baseline_nv = task.evaluate(model, probe, 'anatomy/nvfp4-baseline')
    for handle in handles:
        handle.remove()
    torch.save(energies, out/'activation_energy.pt')
    geometry = []
    for (name, row, column), errors in tile_errors.items():
        covariance = (covariances[(name, column)]/counts[name]).double().cuda()
        alt_error = errors['e0m3'].double().cuda()
        for baseline in ('nvfp4', 'four_over_six'):
            base_error = errors[baseline].double().cuda()
            gram = alt_error.T @ alt_error - base_error.T @ base_error
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            worst_input = eigenvectors[:, -1]
            witness = (alt_error @ worst_input).square().sum() - (base_error @ worst_input).square().sum()
            expected_change = (gram*covariance.T).sum()
            nuclear = eigenvalues.abs().sum()
            covariance_norm = torch.linalg.eigvalsh(covariance)[-1].clamp_min(1e-300)
            radius = (-expected_change/nuclear.clamp_min(1e-300)).clamp_min(0)
            geometry.append({'module': name, 'row_tile': row, 'column_tile': column, 'baseline': baseline,
                             'activation_reference': 'nvfp4', 'gram_nuclear_norm': float(nuclear),
                             'max_gram_eigenvalue': float(eigenvalues[-1]),
                             'adverse_input_witness': float(witness),
                             'adverse_input_direction': worst_input.cpu().tolist(),
                             'calibration_output_error_change': float(expected_change),
                             'sufficient_covariance_radius': float(radius),
                             'relative_covariance_radius': float(radius/covariance_norm)})
    task.atomic_json(out/'local_error_geometry.json', geometry)
    report['activation_anatomy'] = {
        name: {'peak_channel': int(energy.argmax()),
               'peak_over_median': float(energy.max()/energy.median().clamp_min(1e-30)),
               'channel_region_energy_fraction': float(energy[3968:4032].sum()/energy.sum().clamp_min(1e-30))}
        for name, energy in energies.items() if energy.numel() == 5120}
    for label, base in [('nvfp4', base_nv), ('four_over_six', base_four)]:
        task.install(modules, base, alt)
        baseline = baseline_nv if label == 'nvfp4' else task.evaluate(model, probe, label+'/baseline')
        report['results'][label] = {'baseline_nll': baseline, 'maps': {}}
        for map_name, masks in [('old_nvfp4_map', old_masks), ('new_four_over_six_map', new_masks)]:
            results = {}
            for name, selection in variants(masks, modules).items():
                task.install(modules, base, alt, selection)
                values = task.evaluate(model, probe, label+'/'+map_name+'/'+name)
                results[name] = {'tiles': sum(int(m.sum()) for m in selection.values()),
                                 'nll': values, 'vs_baseline': task.paired(values, baseline)}
                print(f'ABLATION {label} {map_name} {name} {results[name]["vs_baseline"]}', flush=True)
            report['results'][label]['maps'][map_name] = results
            save()
    report['complete'] = True
    save()
    print('MECHANISM ABLATIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
