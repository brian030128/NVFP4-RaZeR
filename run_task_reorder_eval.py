"""Matched pilot quality evaluation of frozen layouts; run only on a Slurm GPU.

Other text linear weights use the selected fixed background. This undoes
permutations on fake-quantized weights and cannot measure deployment throughput.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha


def validate_evaluation_data(actual, reference):
    """Compare exact published token windows, whose counts depend on tokenizer."""
    for domain in ('wiki', 'c4_paper'):
        for key in ('revision', 'windows', 'window_tokens', 'token_sha256'):
            if actual[domain][key] != reference[domain][key]:
                raise ValueError(f'Published evaluation mismatch: {domain}.{key}')


def coarse_mask(ce, kl, weight_shape, k=3):
    """Aggregate historical 8x64 sequence scores before computing 256x64 SEs."""
    n, width = weight_shape
    if n % 8 or width % 64 or ce.shape != kl.shape or ce.shape != (128, n // 8 * (width // 64)):
        raise ValueError('Need 128 matching historical 8x64 score sequences')
    bounds = []
    for score in (ce, kl):
        values = score.double().reshape(128, n // 8, width // 64)
        values = F.pad(values, (0, 0, 0, (-values.shape[1]) % 32))
        values = values.reshape(128, -1, 32, width // 64).sum(2)
        bounds.append(values.mean(0) + k * values.std(0, unbiased=True) / math.sqrt(128))
    if not all(torch.isfinite(bound).all() for bound in bounds):
        raise ValueError('Nonfinite coarse bound')
    return torch.maximum(*bounds) < 0


def validate_compact_masks(bundle, prior):
    if (bundle.get('schema'), bundle.get('sequences'), bundle.get('k'), bundle.get('revision')) != (
            'mixfp4_compact_masks_v1', 128, 3, prior['revision']):
        raise ValueError('Compact mask provenance mismatch')
    expected = {name: value['source_sha256'] for name, value in prior['matrices'].items()}
    if bundle['weight_sha256'] != expected:
        raise ValueError('Compact mask source weights mismatch')
    for kind, height in (('raw256', 256), ('fine8x64', 8)):
        if set(bundle[kind]) != set(expected):
            raise ValueError('Compact masks do not cover the model')
        for name, mask in bundle[kind].items():
            n, k = prior['matrices'][name]['shape']
            if mask.dtype != torch.bool or tuple(mask.shape) != (math.ceil(n / height), k // 64):
                raise ValueError('Compact mask shape/dtype mismatch')
    return bundle


def raw256_masks(directory, prior):
    if prior.get('compact_mask_sha256'):
        path = directory / 'compact_masks.pt'
        if digest_file(path) != prior['compact_mask_sha256']:
            raise ValueError('Compact mask digest mismatch')
        bundle = validate_compact_masks(torch.load(path, map_location='cpu', weights_only=True), prior)
        masks = bundle['raw256']
        print(f'RAW256 CACHED {sum(int(mask.sum()) for mask in masks.values())}', flush=True)
        return masks
    masks = {}
    for index, (name, metadata) in enumerate(prior['matrices'].items()):
        shard = torch.load(directory / 'scores' / f'{index:03d}.pt', map_location='cpu', weights_only=True)
        if shard['name'] != name:
            raise ValueError('Historical score/module mismatch')
        masks[name] = coarse_mask(shard['ce'], shard['kl'], metadata['shape'])
    print(f'RAW256 ELECTED {sum(int(mask.sum()) for mask in masks.values())}', flush=True)
    return masks


def mix_coarse(base, alternative, mask):
    expanded = mask.to(base.device).repeat_interleave(256, 0).repeat_interleave(64, 1)
    return torch.where(expanded[:base.shape[0], :base.shape[1]], alternative, base)


def load_layouts(specifications, prior, expected_modules=None, allow_rowbands=False):
    expected_modules = set(prior['reorder_modules'] if expected_modules is None else expected_modules)
    if not expected_modules or not expected_modules <= set(prior['matrices']):
        raise ValueError('Invalid evaluation module scope')
    panels, hashes = {}, {}
    for specification in specifications:
        label, directory = specification.split('=', 1)
        if not label or label in panels or label in ('four_over_six', 'identity', 'identity_8x64'):
            raise ValueError('Layout labels must be unique and distinct from controls')
        paths = sorted(Path(directory).glob('*/layout.pt'))
        if not paths:
            raise ValueError(f'No frozen layouts in {directory}')
        panel = {}
        for path in paths:
            report = json.loads(path.with_name('report.json').read_text())
            if report['status'] != 'complete':
                raise ValueError(f'Incomplete layout {path}')
            layout = torch.load(path, map_location='cpu', weights_only=True)
            name = layout['name']
            if name in panel or name not in expected_modules:
                raise ValueError(f'Unexpected or duplicate module {name}')
            if layout['schema'] != 'mixfp4_task_reorder_v1':
                raise ValueError('Unknown layout schema')
            cfg = layout['config']
            geometry = (cfg['tile_rows'], cfg['tile_cols'], cfg['atom_rows'], cfg['atom_cols'])
            if geometry != (256, 64, 1, 16):
                if not allow_rowbands or geometry not in ((256, 64, 8, 64), (256, 64, 1, 64)):
                    raise ValueError('Expected 256x64 tiles and explicitly allowed score atoms')
                n, width = layout['weight_shape']
                atom_rows = cfg['atom_rows']
                grouped = layout['row_perm'].reshape(-1, atom_rows)
                if cfg['axes'] != 'rows' or not torch.equal(layout['col_perm'], torch.arange(width)):
                    raise ValueError('Historical row-band layouts cannot reorder columns')
                if not torch.equal(grouped, (grouped[:, :1] // atom_rows) * atom_rows + torch.arange(atom_rows)):
                    raise ValueError('Row-band layout splits an eight-row atom')
            origin = layout['provenance']
            if origin['weight_sha256'] != prior['matrices'][name]['source_sha256']:
                raise ValueError('Layout/model source weight mismatch')
            if origin['revision'] != prior['revision'] or origin['status'] != 'complete':
                raise ValueError('Incomplete or mismatched score provenance')
            if set(layout['fit_sequence_ids']) & set(layout['election_sequence_ids']):
                raise ValueError('Search/election leakage')
            if set(layout['fit_sequence_ids'] + layout['election_sequence_ids']) != set(origin['sequence_ids']):
                raise ValueError('Split does not cover the recorded scores')
            panel[name] = layout
            hashes[str(path.resolve())] = digest_file(path)
        if set(panel) != expected_modules:
            raise ValueError('Every policy must cover exactly the requested modules')
        panels[label] = panel
    control = next(iter(panels.values()))
    for panel in panels.values():
        for name, layout in panel.items():
            reference = control[name]
            for key in ('fit_sequence_ids', 'election_sequence_ids'):
                if layout[key] != reference[key]:
                    raise ValueError('Policies must use identical search/election splits')
            if layout['config']['k'] != reference['config']['k']:
                raise ValueError('Policies must use the same election threshold')
            for key in ('identity_mask', 'identity_8x64_mask'):
                if not torch.equal(layout[key], reference[key]):
                    raise ValueError('Identity controls differ across layout policies')
    return panels, hashes


def paired(a, b):
    difference = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    mean = float(difference.mean())
    se = float(difference.std(unbiased=True) / math.sqrt(len(difference)))
    return dict(delta_nll=mean, two_se=2 * se, interval=[mean - 2 * se, mean + 2 * se])


def write_summary(out, report):
    lines = [f'# Reordering pilot: {report["model"]}', '',
             f'Reordering {len(report["pilot_modules"])} matrices; background policy: {report["background"]}. '
             'Layouts and maps were frozen before held-out evaluation. '
             'Activation factors are tensor-wide; sequence length is 2048.', '',
             '| Policy | E0M3 tiles | WikiText-2 | C4 |', '|---|---:|---:|---:|']
    for policy, ev in report['evaluation'].items():
        lines.append(f'| {policy} | {report["elected_tiles"][policy]} | '
                     f'{ev["wiki"]["ppl"]:.6f} | {ev["c4"]["ppl"]:.6f} |')
    lines += ['', 'identity_8x64 uses 8x64 tiles inside the pilot; background tiles, if any, '
              'remain 256x64. All other mixed policies use 256x64 throughout. '
              'The FourOverSix row always has all weights at FourOverSix. '
              'raw256 elects on all 128 sequences. Original pilot maps use the 64-sequence election split; '
              'finite-refined maps use the selection and confirmation protocol recorded in mask_selection.', '',
              '| Comparison | Wiki ΔNLL ±2SE | C4 ΔNLL ±2SE |', '|---|---:|---:|']
    for name, values in report['paired'].items():
        cells = [f'{values[d]["delta_nll"]:+.6f} ±{values[d]["two_se"]:.6f}' for d in ('wiki', 'c4')]
        lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
    lines += ['', 'Intervals are descriptive paired window intervals. This is a quality '
              'simulation, with no native permutation-overhead measurement. The user has '
              'separately verified equal throughput for unreordered 256x64 MixFP4 and NVFP4.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run evaluation through Slurm')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('llama8b', 'qwen27b'), required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--layouts', nargs='+', required=True, help='LABEL=directory of module layouts')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layout-module-regex', help='Explicit expanded scope within the calibrated matrices')
    ap.add_argument('--allow-rowbands', action='store_true', help='Accept legal eight-row historical atoms')
    ap.add_argument('--policies', nargs='+', help='Evaluate only these policies, for independent GPU shards')
    ap.add_argument('--background', choices=('four_over_six', 'raw256'), default='four_over_six',
                    help='Policy for modules outside the reordered pilot. raw256 uses all 128 calibration sequences.')
    args = ap.parse_args()
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', 4))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    prior = json.loads((args.calib / 'report.json').read_text())
    pilot_only = prior['status'] == 'pilot_complete'
    if pilot_only:
        assert prior['pilot_scores_verified'] and args.background == 'four_over_six'
    else:
        assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_wiki_calibration'] and not prior['uses_c4_calibration']
    assert transformers.__version__ == prior['transformers_version']
    if not pilot_only:
        assert digest_file(args.calib / 'maps.json') == prior['map_sha256']
    expected = ([name for name in prior['matrices'] if re.search(args.layout_module_regex, name)]
                if args.layout_module_regex else None)
    panels, hashes = load_layouts(args.layouts, prior, expected, args.allow_rowbands)
    controls = next(iter(panels.values()))
    raw_masks = raw256_masks(args.calib, prior) if args.background == 'raw256' else {}
    args.out.mkdir(parents=True, exist_ok=False)
    policies = ['four_over_six', *(['raw256'] if raw_masks else []), 'identity', 'identity_8x64', *panels]
    if args.policies:
        if len(set(args.policies)) != len(args.policies) or not set(args.policies) <= set(policies):
            raise ValueError('Unknown or duplicate policy in evaluation shard')
        policies = [policy for policy in policies if policy in args.policies]
    report = dict(status='running', model=args.model, calibration=str(args.calib),
                  source=prior['source'], revision=prior['revision'], job_id=os.environ['SLURM_JOB_ID'],
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  gpu=torch.cuda.get_device_name(), layout_sha256=hashes,
                  source_sha256={p: digest_file(p) for p in (
                      'run_task_reorder_eval.py', 'quantize/task_reorder.py', 'quantize/quantizer.py',
                      'run_baseline_protocol_audit.py')},
                  pilot_modules=list(controls), policies=policies, length=2048,
                  layout_module_regex=args.layout_module_regex, allow_rowbands=args.allow_rowbands,
                  background=args.background,
                  activation='FourOverSix tensor-wide factor', attention='sdpa',
                  wiki_use_cache=True, c4_use_cache=False, native_overhead_measured=False,
                  evaluation={}, paired={}, elected_tiles={'four_over_six': 0})
    report['mask_selection'] = {label: {name: layout.get('mask_selection',
        {'method': 'directional CE/KL mean+3SE on 64 outer election sequences'})
        for name, layout in panel.items()} for label, panel in panels.items()}
    calibration_job = '333779' if args.model == 'llama8b' else '333787'
    shipped = Path(f'results/math_code_adaptive/calibration_{calibration_job}_{args.model}/maps.json')
    regenerated = {} if pilot_only else json.loads((args.calib / 'maps.json').read_text())['maps']
    historical = json.loads(shipped.read_text())['maps']
    map_audit = {}
    for policy in sorted(set(regenerated) & set(historical)):
        now, old = regenerated[policy], historical[policy]
        difference = sum(len(set(now.get(n, [])) ^ set(old.get(n, []))) for n in set(now) | set(old))
        map_audit[policy] = dict(identical=now == old, changed_tiles=difference)
    report['historical_map_audit'] = dict(reference=str(shipped), policies=map_audit,
                                         available=not pilot_only,
                                         all_identical=None if pilot_only else regenerated == historical)
    nonpilot_count = sum(int(mask.sum()) for name, mask in raw_masks.items() if name not in controls)
    if raw_masks:
        report['elected_tiles']['raw256'] = sum(int(mask.sum()) for mask in raw_masks.values())
        report['raw256_calibration_sequences'] = 128
        report['raw256_k'] = 3
        report['raw256_accumulation'] = 'float64 sums of per-sequence 8x64 scores; then mean + 3 SE'
        torch.save(raw_masks, args.out / 'raw256_masks.pt')
        report['raw256_mask_sha256'] = digest_file(args.out / 'raw256_masks.pt')
        report['background_elected_tiles'] = nonpilot_count
    report['pilot_elected_tiles'] = {}
    for policy in ['identity', 'identity_8x64', *panels]:
        selected = panels.get(policy, controls)
        key = 'mask' if policy in panels else policy + '_mask'
        report['elected_tiles'][policy] = nonpilot_count + sum(int(layout[key].sum()) for layout in selected.values())
        report['pilot_elected_tiles'][policy] = report['elected_tiles'][policy] - nonpilot_count
    save(args.out, report)
    loader = AutoModelForCausalLM
    if args.model == 'qwen27b':
        from transformers import Qwen3_5ForConditionalGeneration
        loader = Qwen3_5ForConditionalGeneration
    model, loading = loader.from_pretrained(
        prior['source'], revision=prior['revision'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='cuda', output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and (('language_model' in n and 'head' not in n) if args.model == 'qwen27b'
                    else m is not model.get_output_embeddings())}
    assert list(modules) == list(prior['matrices'])
    base, alternative, raw_weights = {}, {}, {}
    for name, module in modules.items():
        assert sha(module.weight) == prior['matrices'][name]['source_sha256'], name
        b = quant_nvfp4_4over6(module.weight, 4, 16)
        raw_active = name in raw_masks and bool(raw_masks[name].any())
        if name in controls or raw_active:
            a = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        if name in controls:
            base[name] = b.cpu()
            alternative[name] = a.cpu()
        if raw_active:
            raw_weights[name] = mix_coarse(b, a, raw_masks[name]).cpu()
        if name in controls or raw_active:
            del a
        module.weight.copy_(b)
    del b
    report['source_weights_verified'] = True
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    batches, report['data'] = data(tok, prior, 2048)
    published_job = '336969' if args.model == 'qwen27b' else '336566'
    published_path = Path(f'results/kse_paper/job_{published_job}/{args.model}/report.json')
    published = json.loads(published_path.read_text())
    validate_evaluation_data(report['data'], published['data'])
    report['published_token_windows_verified'] = str(published_path)
    excluded = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
    assert excluded.isdisjoint(d['document_sha256'] for d in report['data']['c4_paper']['documents'])
    save(args.out, report)
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for m in modules.values()]
    raw_installed = False
    for policy in policies:
        # All modules start at FourOverSix. Install the full raw background once;
        # subsequent policies replace only the pilot with their matched maps.
        if raw_masks and policy != 'four_over_six' and not raw_installed:
            for name, weight in raw_weights.items():
                modules[name].weight.copy_(weight.to(modules[name].weight.device))
            raw_installed = True
        for name in (() if policy == 'raw256' else controls):
            module = modules[name]
            b = base[name].to(module.weight.device)
            if policy != 'four_over_six':
                a = alternative[name].to(module.weight.device)
                layout = panels.get(policy, controls)[name]
                b = original_order_weight_reference(b, a, layout,
                                                    'reordered' if policy in panels else policy)
                del a
            module.weight.copy_(b)
            del b
        evaluation = {}
        for domain, sequences in batches.items():
            values = []
            for index, ids in enumerate(sequences):
                ids = ids.cuda()
                logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
                value = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              ids[:, 1:].reshape(-1)))
                assert math.isfinite(value)
                values.append(value)
                del logits
                if (index + 1) % 32 == 0:
                    print(f'EVAL {policy} {domain} {index + 1}/{len(sequences)}', flush=True)
            losses = torch.tensor(values, dtype=torch.float32) * 2048
            key = 'c4' if domain == 'c4_paper' else domain
            evaluation[key] = dict(nll=values, windows=len(values),
                                   ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
            print(f'PPL {policy} {key} {evaluation[key]["ppl"]:.6f}', flush=True)
        report['evaluation'][policy] = evaluation
        for reference in ('four_over_six', 'raw256', 'identity'):
            if policy == reference or reference not in report['evaluation']:
                continue
            report['paired'][f'{policy} - {reference}'] = {
                d: paired(evaluation[d]['nll'], report['evaluation'][reference][d]['nll']) for d in evaluation}
        save(args.out, report)
        write_summary(args.out, report)
    for handle in handles:
        handle.remove()
    assert all(digest_file(path) == digest for path, digest in hashes.items())
    supplied = json.loads(Path('results/task_reorder/cluster_20260919/raw256_reference.json').read_text())
    report['user_reference_provenance'] = supplied['provenance']
    report['user_reference_difference'] = {
        policy: dict(tiles=report['elected_tiles'][policy] - expected['tiles'],
                     **{domain: report['evaluation'][policy][domain]['ppl'] - expected[domain]
                        for domain in ('wiki', 'c4')})
        for policy, expected in supplied['models'][args.model].items() if policy in report['evaluation']}
    report['status'] = 'complete'
    save(args.out, report)
    write_summary(args.out, report)


if __name__ == '__main__':
    main()
