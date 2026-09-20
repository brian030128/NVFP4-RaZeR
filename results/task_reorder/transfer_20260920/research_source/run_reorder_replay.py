"""Finite-step CE/KL calibration of frozen last-MLP layouts, with exact replay.

Uses only inner calibration sequences, never WikiText/C4. Replay includes the
nonlinear gate, quantized down input, residual, final norm, and language head.
It must reproduce full-model logits exactly before its scores can select a
candidate. This is calibration refinement, not held-out model evaluation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_conditional_format import save, sha
from run_c4_frozen import digest_file
from run_math_code_calibration import math_code_data
from run_task_reorder_eval import load_layouts
from run_task_reorder import split_sequences
from run_shared_mlp_reorder import nested_split


def paired_bounds(values, reference, k=1.):
    differences = torch.tensor(values, dtype=torch.float64) - torch.tensor(reference, dtype=torch.float64)
    mean = float(differences.mean())
    se = float(differences.std() / math.sqrt(len(differences)))
    return dict(mean=mean, se=se, upper=mean + k * se, two_se=2 * se)


def select_policy(scores, allowed):
    eligible = [label for label in allowed if label != 'four_over_six'
                and scores[label]['ce']['upper'] < 0 and scores[label]['kl']['upper'] < 0]
    return min(eligible, key=lambda label: scores[label]['ce']['mean']) if eligible else 'four_over_six'


def confirm_choices(scores, choices):
    """Accept/reject each preselected winner independently; never rerank."""
    return {key: label if scores[label]['ce']['upper'] < 0 and scores[label]['kl']['upper'] < 0
            else 'four_over_six' for key, label in choices.items()}


@torch.no_grad()
def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm H200 worker')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--layouts', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--split', choices=('inner', 'election'), default='inner')
    ap.add_argument('--candidate-only', action='store_true', help='Exclude the old outer-elected identity control')
    ap.add_argument('--policies-from', type=Path, help='Evaluate only winners frozen in this earlier replay report')
    ap.add_argument('--foldable-prefixes', nargs='+', default=['shared_', 'identity'])
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    prior = json.loads((args.calib / 'report.json').read_text())
    assert prior['status'] == 'complete'
    assert transformers.__version__ == prior['transformers_version']
    panels, frozen_hashes = load_layouts(args.layouts, prior)
    controls = next(iter(panels.values()))
    fixed_choices = None
    if args.policies_from:
        selection = json.loads(args.policies_from.read_text())
        assert selection['status'] == 'complete' and selection['calibration_split'] == 'inner'
        fixed_choices = {key: selection[key] for key in ('selected_any', 'selected_foldable')}
        panels = {label: panel for label, panel in panels.items() if label in fixed_choices.values()}
    manifest = next(iter(controls.values()))['provenance']
    fit, election = split_sequences(manifest, .5, 0)
    _, validation_positions = nested_split(manifest, fit)
    indices = [fit[i] for i in validation_positions] if args.split == 'inner' else election
    if args.split == 'election':
        assert args.policies_from and args.candidate_only, 'Confirmation requires frozen inner winners only'
        assert not set(selection['selection_sequences']) & {manifest['sequence_ids'][i] for i in indices}
    else:
        assert not set(indices) & set(election)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], model=prior['model'],
                  source=prior['source'], revision=prior['revision'], layout_sha256=frozen_hashes,
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  selection_sequences=[manifest['sequence_ids'][i] for i in indices],
                  selection_sources=[manifest['sequence_sources'][i] for i in indices],
                  calibration_only=True, uses_wiki=False, uses_c4=False,
                  calibration_split=args.split, fixed_choices=fixed_choices,
                  fixed_choices_report_sha256=digest_file(args.policies_from) if args.policies_from else None,
                  activation='FourOverSix tensor-wide factor', background='FourOverSix',
                  selection='Minimum CE change among candidates with paired mean+1SE <0 for CE and teacher KL',
                  warning=('Calibration refinement; these inner sequences helped rank shared layouts and fit unconstrained layouts.'
                           if args.split == 'inner' else 'Confirmation of fixed winners; no candidate reranking is permitted.'),
                  source_sha256={p: digest_file(p) for p in ('run_reorder_replay.py', 'quantize/quantizer.py',
                                                           'quantize/task_reorder.py')}, metrics={})
    save(args.out, report)
    loader = AutoModelForCausalLM
    if prior['model'] == 'qwen27b':
        from transformers import Qwen3_5ForConditionalGeneration
        loader = Qwen3_5ForConditionalGeneration
    model, loading = loader.from_pretrained(prior['source'], revision=prior['revision'],
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda', output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    model.eval().requires_grad_(False)
    modules = {name: model.get_submodule(name) for name in prior['matrices']}
    for name, module in modules.items():
        assert sha(module.weight) == prior['matrices'][name]['source_sha256'], name
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    batches, fit_metadata = math_code_data(tok, prior['fit'])
    assert fit_metadata == prior['fit']
    all_batches = [ids for source in batches.values() for ids in source]
    selected = [all_batches[i] for i in indices]
    assert [sha(ids) for ids in selected] == report['selection_sequences']
    teacher_dir = Path(os.environ['TMPDIR']) / ('reorder_replay_teacher_' + args.out.name)
    teacher_dir.mkdir()
    for i, ids in enumerate(selected):
        logits = model(input_ids=ids.cuda(), use_cache=False).logits
        lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        torch.save(lp.cpu(), teacher_dir / f'{i:03d}.pt')
        del lp, logits
        print(f'REPLAY TEACHER {i + 1}/{len(selected)}', flush=True)
    base, alternative = {}, {}
    for name, module in modules.items():
        b = quant_nvfp4_4over6(module.weight, 4, 16)
        if name in controls:
            base[name] = b
            alternative[name] = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        module.weight.copy_(b)
    del b
    prepared = {'four_over_six': base, 'identity': {
        name: original_order_weight_reference(base[name], alternative[name], controls[name], 'identity') for name in controls}}
    if args.candidate_only:
        del prepared['identity']
    for label, panel in panels.items():
        prepared[label] = {name: original_order_weight_reference(base[name], alternative[name], panel[name]) for name in controls}
    del alternative
    act_handles = [module.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for module in modules.values()]
    mlp_names = {name.rsplit('.', 1)[0] for name in controls}
    assert len(mlp_names) == 1
    mlp_name = next(iter(mlp_names))
    layer_name = mlp_name.rsplit('.', 1)[0]
    backbone_name, layer_index = layer_name.rsplit('.layers.', 1)
    backbone = model.get_submodule(backbone_name)
    assert int(layer_index) == len(backbone.layers) - 1
    mlp = model.get_submodule(mlp_name)
    prenorm = model.get_submodule(layer_name + '.post_attention_layernorm')
    final_norm, head = backbone.norm, model.get_output_embeddings()
    policies = list(prepared)
    report['metrics'] = {policy: dict(ce=[], kl=[]) for policy in policies}
    report['full_model_replay_verified_sequences'] = 0
    report['modified_policy_replay_verified'] = []

    def install(policy):
        for name, weight in prepared[policy].items():
            modules[name].weight.copy_(weight)

    for i, ids in enumerate(selected):
        ids = ids.cuda()
        teacher_path = teacher_dir / f'{i:03d}.pt'
        teacher = torch.load(teacher_path, map_location='cpu', weights_only=True).cuda()
        capture = {}
        def residual_hook(module, inputs):
            capture['residual'] = inputs[0].detach().clone()
        def mlp_hook(module, inputs):
            capture['input'] = inputs[0].detach().clone()
        install('four_over_six')
        handles = [prenorm.register_forward_pre_hook(residual_hook), mlp.register_forward_pre_hook(mlp_hook)]
        direct = model(input_ids=ids, use_cache=False).logits
        for handle in handles:
            handle.remove()
        for policy in policies:
            install(policy)
            logits = head(final_norm(capture['residual'] + mlp(capture['input'])))
            if policy == 'four_over_six':
                assert torch.equal(logits, direct), 'Baseline replay must match full logits bitwise'
                report['full_model_replay_verified_sequences'] += 1
                del direct
            elif i == 0:
                direct_policy = model(input_ids=ids, use_cache=False).logits
                assert torch.equal(logits, direct_policy), f'Modified replay mismatch: {policy}'
                report['modified_policy_replay_verified'].append(policy)
                del direct_policy
            lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            ce = float(F.nll_loss(lp, ids[:, 1:].reshape(-1)))
            kl = float(F.kl_div(lp, teacher, reduction='batchmean', log_target=True))
            assert math.isfinite(ce) and math.isfinite(kl)
            report['metrics'][policy]['ce'].append(ce)
            report['metrics'][policy]['kl'].append(kl)
            del lp, logits
        del teacher, capture
        teacher_path.unlink()
        save(args.out, report)
        print(f'FINITE REPLAY {i + 1}/{len(selected)}', flush=True)
    for handle in act_handles:
        handle.remove()
    report['paired'] = {policy: {objective: paired_bounds(values[objective], report['metrics']['four_over_six'][objective])
                                for objective in ('ce', 'kl')} for policy, values in report['metrics'].items()}
    report['selected_any'] = select_policy(report['paired'], policies)
    foldable = [p for p in policies if p == 'four_over_six' or any(p.startswith(prefix) for prefix in args.foldable_prefixes)]
    report['selected_foldable'] = select_policy(report['paired'], foldable)
    if fixed_choices is not None:
        report.update(confirm_choices(report['paired'], fixed_choices))
        assert digest_file(args.policies_from) == report['fixed_choices_report_sha256']
    assert all(digest_file(path) == digest for path, digest in frozen_hashes.items())
    report['status'] = 'complete'
    save(args.out, report)
    print('FINITE REPLAY SELECTED ' + json.dumps({k: report[k] for k in ('selected_any', 'selected_foldable', 'paired')}), flush=True)


if __name__ == '__main__':
    main()
