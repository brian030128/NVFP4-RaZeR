"""Map-conditioned GPTQ for MixFP4 at a frozen tile map; run only on a Slurm GPU.

The tile-format maps are the frozen task-loss maps from the transfer calibration
(raw 256x64 k=3 and fine 8x64 k=3). GPTQ never changes a map: it only re-rounds
codes and block scales within it, so the runtime kernel and its metadata are
unchanged. Input second moments come from the same 128 OpenWebMath/CodeParrot
calibration sequences, with FourOverSix W4A4 activation quantization applied to
the inputs, in one pass through the BF16-weight model (not layer-sequential).
Evaluation is the released 2048-token WikiText-2 / seed-0 C4 protocol.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.mapped_gptq import factor, quantize_mapped
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_conditional_model import group_name
from run_math_code_calibration import math_code_data
from run_task_reorder_eval import mix_coarse, raw256_masks, validate_compact_masks, validate_evaluation_data

POLICIES = ('rtn_four_over_six', 'rtn_raw256', 'gptq_four_over_six', 'gptq_raw256', 'gptq_fine8x64')


def policy_masks(policy, calib, prior, modules):
    kind = policy.split('_', 1)[1]
    if kind == 'four_over_six':
        return {n: torch.zeros((math.ceil(m.weight.shape[0] / 256), m.weight.shape[1] // 64), dtype=torch.bool)
                for n, m in modules.items()}, 256
    if kind == 'raw256':
        return raw256_masks(calib, prior), 256
    assert digest_file(calib / 'compact_masks.pt') == prior['compact_mask_sha256']
    bundle = validate_compact_masks(torch.load(calib / 'compact_masks.pt', map_location='cpu', weights_only=True), prior)
    return bundle['fine8x64'], 8


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', choices=POLICIES, required=True)
    ap.add_argument('--calib', type=Path, default=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', 4))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    prior = json.loads((args.calib / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_wiki_calibration'] and not prior['uses_c4_calibration']
    assert transformers.__version__ == prior['transformers_version']
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', policy=args.policy, calibration=str(args.calib), source=prior['source'],
                  revision=prior['revision'], job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  source_sha256={p: digest_file(p) for p in (
                      'run_mapped_gptq.py', 'quantize/mapped_gptq.py', 'quantize/branched_format.py',
                      'quantize/quantizer.py', 'run_task_reorder_eval.py')},
                  activation='FourOverSix tensor-wide factor', attention='sdpa', length=2048,
                  hessian='E[xq xq^T], xq = FourOverSix-quantized input, BF16 weights, one pass, 1% damping',
                  calibration_sources=['OpenWebMath', 'CodeParrot'], uses_wiki_calibration=False,
                  uses_c4_calibration=False, evaluation={}, matrices={})
    save(args.out, report)
    model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
                                                 torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                 device_map='cuda')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and m is not model.get_output_embeddings()}
    assert list(modules) == list(prior['matrices'])
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    masks, tile_rows = policy_masks(args.policy, args.calib, prior, modules)
    report['tile_rows'] = tile_rows
    report['elected_tiles'] = sum(int(mask.sum()) for mask in masks.values())

    moments, counts, groups = {}, {}, {}
    if args.policy.startswith('gptq'):
        fit, report['fit'] = math_code_data(tok, prior['fit'])
        for n in modules:
            groups.setdefault(group_name(n), []).append(n)

        def make_hook(key):
            def hook(module, inputs):
                x = quant_nvfp4_4over6(inputs[0], 4, 16).reshape(-1, inputs[0].shape[-1]).float()
                moments[key] = moments.get(key, 0) + x.T @ x
                counts[key] = counts.get(key, 0) + len(x)
            return hook
        handles = [modules[ns[0]].register_forward_pre_hook(make_hook(k)) for k, ns in groups.items()]
        start = time.perf_counter()
        for source in ('math', 'code'):
            for ids in fit[source]:
                model(input_ids=ids.cuda(), use_cache=False)
        for h in handles:
            h.remove()
        report['hessian_seconds'] = time.perf_counter() - start
        report['hessian_tokens'] = sorted(set(counts.values()))
        save(args.out, report)

    start = time.perf_counter()
    u, checked = None, 0
    for index, (n, m) in enumerate(modules.items()):
        w = m.weight.detach()
        mask = masks[n]
        if args.policy.startswith('rtn'):
            q = quant_nvfp4_4over6(w, 4, 16)
            if bool(mask.any()):
                q = mix_coarse(q, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'), mask)
            if index % 32 == 0 or (bool(mask.any()) and checked < 3):
                checked += bool(mask.any())
                # RTN limit of the GPTQ quantizer must reproduce the evaluated RTN weights.
                diag = torch.eye(w.shape[1], device=w.device, dtype=torch.float64)
                limit = quantize_mapped(w, diag, mask, tile_rows)
                report.setdefault('rtn_limit_agreement', {})[n] = float((limit == q).float().mean())
        else:
            key = group_name(n)
            if groups[key][0] == n:
                u = factor(moments.pop(key) / counts[key])
            q = quantize_mapped(w, u, mask, tile_rows)
            e = q.double() - w.double()
            report['matrices'][n] = dict(relative_weight_error=float(e.square().sum() / w.double().square().sum()))
        m.weight.copy_(q)
        del q
        if u is not None and groups[group_name(n)][-1] == n:
            u = None
        if (index + 1) % 28 == 0:
            print(f'QUANTIZED {index + 1}/{len(modules)} {time.perf_counter() - start:.0f}s', flush=True)
            save(args.out, report)
    report['quantize_seconds'] = time.perf_counter() - start
    del moments
    torch.cuda.empty_cache()

    batches, report['data'] = data(tok, prior, 2048)
    published = json.loads(Path('results/kse_paper/job_336566/llama8b/report.json').read_text())
    validate_evaluation_data(report['data'], published['data'])
    report['published_token_windows_verified'] = True
    save(args.out, report)
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])) for m in modules.values()]
    for domain, sequences in batches.items():
        values = []
        for index, ids in enumerate(sequences):
            ids = ids.cuda()
            logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
            value = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)))
            assert math.isfinite(value)
            values.append(value)
            del logits
        losses = torch.tensor(values, dtype=torch.float32) * 2048
        key = 'c4' if domain == 'c4_paper' else domain
        report['evaluation'][key] = dict(nll=values, windows=len(values),
                                         ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
        print(f'PPL {args.policy} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
        save(args.out, report)
    for h in handles:
        h.remove()
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()
