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

POLICIES = ('rtn_four_over_six', 'rtn_raw256', 'gptq_four_over_six', 'gptq_raw256', 'gptq_fine8x64',
            'rtn_rule', 'rtn_mse1x16', 'rtn_fine1x16', 'rtn_mapfile')


def bound(mean, std, k, n=128):
    return mean + k * std / math.sqrt(n)


def rule_masks(rule, scores, frozen, modules):
    """rule = OBJECTIVE:K:TILE_ROWS, OBJECTIVE in {both, ce, kl}, TILE_ROWS in {8, 256}.

    Uses the per-tile statistics persisted by run_math_code_calibration --summary-scores and
    first checks that k=3 CE+KL reproduces the frozen compact maps exactly."""
    objective, k, rows = rule.split(':'); k = float(k); rows = int(rows)
    assert objective in ('both', 'ce', 'kl') and rows in (8, 256)
    masks, mismatched = {}, 0
    for i, (n, m) in enumerate(modules.items()):
        s = torch.load(scores / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert s['name'] == n
        o, c = s['shape']
        stats = {}
        for key in ('ce', 'kl'):
            seq = s[key + '_seq256']
            stats[key, 256] = lambda kk, seq=seq: bound(seq.mean(0), seq.std(0, unbiased=True), kk)
            mean, std = s[key + '_mean8'].double(), s[key + '_std8'].double()
            stats[key, 8] = lambda kk, mean=mean, std=std, o=o, c=c: bound(mean, std, kk).reshape(o // 8, c // 64)
        for tr in (() if frozen is None else (8, 256)):
            check = torch.maximum(stats['ce', tr](3.), stats['kl', tr](3.)) < 0
            mismatched += int((check != frozen['raw256' if tr == 256 else 'fine8x64'][n]).sum())
        if objective == 'both':
            masks[n] = torch.maximum(stats['ce', rows](k), stats['kl', rows](k)) < 0
        else:
            masks[n] = stats[objective, rows](k) < 0
    return masks, rows, mismatched


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
    ap.add_argument('--model', choices=('llama8b', 'qwen27b'), default='llama8b')
    ap.add_argument('--calib', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--map', type=Path, help='Frozen {module: bool tile mask} for --policy rtn_mapfile')
    ap.add_argument('--map-tile-rows', type=int, default=256)
    ap.add_argument('--rule', help='OBJECTIVE:K:TILE_ROWS for --policy rtn_rule')
    ap.add_argument('--fine-dir', type=Path, default=Path('/work/u4320956/mixfp4_potential/llama8b_fine1x16/fine_masks'))
    ap.add_argument('--fine-rule', default='both', help='both|ce|kl, optionally suffixed _k0/_k1/_k2')
    ap.add_argument('--scores', type=Path, default=None)
    args = ap.parse_args()
    qwen = args.model == 'qwen27b'
    if args.calib is None:
        args.calib = Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration' if qwen
                          else '/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration')
    if args.scores is None:
        args.scores = Path('/work/u4320956/mixfp4_potential/qwen27b_scores' if qwen
                           else '/work/u4320956/mixfp4_potential/llama8b_calibration/scores')
    assert not qwen or args.policy in ('rtn_rule', 'rtn_four_over_six', 'rtn_mapfile'), 'Qwen supports RTN rule arms only'
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
    loader = AutoModelForCausalLM
    if qwen:
        from transformers import Qwen3_5ForConditionalGeneration
        loader = Qwen3_5ForConditionalGeneration
    model = loader.from_pretrained(prior['source'], revision=prior['revision'], torch_dtype=torch.bfloat16,
                                   attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and (('language_model' in n and 'head' not in n) if qwen else m is not model.get_output_embeddings())}
    assert list(modules) == list(prior['matrices'])
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    if args.policy == 'rtn_rule':
        # Qwen summaries were converted from the full historical tables with the raw256
        # equality asserted per shard (convert_scores_summary.py); Llama re-checks here.
        frozen = None if qwen else validate_compact_masks(torch.load(args.calib / 'compact_masks.pt', map_location='cpu',
                                                                     weights_only=True), prior)
        masks, tile_rows, mismatched = rule_masks(args.rule, args.scores, frozen, modules)
        report.update(rule=args.rule, scores=str(args.scores), frozen_k3_mismatched_tiles=mismatched)
        assert mismatched == 0, f'Re-scored k=3 maps differ from frozen maps in {mismatched} tiles'
    elif args.policy == 'rtn_mapfile':
        masks, tile_rows = torch.load(args.map, map_location='cpu', weights_only=True), args.map_tile_rows
        assert set(masks) == set(modules)
        report.update(map=str(args.map), map_sha256=digest_file(args.map))
    elif args.policy == 'rtn_fine1x16':
        import numpy as np
        masks, tile_rows = {}, 1
        for i, n in enumerate(modules):
            f = torch.load(args.fine_dir / f'{i:03d}.pt', map_location='cpu', weights_only=True)
            assert f['name'] == n and f['k'] == 3
            count = f['shape'][0] * f['shape'][1]
            bits = np.unpackbits(f['packed'][args.fine_rule].numpy())[:count]
            masks[n] = torch.from_numpy(bits.astype(bool)).reshape(f['shape'])
        report.update(fine_rule=args.fine_rule, fine_dir=str(args.fine_dir))
    elif args.policy == 'rtn_mse1x16':
        masks, tile_rows = {n: torch.zeros(1, 1, dtype=torch.bool) for n in modules}, 1
    else:
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
        if args.policy == 'rtn_fine1x16':
            # Accuracy ceiling for granularity with the task-loss rule: E0M3 per scale block.
            b = quant_nvfp4_4over6(w, 4, 16)
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            pick = mask.to(w.device).reshape(-1)
            q = torch.where(pick[:, None], a.reshape(-1, 16), b.reshape(-1, 16)).reshape(w.shape)
            del a, b
        elif args.policy == 'rtn_mse1x16':
            # Accuracy ceiling for granularity: choose E0M3 alpha=1 or FourOverSix per scale block.
            b = quant_nvfp4_4over6(w, 4, 16)
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            pick = ((a.float() - w.float()).square().reshape(-1, 16).sum(-1)
                    < (b.float() - w.float()).square().reshape(-1, 16).sum(-1))
            q = torch.where(pick[:, None], a.reshape(-1, 16), b.reshape(-1, 16)).reshape(w.shape)
            report['elected_tiles'] = report.get('elected_tiles', 0) + int(pick.sum())
            del a, b
        elif args.policy.startswith('rtn'):
            q = quant_nvfp4_4over6(w, 4, 16)
            if bool(mask.any()):
                a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                if tile_rows == 256:
                    q = mix_coarse(q, a, mask)
                else:
                    q = torch.where(mask.to(w.device).repeat_interleave(tile_rows, 0).repeat_interleave(64, 1), a, q)
                del a
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
    published = json.loads(Path(f'results/kse_paper/job_{"336969" if qwen else "336566"}/{args.model}/report.json').read_text())
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
