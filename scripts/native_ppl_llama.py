"""Native-kernel perplexity on the published windows, against the simulated reference.

The MixFP4 report carries quality from a BF16 fake-quantization simulator and
speed from the native SM100 kernel, and no single artifact has shown both. The
native path's own limitation note records that mixed-policy full logits do not
match the reference -- about a 10.8% relative gap -- while single-layer replay
finds only 0-4 BF16 differences per projection against FP32/FP64 oracles, which
later FP4 activation quantization amplifies. Two correct implementations that
round in different orders diverge exactly that way, so full-logit equivalence is
the wrong acceptance test. Perplexity parity is the one that matters, and this
measures it.

Setup is copied from `benchmark_native_llama.py` so the packing audits are the
same ones that gate the latency work: every matrix is packed, decoded and
compared bitwise against its simulator reference before any forward runs. Only
the measurement differs -- the published evaluation protocol replaces the timing
loop.

`base` is the control that makes the result readable. If native FourOverSix
reproduces the published 6.875525, the harness is sound and any `arranged`
discrepancy is about the mixed-format path specifically rather than about this
script.

Run on GB200 (SM100). Nothing here elects tiles, changes a map, or promotes a
candidate; it re-measures frozen maps on frozen windows.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_native_llama import PolicyLinear, ROOT, digest, write
from native_model_runtime import Linear, Runtime, decode
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference

# Published simulated values this run is compared against, from
# results/task_reorder/transfer_20260920/renewed_llama/joint192_ppl and the
# report's section 4 table. Policy names are the benchmark's.
def tensor_sha(t):
    """Token-window digest, matching run_conditional_format.sha.

    Inlined rather than imported: that module pulls in `datasets`, which the
    native venv does not carry and must not be made to carry, since it is shared
    with the gated latency jobs.
    """
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


REFERENCE = {'base': dict(wiki=6.875525, c4=9.823733, label='FourOverSix'),
             'raw': dict(wiki=6.866879, c4=9.801361, label='raw 256x64'),
             'arranged': dict(wiki=6.864886, c4=9.796946, label='refined 147-tile')}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--kernel-gate', type=Path, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--policies', default='base,arranged',
                    help='Comma separated subset of base,raw,arranged')
    ap.add_argument('--windows', type=Path, required=True,
                    help='Frozen published windows from prepare_published_windows.py')
    ap.add_argument('--limit-windows', type=int, default=None,
                    help='Smoke mode: evaluate only this many windows per domain')
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    assert torch.cuda.get_device_capability(0)[0] == 10, 'Native path needs SM100 (GB200)'
    policies = [p.strip() for p in args.policies.split(',') if p.strip()]
    assert all(p in REFERENCE for p in policies), policies
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    PolicyLinear.audit_dir = out

    rt = Runtime(args.lib)
    prior = json.loads((ROOT / 'calibration/report.json').read_text())
    # Same gate the latency jobs require: a passed kernel check whose recorded
    # library digest matches the library actually loaded here.
    assert json.loads((args.kernel_gate / 'report.json').read_text())['status'] == 'passed'
    assert digest(args.lib) == json.loads(
        (args.kernel_gate / 'library.json').read_text())['library_sha256']
    masks = torch.load(ROOT / 'calibration/compact_masks.pt', map_location='cpu', weights_only=True)
    assert digest(ROOT / 'calibration/compact_masks.pt') == prior['compact_mask_sha256']
    assert masks['revision'] == prior['revision']
    assert masks['weight_sha256'] == {n: v['source_sha256'] for n, v in prior['matrices'].items()}

    layouts, hashes = {}, {}
    for path in sorted((ROOT / 'joint192/layouts').glob('*/layout.pt')):
        layout = torch.load(path, map_location='cpu', weights_only=True)
        layouts[layout['name']] = layout
        hashes[str(path)] = digest(path)
    assert len(layouts) == 3
    tiles = sum(int(l['mask'].sum()) for l in layouts.values()) + sum(
        int(m.sum()) for n, m in masks['raw256'].items() if n not in layouts)
    assert tiles == 147, tiles

    report = dict(status='preparing', job=os.environ['SLURM_JOB_ID'],
                  device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  source=prior['source'], revision=prior['revision'],
                  layout_sha256=hashes, total_tiles=tiles, policies=policies,
                  library_sha256=digest(args.lib), length=2048,
                  protocol='published windows; wiki uses cache, c4 does not; '
                           'float32 NLL aggregation; exp(sum/(windows*2048))',
                  reference_provenance='simulated values from joint192_ppl report 405707',
                  limit_windows=args.limit_windows, elects_nothing=True,
                  source_sha256={p: digest(p) for p in (
                      'scripts/native_ppl_llama.py', 'scripts/benchmark_native_llama.py',
                      'scripts/native_model_runtime.py', 'native/model_runtime.cu',
                      'quantize/quantizer.py', 'quantize/task_reorder.py',
                      'run_baseline_protocol_audit.py')},
                  matrix_audits=[], evaluation={})
    write(out / 'report.json', report)

    checkpoint = Path(os.environ['HF_HOME']) / 'hub' / (
        'models--' + prior['source'].replace('/', '--')) / 'snapshots' / prior['revision']
    assert checkpoint.is_dir(), checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa', local_files_only=True).eval()

    # Packing, byte for byte against the simulator's weights. Copied from
    # benchmark_native_llama.main so the same assertions gate this measurement.
    started = time.perf_counter()
    modules = []
    for index, (name, meta) in enumerate(prior['matrices'].items()):
        old = model.get_submodule(name)
        w = old.weight.data
        sha = hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert sha == meta['source_sha256'], name
        assert old.bias is None, name
        base = Linear(rt, w)
        base_ref = quant_nvfp4_4over6(w, 4, 16)
        decoded = decode(base.packed, base.flat_scales, base.global_scale).to(torch.bfloat16)
        assert torch.equal(decoded, base_ref), (name, 'base packing')
        del decoded
        variants, refs = {'base': base}, {'base': base_ref}
        rawmask = masks['raw256'][name]
        if rawmask.any():
            raw = Linear(rt, w, rawmask)
            alt = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            fullmask = rawmask.cuda().repeat_interleave(256, 0).repeat_interleave(
                64, 1)[:w.shape[0], :w.shape[1]]
            rawref = torch.where(fullmask, alt, base_ref)
            del fullmask
            decoded = decode(raw.packed, raw.flat_scales, raw.global_scale,
                             raw.mask.bool()).to(torch.bfloat16)
            assert torch.equal(decoded, rawref), (name, 'raw packing')
            del decoded
        else:
            raw, rawref, alt = base, base_ref, None
        variants['raw'], refs['raw'] = raw, rawref
        if name in layouts:
            layout = layouts[name]
            arranged = Linear(rt, w, layout['mask'], layout['row_perm'], layout['col_perm'])
            if alt is None:
                alt = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            arrref = original_order_weight_reference(base_ref, alt, layout)
            decoded = decode(arranged.packed, arranged.flat_scales, arranged.global_scale,
                             arranged.mask.bool() if arranged.mask is not None else None
                             ).to(torch.bfloat16)
            decoded = decoded[torch.argsort(layout['row_perm']).cuda()][
                :, torch.argsort(layout['col_perm']).cuda()]
            assert torch.equal(decoded, arrref), (name, 'arranged packing')
            del decoded
        else:
            arranged, arrref = raw, rawref
        variants['arranged'], refs['arranged'] = arranged, arrref
        replacement = PolicyLinear(variants, refs)
        replacement.name = name
        model.set_submodule(name, replacement)
        modules.append(replacement)
        report['matrix_audits'].append(
            dict(name=name, source_sha256=sha, packed_weights_bitwise=True))
        del w, old, alt
        if (index + 1) % 32 == 0:
            print(f'PACKED {index + 1}/{len(prior["matrices"])} '
                  f'{round(time.perf_counter() - started, 2)}s', flush=True)
    assert len(modules) == len(prior['matrices'])
    report['status'] = 'weights_audited'
    write(out / 'report.json', report)

    # Frozen windows rather than a re-derivation, so the native run consumes the
    # same token tensors the simulated run did; the digests are re-checked here.
    frozen = torch.load(args.windows, map_location='cpu', weights_only=False)
    assert frozen['revision'] == prior['revision'] and frozen['length'] == 2048
    batches, report['data'] = frozen['batches'], frozen['meta']
    for domain, windows in batches.items():
        assert [tensor_sha(w) for w in windows] == frozen['meta'][domain]['token_sha256'], domain
    report['windows_sha256'] = digest(args.windows)
    if args.limit_windows:
        batches = {d: b[:args.limit_windows] for d, b in batches.items()}
    print('WINDOWS ' + json.dumps({d: len(b) for d, b in batches.items()}), flush=True)

    PolicyLinear.reference = False
    PolicyLinear.audit = False
    for policy in policies:
        PolicyLinear.policy = policy
        evaluation = {}
        for domain, sequences in batches.items():
            values = []
            for index, ids in enumerate(sequences):
                ids = ids.cuda()
                logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
                value = float(F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1)))
                assert math.isfinite(value), (policy, domain, index)
                values.append(value)
                del logits
                if (index + 1) % 32 == 0:
                    print(f'EVAL {policy} {domain} {index + 1}/{len(sequences)} '
                          f'{round(time.perf_counter() - started, 1)}s', flush=True)
            losses = torch.tensor(values, dtype=torch.float32) * 2048
            key = 'c4' if domain == 'c4_paper' else domain
            evaluation[key] = dict(nll=values, windows=len(values),
                                   ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
            print(f'PPL {policy} {key} {evaluation[key]["ppl"]:.6f}', flush=True)
        # A partial sweep is not comparable with the published full-window value,
        # so the delta is only reported when every window was evaluated.
        full = args.limit_windows is None
        evaluation['reference'] = REFERENCE[policy]
        evaluation['delta_vs_simulated'] = {
            key: evaluation[key]['ppl'] - REFERENCE[policy][key] for key in ('wiki', 'c4')
        } if full else None
        report['evaluation'][policy] = evaluation
        write(out / 'report.json', report)

    report['status'] = 'complete'
    write(out / 'report.json', report)
    print('RESULT ' + json.dumps({p: {
        'label': REFERENCE[p]['label'],
        'wiki': report['evaluation'][p]['wiki']['ppl'],
        'c4': report['evaluation'][p]['c4']['ppl'],
        'delta': report['evaluation'][p]['delta_vs_simulated']} for p in policies}), flush=True)


if __name__ == '__main__':
    main()
