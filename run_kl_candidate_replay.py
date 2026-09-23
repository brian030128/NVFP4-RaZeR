"""Finite-loss replay of the KL-grouped, unrefined candidate on development data.

Step 1 of this branch selected a layout by spectral co-clustering on the KL
channel with the hinge refinement disabled, elected under the unchanged CE/KL
`k=3` conjunction. On Llama layer 31 `down_proj` it elects 36 tiles against
identity's 22, at a fit-over-null of 15.92 against the deployed search's 3.46,
and `gate_proj`/`up_proj` elect below identity, so the candidate is `down_proj`
alone. All of that is held-out surrogate score. This script measures actual loss.

It is self contained on purpose. The original replay chain wrote its capture and
state caches into a worker's job-local `/tmp`, so those are gone, and its two
stages passed caches between jobs by pinning a node. Everything needed to rebuild
persists on `/work`: the calibration report, `compact_masks.pt` for the raw256
background, and the three recorded document sets. This captures, audits and
replays in one job instead.

Protocol. These 192 documents are the previously observed development pool, made
of one rejected confirmation set and two further rejected confirmation sets. They
are explicitly **not** independent validation and no result here promotes
anything. A candidate that improves here still has to be frozen and confirmed on
64 new documents under the existing CE-primary gate before any perplexity run.
Prior failures stay failures.

Two controls, both required by that gate: the full raw 256x64 model, and a
matched identity that elects on the same split with the same rule but no
permutation, which controls for the election sample as well as the background.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_c4_frozen import digest_file
from run_conditional_format import sha
from run_llama_suffix_diagnose import ce, write
from run_math_code_calibration import load_model
from run_reorder_replay import paired_bounds
from run_task_reorder_eval import mix_coarse, raw256_masks

ROOT = Path('/work/u4320956/task_reorder/transfer_20260920/llama8b')
# The previously observed development pool, in the order the earlier study used.
DEVELOPMENT = ('confirmation', 'gate_up_confirm', 'tile_refine_confirm')


def load_development(limit=None):
    """Return the recorded development documents with their digests verified."""
    records, provenance = [], []
    for name in DEVELOPMENT:
        report = json.loads((ROOT / name / 'report.json').read_text())
        if report.get('status') != 'complete':
            raise ValueError(f'{name} is not a complete recorded set')
        if report.get('passed') is not False and name != 'confirmation':
            raise ValueError(f'{name} is not a recorded failure; do not reuse it as development')
        path = ROOT / name / 'fresh.pt'
        if digest_file(path) != report['fresh_sha256']:
            raise ValueError(f'{name} fresh.pt digest mismatch')
        rows = torch.load(path, map_location='cpu', weights_only=True)
        records.extend(rows)
        provenance.append(dict(name=name, documents=len(rows), sha256=report['fresh_sha256']))
    if limit is not None:
        # Smoke mode keeps both domains rather than the first n of one source.
        keep = [i for source in ('math', 'code')
                for i in [j for j, r in enumerate(records) if r['source'] == source][:max(1, limit // 2)]]
        records = [records[i] for i in sorted(keep)]
    return records, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layout', type=Path, required=True,
                    help="down_proj layout.pt from the step-1 spectral/KL run")
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--documents', type=int, default=None,
                    help='Smoke mode: use this many development documents instead of all 192')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run this on a Slurm worker with a GPU')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False

    args.out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((ROOT / 'calibration/report.json').read_text())
    names = list(prior['reorder_modules'])
    if [n.split('.')[-1] for n in names] != ['gate_proj', 'up_proj', 'down_proj']:
        raise ValueError('Unexpected final-MLP module order')
    target = names[2]
    layout = torch.load(args.layout, map_location='cpu', weights_only=True)
    if layout['config']['tile_rows'] != 256 or layout['config']['tile_cols'] != 64:
        raise ValueError('Candidate must keep the deployed 256x64 geometry')
    records, provenance = load_development(args.documents)

    plan = dict(
        status='frozen', protocol_family='development_only_finite_loss',
        development_only=True, previously_observed=list(DEVELOPMENT),
        independent_validation=False, new_fresh_required_before_ppl=True,
        no_candidate_promotion=True, target=target,
        candidate='spectral co-clustering on KL, no hinge refinement, CE/KL k=3 election',
        elected_tiles=int(layout['mask'].sum()),
        identity_elected_tiles=int(layout['identity_mask'].sum()) if 'identity_mask' in layout else None,
        layout_sha256=digest_file(args.layout), source_sha256=digest_file(__file__),
        documents=len(records), document_provenance=provenance,
        smoke=args.documents is not None)
    write(args.out / 'plan.json', plan)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], plan=plan,
                  completed_sequences=0, metrics={k: [] for k in ('raw256', 'identity', 'candidate')},
                  audits=dict(raw_suffix=0))
    write(args.out / 'report.json', report)

    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    raw = raw256_masks(ROOT / 'calibration', prior)
    prepared = {}
    for name, module in modules.items():
        if sha(module.weight) != prior['matrices'][name]['source_sha256']:
            raise ValueError(f'Source weight mismatch for {name}')
        base = quant_nvfp4_4over6(module.weight, 4, 16)
        if name == target or raw[name].any():
            # alt is the E0M3 alpha=1 candidate; elect='always' forces it
            # everywhere, so its type_block does not matter here.
            alt = quant_mix_4_6(module.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            mixed = mix_coarse(base, alt, raw[name])
            if name == target:
                prepared[name] = dict(
                    raw=mixed,
                    candidate=original_order_weight_reference(base, alt, layout, 'reordered'),
                    identity=original_order_weight_reference(base, alt, layout, 'identity'))
            base = mixed
        module.weight.copy_(base)

    last = model.model.layers[-1]
    head, norm = model.get_output_embeddings(), model.model.norm
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:]))
        for m in modules.values()]
    captured, capture = {}, True

    def remember(key):
        def hook(module, inputs):
            if capture:
                captured[key] = inputs[0].detach().clone()
        return hook

    h1 = last.post_attention_layernorm.register_forward_pre_hook(remember('residual'))
    h2 = last.mlp.register_forward_pre_hook(remember('x'))

    start = time.monotonic()
    for index, row in enumerate(records):
        ids = row['ids'].cuda()
        modules[target].weight.copy_(prepared[target]['raw'])
        capture = True
        logits = model(input_ids=ids, use_cache=False).logits
        raw_full = ce(logits, ids)
        del logits
        capture = False
        for policy in ('raw', 'identity', 'candidate'):
            modules[target].weight.copy_(prepared[target][policy])
            logits = head(norm(captured['residual'] + last.mlp(captured['x'])))
            value = ce(logits, ids)
            del logits
            if policy == 'raw':
                # The cached suffix must reproduce the full model bit for bit,
                # otherwise every later comparison is measuring the shortcut.
                if value != raw_full:
                    raise ValueError(f'Suffix replay differs from full model at {index}: '
                                     f'{value} vs {raw_full}')
                report['audits']['raw_suffix'] += 1
            report['metrics']['raw256' if policy == 'raw' else policy].append(value)
        report['completed_sequences'] = index + 1
        write(args.out / 'report.json', report)
        if (index + 1) % 16 == 0:
            print('REPLAY', index + 1, 'seconds', round(time.monotonic() - start, 2), flush=True)
    h1.remove()
    h2.remove()
    for handle in handles:
        handle.remove()

    paired = {}
    for reference in ('raw256', 'identity'):
        domains = {}
        for domain in ('all', 'math', 'code'):
            idx = [i for i, r in enumerate(records) if domain == 'all' or r['source'] == domain]
            domains[domain] = paired_bounds([report['metrics']['candidate'][i] for i in idx],
                                            [report['metrics'][reference][i] for i in idx])
        paired[reference] = domains
    report['paired'] = paired
    # The development screen, deliberately the same shape as the frozen fresh
    # gate so a pass here is a reason to draw new documents and nothing more.
    report['development_screen'] = all(
        paired[ref]['all']['mean'] + 2 * paired[ref]['all']['se'] < 0
        for ref in ('raw256', 'identity')) and all(
        paired['raw256'][domain]['mean'] <= 0 for domain in ('math', 'code'))
    report['status'] = 'complete'
    write(args.out / 'report.json', report)
    print('RESULT ' + json.dumps(dict(
        documents=len(records), audits=report['audits'],
        development_screen=report['development_screen'],
        raw256=paired['raw256']['all'], identity=paired['identity']['all'],
        math=paired['raw256']['math']['mean'], code=paired['raw256']['code']['mean'])), flush=True)


if __name__ == '__main__':
    main()
