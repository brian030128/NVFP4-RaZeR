"""MixFP4 tile selection by gradient training of per-tile logits on the KL loss (no backtracking).

Every type tile u of every text linear layer gets a real latent logit theta_u.
The weights are W(theta) = B + expand(m(theta)) * (A - B), with B the FourOverSix
E2M1 candidate and A the E0M3 alpha=1 candidate (identical to run_multiround.py).

  --alt scale      NVFP4-only control: A is instead the FourOverSix E2M1 candidate
                   with the OTHER block scale (block max -> 6 <-> block max -> 4) in
                   every 16-element scale block (quant_nvfp4_4over6_pair). With
                   --unit 1x16 this is a KL-trained FourOverSix block-scale search:
                   plain NVFP4, no E0M3, no metadata beyond the existing ue4m3 scale.
  --alt joint      MixFP4 + NVFP4 scale search, trained together (STE only). Every
                   16-element scale block b also gets a logit psi_b choosing its
                   FourOverSix scale: E = B + s * (F - B), s = 1[psi > 0], F the
                   flipped-scale candidate; the weight is W = E + t * (A - E) with the
                   tile type t and A the E0M3 candidate. Straight-through gradients:
                   dL/dt_u = <G, P_u (A - E)>, dL/ds_b = <G, P_b (1 - t)(F - B)>.

  --param ste      m = 1[theta > 0] in the forward pass (the deployable hard map),
                   straight-through backward dm/dtheta = 1 (BinaryConnect-style
                   latent weights). The gradient dKL/dm_u = <G, P_u (A - B)> is the
                   same first-order tile score run_multiround.py ranks by.
  --param sigmoid  m = sigmoid(theta / tau), a soft interpolation with tau annealed
                   geometrically from --tau-start to --tau-end; the map is rounded
                   (theta > 0) for every development evaluation and at the end.

The loss is the token-mean KL to the BF16 teacher on the 128 math/code calibration
sequences, with the scoring forward convention of run_multiround.py (causal
per-token FourOverSix activations, straight-through). Weight gradients are never
materialized as parameters: a backward hook on each linear output forms
G = dy^T x for the batch and reduces G * (A - B) to tiles. The optimizer updates
the logits; there is no candidate filter and no acceptance test. The 192 held-out
development documents are evaluated (hard map, tensor-wide per-document
activations, as in evaluation) only to MONITOR training; they never change the
map. WikiText-2 / C4 are evaluated once, on the final map.
"""
import argparse
import json
import math
import os
import random
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from quantize.causal_four_over_six import quantize_rows
from quantize.fast_act import check as check_act, quant_per_document
from quantize.packed_candidates import decode_alt, decode_base, nbytes, pack
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6, quant_nvfp4_4over6_pair
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_multiround import CALIBRATIONS, DEVELOPMENT, PUBLISHED, UNITS, expand, load_development, reduce
from run_task_reorder_eval import validate_evaluation_data

# Llama-3.2-1B/3B-Instruct share the Llama-3 tokenizer, so the base Llama development
# documents and published evaluation windows apply unchanged (make_instruct_prior.py).
for _name in ('llama1b_ins', 'llama3b_ins'):
    CALIBRATIONS[_name] = Path(f'/work/u4320956/mixfp4_potential/{_name}_calibration')
    DEVELOPMENT[_name] = DEVELOPMENT['llama8b']
    PUBLISHED[_name] = PUBLISHED['llama8b']


def extra_math_code(tok, previous_fit, per_source, model):
    """`per_source` more 512-token windows from each of the pinned math/code parquet shards.

    Documents are taken in stream order, skipping the 128 pinned calibration documents and
    every development document (by document_sha256 from the development manifests). The
    window offset is derived from the document hash, so the selection is deterministic."""
    import hashlib
    from datasets import load_dataset
    excluded = {d['document_sha256'] for meta in previous_fit.values() for d in meta['documents']}
    for directory in DEVELOPMENT[model]:
        manifest = json.loads((directory / 'fresh_manifest.json').read_text())
        excluded |= {r['document_sha256'] for r in manifest['records']}
    windows, records = [], []
    for source in ('math', 'code'):
        meta = previous_fit[source]
        stream = load_dataset(meta['repo'], revision=meta['revision'],
                              data_files={'train': meta['path']}, split='train', streaming=True)
        taken = 0
        for row in stream:
            text = row['text' if source == 'math' else 'content']
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in excluded:
                continue
            excluded.add(digest)
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < 513:
                continue
            offset = int(digest[:12], 16) % (ids.shape[1] - 512 + 1)
            window = ids[:, offset:offset + 512].clone()
            windows.append(window)
            records.append(dict(source=source, document_sha256=digest, offset=offset, token_sha256=sha(window)))
            taken += 1
            if taken == per_source:
                break
        assert taken == per_source, (source, taken)
    return windows, dict(per_source=per_source, excluded_documents=len(excluded) - len(records), records=records)


C4_TRAIN = 'en/c4-train.00000-of-01024.json.gz'


def c4_train_windows(tok, count):
    """`count` 512-token windows of general web text from one C4 TRAIN shard, one window per document,
    in stream order; documents shorter than 513 tokens are skipped. The offset is derived from the document
    hash. Evaluation uses only the C4 validation file and WikiText-2 test, so this never overlaps them."""
    import hashlib
    from datasets import load_dataset
    from run_c4_frozen import REVISION
    stream = load_dataset('allenai/c4', revision=REVISION, data_files={'train': C4_TRAIN}, split='train', streaming=True)
    windows, records, seen = [], [], set()
    for row in stream:
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(row['text'], return_tensors='pt').input_ids
        if ids.shape[1] < 513:
            continue
        offset = int(digest[:12], 16) % (ids.shape[1] - 512 + 1)
        window = ids[:, offset:offset + 512].clone()
        windows.append(window)
        records.append(dict(document_sha256=digest, offset=offset, token_sha256=sha(window)))
        if len(windows) == count:
            break
    assert len(windows) == count, len(windows)
    return windows, dict(repo='allenai/c4', revision=REVISION, path=C4_TRAIN, records=records)


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=tuple(CALIBRATIONS), default='llama8b')
    ap.add_argument('--unit', choices=tuple(UNITS), required=True)
    ap.add_argument('--param', choices=('ste', 'sigmoid'), default='ste')
    ap.add_argument('--alt', choices=('e0m3', 'scale', 'joint'), default='e0m3',
                    help='Candidate A: E0M3 alpha=1 (MixFP4), the other FourOverSix block scale (NVFP4 control), '
                         'or joint: E0M3 tiles plus a trained per-block E2M1 scale')
    ap.add_argument('--scale-init', choices=('four_over_six', 'nvfp4'), default='four_over_six',
                    help='--alt scale/joint: the 6-vs-4 scale pair starts at FourOverSix\'s MSE choice, or at '
                         'plain NVFP4 (max -> 6 everywhere) with no FourOverSix prior')
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default='adam')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=1e-12,
                    help='Adam epsilon; tile gradients are ~1e-6, so the torch default 1e-8 would dominate')
    ap.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument('--schedule', choices=('constant', 'cosine'), default='constant')
    ap.add_argument('--init-logit', type=float, default=-1.0,
                    help='Initial theta of every tile (negative = start at FourOverSix E2M1)')
    ap.add_argument('--tau-start', type=float, default=1.0)
    ap.add_argument('--tau-end', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--scale-epochs', type=int, default=0,
                    help='--alt joint: train only the per-block scale logits for the first N epochs (tiles stay E2M1), '
                         'then let tiles flip on top')
    ap.add_argument('--stage2', choices=('joint', 'tiles'), default='joint',
                    help='After --scale-epochs: keep training the scale with the tiles (joint), or freeze it (tiles)')
    ap.add_argument('--fit-source', choices=('mathcode', 'c4'), default='mathcode',
                    help='Calibration text: the pinned math/code windows, or general web text from a C4 TRAIN shard')
    ap.add_argument('--fit-count', type=int, default=512, help='--fit-source c4: number of 512-token windows')
    ap.add_argument('--dev-source', choices=('mathcode', 'c4'), default='mathcode',
                    help='Monitoring/epoch-selection set; with c4 the math/code set is also reported as dev2')
    ap.add_argument('--dev-count', type=int, default=192, help='--dev-source c4: number of held-out 512-token windows')
    ap.add_argument('--no-epoch-maps', action='store_true', help='Save only the final hard maps (disk quota)')
    ap.add_argument('--no-logits', action='store_true', help='Do not save the FP32 latent logits (disk quota)')
    ap.add_argument('--extra-fit', type=int, default=0,
                    help='Additional calibration windows PER SOURCE (math, code) beyond the pinned 64+64, '
                         'excluding the pinned and development documents')
    ap.add_argument('--batch', type=int, default=8, help='Calibration sequences per forward/backward')
    ap.add_argument('--accum', type=int, default=1, help='Forward/backward passes per optimizer step')
    ap.add_argument('--eval-every', type=int, default=2, help='Epochs between development evaluations (monitor only)')
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpus', type=int, default=None, help='Qwen: 1 loads on one device, else balanced')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = UNITS[args.unit]
    qwen = args.model == 'qwen27b'
    if qwen:
        assert args.batch == 1 and args.eval_batch == 1, 'Qwen batched forward is not identical to batch 1'
    prior = json.loads((CALIBRATIONS[args.model] / 'report.json').read_text())
    assert transformers.__version__ == prior['transformers_version']
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'],
                  args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  source_sha256={p: digest_file(p) for p in ('run_train_map.py', 'run_multiround.py', 'quantize/quantizer.py',
                                                             'quantize/causal_four_over_six.py')},
                  epochs=[])
    save(args.out, report)
    if qwen:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], revision=prior['revision'], dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda' if args.gpus == 1 else 'balanced', output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        model.eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
        assert list(modules) == list(prior['matrices'])
    else:
        model, modules = load_model(prior, False)
        model.set_attn_implementation('sdpa')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    c4_needed = (args.fit_count if args.fit_source == 'c4' else 0) + (args.dev_count if args.dev_source == 'c4' else 0)
    if c4_needed:
        # Calibration windows first, then development windows: disjoint documents of the same C4 train shard.
        c4, c4_meta = c4_train_windows(tok, c4_needed)
    if args.fit_source == 'c4':
        assert not args.extra_fit, '--extra-fit applies to --fit-source mathcode'
        fit = c4[:args.fit_count]
        report['fit_c4'] = dict(c4_meta, records=c4_meta['records'][:args.fit_count])
    else:
        fit, _ = math_code_data(tok, prior['fit'])
        fit = [b for source in ('math', 'code') for b in fit[source]]
        if args.extra_fit:
            extra, report['extra_fit'] = extra_math_code(tok, prior['fit'], args.extra_fit, args.model)
            fit += extra
    report['fit_source'], report['fit_sequences'] = args.fit_source, len(fit)
    save(args.out, report)
    print(f'FIT {len(fit)} {args.fit_source} calibration sequences', flush=True)
    mathcode_dev, report['development'] = load_development(args.model)
    if args.dev_source == 'c4':
        start = args.fit_count if args.fit_source == 'c4' else 0
        dev = [dict(ids=w) for w in c4[start:start + args.dev_count]]
        report['development_c4'] = dict(c4_meta, records=c4_meta['records'][start:start + args.dev_count])
        second_dev = mathcode_dev  # also monitored, as dev2
    else:
        dev, second_dev = mathcode_dev, None
    report['dev_source'] = args.dev_source
    device = model.get_input_embeddings().weight.device

    def teacher_of(records):
        out = []
        for r in records:
            logits = model(input_ids=r['ids'].to(device), use_cache=False).logits
            out.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
        return out
    dev_teacher = teacher_of(dev)
    second_teacher = teacher_of(second_dev) if second_dev is not None else None
    teacher = []
    for ids in fit:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    del logits
    # Candidates exactly as run_multiround.py: packed 4-bit codes + FP8 scales,
    # verified bitwise by pack(), dense BF16 fallback otherwise.
    joint = args.alt == 'joint'
    assert not joint or args.param == 'ste', '--alt joint supports only --param ste'
    assert joint or (args.scale_epochs == 0 and args.stage2 == 'joint'), '--scale-epochs/--stage2 need --alt joint'
    assert args.scale_init == 'four_over_six' or args.alt != 'e0m3', '--scale-init applies to --alt scale/joint'

    def scale_pair(w):
        """The two E2M1 block-scale candidates (lo = start at logit <= 0, hi = logit > 0), both per 16-element block.

        four_over_six: lo = FourOverSix's MSE choice, hi = the scale it rejected.
        nvfp4:         lo = block max -> 6 everywhere (plain NVFP4), hi = block max -> 4 everywhere,
                       so the 6-vs-4 decision is learned from the KL loss with no FourOverSix prior."""
        chosen, other, select_4 = quant_nvfp4_4over6_pair(w, 4, 16)
        if args.scale_init == 'four_over_six':
            return chosen, other
        s = select_4.reshape(-1, 1)
        c, o = chosen.reshape(-1, 16), other.reshape(-1, 16)
        return torch.where(s, o, c).view(w.shape), torch.where(s, c, o).view(w.shape)

    packed, dense, theta = {}, {}, {}
    scale_cands, psi = {}, {}  # joint only: (lo, hi) E2M1 scale candidates and per-scale-block logits
    nvfp4_equal = []
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        b = quant_nvfp4_4over6(m.weight, 4, 16)
        if args.alt in ('scale', 'joint'):
            lo, hi = scale_pair(m.weight)
            if args.scale_init == 'four_over_six':
                assert torch.equal(lo, b), n
            else:
                nvfp4_equal.append(bool(torch.equal(lo, quant_nvfp4(m.weight, 4, 16))))
        if joint:
            scale_cands[n] = (lo, hi)
            psi[n] = torch.full((m.weight.shape[0], m.weight.shape[1] // 16), args.init_logit,
                                dtype=torch.float32, device=m.weight.device)
        if args.alt == 'scale':
            a, b = hi, lo
            p = None  # pack() stores only the E0M3 alternative; keep both candidates dense
        else:
            a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            p = pack(m.weight, b, a)  # b = FourOverSix: only the packing reference under joint
        if p is None:
            dense[n] = (b, a)
        else:
            packed[n] = p
        o, k = m.weight.shape
        assert k % cols == 0
        theta[n] = torch.full((-(-o // rows), k // cols), args.init_logit, dtype=torch.float32, device=m.weight.device)
        m.weight.copy_(scale_cands[n][0] if joint else b)
        del a, b
    if nvfp4_equal:
        # Informational: the max->6 branch vs quant_nvfp4 (which differ only if quant_nvfp4 rounds differently).
        report['lo_equals_quant_nvfp4_modules'] = sum(nvfp4_equal)
        print(f'lo == quant_nvfp4 on {sum(nvfp4_equal)}/{len(nvfp4_equal)} modules', flush=True)
    torch.cuda.empty_cache()
    report['candidate_storage'] = dict(packed_modules=len(packed), dense_fallback_modules=sorted(dense),
                                       packed_gib=sum(nbytes(p) for p in packed.values()) / 2 ** 30)
    report['tiles'] = sum(t.numel() for t in theta.values())
    report['scale_blocks'] = sum(t.numel() for t in psi.values())
    print('CANDIDATES ' + json.dumps(report['candidate_storage']) + f' tiles={report["tiles"]} '
          f'scale_blocks={report["scale_blocks"]}', flush=True)

    def base(n):
        return dense[n][0] if n in dense else decode_base(packed[n])

    def alt(n):
        return dense[n][1] if n in dense else decode_alt(packed[n])

    def e2m1(n):
        """The E2M1 branch: base(n), or under --alt joint each scale block's trained scale choice."""
        if not joint:
            return base(n)
        lo, hi = scale_cands[n]
        return torch.where(expand(psi[n] > 0, 1, 16), hi, lo)

    tau = [args.tau_start]

    def apply(n, hard):
        b = e2m1(n)
        if hard or args.param == 'ste':
            # Bitwise the multiround decode: each element is exactly B or A.
            modules[n].weight.copy_(torch.where(expand(theta[n] > 0, rows, cols, b.shape[0]), alt(n), b))
            return
        m = expand(torch.sigmoid(theta[n] / tau[0]), rows, cols, b.shape[0])
        modules[n].weight.copy_((b.float() + m * (alt(n).float() - b.float())).to(b.dtype))

    def hard_map():
        return {n: (t > 0) for n, t in theta.items()}

    eval_handles = []
    act_checks = [0]
    current_batch = [1]

    def per_document_act(module, inputs):
        # Tensor-wide activation scale per document, as in run_multiround.py's dev_eval.
        x = inputs[0]
        batch = current_batch[0]
        if batch == 1 or (x.dim() >= 3 and x.shape[0] == batch):
            view = x
        elif x.shape[0] % batch == 0 and x.shape[0] // batch > 1:
            view = x.reshape(batch, -1, x.shape[-1])
        else:
            raise RuntimeError(f'Cannot recover the document axis of a {tuple(x.shape)} input at batch {batch}')
        if act_checks[0] < 64:
            assert check_act(view), 'vectorized activation quantizer differs from quant_nvfp4_4over6'
            act_checks[0] += 1
        return (quant_per_document(view).reshape(x.shape), *inputs[1:])

    def per_sequence_kl(lp, t):
        return (t.exp() * (t - lp)).sum(-1).mean(-1)

    def eval_hooks(on):
        nonlocal eval_handles
        for h in eval_handles:
            h.remove()
        eval_handles = [m.register_forward_pre_hook(per_document_act) for m in modules.values()] if on else []

    def dev_eval(records=None, teachers=None):
        """Development KL/CE of the HARD map (monitor only); leaves the training weights in place."""
        records = dev if records is None else records
        teachers = dev_teacher if teachers is None else teachers
        if args.param == 'sigmoid':
            for n in modules:
                apply(n, True)
        eval_hooks(True)
        ce, kl = [], []
        for start in range(0, len(records), args.eval_batch):
            chunk = records[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            current_batch[0] = ids.shape[0]
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(teachers[start:start + args.eval_batch]).to(lp.device).float()
            ce.extend(F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1).tolist())
            kl.extend(per_sequence_kl(lp, t).tolist())
            del lp, t
        current_batch[0] = 1
        eval_hooks(False)
        if args.param == 'sigmoid':
            for n in modules:
                apply(n, False)
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    # Training hooks: straight-through per-token activations (the scoring convention
    # of run_multiround.py) and a backward hook that turns each linear layer's output
    # gradient into dLoss/dm for every tile of that layer.
    grads = {n: torch.zeros_like(t) for n, t in theta.items()}
    psi_grads = {n: torch.zeros_like(t) for n, t in psi.items()}

    def act(module, inputs):
        x = inputs[0]
        return (quantize_rows(x.detach()) + (x - x.detach()), *inputs[1:])

    def make_hook(n):
        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])

            def backward(dy):
                dy = dy.detach().reshape(-1, dy.shape[-1])
                g = dy.float().T @ x.float()
                b = e2m1(n)
                d = alt(n).float() - b.float()
                del b
                # dLoss/dm_u = sum over tile u of G * (A - E), with G = dy^T x (E = B unless joint).
                grads[n] += reduce(g * d, rows, cols)
                if joint:
                    # dLoss/ds_b = sum over scale block b of G * (1 - t) * (hi - lo).
                    e2 = ~expand(theta[n] > 0, rows, cols, g.shape[0])
                    lo, hi = scale_cands[n]
                    psi_grads[n] += reduce(g * e2 * (hi.float() - lo.float()), 1, 16)
            output.register_hook(backward)
        return forward

    params = list(theta.values()) + list(psi.values())
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps)
    else:
        opt = torch.optim.SGD(params, lr=args.lr)
    steps_per_epoch = math.ceil(math.ceil(len(fit) / args.batch) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)

    initial = dev_eval()
    report['initial_dev'] = initial
    if second_dev is not None:
        report['initial_dev2'] = dev_eval(second_dev, second_teacher)
        print(f'START dev2 (math/code) KL {report["initial_dev2"]["kl"]:.6f}', flush=True)
    report['setup_seconds'] = time.time() - started
    save(args.out, report)
    print(f'START dev CE {initial["ce"]:.6f} KL {initial["kl"]:.6f} tiles={report["tiles"]} '
          f'steps/epoch={steps_per_epoch} total={total_steps}', flush=True)
    if args.param == 'sigmoid':
        for n in modules:
            apply(n, False)
    step = 0
    previous = hard_map()
    previous_psi = {n: t > 0 for n, t in psi.items()}
    for epoch in range(args.epochs):
        t0 = time.time()
        order = list(range(len(fit)))
        rng.shuffle(order)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses, flips_epoch, scale_flips_epoch = [], 0, 0
        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
        for g0 in range(0, len(batches), args.accum):
            group = batches[g0:g0 + args.accum]
            n_seq = sum(len(b) for b in group)
            for g in (*grads.values(), *psi_grads.values()):
                g.zero_()
            with torch.enable_grad():
                for idx in group:
                    ids = torch.cat([fit[i] for i in idx]).to(device)
                    embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
                    lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                    t = torch.cat([teacher[i] for i in idx]).to(lp.device).float()
                    kl = per_sequence_kl(lp, t)
                    # Mean KL over the sequences of one optimizer step.
                    (kl.sum() / n_seq).backward()
                    losses.extend(kl.tolist())
                    del embeds, lp, t, kl
            if args.schedule == 'cosine':
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            # Staged joint training: grad None (not zero) keeps Adam from stepping, and from
            # advancing the bias-correction count of, the logits that are frozen this stage.
            scale_stage = epoch < args.scale_epochs
            train_tiles = not scale_stage
            train_scale = scale_stage or args.stage2 == 'joint'
            for n, p in theta.items():
                g = grads[n]
                if args.param == 'sigmoid':
                    s = torch.sigmoid(p / tau[0])
                    g = g * s * (1 - s) / tau[0]
                p.grad = g.clone() if train_tiles else None
            for n, p in psi.items():
                p.grad = psi_grads[n].clone() if train_scale else None
            opt.step()
            step += 1
            if args.param == 'sigmoid':
                tau[0] = args.tau_start * (args.tau_end / args.tau_start) ** (step / total_steps)
                for n in modules:
                    apply(n, False)
            for n in modules:
                now = theta[n] > 0
                changed = int((now != previous[n]).sum())
                if changed:
                    flips_epoch += changed
                    previous[n] = now
                if joint:
                    now_psi = psi[n] > 0
                    changed_psi = int((now_psi != previous_psi[n]).sum())
                    if changed_psi:
                        scale_flips_epoch += changed_psi
                        previous_psi[n] = now_psi
                        changed += changed_psi
                if changed and args.param == 'ste':
                    apply(n, True)
        for h in handles:
            h.remove()
        e0m3 = sum(int((t > 0).sum()) for t in theta.values())
        entry = dict(epoch=epoch, step=step, train_kl=sum(losses) / len(losses), e0m3_units=e0m3,
                     hard_flips=flips_epoch, tau=tau[0], lr=opt.param_groups[0]['lr'], epoch_seconds=time.time() - t0)
        if joint:
            entry['scale_flipped_blocks'] = sum(int((t > 0).sum()) for t in psi.values())
            # Scale blocks whose flipped scale is actually in effect (inside E2M1 tiles).
            entry['scale_flipped_active'] = sum(int(((psi[n] > 0) & ~expand(theta[n] > 0, rows, cols // 16, psi[n].shape[0])).sum())
                                                for n in psi)
            entry['scale_hard_flips'] = scale_flips_epoch
        if args.param == 'sigmoid':
            soft = torch.cat([torch.sigmoid(t / tau[0]).reshape(-1) for t in theta.values()])
            entry['undecided_fraction'] = float(((soft > 0.05) & (soft < 0.95)).float().mean())
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            d = dev_eval()
            entry['dev_ce'], entry['dev_kl'] = d['ce'], d['kl']
            if second_dev is not None:
                d2 = dev_eval(second_dev, second_teacher)
                entry['dev2_ce'], entry['dev2_kl'] = d2['ce'], d2['kl']
            entry['dev_kl_values'] = d['kl_values']
            if not args.no_epoch_maps:
                torch.save({n: m.cpu() for n, m in hard_map().items()}, args.out / f'map_epoch{epoch + 1:03d}.pt')
                if joint:
                    torch.save({n: (t > 0).cpu() for n, t in psi.items()}, args.out / f'scale_map_epoch{epoch + 1:03d}.pt')
        report['epochs'].append(entry)
        save(args.out, report)
        print('EPOCH ' + json.dumps({k: v for k, v in entry.items() if k != 'dev_kl_values'}), flush=True)
    final_map = hard_map()
    torch.save({n: m.cpu() for n, m in final_map.items()}, args.out / 'map.pt')
    if not args.no_logits:
        torch.save({n: t.cpu() for n, t in theta.items()}, args.out / 'theta.pt')
    if joint:
        # True = the block uses its `hi` scale candidate (see scale_pair; only matters inside E2M1 tiles).
        torch.save({n: (t > 0).cpu() for n, t in psi.items()}, args.out / 'scale_map.pt')
        if not args.no_logits:
            torch.save({n: t.cpu() for n, t in psi.items()}, args.out / 'psi.pt')
        report['final_scale_flipped_active'] = report['epochs'][-1]['scale_flipped_active'] if report['epochs'] else 0
        report['scale_map_sha256'] = digest_file(args.out / 'scale_map.pt')
    report['final_dev'] = dev_eval()
    if second_dev is not None:
        report['final_dev2'] = dev_eval(second_dev, second_teacher)
    report['final_e0m3_units'] = sum(int(m.sum()) for m in final_map.values())
    report['map_sha256'] = digest_file(args.out / 'map.pt')
    report['optimization_seconds'] = time.time() - started - report['setup_seconds']
    save(args.out, report)
    print(f'FINAL dev CE {report["final_dev"]["ce"]:.6f} KL {report["final_dev"]["kl"]:.6f} '
          f'e0m3={report["final_e0m3_units"]}', flush=True)
    # Released PPL windows on the final HARD map.
    for n in modules:
        apply(n, True)
    batches, report['data'] = data(tok, prior, 2048)
    published = json.loads(Path(PUBLISHED[args.model]).read_text())
    validate_evaluation_data(report['data'], published['data'])
    eval_hooks(True)
    report['evaluation'] = {}
    for domain, sequences in batches.items():
        values = []
        for ids in sequences:
            ids = ids.to(device)
            logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
            values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                ids[:, 1:].reshape(-1).to(logits.device))))
            del logits
        losses = torch.tensor(values, dtype=torch.float32) * 2048
        key = 'c4' if domain == 'c4_paper' else domain
        report['evaluation'][key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
        print(f'PPL train_{args.alt}_{args.param}_{args.unit} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
        save(args.out, report)
    eval_hooks(False)
    report['resources'] = dict(
        total_seconds=time.time() - started,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        gpu_peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())],
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()
