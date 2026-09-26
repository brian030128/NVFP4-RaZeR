"""MixFP4 tile selection by gradient training of per-tile logits on the KL loss (no backtracking).

Every type tile u of every text linear layer gets a real latent logit theta_u.
The weights are W(theta) = B + expand(m(theta)) * (A - B), with B the FourOverSix
E2M1 candidate and A the E0M3 alpha=1 candidate (identical to run_multiround.py).

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
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_multiround import CALIBRATIONS, PUBLISHED, UNITS, expand, load_development, reduce
from run_task_reorder_eval import validate_evaluation_data


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=tuple(CALIBRATIONS), default='llama8b')
    ap.add_argument('--unit', choices=tuple(UNITS), required=True)
    ap.add_argument('--param', choices=('ste', 'sigmoid'), default='ste')
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
    fit, _ = math_code_data(tok, prior['fit'])
    fit = [b for source in ('math', 'code') for b in fit[source]]
    dev, report['development'] = load_development(args.model)
    device = model.get_input_embeddings().weight.device
    dev_teacher = []
    for r in dev:
        logits = model(input_ids=r['ids'].to(device), use_cache=False).logits
        dev_teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    teacher = []
    for ids in fit:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    del logits
    # Candidates exactly as run_multiround.py: packed 4-bit codes + FP8 scales,
    # verified bitwise by pack(), dense BF16 fallback otherwise.
    packed, dense, theta = {}, {}, {}
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        b = quant_nvfp4_4over6(m.weight, 4, 16)
        a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        p = pack(m.weight, b, a)
        if p is None:
            dense[n] = (b, a)
        else:
            packed[n] = p
        o, k = m.weight.shape
        assert k % cols == 0
        theta[n] = torch.full((-(-o // rows), k // cols), args.init_logit, dtype=torch.float32, device=m.weight.device)
        m.weight.copy_(b)
        del a, b
    torch.cuda.empty_cache()
    report['candidate_storage'] = dict(packed_modules=len(packed), dense_fallback_modules=sorted(dense),
                                       packed_gib=sum(nbytes(p) for p in packed.values()) / 2 ** 30)
    report['tiles'] = sum(t.numel() for t in theta.values())
    print('CANDIDATES ' + json.dumps(report['candidate_storage']) + f' tiles={report["tiles"]}', flush=True)

    def base(n):
        return dense[n][0] if n in dense else decode_base(packed[n])

    def alt(n):
        return dense[n][1] if n in dense else decode_alt(packed[n])

    tau = [args.tau_start]

    def apply(n, hard):
        b = base(n)
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

    def dev_eval():
        """Development KL/CE of the HARD map (monitor only); leaves the training weights in place."""
        if args.param == 'sigmoid':
            for n in modules:
                apply(n, True)
        eval_hooks(True)
        ce, kl = [], []
        for start in range(0, len(dev), args.eval_batch):
            chunk = dev[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            current_batch[0] = ids.shape[0]
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
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

    def act(module, inputs):
        x = inputs[0]
        return (quantize_rows(x.detach()) + (x - x.detach()), *inputs[1:])

    def make_hook(n):
        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])

            def backward(dy):
                dy = dy.detach().reshape(-1, dy.shape[-1])
                b = base(n)
                d = alt(n).float() - b.float()
                del b
                # dLoss/dm_u = sum over tile u of G * (A - B), with G = dy^T x.
                grads[n] += reduce((dy.float().T @ x.float()) * d, rows, cols)
            output.register_hook(backward)
        return forward

    params = list(theta.values())
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps)
    else:
        opt = torch.optim.SGD(params, lr=args.lr)
    steps_per_epoch = math.ceil(math.ceil(len(fit) / args.batch) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)

    initial = dev_eval()
    report['initial_dev'] = initial
    report['setup_seconds'] = time.time() - started
    save(args.out, report)
    print(f'START dev CE {initial["ce"]:.6f} KL {initial["kl"]:.6f} tiles={report["tiles"]} '
          f'steps/epoch={steps_per_epoch} total={total_steps}', flush=True)
    if args.param == 'sigmoid':
        for n in modules:
            apply(n, False)
    step = 0
    previous = hard_map()
    for epoch in range(args.epochs):
        t0 = time.time()
        order = list(range(len(fit)))
        rng.shuffle(order)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses, flips_epoch = [], 0
        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
        for g0 in range(0, len(batches), args.accum):
            group = batches[g0:g0 + args.accum]
            n_seq = sum(len(b) for b in group)
            for g in grads.values():
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
            for n, p in theta.items():
                g = grads[n]
                if args.param == 'sigmoid':
                    s = torch.sigmoid(p / tau[0])
                    g = g * s * (1 - s) / tau[0]
                p.grad = g.clone()
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
                    if args.param == 'ste':
                        apply(n, True)
        for h in handles:
            h.remove()
        e0m3 = sum(int((t > 0).sum()) for t in theta.values())
        entry = dict(epoch=epoch, step=step, train_kl=sum(losses) / len(losses), e0m3_units=e0m3,
                     hard_flips=flips_epoch, tau=tau[0], lr=opt.param_groups[0]['lr'], epoch_seconds=time.time() - t0)
        if args.param == 'sigmoid':
            soft = torch.cat([torch.sigmoid(t / tau[0]).reshape(-1) for t in theta.values()])
            entry['undecided_fraction'] = float(((soft > 0.05) & (soft < 0.95)).float().mean())
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            d = dev_eval()
            entry['dev_ce'], entry['dev_kl'] = d['ce'], d['kl']
            entry['dev_kl_values'] = d['kl_values']
            torch.save({n: m.cpu() for n, m in hard_map().items()}, args.out / f'map_epoch{epoch + 1:03d}.pt')
        report['epochs'].append(entry)
        save(args.out, report)
        print('EPOCH ' + json.dumps({k: v for k, v in entry.items() if k != 'dev_kl_values'}), flush=True)
    final_map = hard_map()
    torch.save({n: m.cpu() for n, m in final_map.items()}, args.out / 'map.pt')
    torch.save({n: t.cpu() for n, t in theta.items()}, args.out / 'theta.pt')
    report['final_dev'] = dev_eval()
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
        print(f'PPL train_{args.param}_{args.unit} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
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
