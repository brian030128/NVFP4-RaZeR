"""Multi-round relinearized MixFP4 election with backtracking on measured loss (Llama-3.1-8B).

Each round re-scores every unit's legal flip (E2M1 FourOverSix <-> E0M3 alpha=1,
undo included) with per-sequence CE and KL(BF16 teacher) weight gradients at the
CURRENT quantized model on the 128 calibration sequences, exactly the scoring
convention of run_math_code_calibration.py (causal per-token activation factors,
straight-through). Candidates are units whose CE and KL upper bounds
mean + FILTER_K * SE are both negative, ranked by that bound. The step is chosen
by backtracking: the top n, n/2, n/4, ... candidates are applied and the first
set that lowers mean CE on the 192 held-out development documents (tensor-wide
W4A4, as in evaluation) is accepted. The loop stops when no step lowers it.
WikiText-2 / C4 are evaluated once, on the final map only.
"""
import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from quantize.causal_four_over_six import quantize_rows
from quantize.packed_candidates import decode_alt, decode_base, nbytes, pack
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_task_reorder_eval import validate_evaluation_data

CALIBRATIONS = {'llama8b': Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration'),
                'qwen27b': Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration')}
# Held-out math/code documents from recorded earlier confirmation gates, tokenized per model.
DEVELOPMENT = {
    'llama8b': [Path('/work/u4320956/task_reorder/transfer_20260920/llama8b') / s
                for s in ('confirmation', 'gate_up_confirm', 'tile_refine_confirm')],
    'qwen27b': [Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/fine_rows_v2') / s
                for s in ('fisher_subset_validate_v2_confirm', 'preserved_row_confirm', 'ce_target_combinations_confirm')]}
PUBLISHED = {'llama8b': 'results/kse_paper/job_336566/llama8b/report.json',
             'qwen27b': 'results/kse_paper/job_336969/qwen27b/report.json'}


def load_development(model):
    records, provenance = [], []
    for directory in DEVELOPMENT[model]:
        report = json.loads((directory / 'report.json').read_text())
        assert report.get('status') == 'complete', directory
        assert digest_file(directory / 'fresh.pt') == report['fresh_sha256'], directory
        rows = torch.load(directory / 'fresh.pt', map_location='cpu', weights_only=True)
        records.extend(rows)
        provenance.append(dict(name=str(directory), documents=len(rows), sha256=report['fresh_sha256']))
    return records, provenance


UNITS = {'256x64': (256, 64), '8x64': (8, 64), '1x16': (1, 16)}


def expand(mask, rows, cols, height=None):
    # The last row tile may be partial; its padded rows are trimmed (raw256 convention).
    return mask.repeat_interleave(rows, 0).repeat_interleave(cols, 1)[:height]


def reduce(x, rows, cols):
    o, k = x.shape
    x = F.pad(x, (0, 0, 0, (-o) % rows))
    return x.reshape(-1, rows, k // cols, cols).sum((1, 3))


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=tuple(CALIBRATIONS), default='llama8b')
    ap.add_argument('--unit', choices=tuple(UNITS), required=True)
    ap.add_argument('--objective', choices=('both', 'ce', 'kl'), required=True,
                    help='Scores that rank candidates AND the development loss(es) a step must lower')
    ap.add_argument('--filter-k', type=float, default=2.0)
    ap.add_argument('--max-rounds', type=int, default=40)
    ap.add_argument('--max-tries', type=int, default=22)
    ap.add_argument('--budget-hours', type=float, default=3.0)
    ap.add_argument('--significant-steps', action='store_true',
                    help='Accept a step only if the paired development change is significant: mean + 2 SE < 0 '
                         'over documents for every accepted objective (same 2 SE convention as the candidate filter)')
    ap.add_argument('--warm-start', action='store_true',
                    help='Start each round\'s backtracking at twice the previous accepted step instead of all candidates')
    ap.add_argument('--eval-batch', type=int, default=1,
                    help='Development documents per forward pass (activation scales stay per document)')
    ap.add_argument('--score-batch', type=int, default=1,
                    help='Calibration sequences per scoring forward/backward (gradients stay per sequence)')
    ap.add_argument('--gpus', type=int, default=None, help='Qwen: 1 loads on one device, else balanced')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = UNITS[args.unit]
    qwen = args.model == 'qwen27b'
    prior = json.loads((CALIBRATIONS[args.model] / 'report.json').read_text())
    assert transformers.__version__ == prior['transformers_version']
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], model=args.model, unit=args.unit, objective=args.objective,
                  eval_batch=args.eval_batch, score_batch=args.score_batch,
                  significant_steps=args.significant_steps, warm_start=args.warm_start,
                  filter_k=args.filter_k, max_tries=args.max_tries,
                  acceptance={'both': 'mean dev CE and mean dev KL must both decrease', 'ce': 'mean dev CE must decrease',
                              'kl': 'mean dev KL must decrease'}[args.objective],
                  source_sha256={p: digest_file(p) for p in ('run_multiround.py', 'quantize/quantizer.py',
                                                             'quantize/causal_four_over_six.py')},
                  rounds=[])
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
    logits_width = logits.shape[-1]
    del logits
    teacher = []
    for ids in fit:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    # Both candidates are stored in deployment format (4-bit codes + FP8 scales) and
    # decoded per module on use; pack() verifies bitwise equality with the reference
    # quantizers, and any module that fails keeps dequantized BF16 copies instead.
    packed, dense, sel = {}, {}, {}
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
        sel[n] = torch.zeros(-(-o // rows), k // cols, dtype=torch.bool, device=m.weight.device)
        m.weight.copy_(b)
        del a, b
    torch.cuda.empty_cache()
    report['candidate_storage'] = dict(packed_modules=len(packed), dense_fallback_modules=sorted(dense),
                                       packed_gib=sum(nbytes(p) for p in packed.values()) / 2 ** 30)
    print('CANDIDATES ' + json.dumps(report['candidate_storage']), flush=True)

    def base(n):
        return dense[n][0] if n in dense else decode_base(packed[n])

    def alt(n):
        return dense[n][1] if n in dense else decode_alt(packed[n])

    def apply(n):
        b = base(n)
        modules[n].weight.copy_(torch.where(expand(sel[n], rows, cols, b.shape[0]), alt(n), b))

    eval_handles = []

    def per_document_act(module, inputs):
        # Tensor-wide activation scales are computed per document, exactly as with
        # one document per forward pass, however many documents are batched.
        x = inputs[0]
        if x.dim() >= 3 and x.shape[0] > 1:
            q = torch.stack([quant_nvfp4_4over6(x[i], 4, 16) for i in range(x.shape[0])])
        else:
            q = quant_nvfp4_4over6(x, 4, 16)
        return (q, *inputs[1:])

    def per_sequence_losses(lp, ids, t):
        # lp, t: (B, T-1, V) log-probabilities; CE and KL averaged over each sequence's tokens.
        ce = F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1)
        kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
        return ce, kl

    def eval_hooks(on):
        nonlocal eval_handles
        for h in eval_handles:
            h.remove()
        eval_handles = [m.register_forward_pre_hook(per_document_act) for m in modules.values()] if on else []

    def dev_eval():
        eval_hooks(True)
        ce, kl = [], []
        for start in range(0, len(dev), args.eval_batch):
            chunk = dev[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
            c, k = per_sequence_losses(lp, ids, t)
            ce.extend(c.tolist()); kl.extend(k.tolist())
            del lp, t
        eval_hooks(False)
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    def improves(new, old):
        keys = ('ce', 'kl') if args.objective == 'both' else (args.objective,)
        if not args.significant_steps:
            return all(new[key] < old[key] for key in keys)
        for key, values in (('ce', 'ce_nll'), ('kl', 'kl_values')):
            if key not in keys:
                continue
            diff = torch.tensor(new[values], dtype=torch.float64) - torch.tensor(old[values], dtype=torch.float64)
            if float(diff.mean() + 2 * diff.std(unbiased=True) / math.sqrt(len(diff))) >= 0:
                return False
        return True

    def score():
        """Per-unit mean and SE of CE and KL directional scores for flipping each unit now."""
        sums = {n: [torch.zeros(sel[n].shape, dtype=torch.float64, device=sel[n].device) for _ in range(4)] for n in modules}
        phase = [0]

        def act(module, inputs):
            x = inputs[0]
            return (quantize_rows(x.detach()) + (x - x.detach()), *inputs[1:])

        def make_hook(n):
            def forward(module, inputs, output):
                x = inputs[0].detach()
                x = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() >= 3 else x.reshape(1, -1, x.shape[-1])

                def backward(dy):
                    # Each sequence's loss reaches only its own slice of dy, so
                    # dy[i]^T x[i] is exactly sequence i's weight gradient.
                    dy = dy.detach().reshape(x.shape[0], -1, dy.shape[-1])
                    b = base(n)
                    d = (alt(n).float() - b.float()) * torch.where(expand(sel[n], rows, cols, b.shape[0]), -1., 1.)
                    del b
                    s = sums[n]
                    for i in range(x.shape[0]):
                        value = reduce((dy[i].float().T @ x[i].float()) * d, rows, cols).double()
                        s[2 * phase[0]] += value
                        s[2 * phase[0] + 1] += value.square()
                output.register_hook(backward)
            return forward

        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
        with torch.enable_grad():
            for start in range(0, len(fit), args.score_batch):
                ids = torch.cat(fit[start:start + args.score_batch]).to(device)
                embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
                lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                t = torch.cat(teacher[start:start + args.score_batch]).to(lp.device).float()
                ce, kl = per_sequence_losses(lp, ids, t)
                phase[0] = 0; ce.sum().backward(retain_graph=True)
                phase[0] = 1; kl.sum().backward()
                del embeds, lp, t, ce, kl
        for h in handles:
            h.remove()
        n_seq = len(fit)
        stats = {}
        for n, (cs, cq, ks, kq) in sums.items():
            out = []
            for total, square in ((cs, cq), (ks, kq)):
                mean = total / n_seq
                se = ((square - n_seq * mean.square()).clamp_min(0) / (n_seq - 1)).sqrt() / math.sqrt(n_seq)
                out.append((mean, se))
            stats[n] = out
        return stats

    current = dev_eval()
    report['initial_dev'] = current
    save(args.out, report)
    timing = dict(setup_seconds=time.time() - started)
    print(f'START {args.objective} dev CE {current["ce"]:.6f} KL {current["kl"]:.6f}', flush=True)
    names = list(modules)
    previous_accepted = None
    for rnd in range(args.max_rounds):
        if time.time() - started > args.budget_hours * 3600:
            report['stopped'] = 'time budget'; break
        t0 = time.time()
        stats = score()
        bounds, flat_mean_ce, flat_mean_kl, owners = [], [], [], []
        for i, n in enumerate(names):
            (cm, cse), (km, kse) = stats[n]
            ub = {'ce': cm + args.filter_k * cse, 'kl': km + args.filter_k * kse}
            u = (torch.maximum(ub['ce'], ub['kl']) if args.objective == 'both' else ub[args.objective]).reshape(-1)
            bounds.append(u.to(device)); flat_mean_ce.append(cm.reshape(-1).to(device)); flat_mean_kl.append(km.reshape(-1).to(device))
            owners.append(torch.full_like(u, i, dtype=torch.int32, device=device))
        bounds = torch.cat(bounds); owners = torch.cat(owners)
        flat_mean_ce = torch.cat(flat_mean_ce); flat_mean_kl = torch.cat(flat_mean_kl)
        offsets = np.cumsum([0] + [sel[n].numel() for n in names])
        candidates = (bounds < 0).nonzero().squeeze(-1)
        candidates = candidates[bounds[candidates].argsort()]
        score_seconds = time.time() - t0
        entry = dict(round=rnd, candidates=int(candidates.numel()), score_seconds=score_seconds, tries=[])
        if candidates.numel() == 0:
            entry['accepted'] = 0; report['rounds'].append(entry); report['stopped'] = 'no candidates'; break
        size, accepted = int(candidates.numel()), None
        if args.warm_start and previous_accepted:
            size = min(size, 2 * previous_accepted)
        for _ in range(args.max_tries):
            chosen = candidates[:size]
            touched = {}
            who = owners[chosen]
            for i in who.unique().tolist():
                touched[names[i]] = chosen[who == i] - int(offsets[i])
            for n, idx in touched.items():
                flat = sel[n].view(-1); flat[idx.to(flat.device)] ^= True; apply(n)
            new = dev_eval()
            pce, pkl = float(flat_mean_ce[chosen].sum()), float(flat_mean_kl[chosen].sum())
            entry['tries'].append(dict(size=size, predicted_ce=pce, predicted_kl=pkl,
                                       dev_delta_ce=new['ce'] - current['ce'], dev_delta_kl=new['kl'] - current['kl']))
            print(f'ROUND {rnd} try size={size} pred CE {pce:+.6f} KL {pkl:+.6f} | dev dCE {new["ce"] - current["ce"]:+.6f} '
                  f'dKL {new["kl"] - current["kl"]:+.6f}', flush=True)
            if improves(new, current):
                accepted = size; current = new; break
            for n, idx in touched.items():
                flat = sel[n].view(-1); flat[idx.to(flat.device)] ^= True; apply(n)
            if size == 1:
                break
            size = max(1, size // 2)
        entry['accepted'] = accepted or 0
        previous_accepted = accepted
        entry['e0m3_units'] = sum(int(s.sum()) for s in sel.values())
        entry['dev_ce'], entry['dev_kl'] = current['ce'], current['kl']
        entry['round_seconds'] = time.time() - t0
        report['rounds'].append(entry)
        torch.save({n: s.cpu() for n, s in sel.items()}, args.out / 'map.pt')
        save(args.out, report)
        print(f'ROUND {rnd} accepted={accepted} e0m3={entry["e0m3_units"]} dev CE {current["ce"]:.6f} KL {current["kl"]:.6f} '
              f'{entry["round_seconds"]:.0f}s', flush=True)
        if not accepted:
            report['stopped'] = 'no step lowers the development objective'; break
    report['final_dev'] = current
    timing['optimization_seconds'] = time.time() - started - timing['setup_seconds']
    report['final_e0m3_units'] = sum(int(s.sum()) for s in sel.values())
    report['map_sha256'] = digest_file(args.out / 'map.pt') if (args.out / 'map.pt').exists() else None
    save(args.out, report)
    # One evaluation of the final map on the released PPL windows.
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
            values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1).to(logits.device))))
            del logits
        losses = torch.tensor(values, dtype=torch.float32) * 2048
        key = 'c4' if domain == 'c4_paper' else domain
        report['evaluation'][key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
        print(f'PPL multiround_{args.unit}_{args.objective} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
        save(args.out, report)
    eval_hooks(False)
    timing['evaluation_seconds'] = time.time() - started - timing['setup_seconds'] - timing['optimization_seconds']
    timing['total_seconds'] = time.time() - started
    report['resources'] = dict(
        timing, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        gpu_peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())],
        gpu_peak_reserved_gib=[torch.cuda.max_memory_reserved(i) / 2 ** 30 for i in range(torch.cuda.device_count())],
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        scoring_passes=len(report['rounds']),
        development_evaluations=sum(len(r['tries']) for r in report['rounds']) + 1)
    print('RESOURCES ' + json.dumps(report['resources']), flush=True)
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()
