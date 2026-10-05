"""run_train_map.py --teacher-topk against flipquant's definition (results/topk_cal/PROTOCOL.md, check (b)).

flipquant's functions are read from its source and executed as written: tail_logprob, kl_topk_per_sequence and the
TopKTeacher class, from calibration/train_map.py in $FLIPQUANT_ROOT (default ~/flipquant) at $FLIPQUANT_REF (default
d2dd92e). Checks:
  1. random logits, for the four final models' vocabularies, 512-token documents, batches 8 (the training micro-batch)
     and 2 (Qwen's micro-batch), K = 256:
     - the teacher rows (values, ids, tail) and the tail mass: topk_teacher.topk_rows against TopKTeacher.append_logprobs;
     - the per-sequence KL and the bf16 gradient into the logits of one training step,
       chunked_loss.train_kl_gradient_topk against flipquant's whole-batch
       (kl_topk_per_sequence(...).sum() / n_seq).backward();
     - run_train_map's whole-batch branch (topk_teacher.kl_topk_per_sequence) against flipquant's.
  2. real logits (--real): Llama-3.1-8B, the BF16 teacher against a FourOverSix-weight student (W4A16) on the first 8
     calibration windows. The comparisons of 1 are repeated on these logits.
  3. finite differences, float64: the gradient of flipquant's expression from autograd against central differences, and
     against the closed form d/dz_j = q_j S - p_j (j in the top K) or q_j S - p_tail q_j / q_tail (j outside), with
     S = sum_K p + p_tail.
Deterministic algorithms are on, as in the training runs. Bitwise means equal bit patterns (FP32 as int32, bf16 as int16). If a comparison is not bitwise, the maximum relative
difference is reported against a tolerance of 1e-6.

python repro_local/realquant/test_topk_teacher.py OUT_JSON [--real --data-root DIR --transformers-deviation]
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)     # as in the training runs (--deterministic)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import chunked_loss  # noqa: E402
import topk_teacher  # noqa: E402

VOCABS = {'mistral7b': 32768, 'phi4': 100352, 'llama8b': 128256, 'qwen27b': 248320}
T, K, TOL = 512, 256, 1e-6
N_SEQ = 8          # sequences per optimizer step (Qwen: micro-batch 2 x accumulation 4)
WANTED = ('tail_logprob', 'kl_topk_per_sequence', 'TopKTeacher')


def flipquant_functions():
    """flipquant's tail_logprob, kl_topk_per_sequence and TopKTeacher, from its source, executed as written."""
    root = Path(os.environ.get('FLIPQUANT_ROOT', Path.home() / 'flipquant'))
    ref = os.environ.get('FLIPQUANT_REF', 'd2dd92e')
    source = subprocess.run(['git', '-C', str(root), 'show', f'{ref}:calibration/train_map.py'], capture_output=True,
                            check=True).stdout
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in WANTED]
    assert sorted(n.name for n in nodes) == sorted(WANTED), [n.name for n in nodes]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'flipquant/calibration/train_map.py', 'exec'), namespace)
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', ref], capture_output=True, text=True,
                            check=True).stdout.strip()
    return namespace, dict(root=str(root), ref=ref, commit=commit, file_sha256=hashlib.sha256(source).hexdigest(),
                           functions=[ast.get_source_segment(source.decode(), n) for n in nodes])


def bits(a):
    return a.view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype])


def compare(a, b):
    """Bitwise, else the maximum relative difference (relative to the larger magnitude of b)."""
    if torch.equal(bits(a), bits(b)):
        return dict(bitwise=True, max_rel=0.0)
    d = (a.double() - b.double()).abs().max().item()
    return dict(bitwise=False, max_rel=d / max(b.double().abs().max().item(), 1e-30))


def compare_case(fq, student, teacher_logits):
    """The checks of 1 for one batch: student bf16 [B, T, V]; teacher_logits bf16 [B, T, V]."""
    b = student.shape[0]
    store = fq['TopKTeacher'](K, 'cuda')
    rows, mass = [], []
    for i in range(b):
        lp_t = teacher_logits[i:i + 1, :-1].float().log_softmax(-1)
        store.append_logprobs(lp_t)
        r, m = topk_teacher.topk_rows(lp_t, K)
        rows.append(r)
        mass.append(m)
        del lp_t
    case = dict(rows_bitwise=all(x.dtype == y.dtype and torch.equal(x, y) for r, s in zip(rows, store.rows)
                                 for x, y in zip(r, s)),
                tail_mass_equal=mass == store.tail_mass, mean_tail_mass=sum(mass) / len(mass))
    # flipquant: the whole batch, kl_topk_per_sequence(...).sum() / n_seq, backward
    with torch.enable_grad():
        leaf = student.clone().requires_grad_()
        lp = leaf[:, :-1].float().log_softmax(-1)
        kl0 = fq['kl_topk_per_sequence'](lp, *store.batch(list(range(b)), 'cuda'))
        (kl0.sum() / N_SEQ).backward()
        grad0, kl0 = leaf.grad.clone(), kl0.detach()
        del leaf, lp
        grad1, kl1 = chunked_loss.train_kl_gradient_topk(student, rows, 'cuda', N_SEQ)
    case['chunked_kl'] = compare(kl1, kl0)
    case['chunked_grad'] = compare(grad1, grad0)
    with torch.no_grad():
        lp = student[:, :-1].float().log_softmax(-1)
        kl2 = topk_teacher.kl_topk_per_sequence(lp, *topk_teacher.batch_rows(rows, 'cuda'))
        del lp
    case['whole_batch_kl'] = compare(kl2, kl0)
    case['kl_mean'] = float(kl0.mean())
    case['passed'] = (case['rows_bitwise'] and case['tail_mass_equal']
                      and all(case[c]['max_rel'] <= TOL for c in ('chunked_kl', 'chunked_grad', 'whole_batch_kl')))
    del grad0, grad1, kl0, kl1, kl2
    torch.cuda.empty_cache()
    return case


def finite_differences(fq):
    """3: float64, B = 2, T = 9, V = 64, k in (8, 64)."""
    out = {}
    g = torch.Generator().manual_seed(0)
    for k in (8, 64):
        z_t = torch.randn(2, 9, 64, generator=g, dtype=torch.float64) * 3
        z = torch.randn(2, 9, 64, generator=g, dtype=torch.float64) * 3
        lp_t = z_t[:, :-1].log_softmax(-1)
        values, idx = lp_t.topk(k, dim=-1)
        values = values.bfloat16().double()          # as stored by TopKTeacher
        tail = fq['tail_logprob'](lp_t, idx)

        def loss(x):
            return fq['kl_topk_per_sequence'](x[:, :-1].log_softmax(-1), values, idx, tail).sum() / 2
        leaf = z.clone().requires_grad_()
        loss(leaf).backward()
        auto = leaf.grad
        fd = torch.zeros_like(z)
        h = 1e-6
        for i in range(z.numel()):
            e = torch.zeros(z.numel(), dtype=torch.float64)
            e[i] = h
            e = e.view_as(z)
            fd.view(-1)[i] = (loss(z + e) - loss(z - e)) / (2 * h)
        # the closed form, per token, then the token mean and the /2
        q = z[:, :-1].softmax(-1)
        p = values.exp()
        p_tail = torch.where(torch.isfinite(tail), tail.exp(), torch.zeros_like(tail))
        s = p.sum(-1) + p_tail
        inside = torch.zeros_like(q, dtype=torch.bool).scatter(-1, idx, True)
        q_tail = (q * ~inside).sum(-1)
        closed = q * s.unsqueeze(-1) - torch.where(inside, torch.zeros_like(q).scatter(-1, idx, p),
                                                   p_tail.unsqueeze(-1) * q / q_tail.clamp_min(1e-300).unsqueeze(-1))
        closed = torch.cat([closed, torch.zeros_like(closed[:, :1])], 1) / (z.shape[1] - 1) / 2
        scale = auto.abs().max().item()
        out[f'k{k}'] = dict(fd_max_rel=(auto - fd).abs().max().item() / scale,
                            closed_max_rel=(auto - closed).abs().max().item() / scale)
    out['passed'] = all(v['fd_max_rel'] < 1e-6 and v['closed_max_rel'] < 1e-12 for v in out.values() if isinstance(v, dict))
    return out


def real_logits(args):
    """2: Llama-3.1-8B BF16 teacher against a FourOverSix-weight student on the first 8 calibration windows."""
    import transformers
    from transformers import AutoTokenizer
    from quantize.quantizer import quant_nvfp4_4over6
    from run_math_code_calibration import load_model, math_code_data
    from run_multiround import data_paths
    calibration, _ = data_paths('llama8b', args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    assert transformers.__version__ == prior['transformers_version'] or args.transformers_deviation
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fit, _ = math_code_data(tok, prior['fit'])
    fit = [b for source in ('math', 'code') for b in fit[source]][:8]
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    ids = torch.cat(fit).cuda()
    with torch.no_grad():
        teacher_logits = model(input_ids=ids, use_cache=False).logits
        for m in modules.values():
            m.weight.copy_(quant_nvfp4_4over6(m.weight.data, 4, 16).to(m.weight.dtype))
        student = model(input_ids=ids, use_cache=False).logits
    del model
    torch.cuda.empty_cache()
    return compare_case(args.fq, student, teacher_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--data-root', type=Path, default=None)
    ap.add_argument('--transformers-deviation', action='store_true')
    args = ap.parse_args()
    args.fq, source = flipquant_functions()
    result = dict(flipquant=source, k=K, tokens=T, tolerance=TOL, random={}, failures=[])
    g = torch.Generator(device='cuda').manual_seed(0)
    for model, v in VOCABS.items():
        for batch in (8, 2):
            student = (torch.randn(batch, T, v, generator=g, device='cuda') * 3).bfloat16()
            teacher_logits = (torch.randn(batch, T, v, generator=g, device='cuda') * 3).bfloat16()
            case = compare_case(args.fq, student, teacher_logits)
            result['random'][f'{model}_b{batch}'] = case
            print(model, batch, {k: case[k] for k in ('rows_bitwise', 'tail_mass_equal', 'chunked_kl', 'chunked_grad',
                                                       'whole_batch_kl', 'passed')}, flush=True)
            if not case['passed']:
                result['failures'].append(f'random {model} b{batch}')
            del student, teacher_logits
            torch.cuda.empty_cache()
    result['finite_differences'] = finite_differences(args.fq)
    print('finite differences', result['finite_differences'], flush=True)
    if not result['finite_differences']['passed']:
        result['failures'].append('finite differences')
    if args.real:
        result['real'] = real_logits(args)
        print('real', {k: v for k, v in result['real'].items()}, flush=True)
        if not result['real']['passed']:
            result['failures'].append('real logits')
    result['passed'] = not result['failures']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print('PASSED' if result['passed'] else f"FAILED: {result['failures']}", flush=True)
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
