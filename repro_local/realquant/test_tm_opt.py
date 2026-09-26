"""TM-OPT unit tests (results/tm_opt/PROTOCOL.md): run_train_map.py's optimizations against its legacy expressions.

1. chunked_loss.train_kl_gradient vs the legacy training step, bitwise: for batches 1-8 of 512-token documents at
   the vocabularies of Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, the gradient into the logits of
   (kl.sum() / batch).backward() and the per-sequence KL.
2. tile_score.tile_sums (B1) vs the legacy hook reduce((dy^T x) * (A - B)): one real matrix of every distinct layer-0
   shape of the three models and a random matrix of each shape, at 8x64, 16x64 and 256x64, for 4 batches of
   8 x 512 tokens (synthetic heavy-tailed output gradients, per-token-FourOverSix-quantized inputs). Reported: max
   absolute, normwise relative (max|d| / max|ref|) and elementwise relative error (tiles with |ref| >= 1e-3 max|ref|)
   of the per-batch tile sums, and the time of one batch. Registered tolerance: normwise relative <= 1e-6.

python repro_local/realquant/test_tm_opt.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunked_loss  # noqa: E402
from candidate_store import CandidateStore  # noqa: E402
from quantize.fused_fourover6 import fourover6_rows  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_multiround import reduce  # noqa: E402
from test_tile_score import RECORDS, errors, layer0_matrices, load_weight, timed  # noqa: E402
from tile_score import tile_sums  # noqa: E402

MODELS = ('llama8b', 'mistral7b', 'phi4')
VOCABS = {'mistral7b': 32768, 'phi4': 100352, 'llama8b': 128256}
TOLERANCE = 1e-6


def chunked_tests(g):
    out = []
    for model, vocab in VOCABS.items():
        for batch in range(1, 9):
            logits = (torch.randn(batch, 512, vocab, generator=g, device='cuda') * 3).bfloat16()
            teacher = [torch.randn(1, 511, vocab, generator=g, device='cuda').log_softmax(-1).bfloat16().cpu()
                       for _ in range(batch)]
            leaf = logits.clone().requires_grad_()
            with torch.enable_grad():
                lp = leaf[:, :-1].float().log_softmax(-1)
                t = torch.cat(teacher).to(lp.device).float()
                kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
                (kl.sum() / batch).backward()
            ref_grad, ref_kl = leaf.grad, kl.detach()
            del lp, t
            leaf2 = logits.clone().requires_grad_()
            with torch.enable_grad():
                grad, kl2 = chunked_loss.train_kl_gradient(leaf2, teacher, 'cuda', batch)
            row = dict(model=model, vocab=vocab, batch=batch,
                       gradient_equal=bool(torch.equal(grad.view(torch.int16), ref_grad.view(torch.int16))),
                       kl_equal=bool(torch.equal(kl2, ref_kl)))
            out.append(row)
            print('chunked', json.dumps(row), flush=True)
            del logits, teacher, leaf, leaf2, grad, kl, kl2, ref_grad, ref_kl
            torch.cuda.empty_cache()
    return out


@torch.no_grad()
def b1_tests(g):
    out = {}
    for model in MODELS:
        source, mats = layer0_matrices(RECORDS[model])
        for name, (n, k) in mats:
            for kind in ('real', 'random'):
                w = load_weight(source, name).cuda() if kind == 'real' else (torch.randn(n, k, generator=g, device='cuda') * 0.02).bfloat16()
                b, a = quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                p = pack(w, b, a)
                assert p is not None, name
                for rows in (8, 16, 256):
                    store = CandidateStore(rows, 64)
                    store.add(name, w, decode_base(p), decode_alt(p))
                    base = store.decode(name, which='base')
                    d = store.decode(name, which='alt').float() - base.float()
                    worst = None
                    for _ in range(4):
                        dy = (torch.randn(8 * 512, n, generator=g, device='cuda') * 1e-4
                              * torch.logspace(-1, 1, n, device='cuda')).bfloat16()
                        x = fourover6_rows((torch.randn(8 * 512, k, generator=g, device='cuda')
                                            * torch.logspace(-2, 1, k, device='cuda')[torch.randperm(k, generator=g, device='cuda')]).bfloat16())
                        shape = (-(-n // rows), k // 64)
                        ref = torch.zeros(shape, dtype=torch.float32, device='cuda')
                        ref += reduce((dy.float().T @ x.float()) * d, rows, 64)
                        got = torch.zeros_like(ref)
                        tile_sums(dy, x, store.cand[name], rows, 64, store._luts(w.device), got)
                        e = errors(got.double(), ref.double())
                        worst = e if worst is None else {key: max(worst[key], e[key]) for key in e}
                    scratch = torch.zeros_like(ref)
                    t_legacy = timed(lambda: scratch.add_(reduce((dy.float().T @ x.float()) * d, rows, 64)))
                    t_b1 = timed(lambda: tile_sums(dy, x, store.cand[name], rows, 64, store._luts(w.device), scratch))
                    key = f'{model}:{name}:{kind}:{rows}x64'
                    out[key] = dict(model=model, matrix=name, shape=[n, k], kind=kind, unit=f'{rows}x64', errors=worst,
                                    ms_per_batch=dict(legacy=t_legacy, b1=t_b1))
                    print(f"{key:70s} normwise {worst['normwise_rel']:.1e} elementwise {worst['elementwise_rel_max']:.1e} "
                          f"legacy {t_legacy:.1f} ms  B1 {t_b1:.1f} ms", flush=True)
                    del store, base, d
                del w, b, a, p
                torch.cuda.empty_cache()
    return out


def main():
    out = Path(sys.argv[1])
    g = torch.Generator(device='cuda').manual_seed(0)
    chunked = chunked_tests(g)
    b1 = b1_tests(g)
    summary = dict(
        chunked_cases=len(chunked), chunked_all_bitwise=all(r['gradient_equal'] and r['kl_equal'] for r in chunked),
        b1_cases=len(b1),
        b1_max_normwise_rel=max(r['errors']['normwise_rel'] for r in b1.values()),
        b1_max_elementwise_rel=max(r['errors']['elementwise_rel_max'] for r in b1.values()),
        b1_within_tolerance=all(r['errors']['normwise_rel'] <= TOLERANCE for r in b1.values()),
        time_ms=dict(legacy=sum(r['ms_per_batch']['legacy'] for r in b1.values()),
                     b1=sum(r['ms_per_batch']['b1'] for r in b1.values())))
    summary['passed'] = summary['chunked_all_bitwise'] and summary['b1_within_tolerance']
    out.write_text(json.dumps(dict(summary=summary, chunked=chunked, b1=b1), indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    print('PASS' if summary['passed'] else 'FAIL')
    sys.exit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
