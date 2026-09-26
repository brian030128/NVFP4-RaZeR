"""TM-OPT+TC unit test (results/tm_opt/PROTOCOL_TC.md): the tile-gradient sums with the GEMM on the BF16 tensor cores
(FP32 accumulation and output) vs the FP32 path (TF32 off) vs an FP64 reference.

For one real matrix of every distinct layer-0 shape of Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 and a random matrix of
each shape, at 8x64, 16x64 and 256x64, for 4 batches of 4,096 tokens (synthetic heavy-tailed output gradients and
per-token-FourOverSix-quantized inputs, as test_tm_opt.py):
    ref  = reduce((dy^T x in FP64) * (A - B), rows, 64)
    fp32 = reduce((dy.float()^T @ x.float(), TF32 off) * (A - B), rows, 64)     run_train_map.py's hook
    tc   = reduce(tc_matmul(dy^T, x) * (A - B), rows, 64)                         the hook with --tile-grad-tc
Per case, each path's normwise relative error vs ref (max|d| / max|ref|) is the maximum over the 4 batches.
Registered tolerance, every case: error(tc) <= max(2 * error(fp32), 2e-6). Also reported: the GEMM time, FP32 vs TC.

python repro_local/realquant/test_tile_grad_tc.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_store import CandidateStore  # noqa: E402
from quantize.fused_fourover6 import fourover6_rows  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_multiround import reduce  # noqa: E402
from run_train_map import tc_matmul  # noqa: E402
from test_tile_score import RECORDS, layer0_matrices, load_weight, timed  # noqa: E402

MODELS = ('llama8b', 'mistral7b', 'phi4')
TOKENS, BATCHES = 4096, 4


def normwise(got, ref):
    return float((got - ref).abs().max() / ref.abs().max())


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device='cuda').manual_seed(0)
    cases = {}
    for model in MODELS:
        source, mats = layer0_matrices(RECORDS[model])
        for name, (n, k) in mats:
            for kind in ('real', 'random'):
                w = load_weight(source, name).cuda() if kind == 'real' else (torch.randn(n, k, generator=g, device='cuda') * 0.02).bfloat16()
                b, a = quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                p = pack(w, b, a)
                assert p is not None, name
                store = CandidateStore(8, 64)
                store.add(name, w, decode_base(p), decode_alt(p))
                base = store.decode(name, which='base')
                d = store.decode(name, which='alt').float() - base.float()
                d64 = d.double()
                batches = []
                for _ in range(BATCHES):
                    dy = (torch.randn(TOKENS, n, generator=g, device='cuda') * 1e-4 * torch.logspace(-1, 1, n, device='cuda')).bfloat16()
                    x = fourover6_rows((torch.randn(TOKENS, k, generator=g, device='cuda')
                                        * torch.logspace(-2, 1, k, device='cuda')[torch.randperm(k, generator=g, device='cuda')]).bfloat16())
                    g64 = dy.double().T @ x.double()
                    g32 = dy.float().T @ x.float()
                    gtc = tc_matmul(dy.T, x)
                    batches.append((g64, g32, gtc))
                    del dy, x
                assert not torch.backends.cuda.matmul.allow_tf32, 'TF32 must stay off'
                for rows in (8, 16, 256):
                    err = dict(fp32=0.0, tc=0.0)
                    for g64, g32, gtc in batches:
                        ref = reduce(g64 * d64, rows, 64)
                        err['fp32'] = max(err['fp32'], normwise(reduce(g32 * d, rows, 64).double(), ref))
                        err['tc'] = max(err['tc'], normwise(reduce(gtc * d, rows, 64).double(), ref))
                    threshold = max(2 * err['fp32'], 2e-6)
                    key = f'{model}:{name}:{kind}:{rows}x64'
                    cases[key] = dict(shape=[n, k], unit=f'{rows}x64', error_vs_fp64=err, threshold=threshold,
                                      passed=err['tc'] <= threshold)
                    print(f"{key:70s} fp32 {err['fp32']:.2e}  tc {err['tc']:.2e}  threshold {threshold:.2e}  "
                          f"{'ok' if cases[key]['passed'] else 'FAIL'}", flush=True)
                dy = (torch.randn(TOKENS, n, generator=g, device='cuda') * 1e-4).bfloat16()
                x = torch.randn(TOKENS, k, generator=g, device='cuda').bfloat16()
                t32 = timed(lambda: dy.float().T @ x.float())
                ttc = timed(lambda: tc_matmul(dy.T, x))
                for rows in (8, 16, 256):
                    cases[f'{model}:{name}:{kind}:{rows}x64']['gemm_ms'] = dict(fp32=t32, tc=ttc)
                del batches, store, base, d, d64, w, b, a, p, dy, x
                torch.cuda.empty_cache()
    summary = dict(cases=len(cases), passed=all(c['passed'] for c in cases.values()),
                   failures=[k for k, c in cases.items() if not c['passed']],
                   max_error_fp32=max(c['error_vs_fp64']['fp32'] for c in cases.values()),
                   max_error_tc=max(c['error_vs_fp64']['tc'] for c in cases.values()),
                   max_ratio_tc_over_fp32=max(c['error_vs_fp64']['tc'] / c['error_vs_fp64']['fp32'] for c in cases.values()),
                   gemm_ms_total=dict(fp32=sum(c['gemm_ms']['fp32'] for k, c in cases.items() if k.endswith(':8x64')),
                                      tc=sum(c['gemm_ms']['tc'] for k, c in cases.items() if k.endswith(':8x64'))))
    out.write_text(json.dumps(dict(summary=summary, cases=cases), indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    print('PASS' if summary['passed'] else 'FAIL')
    sys.exit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
