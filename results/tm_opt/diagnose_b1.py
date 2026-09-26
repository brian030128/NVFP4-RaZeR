"""Diagnostic after the registered B1 unit-test failure (PROTOCOL.md deviation 1): how accurate are the legacy hook
and B1 against an FP64 reference, and how fast is B1 for the batch-sum shape under other launch configurations?

For one real matrix of every distinct layer-0 shape of Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, at 8x64 and 16x64,
4 batches of 8 x 512 tokens (the unit test's generators): normwise relative error (max|d| / max|ref|) of the legacy
FP32 hook and of B1 against reduce((dy^T x) * (A - B)) in FP64, and of B1 against the legacy hook. Then the time of one
batch, legacy hook vs B1 at several (block rows, token block, warps), on the three largest shapes. Not a criterion.

python results/tm_opt/diagnose_b1.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'repro_local' / 'realquant'))
from candidate_store import CandidateStore  # noqa: E402
from quantize.fused_fourover6 import fourover6_rows  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_multiround import reduce  # noqa: E402
from test_tile_score import RECORDS, layer0_matrices, load_weight, timed  # noqa: E402
from tile_score import tile_scores, tile_sums  # noqa: E402

CONFIGS = ((64, 32, 4), (64, 64, 4), (64, 128, 4), (64, 64, 8), (128, 32, 8), (128, 64, 8), (32, 64, 4))


def normwise(got, ref):
    return float((got - ref).abs().max() / ref.abs().max())


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    g = torch.Generator(device='cuda').manual_seed(0)
    accuracy, speed = {}, {}
    for model in ('llama8b', 'mistral7b', 'phi4'):
        source, mats = layer0_matrices(RECORDS[model])
        for name, (n, k) in mats:
            w = load_weight(source, name).cuda()
            b, a = quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            p = pack(w, b, a)
            for rows in (8, 16):
                store = CandidateStore(rows, 64)
                store.add(name, w, decode_base(p), decode_alt(p))
                base = store.decode(name, which='base')
                d = store.decode(name, which='alt').float() - base.float()
                errs = dict(legacy_vs_fp64=0.0, b1_vs_fp64=0.0, b1_vs_legacy=0.0)
                for _ in range(4):
                    dy = (torch.randn(8 * 512, n, generator=g, device='cuda') * 1e-4
                          * torch.logspace(-1, 1, n, device='cuda')).bfloat16()
                    x = fourover6_rows((torch.randn(8 * 512, k, generator=g, device='cuda')
                                        * torch.logspace(-2, 1, k, device='cuda')[torch.randperm(k, generator=g, device='cuda')]).bfloat16())
                    ref = reduce((dy.double().T @ x.double()) * d.double(), rows, 64)
                    legacy = reduce((dy.float().T @ x.float()) * d, rows, 64)
                    got = torch.zeros(legacy.shape, dtype=torch.float32, device='cuda')
                    tile_sums(dy, x, store.cand[name], rows, 64, store._luts(w.device), got)
                    errs = dict(legacy_vs_fp64=max(errs['legacy_vs_fp64'], normwise(legacy.double(), ref)),
                                b1_vs_fp64=max(errs['b1_vs_fp64'], normwise(got.double(), ref)),
                                b1_vs_legacy=max(errs['b1_vs_legacy'], normwise(got.double(), legacy.double())))
                    del ref, legacy, got
                key = f'{model}:{name}:{rows}x64'
                accuracy[key] = errs
                print(key, json.dumps({k: f'{v:.2e}' for k, v in errs.items()}), flush=True)
                if rows == 8 and n * k >= 4096 * 14336:
                    zero = torch.zeros(-(-n // rows), k // 64, dtype=torch.bool, device='cuda')
                    tot = torch.zeros(zero.shape, dtype=torch.float64, device='cuda')
                    scratch = torch.zeros(zero.shape, dtype=torch.float32, device='cuda')
                    row = dict(legacy=timed(lambda: scratch.add_(reduce((dy.float().T @ x.float()) * d, rows, 64))))
                    for br, bt, warps in CONFIGS:
                        try:
                            row[f'b1 br{br} bt{bt} w{warps}'] = timed(lambda: tile_scores(
                                dy.reshape(1, -1, n), x.reshape(1, -1, k), store.cand[name], zero, rows, 64,
                                store._luts(w.device), tot, torch.zeros_like(tot), br=br, bt=bt, num_warps=warps))
                        except Exception as error:          # e.g. the launch needs more shared memory than the GPU has
                            row[f'b1 br{br} bt{bt} w{warps}'] = f'not launchable: {type(error).__name__}'
                    speed[key] = row
                    print(key, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                del store, base, d
            del w, b, a, p
            torch.cuda.empty_cache()
    summary = dict(max_legacy_vs_fp64=max(v['legacy_vs_fp64'] for v in accuracy.values()),
                   max_b1_vs_fp64=max(v['b1_vs_fp64'] for v in accuracy.values()),
                   max_b1_vs_legacy=max(v['b1_vs_legacy'] for v in accuracy.values()),
                   cases_b1_closer_to_fp64=sum(v['b1_vs_fp64'] <= v['legacy_vs_fp64'] for v in accuracy.values()),
                   cases=len(accuracy),
                   total_ms=({k: sum(r[k] for r in speed.values()) for k in next(iter(speed.values()))
                              if all(isinstance(r[k], float) for r in speed.values())} if speed else {}))
    out.write_text(json.dumps(dict(summary=summary, accuracy=accuracy, speed=speed), indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
