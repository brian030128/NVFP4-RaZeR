"""Phase 2 unit tests for B1 (results/speedups/PROTOCOL_PHASE2.md): tile_score.tile_scores vs the legacy hook.

For real layers of the four final models (one matrix of every distinct shape, layer 0) and random layers of the
same shapes, at units 8x64 and 256x64, with a random current map (30 % E0M3 tiles): 16 batches of 8 sequences of 512
tokens (128 sequences, as a scoring pass), synthetic heavy-tailed output gradients and per-token-FourOverSix-quantized
inputs. Both paths accumulate the FP64 per-tile sum and sum of squares; mu and SE follow as in run_multiround.score.
Reported per layer and unit: max abs / normwise relative / elementwise relative error of g (per sequence and tile,
first batch), mu and SE; the agreement of the candidate sets {mu + 2 SE < 0}; and the time of one batch, legacy hook
vs B1.

python repro_local/realquant/test_tile_score.py OUT_JSON                       # Phase 2: 8x64 and 256x64, four models
python repro_local/realquant/test_tile_score.py OUT_JSON --rows 16 --models llama8b mistral7b phi4   # 16x64 addendum
"""
import argparse
import json
import math
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
from run_multiround import expand, reduce  # noqa: E402
from tile_score import tile_scores  # noqa: E402

RECORDS = {'llama8b': '/home/dev/n16k64_campaign/cost_comparison/data/llama8b/calibration/report.json',
           'mistral7b': '/home/dev/n16k64_campaign/multimodel/data/mistral7b/calibration/report.json',
           'phi4': '/home/dev/n16k64_campaign/multimodel/data/phi4/calibration/report.json',
           'qwen27b': '/home/dev/n16k64_campaign/multimodel/data/qwen27b/calibration/report.json'}
BATCHES, B, T = 16, 8, 512


def layer0_matrices(record):
    prior = json.loads(Path(record).read_text())
    seen, out = set(), []
    for name, m in prior['matrices'].items():
        if '.layers.0.' not in name and '.layers.3.' not in name:      # Qwen3.8-27B: layer 3 is full attention
            continue
        shape = tuple(m['shape'])
        if shape not in seen:
            seen.add(shape)
            out.append((name, shape))
    return prior['source'], out


def load_weight(source, name):
    from safetensors import safe_open
    index = json.loads((Path(source) / 'model.safetensors.index.json').read_text())['weight_map']
    key = name + '.weight'
    if key not in index:
        key = next(k for k in index if k.endswith(name.split('model.', 1)[-1] + '.weight'))
    with safe_open(str(Path(source) / index[key]), 'pt') as f:
        return f.get_tensor(key)


def legacy_batch(dy, x, store, name, sel, rows, total, square, keep=None):
    base = store.decode(name, which='base')
    d = (store.decode(name, which='alt').float() - base.float()) * torch.where(expand(sel, rows, 64, base.shape[0]), -1., 1.)
    for i in range(dy.shape[0]):
        value = reduce((dy[i].float().T @ x[i].float()) * d, rows, 64).double()
        if keep is not None:
            keep.append(value)
        total += value
        square += value.square()


def stats(total, square, n):
    mean = total / n
    se = ((square - n * mean.square()).clamp_min(0) / (n - 1)).sqrt() / math.sqrt(n)
    return mean, se


def errors(got, ref):
    diff = (got - ref).abs()
    big = ref.abs() >= 1e-3 * ref.abs().max()
    return dict(max_abs=float(diff.max()), normwise_rel=float(diff.max() / ref.abs().max().clamp_min(1e-300)),
                elementwise_rel_max=float((diff[big] / ref.abs()[big]).max()) if big.any() else 0.0)


def timed(fn, reps=3):
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--rows', type=int, nargs='+', default=[8, 256], choices=(8, 16, 256), help='unit rows (x 64 columns)')
    ap.add_argument('--models', nargs='+', default=list(RECORDS), choices=tuple(RECORDS))
    args = ap.parse_args()
    out = args.out
    g = torch.Generator(device='cuda').manual_seed(0)
    result = dict(batches=BATCHES, batch=B, tokens=T, units=[f'{r}x64' for r in args.rows], models=args.models, layers={})
    for model, record in RECORDS.items():
        if model not in args.models:
            continue
        source, mats = layer0_matrices(record)
        for name, (n, k) in mats:
            for kind in ('real', 'random'):
                w = load_weight(source, name).cuda() if kind == 'real' else (torch.randn(n, k, generator=g, device='cuda') * 0.02).bfloat16()
                b, a = quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
                p = pack(w, b, a)
                assert p is not None, name
                for rows in args.rows:
                    store = CandidateStore(rows, 64)
                    store.add(name, w, decode_base(p), decode_alt(p))
                    sel = torch.rand(-(-n // rows), k // 64, generator=g, device='cuda') < 0.3
                    tot = [torch.zeros(sel.shape, dtype=torch.float64, device='cuda') for _ in range(4)]
                    first_legacy, first_fused = [], []
                    time_legacy = time_fused = None
                    for batch in range(BATCHES):
                        dy = (torch.randn(B, T, n, generator=g, device='cuda') * 1e-4
                              * torch.logspace(-1, 1, n, device='cuda')).bfloat16()
                        x = fourover6_rows((torch.randn(B, T, k, generator=g, device='cuda')
                                            * torch.logspace(-2, 1, k, device='cuda')[torch.randperm(k, generator=g, device='cuda')]).bfloat16())
                        legacy_batch(dy, x, store, name, sel, rows, tot[0], tot[1], first_legacy if batch == 0 else None)
                        tile_scores(dy, x, store.cand[name], sel, rows, 64, store._luts(w.device), tot[2], tot[3])
                        if batch == 0:
                            for i in range(B):     # one sequence at a time from zero: that sequence's g
                                s1 = torch.zeros(sel.shape, dtype=torch.float64, device='cuda')
                                tile_scores(dy[i:i + 1], x[i:i + 1], store.cand[name], sel, rows, 64, store._luts(w.device),
                                            s1, torch.zeros_like(s1))
                                first_fused.append(s1)
                            scratch = [torch.zeros_like(tot[0]) for _ in range(2)]
                            time_legacy = timed(lambda: legacy_batch(dy, x, store, name, sel, rows, *scratch))
                            time_fused = timed(lambda: tile_scores(dy, x, store.cand[name], sel, rows, 64,
                                                                   store._luts(w.device), *scratch))
                    n_seq = BATCHES * B
                    (lm, ls), (fm, fs) = stats(tot[0], tot[1], n_seq), stats(tot[2], tot[3], n_seq)
                    cand_l, cand_f = (lm + 2 * ls) < 0, (fm + 2 * fs) < 0
                    row = dict(model=model, matrix=name, shape=[n, k], kind=kind, unit=f'{rows}x64',
                               g=errors(torch.stack(first_fused), torch.stack(first_legacy)),
                               mu=errors(fm, lm), se=errors(fs, ls),
                               candidates=dict(legacy=int(cand_l.sum()), b1=int(cand_f.sum()),
                                               agree=int((cand_l == cand_f).sum()), tiles=int(cand_l.numel()),
                                               disagree=int((cand_l != cand_f).sum())),
                               ms_per_batch=dict(legacy=time_legacy, b1=time_fused))
                    key = f'{model}:{name}:{kind}:{rows}x64'
                    result['layers'][key] = row
                    print(f"{key:70s} g {row['g']['normwise_rel']:.1e} mu {row['mu']['normwise_rel']:.1e} "
                          f"se {row['se']['normwise_rel']:.1e} cand disagree {row['candidates']['disagree']}/"
                          f"{row['candidates']['tiles']}  legacy {time_legacy:.1f} ms  B1 {time_fused:.1f} ms", flush=True)
                    del store
                del w, b, a, p
                torch.cuda.empty_cache()
    rows_ = list(result['layers'].values())
    result['summary'] = dict(
        max_normwise_rel=dict(g=max(r['g']['normwise_rel'] for r in rows_), mu=max(r['mu']['normwise_rel'] for r in rows_),
                              se=max(r['se']['normwise_rel'] for r in rows_)),
        max_elementwise_rel=dict(g=max(r['g']['elementwise_rel_max'] for r in rows_),
                                 mu=max(r['mu']['elementwise_rel_max'] for r in rows_),
                                 se=max(r['se']['elementwise_rel_max'] for r in rows_)),
        candidate_disagreements=sum(r['candidates']['disagree'] for r in rows_),
        candidate_tiles=sum(r['candidates']['tiles'] for r in rows_),
        time_ms=dict(legacy=sum(r['ms_per_batch']['legacy'] for r in rows_), b1=sum(r['ms_per_batch']['b1'] for r in rows_)))
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
