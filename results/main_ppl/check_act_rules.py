#!/usr/bin/env python3
"""The main PPL table's activation rules (quantize/adaptive_formats.quantize_rows) against the originals, on real
activations (CPU; run once before registration, results in act_rules_check.json).

    python results/main_ppl/check_act_rules.py --fouroversix PATH/fouroversix --model qwen3_1p7b --out act_rules_check.json

The per-token rule is the per-tensor rule of Experiment A applied to one token at a time. So, for every token (row) r
of a real activation X:
1. IF4: quantize_rows(X, 'if4')[r] equals, bitwise in BF16:
   - Cook et al.'s official reference (mit-han-lab/fouroversix @ dadfad69, intfloat_block_scaled_quantization with
     DataType.if4, ScaleRule.mse, RoundStyle.nearest) run on X[r:r+1] alone, its amax being that row's. The same
     per-block choice is required too.
   - Experiment A's quantize(X[r:r+1], 'if4').
2. MixFP4 (Zou et al., Algorithm 1; no official code): quantize_rows(X, 'zou')[r] equals quantize(X[r:r+1], 'zou')
   bitwise.
3. Edge rows, through the same comparisons:
   - an all-zero token;
   - a token whose small blocks underflow the E4M3 block scale;
   - blocks with exact ties between the two candidates;
   - one large outlier.
The activations are the inputs of q_proj, o_proj, gate_proj and down_proj in three layers of the model, on the first
WikiText-2 window (256 tokens), in BF16 as the model computes them.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from quantize import adaptive_formats as AF  # noqa: E402


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture(C, key, tokens, layers):
    model = C.load_model(key, device_map='cpu')
    tok = C.load_tokenizer(key)
    wins, _, status = C.windows(tok, key, 'wiki', check=False)
    ids = wins[0][:, :tokens]
    n_layers = model.config.num_hidden_layers
    picks = sorted({0, n_layers // 2, n_layers - 1}) if layers is None else layers
    want = {f'model.layers.{i}.{p}' for i in picks for p in ('self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj',
                                                              'mlp.down_proj')}
    acts, hooks = {}, []
    for n, m in model.named_modules():
        if n in want:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp, _n=n: acts.__setitem__(_n, inp[0].detach().clone())))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    return {n: a.reshape(-1, a.shape[-1]) for n, a in sorted(acts.items())}, dict(window_status=status, tokens=tokens,
                                                                                    layers=picks)


def edge_rows(k, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    rows = {}
    rows['all_zero'] = torch.zeros(1, k)
    x = torch.randn(1, k, generator=g) * 1e-6
    x[0, 0] = 1e4                                                 # the small blocks underflow the E4M3 block scale
    rows['underflow'] = x
    t = torch.zeros(1, k)
    for b in range(k // 16):                                     # values on both grids: E2M1 {0, .5, ..., 6} x 7/6 ...
        t[0, 16 * b:16 * b + 16] = torch.tensor([6, 3, 2, 1, 0, -1, -2, -3, 6, 0, 0, 0, 1, 2, 3, -6]) * (b + 1)
    rows['ties'] = t
    o = torch.randn(1, k, generator=g)
    o[0, 5] = 300.0
    rows['outlier'] = o
    return {n: r.to(dtype) for n, r in rows.items()}


def compare(ref, U, official_if4, X):
    """Per row: IF4 against the official reference and Experiment A's per-tensor rule; Zou against A's rule."""
    mine_if4 = AF.quantize_rows(X, 'if4')
    mine_zou = AF.quantize_rows(X, 'zou')
    st = AF.RowStats()
    AF.quantize_rows(X, 'if4', st)
    out = dict(rows=X.shape[0], cols=X.shape[1], if4_rows_equal_official=0, if4_choice_equal_official=0,
               if4_rows_equal_A=0, zou_rows_equal_A=0, if4_int_blocks=0, zou_uniform_blocks=0, blocks=0)
    stz = AF.RowStats()
    AF.quantize_rows(X, 'zou', stz)
    for r in range(X.shape[0]):
        row = X[r:r + 1]
        is_int, _, _, d_off = official_if4(ref, U, row)
        d_fp, d_int, e_fp, e_int, _ = AF.candidates(row, 'if4')
        mine_choice = AF.select(e_fp, e_int, 'if4').reshape(-1)
        out['if4_choice_equal_official'] += int(torch.equal(is_int, mine_choice))
        out['if4_rows_equal_official'] += int(torch.equal(d_off.reshape(1, -1).to(X.dtype).view(torch.int16),
                                                          mine_if4[r:r + 1].view(torch.int16)))
        a_if4, _ = AF.quantize(row, 'if4')
        out['if4_rows_equal_A'] += int(torch.equal(a_if4.view(torch.int16), mine_if4[r:r + 1].view(torch.int16)))
        a_zou, _ = AF.quantize(row, 'zou')
        out['zou_rows_equal_A'] += int(torch.equal(a_zou.view(torch.int16), mine_zou[r:r + 1].view(torch.int16)))
    s, z = st.as_dict(), stz.as_dict()
    out.update(blocks=s['blocks'], if4_int_blocks=s['uniform_blocks'], if4_zero_scale_blocks=s['zero_scale_blocks'],
               zou_uniform_blocks=z['uniform_blocks'], zou_zero_scale_blocks=z['zero_scale_blocks'])
    out['all_equal'] = (out['if4_rows_equal_official'] == out['if4_choice_equal_official'] == out['if4_rows_equal_A']
                        == out['zou_rows_equal_A'] == out['rows'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fouroversix', type=Path, required=True, help='a checkout of mit-han-lab/fouroversix @ dadfad69')
    ap.add_argument('--model', default='qwen3_1p7b')
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    A1 = load_path('a1_check_formats', REPO / 'results' / 'paper_extra' / 'A' / 'check_formats.py')
    ref, U = A1.load_official(args.fouroversix)
    sys.path.insert(0, str(REPO / 'sm120'))
    C = load_path('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    acts, meta = capture(C, args.model, args.tokens, None)
    res = dict(model=args.model, fouroversix=str(args.fouroversix), capture=meta, activations={}, edge_rows={})
    for n, X in acts.items():
        r = res['activations'][n] = compare(ref, U, A1.official_if4, X)
        print(n, json.dumps(r), flush=True)
    k = next(iter(acts.values())).shape[1]
    for n, X in edge_rows(k).items():
        r = res['edge_rows'][n] = compare(ref, U, A1.official_if4, X)
        print('edge', n, json.dumps(r), flush=True)
    allr = list(res['activations'].values()) + list(res['edge_rows'].values())
    res['summary'] = dict(matrices=len(res['activations']), rows=sum(r['rows'] for r in allr),
                          all_equal=all(r['all_equal'] for r in allr),
                          if4_int_fraction=sum(r['if4_int_blocks'] for r in res['activations'].values())
                          / sum(r['blocks'] for r in res['activations'].values()),
                          zou_uniform_fraction=sum(r['zou_uniform_blocks'] for r in res['activations'].values())
                          / sum(r['blocks'] for r in res['activations'].values()),
                          underflow_row_zero_scale_blocks=dict(if4=res['edge_rows']['underflow']['if4_zero_scale_blocks'],
                                                               zou=res['edge_rows']['underflow']['zou_zero_scale_blocks']))
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print('SUMMARY', json.dumps(res['summary']), flush=True)
    sys.exit(0 if res['summary']['all_equal'] else 1)


if __name__ == '__main__':
    main()
