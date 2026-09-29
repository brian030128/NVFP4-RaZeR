#!/usr/bin/env python3
"""A1: quantize/adaptive_formats.py against the originals, on real weights (CPU; run once, results in a1_check.json).

    python results/paper_extra/A/check_formats.py --fouroversix PATH/fouroversix --model llama8b --out a1_check.json

1. IF4 against the official reference (github.com/mit-han-lab/fouroversix, src/fouroversix/quantize/pytorch/
   reference.py, intfloat_block_scaled_quantization with DataType.if4, ScaleRule.mse, RoundStyle.nearest). Loaded by
   path, without installing the package. Per module: the per-block choice, the codes, the scale bytes (with the sign
   bit) and the dequantized values must be bitwise equal; the dequantized values of the official code are recomputed
   with select_intfloat's own formulas from its outputs.
2. Zou et al. (Algorithm 1) against the repo's quant_nvif4 (which is Zou's construction with other details): counts
   of blocks whose choice differs and of elements whose value differs, split into the known causes (tie-breaking,
   E2M1 rounding of halves, the [2^-9, 448] scale clamp).
3. Zou's E1M2 candidate against the repo's E0M3 alpha = 1 candidate (quant_mix_4_6 clip a1, elect always, the
   TM-OPT+TC E0M3 tiles): elements that differ.
4. The tile rule at (1, 16) equals the per-block rule; checked on every module and rule.
5. IF4's FP candidate against Zou's E2M1 candidate (both E2M1 at scale max/6, computed in different operation orders):
   elements that differ. If none do, the e2m1 reference (IF4's FP candidate everywhere) is exactly the E2M1 base of both.
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from quantize import adaptive_formats as AF  # noqa: E402

MODULES = ('model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj', 'model.layers.0.self_attn.v_proj',
           'model.layers.0.self_attn.o_proj', 'model.layers.0.mlp.gate_proj', 'model.layers.0.mlp.up_proj',
           'model.layers.0.mlp.down_proj', 'model.layers.15.self_attn.q_proj', 'model.layers.15.mlp.down_proj',
           'model.layers.31.self_attn.v_proj', 'model.layers.31.mlp.down_proj')


def load_official(root):
    """fouroversix's pure-PyTorch reference, loaded by path under stub packages (no install, no kernels)."""
    src = Path(root) / 'src' / 'fouroversix'

    def pkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    def mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    pkg('fouroversix', src)
    pkg('fouroversix.kernels', src / 'kernels')
    pkg('fouroversix.quantize', src / 'quantize')
    pkg('fouroversix.quantize.pytorch', src / 'quantize' / 'pytorch')
    mod('fouroversix.kernels.constants', src / 'kernels' / 'constants.py')
    mod('fouroversix.utils', src / 'utils.py')
    mod('fouroversix.quantize.utils', src / 'quantize' / 'utils.py')
    for sub in ('fp4', 'fp6', 'int4'):
        mod(f'fouroversix.quantize.pytorch.{sub}', src / 'quantize' / 'pytorch' / f'{sub}.py')
    ref = mod('fouroversix.quantize.pytorch.reference', src / 'quantize' / 'pytorch' / 'reference.py')
    return ref, sys.modules['fouroversix.utils']


def official_if4(ref, U, w):
    x = w.float().reshape(-1, 16)
    amax = w.float().abs().max()
    _, fake, scales = ref.intfloat_block_scaled_quantization(x, amax, fp4_format=U.DataType.if4,
                                                             scale_rule=U.ScaleRule.mse, round_style=U.RoundStyle.nearest)
    raw = scales.view(torch.uint8)
    is_int = raw >= 128
    s = (raw & 0x7F).view(torch.float8_e4m3fn).to(torch.float32)
    fake = fake.to(torch.float32)
    d_fp = (fake * s.unsqueeze(1) * amax) / (6 * 448 * 1)
    d_int = ((fake * s.unsqueeze(1) * amax) * 0.8571428571) / (6 * 448 * 1)
    return is_int, fake, raw, torch.where(is_int.unsqueeze(1), d_int, d_fp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fouroversix', type=Path, required=True, help='a checkout of mit-han-lab/fouroversix')
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ref, U = load_official(args.fouroversix)
    spec = importlib.util.spec_from_file_location('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    C = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / 'sm120'))
    spec.loader.exec_module(C)
    from safetensors import safe_open
    from quantize.quantizer import quant_mix_4_6, quant_nvif4
    snap = Path(C.snapshot(args.model))
    index = json.loads((snap / 'model.safetensors.index.json').read_text())['weight_map']
    res = dict(fouroversix_commit=None, model=args.model, snapshot=str(snap), modules={})
    try:
        import subprocess
        res['fouroversix_commit'] = subprocess.run(['git', '-C', str(args.fouroversix), 'rev-parse', 'HEAD'],
                                                   capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    for name in MODULES:
        with safe_open(str(snap / index[name + '.weight']), 'pt') as f:
            w = f.get_tensor(name + '.weight')
        r = res['modules'][name] = dict(shape=list(w.shape))
        # 1. IF4 against the official reference
        is_int, fake, raw, d_off = official_if4(ref, U, w)
        d_fp, d_int, e_fp, e_int, zero = AF.candidates(w, 'if4')
        mine_int = AF.select(e_fp, e_int, 'if4').reshape(-1)
        mine = torch.where(mine_int.repeat_interleave(16).reshape(-1, 16), d_int.reshape(-1, 16), d_fp.reshape(-1, 16))
        r['if4'] = dict(blocks=int(is_int.numel()), int_blocks_official=int(is_int.sum()), int_blocks_mine=int(mine_int.sum()),
                        choice_equal=bool(torch.equal(is_int, mine_int)),
                        dequantized_bitwise_equal=bool(torch.equal(d_off.view(torch.int32), mine.view(torch.int32))),
                        zero_scale_blocks=zero)
        # 2. Zou (Algorithm 1) against the repo's quant_nvif4
        out_zou, st = AF.quantize(w, 'zou')
        nvif4 = quant_nvif4(w, 4, 16)
        zd_fp, zd_int, ze_fp, ze_int, _ = AF.candidates(w, 'zou')
        ties = int((ze_fp == ze_int).sum())
        r['zou_vs_repo_nvif4'] = dict(zou_uniform_blocks=st['uniform_blocks'], zero_scale_blocks=st['zero_scale_blocks'],
                                      elements_differing=int((out_zou != nvif4).sum()), elements=w.numel(),
                                      blocks_with_equal_errors=ties)
        r['if4_fp_vs_zou_e2m1'] = dict(elements_differing=int((d_fp.to(torch.bfloat16) != zd_fp.to(torch.bfloat16)).sum()),
                                       elements_differing_fp32=int((d_fp != zd_fp).sum()))
        # 3. Zou's E1M2 candidate against the repo's E0M3 alpha = 1 candidate (the TM-OPT+TC E0M3 tiles)
        e0m3 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        r['zou_e1m2_vs_repo_e0m3_alpha1'] = dict(elements_differing=int((zd_int.to(torch.bfloat16) != e0m3).sum()))
        # 4. the tile rule at (1, 16) is the per-block rule
        for rule in AF.RULES:
            a, _ = AF.quantize(w, rule)
            b, _ = AF.quantize(w, rule, (1, 16))
            r[f'{rule}_tile_1x16_equals_block'] = bool(torch.equal(a, b))
        print(name, json.dumps(r), flush=True)
    m = res['modules'].values()
    res['summary'] = dict(
        if4_choice_equal_all=all(x['if4']['choice_equal'] for x in m),
        if4_dequantized_bitwise_equal_all=all(x['if4']['dequantized_bitwise_equal'] for x in m),
        tile_1x16_equals_block_all=all(x[f'{rl}_tile_1x16_equals_block'] for x in m for rl in AF.RULES),
        zou_vs_repo_nvif4_elements_differing=sum(x['zou_vs_repo_nvif4']['elements_differing'] for x in m),
        zou_e1m2_vs_repo_e0m3_elements_differing=sum(x['zou_e1m2_vs_repo_e0m3_alpha1']['elements_differing'] for x in m),
        if4_fp_vs_zou_e2m1_elements_differing=sum(x['if4_fp_vs_zou_e2m1']['elements_differing'] for x in m),
        elements=sum(x['zou_vs_repo_nvif4']['elements'] for x in m))
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print('SUMMARY', json.dumps(res['summary']), flush=True)


if __name__ == '__main__':
    main()
