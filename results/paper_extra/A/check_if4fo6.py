#!/usr/bin/env python3
"""Amendment 2 of results/paper_extra/A/PROTOCOL.md: the IF4 (Cook et al.) + FourOverSix arm (rule if4fo6), checked on
real weights before any of its GPU runs (CPU; results in if4fo6_check.json).

    python results/paper_extra/A/check_if4fo6.py --model llama8b --out if4fo6_check.json

Registered checks (a failure stops the amendment's runs), on the 11 Llama-3.1-8B modules of the A1 check:
1. The FP candidate equals zoufo6's FP candidate (the repo's FourOverSix, quant_nvfp4_4over6) bitwise.
2. The uniform candidate equals IF4's INT4 candidate (quantize/adaptive_formats.py, verified against the official
   reference in A1) cast to the installed BF16, bitwise.
3. The per-block choice is INT4 exactly where its error is strictly lower (IF4's tie rule: ties keep the FP
   candidate), and the tile rule at (1, 16) equals the per-block rule.
4. The existing rules are unchanged: on 3 of the modules, for if4, zou, zoufo6, e2m1 and e2m1zou at 1x16 and 16x64, the
   installed weights equal those of the registered adaptive_formats.py (commit c923f32) bitwise.
Recorded, not rules: the INT4 share at 1x16, and the blocks whose two errors are equal.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'sm120'))
from quantize import adaptive_formats as AF  # noqa: E402

MODULES = ('model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj', 'model.layers.0.self_attn.v_proj',
           'model.layers.0.self_attn.o_proj', 'model.layers.0.mlp.gate_proj', 'model.layers.0.mlp.up_proj',
           'model.layers.0.mlp.down_proj', 'model.layers.15.self_attn.q_proj', 'model.layers.15.mlp.down_proj',
           'model.layers.31.self_attn.v_proj', 'model.layers.31.mlp.down_proj')
UNCHANGED_MODULES = ('model.layers.0.self_attn.q_proj', 'model.layers.31.self_attn.v_proj', 'model.layers.31.mlp.down_proj')
OLD_RULES = ('if4', 'zou', 'zoufo6', 'e2m1', 'e2m1zou')
REGISTERED = 'c923f32'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    from safetensors import safe_open
    old_src = subprocess.run(['git', '-C', str(REPO), 'show', f'{REGISTERED}:quantize/adaptive_formats.py'],
                             capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(old_src)
    OLD = load('adaptive_formats_registered', f.name)
    snap = Path(C.snapshot(args.model))
    index = json.loads((snap / 'model.safetensors.index.json').read_text())['weight_map']
    res = dict(model=args.model, snapshot=str(snap), registered_commit=REGISTERED, modules={}, unchanged={})
    ok = True
    for name in MODULES:
        with safe_open(str(snap / index[name + '.weight']), 'pt') as f:
            w = f.get_tensor(name + '.weight')
        d_fp, d_int, e_fp, e_int, zero = AF.candidates(w, 'if4fo6')
        z_fp = AF.candidates(w, 'zoufo6')[0]
        i_int = AF.candidates(w, 'if4')[1].to(w.dtype).float()
        uni = AF.select(e_fp, e_int, 'if4fo6')
        r = res['modules'][name] = dict(
            shape=list(w.shape),
            fp_equals_zoufo6_fp=bool(torch.equal(d_fp.view(torch.int32), z_fp.view(torch.int32))),
            int_equals_if4_int_installed=bool(torch.equal(d_int.view(torch.int32), i_int.view(torch.int32))),
            choice_is_strictly_lower=bool(torch.equal(uni, e_int < e_fp)),
            tile_1x16_equals_block=bool(torch.equal(AF.select(e_fp, e_int, 'if4fo6', (1, 16)), uni)),
            int_share_1x16=float(uni.float().mean()), blocks_with_equal_errors=int((e_fp == e_int).sum()), zero_scale_blocks=zero)
        passed = all(r[k] for k in ('fp_equals_zoufo6_fp', 'int_equals_if4_int_installed', 'choice_is_strictly_lower',
                                    'tile_1x16_equals_block'))
        ok &= passed
        print(name, 'PASS' if passed else 'FAIL', f"INT4 share {100 * r['int_share_1x16']:.2f} %", flush=True)
        if name in UNCHANGED_MODULES:
            for rule in OLD_RULES:
                for tile in ((1, 16), (16, 64)):
                    a = AF.quantize(w, rule, tile)[0]
                    b = OLD.quantize(w, rule, tile)[0]
                    same = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
                    res['unchanged'][f'{name} {rule} {tile[0]}x{tile[1]}'] = same
                    ok &= same
    res['passed'] = bool(ok)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    if not ok:
        raise SystemExit('if4fo6 check FAILED')
    print('if4fo6 check PASSED;', sum(res['unchanged'].values()), 'of', len(res['unchanged']), 'existing (rule, tile, module) '
          'outputs unchanged')


if __name__ == '__main__':
    main()
