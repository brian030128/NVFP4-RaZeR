"""Checks 1-3 of PROTOCOL.md for the N16K64 Qwen3-4B one-shot k=3 maps, and their conversion.

Checks 1-2 are device-independent; check 3 runs on the GPU, where both evaluators install weights (deviation 1).

python results/multiround_models/qwen4b/oneshot_k3_eval/convert_check.py DATA_ROOT OUT_DIR
Writes OUT_DIR/<label>.pt (run_multiround map format, 8x64 unit) and prints/writes checks.json here.
Exit 1 on any failed check (a sign-of-zero-only difference in E0M3 tiles is reported, not a failure).
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'research/n16k64/software/primary'))
from campaign import mapio, quant as Q, tiles as T  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_multiround import expand  # noqa: E402

MAPS = Path('/home/dev/n16k64_campaign/runs/calib_qwen4b_attempt1/maps')
FILES = {'N16K64-n8-k3': 'qwen4b_seed0_n8_k3.mixfp4map', 'N16K64-n16-k3': 'qwen4b_seed0_n16_k3.mixfp4map'}
MODEL_ID, REVISION = 'Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'
DEVICE = 'cuda'


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main():
    data_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    prior = json.loads((data_root / 'qwen4b' / 'calibration' / 'report.json').read_text())
    shapes = {n: tuple(m['shape']) for n, m in prior['matrices'].items()}
    result, converted, failed = dict(maps={}, candidates={}), {}, []
    for label, name in FILES.items():
        path = MAPS / name
        prov = json.loads((path.parent / (name + '.provenance.json')).read_text())
        header, masks, digest = mapio.read_map(path, expected_sha256=prov['map_sha256'])
        tb = tuple(header['type_block'])
        checks = dict(file_sha256=digest, file_sha256_equals_provenance=digest == prov['map_sha256'],
                      payload_sha256_equals_provenance=mapio.payload_sha256(masks) == prov['mask_payload_sha256'],
                      model_id=header['model'].get('model_id') == MODEL_ID,
                      model_revision=header['model'].get('revision') == REVISION,
                      tokenizer_revision=header['model'].get('tokenizer_revision') == REVISION,
                      type_block=list(tb), module_names_and_order=[m['name'] for m in header['modules']] == list(shapes),
                      weight_shapes=all(tuple(m['weight_shape']) == shapes[m['name']] for m in header['modules']),
                      rule=prov.get('rule'), header_policy=header['policy'], selected_tiles=header['totals']['selected_tiles'])
        rows = tb[0]
        assert rows in (8, 16) and tb[1] == 64
        sel, per_module_ok = {}, True
        for m in header['modules']:
            n, mask = m['name'], masks[m['name']]
            o, k = shapes[n]
            fine = mask.repeat_interleave(rows // 8, 0)[:-(-o // 8)]
            if not torch.equal(expand(fine, 8, 64, o), expand(mask, rows, 64, o)):
                per_module_ok = False
            if int(fine.sum()) != m['selected'] * (rows // 8):
                per_module_ok = False
            sel[n] = fine
        checks['element_masks_equal_after_conversion_and_per_module_counts'] = per_module_ok
        checks['converted_8x64_tiles'] = sum(int(s.sum()) for s in sel.values())
        checks['converted_total_equals_header'] = checks['converted_8x64_tiles'] == header['totals']['selected_tiles'] * (rows // 8)
        torch.save(sel, out_dir / f'{label}.pt')
        checks['converted_file'] = str(out_dir / f'{label}.pt')
        checks['converted_sha256'] = hashlib.sha256((out_dir / f'{label}.pt').read_bytes()).hexdigest()
        ok = all(v for k, v in checks.items() if isinstance(v, bool))
        checks['passed'] = ok
        failed += [] if ok else [label]
        result['maps'][label] = checks
        converted[label] = (masks, tb, sel)
        print(label, json.dumps({k: v for k, v in checks.items() if k != 'header_policy'}), flush=True)
    # Check 3 on every module: candidate functions, and the installed weight for both maps
    from safetensors import safe_open
    snapshot = Path(prior['source'])
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {}
    stats = dict(modules=0, functions_bitwise=0, **{label: dict(bitwise=0, value_equal=0, differing_elements=0,
                                                                  differing_only_zero_sign=True, sign_zero_in_e0m3_tiles_only=True)
                                                     for label in FILES})
    for n in shapes:
        key = n + '.weight'
        f = index[key]
        if f not in handles:
            handles[f] = safe_open(str(snapshot / f), 'pt')
        w = handles[f].get_tensor(key).to(DEVICE)
        o, k = w.shape
        b, a = quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        cb, ca = Q.four_over_six(w), Q.e0m3(w)
        stats['functions_bitwise'] += int(bits_equal(b, cb) and bits_equal(a, ca))
        p = pack(w, b, a)
        assert p is not None, n
        base, alt = decode_base(p), decode_alt(p)
        for label, (masks, tb, sel) in converted.items():
            campaign = T.apply_mask(cb, ca, masks[n].to(DEVICE), tb)
            ours = torch.where(expand(sel[n].to(DEVICE), 8, 64, o), alt, base)
            s = stats[label]
            if bits_equal(campaign, ours):
                s['bitwise'] += 1
            differ = campaign.view(torch.int16) != ours.view(torch.int16)
            if torch.equal(campaign, ours):
                s['value_equal'] += 1
            s['differing_elements'] += int(differ.sum())
            if differ.any():
                zero = (campaign == 0) & (ours == 0)
                s['differing_only_zero_sign'] &= bool(zero[differ].all())
                s['sign_zero_in_e0m3_tiles_only'] &= bool(expand(sel[n].to(DEVICE), 8, 64, o)[differ].all())
        stats['modules'] += 1
    result['candidates'] = dict(stats, device=torch.cuda.get_device_name(0))
    cand_ok = stats['functions_bitwise'] == stats['modules'] and all(
        stats[label]['value_equal'] == stats['modules'] and stats[label]['differing_only_zero_sign']
        and stats[label]['sign_zero_in_e0m3_tiles_only'] for label in FILES)
    result['candidates_passed'] = cand_ok
    result['passed'] = not failed and cand_ok
    (HERE / 'checks.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['candidates'], indent=1))
    print('PASS' if result['passed'] else 'FAIL', flush=True)
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
