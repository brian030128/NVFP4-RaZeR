"""GPU unit gate for native_dev.py (run_multiround.py --shadow-native) on real Llama-3.1-8B weights.

For the 7 matrices of the first and last decoder layer:
  * both packed candidates decode bitwise to what run_multiround.py's apply() installs:
    decode_base / decode_alt of quantize/packed_candidates, which are value-equal to
    quant_nvfp4_4over6 / quant_mix_4_6(elect='always'), and bitwise equal up to zero signs;
  * the packed weight of a random 256x64 and 8x64 map decodes bitwise to
    torch.where(expand(map), alternative, base), the weight run_multiround.py's apply() installs;
  * a native forward on a (16, 512, K) batch reproduces quant_per_document's activation bitwise,
    and its output is close to the fake-quant F.linear (FP32 FP4 accumulation vs BF16 GEMM).

python repro_local/realquant/test_native_dev.py DATA_ROOT
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import native_dev  # noqa: E402
from quantize.fast_act import quant_per_document  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_math_code_calibration import load_model  # noqa: E402
from run_multiround import data_paths, expand  # noqa: E402


@torch.no_grad()
def main():
    prior = json.loads((data_paths('llama8b', Path(sys.argv[1]))[0] / 'report.json').read_text())
    model, modules = load_model(prior, False)
    names = [n for n in modules if n.startswith(('model.layers.0.', 'model.layers.31.'))]
    tokens = 16 * 512
    results = {}
    for rows, cols in ((256, 64), (8, 64)):
        sub = {n: modules[n] for n in names}
        nd = native_dev.NativeDev(sub, rows, cols, tokens, check_calls=10 ** 9)
        g = torch.Generator(device='cuda').manual_seed(0)
        sel, fake = {}, {}
        for n, m in sub.items():
            w = m.weight
            # the candidates apply() installs: decode_base / decode_alt of the packed candidates
            p = pack(w, quant_nvfp4_4over6(w, 4, 16),
                     quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'))
            b, a = decode_base(p), decode_alt(p)
            nd.add(n, w, b, a)
            o, k = w.shape
            sel[n] = torch.rand(-(-o // rows), k // cols, generator=g, device='cuda') < 0.3
            fake[n] = torch.where(expand(sel[n], rows, cols, o), a, b)
        bad = nd.verify_map(sel, lambda n: fake[n])
        assert not bad, bad
        nd.install(sel)
        x = (torch.randn(16, 512, 4096, generator=g, device='cuda') *
             torch.logspace(-1, 1, 16, device='cuda')[:, None, None])
        x[:, :, torch.randperm(4096, generator=g, device='cuda')[:8]] *= 20
        x = x.bfloat16()
        nd.documents = 16
        errs = {}
        for n in names:
            if sub[n].weight.shape[1] != 4096:
                continue
            y = sub[n](x)
            ref = F.linear(quant_per_document(x), fake[n])
            errs[n] = float((y.float() - ref.float()).norm() / ref.float().norm())
        nd.remove()
        results[f'{rows}x{cols}'] = dict(maps_bitwise=True, activation_checks=nd.checked, rel_err_vs_fake=errs)
        print(f'{rows}x{cols}: candidates and map decode bitwise; {nd.checked} activation calls bitwise; '
              f'max relative error vs fake F.linear {max(errs.values()):.2e}', flush=True)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
