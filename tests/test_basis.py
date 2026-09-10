"""Gate the basis ablation: the default path must be bitwise unchanged.

The ablation adds `--basis` to the scoring and evaluation stages. The whole
comparison is worthless if adding the flag perturbs the reported arm, so the
first check is that `e0m3` reproduces the original two calls exactly. The rest
records what each direction actually is, including its length -- a basis whose
direction is much shorter than another's is taking a smaller step, not
necessarily a worse one, and a null result on it would be uninformative.
"""
import sys

import torch

from quantize.basis import BASES, build_pair, direction_statistics
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights(rows=256, cols=512, seed=20261001):
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    # A few heavy channels, so alpha>1 (which discards the top codes) is a real
    # decision rather than a formality -- this is the Qwen3-4B failure mode.
    w[:, ::37] *= 8.0
    return w.to(torch.bfloat16).to(DEVICE)


def test_e0m3_is_bitwise_the_reported_direction():
    w = weights()
    base, alt = build_pair(w, 'e0m3')
    assert torch.equal(base, quant_nvfp4_4over6(w, 4, 16))
    assert torch.equal(alt, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'))
    print('OK e0m3 reproduces the reported direction bitwise')


def test_type_pure_has_a_different_baseline():
    w = weights()
    e0_base, e0_alt = build_pair(w, 'e0m3')
    tp_base, tp_alt = build_pair(w, 'type_pure')
    # Same alternative, different baseline: that is exactly what isolating the
    # type election from the alpha change means.
    assert torch.equal(tp_alt, e0_alt)
    assert not torch.equal(tp_base, e0_base), 'NVFP4 alpha=1 must differ from FourOverSix'
    assert torch.equal(tp_base, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='never'))
    print('OK type_pure shares the alternative and moves only the baseline')


def test_alpha_direction_is_nonzero_and_pure_e2m1():
    w = weights()
    base, alt = build_pair(w, 'alpha')
    assert torch.equal(base, quant_nvfp4_4over6(w, 4, 16))
    assert not torch.equal(alt, base), 'dense9 must move some scale blocks off {1, 1.5}'
    stats = direction_statistics(w, base, alt)
    assert stats['moved_tiles'] > 0
    print(f"OK alpha direction moves {stats['moved_tiles']}/{stats['tiles']} tiles")
    # dense9 = {1.0, 1.0625, ..., 1.5} is a superset of FourOverSix's {1.0, 1.5},
    # so an MSE-optimal choice over it cannot reconstruct worse. Reported rather
    # than asserted: the two functions round the ue4m3 scale independently.
    verdict = 'PASS' if stats['alternative_error_sq'] <= stats['baseline_error_sq'] else 'WARN'
    print(f"{verdict} superset check: alt_err={stats['alternative_error_sq']:.6g} "
          f"base_err={stats['baseline_error_sq']:.6g}")


def test_report_direction_lengths():
    w = weights()
    print(f'{"basis":<10} {"|d|/|W|":>10} {"moved/tiles":>14} {"base_err/|W|":>13} {"alt_err/|W|":>12}')
    for name in BASES:
        base, alt = build_pair(w, name)
        s = direction_statistics(w, base, alt)
        print(f'{name:<10} {(s["direction_sq"]/s["weight_sq"])**.5:>10.6f} '
              f'{s["moved_tiles"]:>6}/{s["tiles"]:<7} '
              f'{(s["baseline_error_sq"]/s["weight_sq"])**.5:>13.6f} '
              f'{(s["alternative_error_sq"]/s["weight_sq"])**.5:>12.6f}')


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except Exception as exc:
                failures += 1
                print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    print('BASIS GATE', 'PASS' if not failures else f'FAIL ({failures})')
    sys.exit(1 if failures else 0)
