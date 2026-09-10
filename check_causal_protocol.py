"""Does the below-BF16 result survive a causal activation convention?

The 2048-token paper-aligned protocol scales activations with a tensor-wide
factor taken over the whole window, so it can see future tokens; BF16 has no
activation quantization at all and so gets no such help. That asymmetry is a
candidate explanation for a quantized model beating BF16, and it has to be
excluded before any other account is worth stating.

The 512-token tile-count sweep uses row-wise (per-token) activation factors,
which are causal. Its FourOverSix baseline should match the audit's
`four_over_six_row` row exactly; if it does, the sweep's numbers can be compared
against the audit's matching BF16 row on the same sample.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / 'results/cap_sweep/model_335887_qwen4b/report.json'
AUDIT = ROOT / 'results/baseline_protocol_audit/REPORT.md'


def audit_row(text, model, context, method):
    pattern = rf'\|\s*{re.escape(model)}\s*\|\s*{context}\s*\|\s*{re.escape(method)}\s*\|([^|]*)\|([^|]*)\|([^|]*)\|'
    m = re.search(pattern, text)
    assert m, (model, context, method)
    return [c.strip() for c in m.groups()]


def main():
    text = AUDIT.read_text()
    bf16 = audit_row(text, 'Qwen3-4B', 512, 'bf16')
    fos = audit_row(text, 'Qwen3-4B', 512, 'four_over_six_row')
    print('audit @512, row-wise (causal) activation factors, Qwen3-4B')
    print(f'  bf16              wiki {bf16[0]}  c4-paper {bf16[1]}  c4-historical {bf16[2]}')
    print(f'  four_over_six_row wiki {fos[0]}  c4-paper {fos[1]}  c4-historical {fos[2]}')

    r = json.loads(SWEEP.read_text())
    ev = r.get('evaluation', r)
    cells = {}
    for key, val in ev.items():
        if isinstance(val, dict) and 'c4' in val and isinstance(val['c4'], dict):
            cells[key] = val['c4']['ppl']
        elif isinstance(val, dict) and 'ppl' in val:
            cells[key] = val['ppl']
    print('\ncap_sweep @512 held-out C4 perplexity by tile count')
    for k, v in cells.items():
        print(f'  {k:16s} {v:.6f}')

    base = cells.get('four_over_six')
    ref = float(fos[2])
    print(f'\nsweep baseline {base} vs audit four_over_six_row c4-historical {ref}: '
          f'{"MATCH" if base is not None and abs(base - ref) < 1e-6 else "DIFFERENT PROTOCOL"}')
    if base is not None and abs(base - ref) < 1e-6:
        bf = float(bf16[2])
        best = min(cells.values())
        print(f'causal BF16 reference on the same sample: {bf:.6f}')
        print(f'best adaptive count on the same sample:   {best:.6f}  '
              f'({best - bf:+.6f} vs BF16)')


if __name__ == '__main__':
    main()
