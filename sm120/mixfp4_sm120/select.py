"""Shape-dependent kernel selection: one CTA-tile width per (out, in, token-count bucket).

With the weights on operand A the token count T is the GEMM's N. A 128-wide CTA tile computes 128
token columns per k-tile however few are real, which makes small-T GEMMs MMA-bound on padding
(RTX 5090: ~0.37 us per 128x128x128 k-tile per CTA regardless of T). Builds with 64/32/16-wide
tiles remove that waste; which width is fastest depends on (out, in, T) through the number of CTA
tiles versus SMs, so the choice is looked up in a per-GPU table measured on the real model shapes
(bench/tune_tiles.py -> sm120/configs/<gpu>.json).

All widths of one family execute the same MMA instruction sequence per output element (verified
bitwise, tests/test_select.py), so the selection never changes a result -- the output of a token
does not depend on how many other tokens are in the batch.
"""
import json
import math
import re
from pathlib import Path

import torch

from .lib import Kernel

FAMILIES = {
    'mixed': {16: 'n16k64_wA_n16', 32: 'n16k64_wA_n32', 64: 'n16k64_wA_n64', 128: 'n16k64_wA'},
    'stock': {16: 'stock_wA_n16', 32: 'stock_wA_n32', 64: 'stock_wA_n64', 128: 'stock_wA'},
}
TABLE_DIR = Path(__file__).resolve().parents[1] / 'configs'
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def gpu_slug(device=0):
    return re.sub(r'[^a-z0-9]+', '_', torch.cuda.get_device_name(device).lower()).strip('_')


def bucket(t):
    for b in BUCKETS:
        if t <= b:
            return b
    return BUCKETS[-1]


def fallback_width(t):
    """Rule used for shapes the table does not cover: the narrowest width holding all tokens."""
    return min(128, max(16, 1 << max(0, math.ceil(math.log2(max(t, 1))))))


class KernelSet:
    """The builds of one family, and the table choosing among them."""

    def __init__(self, family='mixed', table=None, widths=None):
        names = FAMILIES[family]
        if widths is not None:
            names = {w: n for w, n in names.items() if w in widths}
        self.family = family
        self.kernels = {w: Kernel.load(n) for w, n in sorted(names.items())}
        ref = self.kernels[max(self.kernels)]
        for k in self.kernels.values():
            if (k.weight_operand, k.type_block, k.d_colmajor) != (ref.weight_operand, ref.type_block, ref.d_colmajor):
                raise ValueError(f'{k.cfg.name} is not interchangeable with {ref.cfg.name}')
            k.check_sf_formula([(128, 8, 256), (48, 3, 2560)])
        self.primary = ref
        self.weight_operand, self.type_block, self.d_colmajor = ref.weight_operand, ref.type_block, ref.d_colmajor
        self.cfg = ref.cfg
        self.sha256 = {k.cfg.name: k.sha256 for k in self.kernels.values()}
        self.table, self.table_source = {}, None
        if table is None:
            path = TABLE_DIR / f'{gpu_slug()}.json'
            if path.exists():
                table = path
        if table is not None:
            data = json.loads(Path(table).read_text()) if not isinstance(table, dict) else table
            self.table = {tuple(int(v) for v in key.split('x')): {int(b): int(w) for b, w in row.items()}
                          for key, row in data.get(family, {}).items()}
            self.table_source = str(table) if not isinstance(table, dict) else 'dict'
        self.stats = {}

    def width(self, m, k, t):
        row = self.table.get((m, k))
        w = row.get(bucket(t)) if row else None
        if w is None or w not in self.kernels:
            w = fallback_width(t)
            while w not in self.kernels:
                w *= 2
        return w

    def pick(self, m, k, t):
        w = self.width(m, k, t)
        self.stats[w] = self.stats.get(w, 0) + 1
        return self.kernels[w]

    def describe(self):
        return dict(family=self.family, kernels=self.sha256, table=self.table_source, table_shapes=len(self.table),
                    calls_by_width=dict(sorted(self.stats.items())))
