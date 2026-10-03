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

Keys: an int is the CTA tile's token width, and the fallback rule uses only those. A string key
'<tokens>x<weights>' names an alternative build whose token width is taken by another key but whose
weight extent is narrower (kernel-opt: 'mixed_wB' '128x64', a cooperative 128 x 64 tile); only the
table selects it.
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
    # weights on B (8x64 maps): the width is the CTA tile's M, i.e. again the tokens (kernel-opt)
    'mixed_wB': {16: 'n8k64_wB_m16', 32: 'n8k64_wB_m32', 64: 'n8k64_wB_m64', 128: 'n8k64_wB', '128x64': 'n8k64_wB_n64'},
    # kernel-opt A': the 4-arm 32-row-granule builds for 256x64 (and coarser) maps
    'mixed256': {16: 'n16k64_wA_g32_n16', 32: 'n16k64_wA_g32_n32', 64: 'n16k64_wA_g32_n64', 128: 'n16k64_wA_g32'},
    # kernel-opt #4: the 64 x 64 epilogue tile at width 128, for the mixed and the stock family alike. Width 64 keeps its
    # build: there the tile costs a mainloop stage (6 -> 5) and was up to +4.9 % slower (results/kernel_opt/4)
    'mixed_e': {16: 'n16k64_wA_n16', 32: 'n16k64_wA_n32', 64: 'n16k64_wA_n64', 128: 'n16k64_wA_e64'},
    'stock_e': {16: 'stock_wA_n16', 32: 'stock_wA_n32', 64: 'stock_wA_n64', 128: 'stock_wA_e64'},
    # kernel-opt t0: the 16x64 and 256x64 families without the site-0 prmt tags (bitwise equal to 'mixed' / 'mixed256')
    'mixed_t0': {16: 'n16k64_wA_n16_t0', 32: 'n16k64_wA_n32_t0', 64: 'n16k64_wA_n64_t0', 128: 'n16k64_wA_t0'},
    'mixed256_t0': {16: 'n16k64_wA_g32_n16_t0', 32: 'n16k64_wA_g32_n32_t0', 64: 'n16k64_wA_g32_n64_t0',
                    128: 'n16k64_wA_g32_t0'},
    # kernel-opt adoption (amendment 7, 2026-10-01): the deployed 16x64 path -- t0 at every width (no site-0 prmt tags),
    # #4's 64 x 64 epilogue tile at width 128 -- and stock tuned the same way (#4). Both read the adopted table
    # (TABLE_FILE): the 4b widths and per-call scheduler rows tuned at them. The paper sets above are unchanged.
    'mixed_ko': {16: 'n16k64_wA_n16_t0', 32: 'n16k64_wA_n32_t0', 64: 'n16k64_wA_n64_t0', 128: 'n16k64_wA_e64_t0'},
    'stock_ko': {16: 'stock_wA_n16', 32: 'stock_wA_n32', 64: 'stock_wA_n64', 128: 'stock_wA_e64'},
    # kernel-opt C3k: the adopted path's no-dispatch ceiling (E2M1 only); the adopted table and mixed_ko's scheduler rows
    'nodisp_ko': {16: 'n16k64_wA_nodisp_n16_t0', 32: 'n16k64_wA_nodisp_n32_t0', 64: 'n16k64_wA_nodisp_n64_t0',
                  128: 'n16k64_wA_nodisp_e64_t0'},
    # kernel-opt 8x64 plan P2 (amendment 11): the weights-on-B family without the site-0 prmt tags (bitwise equal to
    # 'mixed_wB'), on the 'mixed_wB' rows
    'mixed_wB_t0': {16: 'n8k64_wB_m16_t0', 32: 'n8k64_wB_m32_t0', 64: 'n8k64_wB_m64_t0', 128: 'n8k64_wB_t0',
                    '128x64': 'n8k64_wB_n64_t0'},
    # kernel-opt 8x64 adoption (amendments 11-12b, 2026-10-02): the 8x64 path -- the t0 builds, deployed with #2's
    # dispatch (MIXFP4_DISPATCH_FREQ=1 in their build directory; amendment 17 adds MIXFP4_PIPE_FLAGS=1 at widths 64,
    # '128x64' and 128, and for 'mixed_ko' MIXFP4_UNIFORM_DISPATCH=1 at widths 64 and 128: docs/BUILD_AND_USE.md) --
    # and stock_wB tuned the same way (#4's epilogue tile).
    # Both read the adopted table (TABLE_FILE): the 8x64 path its 'mixed_wB' width rows (1b's with 11 re-tuned cells; no
    # scheduler rows), stock_wB_ko its scheduler rows.
    'mixed_wB_ko': {16: 'n8k64_wB_m16_t0', 32: 'n8k64_wB_m32_t0', 64: 'n8k64_wB_m64_t0', 128: 'n8k64_wB_t0',
                    '128x64': 'n8k64_wB_n64_t0'},
    'stock_wB_ko': {128: 'stock_wB_e64'},
    # kernel-opt 8x64 plan P5: the adopted 8x64 path's no-dispatch ceiling (E2M1 only), on the adopted table's 'mixed_wB'
    # rows and mixed_wB_ko's scheduler rows (it has none)
    'nodisp_wB_ko': {16: 'n8k64_wB_m16_nodisp_t0', 32: 'n8k64_wB_m32_nodisp_t0', 64: 'n8k64_wB_m64_nodisp_t0',
                     128: 'n8k64_wB_nodisp_t0', '128x64': 'n8k64_wB_n64_nodisp_t0'},
    # kernel-opt amendment 18: the 256x64 path brought to the 16x64 state -- A''s 4-arm 32-row-granule builds without the
    # site-0 tags (t0), #4's 64 x 64 epilogue tile at width 128, deployed with the uniform-branch dispatch
    # (MIXFP4_UNIFORM_DISPATCH=1 in their build directory; docs/BUILD_AND_USE.md), on its own 'mixed256' width rows and
    # 'mixed256_ko' scheduler rows of the adopted table. 'mixed256' (A') stays the paper-table set.
    'mixed256_ko': {16: 'n16k64_wA_g32_n16_t0', 32: 'n16k64_wA_g32_n32_t0', 64: 'n16k64_wA_g32_n64_t0',
                    128: 'n16k64_wA_g32_e64_t0'},
    # its no-dispatch ceiling (E2M1 only): the same CTA tiles (C3k's builds) on its widths and scheduler rows
    'nodisp256_ko': {16: 'n16k64_wA_nodisp_n16_t0', 32: 'n16k64_wA_nodisp_n32_t0', 64: 'n16k64_wA_nodisp_n64_t0',
                     128: 'n16k64_wA_nodisp_e64_t0'},
}
# A family that takes another family's tile-table rows: 'mixed256' has the CTA tile of 'mixed' at every width, and uses
# its widths so that the two differ only in the dispatch granule (kernel-opt A').
TABLE_FAMILY = {'mixed256': 'mixed', 'mixed_e': 'mixed', 'stock_e': 'stock', 'mixed_t0': 'mixed', 'mixed256_t0': 'mixed',
                'mixed_ko': 'mixed', 'stock_ko': 'stock', 'nodisp_ko': 'mixed', 'mixed_wB_t0': 'mixed_wB',
                'mixed_wB_ko': 'mixed_wB', 'stock_wB_ko': 'stock_wB', 'nodisp_wB_ko': 'mixed_wB', 'mixed256_ko': 'mixed256',
                'nodisp256_ko': 'mixed256'}
# A family that takes another family's scheduler rows (the same tiles in the same order)
SCHEDULE_FAMILY = {'nodisp_ko': 'mixed_ko', 'nodisp_wB_ko': 'mixed_wB_ko', 'nodisp256_ko': 'mixed256_ko'}
TABLE_DIR = Path(__file__).resolve().parents[1] / 'configs'
# The table file a family reads by default: '<gpu>.<suffix>.json' if listed here and present, else '<gpu>.json' (the
# paper table). The adopted kernel-opt sets read '<gpu>.ko.json' (amendment 7).
TABLE_FILE = {'mixed_ko': 'ko', 'stock_ko': 'ko', 'nodisp_ko': 'ko', 'mixed_wB_ko': 'ko', 'stock_wB_ko': 'ko',
              'nodisp_wB_ko': 'ko', 'mixed256_ko': 'ko', 'nodisp256_ko': 'ko'}
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def gpu_slug(device=0):
    return re.sub(r'[^a-z0-9]+', '_', torch.cuda.get_device_name(device).lower()).strip('_')


def bucket(t):
    for b in BUCKETS:
        if t <= b:
            return b
    return BUCKETS[-1]


def key_order(w):
    """Sort key for a family's keys: the int token widths in order, then the string alternatives."""
    return (0, w, '') if isinstance(w, int) else (1, 0, str(w))


def parse_key(w):
    return int(w) if isinstance(w, int) or str(w).isdigit() else str(w)


def fallback_width(t):
    """Rule used for shapes the table does not cover: the narrowest width holding all tokens."""
    return min(128, max(16, 1 << max(0, math.ceil(math.log2(max(t, 1))))))


class KernelSet:
    """The builds of one family, and the table choosing among them."""

    def __init__(self, family='mixed', table=None, widths=None, build_root=None):
        names = FAMILIES[family]
        if widths is not None:
            names = {w: n for w, n in names.items() if w in widths}
        self.family = family
        self.kernels = {w: Kernel.load(n, build_root=build_root) for w, n in sorted(names.items(), key=lambda i: key_order(i[0]))}
        ref = self.kernels[max(w for w in self.kernels if isinstance(w, int))]
        for k in self.kernels.values():
            if (k.weight_operand, k.type_block, k.d_colmajor, k.cfg.map_tile_rows) != \
                    (ref.weight_operand, ref.type_block, ref.d_colmajor, ref.cfg.map_tile_rows):
                raise ValueError(f'{k.cfg.name} is not interchangeable with {ref.cfg.name}')
            k.check_sf_formula([(128, 8, 256), (48, 3, 2560)])
        self.primary = ref
        self.weight_operand, self.type_block, self.d_colmajor = ref.weight_operand, ref.type_block, ref.d_colmajor
        self.cfg = ref.cfg
        self.sha256 = {k.cfg.name: k.sha256 for k in self.kernels.values()}
        self.table, self.table_source, self.schedules = {}, None, {}
        if table is None:
            path = TABLE_DIR / f'{gpu_slug()}.json'
            if family in TABLE_FILE and (TABLE_DIR / f'{gpu_slug()}.{TABLE_FILE[family]}.json').exists():
                path = TABLE_DIR / f'{gpu_slug()}.{TABLE_FILE[family]}.json'
            if path.exists():
                table = path
        if table is not None:
            data = json.loads(Path(table).read_text()) if not isinstance(table, dict) else table
            self.table = {tuple(int(v) for v in key.split('x')): {int(b): parse_key(w) for b, w in row.items()}
                          for key, row in data.get(TABLE_FAMILY.get(family, family), {}).items()}
            # kernel-opt #4: optional per-(shape, bucket) scheduler settings [raster, swizzle] under 'schedule'
            self.schedules = {tuple(int(v) for v in key.split('x')): {int(b): tuple(rs) for b, rs in row.items()}
                              for key, row in data.get('schedule', {}).get(SCHEDULE_FAMILY.get(family, family), {}).items()}
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

    def schedule(self, m, k, t):
        """kernel-opt #4: (raster, swizzle) for this call, from the table's 'schedule' rows; (0, 1) without one."""
        row = self.schedules.get((m, k))
        return (row.get(bucket(t)) if row else None) or (0, 1)

    def pick(self, m, k, t):
        w = self.width(m, k, t)
        self.stats[w] = self.stats.get(w, 0) + 1
        return self.kernels[w]

    def describe(self):
        return dict(family=self.family, kernels=self.sha256, table=self.table_source, table_shapes=len(self.table),
                    table_family=TABLE_FAMILY.get(self.family, self.family), schedule_shapes=len(self.schedules),
                    calls_by_width=dict(sorted(self.stats.items(), key=lambda i: key_order(i[0]))))
