"""Kernel build configurations: the single source of truth for sm120/build.py and the runtime.

Each configuration is one shared library. The mixed ones follow the recipes in
sm120/kernel/docs/mixed_nvfp4_report.md section 6 ("Using it"); the table below records which
operand carries the weights, the format granule the kernel honours, and the SASS census the
patcher must report. A build whose census differs from `expected_census` is rejected.

Conventions:
  * weight_operand 0: weights are GEMM operand A, D = W X^T is [out, tokens]; with a column-major D
    that is the row-major [tokens, out] tensor nn.Linear returns.
  * weight_operand 1: weights are operand B, D = X W^T is [tokens, out] row-major.
  * type_block is (output channels, K) of one format granule, i.e. the selector's N x K block.
"""
import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KernelConfig:
    name: str
    kind: str                           # 'mixed' | 'stock'
    weight_operand: int                 # 0 = A, 1 = B
    type_block: tuple | None            # (rows, k) of the weight format granule; None = E2M1 only
    defines: dict = field(default_factory=dict)
    blob_gen: dict = field(default_factory=dict)   # env for kernel/scripts/gen_mixed_mma_blob.py
    patch: bool = True
    # Per-site OMMA counts the patcher must report ({site: count}); None = record only.
    expected_census: dict | None = None
    description: str = ''
    # Smallest contiguous row count that is a union of this kernel's weight granules, when the granule's rows are not
    # contiguous (kernel-opt A': a warp's two m-atoms are 64 rows apart, so a 32-row granule tiles 128-row panels).
    # A map is compatible only if its tile rows are a multiple of it. None = type_block[0] (contiguous granules).
    map_tile_rows: int | None = None

    @property
    def d_colmajor(self):
        return bool(self.defines.get('MIXFP4_D_COLMAJOR', 0))

    @property
    def mixed(self):
        return self.kind == 'mixed'


_WT_AS_A = dict(MIXFP4_BLOB=1, MIXFP4_JOINT_KA=1, MIXFP4_B_ALL_E2M1=1,
                MIXFP4_A_ATOMS_PER_GRANULE=1, MIXFP4_B_ATOMS_PER_GRANULE=8)
_WT_AS_A_GEN = dict(MMA_M=2, MMA_N=8, A_ATOMS=1, B_ATOMS=8)

_WT_AS_A_8X1 = dict(MIXFP4_BLOB=1, MIXFP4_JOINT_KA=1, MIXFP4_B_ALL_E2M1=1, MIXFP4_ATOM_M=8,
                    MIXFP4_EPI_N=32, MIXFP4_A_ATOMS_PER_GRANULE=1, MIXFP4_B_ATOMS_PER_GRANULE=16)
_WT_AS_A_8X1_GEN = dict(MMA_M=1, MMA_N=16, A_ATOMS=1, B_ATOMS=16)

_B8X64 = dict(MIXFP4_BLOB=1, MIXFP4_JOINT_KB=1, MIXFP4_A_ALL_E2M1=1, MIXFP4_ATOM_M=1,
              MIXFP4_PERM_N=128, MIXFP4_A_ATOMS_PER_GRANULE=8, MIXFP4_B_ATOMS_PER_GRANULE=1)
_B8X64_GEN = dict(MMA_M=8, MMA_N=2, A_ATOMS=8, B_ATOMS=1)

CONFIGS = {c.name: c for c in [
    KernelConfig(
        'n16k64_wA', 'mixed', 0, (16, 64), dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1), _WT_AS_A_GEN,
        expected_census={0: 512, 1: 512},
        description='Primary. Weights on A, 16 rows x 64 K granule, stock 4x2 warp arrangement '
                    '(16 joint arms), activations pinned E2M1, column-major D (no transpose).'),
    KernelConfig(
        'n16k64_wA_8x1', 'mixed', 0, (16, 64), dict(_WT_AS_A_8X1, MIXFP4_D_COLMAJOR=1), _WT_AS_A_8X1_GEN,
        expected_census={0: 128, 1: 128},
        description='Same granule via the 8x1 warp arrangement (4 joint arms), epilogue N=32. '
                    'The report measured it best on the RTX 5090 and worse on the RTX PRO 6000.'),
    KernelConfig(
        'n16k64_wA_sk', 'mixed', 0, (16, 64), dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, SM120_STREAMK=1), _WT_AS_A_GEN,
        expected_census={0: 512, 1: 512},
        description='n16k64_wA with the Stream-K tile scheduler (deterministic reduction): splits K '
                    'across CTAs when out/128 x tokens/128 tiles do not fill the GPU (decode, small batches).'),
    KernelConfig(
        'stock_wA_sk', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1, SM120_STREAMK=1), {}, patch=False,
        description='stock_wA with the Stream-K tile scheduler (baseline for n16k64_wA_sk).'),
    # Narrow token tiles for small T (decode): with weights on A the token count is the GEMM's N, and
    # a 128-wide CTA tile computes 128 token columns per k-tile however few are real, which makes
    # decode MMA-bound on padding. bench/splitk.py and bench/kernel.py measure the crossover.
    KernelConfig(
        'n16k64_wA_n64', 'mixed', 0, (16, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=64, MIXFP4_B_ATOMS_PER_GRANULE=4),
        dict(_WT_AS_A_GEN, MMA_N=4, B_ATOMS=4),
        expected_census={0: 256, 1: 256},
        description='n16k64_wA with a 128 x 64 CTA tile (warp 32 x 32): small-T kernel.'),
    KernelConfig(
        'n16k64_wA_n32', 'mixed', 0, (16, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=32, MIXFP4_B_ATOMS_PER_GRANULE=2),
        dict(_WT_AS_A_GEN, MMA_N=2, B_ATOMS=2),
        expected_census={0: 128, 1: 128},
        description='n16k64_wA with a 128 x 32 CTA tile (warp 32 x 16): small-T kernel.'),
    KernelConfig(
        'n16k64_wA_n16', 'mixed', 0, (16, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=16, MIXFP4_B_ATOMS_PER_GRANULE=1, MIXFP4_LDSM_B=2),
        dict(_WT_AS_A_GEN, MMA_N=1, B_ATOMS=1),
        expected_census={0: 64, 1: 64},
        description='n16k64_wA with a 128 x 16 CTA tile (warp 32 x 8): decode kernel.'),
    KernelConfig('stock_wA_n64', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=64), {}, patch=False,
                 description='stock_wA with a 128 x 64 CTA tile.'),
    KernelConfig('stock_wA_n32', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=32), {}, patch=False,
                 description='stock_wA with a 128 x 32 CTA tile.'),
    KernelConfig('stock_wA_n16', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=16), {}, patch=False,
                 description='stock_wA with a 128 x 16 CTA tile.'),
    KernelConfig(
        'n8k64_wB', 'mixed', 1, (8, 64), dict(_B8X64, SM120_BIAS_ON_N=1), _B8X64_GEN,
        expected_census={0: 512, 2: 512},
        description='Comparison. Weights on B, 8 columns x 64 K granule, 1x8 arrangement, '
                    'activations pinned E2M1, row-major D.'),
    # Narrow token tiles with the weights on B (kernel-opt). M is the token count there, so these are
    # n8k64_wB's counterparts of the n16k64_wA_n* builds for small T (decode, short prefill):
    # - MIXFP4_TILE_M narrows the CTA tile; the collective loads the whole 128-row scale-factor block
    #   of A and each CTA reads its sub-tile (the mirror of the narrow-N SFB path).
    # - CUTLASS's cooperative kernel requires a tile M >= 128, so these use the ping-pong schedule
    #   (MIXFP4_PINGPONG): 4 MMA warps per tile. To keep n8k64_wB's 16 dispatch arms (2 n-atoms = 16
    #   weight columns per warp) the warps are 1x4 and the tile is 64 weight columns wide.
    # - A stays pinned E2M1 as one granule per warp (A_ATOMS = MMA_M); the OMMA census scales with the
    #   warp's m-atoms. Bitwise equal to n8k64_wB (the per-output accumulation order is unchanged).
    # - Ping-pong has one 4-warp group on the tensor cores at a time, so these lose at large T: the
    #   'mixed_wB' KernelSet uses them only where the tile table says so.
    KernelConfig(
        'n8k64_wB_m64', 'mixed', 1, (8, 64),
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_PINGPONG=1, MIXFP4_TILE_M=64, MIXFP4_TILE_N=64, MIXFP4_ATOM_N=4,
             MIXFP4_PERM_N=64, MIXFP4_A_ATOMS_PER_GRANULE=4),
        dict(_B8X64_GEN, MMA_M=4, A_ATOMS=4),
        expected_census={0: 256, 2: 256},
        description='n8k64_wB with a 64 x 64 CTA tile (ping-pong, 1x4 warps): small-T kernel.'),
    KernelConfig(
        'n8k64_wB_m32', 'mixed', 1, (8, 64),
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_PINGPONG=1, MIXFP4_TILE_M=32, MIXFP4_TILE_N=64, MIXFP4_ATOM_N=4,
             MIXFP4_PERM_N=64, MIXFP4_A_ATOMS_PER_GRANULE=2),
        dict(_B8X64_GEN, MMA_M=2, A_ATOMS=2),
        expected_census={0: 128, 2: 128},
        description='n8k64_wB with a 32 x 64 CTA tile (ping-pong, 1x4 warps): small-T kernel.'),
    KernelConfig(
        'n8k64_wB_m16', 'mixed', 1, (8, 64),
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_PINGPONG=1, MIXFP4_TILE_M=16, MIXFP4_TILE_N=64, MIXFP4_ATOM_N=4,
             MIXFP4_PERM_N=64, MIXFP4_A_ATOMS_PER_GRANULE=1),
        dict(_B8X64_GEN, MMA_M=1, A_ATOMS=1),
        expected_census={0: 64, 2: 64},
        description='n8k64_wB with a 16 x 64 CTA tile (ping-pong, 1x4 warps): small-T kernel.'),
    # Narrow WEIGHT tiles with the weights on B (kernel-opt 1b): a cooperative 128 (tokens) x 64 (weights)
    # tile, the weights-on-B mirror of stock_wA_n64 / n16k64_wA_n64. At mid token counts (T ~ 128-1024)
    # the 128 x 128 tile leaves most SMs idle (e.g. T = 256, out = 4096: 64 CTAs); halving the weight
    # extent doubles the CTAs while keeping all 8 MMA warps of the cooperative kernel (the ping-pong
    # narrow-M builds above have 4). 2x4 warps: each warp owns 64 tokens x 16 weight columns (4 m-atoms,
    # 2 n-atoms), the same per-warp shape and 16-arm dispatch as n8k64_wB_m64, so the same per-output
    # MMA sequence (bitwise equal to n8k64_wB).
    KernelConfig(
        'n8k64_wB_n64', 'mixed', 1, (8, 64),
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_TILE_N=64, MIXFP4_ATOM_M=2, MIXFP4_ATOM_N=4, MIXFP4_PERM_N=64,
             MIXFP4_A_ATOMS_PER_GRANULE=4),
        dict(_B8X64_GEN, MMA_M=4, A_ATOMS=4),
        expected_census={0: 256, 2: 256},
        description='n8k64_wB with a 128 x 64 CTA tile (cooperative, 2x4 warps): mid-T kernel.'),
    # E0M3 investigation, test B (diagnostic only, never deployed): the same kernels with the dispatch tree's arm
    # positions permuted by MIXFP4_ARM_XOR = arms - 1, which puts the all-E0M3 arm on the all-fall-through path.
    KernelConfig(
        'n16k64_wA_xor', 'mixed', 0, (16, 64), dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_ARM_XOR=15), _WT_AS_A_GEN,
        expected_census={0: 512, 1: 512},
        description='Diagnostic: n16k64_wA with the dispatch arms permuted (all-E0M3 on the fall-through path).'),
    KernelConfig(
        'n8k64_wB_xor', 'mixed', 1, (8, 64), dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_ARM_XOR=15), _B8X64_GEN,
        expected_census={0: 512, 2: 512},
        description='Diagnostic: n8k64_wB with the dispatch arms permuted (all-E0M3 on the fall-through path).'),
    # kernel-opt A': 256x64 maps on a 4-arm kernel. A 256x64 map gives both of a 4x2 warp's m-atoms (weight rows
    # 16 w_m .. +15 and 64 + 16 w_m .. +15 of a 128-row panel) the same format, so the warp's two m-atoms can be one
    # granule (MIXFP4_A_ATOMS_PER_GRANULE=2): 1 bit per k_block, 2 bits per k_tile, 4 joint arms instead of 16 (the
    # 8x1 build's count). The granule's 32 rows are not contiguous, so a map must tile whole 128-row panels
    # (map_tile_rows): 256x64 and coarser. Every output keeps its MMA sequence (bitwise equal to n16k64_wA on such maps).
    KernelConfig(
        'n16k64_wA_g32', 'mixed', 0, (32, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_A_ATOMS_PER_GRANULE=2), dict(_WT_AS_A_GEN, A_ATOMS=2),
        expected_census={0: 128, 1: 128}, map_tile_rows=128,
        description="A': n16k64_wA with a 2-m-atom (32-row, 128-row-panel) granule: 4 joint arms, for 256x64 maps."),
    KernelConfig(
        'n16k64_wA_g32_n64', 'mixed', 0, (32, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_A_ATOMS_PER_GRANULE=2, MIXFP4_TILE_N=64, MIXFP4_B_ATOMS_PER_GRANULE=4),
        dict(_WT_AS_A_GEN, A_ATOMS=2, MMA_N=4, B_ATOMS=4),
        expected_census={0: 64, 1: 64}, map_tile_rows=128,
        description="A': n16k64_wA_g32 with a 128 x 64 CTA tile."),
    KernelConfig(
        'n16k64_wA_g32_n32', 'mixed', 0, (32, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_A_ATOMS_PER_GRANULE=2, MIXFP4_TILE_N=32, MIXFP4_B_ATOMS_PER_GRANULE=2),
        dict(_WT_AS_A_GEN, A_ATOMS=2, MMA_N=2, B_ATOMS=2),
        expected_census={0: 32, 1: 32}, map_tile_rows=128,
        description="A': n16k64_wA_g32 with a 128 x 32 CTA tile."),
    KernelConfig(
        'n16k64_wA_g32_n16', 'mixed', 0, (32, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_A_ATOMS_PER_GRANULE=2, MIXFP4_TILE_N=16, MIXFP4_B_ATOMS_PER_GRANULE=1,
             MIXFP4_LDSM_B=2),
        dict(_WT_AS_A_GEN, A_ATOMS=2, MMA_N=1, B_ATOMS=1),
        expected_census={0: 16, 1: 16}, map_tile_rows=128,
        description="A': n16k64_wA_g32 with a 128 x 16 CTA tile."),
    # kernel-opt #4: the 64 x 64 epilogue tile (auto is 64 x 32 for these tiles), mixed and stock alike, at the widths
    # whose CTA tile holds it (128, 64); the narrower widths keep their builds. Outputs are unchanged.
    KernelConfig(
        'n16k64_wA_e64', 'mixed', 0, (16, 64), dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64),
        _WT_AS_A_GEN, expected_census={0: 512, 1: 512},
        description='kernel-opt #4: n16k64_wA with a 64 x 64 epilogue tile.'),
    KernelConfig(
        'n16k64_wA_n64_e64', 'mixed', 0, (16, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=64, MIXFP4_B_ATOMS_PER_GRANULE=4, MIXFP4_EPI_TILE_M=64,
             MIXFP4_EPI_TILE_N=64),
        dict(_WT_AS_A_GEN, MMA_N=4, B_ATOMS=4), expected_census={0: 256, 1: 256},
        description='kernel-opt #4: n16k64_wA_n64 with a 64 x 64 epilogue tile.'),
    # kernel-opt #4 on the 256x64 path (amendment 8): A''s 4-arm 32-row-granule build with the 64 x 64 epilogue tile.
    KernelConfig(
        'n16k64_wA_g32_e64', 'mixed', 0, (32, 64),
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_A_ATOMS_PER_GRANULE=2, MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64),
        dict(_WT_AS_A_GEN, A_ATOMS=2), expected_census={0: 128, 1: 128}, map_tile_rows=128,
        description="kernel-opt #4: n16k64_wA_g32 with a 64 x 64 epilogue tile."),
    KernelConfig('stock_wA_e64', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1, MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64), {},
                 patch=False, description='kernel-opt #4: stock_wA with a 64 x 64 epilogue tile.'),
    KernelConfig('stock_wA_n64_e64', 'stock', 0, None,
                 dict(MIXFP4_D_COLMAJOR=1, MIXFP4_TILE_N=64, MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64), {}, patch=False,
                 description='kernel-opt #4: stock_wA_n64 with a 64 x 64 epilogue tile.'),
    KernelConfig(
        'n16k64_wA_nodisp', 'mixed', 0, None,
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), _WT_AS_A_GEN,
        patch=False,
        description='Latency ceiling of n16k64_wA: identical tile/arrangement, format dispatch compiled '
                    'out (E2M1 only). Not a deployment kernel.'),
    # kernel-opt t0 (nodisp vs stock): n16k64_wA_nodisp without the blob's site-0 prmt tags (TAG0=0) -- the
    # no-dispatch ceiling the t0 builds are measured against.
    KernelConfig(
        'n16k64_wA_nodisp_t0', 'mixed', 0, None,
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), dict(_WT_AS_A_GEN, TAG0=0),
        patch=False, description='Diagnostic: n16k64_wA_nodisp without the site-0 prmt tags. Not a deployment kernel.'),
    # kernel-opt 8x64 baseline (amendment 10): n8k64_wB's no-dispatch ceiling -- its tile and 1x8 arrangement with the
    # format dispatch compiled out (E2M1 only) -- with and without the blob's site-0 prmt tags (TAG0=0).
    KernelConfig(
        'n8k64_wB_nodisp', 'mixed', 1, None,
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), _B8X64_GEN, patch=False,
        description='Latency ceiling of n8k64_wB: identical tile/arrangement (1x8), format dispatch compiled out '
                    '(E2M1 only). Not a deployment kernel.'),
    KernelConfig(
        'n8k64_wB_nodisp_t0', 'mixed', 1, None,
        dict(_B8X64, SM120_BIAS_ON_N=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), dict(_B8X64_GEN, TAG0=0),
        patch=False, description='Diagnostic: n8k64_wB_nodisp without the site-0 prmt tags. Not a deployment kernel.'),
    KernelConfig(
        'stock_wA', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1), {}, patch=False,
        description='Baseline. Stock CUTLASS SM120 NVFP4 mainloop, weights on A, column-major D, '
                    'same fused epilogue.'),
    KernelConfig(
        'stock_wB', 'stock', 1, None, dict(SM120_BIAS_ON_N=1), {}, patch=False,
        description='Baseline. Stock CUTLASS SM120 NVFP4 mainloop, weights on B, row-major D.'),
]}

# kernel-opt t0 (nodisp vs stock): the 16x64 and 256x64 (A') kernels with the blob's site-0 prmt tags dropped
# (TAG0=0). The tags -- one identity PRMT per (m-atom, k_block) and k_tile in every arm -- were the whole gap between
# n16k64_wA_nodisp and stock_wA (results/kernel_opt/t0). Site 0 is never patched, so only the E0M3 sites keep a tag; the
# patcher reads the sites from reaching definitions (--untagged-site0). Same MMAs in the same order: bitwise equal to
# their bases.
for _base in ('n16k64_wA', 'n16k64_wA_n64', 'n16k64_wA_n32', 'n16k64_wA_n16',
              'n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16',
              'n16k64_wA_e64',    # amendment 7: the adopted 16x64 path's width-128 build (t0 + #4's epilogue tile)
              'n16k64_wA_g32_e64',  # amendment 18: the 256x64 path's width-128 build (A' + #4's tile + t0)
              # amendment 11 (the 8x64 plan's P2): the weights-on-B family 'mixed_wB', whose blob tags every MMA with
              # 8 m-atoms per warp (32 identity PRMTs per k_tile and MMA warp, against 8 for 16x64)
              'n8k64_wB', 'n8k64_wB_m64', 'n8k64_wB_m32', 'n8k64_wB_m16', 'n8k64_wB_n64'):
    _c = CONFIGS[_base]
    CONFIGS[_base + '_t0'] = dataclasses.replace(
        _c, name=_base + '_t0', blob_gen=dict(_c.blob_gen, TAG0=0),
        description=f'kernel-opt t0: {_base} without the site-0 prmt tags (TAG0=0).')

# kernel-opt C3k (amendment 8): the no-dispatch ceiling of the adopted 16x64 path at every width ('nodisp_ko'): its
# tiles and epilogue with the format dispatch compiled out (E2M1 only, no site-0 tags). Not deployment kernels.
_NODISP = dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0)
for _name, _defs, _gen in (
        ('n16k64_wA_nodisp_n16_t0', dict(MIXFP4_TILE_N=16, MIXFP4_B_ATOMS_PER_GRANULE=1, MIXFP4_LDSM_B=2),
         dict(MMA_N=1, B_ATOMS=1)),
        ('n16k64_wA_nodisp_n32_t0', dict(MIXFP4_TILE_N=32, MIXFP4_B_ATOMS_PER_GRANULE=2), dict(MMA_N=2, B_ATOMS=2)),
        ('n16k64_wA_nodisp_n64_t0', dict(MIXFP4_TILE_N=64, MIXFP4_B_ATOMS_PER_GRANULE=4), dict(MMA_N=4, B_ATOMS=4)),
        ('n16k64_wA_nodisp_e64_t0', dict(MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64), {})):
    CONFIGS[_name] = KernelConfig(
        _name, 'mixed', 0, None, dict(_NODISP, **_defs), dict(_WT_AS_A_GEN, TAG0=0, **_gen), patch=False,
        description="kernel-opt C3k: the adopted 16x64 path's tile with the format dispatch compiled out (E2M1 only). "
                    'Not a deployment kernel.')

# kernel-opt 8x64 plan P3 (amendment 12, with P4): stock_wB with #4's 64 x 64 epilogue tile, so the same-placement
# reference is tuned as stock_ko is; it keeps its 4 mainloop stages. The weights-on-B mixed builds get no counterpart:
# the 1x8 arrangement already names a 64 x 64 epilogue tile at width 128 (MIXFP4_EPI_M/N), and the '128x64' build would
# lose a mainloop stage with it (6 -> 5).
CONFIGS['stock_wB_e64'] = dataclasses.replace(
    CONFIGS['stock_wB'], name='stock_wB_e64',
    defines=dict(CONFIGS['stock_wB'].defines, MIXFP4_EPI_TILE_M=64, MIXFP4_EPI_TILE_N=64),
    description='kernel-opt P3: stock_wB with a 64 x 64 epilogue tile.')

# kernel-opt 8x64 plan P5 (decision a): the adopted 8x64 path's no-dispatch ceiling at every width ('nodisp_wB_ko'), the
# same-placement reference: each width's tile and 1x8 / 1x4 arrangement with the format dispatch compiled out (E2M1
# only, no site-0 tags), as n8k64_wB_nodisp_t0 is for width 128. Not deployment kernels.
for _base in ('n8k64_wB_m16', 'n8k64_wB_m32', 'n8k64_wB_m64', 'n8k64_wB_n64'):
    _c = CONFIGS[_base]
    CONFIGS[_base + '_nodisp_t0'] = dataclasses.replace(
        _c, name=_base + '_nodisp_t0', type_block=None, expected_census=None, patch=False,
        defines=dict(_c.defines, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), blob_gen=dict(_c.blob_gen, TAG0=0),
        description=f"kernel-opt P5: {_base}'s tile with the format dispatch compiled out (E2M1 only). "
                    'Not a deployment kernel.')

DEFAULT = 'n16k64_wA'


def get(name):
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f'unknown kernel config {name!r}; known: {sorted(CONFIGS)}') from None
