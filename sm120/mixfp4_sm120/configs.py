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
    KernelConfig(
        'n16k64_wA_nodisp', 'mixed', 0, None,
        dict(_WT_AS_A, MIXFP4_D_COLMAJOR=1, MIXFP4_NO_DISPATCH=1, MIXFP4_PIPE_FLAGS=0), _WT_AS_A_GEN,
        patch=False,
        description='Latency ceiling of n16k64_wA: identical tile/arrangement, format dispatch compiled '
                    'out (E2M1 only). Not a deployment kernel.'),
    KernelConfig(
        'stock_wA', 'stock', 0, None, dict(MIXFP4_D_COLMAJOR=1), {}, patch=False,
        description='Baseline. Stock CUTLASS SM120 NVFP4 mainloop, weights on A, column-major D, '
                    'same fused epilogue.'),
    KernelConfig(
        'stock_wB', 'stock', 1, None, dict(SM120_BIAS_ON_N=1), {}, patch=False,
        description='Baseline. Stock CUTLASS SM120 NVFP4 mainloop, weights on B, row-major D.'),
]}

DEFAULT = 'n16k64_wA'


def get(name):
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f'unknown kernel config {name!r}; known: {sorted(CONFIGS)}') from None
