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
