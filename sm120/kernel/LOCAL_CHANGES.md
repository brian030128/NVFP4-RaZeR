# Local changes to the vendored mixfp4 kernel

Upstream: `https://github.com/brian030128/mixfp4` at `7b3ab34ebc4a31b396a27fa6aea3650259714cf9`
(the content of `mixfp4-main.zip`). `VENDORED.json` lists the upstream SHA-256 of every vendored
file *before* modification. Two files are modified, `src/mixed_nvfp4_gemm.cu` and (since the
`kernel-opt` branch) `src/collective/sm120_blockscaled_mma_tma_mixed.hpp`; the exact diff of both is
`LOCAL_CHANGES.patch`. Every change is a compile-time hook that is inactive unless its macro is
defined (or, for the collective, unless the CTA tile's M is below 128, which no upstream build
uses), so a build without them is the upstream kernel (the upstream self-test driver built by
`sm120/build.py --selftest` is compiled from this file with the hooks off). Every existing
configuration rebuilt from the kernel-opt sources has the same patched and unpatched SASS as before
(`results/kernel_opt/`, gate G1).

| macro | effect | why |
|---|---|---|
| `MIXFP4_D_COLMAJOR=1` | C/D layout tags become `ColumnMajor` | weights-on-A computes D = W Xᵀ (out × tokens); column-major D is the row-major [tokens, out] tensor a Linear returns, so no transpose/copy is needed. Mainloop and dispatch unchanged (identical OMMA census). |
| `MIXFP4_EPILOGUE_FUSION_T=<template>` | the epilogue's fusion type becomes `<template><ThreadBlockShape>` instead of `LinearCombination` | `sm120/csrc/mixfp4_sm120.cu` supplies an EVT computing `bf16((s_m[m]·s_n[n])·acc + bias)`, which folds the per-token activation scale, the weight global scale and the bias into the GEMM's single output rounding. |
| `MIXFP4_NO_SELFTEST=1` | omits `run()` / `main()` | the self-test driver builds LinearCombination epilogue arguments, which do not exist for the EVT. |
| `MIXFP4_TILE_SCHEDULER=<type>` | the kernel's tile scheduler (upstream: `void`, the default persistent scheduler) | `cutlass::gemm::StreamKScheduler` for the `*_sk` builds, which split K across CTAs when the output has fewer tiles than SMs (decode). The mainloop already consumes arbitrary k-tile ranges; identical OMMA census; the self-test gate passes. |
| `MIXFP4_TILE_M=<64\|32\|16>` (kernel-opt) | the CTA tile's M (upstream: 128) | weights on B (`n8k64_wB_m64/m32/m16`): M is the token count, so this is the weights-on-B counterpart of `MIXFP4_TILE_N`'s narrow token tiles for small T. For M < 128 the SFA shared-memory atom is taken from a 128-row builder (the builder's own M/128 atom is empty), the stage count is recomputed with it by the builder's stage function, and the epilogue tile's M is capped at the CTA tile's. Each output is still computed by the same MMA sequence over the same k-tiles in the same order, so the result is bitwise the 128-row build's (gates G4/G5). |
| `MIXFP4_PINGPONG=1` (kernel-opt) | CUTLASS's ping-pong kernel schedule (4 MMA warps, two consumer warp groups on alternating tiles) instead of the builder's default cooperative one (8 MMA warps) | the cooperative kernel static_asserts a CTA tile M >= 128, so the narrow-M builds need ping-pong, with a 4-warp arrangement (`MIXFP4_ATOM_M * MIXFP4_ATOM_N == 4`; the narrow builds use 1x4 and a 64-wide N tile, which keeps n8k64_wB's per-warp dispatch of 2 n-atoms x 2 k-blocks). |
| `MIXFP4_ARM_XOR=<mask>` (kernel-opt, diagnostic only) | in the collective's `dispatch_pattern`: the arm tree is searched on `pattern ^ mask` and the leaf at position L runs pattern L ^ mask's arm | the E0M3 investigation's test B (`results/kernel_opt/e0m3/`): with mask 15 the all-E0M3 arm gets the all-fall-through path that the all-E2M1 arm has by default, to see whether E0M3's cost follows the code position. Every arm computes exactly as before. Unset (0), the dispatch is the upstream one (identical SASS). Used only by the never-deployed `*_xor` builds. |
| collective, narrow M (kernel-opt) | in `sm120_blockscaled_mma_tma_mixed.hpp`, only when the CTA tile's M < 128: the SFA smem tile and TMA box cover the whole 128-row scale-factor block (`TileShapeSFA`), the producer loads the block that holds the CTA's rows (`broadcast_m`), and the consumer reads its M sub-tile (`m % (128/M)`) | the mirror of the collective's existing narrow-N handling of SFB (`TileShapeSFB`, `broadcast_n`): the scale-factor layout's indivisible unit is a 128-row block. For M >= 128 every type is the upstream one (identical SASS, gate G1). |

The mainloop collective (`src/collective/*`) is modified only by the narrow-M path and the diagnostic
arm permutation above; the MMA atom, the blob/PTX generators and the SASS patcher are unmodified. `sm120/build.py` runs `scripts/gen_mixed_mma_blob.py` from a copy in the build
directory, so the generated header never overwrites `src/collective/mixed_mma_blob_generated.hpp`
(which is kept as the upstream committed default and is shadowed by include order).
