# Local changes to the vendored mixfp4 kernel

Upstream: `https://github.com/brian030128/mixfp4` at `7b3ab34ebc4a31b396a27fa6aea3650259714cf9`
(the content of `mixfp4-main.zip`). `VENDORED.json` lists the upstream SHA-256 of every vendored
file *before* modification. Only `src/mixed_nvfp4_gemm.cu` is modified; the exact diff is
`LOCAL_CHANGES.patch`. All three changes are compile-time hooks that are inactive unless their
macro is defined, so a build without them is the upstream kernel (the upstream self-test driver
built by `sm120/build.py --selftest` is compiled from this file with the hooks off).

| macro | effect | why |
|---|---|---|
| `MIXFP4_D_COLMAJOR=1` | C/D layout tags become `ColumnMajor` | weights-on-A computes D = W Xᵀ (out × tokens); column-major D is the row-major [tokens, out] tensor a Linear returns, so no transpose/copy is needed. Mainloop and dispatch unchanged (identical OMMA census). |
| `MIXFP4_EPILOGUE_FUSION_T=<template>` | the epilogue's fusion type becomes `<template><ThreadBlockShape>` instead of `LinearCombination` | `sm120/csrc/mixfp4_sm120.cu` supplies an EVT computing `bf16((s_m[m]·s_n[n])·acc + bias)`, which folds the per-token activation scale, the weight global scale and the bias into the GEMM's single output rounding. |
| `MIXFP4_NO_SELFTEST=1` | omits `run()` / `main()` | the self-test driver builds LinearCombination epilogue arguments, which do not exist for the EVT. |

The mainloop (`src/collective/*`), the MMA atom, the blob/PTX generators and the SASS patcher are
unmodified. `sm120/build.py` runs `scripts/gen_mixed_mma_blob.py` from a copy in the build
directory, so the generated header never overwrites `src/collective/mixed_mma_blob_generated.hpp`
(which is kept as the upstream committed default and is shadowed by include order).
