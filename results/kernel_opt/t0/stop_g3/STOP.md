# Kernel-opt t0 (amendment 6): stopped at gate G3 — awaiting a decision

The registered chain `run_T.sh` (registration 75dade0) began at 02:40:12 UTC on 2026-10-01 and stopped at 03:00:27 in
its first step, G3, with no deviation before that point (`commands.log`). No other gate and no measurement ran, so
there are no GPU results.

## What failed

`build.py --selftest` failed for all 8 t0 configurations at the same step: patching the upstream self-test driver
executable (`G3_selftest.log`). The error was `BRX at 0x...; --untagged-site0 supports direct branches only`.

- **Where the BRX comes from.** The driver's kernel (`src/mixed_nvfp4_gemm.cu` with the hooks off: CUTLASS's
  `LinearCombination` epilogue, not our EVT) compiles CUTLASS's warp-role switch at kernel entry into a jump table:
  ```
  VIMNMX.U32 R3, R8, 0x3, PT
  IMAD.SHL R3, R3, 0x4
  LDC R12, c[0x2][R3]
  BRX R12 -0x1050
  ```
  - The table has 4 entries in the kernel's `.nv.constant2` section, e.g. `[0x16b0, 0x1050, 0x25e0, 0x1050]`: the role
    entry points.
  - The first OMMA is at 0x4fb0.
- **Why our libraries were not affected.** In the library kernels ptxas emits the same switch as compare-and-branch.
  They contain no indirect branch, and were built and patched with the expected census (G2 pre-check: pass).
- **Why the patcher stopped.** The `--untagged-site0` reaching-definitions analysis refuses indirect branches by
  design, because it cannot know their targets.
- **It is not a numerical failure.** No output was compared or found wrong.

## State left behind

- G3 rebuilds each configuration before its self-test, so the 8 t0 directories in `build_T` were rebuilt from 75dade0.
- Their patched and unpatched SASS equal the registered hashes (checked).
- `build.py` writes the manifest only after the self-test, so these 8 directories have no `manifest.json`. They are
  not in the registered state.
- `build_Tfreq` and the other 29 `build_T` directories are untouched.

## Proposed fix (needs approval; would be amendment 6b, re-registered)

Resolve each `BRX`'s targets exactly from the kernel's jump table:
- read the kernel's `.nv.constant2` section from the cubin (`cuobjdump -xelf`);
- target = table word + the BRX's next PC + its immediate;
- add those edges to the control-flow graph;
- any indirect branch whose table cannot be resolved, or whose entries are not instruction addresses, stays an error.

Offline check (scratch copy, CPU only: `diagnostic_patcher_jumptable.py`):
- all 8 self-test kernels then classify with no ambiguity;
- their per-site counts are each library's census (e.g. n16k64_wA_t0: 512 / 512).

The 74 existing builds contain no BRX, so their validation is unchanged.

After approval:
1. Implement and re-validate (74 builds plus the 8 self-test kernels).
2. Rebuild the 8 t0 directories (CPU).
3. Append amendment 6b and re-register.
4. Re-run `run_T.sh` from the start.

An alternative is to drop the self-test gate for the t0 builds. The library-level tests (test_gemm / test_g32 against
the reference with negative controls), G4 and G5 would still apply. It is weaker and not recommended.
