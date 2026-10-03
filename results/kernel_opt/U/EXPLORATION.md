# Kernel-opt U: the uniform-branch dispatch — exploration (disclosed, before registration)

2026-10-03, branch `kernel-opt` at 455af5d (worktree `wtU`, then the main checkout).

- Nothing here is registered. Amendment 17 registers the design chosen from it, and its fresh 4-model M1 is the test.
- Every timed output was checked bitwise against today's adopted path, and all were equal.
- **Maps and tags:** the paper's TM-OPT+TC 16x64 and 8x64 artifacts. Typical = the lower-median module per projection;
  worst = the densest. The FlipQuant calibration will change later.
- **Scripts and outputs** are in `exploration/`. The raw M1-style records are in
  `/home/dev/n16k64_campaign/kernel_opt/U/explore`, with sha256 in `exploration/raw_sha256.txt`.

**The request**, relayed by the coordinator:
- "make ptxas emit uniform branches without BSSY/BSYNC/WARPSYNC around the per-k_tile dispatch; keep the if-conversion
  poison and the no-predicated-OMMA invariant."
- "CPU SASS check first; then a disclosed exploration timing (4096³ all-E2M1/real/all-E0M3 plus narrow widths) against
  today's adopted builds and the no-dispatch ceilings."
- "Register and run the full chain only if it beats noise."

## The change

`-DMIXFP4_UNIFORM_DISPATCH=1` passes the dispatch index through `__reduce_or_sync(0xffffffff, ·)` wherever it is
computed: the pipelined next-index read and the non-pipelined `dispatch` lambda.
- **Why it works:** the index is warp-uniform by construction, so REDUX.OR is the identity on it. Its result sits in a
  uniform register, so ptxas can prove the tree's branches are uniform.
- **Tried and dropped:** a warp vote (`__any_sync`) per tree compare. ptxas duplicated arm 0 (sites {0: 576, 1: 512}),
  and the patcher refuses that.

## 1. SASS, CPU only (`exploration/sass_U.json`)

Every REDUX build, both families and every width, against the adopted build:

| | adopted | REDUX build |
|---|---|---|
| BSSY / BSYNC / WARPSYNC between the first and last OMMA (the k_tile loop and its peeled last iteration) | 16–17 / 18 / 47 | 0 / 0 / 0 |
| WARPSYNC in the kernel | 50–51 | 1 |
| instructions | — | 40–144 fewer |
| predicated OMMAs | 0 | 0 |
| patcher census | — | unchanged |

- **The poison stays in the source.** It is `asm volatile("bar.warp.sync -1")` per arm. What disappears is its
  WARPSYNC: ptxas elides it once the warp is provably converged. The arms remain real branches (0 predicated OMMAs).
- **The tree itself is unchanged.** A SASS walker gives the same compares and the same taken branches per pattern in
  both builds. Pattern 0 takes 0 taken branches in the head and 2 in the tail; pattern 15 takes 5 + 1.
- **What goes is the reconvergence code:** two BSSY/BSYNC pairs and two WARPSYNC per arm.
- **The REDUX result arrives late in each arm,** because its input is the next stage's flags, read after the barrier:

  | build | REDUX position in the arm | OMMAs after the instruction that consumes it |
  |---|---|---|
  | 16x64, width 128 | 70–82 of 88–94 | about 10 |
  | 8x64, width 128, pipelined | 86–116 of 110–126 | 0–8 |

- **The final hook reproduces the exploration builds.** It is the main checkout's version, with the vote variant
  removed. With the define, it reproduces the exploration builds' SASS exactly (n16k64_wA_n64_t0, n8k64_wB_m16_t0).
  Without it, it reproduces today's adopted SASS.

## 2. Fixed widths, cold and back-to-back (`exploration/quick_U.out`, `quick_U2.out`)

The cells: 4096³; 4096x4096 T=1024; 14336x4096 T=2048; 4096x14336 T=512 at width 128; the three shapes at T=16 at
width 16. Cold is the M1 condition. Three rotated rounds; % against the adopted build.

| build | width 128, all-E2M1 / real | width 128, all-E0M3 | width 16 |
|---|---|---|---|
| **16x64, REDUX (pipelined, as today)** | −0.3 … −1.0 % | +0.3 … +3.7 % | −0.0 … −0.9 % |
| **8x64, REDUX, non-pipelined** | −3.6 … +1.5 % | +1.1 … +2.9 % (two runs) | −1.5 … +0.2 % |
| **8x64, REDUX + `MIXFP4_PIPE_FLAGS=1`** | −1.6 % at T=1024; +1.4 / +1.7 % at 4096³; +4 % at 4096x14336 T=512 | −0.9 … 0.0 % | −0.5 … −2.3 % |
| **8x64, `MIXFP4_PIPE_FLAGS=1` alone (no REDUX)** | −0.1 … −1.7 % | −0.4 … −1.1 % | −0.1 … −2.5 % |

The all-E0M3 slowdown of the 16x64 REDUX build is not a longer tree path (section 1). Its cause is not isolated.

## 3. Per T through the adopted tables, M1's method (`exploration/quick_U3_*.out`)

Llama-3.1-8B and Phi-4; per-forward sums over the typical or worst modules; % against the adopted path; median over
the 12 T. Cells marked "every round" were above or below zero in all three rounds.

| candidate | Llama typical | Llama worst | Phi-4 typical | Phi-4 worst |
|---|---|---|---|---|
| **16x64 REDUX, all widths** | −0.38 % (every T below 0 in every round) | −0.73 %; T=1/4 +1.5 / +0.6 % (every round) | −0.56 % (every T below 0 in every round) | −0.24 %; T ≤ 64 +0.25 … +0.60 % (every round) |
| **8x64 REDUX, non-pipelined** | +0.05 % | — | +0.34 % | — |
| **8x64 REDUX + pipelined flags** | −0.28 %: T ≤ 256 −0.3 … −2.5 %, T ≥ 512 +0.7 … +2.0 % | — | +0.23 %: T ≤ 16 +0.0 … +0.2 % | — |
| **8x64 pipelined flags alone** | −0.21 % | −0.29 %; T ≤ 32 +0.15 … +0.8 % | −0.21 % | −0.14 %; T ≤ 32 +0.1 … +0.6 % |
| **The hybrid** (REDUX + pipelined at 16/32/64/'128x64', pipelined alone at 128) | — | — | −0.13 %; T ≤ 32 +0.05 … +0.37 % | — |

- **Both changes help on the wide tiles,** where the arms are long: 16x64 at T ≥ 128, 8x64 at T ≥ 64.
- **On the narrow decode tiles they cost a little,** above all on E0M3-heavy modules. Widths 16 and 32 carry 4–8 OMMAs
  per k_tile and warp, and the kernels there are latency-bound.
- **The hybrid was dropped:** it was chosen on Llama, and Phi-4 did not confirm it.

**The per-width composite** (`quick_U3_composite.out`) takes the candidate where the table's width is 64 or wider, and
today's build below. Both were measured in the same rounds, but on the same data the cut was chosen from.

| | Llama typical | Llama worst | Phi-4 typical | Phi-4 worst |
|---|---|---|---|---|
| **16x64 REDUX at widths 64 and 128** | −0.32 % | −0.48 % | −0.44 % | −0.24 % |
| **8x64 pipelined flags at 64, '128x64', 128** | −0.18 % | −0.22 % | −0.14 % | −0.12 % |

- No cell was above zero in every round.
- At T ≥ 128: 16x64 −0.04 … −1.58 %, 8x64 −0.1 … −1.5 %.

## 4. The 16x64 width-64 build (item 2; `exploration/quick_W64.out`)

Three Llama shapes at T = 64 … 512, every width; the default scheduler setting; cold.

**The tile is not slow.** `n16k64_wA_nodisp_n64_t0` is `stock_wA_n64`: the same 1,424 instructions with the same
opcode counts, 6 mainloop stages, 100,352 B of shared memory, the same tile, warps and epilogue. Only the register
allocation differs. At width 64 their times are within ±0.5 % in every cell.

**The excess is the dispatch's reconvergence code.** Over its own ceiling, at 4096x4096 T=128:

| build | all-E2M1 | real map |
|---|---|---|
| adopted n64 | +6.3 % | +10.6 % |
| REDUX n64 | +0.4 % | +7.7 % |

- On all-E2M1 the adopted builds are +1.5 % over the ceiling at width 32 and −0.6 % at width 128. The cost is that
  large at width 64 alone.
- The same pattern holds at 4096x14336 T=128 and at 14336x4096 T=256.
- With REDUX, width 64 becomes the fastest at T=128 for 4096x4096 and for 4096x14336, as it is for stock. Today's
  mixed table picks width 32 there.

**One outlier (item 3).** At 14336x4096 T=512, width 128, today's dispatch costs +4.2 % even with all-E2M1 tags. At
width 128 elsewhere it is −0.6 … +0.6 %.

## 5. What is registered (amendment 17)

The coordinator approved this design. The width cut was chosen after the worst-tag data above; the fresh 4-model M1 is
the test.

- **Part A**, the builds:
  - 16x64: `MIXFP4_UNIFORM_DISPATCH=1` at widths 64 and 128;
  - 8x64: `MIXFP4_PIPE_FLAGS=1` at widths 64, '128x64' and 128;
  - widths 16 and 32 in both families stay today's builds, with identical SASS.
- **Part B:** re-tune the six 16x64 cells where the adopted mixed and stock rows disagree and one of the two widths is
  64 or 128, on part A's builds. Where a width changes, re-tune its scheduler row too.
