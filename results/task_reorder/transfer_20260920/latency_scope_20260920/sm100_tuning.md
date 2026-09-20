# SM100 NVFP4 tuning — 2026-09-18

Measured on the allocated **NVIDIA GB200**, node `25a-ggpn21`, Slurm job `400605`,
account **mst114554**. Compute capability 10.0, 152 SMs, CUDA 13.0.88, driver
580.105.08, vendored CUTLASS 4.6.1. These are measurements on GB200, not a separate
PCIe/SXM B200 system. The kernels target `sm_100a`.

## Results

Median of three measurements after configuration selection; units TFLOP/s.
Ordinary launches, with PDL enabled for CUTLASS:

| M×N×K | cuBLAS NVFP4 | Tuned CUTLASS | CUTLASS / cuBLAS |
|---|---:|---:|---:|
| 1024³ | 538.1 | 713.5 | 1.326× |
| 2048³ | 3177.5 | 3467.5 | 1.091× |
| 4096³ | 6164.6 | 6300.6 | 1.022× |
| 8192³ | 7569.9 | **8045.0** | **1.063×** |
| 4096×4096×16384 | 7181.5 | 7114.5 | 0.991× |
| 8192×8192×2048 | 5151.7 | 5851.1 | 1.136× |
| 16384×16384×2048 | 5850.5 | 6483.6 | 1.108× |

The earlier 8192³ result was 6137.8 TFLOP/s; the new steady-state launch path is
**31.1% faster**. Six shapes beat cuBLAS and the remaining shape is within 1%.
Full samples, selected configurations, and BF16/FP8 baselines are in
[sm100_launch.json](sm100_launch.json).

CUDA Graph replay, using the same timing mode for both implementations:

| M×N×K | cuBLAS NVFP4 | Tuned CUTLASS | CUTLASS / cuBLAS |
|---|---:|---:|---:|
| 1024³ | 817.8 | 823.1 | 1.006× |
| 2048³ | 3539.0 | 3599.8 | 1.017× |
| 4096³ | 6311.9 | 6377.0 | 1.010× |
| 8192³ | 7724.5 | **8061.4** | **1.044×** |
| 4096×4096×16384 | 7275.2 | 7143.6 | 0.982× |
| 8192×8192×2048 | 5518.9 | 5908.7 | 1.071× |
| 16384×16384×2048 | 5950.8 | 6498.2 | 1.092× |

Graph replay reduces CPU launch gaps, particularly for small shapes. Six shapes
remain faster than cuBLAS; 4096×4096×16384 is 1.8% behind. Full graph samples and
independently selected configurations are in [sm100_graph.json](sm100_graph.json).

Mixed-format results at 8192³, using the same 256×256×256 tile / 128×64 epilogue:

| A | B | Median TFLOP/s |
|---|---|---:|
| E2M1 | E2M1 | 8043.8 |
| E2M1 | E0M3 | 8044.8 |
| E0M3 | E2M1 | 8058.2 |
| E0M3 | E0M3 | 8046.0 |

The spread is under 0.2%, with no measurable mixed-format penalty in this test.
Graph-mode mixed results likewise range from 8057.7 to 8067.9 TFLOP/s across the
four format combinations (under 0.2% spread).
For smaller shapes, the mixed executable's fixed wide tile should not be compared
to a different stock tile as a measure of format-selection overhead.

## Changes

- Specialize the epilogue for `D = A * B` (`beta = 0`, no C-source load).
- Use a **128×64 epilogue** for the **256×256×256, 2-SM** kernel instead of the
  automatic 128×128 epilogue. This reduces epilogue shared-memory storage and
  changes the overlapping accumulator layout. The resulting mainloop has five
  shared-memory stages. Keep K=256: the K=512 / 64-column epilogue experiment
  exceeded the available tensor-memory space and was discarded.
- Add 128×64 and 256×128 configurations for smaller problems, plus static
  persistent scheduling alternatives. Retain the original configurations as
  controls in the tuning executable.
- Initialize CUTLASS once. Cache the kernel handle and launch geometry, and enable
  **programmatic dependent launch (PDL)**. CUTLASS's existing grid-dependency
  waits protect input accesses, while dependent kernels can begin their setup
  earlier. `MIXFP4_PDL=0` disables this path for comparisons.
- Apply the same optimized epilogue and launch path to the mixed E0M3/E2M1 kernel.
  Its default tile remains 256×256×256 with a 2×1×1 cluster. Format selection is
  uniform per launch; per-granule format metadata is outside this change.

## Profiling and controlled experiments

Nsight Compute showed high tensor-pipeline utilization rather than saturated HBM:
the original 256×128 kernel had 93.47% L2 hit rate, zero register-spill requests,
and 84.94% TC-pipeline utilization in the workload analysis capture. Larger
multicast clusters did not close the gap. At 8192³, the original best plain tile
reached about 6,149 TFLOP/s and a 2×2 cluster about 6,285.

With PDL disabled, the 256×256 no-C kernel reached about 7,232 TFLOP/s with a
128-column epilogue and 7,874 with a 64-column epilogue. Enabling PDL raised the
latter to about 8,060. At 2048³, PDL raised this same wide tile from about 2,090
to 3,101; the narrower 256×128 tile performs better there.

Nsight Systems identified cuBLAS's large NVFP4 kernel as a CUTLASS-based
256×256×256 2-SM kernel. In separate Nsight Compute captures with its default
clock/cache controls, the tuned kernel took 232.42 µs versus 235.39 µs for the
captured cuBLAS candidate, using the same 229.38 KB dynamic shared memory.
Profiler replay changes clocks/cache state and timing: **use the unprofiled
benchmark results for throughput**, not the application's TFLOP/s printed while
profiling. The first cuBLAS profiling capture is one heuristic candidate, whereas
the throughput benchmark measures all returned candidates and keeps the fastest.

Raw diagnostic captures: [stock workload analysis](sm100_profiles/mixfp4-stock-profile.txt),
[tuned kernel](sm100_profiles/mixfp4-tuned-profile.txt),
[cuBLAS candidate](sm100_profiles/mixfp4-cublas-profile.txt).

## Measurement method

Both implementations use packed E2M1 inputs, UE4M3 scales per 16 K elements,
FP32 accumulation, BF16 row-major output, alpha=1 and beta=0. The old cuBLAS
benchmark used column-major output; this comparison makes the output layout
identical too. Throughput-only data is filled directly on the GPU outside timing;
correctness uses independent, randomized inputs spanning the entire FP4 codebook
and varying positive scales.

Both libraries use `gemm_benchmark.hpp`, a minimum 50 ms untimed warmup, and CUDA
events on a dedicated stream. Initialization, allocation and graph construction
are excluded. Ordinary launch timing and graph replay are reported separately;
both sides use the same mode. These are repeated-buffer throughput measurements,
not cold-cache or end-to-end quantization measurements.
Each CUTLASS sample uses 100 iterations; each cuBLAS candidate uses 50 iterations.
The tables report the elapsed event time divided by the corresponding count.

`benchmark_sm100.py` tunes CUTLASS once per shape, freezes its configuration,
then records the median of three fresh measurements. cuBLAS tests up to eight
heuristic candidates per measurement. Execution order alternates across
repetitions. `--sweep` additionally tunes raster direction and swizzle. JSON
artifacts contain every sample and the full tuning output.

## Reproduction

Run on an allocated SM100 node; compilation must use its aarch64 toolchain:

```bash
export BUILD=/work/u4320956/b200/bld
bash scripts/build_sm100.sh
bash scripts/check_sm100.sh
MIXFP4_BENCH_GRAPH=1 bash scripts/check_sm100.sh

bash scripts/bench_sm100.sh --sweep --mixed --output docs/results/sm100_launch.json
MIXFP4_BENCH_GRAPH=1 bash scripts/bench_sm100.sh --sweep --mixed \
  --output docs/results/sm100_graph.json

"$BUILD/nvfp4_gemm_sm100" 8192 8192 8192 \
  --config=256x256x256_2sm_fast --raster=0 --swizzle=2 --no-check
"$BUILD/mixed_nvfp4_gemm_sm100" 8192 8192 8192 \
  --fmt-a=e0m3 --fmt-b=e2m1 --no-check
```

When attaching to the existing allocation, prefix commands with
`srun --jobid=400605 --overlap -N1 -n1 -c8`. For a new allocation use
`-A mst114554 -p gb200-dev --gres=gpu:1 --mem=32G -c8` and let the build script's
job-specific default build directory prevent collisions.

## Correctness

The stock kernel is checked against CUTLASS's host block-scaled GEMM reference.
Tests include 144×272×288 and 384×640×768, exercising M/N tile tails and scale
padding along K. Output is checked again after repeated launches / graph replay.
The mixed kernel uses an independent full-precision reference that decodes the
requested codebooks; all four format combinations are checked. Deliberately
using the wrong reference codebook must fail numerically, not merely return a
CUDA error. The correct-codebook relative Frobenius errors are about 0.00166;
the wrong-A-codebook control produces about 0.337.

Compute Sanitizer memcheck is also run on the tuned stock and mixed kernels.
Both report **zero errors** with PDL enabled. Final logs:
[ordinary launches](sm100_check_launch.txt), [graph replay](sm100_check_graph.txt),
and [memcheck](sm100_memcheck.txt).
