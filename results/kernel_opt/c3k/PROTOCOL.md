# Kernel-opt C3k: GEMM latency against the E0M3 tile share on the adopted path — protocol

Registered before any GPU run of the sweep, as amendment 8 of `results/kernel_opt/PROTOCOL.md`. The hashes are in
`registration.json`.

**The request.** The user, through the coordinator: "before B′'s GPU work, produce the GEMM latency vs E0M3 tile
fraction curve (a C3-style sweep) on the current best kernels", with the specification below. The coordinator agreed
to a matched no-dispatch ceiling (`nodisp_ko`) in place of the width-128-only `n16k64_wA_nodisp_t0`.

**What is timed** (`experiments/kernel_opt/c3k_fraction.py`):
- **Shapes:** Llama-3.1-8B's 4096x4096 (o_proj), 14336x4096 (gate_proj) and 4096x14336 (down_proj).
- **Tokens:** T ∈ {1, 16, 128, 512, 2048, 8192}.
- **Maps:**
  - E0M3 tile share f ∈ {0, 1, 2, 5, 10, 25, 50, 75, 100} %, in two patterns as in C3: random (a seeded Bernoulli
    per 16x64 tile) and contiguous (a row-major prefix). f = 0 and f = 100 are one map each.
  - The real map: the typical module (median E0M3 count) of that projection in Llama-3.1-8B's TM-OPT+TC 16x64 map.
- **Kernels**, each a KernelSet that picks the width per T from its table and uses its scheduler rows:

  | kernel | set | builds | table | what it is |
  |---|---|---|---|---|
  | adopted default | `mixed_ko` | `build_7` | adopted | the deployed path |
  | adopted freq | `mixed_ko` | `build_7freq` | adopted | #2's pattern-0-first dispatch |
  | paper | `mixed` | `sm120/build` | paper | the paper kernel: tagged, default dispatch |
  | ceiling | `nodisp_ko` | `build_C3k` | adopted, plus `mixed_ko`'s scheduler rows | new: the adopted tiles with the format dispatch compiled out |
  | references | `stock_ko`; the paper stock | `build_7`; `sm120/build` | adopted; paper | each kernel's own stock |

  The ceiling and the references do not depend on the tags. They are timed once per shape and T, on all-E2M1 weights.
- **Weights:** seeded N(0, 0.02) per shape, quantized with each map: E0M3 with alpha = 1 on tagged tiles, FourOverSix
  elsewhere. The references take the same weights quantized FourOverSix.
- **Method (deviation-2):**
  - cold weights, by rotation and a 512 MiB flush;
  - the activation quantizer run after the flush, untimed;
  - isolated CUPTI GEMM launches, 3 rounds × 30, in a rotated order per (shape, T);
  - the median of all launches is reported; telemetry is sampled.
- **Checks:**
  - adopted default, adopted freq and the paper kernel produce bitwise-equal outputs on every map, shape and T; a
    failure stops the sweep;
  - G2 for the 4 ceiling builds: E2M1 only, nothing predicated.

**Output** (`experiments/kernel_opt/c3k_analyze.py`, into this directory):
- `C3k.md`:
  - the decomposition at f = 0;
  - the overhead vs own stock, and kernel(f) / kernel(0), per shape × T × pattern × kernel;
  - the slope, and where the default dispatch and #2's cross.
- `C3k.csv`, `C3k_summary.json`, `C3k_raw.json`.
- Figures `C3k_overhead{,_mean}.{png,pdf}`.
- `REPORT.md` summarizes them.

**Descriptive only.** Nothing is tuned or adopted from the sweep.

**Builds:**
- `build_C3k` holds the 4 ceiling configurations, built CPU-only from the sweep's sources before registration.
- `build_7`, `build_7freq` (amendment 7) and the paper builds are used as registered earlier.
