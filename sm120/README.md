# SM120 native Mixed-FP4 backend (RTX 5090 / RTX PRO 6000 Blackwell)

The native execution path for MixFP4 maps on GeForce/workstation Blackwell (compute capability
12.0): a frozen selector map is exported to packed FP4 weights with E0M3 format tags, and every
quantized `nn.Linear` of the model runs as a W4A4 block-scaled FP4 tensor-core GEMM on the mixed
E2M1/E0M3 kernel from the mixfp4 project.

```
gradient-guided selector (research/n16k64 campaign)  ->  frozen map  (MIXFP4MAP/1, maps/)
  -> eval/export_artifact.py   packed codes + tagged UE4M3 scales + global scales (artifacts/)
  -> mixfp4_sm120.NativeLinear fused Triton activation quantizer -> patched SM120 GEMM (fused epilogue)
  -> eval/ppl.py, bench/*      full-model quality and performance
```

| Path | Hardware | Where | What runs |
|---|---|---|---|
| **fake quant** | any | `quantize/`, `run_*.py`, the campaign code | dequantized BF16 weights/activations, BF16 GEMM |
| **SM120 native** (this directory) | sm_120 (RTX 5090, RTX PRO 6000) | `sm120/` | FP4 × FP4 tensor-core GEMM, N16K64 weight granule on operand A |
| **SM100 native** | sm_100 (GB200) | `native/`, `scripts/native_*.py`, `MIXFP4_GB200_PLAN.md` | the 256x64 path; unrelated code |

## Layout

| | |
|---|---|
| `kernel/` | the mixfp4 kernel, vendored from `brian030128/mixfp4@7b3ab34` (`VENDORED.json`: per-file upstream hashes). Three macro-guarded hooks were added to `src/mixed_nvfp4_gemm.cu` (`LOCAL_CHANGES.md`); with the macros unset it is the upstream code. |
| `third_party/cutlass` | CUTLASS submodule pinned to `e64a913` (v4.6.0-8), the commit mixfp4 was validated with |
| `csrc/mixfp4_sm120.cu` | C ABI: GEMM with the fused epilogue, scale-factor layout, granule map, config description |
| `build.py` | build → patch → verify → manifest, one shared library per configuration |
| `mixfp4_sm120/` | Python: `configs`, `lib` (verified loading), `numerics` (the spec), `quant_act` (Triton), `mapio`, `artifact`, `linear`, `model` |
| `maps/` | frozen N16K64 / N8K64 k=3 selector maps of the five campaign models, with provenance |
| `tests/` | pytest suite (numerics, quantizer, kernel, artifact, Linear) |
| `eval/` | artifact export, perplexity (bf16 / fake / native on one model), layer-by-layer diagnostics |
| `bench/` | kernel-only, complete-Linear and full-model benchmarks |
| `results/` | committed raw results (JSON) of the runs reported in `RESULTS.md` |
| `NUMERICS.md` | frozen numeric specification; `RESULTS.md`: measured results and open items |

## Requirements (pinned)

- GPU: compute capability 12.0. The kernel is built SASS-only for `sm_120a`.
- CUDA toolkit **13.1** (`nvcc`, `cuobjdump`), found via `--cuda-home` / `$CUDA_HOME`
  (default `/usr/local/cuda-13.1`). `build.py` refuses any other release: the SASS patch sites were
  validated with 13.1. Driver ≥ the one shipping CUDA 13.1.
- Host compiler: any g++ supported by CUDA 13.1 (built and tested with g++ 13.3).
- Python 3.12 with `torch==2.9.0+cu128`, `triton==3.5.0`, `transformers==5.16.1`,
  `safetensors==0.8.0`, `numpy==2.4.4`, `pyarrow==25.0.1`, `pytest` (the research branch's
  `main.lock.txt` versions). The fused quantizer's bit-exactness was established against these
  torch arithmetic semantics; re-run `tests/test_quant_act.py` after any torch upgrade.

```bash
git submodule update --init sm120/third_party/cutlass
uv venv --python 3.12 ~/envs/sm120 && VIRTUAL_ENV=~/envs/sm120 uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  torch==2.9.0+cu128 triton==3.5.0 transformers==5.16.1 accelerate==1.13.0 safetensors==0.8.0 \
  numpy==2.4.4 pyarrow==25.0.1 tokenizers==0.23.2 huggingface-hub==1.31.0 pytest
```

## Build → patch → verify → run

```bash
python sm120/build.py --config n16k64_wA --selftest   # the deployment kernel
python sm120/build.py --all                           # plus baselines / variants
python -m pytest sm120/tests                           # needs the GPU and the built libraries
```

`build.py` for one configuration:

1. checks the toolchain (CUDA 13.1, CUTLASS at the pinned commit);
2. generates the MMA blob header into `build/<config>/gen` (the vendored generator runs from a
   copy; the source tree is never written);
3. compiles `libmixfp4_sm120_<config>.so.unpatched`;
4. installs the E0M3 formats with `kernel/scripts/patch_mixed_nvfp4_gemm.py` and **aborts** unless
   the per-site OMMA census equals the pinned one in `configs.py`, the post-patch disassembly shows
   every site in its intended format, only the weight operand's E0M3 site exists, and no OMMA is
   predicated;
5. loads the library and checks its compiled description (granule, pinning, D layout) against
   `configs.py`;
6. `--selftest` also builds the upstream self-test driver for the same configuration and requires
   PASS patched / FAIL unpatched on random per-granule tagging, and the same OMMA census as the library;
7. writes `build/<config>/manifest.json`: source revision and file hashes, flags, toolchain
   versions, patcher hash, unpatched and patched binary hashes, census, registers/stack.

At run time `mixfp4_sm120.lib.Kernel.load` only accepts a library whose SHA-256 equals its
manifest and whose manifest says it was patched, so an unpatched binary (which silently computes
E2M1 where E0M3 was asked for) cannot be used by accident.

Then, per model:

```bash
python sm120/eval/export_artifact.py --model qwen4b --map sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
    --out sm120/artifacts/qwen4b_n16_k3
python sm120/eval/ppl.py --model qwen4b --policy bf16=bf16 \
    --policy fake_n16_k3=fake:map:sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
    --policy native_n16_k3=native:sm120/artifacts/qwen4b_n16_k3 --out sm120/results/ppl/qwen4b.json
python sm120/eval/layerwise.py --model qwen4b --map ... --artifact ... --out ...
python sm120/bench/kernel.py --out sm120/results/bench/kernel_<gpu>.json
python sm120/bench/linear.py --out sm120/results/bench/linear_<gpu>.json
python sm120/bench/model.py --model qwen4b --policy native:sm120/artifacts/qwen4b_n16_k3 --out ...
```

In code:

```python
from mixfp4_sm120 import model as NM
rep = NM.install(hf_model, 'sm120/artifacts/qwen4b_n16_k3', kernel='n16k64_wA')   # strict by default
```

## Kernel configurations

| config | weights on | weight granule | warp arrangement | role |
|---|---|---|---|---|
| `n16k64_wA` | A | 16 rows × 64 K (contiguous) | 4×2 (stock), 16 joint arms | **deployment** |
| `n16k64_wA_8x1` | A | 16 rows × 64 K (contiguous) | 8×1, 4 arms | per-GPU alternative |
| `n8k64_wB` | B | 8 cols × 64 K (contiguous) | 1×8 | N8K64 comparison |
| `n16k64_wA_nodisp` | A | — (E2M1 only) | 4×2 | dispatch-free ceiling (latency only) |
| `stock_wA`, `stock_wB` | A / B | — (E2M1 only) | CUTLASS default | stock NVFP4 baseline, same epilogue |

All use a 128×128×128 CTA tile, 4 mainloop stages, 168 registers, no stack.

## Format ownership (selector block ↔ kernel granule ↔ MMA format)

- The selector's N16K64 tile `(r, c)` of a weight `[out, in]` covers output channels
  `[16r, 16r+16)` and inputs `[64c, 64c+64)` (row-major grid, `campaign/tiles.py`).
- With weights on A, the kernel's granule map (`lib.Kernel.granule_map(0)`, derived by the
  kernel's own `build_granule_map` from its TiledMma) assigns rows `[16j, 16j+16)` of every CTA tile
  to one granule: **contiguous 16-row blocks**, for both the 4×2 and the 8×1 arrangement; the K
  granule is 64 (`MIXFP4_JOINT_KA`). So selector tile = kernel granule, no reordering is needed.
  (This is specific to one m-atom per granule; the upstream default 32-row granule is two 16-row
  blocks 64 rows apart, which is why ownership is re-derived and tested per configuration.)
- The export sets bit 7 on all 16 × 4 scale bytes of each selected tile and refuses any tile whose
  tags are not uniform (a sub-granule tag split can hang the GPU).
- `tests/test_gemm.py::test_decode_probe` proves the mapping end to end: with an identity
  activation operand the GEMM output equals each weight's hardware-decoded value exactly, so the
  test observes, for every weight element, which format the tensor core executed and compares it
  with the tile the element belongs to.

## Precision boundary and fallback

Native: every text `nn.Linear` except the output head (the campaign's scope rule). BF16 by design:
`lm_head`, embeddings, norms, attention (SDPA), rotary embedding. `model.install` is strict: a
scoped Linear missing from the artifact, or with an unsupported shape, raises; with
`strict=False` it stays BF16 and is listed in `InstallReport.fallback`. NativeLinear has no BF16
weight (reading `.weight` raises), and `eval/ppl.py` records a profiler audit of every dense GEMM
op in a native forward. Shape requirements: `in_features % 64 == 0` and
`out_features % 16 == 0` for N16K64 maps (`in_features % 32 == 0` for E2M1-only artifacts).

## Undocumented-hardware dependencies

- E0M3 is not expressible in PTX. The kernel compiles every MMA as E2M1 × E2M1 and the patcher
  sets bits 14/15 of the second 64-bit word of each OMMA instruction (A / B E0M3). The encoding
  and the E0M3 codebook are empirical (CUDA 13.1, sm_120; `kernel/tests/mma_intrinsics`,
  `revollllt/sm120-e0m3-mma@8b755d9`); a different compiler or GPU stepping must be re-validated
  with `build.py --selftest` and the test suite before use.
- Bit 7 of the UE4M3 scale byte is ignored by the tensor core (verified); any other consumer of
  these scale bytes must mask it.

## Local environment note

This directory's results were produced on a single local workstation (no Slurm); the cluster rules
in the repository's CLAUDE.md/AGENTS.md apply to the cluster, not to this machine.
