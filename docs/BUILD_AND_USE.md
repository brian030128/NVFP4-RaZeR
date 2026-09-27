# Build and use: TM-OPT+TC tile maps on the SM120 MixFP4 kernel

This is the path from a fresh clone to a calibrated model running on the native Mixed-FP4 kernel:
1. build the kernel;
2. prepare the data;
3. calibrate a tile map with TM-OPT+TC;
4. export a deployment artifact;
5. evaluate perplexity through `NativeLinear`;
6. benchmark.

Everything used below is in this repository. The only external downloads are the CUTLASS submodule, the CUDA
toolkit, the Python packages, and the Hugging Face models and datasets. Details on the kernel itself are in
`sm120/README.md`; the final method and its evaluation are in `results/tm_opt/REPORT_QR.md`.

## 1. Clone

```bash
git clone https://github.com/brian030128/NVFP4-RaZeR.git
cd NVFP4-RaZeR
git checkout tm-opt
git submodule update --init sm120/third_party/cutlass     # CUTLASS, pinned at e64a913
```

## 2. Requirements

- **GPU:** compute capability 12.0 (RTX 5090, RTX PRO 6000 Blackwell), with a driver that supports CUDA 13.1.
- **CUDA toolkit 13.1** (`nvcc`, `cuobjdump`), found through `$CUDA_HOME` (default `/usr/local/cuda-13.1`). The
  builds refuse other releases, because the kernel's SASS patch sites were validated with 13.1. One way to get it:
  ```bash
  conda create -y -n cuda131 -c nvidia -c conda-forge cuda-nvcc=13.1.115 cuda-cudart-dev=13.1 \
      cuda-cudart-static=13.1 cuda-cuobjdump=13.1 cuda-nvdisasm=13.1 cuda-cccl=13.1
  export CUDA_HOME=$(conda run -n cuda131 printenv CONDA_PREFIX)     # the env's prefix, e.g. ~/miniconda3/envs/cuda131
  ```
- **Host compiler:** a g++ supported by CUDA 13.1 (`$CXX`, default `g++`). Tested with 11.4 and 13.3.
- **Python:** the pinned environment `sm120/requirements.lock.txt`:
  ```bash
  uv venv --python 3.12 ~/envs/mixfp4 && VIRTUAL_ENV=~/envs/mixfp4 uv pip install \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
      -r sm120/requirements.lock.txt
  ```
  It installs torch 2.9.0+cu128, triton 3.5.0 and transformers 5.16.1. The recorded TM-OPT runs used Python 3.11
  with the same package versions.
- **Hugging Face:** the models (Llama-3.1-8B is gated: accept its license and log in) and the datasets are fetched
  into the cache at `$HF_HOME`, at pinned revisions.

Run everything from the repository root with `PYTHONPATH=.`.

## 3. Build and test the kernel

```bash
export CUDA_HOME=/path/to/cuda-13.1
python sm120/build.py --config n16k64_wA --selftest     # 16x64 maps (and 256x64 maps, as 16x64 granules): weights on A
python sm120/build.py --config n8k64_wB --selftest      # 8x64 maps: weights on B
python sm120/build.py --config stock_wA                 # FourOverSix / NVFP4 baselines (stock CUTLASS mainloop)
python sm120/build.py --config stock_wB
# narrow token tiles for small token counts (decode): kernel='auto' / 'auto_stock' need all four widths of a family
for c in n16k64_wA_n64 n16k64_wA_n32 n16k64_wA_n16; do python sm120/build.py --config $c --selftest; done
for c in stock_wA_n64 stock_wA_n32 stock_wA_n16; do python sm120/build.py --config $c; done
python -m pytest sm120/tests -q                          # needs the GPU; tests of unbuilt configurations are skipped
```

- **What each build checks:** it verifies the toolchain and the CUTLASS commit, and compiles into
  `sm120/build/<config>/`. Then it installs the E0M3 formats with the SASS patcher, and aborts unless the patched
  instruction census is the pinned one.
- **`--selftest`:** also requires the upstream self-test to pass on the patched kernel and fail on the unpatched one.

The native (a) evaluator (`run_multiround.py --eval-backend native`) has its own library, built from the same
vendored kernel. The `--tm-opt` preset uses it for its development monitor and its final evaluation:

```bash
bash repro_local/realquant/build.sh b8x64 lib          # -> repro_local/realquant/build/bin/libb8x64.so
```

## 4. Data

The calibration set (128 math/code windows) and the three development sets (64 windows each) are rebuilt from
pinned datasets and accepted only by their recorded hashes. CPU only.

```bash
export DATA=/path/to/data_root
python prepare_multiround_data.py --out $DATA                                     # Llama-3.1-8B
python prepare_model_data.py --out $DATA --model mistral7b --model phi4 --model qwen27b
```

The WikiText-2 / C4 evaluation windows are drawn at evaluation time: `run_baseline_protocol_audit.data`, validated
against the published records.

## 5. Calibrate: TM-OPT+TC (the final method)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_train_map.py --model mistral7b --data-root $DATA --unit 8x64 \
    --tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 \
    --out runs/tc_mistral7b_8x64
```

- **Units:** `--unit 8x64 | 16x64 | 256x64`.
- **Models:** `mistral7b`, `phi4`, `llama8b` (add `--transformers-deviation`: its calibration record names an older
  transformers), and `qwen27b` (add `--gpus 1 --batch 2 --accum 4`: micro-batch 2 × accumulation 4, so the optimizer
  batch stays 8 sequences).
- **What `--tm-opt` sets:** the verified preset (lean candidate store, fused activation quantization, chunked loss,
  deterministic, native (a) monitor and final evaluation). `--tile-grad-tc` computes the tile gradient on BF16
  tensor cores with FP32 accumulation.
- **Output:** `runs/.../map.pt`, one bool tile grid per quantized Linear, plus `report.json` (cost, development KL
  and PPL).
- **Time and memory:** about 8–16 min of selection per 7–14B model, with peak GPU memory 36–60 GiB. Qwen3.8-27B takes
  80 min and 90 GiB.

## 6. Export the deployment artifact

```bash
python export_map_artifact.py --model mistral7b --data-root $DATA --map runs/tc_mistral7b_8x64/map.pt --unit 8x64 \
    --ownership --out artifacts/mistral7b_tc_8x64
python export_map_artifact.py --model mistral7b --data-root $DATA --kind four_over_six --ownership --out artifacts/mistral7b_fo6
python export_map_artifact.py --model mistral7b --data-root $DATA --kind nvfp4 --ownership --out artifacts/mistral7b_nvfp4
```

- **The map file:** the exporter writes `artifacts/<name>.mixfp4map` (MIXFP4MAP/1).
- **Checks before the artifact is written:**
  - the model's weights equal the calibration record's;
  - the calibration's candidates (FourOverSix E2M1, E0M3 alpha=1) equal the kernel's fake-quant definition;
  - every packed weight decodes bit for bit to the fake-quant weight the map defines.
- **`--ownership`:** runs `sm120/eval/ownership_check.py`. The kernel executes every weight element through an
  identity GEMM, and the check compares its decoded value and format with the stored codes and the map.
- **Units and kernels:**

  | map unit | type block | kernel |
  |---|---|---|
  | 8x64 | (8, 64) | `n8k64_wB` |
  | 16x64 | (16, 64) | `n16k64_wA` |
  | 256x64 | exported as 16x64 granules | `n16k64_wA` |

## 7. Evaluate: perplexity through NativeLinear (convention (c))

```bash
python run_ppl_deploy.py --model mistral7b --data-root $DATA \
    --evaluate BF16=bf16 \
    --evaluate tc-8x64-fake=fake:map:artifacts/mistral7b_tc_8x64.mixfp4map \
    --evaluate tc-8x64=native:artifacts/mistral7b_tc_8x64 \
    --evaluate FourOverSix=native:artifacts/mistral7b_fo6 --evaluate NVFP4=native:artifacts/mistral7b_nvfp4 \
    --out eval/mistral7b
```

- **Windows and batching:** WikiText-2 and C4 on the released protocol windows, one window per forward.
- **NativeLinear:** every quantized Linear runs on the kernel with per-token activation scales.
- **Fake (c):** the same numerics in BF16 fake quant, the like-for-like reference.
- **Kernel defaults:** `auto` for 16x64 artifacts, `n8k64_wB` for 8x64, `auto_stock` for FourOverSix/NVFP4.
- **Output:** `report.json` holds the per-window NLLs, the PPLs, native coverage and wall time.

**How close the deployment path is:** `results/deploy_eval/REPORT.md` compares NativeLinear (c) with fake (c) and
with the (a) evaluator on all four models. Every TM-OPT+TC map stays significantly better than FourOverSix and
NVFP4 under NativeLinear. The kernel-vs-fake differences are |ΔPPL| ≤ 0.021: 37 of 40 cells are not
significant, and 3 exceed 2 SE by 3–5 %, with diagnostics given there.

The older evaluators stay available:
- `run_multiround.py --evaluate-map ... --eval-backend native|fake`, convention (a): one activation scale per window;
- `sm120/eval/ppl.py`, the SM120 campaign's evaluator.

In code:

```python
import sys; sys.path.insert(0, 'sm120')
from mixfp4_sm120 import model as NM
report = NM.install(hf_model, 'artifacts/mistral7b_tc_8x64', kernel='n8k64_wB')   # 16x64 artifacts: kernel='auto'
```

## 8. Benchmark

```bash
python sm120/bench/model.py --model mistral7b --policy native:artifacts/mistral7b_tc_8x64:n8k64_wB \
    --prefill 1x2048,4x2048 --decode 1 --out bench/mistral7b_tc_8x64.json
python sm120/bench/model.py --model mistral7b --policy native:artifacts/mistral7b_fo6:auto_stock --out bench/mistral7b_fo6.json
```

- **What it measures:** prefill latency, and decode tokens/s in eager and CUDA-graph modes. The GPU must be idle.
- **The report's figures:** R2 in `results/tm_opt/REPORT_QR.md`, with 5 shuffled rounds and a profiler
  decomposition.

## License status of the vendored kernel

The mixfp4 kernel in `sm120/kernel` (brian030128/mixfp4@7b3ab34) is vendored with the author's permission (same
research team). Upstream has no license file. An explicit license must be added before any public release; the
team decides which. See `sm120/kernel/NOTICE.md`. CUTLASS (the submodule) is BSD-3-Clause.

## Where things are

| path | what |
|---|---|
| `sm120/` | the vendored mixfp4 kernel (`kernel/`, provenance in `VENDORED.json` / `LOCAL_CHANGES.md`), `build.py`, the `mixfp4_sm120` package (`NativeLinear`, `model.install`, activation quantizer, kernel selection), eval / bench / tests |
| `run_train_map.py` | TM-OPT and TM-OPT+TC calibration |
| `export_map_artifact.py`, `run_ppl_deploy.py` | map → artifact, and PPL through NativeLinear / fake (c) |
| `run_multiround.py` | MR-OPT, and the convention (a) evaluators (native (a), fake (a)) |
| `repro_local/realquant/` | the native (a) evaluator's kernel library (built from the vendored kernel), candidate store, research benches |
| `prepare_multiround_data.py`, `prepare_model_data.py` | calibration and development data |
