# repro_local/realquant

The kernel library and helpers behind the **native (a) evaluator**, `native_dev.NativeDev`. It runs as
`--dev-backend native` / `--eval-backend native` in `run_multiround.py` and `run_train_map.py`. The `--tm-opt`
preset uses it for the development monitor and the final evaluation. The directory also holds:
- the lean candidate store (`candidate_store.py`);
- the fused activation quantizer (`fused_quant.py`);
- the tile-score kernel (`tile_score.py`) used by the TM-OPT / MR-OPT speed-ups.

## Build

Everything comes from this repository: the vendored kernel `sm120/kernel` (brian030128/mixfp4@7b3ab34, see
`sm120/kernel/VENDORED.json`) and the CUTLASS submodule.

```bash
git submodule update --init sm120/third_party/cutlass
CUDA_HOME=/path/to/cuda-13.1 bash repro_local/realquant/build.sh b8x64 lib   # -> repro_local/realquant/build/bin/libb8x64.so
```

- `$CXX` picks the host compiler (default `g++`).
- `$RQ_BUILD_DIR` moves the build directory; `rq.LIB_DIR` reads the same variable.
- Other configurations: `wt_as_A`, `stock`, the `*_nodisp` latency builds and `wt_as_A_colD`, as documented in
  `build.sh`.

**Verified** (2026-09-27): the `b8x64` library built this way has SASS byte-identical to the library the recorded
runs used, and it reproduces a committed native (a) evaluation bitwise (Llama-3.1-8B, FourOverSix and TM-OPT+TC
8x64, all 141 WikiText-2 and 256 C4 windows). That library was built from an external checkout of the same upstream
commit, with a different host compiler.

## Research only

These are not part of the documented workflow (`docs/BUILD_AND_USE.md`):
- the latency studies `bench_*.py`;
- `evaluate_ppl_real.py`;
- the tests that read campaign data at fixed local paths (`test_rq.py`, `test_tile_score.py`,
  `test_phase1_speedups.py`).
