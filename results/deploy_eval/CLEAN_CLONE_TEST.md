# Part 1: clean-clone test

2026-09-27, 13:07–13:15 UTC. The pushed branch `tm-opt` (35ed746) was cloned from GitHub into a scratch directory,
`/home/dev/n16k64_campaign/clean_clone_test`, which was deleted afterwards. It was not a second development
checkout.

**Method.** Only `docs/BUILD_AND_USE.md` sections 1 and 3 were followed (script `clean_clone_test.sh`, log
`clean_clone_test.txt`), in an emptied environment (`env -i`). Only these were kept:
- `HOME`;
- a minimal `PATH` whose `python` is the environment of the recorded runs;
- `CUDA_HOME`, a CUDA 13.1.115 toolkit made with the documented conda recipe.

No `PYTHONPATH`, Hugging Face cache or other variable was set.

| step (docs section) | command | result |
|---|---|---|
| 1 clone | `git clone https://github.com/brian030128/NVFP4-RaZeR.git && git checkout tm-opt` | ok (tracks origin/tm-opt) |
| 1 submodule | `git submodule update --init sm120/third_party/cutlass` | ok, e64a9136dd92… |
| 3 build | `python sm120/build.py --config n16k64_wA --selftest` | ok: census 512 + 512, 0 predicated, selftest PASS, SASS d34b36a112d2… |
| 3 build | `python sm120/build.py --config n8k64_wB --selftest` | ok: census 512 + 512, 0 predicated, selftest PASS, SASS 846c8878c38e… |
| 3 tests | `python -m pytest sm120/tests -q` | **189 passed, 205 skipped, 0 failed** |
| 3 native (a) library | `bash repro_local/realquant/build.sh b8x64 lib` | ok |
| — | `git status --short` | clean: build outputs are ignored |

- **The skips:** every skipped test names a configuration this test did not build (the narrow `*_n16/_n32/_n64`,
  `n16k64_wA_8x1`, `n16k64_wA_sk`) and the command that builds it. The docs say tests of unbuilt configurations
  are skipped.
- **The same SASS as the development checkout:** both kernels have the SASS hashes of the development checkout's
  builds, which are those of the builds R2 benchmarked.
- **The development checkout, for comparison:** with all ten documented configurations built, 323 passed and
  71 skipped. The skips are only `n16k64_wA_8x1` and `n16k64_wA_sk`, which the workflow does not use.

**What a new user must do** (all documented):
1. Clone, check out `tm-opt`, and init the CUTLASS submodule.
2. Provide CUDA 13.1 via `$CUDA_HOME` and a g++ for it.
3. Create the Python environment from `sm120/requirements.lock.txt`.
4. Run the `sm120/build.py` commands, then `pytest sm120/tests`.
5. For calibration, also `repro_local/realquant/build.sh b8x64 lib`, the data preparation, and Hugging Face
   access (Llama-3.1-8B is gated).

**Undocumented steps taken: none** for sections 1 and 3.

**Not exercised here:**
- Creating the CUDA 13.1 and Python environments from scratch. Existing environments with the documented recipe
  and the lock file's package versions were used; the Python one was 3.11 rather than the documented uv 3.12.
- Sections 4–8 (data, calibration, export, evaluation, benchmark). These run in Parts 2–3 from the development
  checkout, with the same scripts.
