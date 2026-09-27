#!/bin/bash
# Part 1 clean-clone test: follow docs/BUILD_AND_USE.md sections 1 and 3 only, in a scratch clone of the pushed branch.
# The environment is emptied (env -i) except HOME, a minimal PATH, CUDA_HOME and the Python interpreter of an
# environment with the lock file's package versions (the conda env of the recorded runs; not re-created here).
set -x
T=/home/dev/n16k64_campaign/clean_clone_test
rm -rf $T
mkdir -p $T
cd $T
date -u +%FT%TZ
# section 1
git clone https://github.com/brian030128/NVFP4-RaZeR.git || exit 1
cd NVFP4-RaZeR
git checkout tm-opt || exit 1
git log --oneline -1
git submodule update --init sm120/third_party/cutlass || exit 1
git -C sm120/third_party/cutlass rev-parse HEAD
# section 3 (the two kernels the test asks for), then the test suite
python sm120/build.py --config n16k64_wA --selftest; echo "RC build n16k64_wA $?"
python sm120/build.py --config n8k64_wB --selftest; echo "RC build n8k64_wB $?"
python -m pytest sm120/tests -q -rs -p no:cacheprovider 2>&1 | tail -8; echo "RC pytest ${PIPESTATUS[0]}"
# the documented realquant build (native (a) evaluator)
bash repro_local/realquant/build.sh b8x64 lib 2>&1 | tail -3; echo "RC realquant ${PIPESTATUS[0]}"
git status --short | head
date -u +%FT%TZ
