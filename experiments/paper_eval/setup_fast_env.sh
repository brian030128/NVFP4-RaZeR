#!/bin/bash
# Amendment 2: the env n16k64-fast = a clone of n16k64 plus the hybrid layers' fast kernels (the user's decision,
# 2026-10-07). pip inside the new env only; --no-deps so torch / triton / transformers stay the clone's; the CUDA
# extensions built from source for sm_120 against the env's torch with the CUDA 12.8 toolkit. Any failure stops here.
set -euo pipefail
OUT=/home/dev/n16k64_campaign/paper_eval/env
mkdir -p $OUT
ENVS=/home/dev/.conda/envs
NEW=$ENVS/n16k64-fast
CONDA=/opt/miniconda/condabin/conda
[ -e $NEW ] && { echo "$NEW exists"; exit 1; }
$ENVS/n16k64/bin/pip freeze > $OUT/freeze_n16k64.txt
$CONDA create -y -q --prefix $NEW --clone $ENVS/n16k64 > $OUT/clone.log 2>&1
P=$NEW/bin/pip
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 \
       PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
$P install --no-deps flash-linear-attention==0.5.2 > $OUT/pip_fla.log 2>&1
CAUSAL_CONV1D_FORCE_BUILD=TRUE $P install -v --no-build-isolation --no-deps causal-conv1d==1.7.0 > $OUT/pip_causal_conv1d.log 2>&1
MAMBA_FORCE_BUILD=TRUE $P install -v --no-build-isolation --no-deps mamba-ssm==2.3.2.post1 > $OUT/pip_mamba_ssm.log 2>&1
$P freeze > $OUT/freeze_n16k64-fast.txt
diff $OUT/freeze_n16k64.txt $OUT/freeze_n16k64-fast.txt > $OUT/freeze_delta.txt || true
$P check > $OUT/pip_check.txt 2>&1 || true
$NEW/bin/python -c "import torch, triton, transformers; print(torch.__version__, triton.__version__, transformers.__version__)" > $OUT/versions.txt
echo done
