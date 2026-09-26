#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reason_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Llama-3.1-8B-Instruct calibration record + weights, and lm-eval 0.4.9.2 (the version
# the RaZeR env pins, which defines gsm8k_llama and mmlu_cot_llama) as an overlay.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/llama8b/bin/python MODEL=llama8b
source slurm/task_reorder_env.sh
DEPS=$MAIN/.joblib/lmeval0492_x86
if [[ ! -d "$DEPS/lm_eval" ]]; then
  $MAIN/.joblib/bootstrap/bin/uv pip install --python "$PYTHON" --target "$DEPS" \
    lm_eval==0.4.9.2 "huggingface_hub>=0.34,<1.0" langdetect immutabledict
  rm -rf "$DEPS"/{transformers,tokenizers,torch,torchvision,torchaudio,triton,nvidia}
  rm -rf "$DEPS"/{transformers,tokenizers,torch,torchvision,torchaudio,triton}-*.dist-info
fi
PYTHONPATH="$DEPS:$PYTHONPATH" "$PYTHON" -c "
import lm_eval, transformers, importlib.metadata as M
from lm_eval.tasks import TaskManager
tm = TaskManager()
print('lm_eval', M.version('lm_eval'), 'transformers', transformers.__version__)
for t in ('gsm8k_llama', 'mmlu_cot_llama'):
    print('TASK', t, t in tm.all_tasks)
"
"$PYTHON" make_instruct_prior.py
