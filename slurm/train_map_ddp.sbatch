#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=train_map_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --time=04:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn111,25a-hgpn150,25a-hgpn036
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Data-parallel run_train_map.py: one rank per GPU, per-tile gradients summed across ranks each step;
# rank 0 evaluates and writes outputs. All ranks share one GPU allocation (per-task GPU binding isolates
# devices and breaks NCCL peer-to-peer) and rank i selects the i-th entry of the job-level
# CUDA_VISIBLE_DEVICES list. The job's GPUs are cgroup-fenced (slurm/gpu_visibility_check.sbatch), so that
# list is the job's own allocation; the count check below guards against a mismatched allocation.
# Usage: sbatch [--ntasks=N --gres=gpu:H200:N] slurm/train_map_ddp.sbatch OUT [run_train_map.py args...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/llama8b/bin/python MODEL=llama8b
source slurm/task_reorder_env.sh
export TRAIN_MAP_DDP=1 MASTER_ADDR=$(hostname) MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

OUT="${1:?out}"; shift
export PYTHON OUT
echo "JOB_GPUS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
IFS=, read -ra ALLOC <<< "${CUDA_VISIBLE_DEVICES:?job has no CUDA_VISIBLE_DEVICES}"
[ "${#ALLOC[@]}" -eq "$SLURM_NTASKS" ] || { echo "allocation has ${#ALLOC[@]} GPUs for $SLURM_NTASKS ranks" >&2; exit 1; }
export JOB_GPU_LIST="$CUDA_VISIBLE_DEVICES"
srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpu-bind=none --wait=0 \
  bash -c 'IFS=, read -ra G <<< "$JOB_GPU_LIST"; export CUDA_VISIBLE_DEVICES=${G[$SLURM_LOCALID]};
           echo "rank $SLURM_PROCID -> GPU $CUDA_VISIBLE_DEVICES (job GPUs $JOB_GPU_LIST)" >&2;
           exec "$PYTHON" run_train_map.py --out "$OUT" "$@"' _ "$@"
