#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=gpu_vis
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn111
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
# Which GPUs can a job / an srun step actually see? Compares Slurm's view with what the driver exposes.
echo "== batch step: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
echo "== srun --gpu-bind=none, per task"
srun --ntasks=4 --gpu-bind=none bash -c 'echo "task $SLURM_LOCALID CVD=${CUDA_VISIBLE_DEVICES:-unset} STEP_GPUS=${SLURM_STEP_GPUS:-unset} visible=$(nvidia-smi -L | wc -l)"' | sort
echo "== srun default binding, per task"
srun --ntasks=4 bash -c 'echo "task $SLURM_LOCALID CVD=${CUDA_VISIBLE_DEVICES:-unset} visible=$(nvidia-smi -L | wc -l) bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | tr "\n" " ")"' | sort
