#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=gpu_vis1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn111
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
# The single-GPU case (as slurm/train_map.sbatch): which physical GPU does CUDA device 0 map to?
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "visible=$(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
