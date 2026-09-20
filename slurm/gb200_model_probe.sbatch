#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_modelprobe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=12G
#SBATCH --time=00:01:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OUT="$PWD/results/task_reorder/full_model_20260920/probe_$SLURM_JOB_ID"
mkdir -p "$OUT"
python3 - <<'PY' > "$OUT/environment.json"
import sys,platform,json,importlib,shutil,os
r=dict(python=sys.executable,version=sys.version,architecture=platform.machine(),nvcc=shutil.which('nvcc'))
for name in ['torch','transformers','tokenizers','safetensors','numpy','pip']:
 try:
  m=importlib.import_module(name);r[name]=dict(version=getattr(m,'__version__',None),file=getattr(m,'__file__',None))
  if name=='torch':r['cuda']=dict(available=m.cuda.is_available(),version=m.version.cuda,name=m.cuda.get_device_name(0),capability=m.cuda.get_device_capability(0))
 except Exception as e:r[name]=dict(error=str(e))
print(json.dumps(r,indent=2))
PY
cat "$OUT/environment.json"
