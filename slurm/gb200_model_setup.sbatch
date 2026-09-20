#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_modelsetup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
ROOT=/home/u4320956/.cache/mixfp4-model-runtime
OLD=/work/u4320956/b200/model-runtime
if [[ ! -d "$ROOT" && -d "$OLD" ]]; then mv "$OLD" "$ROOT"; fi
mkdir -p "$ROOT"
export UV_CACHE_DIR="$ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_LINK_MODE=hardlink
if [[ ! -x "$ROOT/bootstrap/bin/uv" ]]; then python3 -m pip install --prefix "$ROOT/bootstrap" uv; fi
UV="$ROOT/bootstrap/bin/uv"
if [[ ! -f "$ROOT/environment_ready.json" ]]; then
 "$UV" venv --clear --python 3.11 --managed-python "$ROOT/venv"
fi
"$UV" pip install --python "$ROOT/venv/bin/python" 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install --python "$ROOT/venv/bin/python" 'transformers==4.57.3' safetensors accelerate sentencepiece protobuf
OUT="$PWD/results/task_reorder/full_model_20260920/setup_$SLURM_JOB_ID"
mkdir -p "$OUT"
"$ROOT/venv/bin/python" - <<'PY' > "$OUT/environment.json"
import torch,transformers,platform,json
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,arch=platform.machine(),device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0)),indent=2))
x=torch.randn(16,16,device='cuda',dtype=torch.bfloat16);assert torch.isfinite(x@x).all()
PY
cat "$OUT/environment.json"
cp "$OUT/environment.json" "$ROOT/environment_ready.json"
