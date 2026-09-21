"""Process-wide run context shared by the job wrapper and job modules."""
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

run_dir = None
gate = None
gpu_run = False


def install(directory, phase_gate, gpu_run=False):
    global run_dir, gate
    run_dir = Path(directory)
    gate = phase_gate
    globals()['gpu_run'] = gpu_run


def phase(name):
    """Mandatory preflight before an expensive phase. No-op outside a launched run (unit tests)."""
    if gate is not None:
        return gate(name)
    return None


def out_dir(*parts):
    base = run_dir if run_dir is not None else Path(os.environ.get('CAMPAIGN_RUN_DIR', '.'))
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def peak_memory():
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {str(i): int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())}
    except Exception:
        return {}


def environment(attention_backend=None, activation_quantizer=None):
    import importlib.metadata as md
    env = dict(hostname=os.environ.get('CAMPAIGN_HOST', socket.gethostname()), python=platform.python_version())
    root = Path(os.environ.get('CAMPAIGN_ROOT', '.'))
    name = os.environ.get('CAMPAIGN_ENV_NAME', 'main')
    lock = root / 'env' / f'{name}.lock.txt'
    lr = {}
    try:
        lr = json.loads((run_dir / 'launch_record.json').read_text()) if run_dir else {}
    except Exception:
        pass
    lock_sha = sha256_file(lock) if lock.exists() else None
    env['env_name'] = name
    env['lock_sha256'] = lock_sha
    env['image_id'] = lr.get('image_id')
    env['container_or_lock_sha256'] = hashlib.sha256(f'{lr.get("image_id")}|{lock_sha}'.encode()).hexdigest()
    try:
        env['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        env['driver'] = None
    for pkg, key in (('torch', 'torch'), ('transformers', 'transformers'), ('datasets', 'datasets'),
                     ('lm_eval', 'lm_eval'), ('accelerate', 'accelerate'), ('numpy', 'numpy'),
                     ('tokenizers', 'tokenizers'), ('safetensors', 'safetensors'), ('huggingface-hub', 'huggingface_hub')):
        try:
            env[key] = md.version(pkg)
        except md.PackageNotFoundError:
            env[key] = None
    try:
        import torch
        env['cuda'] = torch.version.cuda
        env['cudnn'] = torch.backends.cudnn.version()
        env['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env['compute_capability'] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception:
        env['cuda'] = None
    env['attention_backend'] = attention_backend
    env['activation_quantizer'] = activation_quantizer
    env['env_vars'] = {k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'OMP_NUM_THREADS',
                                                      'HF_HUB_OFFLINE', 'CAMPAIGN_LEASE_ID', 'CAMPAIGN_ALLOCATED_UUIDS')}
    return env

def collect_missing(out_dir):
    """Endpoints a deliverable recorded as unavailable: 'file.json', 'file.json:model' or 'file.json:model:contrast'.
    Analyses mark these with status == 'missing' instead of dropping them, and the run record must list them."""
    import json as _json
    found = []
    for p in sorted(Path(out_dir).glob('*.json')):
        try:
            d = _json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if d.get('status') == 'missing':
            found.append(p.name)
        for m, mm in (d.get('models') or {}).items():
            if not isinstance(mm, dict):
                continue
            if mm.get('status') == 'missing':
                found.append(f'{p.name}:{m}')
                continue
            for ck, c in mm.items():
                if isinstance(c, dict) and c.get('status') == 'missing':
                    found.append(f'{p.name}:{m}:{ck}')
    return sorted(found)
