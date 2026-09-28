"""Shared configuration of the paper-experiment steps (docs/PAPER_EXPERIMENTS.md).

Every path is an environment variable with the default below. Nothing here needs Slurm.

  PAPER_REPO          the repository                          default: two levels above this file
  PAPER_PYTHON        the Python interpreter                  /home/dev/.conda/envs/n16k64/bin/python
  PAPER_CUDA_HOME     CUDA 13.1 for the SM120 kernels         /home/dev/.conda/envs/mixfp4-cuda131
  HF_HOME             the Hugging Face cache                  /home/dev/.cache/huggingface
  PAPER_DATA_LLAMA8B  Llama-3.1-8B's calibration data root    /home/dev/n16k64_campaign/cost_comparison/data
  PAPER_DATA_ROOT     the other models' data root             /home/dev/n16k64_campaign/multimodel/data
  PAPER_MAPS_REF      committed TM-OPT+TC maps to reuse       /home/dev/n16k64_campaign/tm_opt/runs
  PAPER_ARTIFACTS_REF Parts 2-3 artifacts, hash comparison    /home/dev/n16k64_campaign/deploy_eval/artifacts (step 02)
  PAPER_OUT           outputs (maps, artifacts, results)      /home/dev/n16k64_campaign/paper
  PAPER_SMOKE_OUT     outputs of --smoke runs (not results)   /home/dev/n16k64_campaign/paper_smoke

Layout under the output root: maps/<model>_<unit>/, artifacts/<model>_<kind>/, ppl/, lmeval/, latency/, gemm/, tables/,
logs/ (one log per command) and commands.log (every command, with its start, end and exit code).
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get('PAPER_REPO', Path(__file__).resolve().parents[2]))
PY = os.environ.get('PAPER_PYTHON', '/home/dev/.conda/envs/n16k64/bin/python')
CUDA_HOME = os.environ.get('PAPER_CUDA_HOME', '/home/dev/.conda/envs/mixfp4-cuda131')
HF_HOME = os.environ.get('HF_HOME', '/home/dev/.cache/huggingface')
DATA = {'llama8b': os.environ.get('PAPER_DATA_LLAMA8B', '/home/dev/n16k64_campaign/cost_comparison/data')}
for _m in ('mistral7b', 'phi4', 'qwen27b'):
    DATA[_m] = os.environ.get('PAPER_DATA_ROOT', '/home/dev/n16k64_campaign/multimodel/data')
MAPS_REF = Path(os.environ.get('PAPER_MAPS_REF', '/home/dev/n16k64_campaign/tm_opt/runs'))
OUT_DEFAULT = Path(os.environ.get('PAPER_OUT', '/home/dev/n16k64_campaign/paper'))
SMOKE_OUT = Path(os.environ.get('PAPER_SMOKE_OUT', '/home/dev/n16k64_campaign/paper_smoke'))

MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
TITLES = {'llama8b': 'Llama-3.1-8B', 'mistral7b': 'Mistral-7B-v0.3', 'phi4': 'Phi-4', 'qwen27b': 'Qwen3.8-27B'}
UNITS = ('8x64', '16x64', '256x64')
MAIN_UNITS, APPENDIX_UNITS = ('8x64', '16x64'), ('256x64',)
# Llama's calibration record was made with another transformers version (accepted, recorded in every report)
DEVIATION = {'llama8b': ['--transformers-deviation']}
# TM-OPT+TC, the final method: identical settings for every model except Qwen's micro-batch (2 x accumulation 4)
TM_OPT_TC = ['--tm-opt', '--tile-grad-tc', '--param', 'ste', '--lr', '0.02', '--init-logit', '-1', '--epochs', '20',
             '--eval-every', '2', '--no-dev', '--no-eval']
TRAIN_BATCH = {'qwen27b': ['--gpus', '1', '--batch', '2', '--accum', '4']}
LMEVAL_BATCH = {'qwen27b': 8}

# Accuracy policies (steps 03, 04): label -> (artifact kind or None for BF16, kernel)
ACCURACY_POLICIES = {'bf16': (None, None), 'nvfp4': ('nvfp4', 'auto_stock'), 'fo6': ('fo6', 'auto_stock'),
                     'ours-8x64': ('tc_8x64', 'n8k64_wB'), 'ours-16x64': ('tc_16x64', 'auto'),
                     'ours-256x64': ('tc_256x64', 'auto')}
# Latency policies (step 05): label -> (artifact kind, kernel, activation-quantizer override or None).
# 'ours-<u>-nvfp4act' switches only the activation quantizer of the SAME TC artifact to nvfp4_rows: a latency-only
# configuration, never evaluated for accuracy (the maps were calibrated with FourOverSix activations).
LATENCY_POLICIES = {'nvfp4': ('nvfp4', 'auto_stock', None), 'fo6': ('fo6', 'auto_stock', None),
                    'nvfp4-wB': ('nvfp4', 'stock_wB', None), 'fo6-wB': ('fo6', 'stock_wB', None)}
for _u, _k in (('8x64', 'n8k64_wB'), ('16x64', 'auto'), ('256x64', 'auto')):
    LATENCY_POLICIES[f'ours-{_u}'] = (f'tc_{_u}', _k, None)
    LATENCY_POLICIES[f'ours-{_u}-nvfp4act'] = (f'tc_{_u}', _k, 'nvfp4_rows')
LATENCY_ONLY = tuple(p for p in LATENCY_POLICIES if p.endswith('-nvfp4act'))
PREFILL_SHAPES = ('1x128', '1x256', '1x512', '1x1024', '1x2048', '1x4096', '1x8192', '4x2048')


def unit_of(policy):
    for u in UNITS:
        if f'-{u}' in policy:
            return u
    return None


def parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(MODELS), help='comma-separated subset of ' + ','.join(MODELS))
    ap.add_argument('--units', default=','.join(UNITS), help='comma-separated subset of ' + ','.join(UNITS))
    ap.add_argument('--policies', default=None, help='comma-separated subset of the step\'s policies')
    ap.add_argument('--smoke', action='store_true', help='a tiny run to prove the step works; outputs go to PAPER_SMOKE_OUT')
    ap.add_argument('--force', action='store_true', help='redo outputs that already exist')
    ap.add_argument('--out', type=Path, default=None, help='output root (default PAPER_OUT, or PAPER_SMOKE_OUT with --smoke)')
    return ap


def setup(args):
    args.models = [m for m in args.models.split(',') if m]
    args.units = [u for u in args.units.split(',') if u]
    assert set(args.models) <= set(MODELS), args.models
    assert set(args.units) <= set(UNITS), args.units
    args.out = args.out or (SMOKE_OUT if args.smoke else OUT_DEFAULT)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    return args


# The allocator: the calibration, export and accuracy steps (01-04) run with expandable segments, as every recorded
# calibration and evaluation did (Qwen's calibration peaks at 90 of 96 GiB); the latency steps (05, 06) run with the
# default caching allocator, as Part R did.
ACCURACY_ENV = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def env():
    e = dict(os.environ)
    e.update(PYTHONPATH=f'{REPO / "sm120"}:{REPO}', CUDA_HOME=CUDA_HOME, HF_HOME=HF_HOME, HF_HUB_OFFLINE=e.get('HF_HUB_OFFLINE', '0'),
             CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
    e.pop('PYTORCH_CUDA_ALLOC_CONF', None)       # the default caching allocator unless a step passes ACCURACY_ENV
    return e


def log(out, text):
    with open(out / 'commands.log', 'a') as f:
        f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} {text}\n")


def run(out, name, cmd, extra_env=None):
    """Run one command from the repository root; its output goes to logs/<name>.log. Returns the exit code."""
    e = env()
    e.update(extra_env or {})
    line = ' '.join(shlex.quote(str(c)) for c in cmd)
    log(out, f'START {name} {line}')
    with open(out / 'logs' / f'{name}.log', 'w') as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=REPO, env=e, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    log(out, f'END {name} rc={rc}')
    print(f'{name}: rc={rc}', flush=True)
    return rc


def complete(report_path, key='status'):
    try:
        return json.loads(Path(report_path).read_text()).get(key) == 'complete'
    except (OSError, ValueError):
        return False


def artifact(out, model, kind):
    return out / 'artifacts' / f'{model}_{kind}'


def gpu_idle():
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()
    return not [p for p in q if p.strip()]


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)
