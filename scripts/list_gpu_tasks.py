"""
    Which Terminal-Bench tasks require a GPU inside the container?

    harbor's singularity environment declares no GPU capability, so a task that asks for one
    fails before the agent starts:

        RuntimeError: Task requires 1 GPU(s) but EnvironmentType.SINGULARITY ...

    Those tasks say nothing about a quantization policy -- they fail identically for every
    policy -- so they are excluded up front rather than discovered one failure at a time. This
    reads the declaration out of each task.toml and prints the -x arguments to pass to
    `harbor run`.

        harbor download 'terminal-bench/terminal-bench@4.0.0' -o <dir>
        python scripts/list_gpu_tasks.py <dir>/terminal-bench
"""

import sys
import tomllib
from pathlib import Path


def gpu_count(cfg):
    """The GPU count a task declares, wherever the schema puts it."""
    for section in ('environment', 'resources', 'agent', 'verifier'):
        block = cfg.get(section)
        if isinstance(block, dict):
            for key in ('gpus', 'n_gpus', 'gpu_count', 'num_gpus'):
                v = block.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
                if isinstance(v, str) and v.strip().isdigit() and int(v) > 0:
                    return int(v)
    for key in ('gpus', 'n_gpus'):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 0


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    tomls = sorted(root.glob('*/task.toml'))
    if not tomls:
        print(f'no task.toml under {root}')
        return 1

    gpu, total = [], 0
    for path in tomls:
        total += 1
        try:
            cfg = tomllib.loads(path.read_text())
        except Exception as exc:
            print(f'  (unparsed) {path.parent.name}: {exc}')
            continue
        n = gpu_count(cfg)
        if n:
            gpu.append((path.parent.name, n))

    print(f'{total} tasks, {len(gpu)} declare a GPU')
    for name, n in gpu:
        print(f'  {name}  (gpus={n})')
    if gpu:
        print()
        print('exclude arguments:')
        print(' '.join(f'-x {name}' for name, _ in gpu))
    return 0


if __name__ == '__main__':
    sys.exit(main())
