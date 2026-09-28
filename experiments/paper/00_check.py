#!/usr/bin/env python3
"""Step 00: environment and build check (docs/PAPER_EXPERIMENTS.md). Writes <out>/00_check.json; exit 1 on a failure.

Checks:
- the GPU is idle (no other compute process);
- the pinned versions: transformers 5.16.1, lm-eval 0.4.11, torch with CUDA;
- the SM120 kernels are built and load: the mixed and stock width families, n8k64_wB and stock_wB;
- the RTX PRO 6000 tile table is present and picked up by both kernel sets (committed in b37e489);
- every model snapshot is in the HF cache at its pinned revision, and every calibration record is present;
- the reference manifest of the committed TM-OPT+TC maps (maps.sha256.json).
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

TABLE = 'nvidia_rtx_pro_6000_blackwell_workstation_edition.json'
CONFIGS = ('n16k64_wA', 'n16k64_wA_n16', 'n16k64_wA_n32', 'n16k64_wA_n64', 'stock_wA', 'stock_wA_n16', 'stock_wA_n32',
           'stock_wA_n64', 'n8k64_wB', 'stock_wB')


def main():
    args = P.setup(P.parser(__doc__).parse_args())
    os.environ.setdefault('CUDA_HOME', P.CUDA_HOME)
    sys.path.insert(0, str(P.REPO / 'sm120'))
    sys.path.insert(0, str(P.REPO))
    res, fail = dict(checks={}), []

    def check(name, ok, detail=None):
        res['checks'][name] = dict(ok=bool(ok), detail=detail)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            fail.append(name)

    check('gpu idle', P.gpu_idle(), 'no other compute process')
    import torch
    check('torch cuda', torch.cuda.is_available(), f'torch {torch.__version__}, CUDA {torch.version.cuda}, '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
    for pkg, want in (('transformers', '5.16.1'), ('lm_eval', '0.4.11')):
        have = md.version(pkg)
        check(f'{pkg} version', have == want, f'{have} (pinned {want})')
    from mixfp4_sm120.lib import Kernel
    from mixfp4_sm120.select import KernelSet, gpu_slug
    kernels = {}
    for cfg in CONFIGS:
        try:
            kernels[cfg] = Kernel.load(cfg).sha256
        except Exception as e:  # noqa: BLE001
            kernels[cfg] = f'{type(e).__name__}: {e}'
    check('kernels built', all(len(v) == 64 for v in kernels.values()), kernels)
    table = P.REPO / 'sm120' / 'configs' / TABLE
    slug_ok = gpu_slug() + '.json' == TABLE
    sets = {f: KernelSet(f).table_source for f in ('mixed', 'stock')}
    check('tile table', table.exists() and slug_ok and all(v == str(table) for v in sets.values()),
          dict(file=str(table), sha256=hashlib.sha256(table.read_bytes()).hexdigest() if table.exists() else None,
               gpu_slug=gpu_slug(), table_source=sets))
    spec = importlib.util.spec_from_file_location('sm120_eval_common', P.REPO / 'sm120' / 'eval' / 'common.py')
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    for m in args.models:
        try:
            snap = C.snapshot(m)
            ok = Path(snap).exists()
        except Exception as e:  # noqa: BLE001
            snap, ok = f'{type(e).__name__}: {e}', False
        check(f'{m} snapshot', ok, dict(model_id=C.MODELS[m]['model_id'], revision=C.MODELS[m]['revision'], path=str(snap)))
        from run_multiround import data_paths
        cal, dev = data_paths(m, Path(P.DATA[m]))
        check(f'{m} calibration record', (cal / 'report.json').exists(), str(cal))
    manifest = Path(__file__).resolve().parent / 'maps.sha256.json'
    check('committed map manifest', manifest.exists(), str(manifest))
    res['all_ok'] = not fail
    (args.out / '00_check.json').write_text(json.dumps(res, indent=1) + '\n')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
