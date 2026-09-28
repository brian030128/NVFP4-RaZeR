"""Deviation 1 (PROTOCOL.md): the table's effect on Qwen3.8-27B's 1x2048 prefill, table against fallback in ONE process.

The cross-session re-measurement (compare_remeasure.py) cannot resolve an expected ~0.1 % change: Qwen's single-sequence
prefill is bimodal (~760 / ~880 ms) and the sessions differ. Here one process loads the model once, installs the
artifact once, and alternates the kernel set of every NativeLinear between the table ('table') and the fallback rule
('fallback'), ALTERNATIONS times each in ABAB order. Each block is sm120/bench/model.py prefill (2 warm-ups, then REPS
timed forwards, CUDA events). All widths are bitwise identical, so only the speed can differ.

    python results/tile_table/ab_same_process.py --policy fo6@stock_wA|tc-16x64 --out JSON
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

ART = Path('/home/dev/n16k64_campaign/sm120_bench/results_r2/artifacts')
POLICIES = {'fo6@stock_wA': ('qwen27b_fo6', 'stock'), 'tc-16x64': ('qwen27b_tc_16x64', 'mixed')}
ALTERNATIONS, REPS = 5, 7


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, choices=tuple(POLICIES))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load_path('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BM = load_path('sm120_bench_model', REPO / 'sm120' / 'bench' / 'model.py')
    art, family = POLICIES[args.policy]
    model = C.load_model('qwen27b')
    sets = dict(table=KernelSet(family), fallback=KernelSet(family, table={}))
    assert sets['table'].table_source and not sets['fallback'].table
    NM.install(model, str(ART / art), kernel=sets['table'], loader=C.MODELS['qwen27b']['loader'])
    native = NM.native_modules(model).values()
    res = dict(policy=args.policy, family=family, gpu=B.gpu_info(), blocks=[])
    for i in range(ALTERNATIONS):
        for mode in (('table', 'fallback') if i % 2 == 0 else ('fallback', 'table')):
            for m in native:
                m.kernel_set = m.kernel = sets[mode]
            r = BM.prefill(model, 1, 2048, reps=REPS)
            res['blocks'].append(dict(alternation=i, mode=mode, **r))
            print(i, mode, json.dumps(r), flush=True)
    for mode in ('table', 'fallback'):
        med = [b['ms'] for b in res['blocks'] if b['mode'] == mode]
        mins = [b['min_ms'] for b in res['blocks'] if b['mode'] == mode]
        res[mode] = dict(median_of_block_medians=statistics.median(med), median_of_block_minima=statistics.median(mins),
                         block_medians=med, block_minima=mins, calls_by_width=dict(sorted(sets[mode].stats.items())))
    pairs = [(a['ms'], b['ms']) for a, b in zip(res['blocks'][0::2], res['blocks'][1::2])]
    res['paired_table_minus_fallback_ms'] = [(t - f) if a['mode'] == 'table' else (f - t)
                                            for (t, f), a in zip(pairs, res['blocks'][0::2])]
    res['gpu_end'] = B.gpu_info()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
