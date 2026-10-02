#!/usr/bin/env python3
"""The main W4A4 perplexity table (results/main_ppl/PROTOCOL.md): one run_ppl_deploy.py process per (model, row).

    PAPER_PYTHON experiments/main_ppl/run.py [--models ...] [--rows ...]
    PAPER_PYTHON experiments/main_ppl/run.py --smoke-nemotron      # feasibility only (2 windows per corpus)

Rows (the records' labels):
  bf16                     the model as loaded
  nvfp4, fo6               native (c): the NVFP4 / FourOverSix artifact on the paper stock set ('auto_stock')
  ours-8x64                native (c): the FlipQuant 8x64 artifact on n8k64_wB (weights on B)
  ours-16x64, ours-256x64  native (c): the FlipQuant artifacts on the paper mixed set ('auto'; 256x64 as 16x64 granules)
  if4, zou                 simulated W4A4, fake (c): fake:w4a4:if4 (IF4, Cook et al.), fake:w4a4:zou (MixFP4, Zou et al.)
  fo6-fake                 the appendix reference: fake (c) FourOverSix
  if4w, zouw               amendment 1: IF4 / MixFP4 (Zou) on the weights only, FourOverSix per-token activations
                           (fake:format:<rule>:1x16, Experiment A's setting)
Models, in run order: qwen3_1p7b, qwen3_8b, mistral7b_ins, nemotron9b, phi4, qwen27b, then mistral7b (the base model,
the secondary Mistral column).
- **The four new models:** the artifacts are exported here, as step 02 does (export_map_artifact.py --ownership), from
  the map.pt the delivered ~/flipquant map records as its source. Its sha256 and tiles are checked against the delivered
  file first.
- **phi4, qwen27b, mistral7b:** the paper's step 03 records of the native and BF16 rows (PAPER_OUT/ppl) are reused
  after the reuse check (RECHECK). Those rows are re-run into <out>/ppl/<model>/<row>-recheck, and every per-window NLL
  must equal the paper record's. On a mismatch the script stops: nothing of that model is reused.
Outputs: <out>/artifacts/<model>_<kind>, <out>/ppl/<model>/<row>/report.json, <out>/commands.log, <out>/logs/.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402

OUT = Path(os.environ.get('MAIN_PPL_OUT', '/home/dev/n16k64_campaign/main_ppl'))
# the paper's SM120 builds (built at tm-opt 4b85cec on 2026-09-27; step 03 ran on them): this worktree has none
os.environ.setdefault('SM120_BUILD_DIR', '/home/dev/NVFP4-RaZeR/sm120/build')
NEW = ('qwen3_1p7b', 'qwen3_8b', 'mistral7b_ins', 'nemotron9b')
PAPER = ('phi4', 'qwen27b', 'mistral7b')
MODELS = ('qwen3_1p7b', 'qwen3_8b', 'mistral7b_ins', 'nemotron9b', 'phi4', 'qwen27b', 'mistral7b')
ROWS = ('bf16', 'nvfp4', 'fo6', 'ours-8x64', 'ours-16x64', 'ours-256x64', 'fo6-fake', 'if4', 'zou', 'if4w', 'zouw')
# amendment 1 (results/main_ppl/PROTOCOL.md): the rules on the weights only, with FourOverSix per-token activations
WEIGHT_ONLY = {'if4w': 'if4', 'zouw': 'zou'}
NATIVE = {'nvfp4': 'nvfp4', 'fo6': 'fo6', 'ours-8x64': 'tc_8x64', 'ours-16x64': 'tc_16x64', 'ours-256x64': 'tc_256x64'}
UNITS = ('8x64', '16x64', '256x64')
DATA = dict({m: '/home/dev/n16k64_campaign/fqmaps/data' for m in NEW}, **{m: P.DATA[m] for m in PAPER})
REGISTRY = {'qwen3_1p7b': 'qwen3-1.7b', 'qwen3_8b': 'qwen3-8b', 'mistral7b_ins': 'mistral-7b',
            'nemotron9b': 'nemotron-nano-9b-v2'}
DELIVERED = Path.home() / 'flipquant' / 'maps'
MAP_RUNS = Path('/home/dev/n16k64_campaign/fqmaps/runs')
PAPER_ARTIFACTS = Path('/home/dev/n16k64_campaign/paper/artifacts')
PAPER_PPL = Path('/home/dev/n16k64_campaign/paper/ppl')
RECHECK = {'phi4': ('bf16', 'fo6'), 'qwen27b': ('ours-8x64',), 'mistral7b': ('bf16', 'fo6')}
# amendment 1: Experiment A's records of the identical setting (results/paper_extra/A), compared bitwise
A_PPL = Path('/home/dev/n16k64_campaign/paper_extra/A/ppl')
A_LABEL = {'if4w': 'if4-1x16', 'zouw': 'zou-1x16', 'fo6-fake': 'fo6'}
KIND = {'nvfp4': 'nvfp4', 'fo6': 'four_over_six'}


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(model, kind):
    return (PAPER_ARTIFACTS if model in PAPER else OUT / 'artifacts') / f'{model}_{kind}'


def source_map(model, unit):
    """The run map.pt behind the delivered ~/flipquant map: sha256 and tiles checked against the delivered file."""
    path = MAP_RUNS / f'{model}_{unit}' / 'map.pt'
    obj = torch.load(DELIVERED / REGISTRY[model] / f'flipquant_{unit}.pt', map_location='cpu', weights_only=True)
    assert obj['format'] == 'flipquant-map/1' and obj['unit'] == unit, (model, unit)
    assert sha(path) == obj['meta']['source_map_sha256'], f'{path} is not the delivered map\'s source'
    raw = torch.load(path, map_location='cpu', weights_only=True)
    assert set(raw) == set(obj['tiles']) and all(torch.equal(raw[n].bool(), obj['tiles'][n]) for n in raw), (model, unit)
    return path


def export(args, model, kind):
    art = artifact(model, kind)
    if (art / 'artifact.json').exists():
        return art
    assert model in NEW, f'{art} missing (the paper exported it)'
    cmd = [P.PY, 'export_map_artifact.py', '--model', model, '--data-root', DATA[model]]
    if kind.startswith('tc_'):
        unit = kind[3:]
        cmd += ['--map', source_map(model, unit), '--unit', unit]
    else:
        cmd += ['--kind', KIND[kind]]
    rc = P.run(args.out, f'export_{model}_{kind}', [*cmd, '--ownership', '--out', art], extra_env=P.ACCURACY_ENV)
    if rc != 0 or not (art / 'artifact.json').exists():
        P.die(f'export failed: {model} {kind} (log: {args.out}/logs/export_{model}_{kind}.log)')
    return art


def spec(args, model, row):
    if row == 'bf16':
        return 'bf16=bf16'
    if row in NATIVE:
        return f'{row}=native:{export(args, model, NATIVE[row])}'
    if row == 'fo6-fake':
        return 'fo6-fake=fake:four_over_six'
    if row in WEIGHT_ONLY:
        return f'{row}=fake:format:{WEIGHT_ONLY[row]}:1x16'
    return f'{row}=fake:w4a4:{row}'


def ppl(args, model, row, out, limit=None):
    if P.complete(out / 'report.json') and not args.force:
        print(f'{model} {row}: done', flush=True)
        return
    cmd = [P.PY, 'run_ppl_deploy.py', '--model', model, '--data-root', DATA[model], *P.DEVIATION.get(model, []),
           '--evaluate', spec(args, model, row), '--out', out]
    if limit:
        cmd += ['--limit-windows', str(limit)]
    name = f'ppl_{model}_{out.name}' + (f'_limit{limit}' if limit else '')
    rc = P.run(args.out, name, cmd, extra_env=P.ACCURACY_ENV)
    if rc != 0 or not P.complete(out / 'report.json'):
        P.die(f'failed: {model} {row} (log: {args.out}/logs/{name}.log)')


def recheck(args, model, row):
    """A paper row re-run: every per-window NLL must equal the paper record's (else stop: nothing is reused)."""
    out = args.out / 'ppl' / model / f'{row}-recheck'
    ppl(args, model, row, out)
    new = json.loads((out / 'report.json').read_text())['evaluations'][row]['evaluation']
    old = json.loads((PAPER_PPL / model / row / 'report.json').read_text())['evaluations'][row]['evaluation']
    same = {c: new[c]['nll'] == old[c]['nll'] and new[c]['ppl'] == old[c]['ppl'] for c in ('wiki', 'c4')}
    P.log(args.out, f'RECHECK {model} {row} {json.dumps(same)}')
    if not all(same.values()):
        P.die(f'reuse check FAILED: {model} {row} {same}; the paper rows of {model} are not reused (tell the coordinator)')
    return same


def crosscheck_a(args, model, row):
    """Amendment 1: where Experiment A ran the identical setting (fake:format:<rule>:1x16 on the same model, windows and
    harness), its per-window NLLs must equal this run's, bitwise. Logged; a mismatch is reported, not hidden."""
    a = A_PPL / model / A_LABEL[row] / 'report.json'
    if not a.exists():
        return None
    new = json.loads((args.out / 'ppl' / model / row / 'report.json').read_text())['evaluations'][row]['evaluation']
    old = json.loads(a.read_text())['evaluations'][A_LABEL[row]]['evaluation']
    same = {c: new[c]['nll'] == old[c]['nll'] and new[c]['ppl'] == old[c]['ppl'] for c in ('wiki', 'c4')}
    P.log(args.out, f'CROSSCHECK-A {model} {row} vs {a} {json.dumps(same)}')
    return same


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--rows', default=','.join(ROWS))
    ap.add_argument('--smoke-nemotron', action='store_true', help='feasibility: export + 2 windows per corpus')
    args = ap.parse_args()
    args.models = args.models if args.models != ','.join(P.MODELS) else ','.join(MODELS)
    args.out = args.out or OUT
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    if args.smoke_nemotron:
        for row in ('bf16', 'fo6', 'ours-8x64', 'if4'):
            ppl(args, 'nemotron9b', row, args.out / 'smoke' / 'nemotron9b' / row, limit=2)
        return
    rows = [r for r in ROWS if r in args.rows.split(',')]
    for model in [m for m in MODELS if m in args.models.split(',')]:
        for row in rows:
            if model in PAPER and (row in NATIVE or row == 'bf16'):
                if row in RECHECK[model]:
                    recheck(args, model, row)
                continue                      # reused: PAPER_PPL/<model>/<row>/report.json
            ppl(args, model, row, args.out / 'ppl' / model / row)
            if row in A_LABEL:
                crosscheck_a(args, model, row)


if __name__ == '__main__':
    main()
