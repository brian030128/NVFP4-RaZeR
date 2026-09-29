#!/usr/bin/env python3
"""Experiment A: IF4 (Cook et al.) and MixFP4 (Zou et al.) per-block selection, coarsened to hardware tiles.

    PAPER_PYTHON experiments/paper_extra/A_formats.py --models llama8b [--policies ...] [--force]

Protocol: results/paper_extra/A/PROTOCOL.md. One fake (c) simulator for every policy (run_ppl_deploy.py, one process
per (model, policy)), WikiText-2 and C4 on the released protocol windows:
  bf16                     the model as loaded
  nvfp4, fo6               fake:nvfp4 (NVFP4 per-token activations), fake:four_over_six (FourOverSix activations)
  e2m1, e2m1z              the E2M1 bases of IF4 and of Zou (NVFP4 weights: each rule's own E2M1 candidate everywhere)
                           with FourOverSix activations: the rules' own bases for the gain comparison
  tc-8x64/16x64/256x64     the committed TM-OPT+TC maps (fake:map:<the paper run's .mixfp4map>)
  if4-<unit>, zou-<unit>   fake:format:<rule>:<unit>, unit in 1x16 (the original per-block rule), 8x64, 16x64, 256x64;
                           weights only; FourOverSix per-token activations (convention (c)), like every paper number
  zoufo6-<unit>            the user-requested arm: Zou's rule with FourOverSix E2M1 as the E2M1 candidate (the base of
                           our maps); OURS vs zoufo6 isolates the selection method, zoufo6 vs zou the FourOverSix part
Outputs: <A_OUT>/ppl/<model>/<policy>/report.json (A_OUT: PAPER_EXTRA_OUT/A, default
/home/dev/n16k64_campaign/paper_extra/A); commands in <A_OUT>/commands.log, logs in <A_OUT>/logs.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402

OUT = Path(os.environ.get('PAPER_EXTRA_OUT', '/home/dev/n16k64_campaign/paper_extra')) / 'A'
UNITS = ('1x16', '8x64', '16x64', '256x64')
POLICIES = (['bf16', 'nvfp4', 'fo6', 'e2m1', 'e2m1z'] + [f'tc-{u}' for u in P.UNITS] +
            [f'{r}-{u}' for r in ('if4', 'zou', 'zoufo6') for u in UNITS])


def spec(model, pol):
    if pol == 'bf16':
        return 'bf16=bf16'
    if pol == 'nvfp4':
        return 'nvfp4=fake:nvfp4'
    if pol == 'fo6':
        return 'fo6=fake:four_over_six'
    if pol == 'e2m1':
        return 'e2m1=fake:format:e2m1:1x16'
    if pol == 'e2m1z':
        return 'e2m1z=fake:format:e2m1zou:1x16'
    if pol.startswith('tc-'):
        m = P.artifact(P.OUT_DEFAULT, model, f'tc_{pol[3:]}').with_name(f'{model}_tc_{pol[3:]}.mixfp4map')
        if not m.exists():
            P.die(f'{m} missing (the paper run exports it)')
        return f'{pol}=fake:map:{m}'
    rule, unit = pol.split('-')
    return f'{pol}=fake:format:{rule}:{unit}'


def main():
    ap = P.parser(__doc__)
    args = ap.parse_args()
    args.out = args.out or (P.SMOKE_OUT / 'extra_A' if args.smoke else OUT)
    args = P.setup(args)
    names = [p for p in POLICIES if not args.policies or p in args.policies.split(',')]
    for model in args.models:
        for pol in names:
            out = args.out / 'ppl' / model / pol
            if P.complete(out / 'report.json') and not args.force:
                print(f'{model} {pol}: done')
                continue
            cmd = [P.PY, 'run_ppl_deploy.py', '--model', model, '--data-root', P.DATA[model], *P.DEVIATION.get(model, []),
                   '--evaluate', spec(model, pol), '--out', out]
            if args.smoke:
                cmd += ['--limit-windows', '4']
            rc = P.run(args.out, f'A_{model}_{pol}', cmd, extra_env=P.ACCURACY_ENV)
            if rc != 0 or not P.complete(out / 'report.json'):
                P.die(f'A failed: {model} {pol} (log: {args.out}/logs/A_{model}_{pol}.log)')


if __name__ == '__main__':
    main()
