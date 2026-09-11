"""
    Write one of MIXFP4_REPORT.md's policies out as an ordinary BF16 checkpoint.

    This exists for one reason: Terminal-Bench is unaffordable otherwise. Serving the policy
    faithfully -- weights and the W4A4 activation hooks, in-process through HF generate -- was
    measured at roughly three hours per Terminal-Bench trial for a 4B model, so a three-policy
    comparison over fifteen tasks would be about 135 GPU-hours. vLLM is an order of magnitude
    faster but cannot run the hooks.

    So the trade is explicit and recorded: **the exported checkpoint is W4A16, not the W4A4 the
    report measures.** The element-type election survives, because it is a rewrite of the weights
    and bakes in; the activation quantizer does not. The provenance file next to the checkpoint
    says `precision: W4A16`, and any result obtained from it has to say the same. It is a
    measurement of the weight half of the policy, which is the half the element-type decision is
    about, and it is not a measurement of the reported configuration.

        python export_kse_checkpoint.py --model qwen4b --calib <dir> --policy k3 --out <dir>
"""

import argparse
import json
import os
from pathlib import Path

from serve_quantized import POLICIES, build
from run_kse_paper import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--policy', choices=POLICIES, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-map-drift', action='store_true')
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm -- see /home/u4320956/CLAUDE.md'

    model, tok, prov = build(args.model, args.calib, args.policy, args.stage_root,
                             args.allow_map_drift, activation_hooks=False)
    assert prov['precision'] in ('BF16', 'W4A16'), prov

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    prov['exported_to'] = str(out)
    prov['caveat'] = ('activation quantization is NOT present in this checkpoint; the reported '
                      'policy is W4A4 and this is the weight half only')
    (out / 'quantization_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print('PROVENANCE ' + json.dumps(prov), flush=True)
    print(f'wrote {args.policy} ({prov["precision"]}) to {out}', flush=True)


if __name__ == '__main__':
    main()
