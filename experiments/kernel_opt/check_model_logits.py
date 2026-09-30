#!/usr/bin/env python3
"""Kernel-opt gate G5: whole-model logits before vs after, bitwise.

    python experiments/kernel_opt/check_model_logits.py --after-root /home/dev/n16k64_campaign/kernel_opt/build \
        --models llama8b,mistral7b,phi4,qwen27b --out JSON

Protocol: results/kernel_opt/PROTOCOL.md (G5). Per model:
- the model is loaded once (sm120/eval/common.load_model: BF16, SDPA);
- its TC 8x64 artifact is installed (sm120/mixfp4_sm120/model.install) on n8k64_wB from sm120/build ("before"), eager
  forwards run on seeded random tokens (seed 0, as sm120/bench/model.py prefill) at SHAPES (batch x prompt) and the
  logits are kept;
- the same artifact is installed again on the width-selecting 'mixed_wB' KernelSet from --after-root ("after") and the
  same forwards run;
- rule: the logits are equal bitwise at every shape, and every scoped Linear is a NativeLinear and ran in both.
The widths the set used per shape are recorded.
"""
import argparse
import datetime
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

ARTIFACTS = Path('/home/dev/n16k64_campaign/paper/artifacts')
SHAPES = ('1x1', '1x16', '1x100', '1x2048', '4x512')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def forwards(model):
    out = {}
    for s in SHAPES:
        b, p = (int(v) for v in s.split('x'))
        ids = torch.randint(100, 20000, (b, p), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
        out[s] = model(input_ids=ids, use_cache=True).logits.clone()
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--models', default='llama8b,mistral7b,phi4,qwen27b')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    before = Kernel.load('n8k64_wB')
    assert Path(before.path).parent.parent == REPO / 'sm120' / 'build', before.path
    after = KernelSet('mixed_wB', build_root=args.after_root)
    res = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               gpu=B.gpu_info(), shapes=list(SHAPES), before=dict(kernel='n8k64_wB', sha256=before.sha256),
               after=dict(kernel_set='mixed_wB', build_root=args.after_root, describe=after.describe()), models={})
    ok = True
    for model_key in args.models.split(','):
        art = ARTIFACTS / f'{model_key}_tc_8x64'
        model = C.load_model(model_key)
        rec = res['models'][model_key] = dict(artifact=str(art), shapes={})
        logits = {}
        for label, kern in (('before', before), ('after', after)):
            after.stats.clear()
            rep = NM.install(model, art, kernel=kern, loader=C.MODELS[model_key]['loader'])
            assert not rep.fallback, rep.fallback
            nat = NM.native_modules(model)
            for m in nat.values():
                m.calls = 0
            logits[label] = forwards(model)
            called = sum(1 for m in nat.values() if m.calls > 0)
            rec[label] = dict(install=dict(kernel=rep.kernel, kernel_sha256=rep.kernel_sha256, native_modules=len(rep.native)),
                              native_called=called, native=len(nat))
            if label == 'after':
                rec['after']['calls_by_width'] = dict(sorted(after.stats.items()))
            ok &= called == len(nat) == len(rep.native)
        for s in SHAPES:
            a, b = logits['before'][s], logits['after'][s]
            eq = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            rec['shapes'][s] = dict(equal=eq, max_abs_diff=None if eq else (a.float() - b.float()).abs().max().item(),
                                    shape=list(a.shape))
            ok &= eq
        print(model_key, {s: v['equal'] for s, v in rec['shapes'].items()}, rec['after'].get('calls_by_width'), flush=True)
        del model, logits
        torch.cuda.empty_cache()
        B.write(args.out, res)
    res['passed'] = bool(ok)
    res['status'] = 'complete'
    res['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    B.write(args.out, res)
    print('G5', 'PASSED' if ok else 'FAILED')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
