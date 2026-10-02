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

Amendment 2 (G5'): --unit selects the TC artifact (8x64, 16x64, 256x64), and --before / --after the kernels, each a build
name (from sm120/build, or --before-root / --after-root) or 'set:<family>'. Defaults reproduce the gate above.

Amendment 4 (tile-table re-tune): --unit fo6 selects the FourOverSix artifact (for the stock set), and --before-table /
--after-table give a set its tile table (default: sm120/configs/<gpu>.json), so the same builds can be compared under
two tables.

Amendment 14 (the 8x64 plan's P6): --after 'auto:<name>' installs with kernel='<name>' (e.g. 'auto'), so the kernel is
resolved per artifact by model.resolve_kernel, as deployed, from the default build directory ($SM120_BUILD_DIR) and
table. --expect-family requires the resolved kernel to be that set, every library of it in $SM120_BUILD_DIR. The routing
note, the set, its table and its libraries' extra defines are recorded per model.
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet, key_order  # noqa: E402

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
    ap.add_argument('--after-root', default=None, help="required unless --after is 'auto:<name>'")
    ap.add_argument('--unit', default='8x64', choices=('8x64', '16x64', '256x64', 'fo6'))
    ap.add_argument('--before', default='n8k64_wB', help="build name or 'set:<family>' (sm120/build unless --before-root)")
    ap.add_argument('--before-root', default=None)
    ap.add_argument('--after', default='set:mixed_wB', help="build name or 'set:<family>' from --after-root")
    ap.add_argument('--before-table', default=None, help="tile table of a 'set:' before (default: the GPU's table)")
    ap.add_argument('--after-table', default=None, help="tile table of the 'set:' after (default: the GPU's table)")
    ap.add_argument('--expect-family', default=None, help="with --after auto:<name>: the set the routing must resolve to")
    ap.add_argument('--models', default='llama8b,mistral7b,phi4,qwen27b')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    def resolve(spec, root, table=None):
        if spec.startswith('set:'):
            return KernelSet(spec[4:], build_root=root, table=table)
        assert table is None, 'a table needs a set'
        return Kernel.load(spec, build_root=root)
    before = resolve(args.before, args.before_root, args.before_table)
    for k in (before.kernels.values() if isinstance(before, KernelSet) else [before]):
        want = Path(args.before_root) if args.before_root else REPO / 'sm120' / 'build'
        assert Path(k.path).parent.parent == want, k.path
    routed = args.after.startswith('auto:')
    if not routed and args.after_root is None:
        ap.error('--after-root is required')
    if routed:
        assert args.after_root is None and args.after_table is None, 'auto:<name> resolves from the default build dir and table'
        after = args.after[5:]
    else:
        after = resolve(args.after, args.after_root, args.after_table)
        if not isinstance(after, KernelSet):
            raise SystemExit('--after must be a set (its widths are recorded)')
    desc = lambda k: k.describe() if isinstance(k, KernelSet) else dict(kernel=k.cfg.name, sha256=k.sha256)  # noqa: E731
    res = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               gpu=B.gpu_info(), shapes=list(SHAPES), unit=args.unit, before=dict(spec=args.before, root=args.before_root,
               table=args.before_table, describe=desc(before)),
               after=dict(spec=args.after, build_root=args.after_root, table=args.after_table,
                          describe=None if routed else after.describe(), expect_family=args.expect_family,
                          sm120_build_dir=os.environ.get('SM120_BUILD_DIR')),
               models={})
    ok = True
    for model_key in args.models.split(','):
        art = ARTIFACTS / (f'{model_key}_fo6' if args.unit == 'fo6' else f'{model_key}_tc_{args.unit}')
        model = C.load_model(model_key)
        rec = res['models'][model_key] = dict(artifact=str(art), shapes={})
        logits = {}
        for label, kern in (('before', before), ('after', after)):
            for ks in (before, after):
                if isinstance(ks, KernelSet):
                    ks.stats.clear()
            rep = NM.install(model, art, kernel=kern, loader=C.MODELS[model_key]['loader'])
            assert not rep.fallback, rep.fallback
            nat = NM.native_modules(model)
            for m in nat.values():
                m.calls = 0
            logits[label] = forwards(model)
            called = sum(1 for m in nat.values() if m.calls > 0)
            rec[label] = dict(install=dict(kernel=rep.kernel, kernel_sha256=rep.kernel_sha256, native_modules=len(rep.native)),
                              native_called=called, native=len(nat))
            if label == 'after' and routed:
                meta = json.loads((art / 'artifact.json').read_text())
                ks, note = NM.resolve_kernel(after, meta)
                root = Path(os.environ.get('SM120_BUILD_DIR') or REPO / 'sm120' / 'build')
                libs = ks.kernels.values() if isinstance(ks, KernelSet) else [ks]
                fam = ks.family if isinstance(ks, KernelSet) else None
                rec['routing'] = dict(note=rep.routing, family=fam, kernel_set=rep.kernel_set,
                                      roots=sorted({str(Path(k.path).parent.parent) for k in libs}),
                                      extra_defines={k.cfg.name: k.manifest.get('extra_defines') for k in libs})
                ok &= rep.routing == note and (args.expect_family is None or fam == args.expect_family) \
                    and all(Path(k.path).parent.parent == root for k in libs)
            if isinstance(kern, KernelSet):
                rec[label]['calls_by_width'] = {str(w): c for w, c in sorted(kern.stats.items(), key=lambda i: key_order(i[0]))}
            ok &= called == len(nat) == len(rep.native)
        for s in SHAPES:
            a, b = logits['before'][s], logits['after'][s]
            eq = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            rec['shapes'][s] = dict(equal=eq, max_abs_diff=None if eq else (a.float() - b.float()).abs().max().item(),
                                    shape=list(a.shape))
            ok &= eq
        print(model_key, {s: v['equal'] for s, v in rec['shapes'].items()}, rec['before'].get('calls_by_width'),
              rec['after'].get('calls_by_width'), flush=True)
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
