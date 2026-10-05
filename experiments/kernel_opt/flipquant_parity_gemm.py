#!/usr/bin/env python3
"""flipquant GEMM parity: the GEMM time of flipquant's real path against NVFP4-RaZeR kernel-opt's adopted kernels
(results/kernel_opt/flipquant_parity/NOTE.md). A check asked by the user, not a registered study.

    python experiments/kernel_opt/flipquant_parity_gemm.py --model llama8b --fq-root /home/dev/n16k64_campaign/fqport/wt \
        --rz-build /home/dev/n16k64_campaign/kernel_opt/build_V --out JSON

Two worker processes hold the two sides, and this driver (no CUDA context of its own) alternates them cell by cell.
- fq: flipquant, from a clean worktree at --fq-root. Nothing there is modified (no bytecode, no files). For each
  policy, its own CLI plumbing: models.cli.build (models.loader.load, then models.razer.install) with
  `--mode native --native-backend razer --razer-build-dir <fq-root>/kernels/razer_sm120/build_ko --razer-kernel auto`
  (`auto_stock_wB` for stock_wB_ko) and the paper map (`--map <maps>/<model>_<unit>/map.pt --unit <unit>`). The timed
  modules are the NativeLinears that install put into the model.
- rz: NVFP4-RaZeR kernel-opt, the deviation-2 harness's construction (bench_ab_isolated.run, M1): a NativeLinear from
  the paper artifact's packed weights on the set that mixfp4_sm120.model.resolve_kernel gives ('auto' for the maps,
  'auto_stock' / 'auto_stock_wB' for FourOverSix), from the deployment directory --rz-build (SM120_BUILD_DIR).
Policies (POLICIES): the 8x64, 16x64 and 256x64 TM-OPT+TC maps (mixed_wB_ko, mixed_ko, mixed256_ko), and FourOverSix
weights on stock_ko and stock_wB_ko; activations FourOverSix per token. Modules: per projection, M1's choice
(bench_gemm_isolated.tag_modules on the rz artifact): a map's typical module (the lower median of the E0M3 tiles), and
FourOverSix's first module. Both sides time the same module names.
Timing: M1's deviation-2 repetition, the same code on both sides. Cold weights: each launch uses the next of K copies of
the module's packed weights and placed scales (K x size > 4x the L2), after a 512 MiB read-flush; a fresh activation
from a seeded pool of two; the activation quantizer (Kernel.quant_rows of the picked build), synchronized; then the GEMM
with its fused epilogue (Kernel.gemm_ptr with the set's width and scheduler row, as NativeLinear calls it). The value is
the CUPTI device time of the GEMM, the median of all launches; the quantizer's device time and a CUDA-event time are
recorded too. Per (projection, T): 3 rounds of --reps launches after --warmup; in round r the policies start at position
r * len / rounds, and each policy's two sides run back to back, fq first when r + (the policy's index) is even.
Checks (a failure stops the run):
- the two sides' packages are their own (flipquant's vendored mixfp4_sm120, NVFP4-RaZeR's sm120), each install
  resolves to the expected set, and the timed modules' operands (packed codes, placed scales, global scale, bias) are
  equal bit for bit;
- per (policy, projection, T), on both sides: the isolated path equals the module's fused forward (NativeLinear.forward,
  the real path) bit for bit, and the two sides' outputs are equal bit for bit; the build, width, scheduler row, build
  defines and the manifests' SASS hashes are recorded per side (the analysis compares them);
- every profiled block holds exactly --reps GEMM and --reps quantizer launches; no third process on the GPU.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

RAZER = Path(__file__).resolve().parents[2]
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
MAPS = Path('/home/dev/n16k64_campaign/paper/maps')
# NVFP4-RaZeR model key -> flipquant registry key and the checkpoint revision of the paper artifacts
MODELS = {'llama8b': dict(fq='llama3.1-8b', revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'),
          'phi4': dict(fq='phi4-14b', revision='2db69c1c3e91a05d2c64a3185acfbaf36f744e25')}
ACT = 'four_over_six_rows'
# policy -> weights, map unit, flipquant route (--razer-kernel), NVFP4-RaZeR route (resolve_kernel), artifact suffix,
# module choice, and the set both routes must give
POLICIES = OrderedDict([
    ('8x64', dict(weight='mixfp4', unit='8x64', fq='auto', rz='auto', art='tc_8x64', tags='typical', family='mixed_wB_ko')),
    ('16x64', dict(weight='mixfp4', unit='16x64', fq='auto', rz='auto', art='tc_16x64', tags='typical', family='mixed_ko')),
    ('256x64', dict(weight='mixfp4', unit='256x64', fq='auto', rz='auto', art='tc_256x64', tags='typical',
                    family='mixed256_ko')),
    ('stock_ko', dict(weight='fourover6', unit=None, fq='auto', rz='auto_stock', art='fo6', tags='first', family='stock_ko')),
    ('stock_wB_ko', dict(weight='fourover6', unit=None, fq='auto_stock_wB', rz='auto_stock_wB', art='fo6', tags='first',
                         family='stock_wB_ko')),
])
TOKENS = (1, 16, 128, 512, 2048, 8192)
IMPL = dict(fq="flipquant's real path (models.cli.build -> models.razer.install)",
            rz='NVFP4-RaZeR kernel-opt, the deviation-2 harness (NativeLinear from the artifact)',
            rzm="NVFP4-RaZeR's own model path (eval/common.load_model -> mixfp4_sm120.model.install)")


def classify(name):
    """bench_gemm_isolated.classify (and flipquant evaluation/latency.classify), copied so the fq worker imports nothing
    from NVFP4-RaZeR."""
    n = name.lower()
    if 'quant_rows_kernel' in n:
        return 'quant'
    if 'cutlass' in n or 'device_kernel' in n or 'gemm' in n:
        return 'gemm'
    return 'other'


# ------------------------------------------------------------------------------------------------------------- worker

class Worker:
    """One side: its NativeLinears per (policy, projection), their rotation copies, and the timing of M1's repetition."""

    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.side = args.side
        torch.backends.cuda.matmul.allow_tf32 = False
        self.lins, self.copies, self.counter, self.info = {}, {}, {}, {}
        self.pool_key, self.pool, self.x = None, None, None

    # ---- setup
    def setup(self, msg):
        torch = self.torch
        if self.side == 'rz':
            out = self.setup_rz()
        elif self.side == 'rzm':
            out = self.setup_rzm()
        else:
            out = self.setup_fq(msg['modules'])
        import mixfp4_sm120
        from mixfp4_sm120.lib import Kernel
        self.Kernel = Kernel
        out['package'] = str(Path(mixfp4_sm120.__file__).resolve().parent)
        out['python'] = sys.executable
        out['torch'] = torch.__version__
        out['device'] = torch.cuda.get_device_name(0)
        self.l2 = torch.cuda.get_device_properties(0).L2_cache_size
        out['l2_bytes'] = self.l2
        self.flush = torch.ones(self.args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
        self.stream = torch.cuda.current_stream()
        operands, copies = {}, {}
        for key, lin in self.lins.items():
            lin.share_input = False
            size = lin.packed.numel() + lin.sf.numel()
            self.copies[key] = [(lin.packed.clone(), lin.sf.clone()) for _ in range(math.ceil(4 * self.l2 / size) + 1)]
            self.counter[key] = 0
            copies['/'.join(key)] = len(self.copies[key])
            operands['/'.join(key)] = dict(
                module=lin.name, shape=[lin.out_features, lin.in_features], e0m3_tiles=int(lin.e0m3_tiles),
                packed_sha256=self.tsha(lin.packed), sf_sha256=self.tsha(lin.sf), global_scale=repr(lin.global_scale),
                bias_sha256=None if lin.bias_bf16 is None else self.tsha(lin.bias_bf16),
                kernel_set=lin.kernel_set.family if lin.kernel_set is not None else lin.kernel.cfg.name)
        out['operands'], out['rotation_copies'] = operands, copies
        out['gpu_memory_allocated_gib'] = torch.cuda.memory_allocated() / 2 ** 30
        return out

    def tsha(self, t):
        return hashlib.sha256(t.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def setup_rz(self):
        torch = self.torch
        for p in (RAZER / 'sm120' / 'bench', RAZER / 'experiments' / 'paper', RAZER / 'sm120'):
            sys.path.insert(0, str(p))
        import bench_gemm_isolated as G
        import common as B
        from mixfp4_sm120 import artifact as A
        from mixfp4_sm120 import model as RM
        from mixfp4_sm120.linear import NativeLinear
        from mixfp4_sm120.select import KernelSet
        out = dict(gpu=B.gpu_info(), build_dir=os.environ.get('SM120_BUILD_DIR'), routing={}, kernel_sets={}, modules={},
                   tags={})
        loaded = {}
        for pol, cfg in POLICIES.items():
            if pol not in self.args.policies:
                continue
            if cfg['art'] not in loaded:
                loaded = {cfg['art']: A.load(ART / f"{self.args.model}_{cfg['art']}", device='cpu')}
            meta, weights = loaded[cfg['art']]
            tags = G.tag_modules(meta, weights)
            kern, note = RM.resolve_kernel(cfg['rz'], meta)
            family = kern.family if isinstance(kern, KernelSet) else kern.cfg.name
            if family != cfg['family']:
                raise SystemExit(f"rz {pol}: resolve_kernel({cfg['rz']!r}) gave {family}, expected {cfg['family']} ({note})")
            out['routing'][pol] = dict(route=cfg['rz'], set=family, note=note)
            out['kernel_sets'][pol] = kern.describe() if isinstance(kern, KernelSet) else dict(kernel=kern.cfg.name)
            out['modules'][pol], out['tags'][pol] = {}, {}
            for proj, t in tags.items():
                if self.args.projections and proj not in self.args.projections:
                    continue
                name, e0 = t[cfg['tags']]
                w = weights[name]
                w = dataclasses.replace(w, packed=w.packed.cuda(), scales=w.scales.cuda(),
                                        bias=None if w.bias is None else w.bias.cuda())
                self.lins[(pol, proj)] = NativeLinear(w, kern, ACT, name=name)
                out['modules'][pol][proj] = name
                out['tags'][pol][proj] = dict(module=name, e0m3_tiles=e0, modules=t['modules'], shape=t['shape'],
                                              tiles_per_module=t['tiles_per_module'], choice=cfg['tags'])
        del loaded
        gc.collect()
        return out

    def setup_rzm(self):
        """NVFP4-RaZeR's own deployment path, as its evaluations run it: per policy the model is loaded
        (sm120/eval/common.load_model, transformers) and mixfp4_sm120.model.install puts NativeLinears from the paper
        artifact into it, routed as the harness routes ('auto', 'auto_stock', 'auto_stock_wB'); the timed modules are
        the installed NativeLinears of M1's module choice."""
        import importlib.util
        torch = self.torch
        for p in (RAZER / 'experiments' / 'paper', RAZER / 'sm120'):
            sys.path.insert(0, str(p))

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        EC = load('razer_eval_common', RAZER / 'sm120' / 'eval' / 'common.py')
        import bench_gemm_isolated as G
        import common as B                           # sm120/bench/common.py (bench_gemm_isolated put it on the path)
        from mixfp4_sm120 import artifact as A
        from mixfp4_sm120 import model as RM
        from mixfp4_sm120.linear import NativeLinear
        out = dict(gpu=B.gpu_info(), build_dir=os.environ.get('SM120_BUILD_DIR'), routing={}, kernel_sets={}, modules={},
                   tags={}, install={})
        for pol, cfg in POLICIES.items():
            if pol not in self.args.policies:
                continue
            art = ART / f"{self.args.model}_{cfg['art']}"
            meta, weights = A.load(art, device='cpu')
            tags = G.tag_modules(meta, weights)
            del weights
            t0 = time.time()
            model = EC.load_model(self.args.model)
            rep = RM.install(model, art, kernel=cfg['rz'], loader=EC.MODELS[self.args.model]['loader'])
            family = (rep.kernel_set or {}).get('family')
            if family != cfg['family']:
                raise SystemExit(f"rzm {pol}: install routed {cfg['rz']!r} to {family}, expected {cfg['family']} "
                                 f"({rep.routing})")
            mods = dict(model.named_modules())
            out['routing'][pol] = dict(route=cfg['rz'], set=family, note=rep.routing)
            out['kernel_sets'][pol] = rep.kernel_set
            out['install'][pol] = dict(modules=len(rep.native), e0m3_tiles=rep.e0m3_tiles, seconds=round(time.time() - t0, 1),
                                       activation_quantizer=rep.activation_quantizer)
            out['modules'][pol], out['tags'][pol] = {}, {}
            for proj, t in tags.items():
                if self.args.projections and proj not in self.args.projections:
                    continue
                name, e0 = t[cfg['tags']]
                lin = mods[name]
                if not isinstance(lin, NativeLinear) or lin.act_kind != ACT:
                    raise SystemExit(f'rzm {pol}: {name} is not a FourOverSix NativeLinear')
                self.lins[(pol, proj)] = lin
                out['modules'][pol][proj] = name
                out['tags'][pol][proj] = dict(module=name, e0m3_tiles=e0, modules=t['modules'], shape=t['shape'],
                                              tiles_per_module=t['tiles_per_module'], choice=cfg['tags'])
            del model, mods, rep
            gc.collect()
            torch.cuda.empty_cache()
        return out

    def setup_fq(self, modules):
        torch = self.torch
        fq = Path(self.args.fq_root).resolve()
        sys.path.insert(0, str(fq))
        os.chdir(fq)
        from models import cli as FC
        from models import razer as RZ
        out = dict(fq_root=str(fq), build_dir=str(self.args.fq_build), routing={}, kernel_sets={}, argv={}, install={})
        spec = MODELS[self.args.model]
        for pol, cfg in POLICIES.items():
            if pol not in self.args.policies:
                continue
            argv = ['--model', spec['fq'], '--revision', spec['revision'], '--mode', 'native', '--native-backend', 'razer',
                    '--razer-build-dir', str(self.args.fq_build), '--razer-kernel', cfg['fq'], '--weight', cfg['weight']]
            if cfg['weight'] == 'mixfp4':
                argv += ['--map', str(MAPS / f"{self.args.model}_{cfg['unit']}" / 'map.pt'), '--unit', cfg['unit']]
            ap = argparse.ArgumentParser()
            FC.add_model_args(ap)
            a = ap.parse_args(argv)
            t0 = time.time()
            model, tok, mspec, layers, info = FC.build(a)
            rep = info['razer']
            if rep['kernel'] != cfg['family']:
                raise SystemExit(f"fq {pol}: --razer-kernel {cfg['fq']} resolved to {rep['kernel']!r}, expected "
                                 f"{cfg['family']} ({rep['routing']})")
            sets = {id(m.kernel) for m in layers.values()}
            if len(sets) != 1 or len(layers) != rep['modules']:
                raise SystemExit(f'fq {pol}: {len(sets)} kernel objects over {len(layers)} NativeLinears')
            out['argv'][pol] = argv
            out['routing'][pol] = dict(route=cfg['fq'], set=rep['kernel'], note=rep['routing'])
            out['kernel_sets'][pol] = rep['kernel_set']
            out['install'][pol] = dict(modules=rep['modules'], e0m3_tiles=rep['e0m3_tiles'], checked=rep['checked'],
                                       activation_quantizer=rep['activation_quantizer'], unit=rep['unit'],
                                       seconds=round(time.time() - t0, 1), source=info.get('source'))
            for proj, name in modules[pol].items():
                lin = layers[name]
                if not isinstance(lin, RZ.razer().linear.NativeLinear) or lin.act_kind != ACT:
                    raise SystemExit(f'fq {pol}: {name} is not a FourOverSix NativeLinear')
                self.lins[(pol, proj)] = lin
            del model, tok, layers, info
            gc.collect()
            torch.cuda.empty_cache()
        return out

    # ---- device-memory placement (the allocator's segments)
    def segments(self, msg=None):
        """The caching allocator's segments that hold the timed buffers: per buffer kind, how many segments and their
        sizes (GiB); and the process's segments."""
        snap = self.torch.cuda.memory_snapshot()
        segs = sorted((s['address'], s['total_size']) for s in snap)

        def seg_of(t):
            a = t.data_ptr()
            for base, size in segs:
                if base <= a < base + size:
                    return base, size
            return None
        kinds = dict(copies=[t for v in self.copies.values() for pair in v for t in pair],
                     modules=[b for lin in self.lins.values() for b in (lin.packed, lin.sf)],
                     flush=[self.flush])
        out = {}
        for kind, ts in kinds.items():
            held = {seg_of(t) for t in ts}
            out[kind] = dict(segments=len(held), largest_gib=max(s for _, s in held) / 2 ** 30,
                             total_gib=sum(s for _, s in held) / 2 ** 30)
        out['process'] = dict(segments=len(segs), largest_gib=max(s for _, s in segs) / 2 ** 30,
                              reserved_gib=self.torch.cuda.memory_reserved() / 2 ** 30)
        return out

    def rehome(self, msg=None):
        """Every buffer this worker times (the modules' packed weights, placed scales and bias, the rotation copies, the
        flush buffer) goes to host memory, the allocator returns its cached segments to the driver, and the buffers
        come back in a fixed order: per (policy, projection) the module's buffers, then its copies. Afterwards the
        buffers sit in fresh segments, as in a process that never held anything else."""
        torch = self.torch
        host = {}
        for key, lin in self.lins.items():
            host[key] = dict(packed=lin.packed.cpu(), sf=lin.sf.cpu(),
                             bias=None if lin.bias_bf16 is None else lin.bias_bf16.cpu(), copies=len(self.copies[key]))
            lin.packed = lin.sf = None
            if lin.bias_bf16 is not None:
                lin.bias_bf16 = None
        self.copies, self.flush = {}, None
        self.pool_key = self.pool = self.x = None
        for lin in self.lins.values():                 # the kernels' cached workspaces are reallocated on first use
            for k in (lin.kernel_set.kernels.values() if lin.kernel_set is not None else [lin.kernel]):
                k._ws.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.flush = torch.ones(self.args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
        for key, lin in self.lins.items():
            h = host[key]
            lin.packed, lin.sf = h['packed'].cuda(), h['sf'].cuda()
            if h['bias'] is not None:
                lin.bias_bf16 = h['bias'].cuda()
            self.copies[key] = [(lin.packed.clone(), lin.sf.clone()) for _ in range(h['copies'])]
            self.counter[key] = 0
        return self.segments()

    def warm(self, msg):
        """The opposite of rehome: the allocator first caches one segment of msg['gib'] GiB (transformers'
        caching_allocator_warmup does this before it loads a model), then the rotation copies and the activation
        buffers are allocated again, so that they are carved out of it."""
        torch = self.torch
        big = torch.empty(int(msg['gib'] * 2 ** 30), dtype=torch.uint8, device='cuda')
        del big
        self.copies = {k: [(p.clone(), s.clone()) for p, s in v] for k, v in self.copies.items()}
        self.pool_key = self.pool = self.x = None
        gc.collect()
        return self.segments()

    # ---- per (policy, projection, T)
    def pools(self, n, k, t):
        torch = self.torch
        if self.pool_key != (n, k, t):
            self.pool = self.x = None
            g = torch.Generator('cpu').manual_seed(n + k + t)            # M1's seed and pool of two
            self.pool = [torch.randn(t, k, generator=g).to('cuda', torch.bfloat16) for _ in range(2)]
            self.x = torch.empty_like(self.pool[0])
            self.pool_key = (n, k, t)
        return self.pool, self.x

    def call(self, key, t):
        lin = self.lins[key]
        n, k = lin.out_features, lin.in_features
        kset = lin.kernel_set
        kern = kset.pick(n, k, t) if kset is not None else lin.kernel
        sched = kset.schedule(n, k, t) if kset is not None else (0, 1)
        width = kset.width(n, k, t) if kset is not None else None
        return lin, n, k, kern, sched, width

    def gemm(self, lin, kern, sched, q, wp, wsf, y, n, k, t):
        xp, xsf, gs = q
        bptr = None if lin.bias_bf16 is None else lin.bias_bf16.data_ptr()
        if lin.weights_on_a:
            kern.gemm_ptr(wp.data_ptr(), wsf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k, None, lin.global_scale,
                          gs.data_ptr(), 1.0, bptr, y, self.stream.cuda_stream, schedule=sched)
        else:
            kern.gemm_ptr(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), t, n, k, gs.data_ptr(), 1.0, None,
                          lin.global_scale, bptr, y, self.stream.cuda_stream, schedule=sched)
        return y

    def check(self, msg):
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile
        key, t = (msg['policy'], msg['proj']), msg['t']
        lin, n, k, kern, sched, width = self.call(key, t)
        pool, x = self.pools(n, k, t)
        x.copy_(pool[0])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ref = lin(x)                                                       # the real path: the fused forward
            torch.cuda.synchronize()
        fwd = {'gemm': set(), 'quant': set()}
        for e in prof.events():
            if e.device_type.name == 'CUDA' and classify(e.name) in fwd:
                fwd[classify(e.name)].add(e.name)
        ref = ref.clone()
        y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
        got = self.gemm(lin, kern, sched, kern.quant_rows(x, lin.act_kind), lin.packed, lin.sf, y, n, k, t).clone()
        torch.cuda.synchronize()
        m = kern.manifest
        return dict(kernel=kern.cfg.name, width=width, schedule=list(sched),
                    set=lin.kernel_set.family if lin.kernel_set is not None else None,
                    table=lin.kernel_set.table_source if lin.kernel_set is not None else None,
                    library=str(kern.path), library_sha256=kern.sha256, sass_sha256=m['sass_sha256'],
                    unpatched_sass_sha256=m.get('unpatched_sass_sha256'), extra_defines=m.get('extra_defines') or {},
                    blob_gen=m.get('blob_gen'), compiled_description=m.get('compiled_description'),
                    has_quant=bool(kern.has_quant), has_schedule=bool(kern.has_schedule),
                    quant_mode=self.Kernel.QUANT_MODES[lin.act_kind],
                    isolated_equals_forward=bool(torch.equal(ref.view(torch.int16), got.view(torch.int16))),
                    forward_gemm_names=sorted(fwd['gemm']), forward_quant_names=sorted(fwd['quant']),
                    y_sha256=self.tsha(ref))

    def time(self, msg):
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile
        key, t = (msg['policy'], msg['proj']), msg['t']
        lin, n, k, kern, sched, width = self.call(key, t)
        pool, x = self.pools(n, k, t)
        y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
        copies = self.copies[key]
        start = self.counter[key]

        def rep(i, timed):
            self.flush.sum()
            torch.cuda.synchronize()
            x.copy_(pool[i % 2])
            torch.cuda.synchronize()
            q = kern.quant_rows(x, lin.act_kind)
            torch.cuda.synchronize()
            wp, wsf = copies[self.counter[key] % len(copies)]
            self.counter[key] += 1
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            self.gemm(lin, kern, sched, q, wp, wsf, y, n, k, t)
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) * 1e3 if timed else None
        for i in range(msg['warmup']):
            rep(i, False)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ev = [rep(i, True) for i in range(msg['reps'])]
        kt, names = {'gemm': [], 'quant': []}, {'gemm': set(), 'quant': set()}
        for e in prof.events():
            if e.device_type.name == 'CUDA':
                cls = classify(e.name)
                if cls in kt:
                    kt[cls].append(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)
                    names[cls].add(e.name)
        return dict(gemm_us=kt['gemm'], quant_us=kt['quant'], event_us=ev, kernel=kern.cfg.name, width=width,
                    schedule=list(sched), gemm_names=sorted(names['gemm']), quant_names=sorted(names['quant']),
                    copies=[(start + msg['warmup'] + i) % len(copies) for i in range(msg['reps'])])


def worker_main(args):
    if args.side == 'fq':                           # nothing of NVFP4-RaZeR on the fq side's path, this script's dir included
        here = Path(__file__).resolve().parent
        sys.path = [p for p in sys.path if Path(p or '.').resolve() != here]
    proto = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)                                   # anything else printed goes to stderr
    sys.stdout = sys.stderr
    w = Worker(args)
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if msg['op'] == 'quit':
                proto.write(json.dumps(dict(ok=True)) + '\n')
                break
            out = getattr(w, msg['op'])(msg)
            proto.write(json.dumps(dict(ok=True, result=out), default=str) + '\n')
        except BaseException as e:                  # noqa: BLE001 -- reported to the driver, which stops
            proto.write(json.dumps(dict(ok=False, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())) + '\n')
            break


# ------------------------------------------------------------------------------------------------------------- driver

class Peer:
    def __init__(self, side, args, env, cwd, log, worker_side=None):
        cmd = [args.python, __file__, '--role', 'worker', '--side', worker_side or side, '--model', args.model,
               '--fq-root', args.fq_root,
               '--fq-build', str(args.fq_build), '--flush-mib', str(args.flush_mib), '--policies', ','.join(args.policies)]
        if args.projections:
            cmd += ['--projections', ','.join(args.projections)]
        self.side = side
        self.log = open(log, 'w')
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, cwd=cwd,
                                  text=True, bufsize=1)

    def __call__(self, op, **kw):
        self.p.stdin.write(json.dumps(dict(op=op, **kw)) + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit(f'{self.side} worker died (see its log)')
        rep = json.loads(line)
        if not rep['ok']:
            raise SystemExit(f"{self.side} worker failed on {op}: {rep['error']}\n{rep['traceback']}")
        return rep.get('result')

    def close(self):
        try:
            self('quit')
        except SystemExit:
            pass
        self.p.wait(timeout=60)
        self.log.close()


def git_state(root):
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True).stdout
    return dict(head=head, clean=not dirty.strip(), status=dirty.strip().splitlines()[:20])


def stats(v):
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [min(v), statistics.median(v), max(v)]
    return dict(median=statistics.median(v), q1=q[0], q3=q[2], min=min(v), max=max(v), n=len(v))


def driver_main(args):
    import pynvml
    sys.path.insert(0, str(RAZER / 'experiments' / 'paper'))
    sys.path.insert(0, str(RAZER / 'sm120' / 'bench'))
    import bench_gemm_isolated as G      # its Telemetry (NVML, nvidia-smi); this process creates no CUDA context
    import common as B
    B.require_idle()
    out_dir = Path(str(args.out) + '.logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    fq_root = Path(args.fq_root).resolve()
    res = dict(status='running', suite='flipquant GEMM parity (results/kernel_opt/flipquant_parity/NOTE.md)',
               model=args.model, policies={p: POLICIES[p] for p in args.policies}, tokens=args.tokens,
               method=dict(reps=args.reps, warmup=args.warmup, rounds=args.rounds, flush_mib=args.flush_mib,
                           value='median over all rounds of the CUPTI device time of the isolated GEMM launch',
                           order='per (projection, T): round r starts the policies at r * len / rounds; the two sides '
                                 'of a policy back to back, fq first when r + policy index is even'),
               razer=dict(root=str(RAZER), **git_state(RAZER), build=str(args.rz_build)),
               flipquant=dict(root=str(fq_root), **git_state(fq_root), build=str(args.fq_build)),
               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), control=args.control,
               pair=args.pair, sides={slot: IMPL[impl] for slot, impl in args.pair.items()})
    if not res['flipquant']['clean']:
        raise SystemExit(f'the flipquant worktree is not clean: {res["flipquant"]["status"]}')
    tel = G.Telemetry(Path(str(args.out) + '.telemetry.csv'))
    res['power_limit_w'] = tel.power_limit_w()
    base = {k: v for k, v in os.environ.items() if k not in ('SM120_BUILD_DIR', 'PYTHONPATH')}
    base.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    peers = {}
    try:
        def spawn(slot):
            impl = args.pair[slot]
            if impl == 'fq':
                return Peer(slot, args, base, str(fq_root), out_dir / f'{slot}.log', worker_side='fq')
            return Peer(slot, args, dict(base, SM120_BUILD_DIR=str(args.rz_build)), str(RAZER), out_dir / f'{slot}.log',
                        worker_side=impl)
        peers['rz'] = spawn('rz')
        t0 = time.time()
        rz = peers['rz']('setup')
        print(f"rz slot ({args.pair['rz']}) setup {time.time() - t0:.0f} s", flush=True)
        peers['fq'] = spawn('fq')
        t0 = time.time()
        fq = peers['fq']('setup', modules=rz['modules'])
        print(f"fq slot ({args.pair['fq']}) setup {time.time() - t0:.0f} s", flush=True)
        if args.pair['fq'] != 'fq':
            assert fq['modules'] == rz['modules'], 'the two RaZeR workers chose different modules'
        res['setup'] = dict(rz=rz, fq=fq)
        pids = {os.getpid(), peers['rz'].p.pid, peers['fq'].p.pid}
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def others():
            return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid not in pids]
        # the packages are each side's own, and the operands are equal bit for bit
        package = dict(fq=fq_root / 'kernels' / 'razer_sm120' / 'mixfp4_sm120', rz=RAZER / 'sm120' / 'mixfp4_sm120',
                       rzm=RAZER / 'sm120' / 'mixfp4_sm120')
        assert Path(fq['package']) == package[args.pair['fq']], fq['package']
        assert Path(rz['package']) == package[args.pair['rz']], rz['package']
        res['checks'] = dict(operands={}, cells=[], counts_ok=True, other_processes=[])
        for key, a in rz['operands'].items():
            b = fq['operands'][key]
            same = {f: a[f] == b[f] for f in ('module', 'shape', 'e0m3_tiles', 'packed_sha256', 'sf_sha256', 'global_scale',
                                               'bias_sha256', 'kernel_set')}
            res['checks']['operands'][key] = same
            if not all(same.values()):
                raise SystemExit(f'operands differ at {key}: {same}')
        if args.rehome:                              # diagnostic: every timed buffer into fresh segments first
            res['rehome'] = {side: peers[side]('rehome') for side in ('fq', 'rz')}
        res['rows'] = []
        policies = list(args.policies)
        projs = list(next(iter(rz['modules'].values())))
        for proj in projs:
            for t in args.tokens:
                cells = {}
                for pol in policies:
                    if args.no_checks:                   # diagnostic only: no record of the selection or outputs
                        cells[pol] = {side: [] for side in ('fq', 'rz')}
                        continue
                    c = {side: peers[side]('check', policy=pol, proj=proj, t=t) for side in ('fq', 'rz')}
                    cell = dict(policy=pol, proj=proj, tokens=t, fq=c['fq'], rz=c['rz'],
                                y_equal=c['fq']['y_sha256'] == c['rz']['y_sha256'],
                                isolated_equals_forward=c['fq']['isolated_equals_forward'] and c['rz']['isolated_equals_forward'])
                    res['checks']['cells'].append(cell)
                    if not (cell['y_equal'] and cell['isolated_equals_forward']):
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise SystemExit(f'{pol} {proj} T={t}: outputs differ ({cell["y_equal"]}, {cell["isolated_equals_forward"]})')
                    cells[pol] = {side: [] for side in ('fq', 'rz')}
                for r in range(args.rounds):
                    shift = (r * len(policies) // args.rounds) % len(policies)
                    for pos, pol in enumerate(policies[shift:] + policies[:shift]):
                        order = ('fq', 'rz') if (r + policies.index(pol)) % 2 == 0 else ('rz', 'fq')
                        for slot, side in enumerate(order):
                            o = others()
                            if o:
                                res['checks']['other_processes'].append(dict(proj=proj, tokens=t, policy=pol, round=r, pids=o))
                                res['status'] = 'failed'
                                B.write(args.out, res)
                                raise SystemExit(f'another compute process on the GPU: {o}')
                            before = tel.snap()
                            got = peers[side]('time', policy=pol, proj=proj, t=t, reps=args.reps, warmup=args.warmup)
                            after = tel.snap()
                            if len(got['gemm_us']) != args.reps or len(got['quant_us']) != args.reps:
                                res['checks']['counts_ok'] = False
                                res['status'] = 'failed'
                                B.write(args.out, res)
                                raise SystemExit(f"{pol} {proj} T={t} {side} round {r}: {len(got['gemm_us'])} GEMM / "
                                                 f"{len(got['quant_us'])} quantizer launches profiled")
                            cells[pol][side].append(dict(round=r, position=pos, slot=slot, telemetry=G.Telemetry.block(before, after),
                                                         **got))
                for pol in policies:
                    for side in ('fq', 'rz'):
                        blocks = cells[pol][side]
                        g = [v for b in blocks for v in b['gemm_us']]
                        q = [v for b in blocks for v in b['quant_us']]
                        e = [v for b in blocks for v in b['event_us']]
                        res['rows'].append(dict(
                            policy=pol, proj=proj, tokens=t, side=side, kernel=blocks[0]['kernel'], width=blocks[0]['width'],
                            schedule=blocks[0]['schedule'],
                            same_launch_every_round=all((b['kernel'], b['width'], b['schedule']) ==
                                                        (blocks[0]['kernel'], blocks[0]['width'], blocks[0]['schedule'])
                                                        for b in blocks),
                            gemm_names=sorted({n for b in blocks for n in b['gemm_names']}),
                            quant_names=sorted({n for b in blocks for n in b['quant_names']}),
                            gemm_us=statistics.median(g), gemm=stats(g), quant_us=statistics.median(q), quant=stats(q),
                            event_us=statistics.median(e), event=stats(e),
                            rounds=[dict(round=b['round'], position=b['position'], slot=b['slot'],
                                         gemm_us=statistics.median(b['gemm_us']), quant_us=statistics.median(b['quant_us']),
                                         event_us=statistics.median(b['event_us']), telemetry=b['telemetry'])
                                    for b in blocks]))
                line = ' '.join(f"{r['policy']}:{r['side']}={r['gemm_us']:.2f}" for r in res['rows']
                                if r['proj'] == proj and r['tokens'] == t)
                print(f'{args.model} {proj} T={t} {line}', flush=True)
                B.write(args.out, res)
        res['status'] = 'complete'
    finally:
        for p in peers.values():
            p.close()
        tel.close()
        res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        res['flipquant_after'] = git_state(fq_root)
        try:
            res['sampler'] = G.Telemetry.summarize(Path(str(args.out) + '.telemetry.csv'))
        except Exception as e:                       # noqa: BLE001
            res['sampler'] = dict(error=str(e))
        B.write(args.out, res)
    if not res['flipquant_after']['clean']:
        raise SystemExit(f'the flipquant worktree changed: {res["flipquant_after"]["status"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--role', choices=('driver', 'worker'), default='driver')
    ap.add_argument('--side', choices=('fq', 'rz', 'rzm'), default=None)
    ap.add_argument('--model', choices=tuple(MODELS), default='llama8b')
    ap.add_argument('--fq-root', default='/home/dev/n16k64_campaign/fqport/wt', help="a clean flipquant worktree")
    ap.add_argument('--fq-build', default=None, help='default <fq-root>/kernels/razer_sm120/build_ko')
    ap.add_argument('--rz-build', default='/home/dev/n16k64_campaign/kernel_opt/build_V', help="kernel-opt's deployment directory")
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--tokens', default=','.join(map(str, TOKENS)))
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--control', choices=('none', 'aa'), default='none',
                    help="aa: an A/A control, the 'fq' slot runs a second NVFP4-RaZeR worker (= --pair rz:rz)")
    ap.add_argument('--rehome', action='store_true', help="diagnostic: both workers 'rehome' before the first cell")
    ap.add_argument('--no-checks', action='store_true', help='diagnostic: skip the per-cell checks')
    ap.add_argument('--pair', default='fq:rz',
                    help="what the two slots run, '<fq slot>:<rz slot>' from fq (flipquant's real path), rz (the "
                         "kernel-opt harness), rzm (NVFP4-RaZeR's own model path); default fq:rz")
    args = ap.parse_args()
    if args.control == 'aa':
        args.pair = 'rz:rz'
    a, b = args.pair.split(':')
    assert a in IMPL and b in ('rz', 'rzm'), args.pair
    args.pair = dict(fq=a, rz=b)
    args.fq_build = Path(args.fq_build or Path(args.fq_root).resolve() / 'kernels' / 'razer_sm120' / 'build_ko')
    args.policies = [p for p in args.policies.split(',') if p]
    assert all(p in POLICIES for p in args.policies), args.policies
    args.projections = args.projections.split(',') if args.projections else None
    args.tokens = [int(t) for t in str(args.tokens).split(',')]
    if args.role == 'worker':
        return worker_main(args)
    if args.out is None:
        ap.error('--out is required')
    driver_main(args)


if __name__ == '__main__':
    main()
