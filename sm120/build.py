#!/usr/bin/env python3
"""Build -> patch -> verify one SM120 kernel configuration, reproducibly.

    python sm120/build.py --config n16k64_wA            # library used by mixfp4_sm120
    python sm120/build.py --config n16k64_wA --selftest # also build + run the upstream self-test exe
    python sm120/build.py --all                         # every configuration in configs.py

Steps (each one aborts the build on failure):
  1. Toolchain check: nvcc must be the pinned CUDA release (13.1), CUTLASS the pinned commit.
  2. Generate the MMA blob header for the configuration into build/<config>/gen (the vendored
     generator runs from a copy there, so the source tree is never written to).
  3. nvcc -> lib<name>.so.unpatched (SASS-only sm_120a, flags of kernel/scripts/build_mixed.sh).
  4. Install the E0M3 formats with kernel/scripts/patch_mixed_nvfp4_gemm.py. The per-site OMMA
     census must equal configs.KernelConfig.expected_census, the post-patch disassembly must show
     every site in its intended format, and no OMMA may be predicated (a predicated OMMA wastes a
     tensor-pipe issue slot; see kernel/docs/mixed_nvfp4_report.md section 1).
  5. Load the library and check its compiled description (granule, pinning, D layout) against
     the configuration.
  6. Write build/<config>/manifest.json: source revision and hashes, flags, toolchain versions,
     patcher hash, unpatched/patched binary hashes, OMMA census, register/stack usage.

Nothing here depends on an undocumented local checkout or absolute path: CUDA is found through
--cuda-home / $CUDA_HOME (default /usr/local/cuda-13.1), CUTLASS is the git submodule.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from mixfp4_sm120 import configs as CFG  # noqa: E402

KERNEL = HERE / 'kernel'
CUTLASS = HERE / 'third_party' / 'cutlass'
CSRC = HERE / 'csrc' / 'mixfp4_sm120.cu'
PATCHER = KERNEL / 'scripts' / 'patch_mixed_nvfp4_gemm.py'
BLOBGEN = KERNEL / 'scripts' / 'gen_mixed_mma_blob.py'
BUILD_ROOT = Path(os.environ.get('SM120_BUILD_DIR', HERE / 'build'))

PINNED_CUDA = '13.1'
PINNED_CUTLASS = 'e64a9136dd929639e5f7c969fe5af3bf7415cd4f'

NVCC_FLAGS = [
    '-O3', '-DNDEBUG', '-std=c++17', '--generate-code=arch=compute_120a,code=[sm_120a]',
    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-DCUTLASS_ENABLE_GDC_FOR_SM100=1',
    '--expt-relaxed-constexpr', '-ftemplate-backtrace-limit=0',
    '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '-DCUTLASS_SM100_FAMILY_ARCHS_ENABLED',
    '-Xcompiler=-fno-strict-aliasing',
]


class BuildError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    if r.returncode != 0:
        raise BuildError(f'command failed ({r.returncode}): {" ".join(map(str, cmd))}\n{r.stdout[-6000:]}')
    return r.stdout


def git(*args, cwd=REPO):
    return subprocess.run(['git', *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True).stdout.strip()


def toolchain(cuda_home):
    nvcc = cuda_home / 'bin' / 'nvcc'
    cuobjdump = cuda_home / 'bin' / 'cuobjdump'
    for p in (nvcc, cuobjdump):
        if not p.exists():
            raise BuildError(f'{p} not found (set --cuda-home or $CUDA_HOME)')
    ver = run([nvcc, '--version'])
    m = re.search(r'release (\d+\.\d+), V([\d.]+)', ver)
    if not m:
        raise BuildError(f'cannot parse nvcc version:\n{ver}')
    if m.group(1) != PINNED_CUDA:
        raise BuildError(f'nvcc is CUDA {m.group(1)}; this build is pinned to CUDA {PINNED_CUDA} '
                         f'(the release the kernel and its SASS patch sites were validated with)')
    cxx = shutil.which(os.environ.get('CXX', 'g++'))
    cxx_ver = run([cxx, '--version']).splitlines()[0] if cxx else 'unknown'
    head = git('rev-parse', 'HEAD', cwd=CUTLASS)
    if head != PINNED_CUTLASS:
        raise BuildError(f'CUTLASS submodule is at {head or "<missing>"}, expected {PINNED_CUTLASS}. '
                         f'Run: git submodule update --init sm120/third_party/cutlass')
    return dict(nvcc=str(nvcc), cuobjdump=str(cuobjdump), cuda_release=m.group(1), nvcc_version=m.group(2),
                host_compiler=cxx, host_compiler_version=cxx_ver, cutlass_commit=head,
                cutlass_describe=git('describe', '--tags', cwd=CUTLASS))


def generate(cfg, out):
    """Generated headers for one configuration: CUTLASS's version header and the MMA blob."""
    inc = out / 'gen' / 'include' / 'cutlass'
    inc.mkdir(parents=True, exist_ok=True)
    template = (CUTLASS / 'cmake' / 'version_extended.h.in').read_text()
    (inc / 'version_extended.h').write_text(
        template.replace('@CUTLASS_VERSION_BUILD@', '0').replace('@CUTLASS_REVISION@', PINNED_CUTLASS[:8]))
    if not cfg.blob_gen:
        return None
    gscripts = out / 'gen' / 'scripts'
    gscripts.mkdir(parents=True, exist_ok=True)
    (out / 'gen' / 'src' / 'collective').mkdir(parents=True, exist_ok=True)
    shutil.copy2(BLOBGEN, gscripts / BLOBGEN.name)
    env = dict(os.environ, **{k: str(v) for k, v in cfg.blob_gen.items()})
    log = run([sys.executable, gscripts / BLOBGEN.name], env=env)
    blob = out / 'gen' / 'src' / 'collective' / 'mixed_mma_blob_generated.hpp'
    if not blob.exists():
        raise BuildError(f'blob generator did not write {blob}:\n{log}')
    return blob


def compile_lib(cfg, tc, out):
    lib = out / f'libmixfp4_sm120_{cfg.name}.so'
    defines = dict(cfg.defines)
    if cfg.kind == 'stock':
        defines['SM120_STOCK'] = 1
    defines['SM120_CONFIG_NAME'] = f'"{cfg.name}"'
    dflags = [f'-D{k}={v}' for k, v in sorted(defines.items())]
    includes = [
        f'-I{out / "gen" / "src"}',      # generated blob first: it must shadow the vendored default
        f'-I{KERNEL / "src"}',
        f'-I{CUTLASS / "include"}',
        f'-I{out / "gen" / "include"}',
        f'-I{CUTLASS / "tools" / "util" / "include"}',
        f'-I{CUTLASS / "examples" / "common"}',
    ]
    cmd = [tc['nvcc'], *NVCC_FLAGS, *dflags, '-Xcompiler=-fPIC', '-shared', '-ccbin', tc['host_compiler'],
           *includes, CSRC, '-o', f'{lib}.unpatched', '-lcudart_static', '-lrt', '-lpthread', '-ldl']
    log = run(cmd)
    return lib, cmd, log


def omma_census(cuobjdump, binary):
    sass = run([cuobjdump, '--dump-sass', binary])
    ops = re.findall(r'(@!?U?P[T\d]+\s+)?OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)\.', sass)
    formats = Counter(f'{a}x{b}' for _, a, b in ops)
    predicated = sum(1 for p, _, _ in ops if p)
    return dict(formats=dict(formats), total=len(ops), predicated=predicated)


def sass_sha256(cuobjdump, binary):
    """Hash of the disassembled device code: identical for identical kernels wherever the checkout
    lives (the file hash is not -- nvcc names host-side internal symbols after the absolute path)."""
    return hashlib.sha256(run([cuobjdump, '--dump-sass', binary]).encode()).hexdigest()


def resource_usage(cuobjdump, binary):
    out = run([cuobjdump, '--dump-resource-usage', binary])
    kernels = []
    for m in re.finditer(r'Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', out):
        kernels.append(dict(function=m.group(1)[:120], reg=int(m.group(2)), stack=int(m.group(3)),
                            shared=int(m.group(4)), local=int(m.group(5))))
    return kernels


def patch(cfg, tc, lib):
    unpatched = Path(f'{lib}.unpatched')
    if not cfg.patch:
        pre = omma_census(tc['cuobjdump'], unpatched)
        if cfg.mixed and cfg.type_block is not None:
            raise BuildError(f'{cfg.name}: a mixed configuration with a format granule must be patched')
        shutil.copy2(unpatched, lib)
        return dict(patched=False, census_unpatched=pre, census_patched=pre, patcher_log='')
    log = run([sys.executable, PATCHER, '--cuobjdump', tc['cuobjdump'], '--allow-missing-sites', unpatched, lib])
    m = re.search(r'per-site counts: (.*)', log)
    if not m:
        raise BuildError(f'patcher reported no census:\n{log}')
    sites = {int(k): int(v) for k, v in re.findall(r'site (\d+)=(\d+)', m.group(1))}
    if 'verify OK' not in log:
        raise BuildError(f'patcher post-disassembly verification failed:\n{log}')
    if cfg.expected_census is not None and sites != {int(k): v for k, v in cfg.expected_census.items()}:
        raise BuildError(f'{cfg.name}: OMMA site census {sites} != expected {cfg.expected_census}. '
                         f'The kernel or toolchain changed; re-validate before updating configs.py.\n{log}')
    # The weight operand carries the format; the activation operand is pinned to E2M1, so only the
    # sites that put E0M3 on the weight operand may exist (site 1 = E0M3 on A, site 2 = on B).
    allowed = {0, 1} if cfg.weight_operand == 0 else {0, 2}
    if set(sites) != allowed:
        raise BuildError(f'{cfg.name}: sites {sorted(sites)} present, expected exactly {sorted(allowed)}')
    pre = omma_census(tc['cuobjdump'], unpatched)
    post = omma_census(tc['cuobjdump'], lib)
    want = 'E0M3xE2M1' if cfg.weight_operand == 0 else 'E2M1xE0M3'
    if pre['formats'] != {'E2M1xE2M1': pre['total']}:
        raise BuildError(f'unpatched binary already contains non-E2M1 OMMAs: {pre}')
    if post['formats'].get(want, 0) != sites.get(1 if cfg.weight_operand == 0 else 2, 0):
        raise BuildError(f'patched census {post} disagrees with the patcher sites {sites}')
    return dict(patched=True, sites=sites, census_unpatched=pre, census_patched=post, patcher_log=log)


def describe(lib):
    dll = ctypes.CDLL(str(lib))
    dll.sm120_describe.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.sm120_describe.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(4096)
    n = dll.sm120_describe(buf, 4096)
    if n <= 0 or n >= 4096:
        raise BuildError(f'sm120_describe returned {n}')
    return json.loads(buf.value.decode())


def check_description(cfg, d):
    errs = []
    if d['config'] != cfg.name:
        errs.append(f'config {d["config"]}')
    if bool(d['stock']) != (cfg.kind == 'stock'):
        errs.append(f'stock {d["stock"]}')
    if bool(d['d_colmajor']) != cfg.d_colmajor:
        errs.append(f'd_colmajor {d["d_colmajor"]}')
    if bool(d['bias_on_n']) != (cfg.weight_operand == 1):
        errs.append(f'bias_on_n {d["bias_on_n"]}')
    if cfg.type_block is not None:
        wop, aop = cfg.weight_operand, 1 - cfg.weight_operand
        gran = d['granule_a'] if wop == 0 else d['granule_b']
        if tuple(gran) != tuple(cfg.type_block):
            errs.append(f'weight granule {gran} != {cfg.type_block}')
        if d['pinned_e2m1'][wop] or not d['pinned_e2m1'][aop]:
            errs.append(f'pinning {d["pinned_e2m1"]}: weight operand must be mixed, activation pinned E2M1')
        if cfg.blob_gen and d['blobgen_mma_mn'] != [cfg.blob_gen['MMA_M'], cfg.blob_gen['MMA_N']]:
            errs.append(f'blob header {d["blobgen_mma_mn"]} is not the generated one')
    if errs:
        raise BuildError(f'{cfg.name}: compiled description disagrees with configs.py: {errs}\n{d}')


def source_hashes(blob):
    files = [CSRC, CSRC.parent / 'quant_act.cuh', PATCHER, BLOBGEN, Path(__file__), HERE / 'mixfp4_sm120' / 'configs.py']
    files += sorted((KERNEL / 'src').rglob('*'))
    out = {}
    for p in files:
        if p.is_file():
            out[str(p.relative_to(REPO))] = sha256(p)
    if blob is not None:
        out['<generated>/mixed_mma_blob_generated.hpp'] = sha256(blob)
    return out


def build(cfg, tc, selftest=False):
    out = BUILD_ROOT / cfg.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    blob = generate(cfg, out)
    lib, cmd, clog = compile_lib(cfg, tc, out)
    pinfo = patch(cfg, tc, lib)
    if pinfo['census_patched']['predicated']:
        raise BuildError(f'{cfg.name}: {pinfo["census_patched"]["predicated"]} predicated OMMAs '
                         f'(each wastes a tensor-pipe issue slot)')
    desc = describe(lib)
    check_description(cfg, desc)
    res = resource_usage(tc['cuobjdump'], lib)
    manifest = dict(
        config=cfg.name, description=cfg.description, kind=cfg.kind, weight_operand=cfg.weight_operand,
        type_block=cfg.type_block, defines=cfg.defines, blob_gen=cfg.blob_gen,
        built_utc=t0.isoformat(timespec='seconds'), host=platform.node(),
        repo_head=git('rev-parse', 'HEAD'), repo_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
        toolchain=tc, nvcc_command=[str(c) for c in cmd], nvcc_warnings=clog.strip()[-4000:],
        sources=source_hashes(blob), patcher_sha256=sha256(PATCHER),
        library=lib.name, unpatched_sha256=sha256(f'{lib}.unpatched'), library_sha256=sha256(lib),
        sass_sha256=sass_sha256(tc['cuobjdump'], lib), unpatched_sass_sha256=sass_sha256(tc['cuobjdump'], f'{lib}.unpatched'),
        patch=pinfo, compiled_description=desc, resource_usage=res,
    )
    if selftest:
        manifest['selftest'] = st = run_selftest(cfg, tc, out)
        # The library differs from the self-test driver only in its epilogue; the mainloop's
        # dispatch sites must be the same instructions in the same number.
        if st.get('exe_sites') is not None and pinfo.get('sites') is not None and st['exe_sites'] != pinfo['sites']:
            raise BuildError(f'{cfg.name}: library census {pinfo["sites"]} != self-test driver {st["exe_sites"]}')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


# ------------------------------------------------------------------------------ upstream self-test

SELFTEST_SHAPES = [(1024, 1024, 1024), (2048, 512, 4096), (512, 2048, 1024)]


def run_selftest(cfg, tc, out):
    """Build the vendored mixed_nvfp4_gemm.cu driver (its own E0M3-aware host reference) for this
    configuration, patch it, and require PASS patched / FAIL unpatched on per-granule random tagging
    of the weight operand. Mixed configurations only."""
    if cfg.kind != 'mixed' or cfg.type_block is None:
        return dict(skipped='no format granule')
    exe = out / 'selftest'
    defines = [f'-D{k}={v}' for k, v in sorted(cfg.defines.items()) if not k.startswith('SM120_')]
    cmd = [tc['nvcc'], *NVCC_FLAGS, *defines, '-Xcompiler=-fopenmp', '-ccbin', tc['host_compiler'],
           f'-I{out / "gen" / "src"}', f'-I{KERNEL / "src"}', f'-I{CUTLASS / "include"}', f'-I{out / "gen" / "include"}',
           f'-I{CUTLASS / "tools" / "util" / "include"}', f'-I{CUTLASS / "examples" / "common"}',
           KERNEL / 'src' / 'mixed_nvfp4_gemm.cu', '-o', f'{exe}.unpatched',
           '-lcudadevrt', '-lcudart_static', '-lrt', '-lpthread', '-ldl', '-lgomp']
    run(cmd)
    plog = run([sys.executable, PATCHER, '--cuobjdump', tc['cuobjdump'], '--allow-missing-sites', f'{exe}.unpatched', exe])
    m = re.search(r'per-site counts: (.*)', plog)
    exe_sites = {int(k): int(v) for k, v in re.findall(r'site (\d+)=(\d+)', m.group(1))} if m else None
    tag = 'randa' if cfg.weight_operand == 0 else 'randb'
    rows = []
    for m, n, k in SELFTEST_SHAPES:
        for variant, binary in (('patched', exe), ('unpatched', f'{exe}.unpatched')):
            env = dict(os.environ, MIXFP4_TAG=tag)
            r = subprocess.run([str(binary), str(m), str(n), str(k), '3'], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=600)
            mm = re.search(r'Correctness: (PASSED|FAILED)\s+\(relative Frobenius error ([0-9.eE+-]+)', r.stdout)
            rows.append(dict(shape=[m, n, k], binary=variant, tag=tag,
                             result=mm.group(1) if mm else 'ERROR', rel_err=float(mm.group(2)) if mm else None,
                             granule_line=next((ln for ln in r.stdout.splitlines() if 'granule' in ln), '')))
    ok = all(r['result'] == ('PASSED' if r['binary'] == 'patched' else 'FAILED') for r in rows)
    if not ok:
        raise BuildError(f'{cfg.name}: upstream self-test gate failed: {json.dumps(rows, indent=1)}')
    return dict(gate='PASS', rows=rows, exe_sha256=sha256(exe), exe_sites=exe_sites)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', action='append', help='configuration name (repeatable)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--selftest', action='store_true', help='also build and gate the upstream self-test driver')
    ap.add_argument('--cuda-home', type=Path, default=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.1')))
    args = ap.parse_args()
    names = list(CFG.CONFIGS) if args.all else (args.config or [CFG.DEFAULT])
    tc = toolchain(args.cuda_home)
    failed = []
    for name in names:
        cfg = CFG.get(name)
        try:
            m = build(cfg, tc, args.selftest)
            p = m['patch']
            print(f'[ok] {name}: {m["library"]} sass_sha256={m["sass_sha256"][:16]} '
                  f'census={p["census_patched"]["formats"]} predicated={p["census_patched"]["predicated"]} '
                  f'regs={[r["reg"] for r in m["resource_usage"]]} stages={m["compiled_description"]["mainloop_stages"]}'
                  + (f' selftest={m["selftest"].get("gate", m["selftest"])}' if 'selftest' in m else ''), flush=True)
        except BuildError as e:
            failed.append(name)
            print(f'[FAIL] {name}: {e}', file=sys.stderr, flush=True)
    if failed:
        sys.exit(f'build failed for {failed}')


if __name__ == '__main__':
    main()
