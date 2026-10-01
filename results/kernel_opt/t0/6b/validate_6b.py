"""Amendment 6b check (CPU only), with the 6b patcher (worktree wtN2):
1. every existing tagged mixed LIBRARY (6 build dirs): parse_ommas_cfg (reaching definitions, jump tables resolved) gives
   every OMMA the site of parse_ommas (strict, nearest tagged writer);
2. every existing tagged SELF-TEST executable (the upstream driver, whose kernel has the BRX jump table): the same
   comparison, which exercises the BRX path against the strict parser's ground truth;
3. the 8 t0 self-test executables left by the stopped G3: classified with equal per-site counts."""
import collections, glob, importlib.util, subprocess
spec = importlib.util.spec_from_file_location('P', '/home/dev/n16k64_campaign/kernel_opt/wtN2/sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
X = '/home/dev/.conda/envs/mixfp4-cuda131/bin/cuobjdump'
roots = ['/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_4', '/home/dev/n16k64_campaign/kernel_opt/build_A1',
         '/home/dev/n16k64_campaign/kernel_opt/build_freq', '/home/dev/n16k64_campaign/kernel_opt/build_e0m3', '/home/dev/n16k64_campaign/kernel_opt/build']


def has_brx(sass):
    return any(' BRX ' in l for l in sass.split('\n'))


def compare(binary):
    sass = subprocess.run([X, '--dump-sass', binary], check=True, capture_output=True, text=True).stdout
    if 'OMMA' not in sass:
        return None
    try:
        old = sorted((a, s) for a, _, _, s, _ in P.parse_ommas(sass))
    except RuntimeError:
        return 'untagged'
    new = sorted((a, s) for a, _, _, s, _ in P.parse_ommas_cfg(sass, P.jump_tables(X, binary)))
    return ('same' if old == new else 'DIFF', len(old), has_brx(sass), dict(collections.Counter(s for _, s in new)))


bad = 0
for kind, pattern in (('library', 'libmixfp4_sm120_*.so.unpatched'), ('selftest', 'selftest.unpatched')):
    n = n_brx = 0
    for root in roots:
        for b in sorted(glob.glob(f'{root}/*/{pattern}')):
            r = compare(b)
            if r is None or r == 'untagged':
                continue
            n += 1
            n_brx += r[2]
            bad += r[0] != 'same'
            if r[0] != 'same' or kind == 'selftest':
                print(kind, r[0], root.split('/')[-1], b.split('/')[-2], 'OMMAs', r[1], 'BRX' if r[2] else 'no BRX', r[3])
    print(f'{kind}: {n} tagged binaries compared, {n_brx} with a BRX')
for b in sorted(glob.glob('/home/dev/n16k64_campaign/kernel_opt/build_T/*_t0/selftest.unpatched')):
    sass = subprocess.run([X, '--dump-sass', b], check=True, capture_output=True, text=True).stdout
    c = collections.Counter(s for _, _, _, s, _ in P.parse_ommas_cfg(sass, P.jump_tables(X, b)))
    ok = len(set(c.values())) == 1
    bad += not ok
    print('t0 selftest', 'OK ' if ok else 'BAD', b.split('/')[-2], dict(c), 'BRX' if has_brx(sass) else 'no BRX')
print('failures:', bad)
