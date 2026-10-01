"""CPU check: on every existing (fully tagged) unpatched mixed build, parse_ommas_cfg (reaching definitions) must assign
the same site to every OMMA as parse_ommas (nearest tagged writer)."""
import glob, importlib.util, subprocess, sys
spec = importlib.util.spec_from_file_location('P', '/home/dev/n16k64_campaign/kernel_opt/wtN/sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
X = '/home/dev/.conda/envs/mixfp4-cuda131/bin/cuobjdump'
roots = ['/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_4', '/home/dev/n16k64_campaign/kernel_opt/build_A1',
         '/home/dev/n16k64_campaign/kernel_opt/build_freq', '/home/dev/n16k64_campaign/kernel_opt/build_e0m3', '/home/dev/n16k64_campaign/kernel_opt/build']
seen = set()
bad = 0
for root in roots:
    for lib in sorted(glob.glob(f'{root}/*/libmixfp4_sm120_*.so.unpatched')):
        sass = subprocess.run([X, '--dump-sass', lib], check=True, capture_output=True, text=True).stdout
        if 'OMMA' not in sass:
            continue
        try:
            old = [(a, s) for a, _, _, s, _ in P.parse_ommas(sass)]
        except RuntimeError as e:
            print('SKIP (strict parser fails: untagged build)', lib.split('/')[-2], str(e)[:80]); continue
        new = [(a, s) for a, _, _, s, _ in P.parse_ommas_cfg(sass)]
        same = sorted(old) == sorted(new)
        bad += not same
        from collections import Counter
        print('OK  ' if same else 'DIFF', root.split('/')[-1], lib.split('/')[-2], len(old), dict(Counter(s for _, s in new)))
print('mismatching builds:', bad)
