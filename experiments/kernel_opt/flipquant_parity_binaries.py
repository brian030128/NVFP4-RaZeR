#!/usr/bin/env python3
"""flipquant GEMM parity, the binaries (results/kernel_opt/flipquant_parity/NOTE.md): every build that the timing run
loaded, flipquant's (its deployment directory build_ko) against NVFP4-RaZeR's (kernel-opt build_V). CPU only.

    python experiments/kernel_opt/flipquant_parity_binaries.py --timing JSON --out JSON

Per build name, from the libraries the two sides recorded in the timing run's checks:
- the file sha256 of each side's library (and that it is the one its manifest names);
- the manifests: SASS hashes (patched and unpatched), build defines, blob tags, configuration, compiled description;
- the SASS recomputed now (cuobjdump --dump-sass of each loaded file);
- the device code: the cubins extracted from each library (cuobjdump -xelf all), compared ELF section by ELF section;
- the host library: compared ELF section by ELF section; each section that differs is compared again after replacing
  nvcc's 8-hex compilation id in internal symbol names (_INTERNAL_<id>_, _GLOBAL__N__<id>_, tmpxft_<id>_) by a fixed
  string of the same length.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ID = re.compile(rb'(_INTERNAL_|_GLOBAL__N__|tmpxft_)[0-9a-f]{8}_')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def norm(b):
    return ID.sub(lambda m: m.group(1) + b'xxxxxxxx_', b)


def sections(path):
    """{name: (offset, size, allocated)} from readelf -S -W."""
    out = subprocess.run(['readelf', '-S', '-W', str(path)], capture_output=True, text=True, check=True).stdout
    secs = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith('[') or ']' not in line:
            continue
        parts = line.split(']', 1)[1].split()
        if len(parts) < 7 or not parts[0].startswith('.'):
            continue
        name, off, size = parts[0], int(parts[3], 16), int(parts[4], 16)
        flags = parts[6] if not parts[6].isdigit() else ''
        secs[name] = (off, size, 'A' in flags)
    return secs


def compare_elf(a, b):
    """Section-by-section comparison of two ELF files of the same layout."""
    sa, sb = sections(a), sections(b)
    ba, bb = Path(a).read_bytes(), Path(b).read_bytes()
    out = dict(same_layout=[(n, s[:2]) for n, s in sa.items()] == [(n, s[:2]) for n, s in sb.items()], differing={})
    if not out['same_layout']:
        return out
    for name, (off, size, alloc) in sa.items():
        if name in ('.bss', '.tbss'):
            continue
        x, y = ba[off:off + size], bb[off:off + size]
        if x != y:
            out['differing'][name] = dict(allocated=alloc, bytes=sum(1 for i, j in zip(x, y) if i != j),
                                          equal_up_to_compilation_id=norm(x) == norm(y))
    return out


def run(cmd):
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--timing', type=Path, required=True)
    ap.add_argument('--cuobjdump', default='/home/dev/.conda/envs/mixfp4-cuda131/bin/cuobjdump')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    timing = json.loads(args.timing.read_text())
    pairs = {}
    for c in timing['checks']['cells']:
        name = c['fq']['kernel']
        assert c['rz']['kernel'] == name, (c['policy'], c['proj'], c['tokens'], c['fq']['kernel'], c['rz']['kernel'])
        pairs.setdefault(name, (c['fq']['library'], c['rz']['library']))
        assert pairs[name] == (c['fq']['library'], c['rz']['library']), name
    res = dict(timing=str(args.timing), cuobjdump=args.cuobjdump,
               cuobjdump_version=run([args.cuobjdump, '--version']).decode().strip().splitlines()[-1], builds={})
    for name, (fq, rz) in sorted(pairs.items()):
        r = dict(fq=fq, rz=rz)
        mf = json.loads((Path(fq).parent / 'manifest.json').read_text())
        mr = json.loads((Path(rz).parent / 'manifest.json').read_text())
        bf, br = Path(fq).read_bytes(), Path(rz).read_bytes()
        r['library_sha256'] = dict(fq=sha(bf), rz=sha(br), equal=sha(bf) == sha(br),
                                   match_manifests=sha(bf) == mf['library_sha256'] and sha(br) == mr['library_sha256'])
        r['manifest_equal'] = {k: mf.get(k) == mr.get(k) for k in ('sass_sha256', 'unpatched_sass_sha256', 'extra_defines',
                                                                     'blob_gen', 'config', 'compiled_description',
                                                                     'weight_operand', 'type_block', 'kind', 'toolchain')}
        r['manifest'] = dict(sass_sha256=mr['sass_sha256'], extra_defines=mr.get('extra_defines') or {},
                             repo_head=dict(fq=mf.get('repo_head'), rz=mr.get('repo_head')),
                             built_utc=dict(fq=mf.get('built_utc'), rz=mr.get('built_utc')))
        sf, sr = sha(run([args.cuobjdump, '--dump-sass', fq])), sha(run([args.cuobjdump, '--dump-sass', rz]))
        r['sass_recomputed'] = dict(fq=sf, rz=sr, equal=sf == sr, equals_manifest=sr == mr['sass_sha256'])
        with tempfile.TemporaryDirectory() as td:
            cub = {}
            for side, lib in (('fq', fq), ('rz', rz)):
                d = Path(td) / side
                d.mkdir()
                subprocess.run([args.cuobjdump, '-xelf', 'all', str(lib)], cwd=d, capture_output=True, check=True)
                cub[side] = sorted(d.iterdir())
            assert [p.name for p in cub['fq']] == [p.name for p in cub['rz']], name
            r['cubins'] = []
            for a, b in zip(cub['fq'], cub['rz']):
                c = compare_elf(a, b)
                c.update(cubin=a.name, file_equal=a.read_bytes() == b.read_bytes())
                r['cubins'].append(c)
        r['host'] = compare_elf(fq, rz)
        code = [c for c in r['cubins']]
        r['device_code_equal'] = all(c['same_layout'] and all(n == '.strtab' and d['equal_up_to_compilation_id']
                                                              for n, d in c['differing'].items()) for c in code)
        r['host_code_equal'] = r['host']['same_layout'] and '.text' not in r['host']['differing']
        r['loaded_bytes_equal_up_to_compilation_id'] = r['host']['same_layout'] and all(
            d['equal_up_to_compilation_id'] or not d['allocated'] or n == '.note.gnu.build-id'
            for n, d in r['host']['differing'].items())
        r['identical'] = (r['sass_recomputed']['equal'] and r['manifest_equal']['sass_sha256'] and r['device_code_equal']
                          and r['host_code_equal'] and r['loaded_bytes_equal_up_to_compilation_id']
                          and r['library_sha256']['match_manifests']
                          and all(r['manifest_equal'][k] for k in ('extra_defines', 'blob_gen', 'config',
                                                                   'compiled_description')))
        res['builds'][name] = r
        print(name, 'identical (SASS, device code, host code; loaded bytes up to the compilation id)' if r['identical']
              else 'DIFFERENT', '| file bytes equal' if r['library_sha256']['equal'] else '| file bytes differ',
              sorted(r['host']['differing']), flush=True)
    res['all_identical'] = all(r['identical'] for r in res['builds'].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print('ALL IDENTICAL' if res['all_identical'] else 'NOT ALL IDENTICAL')


if __name__ == '__main__':
    main()
