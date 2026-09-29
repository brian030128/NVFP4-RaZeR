#!/usr/bin/env python3
"""Copy the result records and tables of a paper run into the repository (results/paper by default), for commit.

    PAPER_PYTHON experiments/paper/collect_results.py [--out PAPER_OUT] [--dest results/paper]

Collected, from the output root (PAPER_OUT):
  00_check.json, commands.log                  the environment check; every command with its start, end and exit code
  maps/<model>_<unit>/calibration.json         -> maps/<model>_<unit>.calibration.json (reuse or training record)
  artifacts/<model>_<kind>.export.json         export and ownership records
  ppl/<model>/<policy>/report.json             -> ppl/<model>/<policy>.json (per-window NLL)
  lmeval/<model>/<policy>/report.json          -> lmeval/<model>/<policy>.json.gz (compact JSON, gzip: per-example
                                                  correctness, per-choice log-likelihoods, sample hashes)
  latency/<model>/<policy>/round<r>.json       prefill latency, per process
  gemm/<model>.json                            GEMM and quantizer kernel times
  gemm_superseded/<model>.json                 the fixed-order GEMM records superseded by deviation 1 (PROTOCOL.md)
  tables/*                                     the tables of step 07
Not collected: the artifacts, the maps themselves, the logs (logs/*.log), anything else.
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=P.OUT_DEFAULT)
    ap.add_argument('--dest', type=Path, default=P.REPO / 'results' / 'paper')
    args = ap.parse_args()
    src, dst = args.out, args.dest
    copied = []

    def put(a, b):
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a, b)
        copied.append(b)

    for name in ('00_check.json', 'commands.log'):
        if (src / name).exists():
            put(src / name, dst / name)
    for f in sorted(src.glob('maps/*/calibration.json')):
        put(f, dst / 'maps' / f'{f.parent.name}.calibration.json')
    for f in sorted(src.glob('artifacts/*.export.json')):
        put(f, dst / 'artifacts' / f.name)
    for f in sorted(src.glob('ppl/*/*/report.json')):
        put(f, dst / 'ppl' / f.parent.parent.name / f'{f.parent.name}.json')
    for f in sorted(src.glob('lmeval/*/*/report.json')):
        b = dst / 'lmeval' / f.parent.parent.name / f'{f.parent.name}.json.gz'
        b.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(b, 'wt', compresslevel=9) as g:
            json.dump(json.loads(f.read_text()), g, separators=(',', ':'))
        copied.append(b)
    for f in sorted(src.glob('latency/*/*/round*.json')):
        put(f, dst / 'latency' / f.parent.parent.name / f.parent.name / f.name)
    for sub in ('gemm', 'gemm_superseded'):
        for f in sorted(src.glob(f'{sub}/*.json')):
            put(f, dst / sub / f.name)
    for f in sorted(src.glob('tables/*')):
        put(f, dst / 'tables' / f.name)
    size = sum(b.stat().st_size for b in copied)
    big = sorted(((b.stat().st_size, b) for b in copied), reverse=True)[:3]
    print(f'{len(copied)} files, {size / 2 ** 20:.1f} MiB into {dst}; largest: ' +
          ', '.join(f'{b.relative_to(dst)} {n / 2 ** 20:.1f} MiB' for n, b in big))


if __name__ == '__main__':
    main()
