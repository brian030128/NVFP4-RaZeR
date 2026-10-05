"""Writes results/topk_cal/registration.json, CPU only: the sha256 of every file the measured runs read or are judged by
(the protocol, the queue, run_train_map.py and its source closure, the top-K module and its test, the reference maps
and the check records), plus flipquant's reference commit and file hash."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
D = Path('/home/dev/n16k64_campaign/topk')
PAPER = Path('/home/dev/n16k64_campaign/paper/maps')
FILES = ['results/topk_cal/PROTOCOL.md', 'results/topk_cal/run_topk_cal.sh', 'results/topk_cal/make_registration.py',
         'results/topk_cal/test_topk_teacher.json', 'results/topk_cal/check_a_llama8b_8x64.json',
         'repro_local/realquant/test_topk_teacher.py', 'cost_monitor.py', 'run_math_code_calibration.py']
REFERENCES = [PAPER / 'llama8b_16x64' / 'map.pt', PAPER / 'llama8b_8x64' / 'map.pt',
              Path('/home/dev/n16k64_campaign/cost_comparison/data/llama8b/calibration/report.json')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import run_train_map
    git = lambda *a, cwd=REPO: subprocess.run(['git', *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    files = {f: sha(REPO / f) for f in sorted(set(FILES) | set(run_train_map.SOURCES))}
    flipquant = Path.home() / 'flipquant'
    source = subprocess.run(['git', '-C', str(flipquant), 'show', 'd2dd92e:calibration/train_map.py'], check=True,
                            capture_output=True).stdout
    reg = dict(protocol='results/topk_cal/PROTOCOL.md', protocol_sha256=files['results/topk_cal/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any measured run; checks (a) and (b) and a one-epoch smoke run of C were done before, as the '
                     'protocol says'),
               files=files, references={str(p): sha(p) for p in REFERENCES},
               flipquant=dict(repository='brian030128/flipquant', ref='d2dd92e',
                              commit=git('rev-parse', 'd2dd92e', cwd=flipquant),
                              calibration_train_map_py_sha256=hashlib.sha256(source).hexdigest()))
    out = REPO / 'results' / 'topk_cal' / 'registration.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {len(REFERENCES)} references')


if __name__ == '__main__':
    main()
