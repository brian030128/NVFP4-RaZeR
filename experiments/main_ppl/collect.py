#!/usr/bin/env python3
"""Copy the main PPL table's run records into results/main_ppl (the packed artifacts stay outside the repository).

    python experiments/main_ppl/collect.py

- runs/<model>/<row>.json: every run_ppl_deploy.py report (per-window NLL, window hashes, install and coverage records),
  including the reuse rechecks (<row>-recheck);
- exports/<artifact>/: each new export's record (artifact.json), ownership check and map provenance;
- smoke/, diagnostics/: the Nemotron feasibility smoke and the Qwen3-1.7B fake-map diagnostic (not table numbers);
- commands.log and logs/ (one log per command; *.log is git-ignored, so force-add them).
"""
import os
import shutil
from pathlib import Path

OUT = Path(os.environ.get('MAIN_PPL_OUT', '/home/dev/n16k64_campaign/main_ppl'))
DEST = Path(__file__).resolve().parents[2] / 'results' / 'main_ppl'


def main():
    n = 0
    for sub in ('ppl', 'smoke', 'diagnostics'):
        for rep in sorted((OUT / sub).glob('*/*/report.json')):
            model, row = rep.parent.parent.name, rep.parent.name
            dst = DEST / ('runs' if sub == 'ppl' else sub) / model / f'{row}.json'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rep, dst)
            n += 1
    for art in sorted((OUT / 'artifacts').glob('*')):
        if not art.is_dir():
            continue
        for f in ('artifact.json', 'ownership.json'):
            if (art / f).exists():
                dst = DEST / 'exports' / art.name / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(art / f, dst)
                n += 1
        prov = art.parent / f'{art.name}.mixfp4map.provenance.json'
        if prov.exists():
            shutil.copyfile(prov, DEST / 'exports' / art.name / prov.name)
    for name in ('commands.log', 'smoke_nemotron.out', 'run_all.out'):
        if (OUT / name).exists():
            shutil.copyfile(OUT / name, DEST / name)
    if (OUT / 'logs').exists():
        shutil.copytree(OUT / 'logs', DEST / 'logs', dirs_exist_ok=True)
    print(f'{n} records copied to {DEST}')


if __name__ == '__main__':
    main()
