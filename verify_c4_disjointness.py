"""Recompute calibration/evaluation text-hash intersections from saved reports.

Run through Slurm on this cluster. No model, tensor checkpoints or network needed.
"""
import argparse
import json
from pathlib import Path


def audit_report(report_path):
    report_path = Path(report_path)
    r = json.loads(report_path.read_text())
    if 'fit' in r:
        origin = report_path
        fit = r['fit']
    else:
        origin = Path(r['frozen_map_origin']) / 'report.json'
        fit = json.loads(origin.read_text())['fit']
    c4 = {d['document_sha256'] for d in fit['web']['documents']}
    all_fit = {d['document_sha256'] for meta in fit.values() for d in meta['documents']}
    heldout = {d['document_sha256'] for d in r['c4_data']['documents']}
    assert len(c4) == 64 and len(all_fit) == 192 and len(heldout) == 256
    result = dict(model=r['model'], calibration_c4_documents=len(c4),
                  all_calibration_documents=len(all_fit), heldout_c4_documents=len(heldout),
                  c4_calibration_overlap=len(c4 & heldout), all_calibration_overlap=len(all_fit & heldout),
                  calibration_report=str(origin), evaluation_report=str(report_path))
    assert result['c4_calibration_overlap'] == result['all_calibration_overlap'] == 0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reports', nargs='+')
    ap.add_argument('--out')
    args = ap.parse_args()
    results = [audit_report(p) for p in args.reports]
    text = json.dumps(results, indent=2) + '\n'
    if args.out: Path(args.out).write_text(text)
    print(text, end='')


if __name__ == '__main__': main()
