"""Stage 2 of PROTOCOL.md: register one model before its first run (protocol, code, data and kernel hashes).

python results/multiround_models/register_model.py MODEL DATA_ROOT   (writes results/multiround_models/<model>/registration.json)
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = ('run_multiround.py', 'prepare_model_data.py', 'repro_local/realquant/candidate_store.py',
        'repro_local/realquant/native_dev.py', 'repro_local/realquant/rq.py', 'repro_local/realquant/fused_quant.py',
        'quantize/packed_candidates.py', 'quantize/quantizer.py', 'quantize/causal_four_over_six.py',
        'quantize/fast_act.py', 'cost_monitor.py', 'run_baseline_protocol_audit.py', 'run_math_code_calibration.py',
        'run_fine_row_research.py', 'results/multiround_models/compare_batching.py',
        'results/multiround_models/precheck.py', 'results/multiround_models/analyze_model.py')
KERNEL = Path('/home/dev/n16k64_campaign/realquant/bin/libb8x64.so')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    model, data = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    here = REPO / 'results' / 'multiround_models'
    registered = json.loads((here / 'registration.json').read_text())
    summary = json.loads((data / 'prepare_summary.json').read_text())
    development = sorted(p.parent.name for p in data.glob('*/fresh.pt') if not p.parent.name.startswith('_'))
    out = dict(model=model, registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               protocol=dict(registered_sha256=registered['protocol_sha256'],
                             registered_utc=registered['registered_utc'],
                             current_sha256=digest(here / 'PROTOCOL.md'),
                             note='current differs from registered only by appended deviations'),
               code_sha256={p: digest(REPO / p) for p in CODE}, kernel_library=dict(path=str(KERNEL), sha256=digest(KERNEL)),
               data=dict(root=str(data), prepare_summary_sha256=digest(data / 'prepare_summary.json'),
                         calibration_report_sha256=digest(data / 'calibration' / 'report.json'),
                         development_fresh_sha256={d: digest(data / d / 'fresh.pt') for d in development},
                         fit_from=summary['fit_from'], matrices=summary['matrices']))
    target = here / model
    target.mkdir(exist_ok=True)
    (target / 'registration.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(model=model, registered_utc=out['registered_utc'], development=development), indent=1))


if __name__ == '__main__':
    main()
