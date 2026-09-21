"""V00 inventory: source / environment / data / model / archived-artifact manifest with availability audit (CPU)."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from campaign import runtime
from campaign.launcher import source_manifest

CR = Path(os.environ['CAMPAIGN_ROOT'])
SRC = CR / 'source' / 'NVFP4-RaZeR-main'


def sha(p):
    return runtime.sha256_file(p)


def main():
    out = runtime.out_dir('inventory')
    inv = dict(campaign_root=str(CR), host=platform.node())
    inv['input'] = json.loads((CR / 'provenance' / '00_input_resolution.json').read_text())
    inv['handoff_verification'] = json.loads((CR / 'provenance' / 'handoff_sha256_verify.json').read_text())
    src_sha, entries = source_manifest()
    inv['source_manifest'] = dict(sha256=src_sha, files=len(entries))
    (out / 'SOURCE_MANIFEST.txt').write_text(''.join(f'{h}  {r}\n' for r, h in entries))
    handoff_src = CR / 'handoff' / 'agent_handoff' / 'source' / 'NVFP4-RaZeR-main'
    changed, added = [], []
    for r, h in entries:
        hp = handoff_src / r
        if not hp.exists():
            added.append(r)
        elif sha(hp) != h:
            changed.append(r)
    inv['source_delta_vs_handoff'] = dict(modified_archived_files=changed, added_files=added,
                                         note='campaign code lives in campaign/; archived files are not modified')
    inv['environment'] = dict(locks={n: dict(path=f'env/{n}.lock.txt', sha256=sha(CR / 'env' / f'{n}.lock.txt')) for n in ('hist', 'main')},
                              python='3.11.11 (uv-managed)', image='ubuntu@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7',
                              driver='565.57.01 (host nvidia-smi; CUDA 12.7 driver API)')
    fetch = json.loads((CR / 'provenance' / 'INPUT_FETCH_MANIFEST.json').read_text())
    inv['models'] = {k: dict(repo=v['repo'], revision=v['revision'], resolved=v['resolved_revision'], files=len(v['files']),
                             bytes=sum(f['size'] for f in v['files']), lfs_mismatches=v['lfs_mismatches']) for k, v in fetch['models'].items()}
    inv['datasets'] = {k: {kk: vv for kk, vv in v.items() if kk != 'files'} for k, v in fetch['datasets'].items()}
    inv['fetch_failures'] = fetch['failures']
    inv['lm_eval_tasks'] = json.loads((CR / 'provenance' / 'LM_EVAL_TASK_FETCH_offline.json').read_text())['tasks']
    # archived artifacts
    arch = {}
    for key, d in (('llama8b', 'results/math_code_adaptive/calibration_333779_llama8b'), ('qwen4b', 'results/math_code_adaptive/calibration_333779_qwen4b'),
                   ('qwen27b', 'results/math_code_adaptive/calibration_333787_qwen27b')):
        p = SRC / d
        rep = json.loads((p / 'report.json').read_text())
        arch[key] = dict(report_sha256=sha(p / 'report.json'), maps_json_sha256=sha(p / 'maps.json'), maps_json_matches_report=sha(p / 'maps.json') == rep['map_sha256'],
                         score_shards_present=(p / 'scores').exists(), weight_mse_pt_present=(p / 'weight_mse.pt').exists(),
                         archived_k3_map_saved=False, note='k=3 masks were never saved; only counts in kse_paper reports')
    audit = json.loads((SRC / 'results/math_code_adaptive/summary_333786_333788/curvature_audit.json').read_text())
    inv['archived'] = dict(calibrations=arch, score_shard_hashes_recorded=len(audit['score_artifact_sha256']),
                           kse_reports={k: sha(SRC / v / 'report.json') for k, v in (('llama8b', 'results/kse_paper/job_336566/llama8b'),
                                                                                    ('qwen4b', 'results/kse_paper/job_336566/qwen4b'), ('qwen27b', 'results/kse_paper/job_336969/qwen27b'))},
                           original_cluster_paths_available=bool(os.environ.get('NVFP4_ARCHIVED_RESULTS_ROOT')) and
                           os.path.exists(os.environ['NVFP4_ARCHIVED_RESULTS_ROOT']),
                           conclusion='score shards absent on this host and in the handoff ZIP; regenerate under frozen provenance (V20)')
    res_files = sorted(p for p in (SRC / 'results').rglob('*') if p.is_file())
    (out / 'ARCHIVED_RESULTS_SHA256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(SRC).as_posix()}\n' for p in res_files))
    inv['archived_results_files'] = dict(count=len(res_files), manifest='inventory/ARCHIVED_RESULTS_SHA256.txt')
    v01 = CR / 'reports' / 'V01' / 'v01_integration.json'
    inv['gpu_inventory'] = json.loads(v01.read_text())['gpu_table'] if v01.exists() else 'unavailable in CPU container'
    inv['gpu_inventory_source'] = 'host-side nvidia-smi table recorded by campaign.v01_integration'
    inv['scheduler'] = 'none on host; campaign_local_lease + docker device cgroup (provenance/01_owner_decisions.json#D01)'
    runtime.atomic_json(out / 'INVENTORY.json', inv)
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='inventory', protocol_freeze_sha256=os.environ.get('FREEZE_SHA256', ''),
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src_sha),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / 'INVENTORY.json'), str(out / 'SOURCE_MANIFEST.txt'), str(out / 'ARCHIVED_RESULTS_SHA256.txt')],
                                  summary=dict(models=list(inv['models']), fetch_failures=len(fetch['failures']), score_shards_present=False), uncertainty={},
                                  attempted_endpoints=['inventory'], missing_endpoints=[]), logs=[], failures=[]))
    print(json.dumps(dict(source_manifest=src_sha, modified_archived_files=len(changed), added=len(added)), indent=1))


if __name__ == '__main__':
    main()
