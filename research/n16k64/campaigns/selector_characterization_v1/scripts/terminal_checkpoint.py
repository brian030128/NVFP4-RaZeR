"""Close this authorized tranche without declaring missing science complete."""
import argparse
import subprocess
from common import OUT, REPO, load, jsonout, now, sha, log

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    start=now()
    blocks=load(OUT/'results/GPU_RESOURCE_BLOCKERS.json')
    run='granularity_qwen4b_attempt4_authorized'
    lp=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/run/'launch_record.json'
    launch=load(lp)
    assert launch['status']=='complete'
    for n in (8,16):
        audit=load(OUT/f'results/N{n}_REPRODUCTION.json')
        assert all(x['passed'] for x in audit['models'].values())
    assert len(load(OUT/'results/GRANULARITY_TABLE_AUDIT.json')['verified_points'])==20
    blocks['qwen4b_granularity'].update(status='resolved_validated',last_run=run,
        source=str(lp.relative_to(REPO)),gpu_hours=launch['gpu_hours'],
        reason='Authorized fourth attempt passed full baseline, map/runtime/window identity, N8/N16 reproduction and ownership/cadence/postflight admission. Eight new coarse cells accepted; earlier failures remain retained.',
        evidence=['results/GRANULARITY_TABLE_AUDIT.json','results/N8_REPRODUCTION.json','results/N16_REPRODUCTION.json'],
        required_external_resolution='None for Qwen; do not rerun accepted cells.')
    blocks['mistral7b_calibration']['repair']='CPU import repair and restored historical source/options executed in the explicitly authorized fourth attempt; operational completion did not pass historical score/map identity. No fifth attempt authorized.'
    jsonout(OUT/'results/GPU_RESOURCE_BLOCKERS.json',blocks)
    if args.prepare:return
    audit=load(OUT/'ACCEPTANCE_AUDIT.json')
    assert audit['cpu_tests']['passed'] and audit['cpu_tests']['count']==55
    assert audit['T2']['validated_cells']==150 and audit['T3']['validated_cells']==20
    policy=load(OUT/'results/GPU_POLICY_AUDIT.json')
    assert not policy['active_attempts']
    assert policy['observed_launch_intervals_max_physical_concurrency']<=3
    # Inspect processes without signaling them. Ignore this audit itself.
    commands=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    live=[line for line in commands.splitlines() if 'selector_characterization_v1/scripts/' in line
          and any('/'+name+' ' in line for name in ('cadence_launch.py','gpu_run.py','gpu_run_wait.py','accuracy_run.py','calibration_repair_run_v2.py','calibration_llama_retry_run.py','t2_continue.py'))]
    assert not live,live
    costs=load(OUT/'results/GPU_COST.json')
    state=load(OUT/'TASK_STATUS.json')
    for task in state['tasks']:
        if task['id'] in ('T2','T3'):task['status']='blocked'
        if task['id']=='T3':task['validated_cells']=20;task['required_cells']=36
        if task['id']=='T5':
            task['status']='complete_with_explicit_gaps'
            task['blockers']=['Full handoff acceptance remains false because T2 secondary accuracy lacks 16 cells and T3 lacks 16 cells.']
    state.update(last_updated=now(),gpu_hours=costs['total_gpu_hours'],
        execution_wait=dict(status='terminal_no_active_launches',processes=[],controllers=[]),
        disposition='All currently authorized independent work completed; scientific handoff not fully complete.')
    jsonout(OUT/'TASK_STATUS.json',state)
    audit.update(T5='Deliverables checkpoint complete with explicit T2/T3 gaps; no full acceptance',
        all_required_passed=False,terminal_checked_utc=now())
    jsonout(OUT/'ACCEPTANCE_AUDIT.json',audit)
    (OUT/'NEXT_ACTION.md').write_text('''# 下一步：外部 blocker，勿自動重試

本次三項追加嘗試均已結束，沒有活躍 GPU 工作。Qwen coarse evaluation 通過；
Llama/Mistral calibration attempt4 雖完成，但原始 scores/maps 身份不符，不能使用。

- T2 primary PPL：150/150；secondary accuracy：80/96。Llama 的 16 cells
  仍缺，原三次嘗試無效；本次授權不包含第四次 accuracy 嘗試。
- T3：20/36。Qwen 六粒度完整；Llama/Mistral 各缺 N32/64/128/256 × Wiki/C4。
- T0/T1/T4 與可完成的交付整理已完成，完整驗收仍不通過。

最小缺件：Llama/Mistral 原 seed0 全量逐 sequence scores 或完整 child covariance，
需通過既有原始 moments/map/hash gates。若只能重新 calibration，必須先解決
exact-reproduction 差異，再另行取得追加授權；不能盲目啟動 attempt5。
Llama accuracy 需要有效既有 paired predictions，或另外授權額外嘗試及獨占 A6000 時段。

取得資料後先依 REPRODUCE.md 檢查身份，執行 scripts/validate_parent_moments.py；
僅在通過後依 granularity_job_plan.csv 與 plans/ 中的 frozen plan 續跑缺項。
目前禁止直接執行已耗盡授權的 GPU commands。不要重跑已完成的 Qwen 或 T2 PPL。

CPU 檢查（使用 REPRODUCE.md 中的既有 Python 環境）：
`python scripts/granularity_table.py`、`python scripts/accuracy_table.py`。
不要把 CPU table checks 當成補足缺失模型評估。
''')
    jsonout(OUT/'results/TERMINAL_CHECKPOINT.json',dict(checked_utc=now(),
        scope='The three specifically authorized additional attempts; no fifth attempts',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        all_required_passed=False,active_campaign_processes=live,
        gpu_hours=costs,delivery_status='partial_scientific_completion_external_blockers',
        manifest_scope='Campaign directory excluding manifest itself and Python caches; raw local-only runtime evidence is hash-referenced, not duplicated.'))
    log('terminal-checkpoint','python scripts/terminal_checkpoint.py',start,
        [OUT/'TASK_STATUS.json',OUT/'ACCEPTANCE_AUDIT.json',OUT/'NEXT_ACTION.md',OUT/'results/TERMINAL_CHECKPOINT.json'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.sha256' and '__pycache__' not in p.parts)
    (OUT/'ARTIFACT_MANIFEST.sha256').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files))
    print('Terminal partial checkpoint:',len(files),'hashed files;',costs['total_gpu_hours'],'GPU-hours')

if __name__=='__main__':main()
