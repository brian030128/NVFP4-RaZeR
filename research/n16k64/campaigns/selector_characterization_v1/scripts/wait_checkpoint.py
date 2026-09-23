"""Persist a verified-live wait without rerunning scientific analyses/tests."""
from common import *

def main():
    start=now();s=load(OUT/'TASK_STATUS.json');processes=[];controllers=[]
    runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if '--inspect-only' in args:continue
            if any(x.endswith('selector_characterization_v1/scripts/t2_continue.py') for x in args):
                controllers.append(dict(pid=int(p.name),proc_start_ticks=(p/'stat').read_text().split()[21],kind='T2 continuation'))
            scripts=[x for x in args if any(x.endswith('selector_characterization_v1/scripts/'+entry) for entry in ['gpu_run.py','accuracy_run.py','gpu_run_on_uuid.py','gpu_run_wait.py','calibration_repair_run.py','calibration_repair_run_v2.py','calibration_llama_retry_run.py','cadence_launch.py'])]
            if not scripts:continue
            assert len(scripts)==1
            cwd=(p/'cwd').resolve();entry=Path(scripts[0])
            if not entry.is_absolute():entry=cwd/entry
            plan=None
            if '--plan' in args:
                plan_path=Path(args[args.index('--plan')+1])
                if not plan_path.is_absolute():plan_path=cwd/plan_path
                plan=dict(path=str(plan_path),sha256=sha(plan_path))
            run=args[args.index('--name')+1];root=runtime/'runs'/run
            latest=None;cp=root/'preflight_candidates.jsonl'
            if cp.exists():
                with cp.open() as f:
                    for line in f:
                        try:latest=json.loads(line)
                        except json.JSONDecodeError:pass  # only a partial last append, never a terminal state
            processes.append(dict(pid=int(p.name),run=run,proc_start_ticks=(p/'stat').read_text().split()[21],
                command=args,cwd=str(cwd),launcher_source_sha256=sha(entry),plan=plan,
                protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.yaml'),
                state='launched' if (root/'launch_record.json').exists() else 'waiting_preflight',
                last_preflight_time=latest['timestamp_utc'] if latest else None,last_preflight_passed=latest['passed'] if latest else None,
                last_preflight_reasons=latest['reasons'] if latest else None))
        except (OSError,ValueError,UnicodeError):continue
    assert processes,'No live launcher found: inspect terminal state, do not claim a verified wait'
    jsonout(OUT/'results/GPU_QUEUE.json',dict(checked_utc=start,processes=processes,controllers=controllers))
    s['last_updated']=start;s['execution_wait']=dict(status='verified_live',processes=processes,controllers=controllers);jsonout(OUT/'TASK_STATUS.json',s)
    gates=load(OUT/'results/ANCHOR_GATE.json');passed=', '.join(m for m in MODELS if gates.get(m,{}).get('passed'))
    (OUT/'NEXT_ACTION.md').write_text('# 下一步\n\n已驗證 live launcher：'+', '.join(x['run']+' PID '+str(x['pid']) for x in processes)+
        '。等待者僅使用獨占 A6000；最多三張實體 GPU，不把等待 launcher 算成已使用的 GPU。勿重複提交。終止／完成須由 process/session 或 launch_record 實證確認。\n\n'+
        '已通過 prefix anchor：'+passed+'。新 full outcomes 仍需完整 baseline window gate；N8/N16 full reproduction 另見驗收。Mistral/Qwen cache 見 results/RECOVERED_CACHE_VERIFICATION.json。\n\n'+
        ('T2 continuation controller live PID：'+', '.join(str(x['pid']) for x in controllers)+'。由 controller 執行 quality.py 與接續 frozen plans；不要另起並行 T2 ingestion writer。\n\n' if controllers else
         ('T2 primary PPL 已完成 150/150，原 controller 正常結束；不要重跑。Secondary accuracy 仍須核對完整 baseline prompt/response 與各 task outputs 後才能納入。\n\n' if next(t for t in s['tasks'] if t['id']=='T2').get('primary_ppl_status')=='complete' else '未觀察到 T2 controller；先查 terminal 狀態再決定是否續跑。\n\n'))+
        '2026-09-23 使用者明確授權三項各一次追加修復：Llama/Mistral calibration 與 Qwen coarse evaluation；見 plans/additional_retry_authorization_20260923.json 和 results/ADDITIONAL_RETRY_PLAN.json。只限指定 attempt4，不可另開 attempt5。Llama secondary accuracy 不在追加授權內，仍 blocked。Calibration 完成須先 validate_parent_moments.py 通過，才能產生/評估 coarse maps；Qwen evaluation 完成須 granularity_quality.py 與 N8/N16 reproduction audits。Mistral accuracy 完成後執行 accuracy_new_results.py、accuracy_table.py、accuracy_plot.py。所有 terminal 均更新 anchor_audit.py 與 checkpoint；不得重跑已完成 T2 PPL 或 Llama/Mistral N8/N16 reproduction。\n')
    with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(run_id='wait-'+start,task='GPU_WAIT',status='verified_wait',checked_utc=start,processes=processes,gpu_hours_added=0))+'\n')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.sha256' and '__pycache__' not in p.parts)
    (OUT/'ARTIFACT_MANIFEST.sha256').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files))
    print([dict(pid=x['pid'],run=x['run'],state=x['state'],last_preflight_time=x['last_preflight_time']) for x in processes])

if __name__=='__main__':main()
