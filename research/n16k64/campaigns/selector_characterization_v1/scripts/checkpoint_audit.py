"""Save a truthful interim audit. A passing CPU suite is not GPU acceptance."""
from common import *
import ast
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

def main():
    start=now();status=load(OUT/'TASK_STATUS.json');checks=[]
    test=subprocess.run([sys.executable,str(OUT/'scripts/test_math.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    (OUT/'CPU_TEST_REPORT.txt').write_text('Command: '+sys.executable+' scripts/test_math.py\nScope: 9 CPU math/map tests; NOT GPU numerical anchor acceptance\n'+test.stdout+test.stderr)
    assert test.returncode==0
    stream=subprocess.run([sys.executable,str(OUT/'scripts/test_stream_adapter.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_stream_adapter.py\nScope: actual CPU observer and insertion syntax; not model-level calibration\n'+stream.stdout+stream.stderr)
    assert stream.returncode==0
    runpy_probe=subprocess.run([sys.executable,str(OUT/'scripts/test_historical_runpy.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_historical_runpy.py\nScope: reproduce actual sibling-import failure with isolated runpy and verify path-only v2 import; no main/model/GPU execution\n'+runpy_probe.stdout+runpy_probe.stderr)
    assert runpy_probe.returncode==0
    repair_v2=subprocess.run([sys.executable,str(OUT/'scripts/test_calibration_repair_v2.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_calibration_repair_v2.py\nScope: narrow v2 path/provenance transformation and no-authorization refusal; no GPU execution\n'+repair_v2.stdout+repair_v2.stderr)
    assert repair_v2.returncode==0
    cadence_test=subprocess.run([sys.executable,str(OUT/'scripts/test_cadence_guard.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_cadence_guard.py\nScope: fail closed on >60s query/interval, retain foreign-PID rejection, exclude prelaunch waiting; no GPU execution\n'+cadence_test.stdout+cadence_test.stderr)
    assert cadence_test.returncode==0
    backend=subprocess.run([sys.executable,str(OUT/'scripts/test_map_backend.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_map_backend.py\nScope: real map reader and N8–N256 candidate-mask geometry on CPU; not model NLL\n'+backend.stdout+backend.stderr)
    assert backend.returncode==0
    ingestion=subprocess.run([sys.executable,str(OUT/'scripts/test_quality_ingestion.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_quality_ingestion.py\nScope: full-window baseline rejection and historical/new labeling; no GPU inference\n'+ingestion.stdout+ingestion.stderr)
    assert ingestion.returncode==0
    scheduling=subprocess.run([sys.executable,str(OUT/'scripts/test_t2_continue.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_t2_continue.py\nScope: frozen-order scheduling, live-model exclusion, no blind retry; not GPU correctness\n'+scheduling.stdout+scheduling.stderr)
    assert scheduling.returncode==0
    accuracy=subprocess.run([sys.executable,str(OUT/'scripts/test_accuracy_adapter.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_accuracy_adapter.py\nScope: pinned launcher transformation retains lease/preflight/monitor/cleanup and fixed full8 payload; not model correctness\n'+accuracy.stdout+accuracy.stderr)
    assert accuracy.returncode==0
    restricted=subprocess.run([sys.executable,str(OUT/'scripts/test_uuid_launcher.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_uuid_launcher.py\nScope: exact pinned-launcher transformation changes only UUID selection and provenance; no GPU reservation guarantee\n'+restricted.stdout+restricted.stderr)
    assert restricted.returncode==0
    for p in (OUT/'scripts').glob('*.py'):ast.parse(p.read_text());checks.append(dict(path=str(p.relative_to(OUT)),check='Python syntax',passed=True))
    for p in OUT.rglob('*.json'):load(p)
    load(OUT/'FROZEN_PROTOCOL.yaml')
    for p in (OUT/'figures').glob('*.svg'):ET.parse(p)
    figure_sources=load(OUT/'figures/SOURCE_MANIFEST.json')
    assert figure_sources['plot_script_sha256']==sha(OUT/'scripts/plot_results.py')
    for name,digest in figure_sources['sources'].items():
        assert sha(OUT/'results'/name)==digest,('Stale figure input',name)
    checks.append(dict(path='figures/SOURCE_MANIFEST.json',check='Plot script and all source table hashes match current artifacts',passed=True))
    maps=load(OUT/'results/MAP_MANIFEST.json')+load(OUT/'results/GRANULARITY_MAP_MANIFEST.json')
    for m in maps:
        p=OUT/m['path'];assert sha(p)==m['sha256'];_,a=mapread(p);assert maskhash(a)==m['payload_sha256']
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
    runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z';gpu=[]
    waiting=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if '--inspect-only' in argv:continue
            if any(x.endswith('selector_characterization_v1/scripts/'+entry) for x in argv for entry in ['gpu_run.py','accuracy_run.py','gpu_run_on_uuid.py','gpu_run_wait.py','calibration_repair_run.py','calibration_repair_run_v2.py','calibration_llama_retry_run.py','cadence_launch.py']):
                name=argv[argv.index('--name')+1]
                waiting.append(dict(pid=int(proc.name),run=name,argv=argv,
                    state='launched' if (runtime/'runs'/name/'launch_record.json').exists() else 'waiting_preflight'))
        except (OSError,UnicodeError,ValueError):continue
    jsonout(OUT/'results/GPU_QUEUE.json',dict(checked_utc=start,processes=waiting))
    for p in (runtime/'runs').glob('*/launch_record.json'):
        r=load(p);root=p.parent
        if r['status']=='running':continue
        before=[load(x) for x in (root/'preflight').glob('*.json')]
        monitor=[json.loads(x) for x in (root/'gpu_monitor.jsonl').read_text().splitlines()]
        after=load(root/'postflight.json')
        allchecks=before+monitor+[after]
        # Audit elapsed coverage, not merely presence of a monitor log. Include
        # launch/termination boundaries so a late first or last check is visible.
        instants=[datetime.fromisoformat(r['started_utc'].replace('Z','+00:00'))]
        instants += [datetime.fromisoformat(x['timestamp_utc'].replace('Z','+00:00')) for x in monitor]
        instants += [datetime.fromisoformat(r['finished_utc'].replace('Z','+00:00'))]
        gaps=[(b-a).total_seconds() for a,b in zip(instants,instants[1:])]
        cadence_passed=bool(monitor) and all(0<=gap<=60 for gap in gaps)
        if r['status']=='complete':assert cadence_passed,(root.name,'GPU ownership monitoring gap exceeds 60 seconds')
        # The immutable generic launcher used task=T2 for every PPL job,
        # including T3. Add the scientific classification here; never rewrite
        # the original launch records/RUN_LOG events to hide that label defect.
        command=r['command']
        plan_name=Path(command[command.index('--plan')+1]).name if '--plan' in command else ''
        scientific_task=('T0-anchor' if root.name.startswith('pilot_') else
            'T3' if any(str(x).endswith(('stream_calibrate.py','stream_calibrate_historical.py','stream_calibrate_historical_v2.py','stream_calibrate_llama_historical.py')) for x in command)
                or 'granularity' in plan_name or 'reproduction_only' in plan_name else
            'T2-secondary-accuracy' if 'campaign.evaluate_lmeval' in command else 'T2')
        external=load(root/'external_invalidation.json') if (root/'external_invalidation.json').exists() else None
        guard=load(root/'cadence_guard.json') if (root/'cadence_guard.json').exists() else None
        if guard:
            assert sha(REPO/guard['wrapper'])==guard['wrapper_sha256']
            assert guard['maximum_seconds']==60 and guard['scientific_payload_changed'] is False
        parent_rejections=load(OUT/'results/PARENT_MOMENT_REJECTIONS.json') if (OUT/'results/PARENT_MOMENT_REJECTIONS.json').exists() else []
        parent_rejected=any(x['run']==str(root.relative_to(REPO)) for x in parent_rejections)
        gpu.append(dict(run=root.name,scientific_task=scientific_task,status=r['status'],scientific_status='invalid' if external else ('rejected_parent_identity' if parent_rejected else r['status']),external_invalidation=external,cadence_guard=guard,gpu_hours=r['gpu_hours'],gpu_model=r['gpu_model'],
                        ownership_checks=len(allchecks),all_passed=all(x['passed'] for x in allchecks),
                        maximum_ownership_check_gap_seconds=max(gaps),monitoring_within_60_seconds=cadence_passed,
                        source=str(p.relative_to(REPO)),source_sha256=sha(p)))
    events=[];active_attempts=[]
    for p in (runtime/'runs').glob('*/launch_record.json'):
        r=load(p)
        events.append((datetime.fromisoformat(r['started_utc'].replace('Z','+00:00')),1,r['uuid']))
        end=r.get('finished_utc',start)
        events.append((datetime.fromisoformat(end.replace('Z','+00:00')),-1,r['uuid']))
        if r['status']=='running':active_attempts.append(dict(run=p.parent.name,uuid=r['uuid'],status='running',not_final_audit=True))
    active={};peak=0
    for _,direction,uuid in sorted(events):
        active[uuid]=active.get(uuid,0)+direction
        peak=max(peak,sum(n>0 for n in active.values()))
    assert peak<=3,'Physical GPU concurrency exceeds campaign cap'
    reassignments=[]
    helper_versions={sha(OUT/'scripts/reassign_waiting.py'),sha(OUT/'scripts/history/reassign_waiting_pre_accuracy.py')}
    for p in (runtime/'runs').glob('*/waiting_reassignment.json'):
        d=load(p);assert d['additional_scientific_attempt'] is False and d['gpu_hours']==0
        assert d['helper_sha256'] in helper_versions,'Missing historical reassignment helper source'
        assert not (p.parent/'launch_record.json').exists(),'Original wait unexpectedly launched'
        original=d['old_snapshot']['argv'];restored=list(d['new_command'])
        for flag in ['--name','--only-uuid']:restored[restored.index(flag)+1]=original[original.index(flag)+1]
        assert restored==original,'Reassignment changed scientific arguments'
        reassignments.append(dict(source=str(p.relative_to(REPO)),sha256=sha(p),**d))
    guarded_moves=[]
    for p in (runtime/'runs').glob('*/guarded_waiting_reassignment.json'):
        d=load(p);assert d['additional_scientific_attempt'] is False and d['gpu_hours']==0
        assert d['helper_sha256']==sha(OUT/'scripts/reassign_guarded_waiting.py')
        assert d['guard_sha256']==sha(OUT/'scripts/cadence_launch.py')
        assert not (p.parent/'launch_record.json').exists()
        old=d['old_snapshot']['argv'];new=list(d['new_command'])
        for flag in ['--name','--only-uuid']:new[new.index(flag)+1]=old[old.index(flag)+1]
        assert new==old,'Guarded reassignment changed scientific arguments'
        guarded_moves.append(dict(source=str(p.relative_to(REPO)),sha256=sha(p),**d))
    upgrades=[]
    for p in (runtime/'runs').glob('*/waiting_guard_upgrade.json'):
        d=load(p);assert d['additional_scientific_attempt'] is False and d['gpu_hours']==0
        assert d['helper_sha256']==sha(OUT/'scripts/upgrade_wait_guard.py')
        assert not (p.parent/'launch_record.json').exists()
        old=d['old_snapshot']['argv'][2:];new=list(d['new_command'][4:])
        new[new.index('--name')+1]=old[old.index('--name')+1]
        assert new==old,'Guard upgrade changed scientific arguments'
        upgrades.append(dict(source=str(p.relative_to(REPO)),sha256=sha(p),**d))
    jsonout(OUT/'results/GPU_POLICY_AUDIT.json',dict(completed_attempts=gpu,active_attempts=active_attempts,cap=3,
        waiting_guard_upgrades=upgrades,
        guarded_waiting_reassignments=guarded_moves,
        waiting_reassignments=reassignments,
        observed_launch_intervals_max_physical_concurrency=peak,
        qualification='Terminal attempts include failures and invalid co-tenancy attempts. Running intervals stop at this audit time; waiting launchers have no lease. Invalid attempts are retained, not scientifically accepted.'))
    q=list(csv.DictReader((OUT/'results/quality_results.csv').open()));g=list(csv.DictReader((OUT/'results/granularity_quality.csv').open()))
    gate=load(OUT/'results/ANCHOR_GATE.json')
    accuracy_ingestion=subprocess.run([sys.executable,str(OUT/'scripts/test_accuracy_ingestion.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_accuracy_ingestion.py\nScope: exact baseline IDs/responses/correctness/runtime and monitor-gap rejection; not model correctness\n'+accuracy_ingestion.stdout+accuracy_ingestion.stderr)
    assert accuracy_ingestion.returncode==0
    inventory_wait=subprocess.run([sys.executable,str(OUT/'scripts/test_inventory_wait.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_inventory_wait.py\nScope: prelaunch inventory errors wait without changing any in-run/postflight check; no GPU allocation\n'+inventory_wait.stdout+inventory_wait.stderr)
    assert inventory_wait.returncode==0
    reassignment=subprocess.run([sys.executable,str(OUT/'scripts/test_wait_reassignment.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_wait_reassignment.py\nScope: reject foreign/reused/unrelated/leased/child-bearing processes; no process signals in this CPU test\n'+reassignment.stdout+reassignment.stderr)
    assert reassignment.returncode==0
    guarded_test=subprocess.run([sys.executable,str(OUT/'scripts/test_guarded_reassignment.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_guarded_reassignment.py\nScope: guarded-waiter identity/activity/payload rejection; no process signals or GPU query\n'+guarded_test.stdout+guarded_test.stderr)
    assert guarded_test.returncode==0
    llama_repair_test=subprocess.run([sys.executable,str(OUT/'scripts/test_llama_calibration_retry.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    with (OUT/'CPU_TEST_REPORT.txt').open('a') as f:f.write('\nCommand: '+sys.executable+' scripts/test_llama_calibration_retry.py\nScope: four CPU checks for exact Llama historical options, narrow transform, scoped real grant and no-grant refusal; no model execution\n'+llama_repair_test.stdout+llama_repair_test.stderr)
    assert llama_repair_test.returncode==0
    accuracy_table=load(OUT/'results/SECONDARY_ACCURACY_TABLE_AUDIT.json')
    assert accuracy_table['script_sha256']==sha(OUT/'scripts/accuracy_table.py')
    for rel,expected in accuracy_table['source_hashes'].items():assert sha(OUT/rel)==expected,('stale accuracy table',rel)
    accuracy_plot=load(OUT/'figures/ACCURACY_SOURCE_MANIFEST.json')
    assert accuracy_plot['script_sha256']==sha(OUT/'scripts/accuracy_plot.py')
    for rel,expected in {**accuracy_plot['sources'],**accuracy_plot['outputs']}.items():assert sha(OUT/rel)==expected,('stale accuracy figure',rel)
    granularity_table=load(OUT/'results/GRANULARITY_TABLE_AUDIT.json')
    assert granularity_table['script_sha256']==sha(OUT/'scripts/granularity_table.py')
    for rel,expected in granularity_table['source_hashes'].items():assert sha(OUT/rel)==expected,('stale granularity table',rel)
    t2_count=sum(r.get('status') in ['validated_reuse','validated_new'] and r['policy']!='baseline' for r in q)
    t3_count=sum(r['status'] in ['validated_reuse','validated_new'] for r in g)
    assert t2_count<=150 and t3_count<=36
    accuracy_coverage=load(OUT/'results/ACCURACY_NEW_AUDIT.json')['coverage']
    acceptance=dict(cpu_tests=dict(passed=True,count=55,not_equivalent_to='All nine handoff acceptance items; item 8 needs successful numerical model reproduction'),
        T1='complete',T2=dict(validated_cells=t2_count,required_cells=150,status='primary_ppl_complete' if t2_count==150 else 'incomplete',secondary_accuracy=accuracy_coverage),
        T3=dict(validated_cells=t3_count,required_cells=36,status='complete' if t3_count==36 else 'incomplete'),
        T4='bounded review complete',T5='interim outputs available; final acceptance pending',numerical_anchor={m:x['passed'] for m,x in gate.items()},
        n8_reproduction=load(OUT/'results/N8_REPRODUCTION.json')['models'] if (OUT/'results/N8_REPRODUCTION.json').exists() else 'not checked',
        n16_full_reproduction=load(OUT/'results/N16_REPRODUCTION.json')['models'] if (OUT/'results/N16_REPRODUCTION.json').exists() else 'not checked',
        map_hashes_verified=len(maps),all_required_passed=False,checked_utc=start,checks=checks)
    jsonout(OUT/'ACCEPTANCE_AUDIT.json',acceptance)
    # Add derived evidence without overwriting original T4 source lineage.
    ledger=list(csv.DictReader((OUT/'evidence_ledger.csv').open()))
    for question,name in [('map stability','map_pairs.csv'),('objective quality','quality_results.csv'),('granularity measured','granularity_quality.csv'),('granularity surrogate','granularity_results.csv'),('GPU numerical reproduction','anchor_windows.csv'),('historical composition and budget','composition_existing.csv')]:
        rel='results/'+name
        ledger=[x for x in ledger if x['source']!=rel]
        ledger.append(dict(question=question,source=rel,sha256=sha(OUT/rel),verification='new CPU recomputation from verified raw inputs; missing results remain explicit',status='interim'))
    csvout(OUT/'evidence_ledger.csv',ledger)
    checkpoint('T4','done',['individual_tile_review.md','results/effect_diagnostics.csv','results/group_interventions.csv','results/INDIVIDUAL_SUMMARY.json','results/reorder_ratio_audit.csv','results/composition_existing.csv'],
        next_action='查看 GPU_QUEUE 的 live runs；完成後核對完整 baseline windows 與 protocol，再納入 T2/T3。保留所有 invalid/failed attempts。')
    checkpoint('T5','running',['REPORT.md','CLAIMS.md','REPRODUCE.md','figures','CPU_TEST_REPORT.txt','ACCEPTANCE_AUDIT.json'],
        next_action='查看 results/GPU_QUEUE.json 與 live PID，勿重複提交。三模型 prefix anchor 已通過；full-window baseline gate 與 T3 N8 reproduction 仍須驗證。')
    status=load(OUT/'TASK_STATUS.json')
    task2=next(t for t in status['tasks'] if t['id']=='T2')
    task2['primary_ppl_status']='complete' if t2_count==150 else 'incomplete'
    task2['secondary_accuracy']=accuracy_coverage
    task2['status']='done' if t2_count==150 and accuracy_coverage['missing']==0 else 'running'
    bp=OUT/'results/GPU_RESOURCE_BLOCKERS.json'
    if bp.exists():
        resource_blockers=load(bp)
        for task in status['tasks']:
            if task['id'] in ['T2','T3']:
                prefixes=tuple(key+': ' for key in resource_blockers)
                task['blockers']=[b for b in task.get('blockers',[]) if not b.startswith(prefixes)]
                for key,block in resource_blockers.items():
                    target='T2' if key.endswith('_secondary_accuracy') else 'T3'
                    if block.get('status')=='blocked_retry_limit' and task['id']==target:
                        message=key+': '+block['required_external_resolution']
                        if message not in task['blockers']:task['blockers'].append(message)
    if all(gate.get(m,{}).get('passed') for m in MODELS):
        for task in status['tasks']:
            task['blockers']=[b for b in task.get('blockers',[]) if b!='GPU numerical anchor unresolved']
    jsonout(OUT/'TASK_STATUS.json',status)
    log('checkpoint-audit','python scripts/checkpoint_audit.py',start,[OUT/'ACCEPTANCE_AUDIT.json',OUT/'CPU_TEST_REPORT.txt'])
    # A point-in-time manifest, not a promise that mutable live checkpoints stop changing.
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['ARTIFACT_MANIFEST.sha256'] and '__pycache__' not in p.parts)
    (OUT/'ARTIFACT_MANIFEST.sha256').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files))
    print('Interim audit:',acceptance['T2'],acceptance['T3'],'hashed files',len(files))

if __name__=='__main__':main()
