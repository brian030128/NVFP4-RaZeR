"""Preserve launch records while separately classifying monitor-gap invalidation."""
from common import *

def main():
    start=now();runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    rows=[]
    for run in ['secondary_accuracy_llama8b_attempt1','secondary_accuracy_mistral7b_attempt1','reproduction_mistral7b_attempt1']:
        root=runtime/'runs'/run;launch=load(root/'launch_record.json');event=load(root/'external_invalidation.json')
        assert launch['status']!='running' and event['scientific_status']=='invalid'
        rows.append(dict(run=run,launcher_status=launch['status'],scientific_status='invalid',
            gpu_hours=launch['gpu_hours'],reason=event['reason'],gaps=event['gaps'],
            launch_path=str((root/'launch_record.json').relative_to(REPO)),launch_sha256=sha(root/'launch_record.json'),
            event_path=str((root/'external_invalidation.json').relative_to(REPO)),event_sha256=sha(root/'external_invalidation.json')))
    dest=OUT/'results/MONITOR_CADENCE_INCIDENT.json'
    jsonout(dest,dict(checked_utc=start,attempts=rows,total_gpu_hours=sum(x['gpu_hours'] for x in rows),
        cause='Unresolved host/scheduling delay; shared timestamps do not establish physical cause or foreign co-tenancy',
        action='Verified own child PID/UID/start ticks/command/environment before SIGTERM; no foreign process signalled',
        admission='No outputs admitted. Original launcher records remain failed; separate scientific classification is invalid.',
        repair='First reasoned retries use cadence_launch.py with 60-second fail-closed duration/interval checks. Scientific plans unchanged.'))
    log('monitor-cadence-incident','python scripts/record_monitor_incident.py',start,[dest])
    print('Invalid GPU-hours',sum(x['gpu_hours'] for x in rows))

if __name__=='__main__':main()
